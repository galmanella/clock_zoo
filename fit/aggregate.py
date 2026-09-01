"""
fit/aggregate.py
================
Join the per-task outputs of a CAMPAIGN into one campaign-level result.

    $PY -m fit.aggregate --model almeida --tag bmal1seeds            # aggregate + report
    $PY -m fit.aggregate --model almeida --tag bmal1seeds --list     # what it would join
    $PY -m fit.aggregate --model almeida --tag genes --kind genes    # the per-gene campaign

WHY THIS EXISTS AT ALL
    `fit.radial.run_seeds` compares the seeds that share ONE array task, because that is all
    one task can see. A campaign that splits 16 seeds across 4 tasks therefore produces four
    4-seed comparisons and NO 16-seed one -- and the 16-seed comparison is the entire question.
    Sixteen seeds in four separate clusters of four look exactly like one cluster of four when
    the only distances ever measured are within a task.

    The campaign file says so itself: "comparing ACROSS tasks needs a separate aggregation over
    the four seeds_*.npz". This is that aggregation, and it reads the PER-SEED `radial_*.npz`
    rather than those per-task summaries, because a summary carries only (v, cost) and the
    question needs the verdicts and the surfaces too.

WHAT IT REFUSES TO DO
    Joining runs that are not the same experiment produces a distance matrix with no meaning.
    Every run joined must agree on model / target / mode / n_phase / optimizer, on the dose
    grid, and on the pinned target registration (k, psi); a mismatch RAISES rather than being
    quietly averaged over. Duplicate seeds raise too -- the same seed twice is not two
    independent searches, and keeping one of them silently would understate the spread.

WHAT IT WRITES, AND WHY IT IS FAT
    `out/<model>/fit_radial/<tag>/seeds_<target>_<mode>.npz`, with the SAME key names as the
    per-task file (seeds, v, cost, dist, cfg_json) plus everything a figure or a re-analysis
    needs: per-run verdict flags, both twist metrics, and THE RAW SURFACES -- every fitted PTC,
    every twist curve, the base and the target. REPO_MAP hazard 9: these fits cost ~16
    CPU-hours, re-deriving a figure from a scalar table is impossible, and a few MB of npz is
    free by comparison.

    Two things are RECOMPUTED here rather than copied. `accumulated_twist` and `signed_twist`:
    the runs stored `total_twist`, which SATURATES at 0.5 and reads 0.497 for the Almeida/BMAL1
    base -- pinned to its ceiling, so every "twist 0.497 -> 0.10" in those logs is a ratio
    against a number that could not have been larger (PROJECT_SUMMARY 5.9a). Both unbounded
    metrics are pure functions of the saved twist curve, so they cost nothing to re-derive.
"""
import argparse
import glob
import os

import numpy as np

import paths
from analysis import winding as W

#: per-run scalars copied straight through
_SCALARS = ('base__parts_c_ptc', 'fit__parts_c_ptc', 'base__parts_total', 'fit__parts_total',
            'base__parts_alive_frac', 'fit__parts_alive_frac',
            'base__parts_period', 'fit__parts_period',
            'base__parts_re_lambda', 'fit__parts_re_lambda',
            'base__parts_fp_res', 'fit__parts_fp_res',
            'base__parts_osc', 'fit__parts_osc',
            'base__total_twist', 'fit__total_twist',
            'base__S_crit', 'fit__S_crit', 'base__phi_sing', 'fit__phi_sing',
            'base__n_sing', 'fit__n_sing', 'base__scramble', 'fit__scramble',
            'base__amp_lc', 'fit__amp_lc', 'base__period', 'fit__period',
            'base__mu', 'fit__mu', 'seconds', 'seed_used')
_FLAGS = ('collapsed', 'improved', 'less_twist', 'quality_fit', 'off_regime')
#: must be IDENTICAL across every run joined -- see "WHAT IT REFUSES TO DO"
_MUST_MATCH = ('model', 'target', 'mode', 'n_phase', 'optimizer')

VERDICTS = ('RADIALIZED', 'PARTIAL', 'NO PROGRESS', 'UNUSABLE', 'DEGENERATE', 'NO DIAGNOSIS')
#: RADIALIZED green, every failure mode its own colour -- they are not one category
VERDICT_COLOR = {'RADIALIZED': '#1a7f37', 'PARTIAL': '#9a6700', 'NO PROGRESS': '#57606a',
                 'UNUSABLE': '#bc4c00', 'DEGENERATE': '#b31d28', 'NO DIAGNOSIS': '#8250df'}
#: verdicts whose twist / S_crit / surface numbers are measurements
USABLE = ('RADIALIZED', 'PARTIAL')


def verdict(z):
    """The one-word verdict, re-derived from the SAME flags `fit.radial.run` printed.

    Re-derived rather than read back, because the run wrote its verdict only to stdout and a
    campaign's stdout is twenty SLURM logs. Every flag it decided from is in the npz.

    ONE VERDICT IS SPLIT OFF THAT `run` DID NOT SEPARATE. `collapsed` is the OR of three
    tests, and one of them -- `not (0 < mu < 0.99)` -- also fires on a NaN, which is what a
    FAILED DIAGNOSIS leaves behind (`_diagnose_safe`). So a run whose post-hoc orbit re-solve
    diverged lands in the same bucket as a genuinely dead clock. Those are different claims:
    one says the fit found a non-oscillator, the other says the DIAGNOSTIC did not converge and
    we therefore do not know what the fit found. Conflating them lets a tooling failure be
    counted as a scientific result, which is REPO_MAP hazard 15 exactly.

    So the two independent tests are asked FIRST, because they come from the cost's own path
    (which converged -- it produced the surface) rather than from the diagnostic re-solve:

      * `fp_res` -- did Newton reach a fixed point at all? If not, nothing about stability
        was measured.
      * `re_lambda <= 0` -- the physiological fixed point is STABLE, i.e. on the
        non-oscillating side of the Hopf bifurcation. That is a dead clock, on evidence, and
        it stays DEGENERATE whatever the Floquet diagnostic did.

    Only a collapse resting SOLELY on a non-finite mu is demoted to NO DIAGNOSIS.
    """
    coll = bool(z['collapsed'])
    if coll:
        if _f(z, 'fit__parts_fp_res', np.inf) >= 1e-6:
            return 'NO DIAGNOSIS'                      # stability was never measured
        if _f(z, 'fit__parts_re_lambda', np.nan) <= 0.0:
            return 'DEGENERATE'                        # measured, on the stable side
        if not np.isfinite(_f(z, 'fit__mu')):
            return 'NO DIAGNOSIS'                      # only the mu re-solve failed
        return 'DEGENERATE'                            # mu >= 0.99: not an attracting cycle
    if not bool(z['quality_fit']):
        return 'UNUSABLE'
    if bool(z['improved']) and bool(z['less_twist']):
        return 'RADIALIZED'
    if bool(z['improved']):
        return 'PARTIAL'
    return 'NO PROGRESS'


def find_runs(model, tag, analysis='fit_radial'):
    """Every per-seed / per-entry `radial_*.npz` belonging to campaign `tag`.

    A campaign entry directory is `<tag>__<axis-label>` and a seed subdirectory is
    `<entry>__seed<N>` (`fit.campaign._tag_for`, `fit.radial.run_seeds`), so one prefix matches
    both. `<tag>` itself is included as well, so a single-task campaign -- whose output has no
    suffix at all -- aggregates by exactly the same call.
    """
    root = os.path.join(paths.OUT, str(model), analysis)
    hits = []
    for d in sorted(glob.glob(os.path.join(root, tag + '*'))):
        name = os.path.basename(d)
        # prefix-match on the SEPARATOR, so tag 'genes' never swallows a campaign called
        # 'genes2' -- a silent extra run in the matrix is worse than no match at all
        if not os.path.isdir(d) or (name != tag and not name.startswith(tag + '__')):
            continue
        hits.extend(sorted(glob.glob(os.path.join(d, 'radial_*.npz'))))
    return hits


def _f(z, k, default=np.nan):
    if k not in z:
        return default
    try:
        return float(np.asarray(z[k]).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return default


def collect(model, tag, analysis='fit_radial', key='seed'):
    """Load every run of the campaign, validated, sorted by its key.

    `key='seed'` labels runs by `seed_used` (a multi-seed campaign); `key='target'` by the
    perturbation target (one gene per task). The key decides only what the entries are called
    and which field has to be unique -- everything else is identical, which is why a gene sweep
    and a seed sweep share this function instead of there being two of it.
    """
    files = find_runs(model, tag, analysis)
    if not files:
        raise SystemExit(f"no runs found for {model}/{analysis} campaign tag {tag!r}; looked "
                         f"in {os.path.join(paths.OUT, model, analysis, tag + '*')}")
    runs, labels = [], []
    for f in files:
        z = dict(np.load(f, allow_pickle=True))
        z['_file'] = os.path.relpath(f, paths.HERE).replace('\\', '/')
        runs.append(z)
        labels.append(str(int(_f(z, 'seed_used', -1))) if key == 'seed' else str(z['target']))

    # --- the refusals ------------------------------------------------------------------ #
    ref = runs[0]
    for name in _MUST_MATCH:
        if key == 'target' and name == 'target':
            continue                          # that IS the axis being swept
        vals = sorted({str(r[name]) for r in runs if name in r})
        if len(vals) > 1:
            raise SystemExit(f"cannot aggregate: runs disagree on {name!r} -> {vals}. These "
                             f"are not the same experiment, and a distance matrix over them "
                             f"would not mean anything.")
    if len(set(labels)) != len(labels):
        dup = sorted({x for x in labels if labels.count(x) > 1})
        raise SystemExit(f"cannot aggregate: duplicate {key}(s) {dup}. The same {key} twice is "
                         f"not two independent searches; delete or re-tag the stale run.")
    if key == 'seed':
        d0 = np.asarray(ref['doses'], float)
        for r in runs:
            if not np.allclose(np.asarray(r['doses'], float), d0):
                raise SystemExit(f"cannot aggregate: {r['_file']} uses a different dose grid.")
            # THE GAUGE-QUOTIENT BASIS MUST BE THE SAME ONE, and it is not guaranteed to be.
            # `quotient_basis` SVDs a projector whose nonzero singular values are all exactly 1,
            # so any orthonormal basis of that subspace is a valid answer and LAPACK's choice is
            # not reproducible. Two runs with different B put their `v` in different coordinates,
            # and a distance between them is then meaningless -- measured, the same `v` under a
            # re-derived basis reconstructs parameters up to 3.0 decades away. Every run of the
            # Aug-30 campaign happens to share one basis; that is checked here, not assumed.
            if 'B' in r and 'B' in ref and not np.allclose(np.asarray(r['B']),
                                                           np.asarray(ref['B'])):
                raise SystemExit(
                    f"cannot aggregate: {r['_file']} used a DIFFERENT gauge-quotient basis. "
                    f"Its `v` is in different coordinates, so distances across these runs would "
                    f"not mean anything. Compare `theta_fit` instead, or re-run them together.")
            for kk in ('k_used', 'psi_used'):
                a, b = _f(r, kk), _f(ref, kk)
                if np.isfinite(a) != np.isfinite(b) or (np.isfinite(a)
                                                        and not np.isclose(a, b)):
                    raise SystemExit(
                        f"cannot aggregate: {r['_file']} was scored against a DIFFERENT target "
                        f"({kk} {a} vs {b}). Costs measured against different targets are not "
                        f"comparable, so neither is a ranking of them.")

    order = sorted(range(len(runs)),
                   key=lambda i: (int(labels[i]) if key == 'seed' else labels[i]))
    return [runs[i] for i in order], [labels[i] for i in order]


def build(model, tag, analysis='fit_radial', key='seed'):
    """The campaign-level blob: scalars, flags, both twist metrics, and the raw surfaces."""
    runs, labels = collect(model, tag, analysis, key)
    n = len(runs)
    ref = runs[0]
    blob = {k: np.asarray(ref[k]) for k in ('old', 'names', 'model', 'mode',
                                            'optimizer', 'section', 'readout')}
    # `doses` IS PER RUN, ALWAYS, EVEN WHEN THE RUNS SHARE A GRID.
    #
    # Storing one grid for the campaign was wrong and wrong SILENTLY: `fit_dose_grid` derives
    # the window from the target's OWN S_crit, so a per-gene campaign has four different dose
    # axes -- BMAL1 12.5-200, REV 778-12460, a factor of 60 apart -- and a figure that drew all
    # four genes against the first one's axis put every surface at the wrong dose without
    # anything looking wrong. A seed campaign's grids ARE identical (`collect` refuses to join
    # them otherwise), but keeping one shape for both cases means the consumer can never pick
    # up the wrong axis by forgetting which campaign it has.
    blob['doses'] = np.array([np.asarray(r['doses'], float) for r in runs])
    blob['campaign_tag'] = np.asarray(str(tag))
    blob['key'] = np.asarray(str(key))
    blob['labels'] = np.array([str(x) for x in labels])
    blob['n_runs'] = np.asarray(n)
    blob['files'] = np.array([r['_file'] for r in runs])
    blob['target'] = (np.asarray(ref['target']) if key == 'seed'
                      else np.array([str(r['target']) for r in runs]))
    blob['seeds'] = np.array([int(_f(r, 'seed_used', -1)) for r in runs])

    for k in _SCALARS:
        blob[k] = np.array([_f(r, k) for r in runs])
    for k in _FLAGS:
        blob[k] = np.array([bool(r[k]) if k in r else False for r in runs])
    blob['verdict'] = np.array([verdict(r) for r in runs])

    # --- the unsaturated twist metrics, from the saved curves --------------------------- #
    for pref in ('base', 'fit'):
        tw = np.array([np.asarray(r['twist_' + pref], float) for r in runs])
        blob['twist_' + pref] = tw
        blob[pref + '__accum_twist'] = np.array([W.accumulated_twist(t) for t in tw])
        blob[pref + '__signed_twist'] = np.array([W.signed_twist(t) for t in tw])
    # A sign flip between base and fit is evidence of DOSE UNDER-SAMPLING, not of a fit that
    # reversed the spiral: aliasing inverts the direction before it destroys the magnitude
    # (PROJECT_SUMMARY 5.9c). Flagged per run so no twist number gets quoted without it.
    blob['twist_sign_flip'] = (blob['base__signed_twist'] * blob['fit__signed_twist']) < 0

    # --- RAW: every surface, so no figure ever needs a fit re-run ----------------------- #
    blob['ptc_fit'] = np.array([np.asarray(r['ptc_fit'], float) for r in runs])
    blob['ptc_base'] = np.array([np.asarray(r['ptc_base'], float) for r in runs])
    blob['ptc_target_used'] = np.array([np.asarray(r['ptc_target_used'], float) for r in runs])
    blob['alive_fit'] = np.array([np.asarray(r['alive_fit']) for r in runs])
    blob['k_used'] = np.array([_f(r, 'k_used') for r in runs])
    blob['psi_used'] = np.array([_f(r, 'psi_used') for r in runs])
    blob['theta_base'] = np.array([np.asarray(r['theta_base'], float) for r in runs])
    blob['theta_fit'] = np.array([np.asarray(r['theta_fit'], float) for r in runs])
    blob['v'] = np.array([np.asarray(r['v_fit'], float) for r in runs])
    blob['cost'] = blob['fit__parts_total'].copy()

    # descent traces, NaN-padded: they are equal length here only because every run stopped on
    # maxfev, which is not something to rely on
    L = max(int(np.asarray(r['trace_f']).size) for r in runs)
    tr = np.full((n, L), np.nan)
    for i, r in enumerate(runs):
        t = np.asarray(r['trace_f'], float).ravel()
        tr[i, :t.size] = t
    blob['trace_f'] = tr

    # --- the cross-run distance matrix, IN THE GAUGE QUOTIENT --------------------------- #
    # `v` already lives in the quotient (fit.cost.quotient_basis), so a plain Euclidean norm
    # here IS the gauge-quotiented distance: two parameter sets differing only by a change of
    # units sit at distance 0, which is the whole point (REPO_MAP hazard 6). Distances between
    # runs with different TARGETS are meaningless, so a gene campaign gets NaN instead.
    v = blob['v']
    blob['dist'] = (np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1) if key == 'seed'
                    else np.full((n, n), np.nan))
    # the basis and origin the runs actually used -- without them a saved `v` cannot be
    # turned back into parameters by anything but the machine that wrote it
    if 'B' in ref:
        blob['B'] = np.asarray(ref['B'])
    if 'z_base' in ref:
        blob['z_base'] = np.asarray(ref['z_base'])
    blob['cfg_json'] = np.asarray(str(ref['cfg_json']) if 'cfg_json' in ref else '')
    return blob


def merge_viability(model, tag, analysis='fit_radial'):
    """Join the per-task rejection-sampling records into one survey.

    Every REJECTED draw is a viability measurement of a random parameter set, accumulated for
    free while seeding (`fit.radial._save_viability`). Split across tasks it is four small
    samples with four hit rates; joined it is one survey with one, which is the only form in
    which the rate means anything.
    """
    root = os.path.join(paths.OUT, str(model), analysis)
    fs = sorted(glob.glob(os.path.join(root, tag + '*', 'viability_*.npz')))
    if not fs:
        return None
    parts = [dict(np.load(f, allow_pickle=True)) for f in fs]
    out = {}
    for k in ('seeds', 'found', 'draws', 'period', 'min_ratio', 'norm',
              'probe_seed', 'probe_norm', 'probe_ok'):
        vals = [np.asarray(p[k]).ravel() for p in parts if k in p]
        if vals:
            out[k] = np.concatenate(vals)
    if 'draws' in out:
        out['hit_rate'] = np.asarray(float(np.sum(out['found']))
                                     / max(int(np.sum(out['draws'])), 1))
        out['n_tasks'] = np.asarray(len(parts))
    return out


def best_index(b):
    """Index of the cheapest run whose numbers are measurements.

    NOT `argmin(cost)`. A DEGENERATE or NO-DIAGNOSIS run can score arbitrarily well -- that is
    the failure the whole cost was built against (PROJECT_SUMMARY 5.2) -- so 'best' has to mean
    'best among the runs that survived the gates', and falls back to the raw argmin only when
    nothing survived, with the verdict printed alongside so it cannot be misread.
    """
    ok = np.array([v in USABLE for v in b['verdict']])
    cost = np.asarray(b['cost'], float)
    return int(np.argmin(np.where(ok, cost, np.inf))) if ok.any() else int(np.nanargmin(cost))


def report(b, ref_cost=None, ref_label=''):
    """The text answer to 'one optimum or several?', over the WHOLE campaign."""
    n, key = int(b['n_runs']), str(b['key'])
    lab, cost, D = b['labels'], np.asarray(b['cost'], float), np.asarray(b['dist'], float)
    ok = np.array([v in USABLE for v in b['verdict']])
    bi = best_index(b)
    print("\n" + "=" * 94)
    print(f"CAMPAIGN {b['campaign_tag']} -- {n} runs, keyed by {key}")
    print("=" * 94)
    # The distance column is DROPPED, not printed as NaN, when distances are undefined: a gene
    # campaign's runs were scored against different targets, so "how far apart are they in
    # parameter space" is not a question with an answer, and a column of NaN invites one.
    has_d = bool(np.isfinite(D).any())
    print(f"  {key:>7s} {'cost':>9s} {'c_ptc':>9s} {'tw_acc':>8s} {'tw_sgn':>8s} "
          f"{'S_fit':>9s} {'T_cost':>8s} {'mu':>9s}" + (f" {'d(best)':>8s}" if has_d else '')
          + "  verdict")
    for i in np.argsort(cost):
        flag = ' *' if b['twist_sign_flip'][i] else '  '
        print(f"  {str(lab[i]):>7s} {cost[i]:9.4f} {b['fit__parts_c_ptc'][i]:9.4f} "
              f"{b['fit__accum_twist'][i]:8.3f} {b['fit__signed_twist'][i]:+8.3f}{flag}"
              f"{b['fit__S_crit'][i]:9.4g} {b['fit__parts_period'][i]:8.2f} "
              f"{b['fit__mu'][i]:9.4g}" + (f" {D[i, bi]:8.3f}" if has_d else '')
              + f"  {b['verdict'][i]}" + ('   <- best usable' if i == bi else ''))
    print(f"\n  base: cost {np.nanmedian(b['base__parts_total']):.4f}, accumulated twist "
          f"{np.nanmedian(b['base__accum_twist']):.3f} cyc, period "
          f"{np.nanmedian(b['base__parts_period']):.2f} h")
    if ref_cost is not None and np.isfinite(ref_cost):
        print(f"  reference ({ref_label}): cost {ref_cost:.4f}")
    for v in VERDICTS:
        c = int(np.sum(b['verdict'] == v))
        if c:
            print(f"  {v:<14s} {c:2d}/{n}")

    if ok.sum() >= 2 and np.isfinite(D).any():
        cu = cost[ok]
        off = D[np.ix_(ok, ok)][np.triu_indices(int(ok.sum()), 1)]
        print(f"\n  USABLE runs only ({int(ok.sum())} of {n}): cost {cu.min():.4f}-"
              f"{cu.max():.4f}, spread {cu.max() - cu.min():.4f} "
              f"({cu.max() / max(cu.min(), 1e-12):.2f}x)")
        print(f"  pairwise quotient distance: min {off.min():.3f}, median "
              f"{np.median(off):.3f}, max {off.max():.3f}")
        print(f"  -> the cost spread is {(cu.max() - cu.min()) / max(np.median(cu), 1e-12):.0%} "
              f"of the median cost, across a median separation of {np.median(off):.2f}.")
        print("  Comparable cost at large separation = MULTIMODAL; a tight cluster = one "
              "optimum, found repeatedly.")

    nsz = int(np.sum(np.asarray(b['fit__n_sing'])[ok] == 0)) if ok.any() else 0
    if nsz:
        where = (f" -- here at S={1 / np.nanmedian(np.asarray(b['k_used'])[ok]):.4g}"
                 if key == 'seed' else ' (at its own pinned dose, per run)')
        print(f"\n  NOTE: {nsz} of {int(ok.sum())} usable fits have NO singularity left in the "
              f"dose window (n_sing 0). The residual is then scoring a surface whose TOPOLOGY "
              f"differs from the target's: the target carries a defect by "
              f"construction{where}. 'The isochrons went flat' and 'the transition moved out "
              f"of the dose window' are not the same result, and a pointwise cost does not "
              f"separate them.")
    nf = int(np.sum(b['twist_sign_flip']))
    if nf:
        print(f"\n  CAVEAT: {nf} run(s) marked * flip the twist SIGN between base and fit. On "
              f"a {np.asarray(b['doses']).shape[-1]}-point dose axis that is the under-sampling "
              f"signature of PROJECT_SUMMARY 5.9c, not a measurement of direction.")
    return bi


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('WHY THIS')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--tag', required=True, help='campaign tag, e.g. bmal1seeds')
    ap.add_argument('--kind', default='seeds', choices=('seeds', 'genes'),
                    help="'seeds': one entry per seed. 'genes': one entry per target.")
    ap.add_argument('--analysis', default='fit_radial')
    ap.add_argument('--out-tag', default=None,
                    help='tag to write under (default: the campaign tag itself)')
    ap.add_argument('--list', action='store_true', help='list the runs that would join, exit')
    a = ap.parse_args(argv)

    if a.list:
        for f in find_runs(a.model, a.tag, a.analysis):
            print('  ' + os.path.relpath(f, paths.HERE))
        return 0

    key = 'seed' if a.kind == 'seeds' else 'target'
    b = build(a.model, a.tag, a.analysis, key)
    via = merge_viability(a.model, a.tag, a.analysis)
    if via:
        print(f"[viability] {int(np.sum(via['draws']))} draws over {int(via['n_tasks'])} "
              f"task(s), {int(np.sum(via['found']))} accepted "
              f"(rate {float(via['hit_rate']):.3%})")
        for k, v in via.items():
            b['viability__' + k] = np.asarray(v)
    report(b)

    name = (f"seeds_{b['target']}_{b['mode']}.npz" if key == 'seed'
            else f"{a.kind}_{b['mode']}.npz")
    out = paths.out_path(a.model, a.analysis, name, a.out_tag or a.tag)
    paths.savez(out, **b)
    print(f"\n[aggregate] -> {os.path.relpath(out, paths.HERE)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
