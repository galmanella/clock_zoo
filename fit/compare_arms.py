"""
fit/compare_arms.py
===================
Compare several campaigns that used DIFFERENT objectives, on one common yardstick.

    $PY -m fit.compare_arms --model almeida --arms arm0_ctl arm1_ramp arm2_brack \
        arm3_inform arm4_floor --control arm0_ctl

WHY `fit.aggregate` IS NOT ENOUGH
    It joins the array tasks WITHIN one campaign, which is the right unit when every task
    minimised the same function. An arm comparison is the other case: each arm minimised a
    DIFFERENT function, so the cost each one reports is denominated in its own objective and
    the numbers are not comparable. Ranking arms by their own `c_ptc` would reward whichever
    arm happens to have the most permissive cost -- which, since the whole point of these arms
    is to make the cost LESS permissive, would rank them exactly backwards.

THE YARDSTICK
    Every endpoint is re-scored under the CONTROL arm's objective, on the CONTROL arm's grid.
    That is the only number in the table that means the same thing in every row. The arm's own
    cost is shown beside it, labelled as such, because the difference between the two is itself
    informative: a large gap means the arm's extra terms are doing most of the work at its
    optimum.

    Alongside it, the contract (fit.contract) on every endpoint. That is the PRIMARY outcome:
    the arms exist to stop the optimizer reaching surfaces that are not PTCs, and the contract
    is what says whether it did. A lower yardstick cost on an endpoint that fails C5 or C6 is
    not an improvement.

WHAT IT DELIBERATELY DOES NOT DO
    Declare a winner. With 3 seeds per cell the spread within an arm is usually comparable to
    the difference between arms, so it prints the per-seed values and the spread rather than a
    mean with a confidence interval it cannot support. Read the contract columns first.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

import paths


#: two scorings of the same endpoint may differ by this much before it is called unstable.
#: Well-behaved endpoints reproduce to ~1e-13; the fragile ones jump by ~0.7.
YARD_TOL = 1e-3
#: a fixed direction, so the "is this endpoint well defined" test is reproducible run to run
_JIG = np.random.default_rng(20260831).normal(size=256)


def _load(model, tag, analysis='fit_radial'):
    d = paths.out_dir(model, analysis, tag, create=False)
    fs = sorted(glob.glob(os.path.join(d, 'seeds_*.npz')))
    if not fs:
        return None
    return dict(np.load(fs[-1], allow_pickle=True))


def _arm_dirs(model, tag, analysis='fit_radial'):
    """The per-run directories of one arm, which `fit.aggregate` may not have joined yet.

    An arm's six tasks each write their own directory, and they are NOT one aggregate: the arm
    varies `start`, so the six are two groups of three and joining all six would compare runs
    that began in different places. So this walks the directories directly.
    """
    root = os.path.join(paths.OUT, str(model), analysis)
    out = []
    for d in sorted(glob.glob(os.path.join(root, tag + '__*'))):
        fs = sorted(glob.glob(os.path.join(d, 'radial_*.npz')))
        if fs:
            out.append((os.path.basename(d), fs[-1]))
    return out


def _start_of(dirname):
    for part in dirname.split('__')[1:]:
        for token in part.split('_'):
            if token.startswith('start-'):
                return token[len('start-'):]
    return '?'


def _seed_of(z):
    try:
        return str(int(float(np.asarray(z['seed_used']).ravel()[0])))
    except Exception:
        return '?'


def compare(model, arms, control, analysis='fit_radial', with_contract=True):
    from models import get_model
    from fit.cost import make_cost, RadialTarget
    from fit.doses import fit_dose_grid
    if control not in arms:
        raise SystemExit(f"--control {control!r} is not one of --arms {arms}")

    # --- build the CONTROL objective once; every endpoint is re-scored on it ------------ #
    ctl_runs = _arm_dirs(model, control, analysis)
    if not ctl_runs:
        raise SystemExit(f"no runs for the control arm {control!r} under "
                         f"{os.path.join(paths.OUT, model, analysis)}")
    z0 = dict(np.load(ctl_runs[0][1], allow_pickle=True))
    cfg0 = json.loads(str(z0['cfg_json']))
    m = get_model(model)
    if cfg0.get('section'):
        m.reference_variable = cfg0['section']
    doses0, _S = fit_dose_grid(model, cfg0['target'], cfg0['mode'], cfg0['max_factor'],
                               cfg0['n_dose'], lo_factor=cfg0['lo_factor'],
                               include_zero=bool(cfg0.get('include_zero', False)))
    tgt0 = RadialTarget.from_singularity(1.0 / float(z0['k_used']),
                                         (float(z0['psi_used']) + 0.5) % 1.0)
    # THE CONTROL OBJECTIVE, EXPLICITLY: every new term off, whatever the control config said.
    # Reconstructing it from cfg0 would silently inherit any option the control itself set.
    Cy = make_cost(m, cfg0['target'], doses0, tgt0, n_phase=cfg0['n_phase'],
                   mode=cfg0['mode'], backend=cfg0['backend'], dt=cfg0['dt'],
                   w_osc=cfg0['w_osc'], w_amp=cfg0['w_amp'], pulse=cfg0['pulse'],
                   skip_p=cfg0.get('skip_p'), readout_ref=cfg0.get('readout'),
                   basis=np.asarray(z0['B']),
                   amp_ramp=None, row_weight=False, w_brack=0.0)

    rows = []
    for arm in arms:
        for dirname, f in _arm_dirs(model, arm, analysis):
            z = dict(np.load(f, allow_pickle=True))
            th = np.asarray(z['theta_fit'], float)
            # REPROJECT into the yardstick's basis. theta is the invariant; `v` is not
            # portable across bases (hazard 18), and these runs may have been built on a
            # different one.
            zb = np.asarray(z0['z_base'], float)
            B0 = np.asarray(z0['B'], float)
            u = np.log(th) - zb
            v = B0.T @ u
            drift = float(np.max(np.abs(np.log10(np.exp(zb + B0 @ v) / th))))
            r = dict(arm=arm, start=_start_of(dirname), seed=_seed_of(z),
                     own_cost=float(z['fit__parts_total']),
                     own_c_ptc=float(z['fit__parts_c_ptc']),
                     drift=drift, dirname=dirname)
            if drift > 1e-6:
                r['yard'] = r['yard_alt'] = np.nan   # outside the yardstick's quotient
            else:
                # SCORE IT TWICE, THE SECOND TIME AT A 1e-12 RELATIVE JIGGLE.
                #
                # Some endpoints are not a well-defined point of the cost at double precision.
                # MEASURED on bmal1seeds seed 0: reprojecting theta into its OWN basis moves `v`
                # by 3.55e-15 -- pure round-off -- and the surface flips from alive_frac 1.000,
                # c_ptc 0.313 to alive_frac 0.000, c_ptc 1.000. The BVP solve lands on a
                # different branch, every cell dies, and the cost jumps by 0.69. That parameter
                # set has |grad| = 3e10 and a numerical floor of 2.6e-4, so this is the same
                # fragility seen from a third direction (PROJECT_SUMMARY 5.11d).
                #
                # A single evaluation there returns a coin flip. Two evaluations cannot make it
                # well defined, but they can REFUSE TO REPORT IT AS A NUMBER, which is the
                # difference between a comparison and a fiction.
                r['yard'] = float(Cy['parts'](v)['c_ptc'])
                jig = v + 1e-12 * np.maximum(np.abs(v), 1.0) * _JIG[:len(v)]
                r['yard_alt'] = float(Cy['parts'](jig)['c_ptc'])
                # AND, WHEN THE RUN SHARES THE YARDSTICK'S BASIS, score its OWN stored `v` too.
                #
                # The jiggle alone is not enough, and seed 0 is why: at the REPROJECTED v its
                # surface is consistently dead, so two jiggled evaluations agree perfectly at
                # 1.000 and the endpoint looks stable. It is stable and WRONG -- the run itself
                # recorded 0.313 from a `v` differing by 3.55e-15. Round-off moved it across a
                # basin boundary of the BVP solve, not across a noisy patch, so the test has to
                # compare the two representations rather than probe one of them twice.
                if np.allclose(np.asarray(z.get('B', B0), float), B0, atol=0, rtol=1e-12):
                    r['yard_vfit'] = float(Cy['parts'](np.asarray(z['v_fit'], float))['c_ptc'])
                else:
                    r['yard_vfit'] = np.nan
            d_jig = abs(r['yard'] - r['yard_alt'])
            d_rep = (abs(r['yard'] - r['yard_vfit'])
                     if np.isfinite(r.get('yard_vfit', np.nan)) else 0.0)
            r['unstable'] = bool(np.isfinite(r['yard'])
                                 and max(d_jig, d_rep) > YARD_TOL)
            if with_contract:
                from fit.contract import evaluate, CHECKS, fmt
                c = evaluate(Cy, m, v)
                r['contract'] = fmt(c)
                r['n_fail'] = sum(1 for k in CHECKS if c[k] is False)
                r.update({k: c[k] for k in CHECKS})
            rows.append(r)
    return rows, cfg0


def report(rows, control):
    arms = sorted({r['arm'] for r in rows})
    starts = sorted({r['start'] for r in rows})
    print(f"\n{'=' * 100}")
    print("ARM COMPARISON.  'yardstick' = every endpoint re-scored under the CONTROL objective")
    print(f"                 on the CONTROL grid -- the only column comparable across arms.")
    print(f"{'=' * 100}")
    print(f"  {'arm':<12s}{'start':<16s}{'seed':>5s}{'yardstick':>11s}{'own cost':>10s}"
          f"{'own c_ptc':>11s}  {'C1..C6':<8s} fails")
    for st in starts:
        for arm in arms:
            sel = [r for r in rows if r['arm'] == arm and r['start'] == st]
            for r in sorted(sel, key=lambda x: x['seed']):
                mark = '  <- control' if arm == control else ''
                flag = ' !UNSTABLE' if r.get('unstable') else ''
                print(f"  {arm:<12s}{st:<16s}{r['seed']:>5s}{r['yard']:11.5f}"
                      f"{r['own_cost']:10.5f}{r['own_c_ptc']:11.5f}  "
                      f"{r.get('contract', '-'):<8s} {r.get('n_fail', '-')}{mark}{flag}")
        print()
    print("  ('.' passes, 'X' fails, '-' not run;  order C1 C2 C3 C4 C5 C6.  C2 is post hoc:")
    print("   run fit.stability per arm for it.)\n")

    # per (arm, start) summary -- spread, not a mean with an interval 3 seeds cannot support
    print(f"  {'arm':<12s}{'start':<16s}{'yardstick min..max':>22s}{'contract failures':>20s}")
    for st in starts:
        for arm in arms:
            sel = [r for r in rows if r['arm'] == arm and r['start'] == st]
            if not sel:
                continue
            y = np.array([r['yard'] for r in sel], float)
            nf = [r.get('n_fail') for r in sel if r.get('n_fail') is not None]
            print(f"  {arm:<12s}{st:<16s}"
                  f"{np.nanmin(y):10.5f} .. {np.nanmax(y):<9.5f}"
                  f"{(str(min(nf)) + '-' + str(max(nf)) + ' of 5') if nf else '-':>20s}")
        print()
    uns = [r for r in rows if r.get('unstable')]
    if uns:
        print(f"  !UNSTABLE ({len(uns)} of {len(rows)}): re-scoring the SAME endpoint -- after a "
              f"1e-12 jiggle, or from its own stored `v` rather than its theta -- moves the "
              f"yardstick cost by more than {YARD_TOL:g}. The "
              f"cost is not a well-defined function there at double precision, so that row's "
              f"number is a coin flip and MUST NOT be ranked. Check it with fit.stability "
              f"(the numerical floor) before drawing anything from it.\n")
    bad = [r for r in rows if r['drift'] > 1e-6]
    if bad:
        print(f"  WARNING: {len(bad)} endpoint(s) could not be represented in the yardstick's "
              f"gauge quotient (drift > 1e-6 decades) and are scored NaN. Their theta has a "
              f"component outside span(B); they are not comparable and must not be ranked.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('WHY `fit.aggregate`')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--control', required=True)
    ap.add_argument('--analysis', default='fit_radial')
    ap.add_argument('--no-contract', action='store_true')
    a = ap.parse_args(argv)
    rows, _cfg = compare(a.model, a.arms, a.control, a.analysis, not a.no_contract)
    if not rows:
        raise SystemExit("no runs found for any arm -- has the array finished, and has out/ "
                         "been rsynced back? (out/ is gitignored; results do not travel "
                         "through git.)")
    report(rows, a.control)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
