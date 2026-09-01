"""
fit/radial.py
=============
T3: fit a clock model so its single-gene PTC matches a RADIAL-ISOCHRON (Poincare) target.

    $PY -m fit.radial --model almeida --target BMAL1 [--starts 8]

    RUN fit/recover.py FIRST. If the optimizer cannot recover the model's own displaced
    parameter set (an in-class target, guaranteed reachable), it certainly cannot fit an
    out-of-class one, and a failure here would be uninterpretable -- model-cannot-do-it and
    optimizer-cannot-find-it look identical.

WHAT A SUCCESS AND A FAILURE EACH MEAN
    The radial target is almost certainly NOT exactly in Almeida's class: nothing says an
    8-state transcription-factor model can produce perfectly radial isochrons. So the honest
    question is not "does the residual reach zero" but "how close can it get, and does the
    isochron geometry actually become more radial on the way".

    That distinction is why this driver reports twist alongside the residual. Twist is the
    gauge-invariant, smooth measure of isochron shear (REPO_MAP hazard 4 -- do NOT read
    isochron change off the singularity, which is a grid-quantized topological defect). A fit
    that lowers the residual while leaving twist untouched has matched the target's dose
    scaling, not its geometry.

THE FAILURE MODE THIS DRIVER IS BUILT TO DETECT
    input_screen's radialization run reported a converged cost and a twist improved from 0.97
    to 0.36 -- and it was meaningless, because the optimizer had walked the clock toward a Hopf
    bifurcation. The tell was not the cost and not the singularity count; it was the LIMIT
    CYCLE AMPLITUDE going flat. So this driver reports, before and after:

        limit-cycle amplitude      the diagnostic that actually caught it
        leading Floquet multiplier how close to losing stability
        the quality gate           on the fitted surface, so a scrambled "improvement" is
                                   rejected the way CRY was
        twist                      the thing radialization is supposed to change
        cost split by term         so a good total can be attributed

    A twist improvement accompanied by amplitude collapse is reported as a FAILURE, not as a
    result.
"""
import argparse
import json
import time

import numpy as np

import paths
from fit.config import RunConfig, describe, start_points
from fit.cost import make_cost, RadialTarget
from fit.doses import fit_dose_grid
from fit import search


def _diagnose_safe(model, C, v, label):
    """`_diagnose`, but a failure DEGRADES the report instead of destroying the run.

    REPO_MAP hazard 7, violated by this very function. Diagnosis happens AFTER the optimizer
    has finished, so an exception here throws away the whole fit -- and it did: a completed
    BMAL1 campaign task died in `solver.floquet` on a non-finite monodromy
    (LinAlgError: Array must not contain infs or NaNs) and hours of compute went with it.

    The fit result is the expensive, irreplaceable thing; twist and Floquet numbers are
    commentary on it. So a broken diagnosis returns NaNs and an `error` string, the verdict
    says the diagnosis failed, and the raw surfaces and parameters are still written to disk.
    """
    try:
        return _diagnose(model, C, v, label)
    except Exception as exc:
        import traceback
        print(f"{chr(10)}  DIAGNOSIS FAILED for '{label}': "
              f"{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        print("  The FIT ITSELF is unaffected and will still be saved; only the derived "
              "twist / S_crit / Floquet numbers are missing.", flush=True)
        n_ph, n_d = len(C['old']), len(C['doses'])
        nan2 = np.full((n_ph, n_d), np.nan)
        try:
            zz, aa, am = C['surface'](v)
            ptc = np.where(aa, (np.angle(zz) / (2 * np.pi)) % 1.0, np.nan)
        except Exception:
            ptc, aa, am = nan2, np.zeros((n_ph, n_d), bool), nan2
        try:
            parts = C['parts'](v)
        except Exception:
            parts = {}
        return dict(label=label, ptc=ptc, alive=aa, amp=am,
                    twist=np.full(n_d, np.nan), ptc_target=nan2,
                    k_target=float('nan'), psi_target=float('nan'),
                    cyc=np.zeros((0, 0)), obs=[], total_twist=float('nan'),
                    S_crit=float('nan'), phi_sing=float('nan'), n_sing=-1,
                    amp_lc=parts.get('amp_lc', float('nan')),
                    period=parts.get('period', float('nan')), mu=float('nan'),
                    quality_pass=False, scramble=float('nan'), winding_set=[],
                    parts=parts, q={'passed': False, 'failures': [f'diagnosis failed: {exc}']},
                    error=f'{type(exc).__name__}: {exc}')


def _diagnose(model, C, v, label):
    """Everything needed to tell a real improvement from a dying clock."""
    from analysis import winding as W
    from analysis import quality as Q

    z, alive, amp = C['surface'](v)
    ptc = (np.angle(z) / (2 * np.pi)) % 1.0
    ptc = np.where(alive, ptc, np.nan)
    old, doses = C['old'], C['doses']
    tw = W.twist_curve(old, doses, ptc)
    S, phi, nsing = W.detect_grid(old, doses, ptc)
    q = Q.score(old, doses, ptc)
    p = C['parts'](v)

    # leading Floquet multiplier: how close the cycle is to losing stability
    import jax
    import jax.numpy as jnp
    P = model.jax_apply(C['theta'](v), C['names'])
    solver = C['solver']
    y0, T, _r = jax.jit(solver.solve)(P, solver.guess(P))
    mu, _ev = solver.floquet(P, y0, T)

    # The TARGET this surface was scored against. RadialTarget profiles (k, psi) per evaluation,
    # so the target is not fixed -- it is whatever registration best matched THIS surface, and
    # without storing it a figure cannot show what was actually being fitted.
    from fit.target import profile as _profile, radial_z as _radial_z
    kk, pp, _cc = _profile(jnp.asarray(z), jnp.asarray(alive), old, doses)
    zt = np.asarray(_radial_z(old, doses, kk, pp))
    ptc_target = (np.angle(zt) / (2 * np.pi)) % 1.0

    # every observable species over one period, in REAL time
    obs = list(model.observable_states())
    oidx = [int(model.var_index(sname)) for sname in obs]
    cyc_all = np.asarray(solver.cycle(P, y0, T, 256))[:, oidx]

    return dict(label=label, ptc=ptc, alive=alive, amp=amp, twist=tw,
                ptc_target=ptc_target, k_target=float(kk), psi_target=float(pp),
                cyc=cyc_all, obs=obs,
                total_twist=W.total_twist(tw), S_crit=S, phi_sing=phi, n_sing=nsing,
                amp_lc=p['amp_lc'], period=float(T), mu=float(np.asarray(mu)),
                quality_pass=q['passed'], scramble=q['scramble'],
                winding_set=q['winding_set'], parts=p, q=q)


def _report(d, amp_base):
    p = d['parts']
    print(f"  {d['label']:9s} cost={p['total']:.4f} (ptc {p['c_ptc']:.4f} + osc {p['osc']:.4f} "
          f"+ amp {p['amp_pen']:.4f})")
    print(f"            twist={d['total_twist']:.4f} S_crit={d['S_crit']:.4g} "
          f"phi*={d['phi_sing']:.3f} n_sing={int(d['n_sing'])}")
    print(f"            amp_lc={d['amp_lc']:.3f} ({d['amp_lc'] / amp_base:.1%} of base) "
          f"T={d['period']:.3f} mu={d['mu']:.4f}  "
          f"quality={'PASS' if d['quality_pass'] else 'FAIL'} (scramble {d['scramble']:.4f})")


def run(cfg, seed=None, tag=None, v_start=None):
    """One radialization at ONE seed, fully described by `cfg` (see fit/config.RunConfig).

    Everything the run depends on comes from `cfg` and nothing is read from a module default,
    so two runs with equal configs are the same experiment and the .npz records which one it
    was. `seed` overrides cfg for the multi-seed loop in `run_seeds`.
    """
    from models import get_model

    cfg.validate()
    model_name, target, mode = cfg.model, cfg.target, cfg.mode
    n_phase, backend, dt = cfg.n_phase, cfg.backend, cfg.dt
    bound, w_osc, w_amp = cfg.bound, cfg.w_osc, cfg.w_amp
    optimizer, workers = cfg.optimizer, cfg.workers
    seed = cfg.seed_list[0] if seed is None else int(seed)
    n_starts, maxiter = 1, max(4, cfg.maxfev // 10)
    max_factor = cfg.max_factor
    pin_target = cfg.target_mode != 'profiled'

    model = get_model(model_name)
    # the observable is part of the run, not of whatever the model currently defaults to
    if cfg.section:
        model.reference_variable = cfg.section
    if cfg.readout:
        model.readout_variable = cfg.readout
    section = str(model.reference_variable)
    readout = str(getattr(model, 'readout_variable', None) or model.reference_variable)

    doses, s_crit = fit_dose_grid(model_name, target, mode, cfg.max_factor, cfg.n_dose,
                                  lo_factor=cfg.lo_factor, include_zero=cfg.include_zero)
    tag = paths.run_tag(cfg.tag if tag is None else tag)
    if cfg.verbose:
        print(describe(cfg), flush=True)

    from parallel import announce
    announce(analysis='fit.radial', model=model_name, target=target, mode=mode,
             backend=backend, optimizer=optimizer, n_phase=n_phase, n_dose=len(doses),
             S_crit=f'{s_crit:.3g}', dose_max=f'{doses.max():.3g}', starts=n_starts,
             maxiter=maxiter, tag=tag)

    # PIN THE TARGET TO THE SEED'S SINGULARITY, by default.
    #
    # The target's (k, psi) -- its critical dose and singular phase -- are not properties of the
    # model, so they have to come from somewhere. Re-profiling them at every evaluation makes
    # the target a function of the current parameters: `min` over a family is not smooth, so the
    # effective target jumps whenever the argmin changes branch, and costs at different
    # parameter sets are measured against different targets. On a landscape already rugged from
    # the spiral geometry that is a feedback loop worth avoiding.
    #
    # Pinning to the base run's own singularity asks the well-posed question -- flatten the
    # twist while holding the defect where it already is -- and is what input_screen's
    # radialize.py did (`make_radial_target(S_crit, phi_sing, ...)`).
    # ONE dict, so the probe surface and the scored surface can never drift apart. The probe
    # only locates the seed's singularity, but it must see the same surface the cost will.
    cost_opts = dict(amp_ramp=None if cfg.amp_lo is None else (cfg.amp_lo, cfg.amp_hi),
                     row_weight=cfg.row_weight, row_weight_floor=cfg.row_weight_floor,
                     w_brack=cfg.w_brack, brack_lo=cfg.brack_lo, brack_hi=cfg.brack_hi)
    C_probe = make_cost(model, target, doses, RadialTarget(), n_phase=n_phase, mode=mode,
                        backend=backend, dt=dt, w_osc=0.0, w_amp=0.0,
                        pulse=cfg.pulse, skip_p=cfg.skip_p, readout_ref=readout,
                        **{k: v for k, v in cost_opts.items() if k != 'row_weight'},
                        row_weight=False)   # the probe target is PROFILED; weighting needs pinned
    _zb, _ab, _amb = C_probe['surface'](C_probe['v0'])
    from analysis import winding as _W
    _ptc_b = np.where(_ab, (np.angle(_zb) / (2 * np.pi)) % 1.0, np.nan)
    S_seed, phi_seed, _ns = _W.detect_grid(np.asarray(C_probe['old']), doses, _ptc_b)
    if cfg.target_mode == 'explicit':
        # The user placed the target by hand. Useful for asking whether a gene CAN be
        # radialized toward a singularity somewhere other than its own -- a question the
        # pinned default cannot express.
        tgt = RadialTarget.from_singularity(cfg.target_scrit, cfg.target_phi)
        print(f"[radial] target EXPLICIT: S_crit={cfg.target_scrit:.4g}, "
              f"phi*={cfg.target_phi:.3f}  (seed's own is S={S_seed:.4g}, "
              f"phi*={phi_seed:.3f})", flush=True)
    elif pin_target and np.isfinite(S_seed) and np.isfinite(phi_seed):
        tgt = RadialTarget.from_singularity(S_seed, phi_seed)
        print(f"[radial] target PINNED to the seed singularity: "
              f"S_crit={S_seed:.4g}, phi*={phi_seed:.3f}  "
              f"(k={tgt.k:.5g}, psi={tgt.psi:.3f})", flush=True)
    else:
        tgt = RadialTarget()
        why = 'requested' if not pin_target else 'the seed has no detectable singularity'
        print(f"[radial] target PROFILED per evaluation ({why}) -- the target moves with the "
              f"model; read the fit accordingly", flush=True)

    C = make_cost(model, target, doses, tgt, n_phase=n_phase, mode=mode,
                  backend=backend, dt=dt, w_osc=w_osc, w_amp=w_amp,
                  pulse=cfg.pulse, skip_p=cfg.skip_p, readout_ref=readout, **cost_opts)
    before = _diagnose_safe(model, C, C['v0'], 'base')
    # The BASE surface has to be usable or nothing downstream means anything. A run was allowed
    # to proceed from a base that failed the gate (scramble 0.0563 at 16x10) and its "before"
    # twist and S_crit were therefore not measurements.
    if not before['quality_pass']:
        print(f"\n  WARNING: the BASE surface FAILS the PTC quality gate "
              f"(scramble {before['scramble']:.4f}). Its twist and S_crit are not "
              f"measurements, so any before/after comparison built on them is meaningless. "
              f"Raise --n-phase / --n-dose until the base passes before trusting this run.",
              flush=True)

    # the pinned target rendered on this grid, for the record and the figure
    if tgt.k is not None:
        from fit.target import radial_z as _rz
        _ptc_used = (np.angle(np.asarray(_rz(np.asarray(C['old']), doses,
                                             float(tgt.k), float(tgt.psi))))
                     / (2 * np.pi)) % 1.0
    else:
        _ptc_used = np.full((n_phase, len(doses)), np.nan)

    t0 = time.time()
    if optimizer == 'cma':
        # anneal, not ipop -- see fit/search.cma. The obstacle here is small-scale ruggedness,
        # not distinct basins, so a contracting sigma is what is called for.
        ev, ps = None, cfg.popsize
        if workers and workers > 1:
            from fit.parallel import PoolEvaluator, cost_spec, recommend_popsize
            # an EXPLICIT popsize always wins; the recommendation only fills a blank
            ps = cfg.popsize or recommend_popsize(C['n_free'], workers)
            ev = PoolEvaluator(cost_spec(
                model_name, target, doses, n_phase, mode=mode, backend=backend, dt=dt,
                target_k=(tgt.k if tgt.k is not None else None),
                target_psi=(tgt.psi if tgt.k is not None else None),
                section=section, readout=readout,
                pulse=cfg.pulse, skip_p=cfg.skip_p), workers)
            print(f"[radial] population parallelism: {workers} workers, popsize {ps} "
                  f"(oversubscribed so fast members fill the gaps behind a straggler)",
                  flush=True)
        try:
            runs = [search.cma(C, v0=v_start, bound=bound, seed=seed, mode=cfg.cma_mode,
                               sigma0=cfg.sigma0, maxfev=cfg.maxfev,
                               restarts=cfg.restarts, popsize=ps, evaluator=ev,
                               log_every=cfg.log_every, verbose=cfg.verbose)]
        finally:
            if ev is not None:
                ev.close()
    elif optimizer == 'lm':
        runs = [search.levenberg_marquardt(C, C['v0'], bound=bound, maxiter=maxiter)]
    elif optimizer == 'bobyqa':
        # MEASURED WINNER on the T1 self-recovery benchmark at a matched 1500-eval budget:
        # bobyqa 0.0104 against cma-anneal 0.0397 and cma-ipop 0.0674, in the same wall time
        # (684 s vs 667 s), and it stopped on MAXFUN rather than converging -- it was still
        # improving. That ordering is what the spiral picture predicts: a model-based trust
        # region fits a quadratic through interpolation points spread across a region wide
        # enough to average over the fine-scale ruggedness that stalls a gradient and that CMA
        # can only sample through.
        runs = [search.bobyqa(C, C['v0'], bound=bound, maxfev=cfg.maxfev,
                              seek_global=True, verbose=cfg.verbose)]
    elif n_starts > 1:
        runs = search.multistart(C, n_starts=n_starts, bound=bound, maxiter=maxiter, seed=seed)
    else:
        runs = [search.lbfgs(C, C['v0'], bound=bound, maxiter=maxiter, label='nominal')]
    best = runs[0] if optimizer in ('cma', 'bobyqa') else min(runs, key=lambda r: r['f'])
    secs = time.time() - t0
    after = _diagnose_safe(model, C, best['v'], 'fitted')

    print(f"\n{'=' * 78}\nRADIALIZATION -- {model_name}/{target} ({mode})\n{'=' * 78}")
    _report(before, C['amp_base'])
    _report(after, C['amp_base'])

    # THE DEGENERACY CHECK, AND IT MUST BE TWO-SIDED.
    #
    # The first version tested only `amp_lc < 0.5 * base`, because it was written against the
    # Mirsky failure -- the clock DYING. A CMA run then returned "RADIALIZED" on this:
    #
    #     period   24.83 h -> 0.159 h      (nine minutes)
    #     amp_lc    4.337  -> 195.680      (4512% of base)
    #     mu        0.546  -> 1.0000       (not an attracting cycle at all)
    #
    # Residual 0.258 -> 0.011 and twist 0.459 -> 0.213, both "improved", on a surface that is no
    # longer a circadian PTC. Escaping UPWARD is just as degenerate as collapsing, and mu -> 1
    # means the orbit solver is not returning a stable limit cycle in the first place. So every
    # bound is now two-sided and stability is checked explicitly.
    per_ratio = after['period'] / max(before['period'], 1e-12)

    # DEGENERATE == NOT OSCILLATING, decided by which side of the Hopf bifurcation the
    # physiological fixed point sits on. This replaced an amplitude-RATIO test, which was
    # wrong twice over. `amp_lc` is range / |mean| on ONE species, so (a) a species whose
    # BASELINE falls toward zero inflates it without the oscillation growing at all, and
    # (b) the verdict changes when the reference species changes, for the same fit.
    #
    # RAD03 was declared DEGENERATE for 'amplitude x2.54' when BMAL1's absolute range had
    # SHRUNK 6.4x (27.79 -> 4.34) while its mean fell 16x (6.41 -> 0.394). Every other
    # species was healthy or larger: PER 55 -> 86, PER_CRY 22 -> 68, CRY 15 -> 30.
    #
    # Whether a deterministic system still oscillates is not a question about amplitude. The
    # cost already answers it every evaluation as `re_lambda`, with `fp_res` saying whether
    # Newton reached a fixed point at all. MEASURED: base +0.06275, RAD03 +0.02507, RAD05
    # +0.01723, all at fp_res ~1e-15 -- both fits unambiguously oscillate.
    re_fit = float(after['parts']['re_lambda'])
    fp_res = float(after['parts']['fp_res'])
    fp_ok = fp_res < 1e-6
    collapsed = (not fp_ok) or re_fit <= 0.0 or not (0.0 < after['mu'] < 0.99)
    if collapsed:
        why = []
        if not fp_ok:
            why.append(f'fixed-point solve did not converge (residual {fp_res:.2e}), so the '
                       f'stability test is not a measurement')
        elif re_fit <= 0.0:
            why.append(f'Re(lambda_max) = {re_fit:+.5f} at the fixed point -- STABLE, i.e. '
                       f'on the non-oscillating side of the Hopf bifurcation')
        if not (0.0 < after['mu'] < 0.99):
            why.append(f"Floquet mu = {after['mu']:.4f} (not an attracting cycle)")
        print("\n  DEGENERACY: " + "; ".join(why))
    else:
        print(f"\n  oscillation OK: Re(lambda_max) = {re_fit:+.5f} > 0 at "
              f"the fixed point (base "
              f"{float(before['parts']['re_lambda']):+.5f}), Floquet mu "
              f"{after['mu']:.4f}")

    # A SEPARATE claim from degeneracy: still an oscillator, but no longer a circadian one.
    # Period is pure GAUGE in parameter space -- freely rescalable -- so residual bought by
    # stretching time is not radialization, but it is also not a dead clock.
    off_regime = per_ratio < 0.7 or per_ratio > 1.4
    if off_regime and not collapsed:
        print(f"  OUT OF CIRCADIAN RANGE: period {before['period']:.2f} -> "
              f"{after['period']:.2f} h (x{per_ratio:.3g}). The oscillation is genuine.")
    improved = after['parts']['c_ptc'] < 0.9 * before['parts']['c_ptc']
    less_twist = after['total_twist'] < before['total_twist']
    print(f"\n  residual {before['parts']['c_ptc']:.4f} -> {after['parts']['c_ptc']:.4f}"
          f"   twist {before['total_twist']:.4f} -> {after['total_twist']:.4f}"
          f"   amplitude {before['amp_lc']:.3f} -> {after['amp_lc']:.3f}")
    if after.get('error'):
        print(f"  VERDICT: FIT SAVED, DIAGNOSIS INCOMPLETE -- {after['error']}. The fitted "
              f"parameters and surfaces are on disk; the twist / S_crit / Floquet numbers "
              f"are not available for this run.")
    elif collapsed:
        print("  VERDICT: DEGENERATE -- there is no self-sustained oscillation (see above). "
              "Any twist or residual improvement is an artefact of fitting a "
              "different dynamical object, not radialization.")
    elif not after['quality_pass']:
        print("  VERDICT: UNUSABLE -- the fitted surface fails the PTC quality gate, so its "
              "twist and S_crit are not measurements.")
    elif improved and less_twist:
        print("  VERDICT: RADIALIZED -- residual and twist both fell with the oscillation intact.")
    elif improved:
        print("  VERDICT: PARTIAL -- the residual fell but twist did not, so the fit matched "
              "the target's dose scaling rather than its isochron geometry.")
    else:
        print("  VERDICT: NO PROGRESS -- the optimizer did not materially reduce the residual.")

    blob = dict(model=model_name, target=target, mode=mode, backend=backend,
                optimizer=optimizer, n_phase=n_phase, dt=dt, s_crit=s_crit,
                max_factor=max_factor, doses=doses, old=C['old'],
                names=np.array(C['names']), B=C['B'], z_base=C['z_base'],
                # --- RAW ------------------------------------------------------------- #
                v_fit=best['v'], theta_base=C['theta'](C['v0']), theta_fit=C['theta'](best['v']),
                ptc_base=before['ptc'], ptc_fit=after['ptc'],
                ptc_target_base=before['ptc_target'], ptc_target_fit=after['ptc_target'],
                k_target_base=before['k_target'], k_target_fit=after['k_target'],
                psi_target_base=before['psi_target'], psi_target_fit=after['psi_target'],
                # THE TARGET THE COST ACTUALLY USED, which is not the profiled registration
                # above whenever the target is pinned. Stored separately and explicitly rather
                # than reconstructed later, so a figure can never again caption a profiled
                # surface as "the target".
                # THE OBSERVABLE IS PART OF THE RUN. Without it a re-plot silently adopts
                # whatever the model default happens to be today -- see REPO_MAP hazard 14.
                cfg_json=json.dumps(cfg.to_dict(), sort_keys=True),
                cfg_label=cfg.label(), seed_used=int(seed),
                section=str(model.reference_variable),
                readout=str(getattr(model, 'readout_variable', None)
                            or model.reference_variable),
                amp_lo=(np.nan if cfg.amp_lo is None else float(cfg.amp_lo)),
                amp_hi=float(cfg.amp_hi), row_weight=bool(cfg.row_weight),
                w_brack=float(cfg.w_brack), include_zero=bool(cfg.include_zero),
                lo_factor=float(cfg.lo_factor),
                off_regime=bool(off_regime), re_lambda_fit=float(re_fit),
                target_pinned=bool(tgt.k is not None),
                k_used=float(tgt.k) if tgt.k is not None else np.nan,
                psi_used=float(tgt.psi) if tgt.psi is not None else np.nan,
                ptc_target_used=_ptc_used,
                cyc_base=before['cyc'], cyc_fit=after['cyc'],
                obs=np.array(before['obs']),
                alive_base=before['alive'], alive_fit=after['alive'],
                amp_base_grid=before['amp'], amp_fit_grid=after['amp'],
                twist_base=before['twist'], twist_fit=after['twist'],
                trace_f=best.get('trace_f'), trace_v=best.get('trace_v'),
                all_f=np.array([r['f'] for r in runs]),
                all_v=np.array([r['v'] for r in runs]),
                # --- features ---------------------------------------------------------- #
                collapsed=bool(collapsed), improved=bool(improved),
                less_twist=bool(less_twist), quality_fit=bool(after['quality_pass']),
                seconds=secs, n_starts=n_starts)
    for tagname, d in (('base', before), ('fit', after)):
        for k in ('total_twist', 'S_crit', 'phi_sing', 'n_sing', 'amp_lc', 'period', 'mu',
                  'scramble'):
            blob[f'{tagname}__{k}'] = np.asarray(d[k])
        for k, v in d['parts'].items():
            blob[f'{tagname}__parts_{k}'] = np.asarray(v)
    out = paths.out_path(model_name, 'fit_radial', f'radial_{target}_{mode}_{optimizer}.npz',
                         tag)
    paths.savez(out, **blob)
    print(f"\n[radial] -> {out}")
    return blob


def run_seeds(cfg):
    """Every seed in `cfg.seeds`, then a comparison across them.

    SEEDS ARE THE UNIT OF THE MULTIMODALITY QUESTION. One seed says how good an optimum the
    search found; several independent seeds say whether the landscape has ONE good optimum or
    many at comparable cost -- which is the global-identifiability question this project is
    aimed at and which no single run can answer. So they are a first-class part of a run
    rather than something to be assembled by hand from separate jobs afterwards.
    """
    cfg.validate()
    seeds = cfg.seed_list
    base_tag = paths.run_tag(cfg.tag)
    # n_free is a property of the model's gauge quotient, so it can be had without building a
    # cost; the starts are then a joint design over the whole seed list (see start_points).
    from fit.cost import quotient_basis
    from models import get_model
    _m = get_model(cfg.model)
    # unpack by NAME: quotient_basis returns (names, z_base, B, gauge) and indexing it
    # positionally is how this became `list.shape` -- twice, in two different files
    _nm, _zb, _B, _gg = quotient_basis(_m)
    n_free = _B.shape[1]
    if cfg.start == 'viable':
        from fit.viability import find_starts
        starts, via_report = find_starts(
            cfg.model, seeds, workers=cfg.workers, section=cfg.section,
            readout=cfg.readout, bound=cfg.bound, period_lo=cfg.viable_period_lo,
            period_hi=cfg.viable_period_hi, min_ratio=cfg.viable_min_ratio,
            max_draws=cfg.viable_max_draws, verbose=cfg.verbose)
        missing = [sd for sd, v in starts.items() if v is None]
        if missing:
            # LOUD, not silent. A quiet fallback to base would turn a dispersed campaign into
            # a start='base' one while still calling itself 'viable' in the output.
            print(f"[viability] WARNING: seeds {missing} found no healthy circadian clock "
                  f"within {cfg.viable_max_draws} draws and will start from BASE instead. "
                  f"Their results are NOT independent starts.", flush=True)
            for sd in missing:
                starts[sd] = np.zeros(n_free)
    else:
        starts, via_report = start_points(cfg, n_free), None
    if cfg.start != 'base':
        D = np.array([[np.linalg.norm(starts[a] - starts[b]) for b in seeds] for a in seeds])
        off = D[np.triu_indices(len(seeds), 1)] if len(seeds) > 1 else np.array([0.0])
        print(f"[radial] start='{cfg.start}' radius {cfg.start_radius}: seed starts are "
              f"{off.min():.2f}-{off.max():.2f} apart (mean {off.mean():.2f})", flush=True)
    out = []
    for i, sd in enumerate(seeds):
        if len(seeds) > 1:
            print(f"{chr(10)}{'=' * 78}{chr(10)}SEED {sd}  ({i + 1} of {len(seeds)})"
                  f"{chr(10)}{'=' * 78}", flush=True)
        # `__`, not `/`: paths.py requires a tag to be a PLAIN NAME, so the run tag
        # stays one directory level and carries its structure in the name instead.
        tag = base_tag if len(seeds) == 1 else f"{base_tag}__seed{sd}"
        out.append(run(cfg, seed=sd, tag=tag, v_start=starts[sd]))
    if via_report is not None:
        _save_viability(cfg, via_report, base_tag)
    if len(seeds) > 1:
        _compare_seeds(cfg, seeds, out, base_tag)
    return out


def _save_viability(cfg, report, tag):
    """Keep the rejection-sampling record. The REJECTED draws are the point.

    A seed needs on the order of a thousand draws to find a healthy circadian clock, and every
    one of those draws is a viability measurement of a random parameter set. Across a campaign
    that is tens of thousands of samples of where a clock can exist -- accumulated for free, as
    a byproduct of seeding, rather than paid for as a separate survey.
    """
    found = np.array([bool(r['found']) for r in report])
    draws = np.array([int(r['draws']) for r in report])
    rate = float(found.sum()) / max(int(draws.sum()), 1)
    print(f"[viability] recorded {int(draws.sum())} draws, {int(found.sum())} accepted "
          f"(rate {rate:.3%}) -> the campaign accumulates this map for free", flush=True)
    blob = dict(seeds=np.array([r['seed'] for r in report]), found=found, draws=draws,
                hit_rate=rate,
                period=np.array([np.nan if r['period'] is None else r['period']
                                 for r in report]),
                min_ratio=np.array([np.nan if r['min_ratio'] is None else r['min_ratio']
                                    for r in report]),
                norm=np.array([np.nan if r['norm'] is None else r['norm'] for r in report]),
                cfg_json=json.dumps(cfg.to_dict(), sort_keys=True))
    # the sampled |v| and their verdicts, so a viability map can be built from campaign output
    pr = [(r['seed'], nv, ok) for r in report for nv, ok in r['probes']]
    if pr:
        blob['probe_seed'] = np.array([p[0] for p in pr])
        blob['probe_norm'] = np.array([p[1] for p in pr])
        blob['probe_ok'] = np.array([p[2] for p in pr])
    paths.savez(paths.out_path(cfg.model, 'fit_radial',
                               f'viability_{cfg.target}_{cfg.mode}.npz', tag), **blob)


def _compare_seeds(cfg, seeds, results, tag):
    """Do independent seeds find the SAME optimum or different ones?

    Compared in the gauge quotient, because two parameter sets differing only by a gauge motion
    are the same model and would otherwise read as distinct basins (REPO_MAP hazard 6).
    """
    v = np.array([np.asarray(r['v_fit']) for r in results])
    f = np.array([float(r['fit__parts_total']) for r in results])
    D = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
    print(f"{chr(10)}{'=' * 78}{chr(10)}SEEDS -- one optimum or several?{chr(10)}{'=' * 78}")
    print(f"  {'seed':>6s} {'cost':>10s}   distance to the best solution")
    b = int(np.argmin(f))
    for i, sd in enumerate(seeds):
        print(f"  {sd:6d} {f[i]:10.6f}   {D[i, b]:8.3f}"
              + ("   <- best" if i == b else ""))
    off = D[np.triu_indices(len(seeds), 1)]
    print(f"{chr(10)}  cost spread {f.max() - f.min():.6f}"
          f"   pairwise distance: min {off.min():.3f} median "
          f"{np.median(off):.3f} max {off.max():.3f}")
    print("  Distinct solutions at comparable cost = MULTIMODAL; one cluster = the search is "
          "finding a single optimum.")
    paths.savez(paths.out_path(cfg.model, 'fit_radial',
                               f'seeds_{cfg.target}_{cfg.mode}.npz', tag),
                seeds=np.array(seeds), v=v, cost=f, dist=D,
                cfg_json=json.dumps(cfg.to_dict(), sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Radial-isochron fit. Every setting is a flag or a --config key; '
                    'see fit/config.py for the full list and its defaults.')
    RunConfig.add_arguments(ap)
    a = ap.parse_args(argv)
    cfg = RunConfig.from_args(a).validate()
    if getattr(a, 'print_config', False):
        print(describe(cfg))
        return 0
    run_seeds(cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
