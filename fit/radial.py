"""
fit/radial.py
=============
T3: fit a clock model so its single-gene PTC matches a RADIAL-ISOCHRON (Poincare) target.

    python -m fit.radial --model almeida --target BMAL1 [--starts 8]

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
import os
import time

import numpy as np

import paths
from fit.cost import make_cost, RadialTarget
from fit.doses import fit_dose_grid
from fit import search


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


def run(model_name='almeida', target='BMAL1', mode='instant', n_phase=16, n_dose=10,
        max_factor=6.0, backend='diffrax', dt=0.02, seed=0, n_starts=1, maxiter=300,
        bound=3.0, w_osc=0.2, w_amp=1.0, optimizer='lbfgs', tag=None,
        pin_target=True):
    from models import get_model
    model = get_model(model_name)
    doses, s_crit = fit_dose_grid(model_name, target, mode, max_factor, n_dose)
    tag = paths.run_tag(tag)

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
    C_probe = make_cost(model, target, doses, RadialTarget(), n_phase=n_phase, mode=mode,
                        backend=backend, dt=dt, w_osc=0.0, w_amp=0.0)
    _zb, _ab, _amb = C_probe['surface'](C_probe['v0'])
    from analysis import winding as _W
    _ptc_b = np.where(_ab, (np.angle(_zb) / (2 * np.pi)) % 1.0, np.nan)
    S_seed, phi_seed, _ns = _W.detect_grid(np.asarray(C_probe['old']), doses, _ptc_b)
    if pin_target and np.isfinite(S_seed) and np.isfinite(phi_seed):
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
                  backend=backend, dt=dt, w_osc=w_osc, w_amp=w_amp)
    before = _diagnose(model, C, C['v0'], 'base')
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
        runs = [search.cma(C, bound=bound, seed=seed, mode='anneal')]
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
        runs = [search.bobyqa(C, C['v0'], bound=bound, maxfev=maxiter * 10, seek_global=True)]
    elif n_starts > 1:
        runs = search.multistart(C, n_starts=n_starts, bound=bound, maxiter=maxiter, seed=seed)
    else:
        runs = [search.lbfgs(C, C['v0'], bound=bound, maxiter=maxiter, label='nominal')]
    best = runs[0] if optimizer in ('cma', 'bobyqa') else min(runs, key=lambda r: r['f'])
    secs = time.time() - t0
    after = _diagnose(model, C, best['v'], 'fitted')

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
    amp_ratio = after['amp_lc'] / max(before['amp_lc'], 1e-12)
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
    if collapsed:
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
                section=str(model.reference_variable),
                readout=str(getattr(model, 'readout_variable', None)
                            or model.reference_variable),
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


def main(argv=None):
    ap = argparse.ArgumentParser(description='T3 radial-isochron fit')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--target', default=os.environ.get('TARGET', 'BMAL1'))
    ap.add_argument('--mode', default='instant', choices=('pulse', 'instant'))
    ap.add_argument('--n-phase', type=int, default=16)
    ap.add_argument('--n-dose', type=int, default=10)
    ap.add_argument('--max-factor', type=float, default=6.0)
    ap.add_argument('--backend', default='diffrax', choices=('rk4', 'diffrax'))
    ap.add_argument('--dt', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--starts', type=int, default=1)
    ap.add_argument('--maxiter', type=int, default=300)
    ap.add_argument('--optimizer', default='lbfgs',
                    choices=('lbfgs', 'cma', 'bobyqa', 'lm'))
    ap.add_argument('--profile-target', action='store_true',
                    help='re-profile the target (k, psi) at every evaluation instead of pinning them to the seed singularity. The target then moves with the model -- see fit.cost.RadialTarget.')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    run(a.model, a.target, a.mode, a.n_phase, a.n_dose, a.max_factor, a.backend, a.dt,
        a.seed, a.starts, a.maxiter, optimizer=a.optimizer, tag=a.tag,
        pin_target=not a.profile_target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
