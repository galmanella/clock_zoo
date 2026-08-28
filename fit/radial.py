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
    P = model.jax_apply(C['theta'](v), C['names'])
    solver = C['solver']
    y0, T, _r = jax.jit(solver.solve)(P, solver.guess(P))
    mu, _ev = solver.floquet(P, y0, T)

    return dict(label=label, ptc=ptc, alive=alive, amp=amp, twist=tw,
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
        bound=3.0, w_osc=0.2, w_amp=1.0, optimizer='lbfgs', tag=None):
    from models import get_model
    model = get_model(model_name)
    doses, s_crit = fit_dose_grid(model_name, target, mode, max_factor, n_dose)
    tag = paths.run_tag(tag)

    from parallel import announce
    announce(analysis='fit.radial', model=model_name, target=target, mode=mode,
             backend=backend, optimizer=optimizer, n_phase=n_phase, n_dose=len(doses),
             S_crit=f'{s_crit:.3g}', dose_max=f'{doses.max():.3g}', starts=n_starts,
             maxiter=maxiter, tag=tag)

    C = make_cost(model, target, doses, RadialTarget(), n_phase=n_phase, mode=mode,
                  backend=backend, dt=dt, w_osc=w_osc, w_amp=w_amp)
    before = _diagnose(model, C, C['v0'], 'base')

    t0 = time.time()
    if optimizer == 'cma':
        # anneal, not ipop -- see fit/search.cma. The obstacle here is small-scale ruggedness,
        # not distinct basins, so a contracting sigma is what is called for.
        runs = [search.cma(C, bound=bound, seed=seed, mode='anneal')]
    elif optimizer == 'str':
        runs = [search.smoothed_trust_region(C, C['v0'], bound=bound, seed=seed)]
    elif optimizer == 'lm':
        runs = [search.levenberg_marquardt(C, C['v0'], bound=bound, maxiter=maxiter)]
    elif n_starts > 1:
        runs = search.multistart(C, n_starts=n_starts, bound=bound, maxiter=maxiter, seed=seed)
    else:
        runs = [search.lbfgs(C, C['v0'], bound=bound, maxiter=maxiter, label='nominal')]
    best = runs[0] if optimizer == 'cma' else min(runs, key=lambda r: r['f'])
    secs = time.time() - t0
    after = _diagnose(model, C, best['v'], 'fitted')

    print(f"\n{'=' * 78}\nRADIALIZATION -- {model_name}/{target} ({mode})\n{'=' * 78}")
    _report(before, C['amp_base'])
    _report(after, C['amp_base'])

    # the verdict, with the degeneracy check FIRST
    collapsed = after['amp_lc'] < 0.5 * before['amp_lc']
    improved = after['parts']['c_ptc'] < 0.9 * before['parts']['c_ptc']
    less_twist = after['total_twist'] < before['total_twist']
    print(f"\n  residual {before['parts']['c_ptc']:.4f} -> {after['parts']['c_ptc']:.4f}"
          f"   twist {before['total_twist']:.4f} -> {after['total_twist']:.4f}"
          f"   amplitude {before['amp_lc']:.3f} -> {after['amp_lc']:.3f}")
    if collapsed:
        print("  VERDICT: DEGENERATE -- the limit cycle collapsed. Any twist or residual "
              "improvement here is the Mirsky failure repeating, not radialization.")
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
                    choices=('lbfgs', 'cma', 'str', 'lm'))
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    run(a.model, a.target, a.mode, a.n_phase, a.n_dose, a.max_factor, a.backend, a.dt,
        a.seed, a.starts, a.maxiter, optimizer=a.optimizer, tag=a.tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
