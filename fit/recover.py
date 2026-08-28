"""
fit/recover.py
==============
T1, the self-recovery control -- and the go/no-go for the whole radial-fit programme.

    python -m fit.recover --model almeida --target BMAL1 --eps 0.3

WHAT IT ASKS
    Generate a PTC surface from the model itself at a KNOWN displaced parameter set, then try
    to fit it back starting from nominal. The truth is in-class by construction, so the only
    thing being tested is whether the optimizer can find a parameter set it is guaranteed to be
    able to represent.

WHY THIS IS THE DECISIVE TEST
    It is the direct 16-dimensional analogue of the failure that stopped the Mirsky programme.
    There, L-BFGS recovered 132/132 parameters when started AT the truth but stalled around
    26/132 when started away from it -- PROJECT_SUMMARY calls that "from-distance locality" and
    names it the framework's central weakness. Whether it is a property of 132-dimensional
    sloppiness or of the PTC landscape itself is exactly what a 16-dimensional, well-conditioned
    model can answer.

    And the answer is worth having either way. If recovery succeeds, the model switch is
    vindicated and the radial fit (T3) is worth running. If it fails, no out-of-class target
    could possibly be fit, and we learn that in hours rather than weeks.

    Note the asymmetry with the radial target: here the truth is REACHABLE. A failure therefore
    indicts the optimizer or the landscape, never the model class -- which is what makes it a
    clean control.
"""
import argparse
import os
import time

import numpy as np

import paths
from fit.cost import make_cost, FixedTarget, RadialTarget
from fit import search


def displaced_truth(cost, eps, seed=0, direction=None):
    """A physical displacement of size `eps` (RMS in log-parameter units) in the quotient.

    Random by default: a hand-picked direction risks accidentally choosing one the PTC happens
    to see especially well (or especially badly), which would make the control easy or
    impossible for reasons unrelated to the question.
    """
    n = cost['n_free']
    if direction is None:
        rng = np.random.default_rng(seed)
        d = rng.normal(size=n)
    else:
        d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    return eps * np.sqrt(n) * d          # so RMS per free coordinate is eps


def run(model_name='almeida', target='BMAL1', mode='instant', eps=0.3, n_phase=16,
        n_dose=10, max_factor=6.0, backend='diffrax', dt=0.02, seed=0, n_starts=1,
        maxiter=300, bound=3.0, w_osc=0.2, w_amp=1.0, tag=None):
    from models import get_model
    from fit.doses import fit_dose_grid

    model = get_model(model_name)
    doses, s_crit = fit_dose_grid(model_name, target, mode, max_factor, n_dose)
    tag = paths.run_tag(tag)

    from parallel import announce
    announce(analysis='fit.recover', model=model_name, target=target, mode=mode, eps=eps,
             backend=backend, n_phase=n_phase, n_dose=len(doses), S_crit=f'{s_crit:.3g}',
             dose_max=f'{doses.max():.3g}', maxiter=maxiter, starts=n_starts, tag=tag)

    # 1. a throwaway cost only to RENDER the truth surface
    C0 = make_cost(model, target, doses, RadialTarget(), n_phase=n_phase, mode=mode,
                   backend=backend, dt=dt, w_osc=0.0, w_amp=0.0)
    v_true = displaced_truth(C0, eps, seed)
    zt, alive_t, amp_t = C0['surface'](v_true)
    frac = float(np.mean(alive_t))
    print(f"[recover] truth |v|={np.linalg.norm(v_true):.3f} "
          f"(RMS {np.std(v_true):.3f} in log-params), target surface {frac:.1%} alive")
    if frac < 0.99:
        # A target with dead cells is missing data, and this control is supposed to be clean.
        print(f"[recover] WARNING: truth surface is only {frac:.1%} alive -- the control is "
              f"no longer clean. Reduce --eps or --max-factor.")

    # 2. the real cost, fitting that fixed surface
    C = make_cost(model, target, doses, FixedTarget(zt), n_phase=n_phase, mode=mode,
                  backend=backend, dt=dt, w_osc=w_osc, w_amp=w_amp)
    p_true = C['parts'](v_true)
    p_nom = C['parts'](C['v0'])
    for lbl, p in (('TRUTH  ', p_true), ('NOMINAL', p_nom)):
        print(f"[recover] cost at {lbl} = {p['total']:.6f}  "
              f"(ptc {p['c_ptc']:.3e} + osc {p['osc']:.4f} + amp {p['amp_pen']:.4f})  "
              f"amp_lc={p['amp_lc']:.3f} T={p['period']:.2f}")
    # The truth's PTC residual is ~0 by construction, so anything left in its TOTAL is a
    # penalty term binding on a healthy parameter set -- which would move the global minimum
    # off the truth and make this control measure the penalty instead of the landscape.
    if p_true['total'] > 0.02:
        print(f"[recover] WARNING: the truth does not sit at ~0 -- a penalty is binding there "
              f"(osc {p_true['osc']:.4f}, amp {p_true['amp_pen']:.4f}). Loosen it "
              f"(amp_frac / w_osc) before believing this control.")

    t0 = time.time()
    if n_starts > 1:
        runs = search.multistart(C, n_starts=n_starts, bound=bound, maxiter=maxiter, seed=seed)
    else:
        runs = [search.lbfgs(C, C['v0'], bound=bound, maxiter=maxiter, label='nominal')]
    best = runs[0]
    dt_all = time.time() - t0

    # 3. did it recover the PARAMETERS, not just the cost?
    dv = best['v'] - v_true
    u_true, u_fit = C['B'] @ v_true, C['B'] @ best['v']
    th_true, th_fit = C['theta'](v_true), C['theta'](best['v'])
    rms_log = float(np.sqrt(np.mean((u_fit - u_true) ** 2)))
    rms_log0 = float(np.sqrt(np.mean(u_true ** 2)))
    within = int(np.sum(np.abs(np.log(th_fit / th_true)) < 0.1))

    print(f"\n{'=' * 78}\nRECOVERY -- {model_name}/{target} ({mode}), eps={eps}\n{'=' * 78}")
    print(f"  cost      nominal {p_nom['total']:.6f} -> fitted {best['f']:.6f} "
          f"(truth {p_true['total']:.6f})")
    print(f"  parameter RMS log-distance to truth: {rms_log:.4f} "
          f"(started at {rms_log0:.4f}, so {1 - rms_log / max(rms_log0, 1e-12):+.1%} closer)")
    print(f"  {within}/{len(C['names'])} parameters recovered to within 10%")
    print(f"  {best['nit']} iterations, {best['nev']} evaluations, {dt_all:.0f}s total")
    print(f"\n  {'param':12s} {'true':>12s} {'fitted':>12s} {'log err':>9s}")
    for nm, a, b in zip(C['names'], th_true, th_fit):
        print(f"  {nm:12s} {a:12.5g} {b:12.5g} {np.log(b / a):+9.3f}")

    verdict = (rms_log < 0.25 * rms_log0) and (best['f'] < 0.25 * p_nom['total'])
    print(f"\n  VERDICT: {'RECOVERED' if verdict else 'NOT RECOVERED'} "
          f"-- {'T3 (radial fit) is worth running' if verdict else 'the landscape is the problem, not the model class'}")

    blob = dict(model=model_name, target=target, mode=mode, backend=backend, eps=eps,
                seed=seed, n_phase=n_phase, dt=dt, s_crit=s_crit, max_factor=max_factor,
                doses=doses, old=C['old'], names=np.array(C['names']),
                # --- RAW ------------------------------------------------------------- #
                v_true=v_true, v_fit=best['v'], v_nominal=C['v0'], B=C['B'],
                z_base=C['z_base'], theta_true=th_true, theta_fit=th_fit,
                target_surface_re=np.real(zt), target_surface_im=np.imag(zt),
                target_alive=alive_t, target_amp=amp_t,
                trace_f=best['trace_f'], trace_v=best['trace_v'],
                all_f=np.array([r['f'] for r in runs]),
                all_v=np.array([r['v'] for r in runs]),
                all_v0=np.array([r.get('v0', C['v0']) for r in runs]),
                # --- features ---------------------------------------------------------- #
                cost_truth=p_true['total'], cost_nominal=p_nom['total'], cost_fit=best['f'],
                rms_log=rms_log, rms_log0=rms_log0, n_within_10pct=within,
                recovered=bool(verdict), seconds=dt_all, n_starts=n_starts)
    for k, v in p_true.items():
        blob[f'parts_truth__{k}'] = np.asarray(v)
    for k, v in best['parts'].items():
        blob[f'parts_fit__{k}'] = np.asarray(v)
    out = paths.out_path(model_name, 'fit_recover',
                         f'recover_{target}_{mode}_eps{eps:g}_s{seed}.npz', tag)
    paths.savez(out, **blob)
    print(f"\n[recover] -> {out}")
    return blob


def main(argv=None):
    ap = argparse.ArgumentParser(description='T1 self-recovery control')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--target', default=os.environ.get('TARGET', 'BMAL1'))
    ap.add_argument('--mode', default='instant', choices=('pulse', 'instant'))
    ap.add_argument('--eps', type=float, default=0.3)
    ap.add_argument('--n-phase', type=int, default=16)
    ap.add_argument('--n-dose', type=int, default=10)
    ap.add_argument('--max-factor', type=float, default=6.0)
    ap.add_argument('--backend', default='diffrax', choices=('rk4', 'diffrax'))
    ap.add_argument('--dt', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--starts', type=int, default=1)
    ap.add_argument('--maxiter', type=int, default=300)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    run(a.model, a.target, a.mode, a.eps, a.n_phase, a.n_dose, a.max_factor, a.backend,
        a.dt, a.seed, a.starts, a.maxiter, tag=a.tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
