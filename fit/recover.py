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
        maxiter=300, bound=3.0, w_osc=0.2, w_amp=1.0, tag=None, optimizer='lbfgs',
        maxfev=3000, start='nominal', ridge=0.0):
    from models import get_model
    from fit.doses import fit_dose_grid

    model = get_model(model_name)
    doses, s_crit = fit_dose_grid(model_name, target, mode, max_factor, n_dose)
    tag = paths.run_tag(tag)

    from parallel import announce
    announce(analysis='fit.recover', model=model_name, target=target, mode=mode, eps=eps,
             backend=backend, optimizer=optimizer, n_phase=n_phase, n_dose=len(doses),
             S_crit=f'{s_crit:.3g}', dose_max=f'{doses.max():.3g}', maxiter=maxiter,
             starts=n_starts, tag=tag)

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
    # LM needs the residual JACOBIAN, and jacfwd cannot differentiate through diffrax's default
    # RecursiveCheckpointAdjoint (a custom_vjp, reverse-mode only). ForwardMode is the right
    # choice here anyway: the residual has ~200 outputs and only 16 inputs, so forward mode costs
    # 16 JVPs against reverse mode's 200 VJPs.
    gmode = 'fwd' if optimizer in ('lm', 'bh') else 'rev'
    C = make_cost(model, target, doses, FixedTarget(zt), n_phase=n_phase, mode=mode,
                  backend=backend, dt=dt, w_osc=w_osc, w_amp=w_amp, ridge=ridge,
                  grad_mode=gmode)
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

    # `start='truth'` is the control Mirsky ran: begin AT the answer and see whether the
    # optimizer stays. If it drifts away and the cost drops, the truth is not a minimum of this
    # cost and the fault is the objective, not the search. If it stays, the truth IS a minimum
    # and the failure from nominal is genuinely about reaching it.
    v_start = v_true.copy() if start == 'truth' else C['v0']
    t0 = time.time()
    if optimizer == 'bh':
        runs = [search.basin_hopping(C, v_start, bound=bound, seed=seed, label=start)]
    elif optimizer == 'lm':
        runs = [search.levenberg_marquardt(C, v_start, bound=bound, maxiter=maxiter,
                                           label=start)]
    elif optimizer == 'cma':
        # Gradient-free. The right tool when the cost VALUES are sound but the derivatives are
        # not -- which is the situation on the wide dose grid, where the informative doses are
        # exactly the ones whose gradient explodes (fit/doses.py, REPO_MAP hazard 11).
        runs = [search.cma(C, v0=v_start, bound=bound, seed=seed, maxfev=maxfev)]
    elif n_starts > 1:
        runs = search.multistart(C, n_starts=n_starts, bound=bound, maxiter=maxiter, seed=seed)
    else:
        runs = [search.lbfgs(C, v_start, bound=bound, maxiter=maxiter, label=start)]
    best = min(runs, key=lambda r: r['f'])
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

    # SCORE IN THE IDENTIFIED SUBSPACE TOO.
    #
    # "N of 18 parameters within 10%" charges the optimizer equally for every direction, but the
    # data does not constrain every direction: the surface Jacobian at the truth has ~5 singular
    # values above 1e-2 and the rest fall to 1e-4. An optimizer cannot recover what the data does
    # not contain, so the full-space metric measures partly the fit and partly the question being
    # ill-posed. Splitting the error in the singular basis separates them.
    #
    # Measured with this on the first two runs: CMA-ES put 92.5% of its squared error in
    # directions beyond the top four, i.e. it largely got the identified part right and wandered
    # in the null space -- which is CORRECT behaviour, not failure. L-BFGS put 41% in the top
    # four, so it missed even the determined directions.
    sub = None
    try:
        Jr = C['jac'](v_true)
        # drop the penalty/ridge rows: only the surface rows carry identifiability information
        Jr = Jr[:C['n_surface_rows']] if 'n_surface_rows' in C else Jr
        _U, sv, Vt = np.linalg.svd(Jr, full_matrices=False)
        svn = sv / sv[0]
        c = Vt @ dv
        tot = float(np.linalg.norm(dv))
        k_det = int(np.sum(svn > 1e-2))
        det = float(np.linalg.norm(c[:k_det])) / max(tot, 1e-30)
        print(f"\n  identifiability-aware score (Jacobian at the truth):")
        print(f"    {k_det} of {len(svn)} directions determined (sigma > 1e-2); "
              f"spectrum {svn[0]:.3f} .. {svn[-1]:.1e}")
        print(f"    error in the DETERMINED subspace: {det:.1%} of |dv|  "
              f"({np.linalg.norm(c[:k_det]):.4f} of {tot:.4f})")
        print(f"    error in the FLAT subspace:       "
              f"{np.linalg.norm(c[k_det:]) / max(tot, 1e-30):.1%} of |dv|")
        print(f"    -> {'the fit missed directions the data DOES constrain' if det > 0.5 else 'the error is mostly where the data is silent, which is expected'}")
        sub = dict(sv=svn, err_modes=c, k_det=k_det, det_frac=det)
    except Exception as e:
        print(f"  [subspace scoring unavailable: {type(e).__name__}: {e}]")
    print(f"  {best.get('nit', -1)} iterations, {best.get('nev', -1)} evaluations, "
          f"{dt_all:.0f}s total ({optimizer})")
    print(f"\n  {'param':12s} {'true':>12s} {'fitted':>12s} {'log err':>9s}")
    for nm, a, b in zip(C['names'], th_true, th_fit):
        print(f"  {nm:12s} {a:12.5g} {b:12.5g} {np.log(b / a):+9.3f}")

    if start == 'truth':
        # Starting AT the answer is a sanity check on the OBJECTIVE, not a recovery test --
        # reporting "RECOVERED" there would be vacuous. What it establishes is whether the truth
        # is a minimum at all, which is the precondition for reading anything into a failure
        # from nominal.
        stayed = rms_log < 0.02 and best['f'] <= p_true['total'] * 1.5
        msg = ("STAYED -- the truth IS a minimum, so the cost is sound and a failure from "
               "nominal is about REACHING it" if stayed else
               "DRIFTED -- the truth is NOT a minimum of this cost; the objective is at "
               "fault, not the search")
        print(f"\n  CONTROL (started at the truth): {msg}")
        verdict = stayed
    else:
        verdict = (rms_log < 0.25 * rms_log0) and (best['f'] < 0.25 * p_nom['total'])
        msg = ("T3 (radial fit) is worth running" if verdict else
               "from-distance locality -- the truth is a minimum (see --start truth) but the "
               "optimizer cannot reach it from here")
        print(f"\n  VERDICT: {'RECOVERED' if verdict else 'NOT RECOVERED'} -- {msg}")

    blob = dict(model=model_name, target=target, mode=mode, backend=backend, eps=eps,
                seed=seed, n_phase=n_phase, dt=dt, s_crit=s_crit, max_factor=max_factor,
                doses=doses, old=C['old'], names=np.array(C['names']),
                # --- RAW ------------------------------------------------------------- #
                v_true=v_true, v_fit=best['v'], v_nominal=C['v0'], B=C['B'],
                z_base=C['z_base'], theta_true=th_true, theta_fit=th_fit,
                target_surface_re=np.real(zt), target_surface_im=np.imag(zt),
                target_alive=alive_t, target_amp=amp_t,
                trace_f=best.get('trace_f'), trace_v=best.get('trace_v'),
                optimizer=optimizer, start=start,
                all_f=np.array([r['f'] for r in runs]),
                all_v=np.array([r['v'] for r in runs]),
                all_v0=np.array([r.get('v0', C['v0']) for r in runs]),
                # --- features ---------------------------------------------------------- #
                cost_truth=p_true['total'], cost_nominal=p_nom['total'], cost_fit=best['f'],
                rms_log=rms_log, rms_log0=rms_log0, n_within_10pct=within,
                nit=best.get('nit', -1), nev=best.get('nev', -1),
                sub_sv=(sub['sv'] if sub else np.array([])),
                sub_err_modes=(sub['err_modes'] if sub else np.array([])),
                sub_k_det=(sub['k_det'] if sub else -1),
                sub_det_frac=(sub['det_frac'] if sub else float('nan')),
                n_grad_clipped=best.get('n_grad_clipped', 0),
                n_grad_dead=best.get('n_grad_dead', 0),
                grad_max=best.get('grad_max', float('nan')),
                recovered=bool(verdict), seconds=dt_all, n_starts=n_starts)
    for k, v in p_true.items():
        blob[f'parts_truth__{k}'] = np.asarray(v)
    for k, v in best['parts'].items():
        blob[f'parts_fit__{k}'] = np.asarray(v)
    out = paths.out_path(model_name, 'fit_recover',
                         f'recover_{target}_{mode}_eps{eps:g}_{optimizer}_{start}_s{seed}.npz', tag)
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
    ap.add_argument('--optimizer', default='lbfgs', choices=('lbfgs', 'cma', 'bobyqa', 'lm', 'bh'))
    ap.add_argument('--maxfev', type=int, default=3000)
    ap.add_argument('--bound', type=float, default=3.0,
                    help='search box, |v| <= bound in log-parameter units. Mirsky et al. report that loose bounds FAILED to find oscillating sets and that they tightened until the search was constrained enough to work; 3.0 is a factor of ~20 each way, which is loose by that standard.')
    ap.add_argument('--ridge', type=float, default=0.0,
                    help='Tikhonov weight along ALL directions; makes the ~8 flat ones well-posed at the price of shrinking them toward nominal')
    ap.add_argument('--start', default='nominal', choices=('nominal', 'truth'))
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    run(a.model, a.target, a.mode, a.eps, a.n_phase, a.n_dose, a.max_factor, a.backend,
        a.dt, a.seed, a.starts, a.maxiter, bound=a.bound, tag=a.tag,
        optimizer=a.optimizer,
        maxfev=a.maxfev, start=a.start, ridge=a.ridge)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
