"""
analysis/sweep.py
=================
Walk a RANGE of displacements along one parameter direction and watch what moves.

    $PY -m analysis.sweep --model almeida --target BMAL1 [--direction decoupled]
                             [--eps-max 0.6 --n-eps 9]

`analysis/confirm.py` answers "does this direction do what the jacobian claims" at a single
step size, with controls. This answers the different question: **how much does it move, and
over what range does that stay true.** A single eps cannot distinguish a genuinely flat
direction from one that is merely flat at the step you happened to pick, and it cannot show
where the linear regime ends -- which matters here, since the jacobian overstated the
decoupling ~11x at eps = 0.15.

At each eps (both signs, through zero) it records:

    the limit cycle          full phase-aligned profiles, and dLC vs base
    the PTC surface          full grid on the target's adaptive dose grid
    twist                    the fixed-point-vs-dose curve, and its total span
    S_crit, phi_sing         the singularity, dipole-filtered

so the features can be plotted AGAINST eps and the trend read directly. Everything raw is
saved; `analysis/figures.py --which sweep` is a pure read of it.

ENGINE
    The JAX engine, which agrees with the adaptive one to 1.6e-05 on Almeida
    (engine/validate.py). A sweep is many surfaces and the adaptive engine is minutes each;
    the single-point verification in `confirm.py` is where the independent solver earns its
    keep, and this inherits that. `--adaptive` forces the slow path if you want it.
"""
import argparse
import os
import sys
import time

import numpy as np

import analysis  # noqa: F401
import paths


DIRECTIONS = ('decoupled', 'coupled', 'stiffest')


def get_direction(cp, which='decoupled'):
    """(vector, label, rho) from a saved coupling run.

    `which` is one of the named directions, or an INTEGER index into the rho-sorted
    generalized eigenvectors (0 = most PTC-favouring, -1 = most LC-favouring). The integers
    matter because the interesting question is not just "the two extremes" -- the middle of the
    spectrum is where a direction can move the PTC in a qualitatively different WAY rather than
    just by a different amount.
    """
    V, rho = np.asarray(cp['V']), np.asarray(cp['rho'])
    if which == 'decoupled':
        return V[:, 0], 'decoupled (max rho)', float(rho[0])
    if which == 'coupled':
        return V[:, -1], 'coupled (min rho)', float(rho[-1])
    if which == 'stiffest':
        _u, _s, vt = np.linalg.svd(np.nan_to_num(np.asarray(cp['J_LC'])), full_matrices=False)
        return vt[0], 'stiffest LC direction', float('nan')
    try:
        i = int(which)
    except (TypeError, ValueError):
        raise SystemExit(f"--direction must be an integer index or one of {DIRECTIONS}")
    if not (-V.shape[1] <= i < V.shape[1]):
        raise SystemExit(f"direction index {i} out of range for {V.shape[1]} directions")
    return V[:, i], f'eigendirection {i % V.shape[1]}', float(rho[i])


def run(model_name, target, mode='pulse', which='decoupled', eps_max=0.6, n_eps=9,
        n_phase=32, feature='twist', tag=None):
    import jax
    from models import get_model
    from engine.orbit import make_orbit_finder
    from engine.ptc import make_ptc, grid_points, phase_or_nan, valid_mask, recommended_skip
    from analysis.ptc_sens import load_grid
    from analysis.confirm import _full_params
    from analysis import winding as W

    model = get_model(model_name)
    fp = paths.out_path(model_name, 'coupling', f'coupling_{target}_{mode}_{feature}.npz')
    if not os.path.exists(fp):
        raise SystemExit(f"run `python -m analysis.coupling --model {model_name} "
                         f"--target {target}` first")
    cp = dict(np.load(fp, allow_pickle=True))
    names = [str(p) for p in cp['params']]
    v, label, rho = get_direction(cp, which)
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)

    doses, dt = load_grid(model_name, target, mode)
    if doses is None:
        raise SystemExit('no dose grid; run analysis.scrit first')
    dt = dt or 0.02
    skip_p, mu, _r = recommended_skip(model, tol=1e-2, verbose=False)
    # symmetric, through zero, so the base sits exactly in the middle of the sweep
    epss = np.linspace(-eps_max, eps_max, n_eps)
    tag = paths.run_tag(tag)

    from parallel import announce
    announce(analysis='sweep', model=model_name, target=target, mode=mode, direction=which,
             rho=f'{rho:.3g}', eps=f'+/-{eps_max}', n_eps=n_eps, n_phase=n_phase,
             n_dose=len(doses), dt=dt, tag=tag)

    f, _sv = make_ptc(model, target, mode=mode, readout='raw', skip_p=skip_p, dt=dt,
                      track_min=True)
    fj = jax.jit(f)
    gph, gdz = grid_points(n_phase, doses)
    find, _s = make_orbit_finder(model)
    old = np.arange(n_phase) / n_phase

    def one(params, y_seed=None):
        x0, T, C, st = find(params, y_seed)
        if x0 is None:
            return None, None, np.nan, st, None
        z, mn = fj(model.jax_params(params), x0, gph, gdz)
        p, a = phase_or_nan(np.asarray(z))
        p = np.where(valid_mask(np.asarray(mn)), p, np.nan)
        return (C, p.reshape(len(doses), n_phase).T, float(T), 'ok', np.asarray(x0)[:-1])

    base_params = model.get_parameters()
    C0, P0, T0, st0, y0 = one(base_params)
    if C0 is None:
        raise SystemExit(f'base not usable ({st0})')
    scale = np.maximum(C0.max(1) - C0.min(1), 1e-9)
    tw0 = W.twist_curve(old, doses, P0)
    S0, phi0, _n = W.detect_grid(old, doses, P0)

    n = len(epss)
    LCP = np.full((n,) + C0.shape, np.nan)
    PTC = np.full((n, n_phase, len(doses)), np.nan, np.float32)
    TWC = np.full((n, len(doses)), np.nan)
    dLC = np.full(n, np.nan); dPT = np.full(n, np.nan); dTW = np.full(n, np.nan)
    TT = np.full(n, np.nan); SC = np.full(n, np.nan); PHI = np.full(n, np.nan)
    PER = np.full(n, np.nan); NSING = np.full(n, np.nan)
    STAT = np.array(['?'] * n, dtype=object)
    PVALS = np.full((n, len(names)), np.nan)

    t0 = time.time()
    # walk OUTWARD from eps = 0 in both directions, continuing the orbit from the previous
    # solution -- the same trick lc_sens uses, and it matters more here because the far ends
    # of the sweep are much further from base than a single factor step.
    i0 = int(np.argmin(np.abs(epss)))
    order = [i0] + [i for k in range(1, n) for i in (i0 + k, i0 - k) if 0 <= i < n]
    seed = None
    for i in order:
        pd = _full_params(model, names, np.exp(epss[i] * v) *
                          np.array([base_params[nm] for nm in names]))
        PVALS[i] = [pd[nm] for nm in names]
        try:
            C, P, T, st, ys = one(pd, seed)
        except Exception as e:
            print(f"    eps={epss[i]:+.3f} FAILED {type(e).__name__}: {e}", flush=True)
            STAT[i] = 'error'
            continue
        STAT[i] = st
        if C is None:
            continue
        seed = ys
        LCP[i] = C; PTC[i] = P.astype(np.float32); PER[i] = T
        d = (C - C0) / scale[:, None]
        dLC[i] = float(np.sqrt(np.mean(d ** 2)))
        try:
            tw = W.twist_curve(old, doses, P)
            TWC[i] = tw
            TT[i] = W.total_twist(tw)
            dTW[i] = W.circ_rms(tw, tw0)
            dPT[i] = W.circ_rms(P, P0)
            s, ph, ns = W.detect_grid(old, doses, P)
            SC[i], PHI[i], NSING[i] = s, ph, ns
        except Exception as e:
            print(f"    eps={epss[i]:+.3f} DETECT failed {type(e).__name__}: {e}", flush=True)

    print(f"[sweep] {n} displacements in {time.time() - t0:.0f}s", flush=True)
    print(f"\n  {'eps':>7s} {'dLC':>9s} {'dPTC':>9s} {'dTwist':>9s} {'twist':>8s} "
          f"{'S_crit':>9s} {'phi*':>7s} {'T':>8s} {'n_sing':>6s}")
    for i in range(n):
        print(f"  {epss[i]:+7.3f} {dLC[i]:9.4f} {dPT[i]:9.4f} {dTW[i]:9.4f} {TT[i]:8.4f} "
              f"{SC[i]:9.4g} {PHI[i]:7.3f} {PER[i]:8.3f} {NSING[i]:6.0f}"
              + ('' if STAT[i] == 'ok' else f'   <- {STAT[i]}'))

    blob = dict(model=model_name, target=target, mode=mode, direction=which, label=label,
                rho=rho, eps=epss, vector=v, param_names=np.array(names),
                state_names=np.array(list(model.state_names)),
                observables=np.array(list(model.observable_states())),
                doses=doses, old=old, dt=dt, skip_p=skip_p, mu=mu, n_phase=n_phase,
                # --- RAW ---------------------------------------------------------------- #
                lc_profiles=LCP, ptc_grids=PTC, twist_curves=TWC, param_values=PVALS,
                base_profiles=C0, base_ptc=P0.astype(np.float32), base_twist=tw0,
                base_period=T0, base_S=S0, base_phi=phi0, lc_scale=scale,
                # --- features ------------------------------------------------------------ #
                dLC=dLC, dPTC=dPT, dTwist=dTW, total_twist=TT, S_crit=SC, phi_sing=PHI,
                period=PER, n_sing=NSING, status=STAT.astype(str))
    safe = str(which).replace('-', 'm')
    out = paths.out_path(model_name, 'sweep', f'sweep_{target}_{mode}_{safe}.npz', tag)
    paths.savez(out, **blob)
    print(f"\n[sweep] -> {out}")
    return blob


def main(argv=None):
    ap = argparse.ArgumentParser(description='displacement sweep along one direction')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--target', default=os.environ.get('TARGET', 'BMAL1'))
    ap.add_argument('--mode', default='pulse')
    ap.add_argument('--direction', default='decoupled',
                    help='decoupled|coupled|stiffest, or an integer eigendirection index')
    ap.add_argument('--eps-max', type=float, default=0.6)
    ap.add_argument('--n-eps', type=int, default=9)
    ap.add_argument('--n-phase', type=int, default=32)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    run(a.model, a.target, a.mode, a.direction, a.eps_max, a.n_eps, a.n_phase, tag=a.tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
