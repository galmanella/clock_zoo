"""
analysis/levelset.py
====================
GLOBAL identifiability by construction: walk the manifold of parameter sets that leave one
observable unchanged, and watch what happens to the other.

    python -m analysis.levelset --model almeida --target BMAL1 --hold lc    # hold LC, watch PTC
    python -m analysis.levelset --model almeida --target BMAL1 --hold ptc   # hold PTC, watch LC

WHY THIS AND NOT ANOTHER RANK NUMBER
    Every Jacobian-based statement in this project -- rank, condition number, principal angles,
    "PTC adds 5 directions beyond LC" (PROJECT_SUMMARY 5.4b) -- is a TANGENT-SPACE statement at
    one point. It describes the flat directions through nominal and nothing else. It cannot tell
    a single connected valley from several disconnected ones, and it cannot rule out a parameter
    set far away with an identical LC and a completely different PTC.

    That second possibility is the interesting one. It is exactly where perturbation data would
    earn its keep: not by adding local directions, but by SEPARATING globally distinct models
    that trajectory data cannot distinguish. A local rank comparison is structurally incapable of
    detecting it, so no amount of refining those numbers will answer it.

THE CONSTRUCTION
    Predictor-corrector continuation on a level set:

        1. J = d(held observable)/dv at the current point
        2. N = right-singular directions with sigma/sigma_0 below `tol` -- the local null space,
           i.e. the directions along which the held observable does not change
        3. predictor: step h along the null direction closest to the previous one (continuation,
           so the walk does not double back or jump between branches)
        4. corrector: v <- v - J^+ (A(v) - A(v0)), one Gauss-Newton step back onto the level set
        5. record |A - A0| (must stay ~0 for the walk to mean anything) and |B - B0| (the signal)

    Recomputing N every step is the point: the manifold CURVES, so a straight line along the
    nominal null space leaves the level set almost immediately. That curvature is also why a
    local rank argument cannot be extrapolated by hand.

WHAT A RESULT MEANS
    If |A - A0| stays at solver noise while |B - B0| grows large, the walk has EXHIBITED two
    far-apart parameter sets that agree on A and disagree on B. That is a constructive proof, not
    an inference from a spectrum.

    LIMITATION, stated plainly: this follows one path along a connected manifold. It gives a
    LOWER bound on how far the ambiguity extends and can never certify uniqueness, nor find
    disconnected components -- for those, a global search (multistart ensembles, MCMC or nested
    sampling over the level set) is needed. This is a proof of existence, not a census.
"""
import argparse
import os
import time

import numpy as np

import analysis  # noqa: F401
import paths


def _circ_rms(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 1.0
    return float(np.sqrt(np.mean(np.minimum(d, 1.0 - d) ** 2)))


def build(model_name='almeida', target='BMAL1', mode='instant', n_phase=12, n_dose=8,
          max_factor=6.0, backend='diffrax', dt=0.02, lc_species=None, m_cycle=64):
    """(ctx) with the two observable maps on a common parameter coordinate `v`."""
    import jax.numpy as jnp
    from models import get_model
    from engine.orbit import OrbitSolver
    from fit.cost import make_cost, RadialTarget, quotient_basis
    from fit.doses import fit_dose_grid

    model = get_model(model_name)
    doses, s_crit = fit_dose_grid(model_name, target, mode, max_factor, n_dose)
    C = make_cost(model, target, doses, RadialTarget(), n_phase=n_phase, mode=mode,
                  backend=backend, dt=dt, w_osc=0.0, w_amp=0.0)
    names, z_base, B, _g = quotient_basis(model)
    n = B.shape[1]
    solver = OrbitSolver(model)
    species = list(lc_species) if lc_species else [target]
    idx = jnp.asarray([int(model.var_index(t)) for t in species])
    zb, Bj = jnp.asarray(z_base), jnp.asarray(B)

    def lc(v):
        P = model.jax_apply(jnp.exp(zb + Bj @ jnp.asarray(v)), names)
        y0, T, _r = solver.solve(P, solver.guess(P))
        return np.asarray(solver.cycle(P, y0, T, m_cycle)[:, idx]).ravel()

    def ptc(v):
        zu, alive, amp = C['surface'](v)
        return np.concatenate([np.real(zu).ravel(), np.imag(zu).ravel()])

    def ptc_phase(v):
        zu, alive, amp = C['surface'](v)
        return (np.angle(zu) / (2 * np.pi)) % 1.0

    return dict(model=model, cost=C, n=n, names=names, z_base=z_base, B=B, doses=doses,
                s_crit=s_crit, species=species, lc=lc, ptc=ptc, ptc_phase=ptc_phase,
                theta=C['theta'])


def jacobian(fn, v, n, h=1e-3):
    """Central-difference Jacobian, SANITIZED.

    The walk steps into parameter regions where an orbit solve can fail, and a single NaN column
    makes `np.linalg.svd` raise "SVD did not converge" and kills the whole run. A failed column
    means "this direction could not be measured here", which is honestly represented by zero (it
    lowers the rank and so gets treated as a null direction) rather than by a crash. The count is
    returned so a walk that is mostly guesswork can be recognized as such."""
    f0 = np.asarray(fn(v), float)
    J = np.zeros((f0.size, n))
    n_bad = 0
    for i in range(n):
        vp = v.copy(); vp[i] += h
        vm = v.copy(); vm[i] -= h
        try:
            col = (np.asarray(fn(vp), float) - np.asarray(fn(vm), float)) / (2 * h)
        except Exception:
            col = None
        if col is None or not np.all(np.isfinite(col)):
            n_bad += 1
            col = np.zeros(f0.size)
        J[:, i] = col
    if not np.all(np.isfinite(f0)):
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    return J, f0, n_bad


def walk(ctx, hold='lc', n_steps=24, h_step=0.15, tol=1e-2, h_fd=1e-3, bound=3.0,
         n_corr=2, verbose=True):
    """Traverse the level set of the HELD observable; report the other one along the way."""
    n = ctx['n']
    A = ctx['lc'] if hold == 'lc' else ctx['ptc']
    B = ctx['ptc'] if hold == 'lc' else ctx['lc']
    A_name, B_name = ('LC', 'PTC') if hold == 'lc' else ('PTC', 'LC')

    v = np.zeros(n)
    A0, B0 = A(v), B(v)
    nA0 = np.linalg.norm(A0) or 1.0
    nB0 = np.linalg.norm(B0) or 1.0
    ptc0 = ctx['ptc_phase'](v)

    rows = []
    prev_dir = None
    t0 = time.time()
    if verbose:
        print(f"  {'step':>4s} {'|dv|':>7s} {'held ' + A_name:>12s} {'watched ' + B_name:>14s} "
              f"{'dPTC(cyc)':>10s} {'nullity':>8s}", flush=True)

    for k in range(n_steps + 1):
        J, Av, n_bad = jacobian(A, v, n, h_fd)
        try:
            U, sv, Vt = np.linalg.svd(J, full_matrices=True)
        except np.linalg.LinAlgError:
            print(f"  step {k}: SVD failed even after sanitizing -- stopping the walk here",
                  flush=True)
            break
        svn = sv / sv[0]
        rank = int(np.sum(svn > tol))
        N = Vt[rank:]                                   # null directions (rows)

        dA = float(np.linalg.norm(Av - A0) / nA0)
        Bv = B(v)
        dB = float(np.linalg.norm(Bv - B0) / nB0)
        dptc = _circ_rms(ctx['ptc_phase'](v), ptc0)
        rows.append(dict(step=k, v=v.copy(), dv=float(np.linalg.norm(v)), dA=dA, dB=dB,
                         dptc=dptc, nullity=n - rank, n_bad=n_bad,
                         theta=ctx['theta'](v)))
        if verbose:
            note = f"  ({n_bad} unmeasurable cols)" if n_bad else ""
            print(f"  {k:4d} {np.linalg.norm(v):7.3f} {dA:12.3e} {dB:14.3e} {dptc:10.4f} "
                  f"{n - rank:5d}/{n}{note}", flush=True)
        if k == n_steps or N.shape[0] == 0:
            break

        # predictor: continue in the null direction closest to the previous heading
        if prev_dir is None:
            d = N[0]
        else:
            proj = N @ prev_dir
            d = N[int(np.argmax(np.abs(proj)))] * np.sign(proj[int(np.argmax(np.abs(proj)))])
        d = d / np.linalg.norm(d)
        prev_dir = d
        v = np.clip(v + h_step * d, -bound, bound)

        # Corrector: Gauss-Newton back onto the level set of A, through a TRUNCATED pseudo-
        # inverse.
        #
        # `lstsq` with the default rcond inverts singular values down to machine precision. On
        # this Jacobian -- rank 3 of 16 for a single species, with a tail near 1e-6 -- that
        # divides the residual by ~1e-6 and produces a correction of order 1e6. MEASURED before
        # this fix: a 0.15 predictor step became |dv| = 7.7 and the held observable moved by
        # 1.7e6, i.e. the walk left the level set immediately and the run was meaningless.
        #
        # Only the directions the observable actually resolves may be corrected along; the rest
        # ARE the level set and must be left alone. The step is also capped, since a corrector
        # allowed to move further than the predictor is not a corrector.
        for _ in range(n_corr):
            Jc, Ac, _nb = jacobian(A, v, n, h_fd)
            r = Ac - A0
            try:
                Uc, sc, Vtc = np.linalg.svd(Jc, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            keep = (sc / max(sc[0], 1e-300)) > tol
            if not keep.any():
                break
            step = Vtc[keep].T @ ((Uc[:, keep].T @ r) / sc[keep])
            ns = np.linalg.norm(step)
            if ns > 2.0 * h_step:                      # never outrun the predictor
                step = step * (2.0 * h_step / ns)
            v = np.clip(v - step, -bound, bound)

    if verbose:
        print(f"  walked {rows[-1]['dv']:.3f} in {time.time() - t0:.0f}s", flush=True)
    return rows, A_name, B_name


def run(model_name='almeida', target='BMAL1', mode='instant', hold='lc', n_steps=24,
        h_step=0.15, tol=1e-2, n_phase=12, n_dose=8, lc_species=None, tag=None):
    from parallel import announce
    ctx = build(model_name, target, mode, n_phase=n_phase, n_dose=n_dose,
                lc_species=lc_species)
    tag = paths.run_tag(tag)
    announce(analysis='levelset', model=model_name, target=target, mode=mode, hold=hold,
             n_steps=n_steps, h_step=h_step, tol=tol, n_free=ctx['n'],
             lc_species=','.join(ctx['species']), tag=tag)

    rows, A_name, B_name = walk(ctx, hold=hold, n_steps=n_steps, h_step=h_step, tol=tol)
    last = rows[-1]

    print(f"\n{'=' * 78}\nLEVEL-SET WALK -- hold {A_name}, watch {B_name}\n{'=' * 78}")
    print(f"  travelled {last['dv']:.3f} in log-parameter units "
          f"({np.exp(last['dv']):.1f}x overall)")
    print(f"  held      {A_name:4s} deviation {last['dA']:.3e}  (must stay near solver noise)")
    print(f"  watched   {B_name:4s} deviation {last['dB']:.3e}")
    print(f"  PTC phase RMS change {last['dptc']:.4f} cyc")
    ok = last['dA'] < 1e-2
    big = last['dB'] > 20 * max(last['dA'], 1e-12)
    if not ok:
        print(f"  INCONCLUSIVE: the walk drifted off the {A_name} level set, so the two end "
              f"points do not actually agree on {A_name}. Reduce --h-step or add correctors.")
    elif big:
        print(f"  CONSTRUCTED: two parameter sets {last['dv']:.2f} apart that agree on "
              f"{A_name} to {last['dA']:.1e} and differ on {B_name} by {last['dB']:.1e}.")
        print(f"  -> {B_name} distinguishes models {A_name} cannot, GLOBALLY and not merely in "
              f"the tangent space.")
    else:
        print(f"  NO SEPARATION along this path: {B_name} stayed within {last['dB']:.1e} while "
              f"{A_name} was held. Along this branch the two observables are redundant.")
    print(f"  (One path on a connected manifold: a lower bound on the ambiguity, never a "
          f"uniqueness certificate, and blind to disconnected components.)")

    blob = dict(model=model_name, target=target, mode=mode, hold=hold, tol=tol,
                h_step=h_step, n_steps=n_steps, doses=ctx['doses'], s_crit=ctx['s_crit'],
                names=np.array(ctx['names']), B=ctx['B'], z_base=ctx['z_base'],
                lc_species=np.array(ctx['species']),
                step=np.array([r['step'] for r in rows]),
                v=np.array([r['v'] for r in rows]),
                theta=np.array([r['theta'] for r in rows]),
                dv=np.array([r['dv'] for r in rows]),
                d_held=np.array([r['dA'] for r in rows]),
                d_watched=np.array([r['dB'] for r in rows]),
                d_ptc_cyc=np.array([r['dptc'] for r in rows]),
                nullity=np.array([r['nullity'] for r in rows]),
                n_bad=np.array([r['n_bad'] for r in rows]))
    out = paths.out_path(model_name, 'levelset', f'walk_{target}_{mode}_hold{hold}.npz', tag)
    paths.savez(out, **blob)
    print(f"\n[levelset] -> {out}")
    return blob


def main(argv=None):
    ap = argparse.ArgumentParser(description='global identifiability by level-set traversal')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--target', default=os.environ.get('TARGET', 'BMAL1'))
    ap.add_argument('--mode', default='instant', choices=('pulse', 'instant'))
    ap.add_argument('--hold', default='lc', choices=('lc', 'ptc'))
    ap.add_argument('--n-steps', type=int, default=24)
    ap.add_argument('--h-step', type=float, default=0.15)
    ap.add_argument('--tol', type=float, default=1e-2)
    ap.add_argument('--n-phase', type=int, default=12)
    ap.add_argument('--n-dose', type=int, default=8)
    ap.add_argument('--lc-species', default=None,
                    help='comma-separated; default is the perturbation target alone, which is '
                         'the MATCHED comparison')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    run(a.model, a.target, a.mode, a.hold, a.n_steps, a.h_step, a.tol, a.n_phase, a.n_dose,
        a.lc_species.split(',') if a.lc_species else None, a.tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
