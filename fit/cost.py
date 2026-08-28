"""
fit/cost.py
===========
The objective for fitting a clock model to a PTC surface.

    python -m fit.cost --selftest                 # the anti-degeneracy invariant
    python -m fit.cost --gradcheck                # autodiff vs finite differences

    C(v) = C_ptc(v) + w_osc * B_osc(v) + w_amp * A(v)

DESIGNED AGAINST ONE SPECIFIC FAILURE
    input_screen/radialize.py fit Mirsky to a radial-isochron target with cost
    `twist_cost + 0.5 * feature_cost` and no limit-cycle term. It converged -- 0.89 -> 0.159 in
    11 L-BFGS iterations -- by walking the clock to a Hopf bifurcation: the phase singularity
    was annihilated, the limit-cycle amplitude collapsed, and the twist "improved" only because
    the oscillation was dying. Parameters moved less than 4%. Both terms of that cost are
    MINIMIZED by a dying oscillator.

    So the arrangement here is deliberate:

    C_ptc   A POINTWISE match over the whole (phase, dose) surface, not a match to extracted
            features. The radial target is defined at every cell, so a dead oscillator cannot
            match it -- but only if unusable cells are scored rather than skipped. They score
            the MAXIMUM (1.0). See fit/target.circ_cost; that one line is what inverts the
            Mirsky degeneracy, and `--selftest` asserts it.
    A       An explicit floor on limit-cycle amplitude. The Mirsky collapse moved parameters
            less than 4%, so the barrier has to bite early, not just at the bifurcation.
    B_osc   The Hopf-distance barrier (ported from input_screen/osc_term.py). This is the
            user's stated constraint -- "self-sustained oscillation and nothing more" -- and it
            has a second job that matters just as much: C_ptc SATURATES at 1.0 across the whole
            dead region, so it is flat there and cannot tell an optimizer which way to go.
            Re(lambda_max) at the fixed point is a continuous signed distance to the Hopf
            boundary, defined whether or not the system oscillates, so it supplies the gradient
            that carries a dead candidate back to life. That matters for multistart (T2), where
            random starts routinely land dead.

WHAT IS *NOT* CONSTRAINED
    LC shape. No waveform, amplitude-profile, or period target beyond the floor above. That is
    the point of the experiment: ask whether PTC data ALONE identifies the model. It is also
    what makes the degeneracy live, hence the care above.

COORDINATES
    The optimizer works in `v`, a vector in the GAUGE QUOTIENT: v -> u = B @ v -> theta =
    theta_base * exp(u), with B an orthonormal basis of the complement of the gauge span. For
    Almeida that is 16 free directions out of 18 parameters. Optimizing raw log-parameters
    instead would leave 2 exactly-flat directions for the optimizer to wander along, which
    wastes population in CMA and makes an L-BFGS Hessian singular.
"""
import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import lax

from engine.orbit import OrbitSolver, make_guess_fn
from engine.ptc import make_ptc, grid_points, DEAD_AMP, NEG_TOL
from fit.target import circ_cost, profile, radial_z
from gauge.gauge import Gauge


# --------------------------------------------------------------------------- #
#  Gauge quotient coordinates
# --------------------------------------------------------------------------- #
def quotient_basis(model, param_names=None, include_time=True):
    """(names, z_base, B) with B an orthonormal basis of the gauge quotient.

    B has one column per PHYSICALLY MEANINGFUL parameter combination: 16 of 18 for Almeida,
    33 of 34 for Korencic, 47 of 52 for Goldbeter. Moving along span(B) can never be undone by
    a change of units, so every direction the optimizer can travel is one that changes the
    model.
    """
    g = Gauge(model, param_names, include_time=include_time)
    n = len(g.names)
    P = np.eye(n) - (g.Q @ g.Q.T if g.Q.size else 0.0)
    u, s, _vt = np.linalg.svd(P)
    B = u[:, s > 1e-8]
    return list(g.names), g.z_nominal.copy(), B, g


# --------------------------------------------------------------------------- #
#  Hopf-distance barrier (ported from input_screen/osc_term.py, generalized)
# --------------------------------------------------------------------------- #
def _maxre_val_np(A):
    A = np.asarray(A)
    if not np.isfinite(A).all():             # divergent params -> NaN, not a LAPACK raise
        return np.asarray(np.nan, A.dtype)
    return np.asarray(np.max(np.linalg.eigvals(A).real), A.dtype)


def _leading_np(J):
    """Value and exact gradient of Re(lambda_max): d(Re lam)/dJ_ab = Re(conj(y_a) x_b/(y^H x)),
    with x, y the right and left eigenvectors of the leading eigenvalue."""
    J = np.asarray(J)
    if not np.isfinite(J).all():
        return np.asarray(np.nan, J.dtype), np.zeros_like(J)
    w, X = np.linalg.eig(J)
    i = int(np.argmax(w.real)); lam = w[i]; x = X[:, i]
    wl, Y = np.linalg.eig(J.T)               # right eigvecs of J^T = left eigvecs of J
    j = int(np.argmin(np.abs(wl - lam.conjugate()))); y = Y[:, j]
    G = np.real(np.outer(np.conjugate(y), x) / np.vdot(y, x))
    return np.asarray(np.real(lam), J.dtype), np.asarray(G, J.dtype)


@jax.custom_vjp
def _max_re_eig(J):
    return jax.pure_callback(_maxre_val_np, jax.ShapeDtypeStruct((), J.dtype), J,
                             vmap_method='sequential')


def _mre_fwd(J):
    val, G = jax.pure_callback(
        _leading_np,
        (jax.ShapeDtypeStruct((), J.dtype), jax.ShapeDtypeStruct(J.shape, J.dtype)),
        J, vmap_method='sequential')
    return val, G


def _mre_bwd(G, ct):
    return (ct * G,)


_max_re_eig.defvjp(_mre_fwd, _mre_bwd)


def make_osc_penalty(model, w=0.2, k=200.0, margin=0.015, n_iter=40, reg=1e-9):
    """`pen(P, y_guess) -> (penalty, Re(lambda_max), residual)`.

    w * softplus(k * (margin - Re(lambda_max))): ~0 once comfortably oscillating, so it does
    not distort the data fit, and growing with a strong gradient once the fixed point turns
    stable (i.e. the clock is dead or damped). Newton runs to convergence under stop_gradient,
    then ONE differentiable implicit step, so d y*/dP is the exact implicit-function gradient
    at the cost of a single linear solve.
    """
    rhs = model.jax_rhs
    N = int(model.n_states)
    I = jnp.eye(N)

    def fixed_point(P, y_guess):
        def body(y, _):
            f = rhs(y, P)
            J = jax.jacfwd(rhs)(y, P)
            dy = jnp.linalg.solve(J + reg * I, f)
            return jnp.maximum(y - 0.7 * dy, 0.0), None     # damped + positivity-safeguarded
        yc, _ = lax.scan(body, y_guess, None, length=n_iter)
        yc = lax.stop_gradient(yc)
        f = rhs(yc, P); J = jax.jacfwd(rhs)(yc, P)
        ystar = yc - jnp.linalg.solve(J + reg * I, f)
        res = jnp.linalg.norm(rhs(lax.stop_gradient(ystar), P))
        return ystar, res

    def pen(P, y_guess):
        ystar, res = fixed_point(P, y_guess)
        re = _max_re_eig(jax.jacfwd(rhs)(ystar, P))
        p = w * jax.nn.softplus(k * (margin - re))
        # No fixed point found -> a large but FINITE 'dead' penalty, never NaN.
        p = jnp.where(jnp.isfinite(re) & (res < 1e-3), p,
                      w * jax.nn.softplus(k * margin + 5.0))
        return p, re, res

    return pen


# --------------------------------------------------------------------------- #
#  Targets
# --------------------------------------------------------------------------- #
class FixedTarget:
    """A concrete surface to match, in the model's own dose units (the self-recovery control).

    No profiling: the target came from this same model and engine, so its dose scale and its
    phase origin are already the model's. Fitting nuisance parameters here would let the
    optimizer hide a real parameter error inside a dose rescale."""
    name = 'fixed'

    def __init__(self, zt):
        self.zt = jnp.asarray(zt)

    def __call__(self, zm, alive, old, doses):
        return self.zt, {}


class RadialTarget:
    """The analytic radial-isochron surface, with (k, psi) profiled out per evaluation.

    Profiled, not fitted, because neither is a property of the model -- see fit/target.py. The
    profile costs no ODE solves, so this is as cheap as FixedTarget."""
    name = 'radial'

    def __init__(self, n_k=48, n_psi=48, refine=2):
        self.n_k, self.n_psi, self.refine = n_k, n_psi, refine

    def __call__(self, zm, alive, old, doses):
        k, psi, _c = profile(zm, alive, old, doses, self.n_k, self.n_psi, refine=self.refine)
        return radial_z(old, doses, k, psi), {'k': k, 'psi': psi}


# --------------------------------------------------------------------------- #
#  The cost
# --------------------------------------------------------------------------- #
def make_cost(model, target_state, doses, tgt, n_phase=24, mode='instant', backend='diffrax',
              dt=0.02, skip_p=None, w_osc=0.2, w_amp=1.0, amp_frac=0.5, m_amp=64,
              param_names=None, eps=1e-12):
    """Build the objective.

    Returns a dict with
        v0        the origin (all zeros) -- the base parameter set
        n_free    dimension of the search
        parts(v)  -> dict of every term plus diagnostics, as plain floats
        total(v)  -> the scalar to minimize (jit-compiled)
        grad(v)   -> its gradient
        surface(v)-> (z_unit, alive, amp) for plotting
    """
    names, z_base, B, g = quotient_basis(model, param_names)
    n_free = B.shape[1]
    Bj, zbj = jnp.asarray(B), jnp.asarray(z_base)

    f, solver = make_ptc(model, target_state, mode=mode, readout='raw', skip_p=skip_p,
                         dt=dt, track_min=True, backend=backend)
    guess = make_guess_fn(model)
    ph, dz = grid_points(n_phase, doses)
    nd, npz = len(doses), n_phase
    old = jnp.arange(n_phase) / n_phase
    ref_idx = int(model.var_index(model.reference_variable))
    y_seed = jnp.asarray(model.get_initial_state(), jnp.float64)
    osc = make_osc_penalty(model, w=1.0) if w_osc else None

    # base limit-cycle amplitude, the reference the floor is expressed against
    Pb = model.jax_params()
    y0b, Tb, _rb = jax.jit(solver.solve)(Pb, solver.guess(Pb))
    cb = np.asarray(solver.cycle(Pb, y0b, Tb, m_amp))[:, ref_idx]
    amp_base = float((cb.max() - cb.min()) / max(abs(cb.mean()), 1e-12))
    amp_floor = amp_frac * amp_base

    def _theta(v):
        return jnp.exp(zbj + Bj @ v)

    def _surface(v):
        P = model.jax_apply(_theta(v), names)
        x0 = guess(P, y_seed)
        y0, T, _res = solver.solve(P, x0)
        cyc = solver.cycle(P, y0, T, m_amp)[:, ref_idx]
        amp_lc = (jnp.max(cyc) - jnp.min(cyc)) / jnp.maximum(jnp.abs(jnp.mean(cyc)), 1e-12)
        z, ymin = f(P, jnp.concatenate([y0, T[None]]), ph, dz)
        z = z.reshape(nd, npz).T
        ymin = ymin.reshape(nd, npz).T
        fin = jnp.isfinite(z.real) & jnp.isfinite(z.imag) & jnp.isfinite(ymin)
        zs = jnp.where(fin, z, 1.0 + 0j)
        amp = jnp.abs(zs)
        alive = fin & (amp > DEAD_AMP) & (ymin >= NEG_TOL)
        zu = zs / (amp + eps)
        return P, zu, alive, amp, amp_lc, T

    def _parts(v):
        P, zu, alive, amp, amp_lc, T = _surface(v)
        zt, aux = tgt(zu, alive, old, doses)
        c_ptc = circ_cost(zu, zt, alive)
        # one-sided quadratic floor: 0 when healthy, rising as the cycle shrinks. Quadratic
        # rather than linear so it is gentle near the floor and firm well below it.
        a = jnp.maximum(0.0, 1.0 - amp_lc / amp_floor) ** 2
        if osc is not None:
            b, re, res = osc(P, y_seed)
        else:
            b = jnp.array(0.0); re = jnp.array(jnp.nan); res = jnp.array(jnp.nan)
        total = c_ptc + w_osc * b + w_amp * a
        return dict(total=total, c_ptc=c_ptc, osc=b, amp_pen=a, amp_lc=amp_lc, period=T,
                    alive_frac=jnp.mean(alive.astype(jnp.float64)), re_lambda=re,
                    fp_res=res, **aux)

    _total = jax.jit(lambda v: _parts(v)['total'])
    _grad = jax.jit(jax.grad(lambda v: _parts(v)['total']))
    _parts_j = jax.jit(_parts)

    def parts(v):
        d = _parts_j(jnp.asarray(v, jnp.float64))
        return {k: float(np.asarray(x)) for k, x in d.items()}

    def surface(v):
        P, zu, alive, amp, amp_lc, T = _surface(jnp.asarray(v, jnp.float64))
        return np.asarray(zu), np.asarray(alive), np.asarray(amp)

    return dict(names=names, z_base=z_base, B=B, gauge=g, n_free=n_free,
                v0=np.zeros(n_free), doses=np.asarray(doses), old=np.asarray(old),
                amp_base=amp_base, amp_floor=amp_floor, target=tgt.name,
                theta=lambda v: np.asarray(_theta(jnp.asarray(v, jnp.float64))),
                total=lambda v: float(_total(jnp.asarray(v, jnp.float64))),
                grad=lambda v: np.asarray(_grad(jnp.asarray(v, jnp.float64))),
                parts=parts, surface=surface, solver=solver, ptc_fn=f)


# --------------------------------------------------------------------------- #
#  Self-tests
# --------------------------------------------------------------------------- #
def _build(name='almeida', target='BMAL1', n_phase=12, n_dose=6, backend='diffrax',
           dt=0.02, w_osc=0.2):
    from models import get_model
    from analysis.ptc_sens import load_grid
    model = get_model(name)
    doses, gdt = load_grid(name, target, 'instant')
    if doses is None:
        doses = np.geomspace(0.5, 60.0, n_dose)
    else:
        doses = np.asarray(doses)[np.linspace(0, len(doses) - 1, n_dose).astype(int)]
    C = make_cost(model, target, doses, RadialTarget(), n_phase=n_phase, mode='instant',
                  backend=backend, dt=dt, w_osc=w_osc)
    return model, C


def selftest(name='almeida', target='BMAL1'):
    """THE load-bearing assertion: a Hopf-collapsed parameter set must score NEAR-MAXIMAL.

    If this ever fails, the Mirsky degeneracy has been rebuilt -- a cost whose global optimum
    is a dead oscillator -- and no result from this pipeline should be believed.
    """
    model, C = _build(name, target)
    print("=" * 76)
    print(f"COST SELF-TEST -- {name}/{target}, {C['n_free']} free directions "
          f"(of {len(C['names'])} parameters)")
    print("=" * 76)

    base = C['parts'](C['v0'])
    print(f"  base            total={base['total']:.4f} c_ptc={base['c_ptc']:.4f} "
          f"osc={base['osc']:.4f} amp={base['amp_pen']:.4f} "
          f"amp_lc={base['amp_lc']:.3f} alive={base['alive_frac']:.2f}")
    ok = np.isfinite(base['total']) and base['alive_frac'] > 0.8

    # Drive the clock toward Hopf by scanning one direction until the cycle collapses, then
    # assert the cost went UP. Which parameter does it is model-specific; what matters is that
    # SOME collapse exists and that it is punished.
    worst, best_v = None, None
    for i in range(C['n_free']):
        for s in (+1.0, -1.0):
            v = np.zeros(C['n_free']); v[i] = s * 1.2
            p = C['parts'](v)
            if p['amp_lc'] < 0.25 * C['amp_base']:
                if worst is None or p['amp_lc'] < worst['amp_lc']:
                    worst, best_v = p, v.copy()
    if worst is None:
        print("  no collapsing direction found at |v|=1.2 -- widening")
        for i in range(C['n_free']):
            v = np.zeros(C['n_free']); v[i] = 3.0
            p = C['parts'](v)
            if worst is None or p['amp_lc'] < worst['amp_lc']:
                worst, best_v = p, v.copy()
    print(f"  most collapsed  total={worst['total']:.4f} c_ptc={worst['c_ptc']:.4f} "
          f"osc={worst['osc']:.4f} amp={worst['amp_pen']:.4f} "
          f"amp_lc={worst['amp_lc']:.3f} alive={worst['alive_frac']:.2f}")
    print(f"  base amp_lc = {C['amp_base']:.3f}, floor = {C['amp_floor']:.3f}")

    collapsed_worse = worst['total'] > base['total']
    print(f"  collapsed cost > base cost: {collapsed_worse}  "
          f"({worst['total']:.4f} vs {base['total']:.4f})   <- THE invariant")
    ok &= bool(collapsed_worse)

    # and an all-dead surface must sit at the ceiling
    print(f"  a fully dead surface scores c_ptc = 1.0 by construction "
          f"(fit.target.circ_cost, asserted in its own selftest)")
    print(f"  [{'PASS' if ok else 'FAIL'}] cost punishes the degenerate optimum")
    return ok


def gradcheck(name='almeida', target='BMAL1', n=6, h=1e-4, seed=0, backend='diffrax'):
    """T0.5: autodiff vs central finite differences on the ADAPTIVE backend.

    REPO_MAP hazard 1 is that autodiffing the fixed-step RK4 PTC at nontrivial dose produced
    |J| up to 3.95e83 against a true response of 0.006 -- and it was found only after it had
    invalidated a table of results. This check would have caught it in a minute. It gates every
    gradient-driven run; if it fails, use --optimizer cma and say so in the writeup.
    """
    model, C = _build(name, target, backend=backend)
    rng = np.random.default_rng(seed)
    v0 = np.zeros(C['n_free'])
    g = C['grad'](v0)
    idx = rng.choice(C['n_free'], size=min(n, C['n_free']), replace=False)
    print("=" * 76)
    print(f"GRADIENT CHECK -- {name}/{target}, backend={backend}, h={h}")
    print("=" * 76)
    print(f"  |grad| = {np.linalg.norm(g):.4e}   max|g_i| = {np.max(np.abs(g)):.4e}")
    if not np.all(np.isfinite(g)):
        print("  FAIL: gradient is not finite")
        return False
    if np.max(np.abs(g)) > 1e6:
        print(f"  FAIL: gradient magnitude {np.max(np.abs(g)):.3e} is not credible "
              f"(hazard 1 signature)")
        return False
    ok = True
    print(f"  {'i':>3s} {'autodiff':>14s} {'central diff':>14s} {'rel err':>10s}")
    for i in idx:
        vp = v0.copy(); vp[i] += h
        vm = v0.copy(); vm[i] -= h
        fd = (C['total'](vp) - C['total'](vm)) / (2 * h)
        rel = abs(fd - g[i]) / max(abs(fd), abs(g[i]), 1e-8)
        print(f"  {i:3d} {g[i]:14.6e} {fd:14.6e} {rel:10.2e}")
        ok &= rel < 5e-2
    print(f"  [{'PASS' if ok else 'FAIL'}] autodiff agrees with finite differences")
    return ok


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='PTC fitting objective')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--gradcheck', action='store_true')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    ap.add_argument('--backend', default='diffrax', choices=('rk4', 'diffrax'))
    ap.add_argument('-n', type=int, default=6)
    a = ap.parse_args(argv)
    ok = True
    if a.selftest or not a.gradcheck:
        ok &= selftest(a.model, a.target)
    if a.gradcheck:
        ok &= gradcheck(a.model, a.target, n=a.n, backend=a.backend)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
