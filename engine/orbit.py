"""
engine/orbit.py
===============
The periodic orbit as the SOLUTION of a boundary-value problem, not as the by-product of a
long relaxation. Model-agnostic (models.api JAX capability); differentiable; vmappable.

Adapted from input_screen/orbit_jax.py (commit e955873), which is already model-agnostic; the
changes here are the import paths and a registry-driven self-test.

WHY A BVP RATHER THAN RELAX-AND-ESTIMATE
----------------------------------------
The obvious way to get a limit cycle is: relax for ~15 periods, then ESTIMATE the period (FFT
peak) and the phase (windowed DFT), then sample a fixed window. Four things go wrong as soon
as you differentiate that:

  * phase along a limit cycle is NEUTRALLY stable (the tangential Floquet multiplier is
    exactly 1, forced by time-translation invariance), so the arrival phase is arbitrary,
    never self-corrects, and has to be estimated rather than pinned;
  * argmax over FFT bins and argsort over phase labels both have identically ZERO derivative,
    so autodiff silently differentiates "holding the bin and the ordering fixed";
  * the sample window is a fixed nominal period, not one TRUE period T;
  * the transient dominates the cost.

Measured consequence in input_screen: the resulting LC jacobian reported sigma/sigma_max up to
7.2e-5 for a direction that is PROVABLY exactly zero, while its SVD called 1e-16 "null" -- so
a null space could not be read off it, which invalidated a whole analysis. The same quantity
through this solver reads 3.8e-12.

THE FORMULATION
---------------
Unknowns x = (y0, T). Integrate the TIME-RESCALED system over tau in [0, 1]:

        dy/dtau = T * f(y; P)                                                  (1)

so the step count is static (lax.scan-friendly) while T stays a differentiable unknown, and
**tau IS the phase** -- exactly, by construction. No FFT, no window, no sort, no interpolation
onto an estimated phase grid.

        F(x; P) = [ phi(y0, T) - y0 ,  g(y0; P) ] = 0                          (2)

with `phi` the time-T flow and `g` a scalar PHASE CONDITION pinning the neutral direction.
Default: g = f(y0)[ref], i.e. y0 sits at a turning point of the reference species. Chosen
because it is GAUGE-COVARIANT: under a species rescaling S, f(S y)[ref] = mu_ref f(y)[ref], so
the section is preserved and the solve commutes with the symmetry. A fixed hyperplane would
NOT have that property, and would reintroduce exactly the phase artifact this module removes.

Derivatives come from the implicit function theorem applied to (2): Newton runs to convergence
under stop_gradient, then ONE differentiable Newton step makes dx*/dP exact at the cost of a
single (n+1)x(n+1) linear solve.

The initial guess comes from a short relaxation. It never enters the derivative, so its
inaccuracy is irrelevant -- which is the whole point.

Run `python -m engine.orbit [model ...]` for the self-test.
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from jax import lax

from models.api import require


# --------------------------------------------------------------------------- #
#  Time-rescaled flow:  dy/dtau = T f(y),  tau in [0, 1]
# --------------------------------------------------------------------------- #
def _rk4_step(rhs, y, h, T, P):
    k1 = rhs(y, P)
    k2 = rhs(y + 0.5 * h * T * k1, P)
    k3 = rhs(y + 0.5 * h * T * k2, P)
    k4 = rhs(y + h * T * k3, P)
    return y + (h * T / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _flow(rhs, y0, T, P, n_steps, full=False):
    """Integrate (1) over one period. full=False -> endpoint; True -> (n_steps+1, n) trace
    whose row i is the state at phase i/n_steps. The tau grid IS the phase grid."""
    h = 1.0 / n_steps

    def step(y, _):
        y2 = _rk4_step(rhs, y, h, T, P)
        return y2, (y2 if full else None)

    yf, tr = lax.scan(step, y0, None, length=n_steps)
    if not full:
        return yf
    return jnp.concatenate([y0[None], tr], axis=0)


# --------------------------------------------------------------------------- #
#  Solver
# --------------------------------------------------------------------------- #
class OrbitSolver:
    """Periodic-orbit BVP solver for one model.

    solve(P, guess)        -> (y0, T, residual_norm)
    cycle(P, y0, T, m)     -> (m, n_states) states at phases 0, 1/m, ..., (m-1)/m
    guess(P)               -> x0 = concat([y0, T])   (host-side, never differentiated)
    """

    def __init__(self, model, n_steps=1024, ref=None, newton_iters=8, damping=1.0,
                 phase_condition=None):
        # newton_iters=8 by default, not 4: Almeida converges to |F| ~ 1e-13 in 4, but Goodwin
        # (whose guess starts further out) still sits at 1.8e-6 after 4 and only reaches 8e-16
        # at 8. Under-convergence is invisible in the self-convergence check -- a coarse and a
        # fine solver are wrong in the SAME way -- so it must be prevented, not detected.
        # The extra iterations are cheap (Newton is ~1 ms/iter here) and never differentiated.
        require(model, 'jax', consumer='engine.orbit.OrbitSolver')
        self.model = model
        self.rhs = model.jax_rhs
        self.n = int(model.n_states)
        self.n_steps = int(n_steps)
        self.newton_iters = int(newton_iters)
        self.damping = float(damping)
        ref = ref or getattr(model, 'reference_variable', None) or model.state_names[0]
        self.ref_idx = int(model.var_index(ref))
        self.ref_name = ref
        # default: turning point of the reference species -- a genuine Poincare section, and
        # gauge-covariant (see module docstring).
        self._phase = phase_condition or (lambda y0, P: self.rhs(y0, P)[self.ref_idx])

    # -- residual + Newton -------------------------------------------------- #
    def residual(self, x, P):
        y0, T = x[:self.n], x[self.n]
        return jnp.concatenate([_flow(self.rhs, y0, T, P, self.n_steps) - y0,
                                jnp.array([self._phase(y0, P)])])

    def _newton(self, x0, P):
        """Damped Newton, value only (stop_gradient)."""
        eye = jnp.eye(self.n + 1)

        def body(x, _):
            F = self.residual(x, P)
            J = jax.jacfwd(self.residual)(x, P)
            return x - self.damping * jnp.linalg.solve(J + 1e-12 * eye, F), None
        xc, _ = lax.scan(body, x0, None, length=self.newton_iters)
        return lax.stop_gradient(xc)

    def solve(self, P, x0):
        """(y0, T, residual_norm) with EXACT implicit derivatives w.r.t. P.

        Newton converges under stop_gradient; one further DIFFERENTIABLE step then satisfies
        the implicit function theorem, so dx*/dP is exact and costs a single linear solve --
        independent of how many Newton iterations ran.

        NOTE: J must be evaluated AT the converged point. Reusing the last Jacobian from the
        Newton loop looks like a free 2x, but lax.scan's carry hands back J from the PREVIOUS
        iterate; that staleness is invisible in the residual (|F| ~ 1e-14 either way) and
        shows up only in the derivative -- it degraded the exactness test from 1e-13 to 2e-6
        in input_screen. Measured, not hypothetical."""
        xc = self._newton(jnp.asarray(x0, jnp.float64), P)
        F = self.residual(xc, P)                                   # differentiable in P
        J = lax.stop_gradient(jax.jacfwd(self.residual)(xc, P))    # at the CONVERGED point
        x = xc - jnp.linalg.solve(J + 1e-12 * jnp.eye(self.n + 1), F)
        res = jnp.linalg.norm(self.residual(lax.stop_gradient(x), P))
        return x[:self.n], x[self.n], res

    # -- sampling ----------------------------------------------------------- #
    def cycle(self, P, y0, T, m=None):
        """States at m evenly spaced phases starting at y0 (phase 0). If m divides n_steps the
        samples are exact RK4 grid points -- no interpolation anywhere."""
        tr = _flow(self.rhs, y0, T, P, self.n_steps, full=True)[:-1]      # drop the wrap point
        if m is None or m == self.n_steps:
            return tr
        idx = (jnp.arange(m) * (self.n_steps // m)) if self.n_steps % m == 0 else \
            jnp.round(jnp.arange(m) * self.n_steps / m).astype(int)
        return tr[idx]

    # -- stability ---------------------------------------------------------- #
    def monodromy(self, P, y0, T):
        """d(phi_T(y0))/dy0 -- the linearised period map. Its eigenvalues are the Floquet
        multipliers."""
        return jax.jacfwd(lambda y: _flow(self.rhs, y, T, P, self.n_steps))(y0)

    def floquet(self, P, y0, T):
        """(mu_lead, all_multipliers) with the trivial phase direction removed.

        One multiplier is exactly 1: perturbing along the flow just re-times the orbit, and
        time-translation invariance makes that direction neutrally stable. `mu_lead` is the
        largest of the REST -- it sets how fast an off-cycle perturbation relaxes back, i.e.
        how long you must wait before an "asymptotic" phase reading is actually asymptotic.
        """
        ev = np.linalg.eigvals(np.asarray(self.monodromy(P, y0, T)))
        mag = np.abs(ev)
        drop = int(np.argmin(np.abs(mag - 1.0)))       # the one closest to 1 == the phase mode
        rest = np.delete(mag, drop)
        return (float(rest.max()) if len(rest) else 0.0), ev

    def relax_periods(self, P, y0, T, tol=1e-3):
        """How many periods a transient needs to decay to `tol`, from the Floquet multiplier.

        This is what sets `skip_p` in engine/ptc.py. It is NOT a constant across models:
        Almeida's leading multiplier is small, so a few periods suffice, while Goodwin's is
        0.79 -- after 6 periods 24% of the transient is still there, which biases every
        "asymptotic" phase it reports. Deriving the number instead of assuming one turns a
        silent bias into a computed quantity."""
        mu, _ = self.floquet(P, y0, T)
        if not (0.0 < mu < 1.0):
            return float('inf')                        # marginal or unstable: no finite answer
        return float(np.log(tol) / np.log(mu))

    # -- initial guess (never differentiated) ------------------------------- #
    def guess(self, P, y0=None, n_relax=8, gp=None):
        """Short relaxation -> (y0, T) guess. Numpy/host side on purpose: this only has to
        land in Newton's basin, and it is deliberately outside the derivative path.

        Anchors on a MAXIMUM of the reference species. The section f(y)[ref] = 0 has two roots
        per cycle (a maximum and a minimum); if different parameter sets anchored on different
        roots their cycles would be half a period out of register and any absolute LC
        comparison between them would be meaningless."""
        gp = float(gp or getattr(self.model, 'approx_period', None) or 1.0)
        y = jnp.asarray(self.model.get_initial_state() if y0 is None else y0, jnp.float64)
        if not hasattr(self, '_relax_jit'):
            self._relax_jit = jax.jit(
                lambda yy, Pp: lax.fori_loop(0, int(n_relax), lambda _i, s:
                                             _flow(self.rhs, s, gp, Pp, self.n_steps), yy))
            self._trace_jit = jax.jit(
                lambda yy, Pp: _flow(self.rhs, yy, 2.5 * gp, Pp,
                                     int(2.5 * self.n_steps), full=True))
        y = self._relax_jit(y, P)
        tr = np.asarray(self._trace_jit(y, P))
        r = tr[:, self.ref_idx]
        dt = 2.5 * gp / (len(r) - 1)
        pk = [i for i in range(1, len(r) - 1) if r[i] > r[i - 1] and r[i] >= r[i + 1]]
        if len(pk) >= 2:
            T = float((pk[-1] - pk[-2]) * dt)
            y0g = tr[pk[-1]]
        else:                                    # no clean peak: fall back to nominal period
            T, y0g = gp, tr[-1]
        return jnp.concatenate([jnp.asarray(y0g), jnp.array([T])])


# --------------------------------------------------------------------------- #
#  Convenience: params -> (T, cycle) as one jitted, differentiable call
# --------------------------------------------------------------------------- #
def make_cycle_fn(model, m=192, n_steps=1024, ref=None, newton_iters=12):
    """jit(f)(P, x0) -> (T, cycle[m, n_states], residual). The phases are exactly
    arange(m)/m, so no phase array is returned -- there is nothing to estimate."""
    solver = OrbitSolver(model, n_steps=n_steps, ref=ref, newton_iters=newton_iters)

    def f(P, x0):
        y0, T, res = solver.solve(P, x0)
        return T, solver.cycle(P, y0, T, m), res

    return jax.jit(f), solver


# --------------------------------------------------------------------------- #
#  Jittable, vmappable initial guess  (required for any SWEEP or GLOBAL search)
# --------------------------------------------------------------------------- #
def make_guess_fn(model, n_steps=1024, ref=None, n_relax=8, gp=None, scan_frac=1.3):
    """Return jit(guess)(P, y_seed) -> x0 = concat([y0, T]), vmappable over a population.

    WHY THIS EXISTS. `OrbitSolver.guess` does a numpy peak-hunt, so it cannot be vmapped, and
    a sweep that reuses ONE nominal-derived x0 for every candidate simply fails: at a
    displaced parameter set that guess is outside Newton's basin, the solve does not converge,
    and every such candidate is SILENTLY rejected. In input_screen that emptied the
    unconstrained arm of a whole search and produced -inf fitness across a batch -- it read as
    a scientific negative rather than a bug.

    PHASE CONSISTENCY IS THE SUBTLE PART. The section f(y)[ref] = 0 has TWO roots per cycle. So
    after relaxing we advance to the next MAXIMUM of the reference species -- the first index
    where its derivative goes + -> - -- which is jittable via a masked argmax.
    """
    rhs = model.jax_rhs
    ref = ref or getattr(model, 'reference_variable', None) or model.state_names[0]
    ref_idx = int(model.var_index(ref))
    gp = float(gp or getattr(model, 'approx_period', None) or 1.0)
    n_scan = int(round(scan_frac * n_steps))

    def guess(P, y_seed):
        y = lax.fori_loop(0, int(n_relax),
                          lambda _i, s: _flow(rhs, s, gp, P, n_steps), y_seed)
        # one scan_frac-period trace at the NOMINAL rate; tau spacing = gp / n_steps
        tr = _flow(rhs, y, gp * scan_frac, P, n_scan, full=True)
        d = jax.vmap(lambda s: rhs(s, P)[ref_idx])(tr)            # d(ref)/dt along the trace
        up_then_down = (d[:-1] > 0) & (d[1:] <= 0)                # + -> -  == a maximum
        i_max = jnp.argmax(up_then_down)                          # first True (0 if none)
        y0 = tr[i_max + 1]
        later = up_then_down & (jnp.arange(up_then_down.shape[0]) > i_max)
        i2 = jnp.argmax(later)
        dt = gp * scan_frac / n_scan
        T = jnp.where(jnp.any(later), (i2 - i_max) * dt, gp)
        T = jnp.clip(T, 0.3 * gp, 3.0 * gp)                       # keep Newton in a sane range
        return jnp.concatenate([y0, T[None]])

    return jax.jit(guess)


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #
def _scipy_period(model, tol=1e-7, max_periods=1600):
    """Independent period by pure relaxation: adaptive LSODA + quadratic-interpolated peaks,
    relaxing LONGER until the estimate stops moving. Returns (T, n_periods_needed).

    The relaxation length must be converged, not assumed. At a fixed 40 periods this reported
    23.6727 for Goodwin against a true 23.5398 -- a 5.6e-3 relative error that looked like a
    solver bug when it was an under-relaxed REFERENCE. Goodwin's cycle attracts slowly and
    needs ~400 periods; Almeida is converged by 40. That gap is precisely the case for solving
    the orbit as a BVP instead: 8 Newton steps (~5 ms) reach what 400 periods of adaptive
    integration reach.
    """
    from scipy.integrate import solve_ivp
    gp = float(model.approx_period)
    p = model.get_parameters()
    ref = model.var_index(model.reference_variable)

    def period_after(n_periods):
        sol = solve_ivp(lambda t, y: model.derivatives(t, y, p), [0, n_periods * gp],
                        model.get_initial_state(), method='LSODA', dense_output=True,
                        rtol=1e-12, atol=1e-14)
        t = np.linspace((n_periods - 6) * gp, n_periods * gp, 200000)
        x = sol.sol(t)[ref]
        pk = []
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] > x[i + 1]:
                a = 0.5 * (x[i - 1] - 2 * x[i] + x[i + 1])
                b = 0.5 * (x[i + 1] - x[i - 1])
                pk.append(t[i] + (-b / (2 * a) if a < 0 else 0.0) * (t[1] - t[0]))
        return float(np.mean(np.diff(pk))) if len(pk) > 2 else float('nan')

    n, prev = 50, period_after(50)
    while n < max_periods:
        n *= 2
        cur = period_after(n)
        if np.isfinite(cur) and np.isfinite(prev) and abs(cur - prev) / cur < tol:
            return cur, n
        prev = cur
    return prev, n


def selftest(name='almeida'):
    import time
    from models import get_model

    model = get_model(name)
    P = model.jax_params()
    solver = OrbitSolver(model)
    print("=" * 76)
    print(f"ORBIT SELF-TEST -- {type(model).__name__} "
          f"({model.n_states} states, ref={solver.ref_name})")
    print("=" * 76)

    t0 = time.time(); x0 = solver.guess(P); t_guess = time.time() - t0
    sol = jax.jit(solver.solve)
    t0 = time.time(); y0, T, res = sol(P, x0); float(T); t_compile = time.time() - t0
    t0 = time.time(); y0, T, res = sol(P, x0); float(T); t_solve = time.time() - t0

    print("\n1. SOLVE -- Newton on the BVP vs an independent adaptive relaxation")
    T_sp, n_rel = _scipy_period(model)
    print(f"   period   BVP {float(T):.9f}   LSODA {T_sp:.9f}   "
          f"|d|/T = {abs(float(T) - T_sp) / T_sp:.2e}")
    print(f"   residual |F| = {float(res):.2e}   (guess T was {float(x0[-1]):.4f})")
    print(f"   timing: guess {t_guess:.2f}s | solve {t_solve * 1000:.1f} ms "
          f"(compile {t_compile:.1f}s)")
    print(f"   the relaxation reference needed {n_rel} periods to converge; the BVP needed "
          f"{solver.newton_iters} Newton steps")

    print("\n2. CYCLE -- self-convergence in n_steps (RK4 is 4th order)")
    hi = OrbitSolver(model, n_steps=8 * solver.n_steps)
    y0h, Th, _r = jax.jit(hi.solve)(P, hi.guess(P))
    Ch = np.asarray(hi.cycle(P, y0h, Th, 64))
    Cl = np.asarray(solver.cycle(P, y0, T, 64))
    dT = abs(float(T) - float(Th)) / float(Th)
    dy = float(np.max(np.abs(Cl - Ch)) / np.max(np.abs(Ch)))
    print(f"   vs an 8x-finer orbit: |dT|/T = {dT:.1e}, max|dy|/scale = {dy:.1e}")

    print("\n3. GAUGE-COVARIANCE of the phase condition")
    print("   A species rescaling must map the solved orbit to the rescaled orbit EXACTLY:")
    from gauge.gauge import Gauge, random_gauge, z_to_params
    g = Gauge(model)
    w = random_gauge(g, sigma=0.4, seed=1)
    m2 = type(model)(); m2.set_parameters(z_to_params(g.apply(g.z_nominal, w), g))
    S = np.array([np.exp(w[g.spec.col(s)]) for s in model.state_names])
    s2 = OrbitSolver(m2, n_steps=solver.n_steps)
    y02, T2, _r2 = jax.jit(s2.solve)(m2.jax_params(), s2.guess(m2.jax_params()))
    C2 = np.asarray(s2.cycle(m2.jax_params(), y02, T2, 64))
    C1 = np.asarray(solver.cycle(P, y0, T, 64)) * S           # predicted rescaled orbit
    rel = float(np.max(np.abs(C2 - C1)) / np.max(np.abs(C1)))
    print(f"   period invariant: |dT|/T = {abs(float(T2) - float(T)) / float(T):.2e}")
    print(f"   cycle == S * cycle: max rel err = {rel:.2e}")

    ok = (abs(float(T) - T_sp) / T_sp < 1e-4) and float(res) < 1e-8 and dT < 1e-6
    print(f"\n   [{'PASS' if ok else 'FAIL'}] orbit solve is accurate and self-consistent")
    print("=" * 76)
    return ok


if __name__ == '__main__':
    import sys
    from models import MODEL_REGISTRY
    names = sys.argv[1:] or sorted(MODEL_REGISTRY)
    good = all(selftest(n) for n in names)
    raise SystemExit(0 if good else 1)
