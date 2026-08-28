"""
engine/flow.py
==============
Adaptive Tsit5 (diffrax) integration, a PARALLEL backend to the fixed-step RK4 path in
engine/perturb.py -- not a replacement. Ported from input_screen/cycle_diffrax.py (which was
itself a parallel backend to that repo's RK4 cycle build), generalized off Mirsky: the only
model-specific couplings there were `from mirsky_jax import rhs` and `fourier_period`, and both
are supplied here by the ClockModel API and by engine/ptc.py respectively.

WHY IT EXISTS
    Two independent reasons, and the second is the one that motivated this port.

    1. Accuracy/speed. Tsit5 is 5th-order and adaptive: ~1.8x fewer RHS evaluations at equal
       accuracy than fixed RK4, and it stays bounded on the stiff high-dose transient where
       fixed RK4 at dt=0.02 rings negative (engine/perturb.make_forced_flow documents the
       measurement: Goldbeter MP at dose 20 reaches -2.8e-1 at dt=0.02 and -9.7e-3 at
       dt=0.005 -- a step-size instability, not a property of the model).

    2. GRADIENTS. REPO_MAP hazard 1 records that autodiffing the fixed-step RK4 PTC at
       nontrivial dose produced |J| up to 3.95e83 against an actual response of 0.006, and
       invalidated a table of results. That hazard indicts *differentiating fixed-step RK4*,
       NOT gradients as such -- input_screen's own resolution was this adaptive backend, and
       with it L-BFGS reached 132/132 parameter recovery from the truth on Mirsky where CMA-ES
       did worse. A gradient-based fit therefore runs on THIS backend, and only after
       `fit/cost.py --gradcheck` confirms autodiff agrees with finite differences.

HOW FAILURE IS REPORTED
    An adaptive solver has a failure mode the fixed-step one does not: `max_steps_reached` on a
    pathologically stiff parameter set. Rather than invent a second validity channel, that is
    mapped onto the EXISTING one -- the running-minimum `y_min` that engine/ptc.valid_mask
    already thresholds at zero. A failed solve reports `y_min = -inf`, so it is marked invalid
    by exactly the same downstream code that catches an RK4 instability, and no caller changes.

WHAT IS DELIBERATELY IDENTICAL TO THE RK4 PATH
    The readout grid. `sample_window` returns states on the SAME uniform grid of `n` points
    spaced `dt` that the RK4 `_four_vec` scan visits, so engine/ptc.py applies one Hann-window
    formula to either backend and an A/B difference is attributable to the integrator alone.
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

try:
    import diffrax as dfx
    HAVE_DIFFRAX = True
except ImportError:                                    # keep the RK4 path importable without it
    dfx = None
    HAVE_DIFFRAX = False

BACKENDS = ('rk4', 'diffrax')

#: Carried over verbatim from input_screen/cycle_diffrax.py, where they were tuned against the
#: RK4 path. MAX_STEPS: valid solves are small (per-cell ~1.5-5k Tsit5 steps at dose 8), so 50k
#: covers them with margin while a pathologically stiff parameter set fails ~4x faster than at
#: 200k. Under vmap the whole batch runs to the slowest cell, so this cap is load-bearing.
RTOL, ATOL, DT0, MAX_STEPS = 1e-7, 1e-10, 0.05, 50_000


def check_backend(backend):
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == 'diffrax' and not HAVE_DIFFRAX:
        raise ImportError("backend='diffrax' needs `pip install diffrax` "
                          "(cluster note: add it to the env, see slurm/README.md)")
    return backend


def _ok(result):
    """A VALID termination: normal completion or a triggered event. We flag only the failure we
    actually hit -- max_steps_reached, i.e. pathologically stiff parameters. (Comparing against
    the event enum raises a type error in diffrax; comparing to max_steps_reached works across
    all result types.)"""
    return result != dfx.RESULTS.max_steps_reached


#: What `y_min` becomes when the solver failed, so engine/ptc.valid_mask marks the cell invalid
#: through the existing channel rather than a new one.
FAILED_MIN = -jnp.inf


def make_flow(model, target_idx, backend='rk4', track_min=False):
    """`flow(y0, n_steps, dt, P, drive, ymin0=None)` -- the same signature the RK4 path exposes,
    so engine/ptc.py's call sites are backend-independent.

    `n_steps * dt` is the physical duration to advance. The fixed-step path takes that literally
    as a step count; the adaptive path takes only the product and chooses its own steps.

    With track_min=True returns `(y_final, y_min)`, where `y_min` is the most negative value any
    state reached -- carried so a caller can discard a point the integrator was not stable at.
    On the adaptive path a solver failure yields `y_min = -inf` (see FAILED_MIN).
    """
    check_backend(backend)
    if backend == 'rk4':
        from engine.perturb import make_forced_flow
        return make_forced_flow(model, target_idx, track_min=track_min)

    rhs = model.jax_rhs
    n = int(model.n_states)
    e = jnp.zeros(n).at[target_idx].set(1.0)

    def field(t, y, args):
        P, drive = args
        return rhs(y, P) + drive * e

    term = dfx.ODETerm(field)

    def flow(y0, n_steps, dt, P, drive, ymin0=None):
        # The step size is T/n_pp with T the SOLVED period, so under an optimizer it can arrive
        # non-positive or NaN from a parameter set with no usable cycle. diffrax asserts
        # (t1-t0)*dt0 >= 0 and raises through a host callback, which kills the whole jitted
        # cost instead of scoring that candidate badly. Route it into the same failure channel
        # everything else uses: substitute a dummy span, then mark the result invalid.
        t1 = jnp.nan_to_num(n_steps * dt, nan=0.0, posinf=0.0, neginf=0.0)
        bad_span = t1 <= 0.0
        t1s = jnp.where(bad_span, 1.0, t1)
        # Sample the interior as well as the endpoint: `y_min` must reflect the whole
        # trajectory, exactly as the RK4 running minimum does. A coarse probe grid is enough --
        # we are detecting gross excursions, not resolving them.
        n_probe = 64
        ts = jnp.linspace(0.0, t1s, n_probe)
        sol = dfx.diffeqsolve(
            term, dfx.Tsit5(), 0.0, t1s, DT0, y0, args=(P, drive),
            stepsize_controller=dfx.PIDController(rtol=RTOL, atol=ATOL),
            saveat=dfx.SaveAt(ts=ts), max_steps=MAX_STEPS, throw=False)
        ys = sol.ys
        good = _ok(sol.result) & ~bad_span
        mn = jnp.where(good, jnp.min(ys), FAILED_MIN)
        if ymin0 is not None:
            mn = jnp.minimum(mn, ymin0)
        yf = jnp.where(good, ys[-1], jnp.full_like(ys[-1], jnp.nan))
        return (yf, mn) if track_min else yf

    return flow


def make_window_sampler(model, backend='rk4'):
    """`sample(y0, n, dt, P) -> states[n, n_states]` on the uniform grid `0, dt, ..., (n-1)*dt`.

    This is the readout window engine/ptc.py Hann-windows into a Fourier fundamental. The RK4
    path reproduces its scan exactly (same states at the same grid points); the adaptive path
    integrates with Tsit5 and evaluates the dense interpolant at the same grid, so the two
    differ only by integration error and the Hann formula downstream is shared.

    NOTE the grid convention matches the RK4 `_four_vec` scan, which records the state AFTER
    each step -- so sample j is at time (j+1)*dt, not j*dt. Getting this wrong shifts the
    measured phase by one step and shows up immediately in the dose-0 identity check.
    """
    check_backend(backend)
    rhs = model.jax_rhs

    if backend == 'rk4':
        def sample(y0, n, dt, P):
            def step(y, _):
                k1 = rhs(y, P); k2 = rhs(y + 0.5 * dt * k1, P)
                k3 = rhs(y + 0.5 * dt * k2, P); k4 = rhs(y + dt * k3, P)
                y2 = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                return y2, y2
            _yf, ys = jax.lax.scan(jax.checkpoint(step), y0, None, length=n)
            return ys
        return sample

    def field(t, y, args):
        return rhs(y, args)

    term = dfx.ODETerm(field)

    def sample(y0, n, dt, P):
        # Same non-positive/NaN span guard as `make_flow` -- an optimizer reaches parameter sets
        # whose solved period is degenerate, and diffrax raises rather than returning.
        d = jnp.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0)
        bad_span = d <= 0.0
        ds = jnp.where(bad_span, 1.0, d)
        ts = (jnp.arange(n) + 1) * ds                  # state AFTER each step -- see above
        sol = dfx.diffeqsolve(
            term, dfx.Tsit5(), 0.0, ts[-1], DT0, y0, args=P,
            stepsize_controller=dfx.PIDController(rtol=RTOL, atol=ATOL),
            saveat=dfx.SaveAt(ts=ts), max_steps=MAX_STEPS, throw=False)
        return jnp.where(_ok(sol.result) & ~bad_span, sol.ys, jnp.nan)

    return sample


# --------------------------------------------------------------------------- #
#  Self-test: the two backends must agree on the forward problem
# --------------------------------------------------------------------------- #
def selftest(name='almeida', hours=48.0, dt=0.02):
    """A/B the backends on the unforced and forced flow, and on the readout window.

    This is the check that licenses using diffrax anywhere: if the two integrators disagree in
    the FORWARD direction, nothing downstream of either can be trusted. RK4 at dt=0.02 is the
    incumbent that every batch-1 result was computed with, so agreement here means the backend
    swap does not move any published number.
    """
    from models import get_model
    from engine.orbit import OrbitSolver
    from engine.perturb import resolve_target

    if not HAVE_DIFFRAX:
        print("diffrax not installed -- skipping"); return True

    model = get_model(name)
    P = model.jax_params()
    tgt = model.perturbable_targets()[0]
    ti = resolve_target(model, tgt)
    solver = OrbitSolver(model)
    y0, T, _r = jax.jit(solver.solve)(P, solver.guess(P))
    n = int(round(hours / dt))

    print("=" * 76)
    print(f"FLOW BACKEND A/B -- {type(model).__name__}, target {tgt}, {hours:g} h at dt={dt}")
    print("=" * 76)

    ok = True
    for drive, label in ((0.0, 'unforced'), (2.0, 'forced (dose 2)')):
        a = np.asarray(make_flow(model, ti, 'rk4', track_min=True)(y0, n, dt, P, drive)[0])
        b = np.asarray(make_flow(model, ti, 'diffrax', track_min=True)(y0, n, dt, P, drive)[0])
        rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
        print(f"   {label:16s} max rel diff = {rel:.3e}")
        ok &= rel < 1e-5

    m = 512
    sa = np.asarray(make_window_sampler(model, 'rk4')(y0, m, dt, P))
    sb = np.asarray(make_window_sampler(model, 'diffrax')(y0, m, dt, P))
    rel = float(np.max(np.abs(sa - sb)) / np.max(np.abs(sa)))
    print(f"   {'readout window':16s} max rel diff = {rel:.3e} over {m} samples")
    ok &= rel < 1e-5

    # the failure channel must actually fire, and must be readable by valid_mask
    from engine.ptc import valid_mask
    huge = make_flow(model, ti, 'diffrax', track_min=True)(y0, n, dt, P, 1e9)
    print(f"   dose 1e9 -> y_min = {float(np.asarray(huge[1])):.3g}, "
          f"valid_mask says {bool(valid_mask(np.asarray(huge[1])))} (must be False)")
    ok &= not bool(valid_mask(np.asarray(huge[1])))

    print(f"   [{'PASS' if ok else 'FAIL'}] backends agree and failures are flagged")
    return ok


if __name__ == '__main__':
    import sys
    raise SystemExit(0 if selftest(*(sys.argv[1:] or ['almeida'])) else 1)
