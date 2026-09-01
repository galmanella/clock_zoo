"""
engine/validate.py
==================
THE GATE. Nothing in analysis/ runs against a model until this passes for it.

    $PY -m engine.validate [model ...] [--quick]

Eight checks, each of which has already caught something real:

  1 conformance    all five capability tiers declared and self-consistent.
  2 jax-vs-numpy   the JAX RHS and the numpy RHS must agree to ~1e-12. Two implementations
                   of the same equations drift the moment one is edited.
  3 orbit          BVP residual, period against an adaptively-relaxed reference, and
                   self-convergence against an 8x-finer solve. Under-convergence is INVISIBLE
                   to self-convergence alone (a coarse and a fine solver are wrong the same
                   way), so the residual is checked separately.
  4 dt             halve the RK4 step until the PTC stops moving. Establishes that the step
                   size is not the limiting error before anything is read off a PTC.
  5 ptc            the cross-check that matters: JAX/RK4/Fourier against scipy/LSODA/peaks,
                   two engines sharing no numerical machinery, with NO offset fitting and
                   agreement required on which cells are dead.
  6 gauge          the defining symmetry identity, the projection algebra, and forward
                   invariance of the period and relative amplitudes.
  7 positivity     a perturbation must not drive a concentration negative, and a perturbation
                   that kills the oscillator must be REPORTED as dead rather than assigned a
                   phase off a near-zero Fourier coefficient.
  8 timing         cost of one PTC grid, which is what sizes the sweeps in analysis/.

`--quick` skips 4, 5 and 8 (the expensive ones); use it while editing a model, and run the
full gate before trusting a number.
"""
import sys
import time

import numpy as np

import engine  # noqa: F401  -- puts the repo root on sys.path


def _line(ok, label, detail=''):
    print(f"   [{'PASS' if ok else 'FAIL'}] {label:34s} {detail}", flush=True)
    return bool(ok)


# --------------------------------------------------------------------------- #
def check_conformance(model):
    from models.api import check_conformance as cc
    res = cc(model, verbose=False)
    bad = [k for k, v in res.items() if v is not True]
    return _line(not bad, 'conformance (5 tiers)',
                 'all tiers declared' if not bad else f'failing: {bad}')


def check_jax_vs_numpy(model, n=200, seed=0):
    """The two RHS implementations must agree. Random POSITIVE states: both clamp at 0, and
    the negative orthant is unphysical anyway."""
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)
    y0 = np.abs(model.get_initial_state()) + 1e-3
    P = model.jax_params()
    worst = 0.0
    for _ in range(n):
        y = y0 * np.exp(rng.normal(0, 1.0, model.n_states))
        a = np.asarray(model.jax_rhs(jnp.asarray(y), P))
        b = np.asarray(model.derivatives(0.0, y))
        worst = max(worst, float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)))
    return _line(worst < 1e-11, 'jax_rhs == derivatives',
                 f'worst rel err over {n} states: {worst:.2e}')


def check_orbit(model):
    import jax
    from engine.orbit import OrbitSolver, _scipy_period
    s = OrbitSolver(model)
    P = model.jax_params()
    y0, T, res = jax.jit(s.solve)(P, s.guess(P))
    T_sp, n_rel = _scipy_period(model)
    hi = OrbitSolver(model, n_steps=8 * s.n_steps)
    y0h, Th, _r = jax.jit(hi.solve)(P, hi.guess(P))
    dT = abs(float(T) - float(Th)) / float(Th)
    dP = abs(float(T) - T_sp) / T_sp
    mu, _ev = s.floquet(P, y0, T)
    ok = float(res) < 1e-9 and dP < 1e-6 and dT < 1e-6
    _line(ok, 'orbit BVP',
          f'|F|={float(res):.1e}  T={float(T):.6f}  vs relax({n_rel}p) {dP:.1e}  '
          f'self-conv {dT:.1e}')
    print(f"          Floquet mu = {mu:.4f} -> {s.relax_periods(P, y0, T):.0f} periods "
          f"for a transient to decay to 1e-3", flush=True)
    return ok


def check_dt(model, target=None, dose=None, nph=12):
    """Halve the RK4 step until the PTC stops moving."""
    import jax
    import jax.numpy as jnp
    from engine.ptc import make_ptc, recommended_skip
    target = target or model.perturbable_targets()[0]
    dose = _probe_dose(model, target) if dose is None else dose
    skip, _mu, _r = recommended_skip(model, tol=1e-2, verbose=False)
    ph = jnp.arange(nph) / nph
    prev, worst = None, None
    for dt in (0.04, 0.02, 0.01):
        f, s = make_ptc(model, target, dt=dt, skip_p=skip)
        cur = np.asarray(jax.jit(f)(model.jax_params(), s.guess(model.jax_params()),
                                    ph, jnp.full(nph, float(dose))))
        if prev is not None:
            worst = float(np.max(np.abs(((cur - prev + 0.5) % 1.0) - 0.5)))
        prev = cur
    return _line(worst is not None and worst < 1e-4, 'RK4 step-size convergence',
                 f'|change| from dt 0.02 -> 0.01 at dose {dose:g}: {worst:.1e}')


def _probe_dose(model, target, hi=50.0, nph=16):
    """A dose that actually resets, chosen only from doses the integrator can be trusted at.

    Three ways a scanned dose is rejected, in order: the trajectory went negative (the
    fixed-step integrator left its stability region -- Goldbeter's MP pulse does this from
    dose ~10), the oscillator was killed (Goodwin's X pulse does this at 0.5), or type-0 was
    not reached. Returns the smallest VALID dose reaching type-0, else the largest valid dose.

    Hardcoding a probe dose per model would be worse than it looks: the dose that probes
    Goodwin (0.05) is invisible on Almeida, and the one that probes Almeida (2.0) is well past
    where Goldbeter's integrator breaks.
    """
    import jax
    import jax.numpy as jnp
    from engine.ptc import make_ptc, phase_or_nan, valid_mask
    from engine.reference import winding
    f, s = make_ptc(model, target, readout='raw', track_min=True)
    P = model.jax_params(); x0 = s.guess(P); fj = jax.jit(f)
    ph = jnp.arange(nph) / nph
    best = None
    for d in np.geomspace(0.02, hi, 14):
        z, mn = fj(P, x0, ph, jnp.full(nph, float(d)))
        if not np.all(valid_mask(np.asarray(mn))):
            break                                   # integrator unstable beyond here
        p, _a = phase_or_nan(np.asarray(z))
        if not np.all(np.isfinite(p)):
            break                                   # oscillator dead beyond here
        best = float(d)
        if winding(np.asarray(ph), p) == 0:
            return float(d)
    return best if best is not None else 0.5


def check_ptc(model, name):
    from engine.reference import cross_check
    return cross_check(name)


def check_gauge(model):
    from gauge.validate import check_identity, check_algebra, check_invariance
    ok = check_identity(model, verbose=False)
    _line(ok, 'gauge identity f(Sy;th_g)=rho.S.f', 'exact to machine precision')
    a = check_algebra(model, verbose=False)
    _line(a, 'gauge projection algebra')
    i = check_invariance(model, verbose=False)
    _line(i, 'gauge invariance (period, rel-amp)')
    from gauge.gauge import Gauge
    g = Gauge(model)
    print(f"          {g.G.shape[1]} generator(s) of {len(g.names)} parameters -> "
          f"{len(g.names) - g.rank} physically meaningful combinations", flush=True)
    return ok and a and i


def check_positivity(model, target=None, nph=12):
    """Two things, and note which is being asserted.

    (a) At a dose the engine calls VALID, states really do stay non-negative -- i.e. the
        validity flag is not just decorative.
    (b) The engine flags what it should: it reports where the integrator goes unstable and
        where the oscillator dies, instead of returning a confident phase there.

    What is NOT asserted is that no dose ever produces a negative state. That is false for a
    fixed-step integrator on a stiff system, and pretending otherwise is how you end up
    reading PTC values out of numerical ringing.
    """
    import jax
    import jax.numpy as jnp
    from engine.perturb import displace, resolve_target
    from engine.ptc import make_ptc, phase_or_nan, valid_mask

    target = target or model.perturbable_targets()[0]
    ti = resolve_target(model, target)
    P = model.jax_params()
    f, s = make_ptc(model, target, readout='raw', track_min=True)
    x0 = s.guess(P); fj = jax.jit(f)
    ph = jnp.arange(nph) / nph

    d_ok = _probe_dose(model, target)
    _z, mn = fj(P, x0, ph, jnp.full(nph, float(d_ok)))
    worst = float(np.min(np.asarray(mn)))
    yd = np.asarray(displace(jnp.asarray(model.get_initial_state()), ti, -1e9))
    ok_pos = worst >= -1e-8 and yd[ti] == 0.0
    _line(ok_pos, 'valid doses stay non-negative',
          f'min state at the probe dose {d_ok:.3g}: {worst:+.2e}; knockdown clamps at '
          f'{yd[ti]:.1f}')

    # scan upward and report the two ways a point stops being trustworthy
    unstable = dead = None
    for d in np.geomspace(max(d_ok, 1e-3), 5e3, 16):
        z, m = fj(P, x0, ph, jnp.full(nph, float(d)))
        m = np.asarray(m)
        if unstable is None and not np.all(valid_mask(m)):
            unstable = (float(d), float(m.min()))
        p, a = phase_or_nan(np.asarray(z))
        if dead is None and not np.all(np.isfinite(p)):
            dead = (float(d), int(np.sum(~np.isfinite(p))), float(np.nanmin(a)))
        if unstable and dead:
            break
    _line(True, 'engine flags its own limits',
          (f'integrator unstable from dose {unstable[0]:.3g} (min {unstable[1]:+.1e})'
           if unstable else 'stable over the whole scan') + '; ' +
          (f'oscillator dead from {dead[0]:.3g}' if dead else 'clock never dies'))
    return ok_pos


def check_timing(model, target=None, nph=24, ndose=8):
    import jax
    import jax.numpy as jnp
    from engine.ptc import make_ptc, grid_points
    target = target or model.perturbable_targets()[0]
    f, s = make_ptc(model, target, readout='raw')
    P = model.jax_params(); x0 = s.guess(P)
    ph, dz = grid_points(nph, np.geomspace(0.02, 5.0, ndose))
    fj = jax.jit(f)
    t0 = time.time(); np.asarray(fj(P, x0, ph, dz)); t_c = time.time() - t0
    t0 = time.time(); np.asarray(fj(P, x0, ph, dz)); t_r = time.time() - t0
    per = t_r / (nph * ndose) * 1e3
    print(f"   [----] {'timing':34s} {nph}x{ndose} grid in {t_r:.2f}s "
          f"({per:.1f} ms/point, compile {t_c:.1f}s)", flush=True)
    return True


# --------------------------------------------------------------------------- #
def gate(name, quick=False):
    from models import get_model
    model = get_model(name)
    print(f"\n{'=' * 78}\nGATE -- {name}  ({type(model).__name__}: {model.n_states} states, "
          f"{len(model.parameter_names)} parameters)\n{'=' * 78}", flush=True)
    ok = True
    ok &= check_conformance(model)
    ok &= check_jax_vs_numpy(model)
    ok &= check_orbit(model)
    ok &= check_gauge(model)
    if not quick:
        ok &= check_dt(model)
        ok &= check_positivity(model)
        ok &= check_ptc(model, name)
        check_timing(model)
    else:
        ok &= check_positivity(model)
    print(f"\n   ==> {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def _main(argv):
    from models import MODEL_REGISTRY
    quick = '--quick' in argv
    names = [a for a in argv if not a.startswith('--')] or sorted(MODEL_REGISTRY)
    results = {n: gate(n, quick) for n in names}
    print(f"\n{'=' * 78}")
    for n, r in results.items():
        print(f"  {n:12s} {'PASS' if r else 'FAIL'}")
    ok = all(results.values())
    print(f"  [{'PASS' if ok else 'FAIL'}] gate over {len(names)} model(s)")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(_main(sys.argv[1:]))
