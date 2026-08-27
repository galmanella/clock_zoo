"""
gauge.validate
==============
Proof that the gauge derived from a model's declaration is real and correct.

Adapted from input_screen/gauge/validate.py (commit e955873), generalised over the registry.

  identity    THE PROOF. The algebraic identity that DEFINES the symmetry,

                    f(S y ; theta_gauged)  ==  rho * S * f(y ; theta)

              evaluated pointwise at random states, with S = diag(species scales) and rho the
              time rescale. No integration, no transient, no phase anchor -- there is no
              numerics to hide behind, and a wrong dimension declaration fails immediately and
              names the generator that failed. Instant.

  algebra     exact linear-algebra identities on G (projection is idempotent, a pure gauge
              motion has zero physical part, dimensionless parameters never move). Instant.

  invariance  forward check: a gauge motion must leave the PERIOD and every species' RELATIVE
              amplitude unchanged, while moving absolute levels by exactly the predicted
              factors. Seconds, via the orbit solver.

WHY `identity` IS THE PRIMARY PROOF AND THE OTHERS ARE NOT
    Phase along a limit cycle is NEUTRALLY stable. A gauge motion changes parameter values,
    which changes floating-point rounding in the RHS, which leaves a tiny phase offset that
    never decays -- and a phase offset shows up as a state difference of order (slope x
    offset). So integrated comparisons of the ABSOLUTE cycle bottom out far above machine
    precision even for an exactly-exact symmetry. Compare gauge-INVARIANT observables
    (relative amplitude, period) instead, and treat `identity` as the real test.

    (This repo's orbit solver actually does better than that, because it pins the cycle with a
    gauge-COVARIANT phase condition rather than relaxing to it -- see engine/orbit.py, whose
    self-test compares the absolute cycle and gets 1e-14. But the argument above is why you
    should not rely on that in general.)

Run:  python -m gauge.validate [identity|algebra|invariance|all] [model ...]
"""
import sys

import numpy as np

import gauge  # noqa: F401  -- puts the repo root on sys.path
from gauge.gauge import Gauge, random_gauge, z_to_params, gauged_model
from models import MODEL_REGISTRY, get_model


def _hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
def check_identity(model, n_states=5, step=0.4, seed=0, verbose=True):
    """THE PROOF: f(S y; theta_gauged) == rho * S * f(y; theta), pointwise. No integration."""
    g = Gauge(model)
    states = list(model.state_names)
    y_ref = model.get_initial_state()
    rng = np.random.default_rng(seed)
    ys = [y_ref] + [y_ref * np.exp(rng.normal(0, 0.8, len(y_ref))) for _ in range(n_states - 1)]

    if verbose:
        print(f"  {type(model).__name__}: {g.G.shape[1]} generator(s) over "
              f"{len(g.names)} parameters")
        print(f"  {'generator':34s} {'max rel err':>13s}")
    worst = 0.0
    for i in range(g.G.shape[1]):
        w = np.zeros(g.G.shape[1]); w[i] = step
        scales, rho = g.species_scales(w)
        S = np.array([scales[s] for s in states])
        m1 = gauged_model(model, g, w)
        e = 0.0
        for y in ys:
            lhs = np.asarray(m1.derivatives(0.0, S * y))          # f(S y ; gauged)
            rhs = rho * S * np.asarray(model.derivatives(0.0, y))  # rho S f(y ; nominal)
            # Normalise by the VECTOR scale, not per component. A per-component ratio is
            # meaningless wherever a derivative passes through zero -- and one of the test
            # states is the on-cycle initial condition, which sits at a turning point of the
            # reference species BY CONSTRUCTION, so d(ref)/dt is ~0 there. Measured on
            # Almeida: a uniform 8.3e-13 absolute error (machine precision on quantities of
            # size 14) reads as 3.3e-9 "relative" on that one component and 1.3e-13 against
            # the vector scale. The latter is the honest number.
            denom = max(float(np.max(np.abs(rhs))), 1e-300)
            e = max(e, float(np.max(np.abs(lhs - rhs)) / denom))
        worst = max(worst, e)
        if verbose:
            print(f"  {g.gauge_names[i][:34]:34s} {e:13.2e}")
    ok = worst < 1e-10
    if verbose:
        print(f"  worst over {g.G.shape[1]} generator(s) x {n_states} states: {worst:.2e}")
        print(f"  [{'PASS' if ok else 'FAIL'}] the declared symmetry is EXACT")
    return ok


# --------------------------------------------------------------------------- #
def check_algebra(model, verbose=True):
    g = Gauge(model)
    if verbose:
        print(g.summary())
    z = g.z_nominal
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.4, g.G.shape[1])
    z2 = g.apply(z, w)
    tot, ga, ph = g.split(z, z2)
    dimensionless = [i for i, p in enumerate(g.names)
                     if not any(model.parameter_dimensions().get(p, {}).values())]
    checks = [
        ("G has full column rank", g.rank == g.G.shape[1]),
        ("canonicalize(z + Gw) == canonicalize(z)",
         np.allclose(g.canonicalize(z2), g.canonicalize(z))),
        ("canonicalize is idempotent",
         np.allclose(g.canonicalize(g.canonicalize(z2)), g.canonicalize(z2))),
        ("gauge_coords recovers w", np.allclose(g.gauge_coords(z2, z), w)),
        ("a pure gauge motion has ~0 physical part", ph < 1e-12),
        ("dimensionless parameters have exactly-zero rows in G",
         not len(dimensionless) or not np.any(g.G_full[dimensionless])),
    ]
    ok = True
    for label, passed in checks:
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)
    if verbose:
        print(f"  (pure gauge motion: total {tot:.4f}, gauge {ga:.4f}, physical {ph:.2e})")
    return ok


# --------------------------------------------------------------------------- #
def check_invariance(model, sigma=0.4, seed=3, verbose=True):
    """Forward check on the solved orbit: period and RELATIVE amplitudes are invariant, while
    absolute levels move by exactly the predicted per-species factor."""
    import jax
    from engine.orbit import OrbitSolver

    g = Gauge(model)
    w = random_gauge(g, sigma=sigma, seed=seed, include_time=True)
    scales, rho = g.species_scales(w)
    m2 = gauged_model(model, g, w)   # also corrects approx_period for the time rescale

    def cycle_of(m):
        s = OrbitSolver(m)
        P = m.jax_params()
        y0, T, res = jax.jit(s.solve)(P, s.guess(P))
        if float(res) > 1e-6:
            raise RuntimeError(f"{type(m).__name__}: orbit did not converge (|F| = "
                               f"{float(res):.2e}) -- this is a SOLVER failure, not a gauge "
                               f"violation")
        return float(T), np.asarray(s.cycle(P, y0, T, 128))

    T1, C1 = cycle_of(model)
    T2, C2 = cycle_of(m2)
    S = np.array([scales[s] for s in model.state_names])

    # a time rescale rho multiplies every rate, so the period divides by rho
    dT = abs(T2 - T1 / rho) / (T1 / rho)
    rel = lambda C: (C.max(0) - C.min(0)) / np.maximum(np.abs(C.mean(0)), 1e-12)
    d_rel = float(np.max(np.abs(rel(C2) - rel(C1))))
    d_abs = float(np.max(np.abs(C2 - C1 * S) / np.maximum(np.abs(C1 * S), 1e-12)))
    if verbose:
        print(f"  time factor rho = {rho:.4f}; species factors "
              f"{np.round(S, 4)[:6]}{'...' if len(S) > 6 else ''}")
        print(f"  period       T2 vs T1/rho          : rel err {dT:.2e}   (must be invariant)")
        print(f"  relative amplitude, every species  : max diff {d_rel:.2e}   (invariant)")
        print(f"  absolute cycle vs predicted S*cyc  : max rel err {d_abs:.2e}   (covariant)")
    ok = dT < 1e-8 and d_rel < 1e-6 and d_abs < 1e-6
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] gauge motions are invisible to invariant "
              f"observables")
    return ok


# --------------------------------------------------------------------------- #
CHECKS = {'identity': check_identity, 'algebra': check_algebra,
          'invariance': check_invariance}


def _main(argv):
    which = [a for a in argv if a in CHECKS] or list(CHECKS)
    names = [a for a in argv if a in MODEL_REGISTRY] or sorted(MODEL_REGISTRY)
    ok = True
    for n in names:
        m = get_model(n)
        for c in which:
            _hdr(f"{c.upper()} -- {type(m).__name__}")
            ok &= bool(CHECKS[c](m))
    print(f"\n[{'PASS' if ok else 'FAIL'}] gauge validation over {len(names)} model(s)")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(_main(sys.argv[1:]))
