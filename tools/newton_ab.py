"""A/B: early-exit Newton (lax.while_loop) vs the old fixed-count lax.scan.

    $PY -m tools.newton_ab [--model almeida] [--n 6]

The early exit has to be free in ACCURACY and only in accuracy: a converged Newton step is a
fixed point, so stopping at |F| < tol must give the same orbit as grinding out 20 iterations,
and a divergent start must still be rejected. This checks both on the full factor grid rather
than asserting it, because "it converged so the rest is a no-op" is exactly the kind of claim
that is true until the damping or the guess changes.
"""
import argparse
import os
import types
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model                                    # noqa: E402
from engine.orbit import OrbitSolver, make_guess_fn             # noqa: E402
from analysis.lc_sens import FACTORS                            # noqa: E402


def old_newton(sol, x0, P):
    """The previous implementation: fixed count, no test."""
    eye = jnp.eye(sol.n + 1)

    def body(x, _):
        F = sol.residual(x, P)
        J = jax.jacfwd(sol.residual)(x, P)
        return x - sol.damping * jnp.linalg.solve(J + 1e-12 * eye, F), None

    xc, _ = lax.scan(body, x0, None, length=sol.newton_iters)
    return lax.stop_gradient(xc)


def main(argv=None):
    """Compare at the level lc_sens actually consumes: find() -> (T, cycle, status).

    Comparing the raw Newton point is the WRONG test and gave two false alarms. The orbit
    residual is exactly zero at any equilibrium for any T, so the degenerate solutions form a
    continuum -- at `vr x1.6` both loops land on it (|F| = 7.6e-25 and 0.0) at different
    points, a 6e4 "relative difference" that is entirely inside the set `find` then REJECTS on
    amplitude and period band. What has to agree is the verdict and the accepted orbit."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--n', type=int, default=6, help='parameters to sample')
    a = ap.parse_args(argv)

    from engine.orbit import make_orbit_finder
    model = get_model(a.model)
    base = model.get_parameters()

    find_new, _sn = make_orbit_finder(model, m=64)
    find_old, sol_old = make_orbit_finder(model, m=64)
    # Patch the INSTANCE, and never restore it. Patching the class and restoring it in a
    # `finally` silently compares new against new: `_solve` is jax.jit'd, so `self._newton` is
    # looked up when JAX first TRACES the function -- which is on the first call, long after
    # the finally has put the original method back. That version of this test reported "all 49
    # agree, speedup 1.0x", which is exactly what comparing something to itself looks like.
    sol_old._newton = types.MethodType(
        lambda self, x0, P: old_newton(self, jnp.asarray(x0, jnp.float64), P), sol_old)

    names = [p for p in model.parameter_names if base[p] > 0][:a.n]
    cases = [('base', base)] + [(f'{p} x{f:g}', {**base, p: base[p] * f})
                                for p in names for f in FACTORS if f != 1.0]

    tn = to = 0.0
    bad = []
    for label, pd in cases:
        t = time.time(); xn, Tn, Cn, sn = find_new(pd); tn += time.time() - t
        t = time.time(); xo, To, Co, so = find_old(pd); to += time.time() - t
        if sn != so:
            bad.append((label, f'status {sn!r} vs {so!r}'))
            continue
        if sn != 'ok':
            continue                                   # both rejected, for the same reason
        dT = abs(Tn - To) / max(abs(To), 1e-30)
        sc = np.maximum(Co.max(1) - Co.min(1), 1e-9)
        dC = float(np.max(np.abs(Cn - Co) / sc[:, None]))
        if dT > 1e-8 or dC > 1e-6:
            bad.append((label, f'dT/T={dT:.2e} dcycle={dC:.2e} (in units of each '
                               f'species own amplitude)'))
    n_ok = len(cases) - len(bad)
    print(f"{len(cases)} settings on {a.model}, compared through find()")
    print(f"  new (early exit) {tn:7.1f}s      old (fixed 20)  {to:7.1f}s      "
          f"speedup {to / max(tn, 1e-9):.1f}x")
    if bad:
        print(f"  DISAGREE on {len(bad)} of {len(cases)}:")
        for label, why in bad[:12]:
            print(f"    {label:<16s} {why}")
    else:
        print(f"  all {n_ok} agree: same verdict, and where accepted the same period to 1e-8 "
              f"and the same cycle to 1e-6 of amplitude")
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
