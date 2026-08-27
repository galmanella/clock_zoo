"""
engine/perturb.py
=================
The two generic perturbation modes, model-agnostically.

    pulse    an additive PRODUCTION RATE on one state, held for `pulse` hours then removed:
                 dy/dt = f(y; P) + dose * e_target   for t in [0, pulse),
                 dy/dt = f(y; P)                     thereafter.
             `dose` has units of [target]/time. This is the notebook's
             AdditiveRatePerturbation, so PRCs computed here are comparable with the ones
             already in circadian_modeling.ipynb.

    instant  a one-shot state displacement, then free relaxation:
                 y[target] <- max(y[target] + dose, 0)
             `dose` has units of [target]. Zero duration, so it isolates "where the state was
             moved to" from "how long it was driven".

WHY NOT THE ECTOPIC + DEGRON SCHEME
    input_screen modelled the actual CircaSCOPE experiment: ectopic protein produced from an
    inducible ORF and terminated by a degron, with the ectopic share tracked separately
    through every complex it formed so the degron stayed selective. That is faithful, but it
    needs per-model complex bookkeeping (`coupling_arrays` read Mirsky's private topology),
    and it does not even map onto a model like Korencic that has no protein species. The two
    modes above are defined for ANY state of ANY model, which is what makes the whole
    downstream stack model-agnostic. The ectopic mode can return later as a third option for
    the models where it means something.

COMPARING THE TWO MODES
    A pulse dose is a rate and an instant dose is a concentration, so they are NOT on a common
    axis. Matching delivered AMOUNT needs the target's own clearance timescale
    (`tau_eff = (1 - exp(-k*pulse))/k` for first-order clearance k), which is model- and
    target-specific -- and in Mirsky the amount-matched instant kick still turned out 3-34x
    weaker than the pulse, because a sustained drive keeps pushing while an instant jump
    decays. Do that matching in the analysis layer, deliberately; do not assume the axes are
    comparable here.

NON-NEGATIVITY
    `instant` clamps the displaced state at 0 so a negative dose (knockdown -- supported, and
    free) cannot drive a concentration below zero. The models clamp again inside their RHS.
"""
import jax
import jax.numpy as jnp
import numpy as np

from models.api import require

MODES = ('pulse', 'instant')


def resolve_target(model, target):
    """Validate a perturbation target and return its state index.

    Checks against `perturbable_targets()` rather than just `state_names`, so a model can
    declare that some state is not something an experiment can push on."""
    require(model, 'perturbation', consumer='engine.perturb')
    allowed = list(model.perturbable_targets())
    if target not in allowed:
        raise KeyError(f"{type(model).__name__} cannot be perturbed at {target!r}; "
                       f"perturbable_targets() = {allowed}")
    return int(model.var_index(target))


def check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return mode


# --------------------------------------------------------------------------- #
#  JAX: forced RK4 flow
# --------------------------------------------------------------------------- #
def make_forced_flow(model, target_idx):
    """Return `flow(y0, n_steps, dt, P, drive)` -- fixed-step RK4 on

        dy/dt = f(y; P) + drive * e_target

    `drive` is a traced scalar (so the same compiled kernel serves the pulse segment with
    drive=dose and the settle segment with drive=0), while `n_steps` is static, as lax.scan
    requires. The step body is checkpointed so reverse-mode recomputes each RK4 step instead
    of storing its internals -- O(n_steps * state) gradient memory rather than
    O(n_steps * n_ops). It is a no-op in the forward pass.
    """
    rhs = model.jax_rhs
    n = int(model.n_states)
    e = jnp.zeros(n).at[target_idx].set(1.0)

    def f(y, P, drive):
        return rhs(y, P) + drive * e

    def flow(y0, n_steps, dt, P, drive):
        def step(y, _):
            k1 = f(y, P, drive)
            k2 = f(y + 0.5 * dt * k1, P, drive)
            k3 = f(y + 0.5 * dt * k2, P, drive)
            k4 = f(y + dt * k3, P, drive)
            return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None
        yf, _ = jax.lax.scan(jax.checkpoint(step), y0, None, length=n_steps)
        return yf

    return flow


def displace(y, target_idx, dose):
    """The `instant` perturbation, JAX. Clamped at 0 so a negative dose cannot go unphysical."""
    return y.at[target_idx].set(jnp.maximum(y[target_idx] + dose, 0.0))


# --------------------------------------------------------------------------- #
#  Numpy / scipy: the same two definitions, for the adaptive reference path
# --------------------------------------------------------------------------- #
def displace_np(y, target_idx, dose):
    y = np.array(y, float, copy=True)
    y[target_idx] = max(y[target_idx] + dose, 0.0)
    return y


def forced_rhs_np(model, target_idx, dose, pulse, params=None):
    """scipy-style rhs(t, y) for the `pulse` mode. The drive is a step in t, so integrate the
    [0, pulse] and [pulse, end] segments SEPARATELY (see engine/reference.py) -- handing an
    adaptive solver a discontinuous RHS makes it either step over the edge or crawl."""
    p = params if params is not None else model.get_parameters()

    def rhs(t, y):
        d = model.derivatives(t, y, p)
        if 0.0 <= t < pulse:
            d = np.array(d, float, copy=True)
            d[target_idx] += dose
        return d

    return rhs


def selftest(name='almeida'):
    """The forced flow must reduce to the unforced one at drive=0, and a pulse must deliver
    exactly `dose * pulse` extra material into the target in the small-dose limit."""
    from models import get_model
    from engine.orbit import OrbitSolver

    model = get_model(name)
    P = model.jax_params()
    tgt = model.perturbable_targets()[0]
    ti = resolve_target(model, tgt)
    flow = make_forced_flow(model, ti)
    solver = OrbitSolver(model)
    y0, T, _r = jax.jit(solver.solve)(P, solver.guess(P))
    dt = 0.02
    n = int(round(4.0 / dt))

    print(f"PERTURB SELF-TEST -- {type(model).__name__}, target {tgt}")
    # 1. drive = 0 must reproduce the unforced flow exactly.
    # NB `_flow(rhs, y0, T_arg, P, n)` integrates dy/dtau = T_arg*f(y) over tau in [0,1] with
    # step 1/n, i.e. it advances T_arg of PHYSICAL time in n steps. So the matching span is
    # T_arg = n*dt, not n*dt/T.
    a = np.asarray(flow(y0, n, dt, P, 0.0))
    from engine.orbit import _flow
    b = np.asarray(_flow(model.jax_rhs, y0, n * dt, P, n))
    rel = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    print(f"   drive=0 vs the unforced orbit flow: max rel diff = {rel:.2e}")

    # 2. linear-response: a small pulse adds dose*pulse to the target, to first order
    pulse, small = 8.0, 1e-6
    npl = int(round(pulse / dt))
    y_un = np.asarray(flow(y0, npl, dt, P, 0.0))
    y_pt = np.asarray(flow(y0, npl, dt, P, small))
    got = (y_pt[ti] - y_un[ti]) / small
    print(f"   d(target)/d(dose) after an {pulse:g} h pulse = {got:.4f} "
          f"(<= {pulse:g} = dose*duration, less the clearance that acted during it)")

    # 3. instant displacement clamps
    yd = np.asarray(displace(y0, ti, -1e9))
    print(f"   instant dose -1e9 clamps to {yd[ti]:.3f} (must be 0.000)")
    ok = rel < 1e-9 and 0.0 < got <= pulse * 1.001 and yd[ti] == 0.0
    print(f"   [{'PASS' if ok else 'FAIL'}] perturbation definitions are consistent")
    return ok


if __name__ == '__main__':
    import sys
    from models import MODEL_REGISTRY
    good = all(selftest(n) for n in (sys.argv[1:] or sorted(MODEL_REGISTRY)))
    raise SystemExit(0 if good else 1)
