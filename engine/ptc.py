"""
engine/ptc.py
=============
The phase-transition curve, evaluated at arbitrary (old_phase, dose) points, built on the
SOLVED periodic orbit (engine/orbit.py) and the generic perturbations (engine/perturb.py).

Adapted from input_screen/ptc_orbit.py (commit e955873), with the ectopic/degron machinery
replaced by engine.perturb and one convention change (the phase origin, below).

WHY THE ORBIT MATTERS HERE
    The older engine obtained the cycle from a relaxation labelled by an ESTIMATED Fourier
    phase, then argsorted those labels and interpolated onto them. Three consequences: the
    old-phase axis carried the estimator's error; argsort has zero derivative, so autodiff
    differentiated a fixed ordering; and the phase origin drifted between parameter sets, so
    PTCs were not directly comparable. Here the orbit is a solved BVP anchored at a PHYSICALLY
    DEFINED point (a turning point of the reference species), returned on an exact uniform
    phase grid -- so old_phase is exact, sampling is plain periodic interpolation, and the
    origin is the same physical event for every parameter set. That last property is what
    makes the cross-parameter comparisons in analysis/coupling.py mean anything.

THE PHASE ORIGIN (a change from ptc_orbit)
    The post-perturbation asymptotic phase is read out by a windowed Fourier average -- that
    is intrinsic to what a PTC is, and it matches the CircaSCOPE estimator. But the argument
    of the fundamental differs from the tau axis by a CONSTANT: the skip-window offset plus
    the peak-vs-fundamental discrepancy of a non-sinusoidal waveform. ptc_orbit left that
    constant floating and removed it when comparing curves.

    Here it is calibrated away instead, by running the identical readout on an UNPERTURBED
    cell at tau = 0:

        new_phase = ( F(y_perturbed) - F(cycle[0]) - elapsed )  mod 1

    The consequence is that the PTC is EXACTLY the identity at dose 0 -- which is the right
    convention, makes two parameter sets' PTCs directly comparable rather than comparable up
    to an offset, and doubles as a built-in correctness check (see `selftest`).

RK4 AND GRADIENTS
    This is a fixed-step RK4 engine. It is accurate in the FORWARD direction (verified against
    an adaptive solver in engine/reference.py), and that is what the batch-1 analyses use --
    they are finite-difference throughout. Do NOT autodiff it at high dose: in input_screen
    that produced |J| up to 3.95e83 against an actual response of 0.006, and invalidated a
    whole table of results. A gradient-based cost needs an adaptive (Tsit5/diffrax) backend.
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap

from engine.orbit import OrbitSolver
from engine.flow import check_backend, make_flow, make_window_sampler
from engine.perturb import make_forced_flow, displace, resolve_target, check_mode


def _sample_uniform(cyc, phase):
    """State at `phase` on a cycle sampled uniformly at phases 0, 1/m, ..., (m-1)/m.
    Periodic linear interpolation -- no sort, no estimated phase axis."""
    m = cyc.shape[0]
    t = (phase % 1.0) * m
    i0 = jnp.floor(t).astype(jnp.int32) % m
    frac = t - jnp.floor(t)
    return (1.0 - frac) * cyc[i0] + frac * cyc[(i0 + 1) % m]


def recommended_skip(model, tol=1e-2, cap=40, floor=4, n_steps=1024, verbose=True):
    """How many periods to skip before reading the asymptotic phase, DERIVED from the model's
    leading Floquet multiplier: skip = log(tol)/log(mu).

    There is no universal constant here, which is the whole point of computing it. Measured
    leading multipliers: Almeida 0.546 (a transient is down to 1e-3 in 11 periods), Goodwin
    0.951 (137 periods). A fixed skip_p=6 leaves 2.6% of Almeida's transient in the reading
    but 74% of Goodwin's -- and nothing in the output would tell you which case you were in.

    Returns (skip_p, mu, residual_fraction). If the cap binds, `residual_fraction` is the
    transient still present, and it is reported rather than hidden.
    """
    from engine.orbit import OrbitSolver
    s = OrbitSolver(model, n_steps=n_steps)
    P = model.jax_params()
    y0, T, _r = jax.jit(s.solve)(P, s.guess(P))
    mu, _ev = s.floquet(P, y0, T)
    if not (0.0 < mu < 1.0):
        skip = cap
    else:
        skip = int(np.clip(np.ceil(np.log(tol) / np.log(mu)), floor, cap))
    resid = float(mu ** skip) if 0.0 < mu < 1.0 else 1.0
    if verbose:
        note = "  <- CAPPED: the reading is NOT asymptotic" if resid > 3 * tol else ""
        print(f"[ptc] {type(model).__name__}: Floquet mu = {mu:.4f} -> skip_p = {skip} "
              f"(transient residual {resid:.1e}){note}", flush=True)
    return skip, mu, resid


def make_ptc(model, target, mode='pulse', n_steps=1024, m_cycle=256, dt=0.02, pulse=8.0,
             settle_p=1, skip_p=None, ev_p=3, readout='phase', eps=1e-6, newton_iters=8,
             gp=None, skip_tol=1e-2, verbose=False, track_min=False, backend='rk4', grad_mode='rev'):
    # skip_p=None means "derive it from the Floquet multiplier" (see recommended_skip). Pass an
    # integer to override. ev_p=3, not 2: it must be >= 2 for exactness (below), and 3 leaves
    # margin. MEASURED on Almeida at dose 2.0 against the adaptive reference engine: skip_p=3
    # disagrees by 2.8e-3 cyc, skip_p=6 by 6.4e-4, and 10 or 16 give the same 6.3e-4 -- the
    # floor being the intrinsic difference between a Fourier-fundamental and a peak-time phase
    # estimator, not a transient.
    """Return (f, solver) where

        f(P, x0, old_phases, doses) -> new_phase[k]      (or a complex readout)

    evaluated AT the given points (both arrays the same length; use `grid_points` for a grid).

    track_min=True makes `f` return `(value, y_min)`, where `y_min` is the most negative state
    reached along each trajectory. Any point with `y_min < 0` is outside the fixed-step
    integrator's stability region and its value must be discarded -- see
    engine.perturb.make_forced_flow, and `valid_mask` below.

    readout:
      'phase'    new_phase in [0, 1)
      'complex'  the SOFT-NORMALISED Fourier unit vector z/(|z|+eps), rotated to the same
                 origin. Identical to exp(2i*pi*new_phase) where the post-perturbation
                 oscillation is well defined, but as the cell approaches a phase singularity
                 (|z| -> 0) it shrinks smoothly toward 0 instead of an atan2 blow-up -- so a
                 near-singular cell reads as neutral rather than as a NaN. This is the readout
                 the winding/singularity analysis wants.
      'raw'      the coefficient NORMALISED by the unperturbed cycle's, so |.| is the
                 post-perturbation amplitude RELATIVE to the intact clock: 1.0 on the cycle,
                 -> 0 both at a phase singularity and when the perturbation has killed the
                 oscillator outright. This is the readout the analysis layer should use, and
                 `phase_amp` splits it into (phase, relative amplitude).

    A PHASE IS ONLY MEANINGFUL IF THERE IS AN OSCILLATION. `arg` of a near-zero coefficient is
    noise, and the 'phase' readout will return it without complaint. Real case: an 8 h pulse
    of dose 0.5 into Goodwin's X drives Z to 15.3, whose n=4 Hill shuts transcription down by
    ~1e-5, and the oscillator stops permanently -- yet the phase readout reported a confident
    type-0 PTC while the adaptive reference correctly reported "no peaks, no phase". Use
    `phase_amp` + `dead_mask` so "strong resetting" and "you killed the clock" stay distinct;
    conflating them is what made a whole radialization result in input_screen meaningless (its
    cost's trivial global optimum was a dead oscillator).

    STEP COUNTS ARE STATIC, DURATIONS ARE IN TRUE PERIODS
    `lax.scan` needs static lengths, so the number of steps per period, `n_pp`, is fixed at
    build time from the nominal period. But the step SIZE is T/n_pp with T the solved period,
    so the settle and readout windows are exactly `settle_p * T` and `ev_p * T` however far T
    drifts from nominal.

    That exactness is not cosmetic. Windowing over a non-integer number of periods leaks
    higher harmonics into the fundamental and biases its phase, by an amount that depends on
    the phase being measured -- so the dose-0 PTC stops being the identity. MEASURED: with the
    window set from the nominal period, Almeida (nominal 24.83 vs true 24.826, 0.01% off) was
    fine at 9e-5, but Goodwin (24.0 vs 23.540, 2% off) was wrong by 3.6e-3. Over an exact
    integer number of periods (>= 2) the Hann-windowed fundamental is extracted EXACTLY, since
    every harmonic then lands at least two DFT bins from DC where the Hann kernel is zero.
    """
    check_mode(mode)
    if ev_p < 2:
        raise ValueError("ev_p must be >= 2 for the Hann-windowed fundamental to be exact")
    if skip_p is None:
        skip_p, _mu, _resid = recommended_skip(model, tol=skip_tol, n_steps=n_steps,
                                               verbose=verbose)
    ti = resolve_target(model, target)
    check_backend(backend)
    solver = OrbitSolver(model, n_steps=n_steps, newton_iters=newton_iters)
    flow = make_flow(model, ti, backend=backend, track_min=True, grad_mode=grad_mode)
    rhs = model.jax_rhs
    ref_idx = solver.ref_idx
    gp = float(gp or getattr(model, 'approx_period', None) or 24.0)

    n_pp = int(round(gp / dt))                  # steps per period: STATIC
    n_pulse = int(round(pulse / dt))            # the pulse is a physical 8 h, not a period
    n_settle = int(settle_p * n_pp)
    n_skip = int(skip_p * n_pp)
    n_ev = int(ev_p * n_pp)

    def _four_vec_rk4(state, P, w, hT):
        """Hann-windowed fundamental Fourier coefficient of the reference oscillation,
        accumulated inside the scan. arg -> asymptotic phase; |.| -> post-perturbation
        amplitude (-> 0 at a phase singularity). `hT` = T/n_pp is the traced step size.

        Skip and readout are FUSED into one checkpointed scan so no trajectory is ever
        materialized -- gradient memory is O(n_steps * state), not O(n_steps * n_ops), and the
        forward pass under vmap holds only the running carry."""
        def step(carry, i):
            y, c, sw = carry
            k1 = rhs(y, P); k2 = rhs(y + 0.5 * hT * k1, P)
            k3 = rhs(y + 0.5 * hT * k2, P); k4 = rhs(y + hT * k3, P)
            y2 = y + (hT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            li = i - n_skip
            inwin = i >= n_skip
            wgt = jnp.where(inwin, 0.5 - 0.5 * jnp.cos(2 * jnp.pi * li / (n_ev - 1)), 0.0)
            c = c + jnp.where(inwin, y2[ref_idx] * wgt * jnp.exp(-1j * w * (li * hT)), 0j)
            return (y2, c, sw + wgt), None
        (_y, c, sw), _ = lax.scan(jax.checkpoint(step), (state, 0j, 0.0),
                                  jnp.arange(n_skip + n_ev))
        return c / sw

    def _four_vec_dfx(state, P, w, hT):
        """The same quantity on the adaptive backend, with the IDENTICAL Hann formula and the
        identical sample grid -- so an A/B difference is attributable to the integrator alone.

        The skip is integrated as its own solve and discarded rather than being fused into the
        sampled window, so only n_ev states are materialized instead of n_skip + n_ev. For
        Almeida that is ~3.7k states per cell rather than ~19k, which is the difference between
        a vmapped gradient fit fitting in memory and not."""
        y = flow(state, n_skip, hT, P, 0.0)[0]
        ys = _sampler(y, n_ev, hT, P)
        li = jnp.arange(n_ev)
        wgt = 0.5 - 0.5 * jnp.cos(2 * jnp.pi * li / (n_ev - 1))
        c = jnp.sum(ys[:, ref_idx] * wgt * jnp.exp(-1j * w * (li * hT)))
        return c / jnp.sum(wgt)

    _sampler = (make_window_sampler(model, backend, grad_mode)
                if backend != 'rk4' else None)
    _four_vec = _four_vec_rk4 if backend == 'rk4' else _four_vec_dfx

    def f(P, x0, old_phases, doses):
        y0, T, _res = solver.solve(P, x0)
        cyc = solver.cycle(P, y0, T, m_cycle)          # exact uniform phase grid
        w = 2 * jnp.pi / T
        hT = T / n_pp                                  # step size in TRUE periods
        # settle is exactly settle_p whole periods, so it contributes nothing mod 1; only the
        # pulse's physical duration shifts the phase origin of the perturbed cell.
        settle_time = (pulse + settle_p * T) if mode == 'pulse' else (settle_p * T)
        elapsed = (settle_time / T) % 1.0
        # calibration: the same readout on an UNPERTURBED cell at tau = 0. Fixes the constant
        # skip + peak-vs-fundamental offset, so dose 0 gives exactly new_phase == old_phase.
        cref = _four_vec(cyc[0], P, w, hT)
        phi_ref = jnp.angle(cref) / (2 * jnp.pi)
        amp_ref = jnp.abs(cref)                            # the intact clock's amplitude
        rot = jnp.exp(-1j * 2 * jnp.pi * (elapsed + phi_ref))
        dt_p = pulse / n_pulse                             # physical-duration pulse stepping

        def one(oldph, dose):
            base = _sample_uniform(cyc, oldph)
            if mode == 'instant':
                y2, mn = flow(displace(base, ti, dose), n_settle, hT, P, 0.0)
            else:
                y1, m1 = flow(base, n_pulse, dt_p, P, dose)        # drive on
                y2, mn = flow(y1, n_settle, hT, P, 0.0, m1)        # drive off, relax
            cc = _four_vec(y2, P, w, hT)
            if readout == 'complex':
                out = (cc / (jnp.abs(cc) + eps)) * rot
            elif readout == 'raw':
                out = cc * rot / amp_ref           # |.| = amplitude relative to intact
            else:
                out = (jnp.angle(cc) / (2 * jnp.pi) - elapsed - phi_ref) % 1.0
            return (out, mn) if track_min else out

        return vmap(one)(old_phases, doses)

    return f, solver


#: Relative amplitude below which the post-perturbation oscillation is treated as absent, so
#: its phase is meaningless. input_screen used the same 0.01 for its type-0 classifier.
DEAD_AMP = 0.01


def phase_amp(z):
    """Split a 'raw' readout into (new_phase in [0,1), relative amplitude)."""
    z = np.asarray(z)
    return (np.angle(z) / (2 * np.pi)) % 1.0, np.abs(z)


def dead_mask(amp, thr=DEAD_AMP):
    """True where the oscillation is gone and the phase must NOT be believed."""
    return ~(np.asarray(amp) > thr)


def phase_or_nan(z, thr=DEAD_AMP):
    """new_phase from a 'raw' readout, with dead cells set to NaN.

    NaN is the honest value: those cells have no phase. analysis/winding.py decides separately
    whether an isolated hole is a numerical glitch worth imputing or a real dead region."""
    ph, amp = phase_amp(z)
    ph = ph.astype(float).copy()
    ph[dead_mask(amp, thr)] = np.nan
    return ph, amp


#: A state may dip this far below zero before a point is called invalid. Not exactly 0: the
#: cycle itself brushes zero for some models (Goldbeter's smallest phospho-form runs at ~4e-3),
#: so round-off alone can produce a -1e-12 without anything being wrong.
NEG_TOL = -1e-8


def valid_mask(y_min, tol=NEG_TOL):
    """True where the trajectory stayed physical, i.e. the integrator was stable there."""
    return np.asarray(y_min) >= tol


def grid_points(n_phase, doses):
    """(old_phases, doses) flattened over a phase x dose grid, for the `f` above.
    Reshape the result as (n_dose, n_phase) -- dose is the SLOW axis."""
    ph = jnp.tile(jnp.arange(n_phase) / n_phase, len(doses))
    dz = jnp.repeat(jnp.asarray(doses, jnp.float64), n_phase)
    return ph, dz


def ptc_grid(f, P, x0, n_phase, doses):
    """Convenience: evaluate `f` on a full phase x dose grid -> (n_phase, n_dose) array,
    matching the [phase, dose] orientation the analysis kernels expect."""
    ph, dz = grid_points(n_phase, doses)
    out = np.asarray(f(P, x0, ph, dz))
    return out.reshape(len(doses), n_phase).T


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #
def selftest(name='almeida', target=None, mode='pulse'):
    import time
    from models import get_model

    model = get_model(name)
    P = model.jax_params()
    target = target or model.perturbable_targets()[0]
    print("=" * 76)
    print(f"PTC SELF-TEST -- {type(model).__name__}, target {target}, mode {mode}")
    print("=" * 76)

    f, solver = make_ptc(model, target, mode=mode)
    x0 = solver.guess(P)
    fj = jax.jit(f)
    nph = 24
    ph = jnp.arange(nph) / nph

    # 1. THE convention check: at dose 0 the PTC must be exactly the identity.
    t0 = time.time()
    new0 = np.asarray(fj(P, x0, ph, jnp.zeros(nph)))
    t_compile = time.time() - t0
    d0 = np.abs(((new0 - np.asarray(ph) + 0.5) % 1.0) - 0.5)
    print(f"\n1. IDENTITY AT DOSE 0 (the phase-origin calibration)")
    print(f"   max |new - old| = {d0.max():.2e}   rms = {np.sqrt(np.mean(d0 ** 2)):.2e}")
    print(f"   [compile {t_compile:.1f}s]")

    # 2. a real dose must actually move the phase, and monotonically in dose at fixed old
    t0 = time.time()
    for d in (0.5, 2.0):
        new = np.asarray(fj(P, x0, ph, jnp.full(nph, d)))
        shift = np.abs(((new - np.asarray(ph) + 0.5) % 1.0) - 0.5)
        print(f"   dose {d:5.2f}: mean |shift| = {shift.mean():.4f} cyc, "
              f"max = {shift.max():.4f} cyc")
    print(f"   [{2 * nph} points in {time.time() - t0:.2f}s]")

    # 3. all three readouts must agree where the oscillation is well defined
    fc, _ = make_ptc(model, target, mode=mode, readout='complex')
    zc = np.asarray(jax.jit(fc)(P, x0, ph, jnp.full(nph, 0.5)))
    npz = np.asarray(fj(P, x0, ph, jnp.full(nph, 0.5)))
    agree = np.abs(((np.angle(zc) / (2 * np.pi) - npz + 0.5) % 1.0) - 0.5).max()
    print(f"\n2. READOUT CONSISTENCY  max|arg(complex) - phase| = {agree:.2e}")

    ok = d0.max() < 2e-3 and agree < 1e-9
    print(f"\n   [{'PASS' if ok else 'FAIL'}] PTC engine self-consistent")
    print("=" * 76)
    return ok


if __name__ == '__main__':
    import sys
    from models import MODEL_REGISTRY
    names = sys.argv[1:] or sorted(MODEL_REGISTRY)
    good = all(selftest(n) for n in names)
    raise SystemExit(0 if good else 1)
