"""
engine/reference.py
===================
The INDEPENDENT ground truth for the PTC: adaptive scipy `solve_ivp` + peak-based asymptotic
phase. Descended from input_screen/screen_core.py's PhaseReference / compute_ptc (commit
e955873), which was already model-agnostic.

WHY A SECOND ENGINE EXISTS
    engine/ptc.py is fixed-step RK4 with a Fourier phase readout. This module is an adaptive
    (LSODA) solver with a peak-matching phase readout. They share NO numerical machinery, so
    when they agree, the agreement means something. That is not paranoia: input_screen shipped
    a whole table of results computed by autodiffing a fixed-step RK4 PTC, and the numbers
    were wrong by ~80 orders of magnitude at high dose. The finite-displacement results that
    used the adaptive engine were the only ones that survived. Every claim in this repo about
    a PTC feature gets confirmed here before it is cited.

    It is deliberately slower and simpler. Do not optimise it; its value is independence.

PHASE CONVENTION (matched to engine/ptc.py)
    phase 0 = a MAXIMUM of the reference species. `asymptotic_phase(y)` returns the cycle
    phase of the isochron y sits on, by integrating the unforced system and comparing peak
    times against the reference. So an unperturbed cycle point at phase p returns p, and
    `ptc()` below is exactly the identity at dose 0 -- the same convention engine/ptc.py
    achieves by calibrating its Fourier origin.

RELAXATION LENGTH IS NOT A CONSTANT
    `build()` keeps relaxing until the period estimate stops moving rather than trusting a
    fixed transient. A fixed 40 periods gives Goodwin a period of 23.6727 against a true
    23.5398 -- its cycle attracts slowly and needs ~800. Almeida is converged by ~100. Getting
    this wrong looks exactly like a solver bug in whatever you compare against.
"""
import numpy as np
from scipy.integrate import solve_ivp

from engine.perturb import resolve_target, check_mode, displace_np


#: Provisional per-model probe doses for the cross-check only. analysis/scrit.py replaces
#: these with derived, per-target adaptive grids.
_DEFAULT_DOSES = {'almeida': (0.0, 0.5, 2.0), 'goodwin': (0.0, 0.02, 0.1, 0.5)}


def _peaks(t, y):
    """Quadratically interpolated local maxima of y(t) on a dense grid."""
    out = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            a = 0.5 * (y[i - 1] - 2 * y[i] + y[i + 1])
            b = 0.5 * (y[i + 1] - y[i - 1])
            off = (-b / (2 * a)) if a < 0 else 0.0
            out.append(t[i] + off * (t[i + 1] - t[i]))
    return np.array(out)


class AdaptiveReference:
    """Period, phase-indexed cycle table, and asymptotic phase -- all on an adaptive solver."""

    def __init__(self, model, params=None, n_cycle=720, rtol=1e-10, atol=1e-12,
                 num_periods=None, eval_periods=3, dense_per_period=4000):
        """`num_periods` is how long to let a perturbed state relax before its phase is read.

        Default None = DERIVE it from the model's leading Floquet multiplier. There is no safe
        constant: with Goodwin's mu = 0.951 a hardcoded 6 leaves 74% of the transient in the
        reading, and the resulting PTC disagreed with the JAX engine by 0.33 cyc at
        intermediate dose -- while looking perfectly plausible on its own. Deriving by default
        means a weakly-attracting model costs more time instead of silently returning biased
        numbers."""
        self.model = model
        self.params = dict(params or model.get_parameters())
        self.n_cycle = int(n_cycle)
        self.rtol, self.atol = rtol, atol
        if num_periods is None:
            from engine.ptc import recommended_skip
            num_periods, _mu, _res = recommended_skip(model, tol=1e-3, cap=200, verbose=False)
        self.num_periods, self.eval_periods = int(num_periods), int(eval_periods)
        self.dpp = int(dense_per_period)
        self.ref_idx = model.var_index(model.reference_variable)
        self.period = None
        self.cycle_states = None
        self.n_relax = None

    # -- integration -------------------------------------------------------- #
    def _sim(self, y0, t_end):
        return solve_ivp(lambda t, y: self.model.derivatives(t, y, self.params),
                         [0.0, t_end], np.asarray(y0, float), method='LSODA',
                         dense_output=True, rtol=self.rtol, atol=self.atol)

    def _period_from(self, sol, t_end, span):
        t = np.linspace(max(t_end - span, 0.0), t_end, int(self.dpp * span /
                                                           float(self.model.approx_period)))
        pk = _peaks(t, sol.sol(t)[self.ref_idx])
        return (float(np.mean(np.diff(pk))) if len(pk) > 2 else np.nan), pk

    def build(self, tol=1e-8, max_periods=1600):
        """Relax to the cycle (adaptively long), then tabulate exactly one period."""
        gp = float(self.model.approx_period)
        y0 = self.model.get_initial_state()
        n, prev = 50, None
        while True:
            sol = self._sim(y0, n * gp)
            T, _pk = self._period_from(sol, n * gp, 6 * gp)
            if prev is not None and np.isfinite(T) and abs(T - prev) / T < tol:
                break
            prev = T
            n *= 2
            if n > max_periods:
                break
        if not np.isfinite(T):
            raise RuntimeError(f"{type(self.model).__name__}: no clean peaks after "
                               f"{n} periods -- is it oscillating?")
        self.period, self.n_relax = T, n

        # advance to the next reference MAXIMUM: that is phase 0. (The turning-point section
        # has two roots per cycle; anchoring on the wrong one puts the cycle half a period out
        # of register -- the same trap engine/orbit.make_guess_fn documents.)
        t = np.linspace(n * gp - 3 * T, n * gp, int(3 * self.dpp))
        pk = _peaks(t, sol.sol(t)[self.ref_idx])
        y_phase0 = sol.sol(pk[-1])

        one = self._sim(y_phase0, T)
        tc = np.linspace(0.0, T, self.n_cycle, endpoint=False)
        self.cycle_states = one.sol(tc).T.copy()               # (n_cycle, n_states)
        self._state_phase0 = y_phase0
        self._ref_peak = self._first_eval_peak(y_phase0)
        if not np.isfinite(self._ref_peak):
            raise RuntimeError("could not locate the reference peak in the eval window")
        return self

    # -- phase -------------------------------------------------------------- #
    def _first_eval_peak(self, state):
        T = self.period
        t_end = (self.num_periods + self.eval_periods) * T
        sol = self._sim(state, t_end)
        t = np.linspace(self.num_periods * T, t_end, int(self.eval_periods * self.dpp))
        pk = _peaks(t, sol.sol(t)[self.ref_idx])
        return pk[0] if len(pk) else np.nan

    def asymptotic_phase(self, state):
        """Cycle phase of the isochron `state` lies on. Returns p for a cycle point at p."""
        tp = self._first_eval_peak(state)
        if not np.isfinite(tp):
            return np.nan
        return ((self._ref_peak - tp) / self.period) % 1.0

    def state_at_phase(self, phase):
        i = int(round((phase % 1.0) * self.n_cycle)) % self.n_cycle
        return self.cycle_states[i].copy()


def ptc(model, ref, target, dose, n_phases=24, mode='pulse', pulse=8.0):
    """PTC at ONE dose on the adaptive engine. Returns (old_phase, new_phase).

    The pulse is integrated as TWO segments so the drive's discontinuity lands exactly on a
    segment boundary. Handing an adaptive solver a step discontinuity inside one call makes it
    either step straight over the edge (wrong) or crawl (slow); neither is acceptable in the
    thing you are using as ground truth.
    """
    check_mode(mode)
    ti = resolve_target(model, target)
    T = ref.period
    settle = 1 * T
    elapsed = ((pulse + settle) / T) % 1.0 if mode == 'pulse' else (settle / T) % 1.0
    old = np.linspace(0, 1, n_phases, endpoint=False)
    new = np.full(n_phases, np.nan)
    p = ref.params

    def drive_rhs(t, y):
        d = np.array(model.derivatives(t, y, p), float, copy=True)
        d[ti] += dose
        return d

    for i, ph in enumerate(old):
        y = ref.state_at_phase(ph)
        if mode == 'instant':
            y = displace_np(y, ti, dose)
        else:
            s1 = solve_ivp(drive_rhs, [0.0, pulse], y, method='LSODA',
                           rtol=ref.rtol, atol=ref.atol)
            y = s1.y[:, -1]
        s2 = solve_ivp(lambda t, yy: model.derivatives(t, yy, p), [0.0, settle], y,
                       method='LSODA', rtol=ref.rtol, atol=ref.atol)
        phi = ref.asymptotic_phase(s2.y[:, -1])
        new[i] = (phi - elapsed) % 1.0 if np.isfinite(phi) else np.nan
    return old, new


# --------------------------------------------------------------------------- #
#  Cross-check against engine/ptc.py -- the reason this module exists
# --------------------------------------------------------------------------- #
def cross_check(name='almeida', target=None, mode='pulse', doses=None, n_phases=16):
    import time
    import jax
    import jax.numpy as jnp
    from models import get_model
    from engine.ptc import make_ptc, phase_or_nan, recommended_skip

    model = get_model(name)
    target = target or model.perturbable_targets()[0]
    # dose scale is model-specific: Goodwin's whole production rate is v1=0.7, so a dose of 0.5
    # is not a probe, it is an execution. Set per model until analysis/scrit.py derives it.
    doses = doses if doses is not None else _DEFAULT_DOSES.get(name, (0.0, 0.5, 2.0))
    print("=" * 76)
    print(f"CROSS-CHECK  jax/RK4/Fourier  vs  scipy/LSODA/peaks")
    print(f"   {type(model).__name__}, target {target}, mode {mode}")
    print("=" * 76)

    # BOTH engines must allow the SAME transient decay, derived from the model's leading
    # Floquet multiplier. Otherwise they are measuring different quantities and the comparison
    # is meaningless: with Goodwin's mu = 0.951, the JAX engine skipping 40 periods and the
    # reference skipping its default 6 disagreed by 0.33 cyc at intermediate dose -- not
    # because either was buggy, but because 6 periods leaves 74% of the transient and 40
    # leaves 13%. Neither number is the asymptotic phase.
    skip, mu, resid = recommended_skip(model, tol=1e-3, cap=200, verbose=True)

    t0 = time.time()
    ref = AdaptiveReference(model, num_periods=skip).build()
    print(f"   adaptive reference: T = {ref.period:.6f} h after {ref.n_relax} relax periods, "
          f"skip {skip} [{time.time() - t0:.1f}s]")

    P = model.jax_params()
    # 'raw' so the JAX side reports amplitude too: a dead oscillator has no phase, and the two
    # engines must agree about THAT as well as about the phases of the live cells.
    f, solver = make_ptc(model, target, mode=mode, readout='raw', skip_p=skip)
    x0 = solver.guess(P)
    fj = jax.jit(f)
    y0, T_bvp, _r = jax.jit(solver.solve)(P, x0)
    print(f"   BVP orbit         : T = {float(T_bvp):.6f} h   "
          f"|dT|/T = {abs(float(T_bvp) - ref.period) / ref.period:.2e}")

    ph = jnp.arange(n_phases) / n_phases
    worst, agree_dead = 0.0, True
    print(f"\n   {'dose':>7s} {'W jax':>6s} {'W ref':>6s} {'dead j/r':>9s} "
          f"{'rms':>10s} {'max':>10s}")
    for d in doses:
        new_j, amp_j = phase_or_nan(np.asarray(fj(P, x0, ph, jnp.full(n_phases, float(d)))))
        old_r, new_r = ptc(model, ref, target, float(d), n_phases=n_phases, mode=mode)
        dj, dr = ~np.isfinite(new_j), ~np.isfinite(new_r)
        agree_dead &= bool(np.array_equal(dj, dr))
        # NO offset fitting: both engines are calibrated to the identity at dose 0, so any
        # constant discrepancy is a real disagreement and must show up here.
        live = ~(dj | dr)
        r = np.abs(((new_r[live] - new_j[live] + 0.5) % 1.0) - 0.5) if live.any() else \
            np.array([np.nan])
        wj, wr = winding(np.asarray(ph), new_j), winding(old_r, new_r)
        if live.any():
            worst = max(worst, float(np.nanmax(r)))
        fmt = lambda v: 'dead' if v is None else f'{v:4d}'
        print(f"   {d:7.2f} {fmt(wj):>6s} {fmt(wr):>6s} "
              f"{int(dj.sum()):4d}/{int(dr.sum()):<4d} "
              f"{np.sqrt(np.nanmean(r ** 2)):10.2e} {np.nanmax(r):10.2e}")

    ok = worst < 1.5e-3 and agree_dead
    print(f"\n   worst |d new_phase| over live cells: {worst:.2e}")
    print(f"   the engines agree on which cells are dead: {agree_dead}")
    print(f"   [{'PASS' if ok else 'FAIL'}] the two engines agree")
    print("=" * 76)
    return ok


def winding(old, new):
    """Signed winding number of a PTC curve (how many cycles `new` advances over one cycle of
    `old`). |W| = 1 is type-1 (weak) resetting, |W| = 0 is type-0 (strong).

    Returns None if any point is NaN. A winding number is a property of a CLOSED loop, so a
    curve with a hole in it does not have one -- returning a plausible integer computed from
    the surviving points would silently invent a topology. (analysis/winding.py imputes
    isolated holes first, deliberately and with a gap limit, then asks for the winding.)"""
    new = np.asarray(new, float)
    if not np.all(np.isfinite(new)):
        return None
    z = np.exp(2j * np.pi * new)
    d = np.angle(z[1:] * np.conj(z[:-1])).sum() + np.angle(z[0] * np.conj(z[-1]))
    return int(np.round(d / (2 * np.pi)))


if __name__ == '__main__':
    import sys
    from models import MODEL_REGISTRY
    names = sys.argv[1:] or sorted(MODEL_REGISTRY)
    good = all(cross_check(n) for n in names)
    raise SystemExit(0 if good else 1)
