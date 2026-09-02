"""
analysis/relax.py
=================
HOW LONG DOES THE CLOCK TAKE TO COME BACK, AS A FUNCTION OF DOSE -- and is the PTC's readout
window past that point or inside it?

    $PY -m analysis.relax --model goldbeter --mode pulse [--targets MP,MB] [--max-p 48]

WHY THIS EXISTS
    `engine.ptc.recommended_skip` sets the transient allowance from the leading FLOQUET
    MULTIPLIER: skip = log(tol)/log(mu). That is a LINEARISATION about the limit cycle, and it
    is correct for a small perturbation. A large dose starts far outside the linear regime,
    and the nonlinear transient can be far longer than the linear rate predicts.

    MEASURED, Goldbeter / MP / pulse at dose 46.4, at the cell with the lowest amplitude
    (old phase 0.4219): mu = 0.334 gives skip_p = 5, but the trajectory needs about 40 PERIODS
    (~950 h) to return to base amplitude. Inside the window the engine actually reads --
    periods 5 to 8 -- that cell's peak-to-trough is 12.8x the base cycle's, its mean is 24x
    elevated, and its dominant period is 71.5 h against a base of 23.85 h. Its Fourier
    fundamental AT THE BASE FREQUENCY is therefore ~0, which the pipeline reports as
    `min_amp = 0.004` and which reads, wrongly, as "the oscillator died". It had not: the same
    perturbation re-integrated with scipy LSODA oscillates at 109% of base amplitude.

    So a near-zero amplitude at high dose usually means NOT YET ASYMPTOTIC, and the phase read
    there is not the asymptotic phase either. That is a dose-range problem, not a solver
    problem, and this module measures where the range should stop.

WHAT IT MEASURES
    For each (dose, old phase): perturb, then integrate free and record the readout's
    peak-to-trough over each successive period. `relax_p` is the first period after which the
    relative amplitude stays within `TOL` of 1 for the rest of the horizon -- the number of
    periods the transient actually lasts. Reported per dose as the WORST cell, because the PTC
    surface is only as asymptotic as its slowest point.

WHAT TO DO WITH IT
    Two things, and the second is the cheaper:
      * raise `skip_p` -- but the cost is linear in it and the worst doses need 8x;
      * CROP THE DOSE RANGE at the dose where `relax_p` first exceeds the skip. Above it the
        surface is measuring a transient, so the honest grid stops there.
    `suggest_ceiling()` returns that dose, and it belongs in the dose-grid derivation next to
    the integrator's own stability ceiling -- they are two different reasons to stop, and the
    grid should respect whichever binds first.
"""
import argparse

import numpy as np

import analysis  # noqa: F401  -- repo root on sys.path
import paths
from analysis.features import ZOO

#: Relative-amplitude band around 1.0 that counts as "back on the cycle".
TOL = 0.10
#: How many periods to watch before giving up and reporting >= max_p.
MAX_P = 48
#: Old phases sampled per dose. The worst cell sets the answer, so this only has to be dense
#: enough to land near it; 16 puts a sample within 1/32 of a cycle of any phase.
N_PHASE = 16


def make_amp_trace(model, target, mode='pulse', dt=0.005, pulse=8.0, max_p=MAX_P,
                   n_pp=None, readout=None):
    """Return `f(P, y0_states, doses) -> amp[n_point, max_p]`, the readout's peak-to-trough in
    each successive period after the perturbation, relative to nothing (raw units).

    One jitted, vmapped kernel: an inner scan over the steps of one period tracking min/max,
    an outer scan over periods. Nothing is materialised, so the memory is O(state)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from engine.perturb import resolve_target, check_mode

    check_mode(mode)
    ti = resolve_target(model, target)
    rhs = model.jax_rhs
    n = int(model.n_states)
    e = jnp.zeros(n).at[ti].set(1.0)
    ri = int(model.var_index(readout or getattr(model, 'readout_variable', None)
                             or model.reference_variable))
    gp = float(getattr(model, 'approx_period', None) or 24.0)
    n_pp = int(n_pp or round(gp / dt))
    n_pulse = int(round(pulse / dt))

    def rk4(y, P, drive):
        f = lambda s: rhs(s, P) + drive * e
        k1 = f(y); k2 = f(y + 0.5 * dt * k1); k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def one_period(y, P):
        def body(c, _):
            yy, lo, hi = c
            y2 = rk4(yy, P, 0.0)
            v = y2[ri]
            return (y2, jnp.minimum(lo, v), jnp.maximum(hi, v)), None
        (yf, lo, hi), _ = lax.scan(jax.checkpoint(body), (y, jnp.inf, -jnp.inf), None,
                                   length=n_pp)
        return yf, hi - lo

    def one_point(P, y0, dose):
        if mode == 'instant':
            y = y0.at[ti].set(jnp.maximum(y0[ti] + dose, 0.0))
        else:
            y, _ = lax.scan(jax.checkpoint(lambda yy, _: (rk4(yy, P, dose), None)),
                            y0, None, length=n_pulse)
        _yf, amps = lax.scan(lambda yy, _: one_period(yy, P), y, None, length=int(max_p))
        return amps

    return jax.jit(jax.vmap(one_point, in_axes=(None, 0, 0)))


def relax_periods(amps, base_amp, tol=TOL):
    """First period after which the relative amplitude STAYS within `tol` of 1.

    'Stays' matters: a transient can cross 1 on its way past. Returns max_p when it never
    settles inside the horizon, which is a lower bound and is reported as such."""
    a = np.asarray(amps, float) / base_amp
    ok = np.abs(a - 1.0) <= tol
    n = len(a)
    out = n
    for k in range(n - 1, -1, -1):
        if ok[k]:
            out = k
        else:
            break
    return int(out) + 1                       # periods elapsed, 1-based


def run(model_name, targets=None, mode='pulse', dt=0.005, pulse=8.0, max_p=MAX_P,
        n_phase=N_PHASE, doses=None, scrit_tag=None, tag=None, verbose=True):
    import jax
    from models import get_model
    from engine.orbit import OrbitSolver
    from engine.ptc import recommended_skip
    from analysis import genemap as GM
    from analysis.dosegrid import _scrit_files

    model = get_model(model_name)
    targets = targets or GM.scope_targets(model_name) or list(model.perturbable_targets())
    s = OrbitSolver(model)   # n_steps from the model's own declaration
    P = model.jax_params()
    y0, T, _r = jax.jit(s.solve)(P, s.guess(P))
    T = float(T)
    skip, mu, _res = recommended_skip(model, tol=1e-2, verbose=False)
    cyc = np.asarray(s.cycle(P, y0, T, m=n_phase))
    full = np.asarray(s.cycle(P, y0, T, m=512))
    ri = int(model.var_index(getattr(model, 'readout_variable', None) or s.ref_name))
    base_amp = float(full[:, ri].max() - full[:, ri].min())

    if doses is None:
        tg, files = _scrit_files(model_name, mode, scrit_tag)
        z = np.load(files[0], allow_pickle=True)
        doses = np.asarray(z[f'doses__{targets[0]}'], float)
    doses = np.asarray(doses, float)

    blob = dict(model=model_name, mode=mode, targets=np.array(targets), doses=doses,
                period=T, mu=float(mu), skip_p=int(skip), dt=dt, max_p=int(max_p),
                base_amp=base_amp, tol=TOL, n_phase=int(n_phase))
    if verbose:
        print(f"\n{'=' * 92}\nRELAXATION vs DOSE -- {model_name} ({mode}), T={T:.3f} h, "
              f"mu={mu:.3f} -> skip_p={skip}\n{'=' * 92}")
    for t in targets:
        f = make_amp_trace(model, t, mode=mode, dt=dt, pulse=pulse, max_p=max_p)
        ph_idx = np.arange(n_phase)
        Y0 = np.repeat(cyc[ph_idx][None, :, :], len(doses), axis=0).reshape(-1, model.n_states)
        DZ = np.repeat(doses, n_phase)
        amps = np.asarray(f(P, Y0, DZ)).reshape(len(doses), n_phase, int(max_p))
        rp = np.array([[relax_periods(amps[j, i], base_amp) for i in range(n_phase)]
                       for j in range(len(doses))])
        worst = rp.max(axis=1)
        blob[f'amps__{t}'] = amps
        blob[f'relax_p__{t}'] = rp
        blob[f'relax_worst__{t}'] = worst
        if verbose:
            ceil = suggest_ceiling(doses, worst, skip)
            bad = int(np.sum(worst > skip))
            print(f"  {t:9s} worst-cell relaxation exceeds skip_p at {bad}/{len(doses)} doses;"
                  f"  last dose with relax <= skip_p: "
                  f"{'--' if not np.isfinite(ceil) else f'{ceil:.4g}'}", flush=True)
            prof = ' '.join(f"{doses[j]:.2g}:{worst[j]}"
                            for j in range(0, len(doses), max(1, len(doses) // 12)))
            print(f"            relax(dose): {prof}", flush=True)
    out = paths.out_path(ZOO, 'relax', f'relax_{model_name}_{mode}.npz', paths.run_tag(tag))
    paths.savez(out, **blob)
    print(f"[relax] -> {out}", flush=True)
    return blob


def suggest_ceiling(doses, relax_worst, skip_p):
    """Highest dose whose worst cell has settled by `skip_p` periods, walking up from the
    bottom. Above it the PTC is reading a transient, so the grid should stop."""
    d = np.asarray(doses, float)
    r = np.asarray(relax_worst, float)
    ok = r <= skip_p
    if not ok.any():
        return np.nan
    first_bad = int(np.argmin(ok)) if not ok.all() else len(d)
    return float(d[first_bad - 1]) if first_bad > 0 else np.nan


def load(model_name, mode='pulse', tag=None):
    import os
    tag = tag or paths.latest_run(ZOO, 'relax')
    fp = None if tag is None else paths.out_path(ZOO, 'relax',
                                                 f'relax_{model_name}_{mode}.npz', tag)
    if fp is None or not os.path.exists(fp):
        return None
    return dict(np.load(fp, allow_pickle=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description='relaxation time vs dose, and the dose ceiling '
                                             'it implies')
    ap.add_argument('--model', default='goldbeter')
    ap.add_argument('--mode', default='pulse', choices=('pulse', 'instant'))
    ap.add_argument('--targets', default=None)
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--max-p', type=int, default=MAX_P)
    ap.add_argument('--n-phase', type=int, default=N_PHASE)
    ap.add_argument('--scrit-tag', default=None)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    run(a.model, targets=(a.targets.split(',') if a.targets else None), mode=a.mode,
        dt=a.dt, max_p=a.max_p, n_phase=a.n_phase, scrit_tag=a.scrit_tag, tag=a.tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
#  The schedule the PTC engine consumes
# --------------------------------------------------------------------------- #
#: Distinct skip values a schedule may use. Each one costs a separate XLA compile, so the set
#: is deliberately small; rounding UP to the next bucket is also the safety margin on a
#: relaxation time measured with a 10% amplitude tolerance.
SKIP_BUCKETS = (8, 16, 32, 48, 64)


def skip_for_doses(model_name, target, mode, doses, cap=48, floor=None, tag=None,
                   buckets=SKIP_BUCKETS):
    """Per-dose transient skip, measured rather than assumed.

    `recommended_skip` returns ONE number for the whole surface, from the Floquet multiplier,
    and that is a linearisation about the cycle -- fine at low dose and badly short at high
    dose (see this module's header). But the requirement is not uniform either: relaxation
    grows with dose, so paying the worst case on every row multiplies the whole surface's cost
    by the price of its most stubborn corner.

    So: interpolate the MEASURED relaxation onto the requested doses, round up to a bucket,
    and never go below the Floquet floor. `lax.scan` needs a static length, so the skip cannot
    vary per cell -- but it can vary per DOSE ROW, and since relaxation is monotone in dose the
    rows group into a handful of contiguous blocks.

    Returns (skip[n_dose], info). `info['capped']` counts rows whose measured relaxation
    EXCEEDS the cap: those are rows whose phase is not asymptotic and the caller must either
    raise the cap or drop them, not quietly keep them.
    """
    z = load(model_name, mode, tag)
    d = np.asarray(doses, float)
    if z is None:
        f = int(floor or max(buckets[0], 8))
        return np.full(len(d), f, int), dict(measured=False, capped=0, floor=f)
    dz = np.asarray(z['doses'], float)
    w = np.asarray(z[f'relax_worst__{target}'], float)
    floor = int(floor if floor is not None else z['skip_p'])
    need = np.interp(np.log(d), np.log(dz), w, left=w[0], right=w[-1])
    bs = np.array(sorted(set(int(b) for b in buckets) | {floor}), int)
    bs = bs[bs <= int(cap)] if np.any(bs <= int(cap)) else bs[:1]
    out = np.array([bs[np.searchsorted(bs, max(n, floor))] if max(n, floor) <= bs[-1]
                    else bs[-1] for n in need], int)
    return out, dict(measured=True, capped=int(np.sum(need > int(cap))),
                     floor=floor, need=need,
                     blocks=int(len(np.where(np.diff(out) != 0)[0]) + 1))


def identity_lo(model_name, target, mode, thr=0.02, tag=None):
    """Lowest scanned dose whose PTC departs from the identity by more than `thr` cycles.

    The bottom of a wide dose scan is all identity -- MEASURED, 20-45% of the screen's rows
    for the BMAL targets -- and those rows cost exactly as much as informative ones while
    telling you only that a small perturbation does little. This is where a grid should start.
    """
    from analysis.dosegrid import _scrit_files
    tag2, files = _scrit_files(model_name, mode, tag)
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        if target not in [str(x) for x in z['targets']]:
            continue
        d = np.asarray(z[f'doses__{target}'], float)
        p = np.asarray(z[f'ptc__{target}'], float)
        old = np.arange(p.shape[0]) / p.shape[0]
        for k in range(len(d)):
            col = p[:, k]
            if not np.any(np.isfinite(col)):
                continue
            dev = np.nanmax(np.abs(((col - old + 0.5) % 1.0) - 0.5))
            if dev > thr:
                return float(d[k])
    return np.nan


def relax_ceiling(model_name, target, mode, skip_cap=48, tag=None):
    """Highest dose whose worst cell relaxes within `skip_cap` periods. Above it the readout
    is inside the transient however long the skip, so the grid should stop."""
    z = load(model_name, mode, tag)
    if z is None:
        return np.nan
    return suggest_ceiling(np.asarray(z['doses'], float),
                           np.asarray(z[f'relax_worst__{target}'], float), int(skip_cap))
