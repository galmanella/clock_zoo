"""
analysis/deadcheck.py
=====================
WHEN A PERTURBED CELL READS AS AMPLITUDE ~0, WHICH OF THE THREE THINGS IS IT?

    $PY -m analysis.deadcheck --model goldbeter --mode pulse [--targets MP,MB]

The base parameters give a healthy attracting limit cycle in all three models -- residual
~1e-14, Floquet mu 0.33-0.55 -- and the perturbation is transient. So a post-perturbation
amplitude of 0.01 needs an explanation, and there are exactly three candidates:

  1. THE CLOCK REALLY STOPPED. A large enough kick can leave the limit cycle's basin. The
     model then has somewhere else to go -- typically a stable fixed point -- and it goes
     there and stays. Transient forcing, permanent consequence: nothing about "the base
     parameters oscillate" forbids this, because it is a statement about a DIFFERENT initial
     condition, not about the parameters.
  2. THE INTEGRATOR BROKE. Fixed-step RK4 has a stability limit that scales with the largest
     eigenvalue the trajectory visits, and a big pulse visits much stiffer states than the
     cycle does. Past that limit the solution rings and can go negative -- which is how it is
     caught, since every state is a concentration.
  3. THE CELL IS ON THE PHASE SINGULARITY. At (S*, phi*) the perturbation lands the state on
     the phaseless set and the returning oscillation is genuinely near-zero. This is a real
     feature, it is ONE cell wide, and it is what the whole analysis is looking for.

THE TEST
    Re-integrate the same perturbation three ways and compare the amplitude that comes back:

        rk4        the production path, at the dt the scan settled on
        rk4/4      the same, four times finer
        adaptive   scipy LSODA at rtol 1e-10 -- engine/reference.py's engine, which shares no
                   numerical machinery with the RK4 path

    Case 1 gives the same near-zero amplitude on all three. Case 2 gives amplitude back as
    soon as the step shrinks, and the coarse run is the one with a negative state. The
    verdict column says which, and it says UNCLEAR rather than guessing when the three
    disagree without a pattern.

WHY THIS IS WORTH A MODULE
    Because the answer changes what the high-dose half of a surface MEANS. If Goldbeter's
    mRNA targets genuinely stop, its type-0 region is a statement about the clock and belongs
    in the comparison with a caveat. If they only stop in RK4, the region is an artefact and
    the dose grid has to be capped instead. REPO_MAP hazard 15 is the record of a solver limit
    and a divergence rendering identically and being read as the latter.
"""
import argparse

import numpy as np

import analysis  # noqa: F401  -- repo root on sys.path
import paths
from analysis.features import ZOO

#: How long to watch after the perturbation, in periods, before measuring the amplitude.
SETTLE_P = 12
#: Over how many periods the surviving amplitude is measured.
MEASURE_P = 3
#: Relative amplitude below which a trajectory counts as NOT oscillating.
DEAD = 0.02


def _rel_amp(t, y, T, ref_idx, base_amp):
    """Peak-to-trough of the readout over the LAST `MEASURE_P` periods, relative to base."""
    m = t >= (t[-1] - MEASURE_P * T)
    if m.sum() < 8:
        return np.nan
    r = np.asarray(y[ref_idx, m], float)
    if not np.all(np.isfinite(r)):
        return np.nan
    return float((r.max() - r.min()) / base_amp) if base_amp > 0 else np.nan


def _rk4_run(model, ti, y0, dose, mode, pulse, dt, span, ref_idx):
    """Fixed-step RK4, the production path. Returns (t, y[ref], min state seen)."""
    import jax
    import jax.numpy as jnp
    from engine.perturb import make_forced_flow, displace
    # JIT with n_steps STATIC. Without this every call re-traces the lax.scan, and the
    # sampling loop below makes thousands of them -- the first version of this function did
    # not return in fifteen minutes for a calculation that takes seconds.
    flow = jax.jit(make_forced_flow(model, ti, track_min=True), static_argnums=(1,))
    P = model.jax_params()
    y = jnp.asarray(y0, jnp.float64)
    mn = jnp.min(y)
    if mode == 'instant':
        y = displace(y, ti, float(dose))
        mn = jnp.minimum(mn, jnp.min(y))
    else:
        y, mn = flow(y, int(round(pulse / dt)), dt, P, float(dose), mn)
    # sample the free relaxation so an amplitude can be measured
    n_tot = int(round(span / dt))
    n_out = 1200
    step = max(1, n_tot // n_out)
    ts, ys, cur = [], [], y
    for k in range(0, n_tot, step):
        cur, mn = flow(cur, step, dt, P, 0.0, mn)
        ts.append((k + step) * dt)
        ys.append(np.asarray(cur))
    return np.array(ts), np.array(ys).T, float(mn)


def _adaptive_run(model, ti, y0, dose, mode, pulse, span):
    """scipy LSODA -- no numerical machinery shared with the RK4 path."""
    from scipy.integrate import solve_ivp
    from engine.perturb import displace_np
    p = model.get_parameters()
    y = np.array(y0, float)
    if mode == 'instant':
        y = displace_np(y, ti, float(dose))
    else:
        def drive(t, yy):
            d = np.array(model.derivatives(t, yy, p), float, copy=True)
            d[ti] += float(dose)
            return d
        s1 = solve_ivp(drive, [0.0, pulse], y, method='LSODA', rtol=1e-10, atol=1e-12)
        y = s1.y[:, -1]
    s2 = solve_ivp(lambda t, yy: model.derivatives(t, yy, p), [0.0, span], y,
                   method='LSODA', rtol=1e-10, atol=1e-12,
                   t_eval=np.linspace(0, span, 4000))
    return s2.t, s2.y, float(np.min(s2.y))


def check(model_name, target, mode='pulse', dose=None, phases=(0.0, 0.25, 0.5, 0.75),
          dt=None, pulse=8.0, verbose=True):
    """One (target, dose): the three integrations, their amplitudes, and a verdict."""
    import jax
    from models import get_model
    from engine.orbit import OrbitSolver
    from engine.perturb import resolve_target

    model = get_model(model_name)
    s = OrbitSolver(model)   # n_steps from the model's own declaration
    P = model.jax_params()
    y0, T, _r = jax.jit(s.solve)(P, s.guess(P))
    T = float(T)
    cyc = np.asarray(s.cycle(P, y0, T, m=512))
    ref_idx = int(model.var_index(getattr(model, 'readout_variable', None) or s.ref_name))
    base_amp = float(cyc[:, ref_idx].max() - cyc[:, ref_idx].min())
    ti = resolve_target(model, target)
    dt = float(dt or 0.005)
    span = SETTLE_P * T

    rows = []
    for ph in phases:
        yph = np.asarray(s.cycle(P, y0, T, m=512))[int(round(ph * 512)) % 512]
        t1, y1, m1 = _rk4_run(model, ti, yph, dose, mode, pulse, dt, span, ref_idx)
        t2, y2, m2 = _rk4_run(model, ti, yph, dose, mode, pulse, dt / 4, span, ref_idx)
        t3, y3, m3 = _adaptive_run(model, ti, yph, dose, mode, pulse, span)
        a1 = _rel_amp(t1, y1, T, ref_idx, base_amp)
        a2 = _rel_amp(t2, y2, T, ref_idx, base_amp)
        a3 = _rel_amp(t3, y3, T, ref_idx, base_amp)
        rows.append(dict(phase=float(ph), rk4=a1, rk4_fine=a2, adaptive=a3,
                         min_rk4=m1, min_fine=m2, min_adaptive=m3,
                         verdict=_verdict(a1, a2, a3, m1)))
    if verbose:
        report(model_name, target, mode, dose, dt, T, rows)
    return rows


def _verdict(a1, a2, a3, min_rk4):
    """The adaptive run is the arbiter; the two RK4 runs say whether the step was the problem."""
    if not np.isfinite(a3):
        return 'NO ADAPTIVE'
    dead_ad = a3 < DEAD
    dead_rk = np.isfinite(a1) and a1 < DEAD
    if dead_ad and dead_rk:
        return 'CLOCK STOPPED'            # every engine agrees the oscillation is gone
    if dead_rk and not dead_ad:
        return 'SOLVER' + (' (neg state)' if min_rk4 < -1e-8 else '')
    if not dead_rk and dead_ad:
        return 'UNCLEAR (rk4 alive, adaptive dead)'
    return 'ALIVE'


def report(model_name, target, mode, dose, dt, T, rows):
    print(f"\n{'=' * 96}\n{model_name} / {target} / {mode}  dose={dose:g}   "
          f"dt={dt:g} vs {dt / 4:g} vs adaptive   watched {SETTLE_P} periods "
          f"({SETTLE_P * T:.0f} h)\n{'=' * 96}")
    print(f"  {'old phase':>9s} {'rk4':>9s} {'rk4/4':>9s} {'adaptive':>9s} "
          f"{'min state (rk4)':>16s}  verdict")
    for r in rows:
        f = lambda x: '   nan' if not np.isfinite(x) else f'{x:6.4f}'
        print(f"  {r['phase']:9.3f} {f(r['rk4']):>9s} {f(r['rk4_fine']):>9s} "
              f"{f(r['adaptive']):>9s} {r['min_rk4']:16.3e}  {r['verdict']}")
    v = [r['verdict'] for r in rows]
    print(f"\n  relative amplitude of the readout over the last {MEASURE_P} periods; "
          f"< {DEAD} counts as not oscillating.")
    print(f"  verdicts: " + ', '.join(f'{x}={v.count(x)}' for x in sorted(set(v))))


def worst_phase(model_name, target, mode, dose, dt=0.005, n_phase=64, pulse=8.0):
    """The OLD PHASE at which the readout amplitude is lowest at this dose, and that amplitude.

    `min_amp` in the feature table is a minimum OVER PHASES -- `scrit` reduces each dose row to
    its worst cell before saving -- so testing a few round-number phases tests the wrong cells.
    A phase singularity is one cell wide by definition, and it is the cell that sets the
    number, so it has to be located before it can be re-integrated."""
    import jax
    import jax.numpy as jnp
    from models import get_model
    from engine.ptc import make_ptc, phase_amp
    model = get_model(model_name)
    f, solver = make_ptc(model, target, mode=mode, readout='raw', dt=dt, pulse=pulse,
                         track_min=True, verbose=False)
    P = model.jax_params()
    x0 = solver.guess(P)
    ph = jnp.arange(n_phase) / n_phase
    z, _mn = jax.jit(f)(P, x0, ph, jnp.full(n_phase, float(dose)))
    _p, a = phase_amp(np.asarray(z))
    k = int(np.argmin(np.where(np.isfinite(a), a, np.inf)))
    return float(k) / n_phase, float(a[k])


def worst_dose(model_name, target, mode, tag=None):
    """The scan dose at which the relative amplitude was lowest -- where to look."""
    from analysis.dosegrid import _scrit_files
    tag, files = _scrit_files(model_name, mode, tag)
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        ts = [str(x) for x in z['targets']]
        if target in ts:
            d = np.asarray(z[f'doses__{target}'], float)
            a = np.asarray(z[f'amp__{target}'], float)
            ym = np.asarray(z[f'ymin__{target}'], float)
            ok = np.isfinite(a) & (ym >= -1e-8)          # only where the scan was believable
            if not ok.any():
                return None, None
            k = int(np.arange(len(d))[ok][np.argmin(a[ok])])
            return float(d[k]), float(a[k])
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser(description='is a zero-amplitude cell the clock or the solver?')
    ap.add_argument('--model', default='goldbeter')
    ap.add_argument('--mode', default='pulse', choices=('pulse', 'instant'))
    ap.add_argument('--targets', default=None, help='default: the model in-scope targets')
    ap.add_argument('--dose', type=float, default=None,
                    help='default: the scan dose with the LOWEST relative amplitude')
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--phases', default='0,0.25,0.5,0.75')
    ap.add_argument('--scrit-tag', default=None)
    ap.add_argument('--auto-phase', action='store_true', default=True,
                    help='locate the lowest-amplitude CELL first (default)')
    ap.add_argument('--fixed-phases', dest='auto_phase', action='store_false')
    a = ap.parse_args(argv)
    from analysis import genemap as GM
    targets = (a.targets.split(',') if a.targets else GM.scope_targets(a.model))
    phases = tuple(float(x) for x in a.phases.split(','))
    out = {}
    for t in targets:
        dose = a.dose
        amp = None
        if dose is None:
            dose, amp = worst_dose(a.model, t, a.mode, a.scrit_tag)
        if dose is None:
            print(f"[deadcheck] no scan dose for {a.model}/{t}/{a.mode}; skipping")
            continue
        print(f"\n[deadcheck] {a.model}/{t}/{a.mode}: scan's worst relative amplitude was "
              f"{amp if amp is not None else float('nan'):.4g} at dose {dose:.4g}", flush=True)
        ph = phases
        if a.auto_phase:
            wph, wamp = worst_phase(a.model, t, a.mode, dose, a.dt)
            print(f"[deadcheck]   its worst CELL is old phase {wph:.4f} "
                  f"(amplitude {wamp:.4g}); testing that one and its neighbours", flush=True)
            ph = tuple(sorted({round((wph + d) % 1.0, 6)
                               for d in (-2 / 64, 0.0, 2 / 64)} | {0.0}))
        out[t] = check(a.model, t, a.mode, dose, ph, a.dt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
