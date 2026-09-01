"""
analysis/scrit.py
=================
OBJECTIVE (a), step 1: for every perturbable target of a model, find where the PTC changes
type, and derive the dose grid the rest of the analysis should use.

    $PY -m analysis.scrit --model almeida [--mode pulse|instant] [--targets A,B]
                             [--shard i --nshards n] [--merge]

WHAT IT PRODUCES, PER (target, mode)
    S_crit      the dose of the type-1 -> type-0 winding transition, as a geometric mean of
                the bracketing scan doses. NaN if the target never reaches type-0.
    re-entrant  whether the winding comes BACK to type-1 above S_crit. Not a curiosity: in
                Mirsky, PER1's winding went 1 -> 0 -> 1 -> 0 over dose, with real colliding
                singularities, and every feature derived in that band was intrinsically noisy.
    d_max_valid the largest dose at which the result can be believed at all, and WHY it stops:
                the integrator leaving its stability region, or the oscillator dying.
    grid        an adaptive log dose grid for that target (below).

THE ADAPTIVE GRID RULE (inherited, do not re-derive)
    Place S_crit at the ~30th log-percentile of the grid, so the type-0 side spans ~70% of the
    sampled range. The isochron twist lives ABOVE S_crit, so a grid centred on S_crit -- or
    worse, a fixed grid shared across targets -- under-samples exactly the region the
    radialization and coupling questions care about. In Mirsky a fixed 0.02-16 grid left CRY2
    only 26% type-0 and hid ~90% of its twist.

    The grid is then CLAMPED to the validity ceiling. A grid that runs past where the
    integrator is stable is worse than a short one, because the points still return numbers.

WHY DOSE SCALES ARE NOT SHARED ACROSS MODELS OR TARGETS
    They cannot be. A dose is a production rate in the target species' own units, and the
    three models differ by orders of magnitude (Goodwin dies at 0.55; Almeida is barely moved
    by 2). Within a model the absolute doses ARE comparable across targets in the same scale
    group, and that comparison carries real information -- so the grids are derived per target
    but never renormalised per target. See gauge/README.md.
"""
import argparse
import os
import sys
import time

import numpy as np

import analysis  # noqa: F401  -- repo root on sys.path
import paths
from parallel import announce, shard_of

#: Fraction of the log-dose range that should sit BELOW S_crit.
BELOW_FRAC = 0.30
#: Total decades spanned by an adaptive grid.
DECADES = 1.8


def scan_target(model, target, mode='pulse', lo=1e-3, hi=1e4, n_scan=40, n_phase=32,
                skip_p=None, dt0=0.02, dt_floor=1e-3, verbose=True):
    """Wide log dose scan -> winding vs dose, S_crit, and the validity ceiling, REFINING the
    integrator step until S_crit is trustworthy.

    WHY THE REFINEMENT IS NOT OPTIONAL
        The dose at which fixed-step RK4 goes unstable is not a property of the model -- it is
        a property of the step size, and it moves. MEASURED on Almeida's CRY target: the
        ceiling is 6.2 at dt=0.02, 48.6 at dt=0.005 and 3.0e3 at dt=0.00125, roughly 8x per 4x
        smaller step. At dt=0.02 CRY appears NOT to reset at all; at dt=0.005 it reaches
        type-0 at S_crit = 32.2, and dt=0.00125 confirms 32.2. A single default step would
        have reported "CRY does not reset" -- a clean, plausible, wrong scientific claim about
        6 of Almeida's 8 targets.

        So: halve dt while the scan is still ceiling-limited, and only accept an S_crit that
        (a) has headroom below the ceiling and (b) agrees with the previous, coarser step. The
        cost is paid only by the targets that need it; PER converges at the first step because
        its S_crit = 14.1 is already dt-stable.

    Returns a dict with the raw scan (so plotting and re-analysis are pure reads), plus
    `dt_used` and `S_converged` so the reader can see what the number rests on.
    """
    from engine.ptc import recommended_skip
    if skip_p is None:
        skip_p, mu, _resid = recommended_skip(model, tol=1e-2, verbose=False)
    else:
        mu = np.nan

    prev_S, res = np.nan, None
    dt = float(dt0)
    while True:
        res = _scan_once(model, target, mode, lo, hi, n_scan, n_phase, skip_p, dt)
        S, ceil = res['S_crit'], res['d_max_valid']
        headroom = np.isfinite(S) and np.isfinite(ceil) and ceil > 2.0 * S
        agrees = (np.isfinite(S) and np.isfinite(prev_S)
                  and abs(np.log(S / prev_S)) < 0.15)
        # non-resetting is only believable once the scan reaches the top of the range
        exhausted = (not np.isfinite(S)) and np.isfinite(ceil) and ceil >= hi * 0.99
        if (headroom and agrees) or exhausted or dt <= dt_floor:
            break
        prev_S = S
        dt /= 4.0
    res.update(dt_used=dt, S_converged=bool(np.isfinite(res['S_crit']) and
                                            np.isfinite(prev_S) and
                                            abs(np.log(res['S_crit'] / prev_S)) < 0.15),
               mu=mu, skip_p=skip_p)
    res['grid'] = adaptive_grid(res['S_crit'], res['d_max_valid'], lo)
    if verbose:
        _report(res, n_scan)
    return res


def _report(r, n_scan):
    doses, W = r['doses'], r['W']
    valid = np.asarray(r['ymin']) >= -1e-8
    live = np.asarray(r['n_dead']) == 0
    prof = ' '.join(
        f"{doses[j]:.2g}:"
        f"{'x' if not valid[j] else ('d' if not live[j] else ('?' if not np.isfinite(W[j]) else str(int(abs(W[j])))))}"
        for j in range(0, n_scan, max(1, n_scan // 18)))
    print(f"  {r['target']:8s} {r['mode']:7s} S_crit={_fmt(r['S_crit']):>9s} "
          f"conv={str(r['S_converged']):5s} re-ent={str(r['reentrant']):5s} "
          f"ceiling={_fmt(r['d_max_valid']):>9s} ({r['ceiling_reason']}) dt={r['dt_used']:g}",
          flush=True)
    print(f"           W(dose): {prof}", flush=True)


def _scan_once(model, target, mode, lo, hi, n_scan, n_phase, skip_p, dt):
    """One wide dose scan at a fixed integrator step."""
    import jax
    import jax.numpy as jnp
    from engine.ptc import make_ptc, phase_or_nan, valid_mask
    from analysis.winding import winding_curve, impute

    f, solver = make_ptc(model, target, mode=mode, readout='raw', skip_p=skip_p,
                         dt=dt, track_min=True)
    P = model.jax_params()
    x0 = solver.guess(P)
    fj = jax.jit(f)
    ph = jnp.arange(n_phase) / n_phase
    doses = np.geomspace(lo, hi, n_scan)

    W = np.full(n_scan, np.nan)
    amp = np.full(n_scan, np.nan)
    ymin = np.full(n_scan, np.nan)
    n_dead = np.zeros(n_scan, int)
    ptc = np.full((n_phase, n_scan), np.nan)

    for j, d in enumerate(doses):
        z, mn = fj(P, x0, ph, jnp.full(n_phase, float(d)))
        mn = np.asarray(mn)
        p, a = phase_or_nan(np.asarray(z))
        ymin[j] = float(mn.min())
        amp[j] = float(np.nanmin(a)) if np.any(np.isfinite(a)) else np.nan
        n_dead[j] = int(np.sum(~np.isfinite(p)))
        ptc[:, j] = p
        if np.all(valid_mask(mn)):
            W[j] = np.nan if (w := winding_curve(impute(p[:, None])[:, 0])) is None else w

    valid = valid_mask(ymin)
    live = n_dead == 0
    usable = valid & live
    # the ceiling is the first dose that fails, walking up from the bottom
    first_bad = int(np.argmin(usable)) if not np.all(usable) else n_scan
    d_max = float(doses[first_bad - 1]) if first_bad > 0 else np.nan
    reason = ('none' if first_bad >= n_scan
              else ('unstable' if not valid[first_bad] else 'dead'))

    # S_crit: the first type-0 within the usable range
    Wu = np.where(usable, W, np.nan)
    typ0 = np.where(np.isfinite(Wu) & (np.abs(Wu) < 0.5))[0]
    if len(typ0):
        j0 = typ0[0]
        S = float(np.sqrt(doses[j0] * doses[max(j0 - 1, 0)]))
        above = Wu[j0:]
        reentrant = bool(np.any(np.abs(above[np.isfinite(above)]) > 0.5))
    else:
        S, reentrant = np.nan, False

    return dict(target=target, mode=mode, doses=doses, W=W, amp=amp, ymin=ymin,
                n_dead=n_dead, ptc=ptc, S_crit=S, reentrant=reentrant, d_max_valid=d_max,
                ceiling_reason=reason, n_phase=n_phase)


def _fmt(x):
    return 'nan' if not np.isfinite(x) else f'{x:.4g}'


def adaptive_grid(S_crit, d_max, lo, n=24, decades=DECADES, below=BELOW_FRAC):
    """Log grid placing S_crit at the `below` log-percentile, clamped to the validity ceiling.

    If S_crit is unknown (the target never resets), fall back to spanning the usable range --
    a non-resetting target is still worth characterising, it just has no transition to centre
    on."""
    if not np.isfinite(d_max) or d_max <= 0:
        d_max = 1.0
    if not np.isfinite(S_crit) or S_crit <= 0:
        hi = d_max
        return np.geomspace(max(hi * 10 ** -decades, lo), hi, n)
    l0 = np.log10(S_crit) - below * decades
    l1 = np.log10(S_crit) + (1.0 - below) * decades
    l1 = min(l1, np.log10(d_max))                     # never sample past the validity ceiling
    l0 = min(l0, l1 - 0.5)                            # keep at least half a decade
    return np.geomspace(10 ** l0, 10 ** l1, n)


# --------------------------------------------------------------------------- #
def run(model_name, mode='pulse', targets=None, n_scan=40, n_phase=32, lo=1e-3, hi=1e4,
        dt0=0.02, dt_floor=1e-3, shard=None, nshards=None, tag=None):
    from models import get_model
    model = get_model(model_name)
    all_t = list(targets or model.perturbable_targets())
    mine, idx, n = shard_of(all_t, shard, nshards)
    tag = paths.run_tag(tag)
    announce(analysis='scrit', model=model_name, mode=mode, targets=len(all_t),
             shard=f'{idx}/{n}', doing=len(mine), tag=tag)

    t0 = time.time()
    rows = [scan_target(model, t, mode=mode, lo=lo, hi=hi, n_scan=n_scan, n_phase=n_phase,
                        dt0=dt0, dt_floor=dt_floor) for t in mine]
    print(f"[scrit] {len(rows)} target(s) in {time.time() - t0:.0f}s", flush=True)

    blob = dict(model=model_name, mode=mode, targets=np.array([r['target'] for r in rows]),
                S_crit=np.array([r['S_crit'] for r in rows]),
                reentrant=np.array([r['reentrant'] for r in rows]),
                d_max_valid=np.array([r['d_max_valid'] for r in rows]),
                ceiling_reason=np.array([r['ceiling_reason'] for r in rows]),
                mu=np.array([r['mu'] for r in rows]),
                skip_p=np.array([r['skip_p'] for r in rows]),
                dt_used=np.array([r['dt_used'] for r in rows]),
                S_converged=np.array([r['S_converged'] for r in rows]))
    for r in rows:                              # raw scan per target: pure-read downstream
        for k in ('doses', 'W', 'amp', 'ymin', 'n_dead', 'ptc', 'grid'):
            blob[f'{k}__{r["target"]}'] = np.asarray(r[k])
    out = paths.out_path(model_name, 'scrit', paths.shard_filename(f'scrit_{mode}', idx
                                                                   if n > 1 else None), tag)
    paths.savez(out, **blob)
    print(f"[scrit] -> {out}", flush=True)
    return rows


def merge(model_name, mode='pulse', tag=None):
    """Combine shard npzs into one table and print it."""
    tag = tag or paths.latest_run(model_name, 'scrit')
    shards = paths.load_shards(model_name, 'scrit', tag, prefix=f'scrit_{mode}')
    if not shards:
        print(f"[scrit] nothing to merge for {model_name}/{mode} tag={tag}", file=sys.stderr)
        return None
    blob = {}
    for s in shards:
        for k, v in s.items():
            if '__' in k:
                blob[k] = v
    cat = lambda k: np.concatenate([np.atleast_1d(s[k]) for s in shards])
    for k in ('targets', 'S_crit', 'reentrant', 'd_max_valid', 'ceiling_reason', 'mu',
              'skip_p', 'dt_used', 'S_converged'):
        blob[k] = cat(k)
    blob['model'] = model_name
    blob['mode'] = mode
    out = paths.out_path(model_name, 'scrit', f'scrit_{mode}_merged.npz', tag)
    paths.savez(out, **blob)
    print_table(blob)
    print(f"[scrit] merged {len(shards)} shard(s) -> {out}", flush=True)
    return blob


def print_table(blob):
    print(f"\n{'=' * 78}")
    print(f"S_crit table -- {blob['model']} ({blob['mode']} mode)")
    print(f"{'=' * 78}")
    print(f"  {'target':10s} {'S_crit':>10s} {'conv':>5s} {'re-ent':>7s} {'valid to':>10s} "
          f"{'why':>9s} {'dt':>8s} {'grid':>21s}")
    order = np.argsort([s if np.isfinite(s) else np.inf for s in blob['S_crit']])
    for i in order:
        t = str(blob['targets'][i])
        g = blob.get(f'grid__{t}')
        gs = f"{g.min():.3g} .. {g.max():.3g}" if g is not None else '-'
        print(f"  {t:10s} {_fmt(blob['S_crit'][i]):>10s} "
              f"{str(bool(blob['S_converged'][i])):>5s} "
              f"{str(bool(blob['reentrant'][i])):>7s} "
              f"{_fmt(blob['d_max_valid'][i]):>10s} {str(blob['ceiling_reason'][i]):>9s} "
              f"{blob['dt_used'][i]:>8g} {gs:>21s}")
    n0 = int(np.sum(np.isfinite(blob['S_crit'])))
    print(f"\n  {n0}/{len(blob['targets'])} targets reach type-0 within their valid range")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--mode', default=os.environ.get('MODE', 'pulse'),
                    choices=('pulse', 'instant'))
    ap.add_argument('--targets', default=os.environ.get('TARGETS'))
    ap.add_argument('--n-scan', type=int, default=40)
    ap.add_argument('--n-phase', type=int, default=32)
    ap.add_argument('--lo', type=float, default=1e-3)
    ap.add_argument('--hi', type=float, default=1e4)
    ap.add_argument('--dt0', type=float, default=0.02)
    ap.add_argument('--dt-floor', type=float, default=1e-3)
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--nshards', type=int, default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args(argv)
    if a.merge:
        return 0 if merge(a.model, a.mode, a.tag) is not None else 1
    run(a.model, mode=a.mode, targets=(a.targets.split(',') if a.targets else None),
        n_scan=a.n_scan, n_phase=a.n_phase, lo=a.lo, hi=a.hi, dt0=a.dt0,
        dt_floor=a.dt_floor, shard=a.shard, nshards=a.nshards, tag=a.tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
