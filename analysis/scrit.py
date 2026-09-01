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
                skip_p=None, dt0=0.02, dt_floor=1e-3, chunk=None, verbose=True):
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
        t0 = time.time()
        res = _scan_once(model, target, mode, lo, hi, n_scan, n_phase, skip_p, dt,
                         chunk=chunk or CHUNK)
        S, ceil = res['S_crit'], res['d_max_valid']
        if verbose:
            # A dt round is minutes on the larger models, and the loop can run three of them.
            # Silence for that long is indistinguishable from a hang, so say what each round
            # cost and what it found BEFORE deciding whether to refine.
            print(f"    [scan] {target:9s} dt={dt:<8g} S_crit={_fmt(S):>9s} "
                  f"ceiling={_fmt(ceil):>9s} ({res['ceiling_reason']}) "
                  f"{time.time() - t0:.0f}s", flush=True)
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
    live = np.asarray(r['n_dead']) < DEAD_FRAC * float(np.asarray(r['ptc']).shape[0])
    prof = ' '.join(
        f"{doses[j]:.2g}:"
        f"{'x' if not valid[j] else ('d' if not live[j] else ('?' if not np.isfinite(W[j]) else str(int(abs(W[j])))))}"
        for j in range(0, n_scan, max(1, n_scan // 18)))
    print(f"  {r['target']:8s} {r['mode']:7s} S_crit={_fmt(r['S_crit']):>9s} "
          f"conv={str(r['S_converged']):5s} re-ent={str(r['reentrant']):5s} "
          f"ceiling={_fmt(r['d_max_valid']):>9s} ({r['ceiling_reason']}) dt={r['dt_used']:g}",
          flush=True)
    print(f"           W(dose): {prof}", flush=True)


#: Points per vmapped call. Large enough to fill the machine (see `_eval_points`), small
#: enough that the batch and its checkpointed scan carry stay in cache/RAM on 16 states.
CHUNK = 2048


def _eval_points(fj, P, x0, n_phase, doses, chunk=CHUNK):
    """Evaluate the PTC on the full (dose x phase) grid, in chunks.

    Returns (Z, MN), both (n_dose, n_phase): the raw complex readout and the per-trajectory
    minimum state. Chunking is over the FLATTENED point list, so a chunk boundary can fall
    inside a dose row; the reshape at the end puts it back.
    """
    import numpy as _np
    from engine.ptc import grid_points
    ph, dz = grid_points(n_phase, _np.asarray(doses, float))
    n = int(ph.shape[0])
    zs, ms = [], []
    for a in range(0, n, int(chunk)):
        b = min(a + int(chunk), n)
        z, mn = fj(P, x0, ph[a:b], dz[a:b])
        zs.append(_np.asarray(z))
        ms.append(_np.asarray(mn))
    return (_np.concatenate(zs).reshape(len(doses), n_phase),
            _np.concatenate(ms).reshape(len(doses), n_phase))


def _scan_once(model, target, mode, lo, hi, n_scan, n_phase, skip_p, dt, chunk=CHUNK):
    """One wide dose scan at a fixed integrator step."""
    import jax
    from engine.ptc import make_ptc, phase_or_nan, valid_mask
    from analysis.winding import winding_curve, impute

    f, solver = make_ptc(model, target, mode=mode, readout='raw', skip_p=skip_p,
                         dt=dt, track_min=True)
    P = model.jax_params()
    x0 = solver.guess(P)
    fj = jax.jit(f)
    doses = np.geomspace(lo, hi, n_scan)

    # ONE vmapped call over the whole (phase x dose) scan instead of one call per dose. The
    # arithmetic is identical -- vmap is elementwise over the points, and these ARE the same
    # points -- but a 32-point batch does not fill the machine: MEASURED on Korencic at
    # dt=0.005, 32 points cost 36 ms/point and 2048 cost 10.5, so the per-dose loop was
    # spending ~3.5x of the scan on dispatch. Chunked, because the batch is materialised.
    Z, MN = _eval_points(fj, P, x0, n_phase, doses, chunk)
    p_all, a_all = phase_or_nan(Z)                       # (n_dose, n_phase)
    ptc = p_all.T.copy()                                 # (n_phase, n_dose), house orientation
    ymin = np.min(MN, axis=1)
    with np.errstate(invalid='ignore'):
        amp = np.where(np.any(np.isfinite(a_all), axis=1),
                       np.nanmin(np.where(np.isfinite(a_all), a_all, np.inf), axis=1), np.nan)
    amp = np.where(np.isfinite(amp), amp, np.nan)
    n_dead = np.sum(~np.isfinite(p_all), axis=1).astype(int)
    W = np.full(n_scan, np.nan)
    for j in range(n_scan):
        if np.all(valid_mask(MN[j])):
            w = winding_curve(impute(ptc[:, j][:, None])[:, 0])
            W[j] = np.nan if w is None else w

    S, reentrant, d_max, reason = derive_scalars(doses, W, ymin, n_dead, n_phase)
    return dict(target=target, mode=mode, doses=doses, W=W, amp=amp, ymin=ymin,
                n_dead=n_dead, ptc=ptc, S_crit=S, reentrant=reentrant, d_max_valid=d_max,
                ceiling_reason=reason, n_phase=n_phase)


#: Fraction of a dose row's phases that must lose the oscillation before the row is called
#: DEAD. Not `n_dead > 0`, and the difference changed an answer.
#:
#: A phase singularity IS a point of zero amplitude, so the dose row that contains one has a
#: cell with no readable phase BY CONSTRUCTION. MEASURED on Goldbeter / MP / instant: exactly
#: 1 of 32 phases dead, at exactly the dose where the winding flips 1 -> 0, and 0 dead at
#: every dose above it. The strict rule read that single cell as the clock stopping, capped
#: the target's validity at 242 -- and then let S_crit be taken from dose 554, ABOVE the cap
#: the same function had just computed. A genuinely dying clock looks nothing like it: the
#: same model's BC and BN lose ALL 32 phases for two consecutive doses.
DEAD_FRAC = 0.25


def derive_scalars(doses, W, ymin, n_dead, n_phase):
    """(S_crit, reentrant, d_max_valid, ceiling_reason) from a saved scan.

    Split out of `_scan_once` so `--rederive` can re-run it on stored raw arrays: a threshold
    is a definition, and re-integrating 81 minutes of scan to change one is what REPO_MAP
    hazard 9 says the raw arrays exist to prevent.
    """
    doses = np.asarray(doses, float)
    n_scan = len(doses)
    valid = np.asarray(ymin, float) >= -1e-8
    live = np.asarray(n_dead, float) < DEAD_FRAC * float(n_phase)
    usable = valid & live
    # the ceiling is the first dose that fails, walking up from the bottom
    first_bad = int(np.argmin(usable)) if not np.all(usable) else n_scan
    d_max = float(doses[first_bad - 1]) if first_bad > 0 else np.nan
    reason = ('none' if first_bad >= n_scan
              else ('unstable' if not valid[first_bad] else 'dead'))

    # S_crit: the first type-0 STRICTLY BELOW the ceiling. Searching the whole usable mask
    # instead let a transition be read from a dose above `d_max_valid` -- "the largest dose at
    # which the result can be believed at all" -- which is a contradiction in terms, and it
    # happened once in 54 scans.
    Wu = np.where(usable, np.asarray(W, float), np.nan)
    typ0 = np.where(np.isfinite(Wu) & (np.abs(Wu) < 0.5))[0]
    typ0 = typ0[typ0 < first_bad]
    if len(typ0):
        j0 = int(typ0[0])
        S = float(np.sqrt(doses[j0] * doses[max(j0 - 1, 0)]))
        above = Wu[j0:first_bad]
        reentrant = bool(np.any(np.abs(above[np.isfinite(above)]) > 0.5))
    else:
        S, reentrant = np.nan, False
    return S, reentrant, d_max, reason


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
        dt0=0.02, dt_floor=1e-3, chunk=None, shard=None, nshards=None, tag=None):
    from models import get_model
    model = get_model(model_name)
    all_t = list(targets or model.perturbable_targets())
    mine, idx, n = shard_of(all_t, shard, nshards)
    tag = paths.run_tag(tag)
    announce(analysis='scrit', model=model_name, mode=mode, targets=len(all_t),
             shard=f'{idx}/{n}', doing=len(mine), tag=tag)

    t0 = time.time()
    rows = [scan_target(model, t, mode=mode, lo=lo, hi=hi, n_scan=n_scan, n_phase=n_phase,
                        dt0=dt0, dt_floor=dt_floor, chunk=chunk) for t in mine]
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


def rederive(model_name, mode='pulse', tag=None, out_tag=None, verbose=True):
    """Recompute S_crit / ceiling / grid from a SAVED scan, without integrating anything.

    The scan arrays -- doses, W, ymin, n_dead, ptc -- are the measurement; S_crit, the ceiling
    and the grid are DEFINITIONS applied to them. When a definition changes (a threshold, a
    tie-break) the honest move is to re-apply it to the stored measurement and diff, not to
    re-run the integration and hope nothing else moved. This is what REPO_MAP hazard 9 buys.

    Writes to a NEW tag. The raw scan is never overwritten.
    """
    from analysis.dosegrid import _scrit_files
    tag, files = _scrit_files(model_name, mode, tag)
    if not files:
        print(f"[rederive] no scrit_{mode} scan for {model_name}", file=sys.stderr)
        return None
    out_tag = out_tag or f'{tag}_rd'
    changed = []
    for fp in files:
        z = dict(np.load(fp, allow_pickle=True))
        ts = [str(x) for x in z['targets']]
        S = np.array(z['S_crit'], float)
        C = np.array(z['d_max_valid'], float)
        R = np.array(z['reentrant'])
        Q = np.array(z['ceiling_reason'], dtype=object)
        for i, t in enumerate(ts):
            ptc = np.asarray(z[f'ptc__{t}'])
            s, re_, dm, why = derive_scalars(z[f'doses__{t}'], z[f'W__{t}'], z[f'ymin__{t}'],
                                             z[f'n_dead__{t}'], ptc.shape[0])
            old = (S[i], C[i], str(Q[i]))
            if not (_same(s, S[i]) and _same(dm, C[i]) and why == str(Q[i])):
                changed.append((model_name, mode, t, old, (s, dm, why)))
            S[i], C[i], R[i], Q[i] = s, dm, bool(re_), why
            z[f'grid__{t}'] = adaptive_grid(s, dm, float(np.min(z[f'doses__{t}'])))
        z.update(S_crit=S, d_max_valid=C, reentrant=R,
                 ceiling_reason=np.array([str(x) for x in Q]))
        out = paths.out_path(model_name, 'scrit', os.path.basename(fp), out_tag)
        paths.savez(out, **z)
        if verbose:
            print(f"[rederive] {os.path.basename(fp)} -> {out}", flush=True)
    if verbose:
        if changed:
            print(f"[rederive] {len(changed)} target(s) CHANGED:")
            for m, md, t, o, n in changed:
                print(f"    {m}/{md}/{t}: S_crit {_fmt(o[0])} -> {_fmt(n[0])}   "
                      f"ceiling {_fmt(o[1])} -> {_fmt(n[1])} ({o[2]} -> {n[2]})")
        else:
            print("[rederive] nothing changed -- the new definition agrees with the stored one")
    return changed


def _same(a, b):
    return (not np.isfinite(a) and not np.isfinite(b)) or (
        np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-12 * max(1.0, abs(b)))


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
    ap.add_argument('--chunk', type=int, default=None,
                    help=f'points per vmapped call (default {CHUNK}); lower it if RAM is tight')
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--nshards', type=int, default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--rederive', action='store_true',
                    help='recompute S_crit/ceiling/grid from a SAVED scan, no integration')
    ap.add_argument('--out-tag', default=None, help='tag --rederive writes to')
    a = ap.parse_args(argv)
    if a.rederive:
        return 0 if rederive(a.model, a.mode, a.tag, a.out_tag) is not None else 1
    if a.merge:
        return 0 if merge(a.model, a.mode, a.tag) is not None else 1
    run(a.model, mode=a.mode, targets=(a.targets.split(',') if a.targets else None),
        n_scan=a.n_scan, n_phase=a.n_phase, lo=a.lo, hi=a.hi, dt0=a.dt0,
        dt_floor=a.dt_floor, chunk=a.chunk, shard=a.shard, nshards=a.nshards, tag=a.tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
