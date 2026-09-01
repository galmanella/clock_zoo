"""
analysis/dosegrid.py
====================
ONE dose grid per (model, mode), shared by every target of that model -- the grid a
cross-gene comparison needs, as opposed to the per-target adaptive grid `analysis.scrit`
derives for the per-gene question.

    $PY -m analysis.dosegrid --model almeida --mode pulse [--decades 3.0] [--n 48]
    $PY -m analysis.dosegrid --model almeida --mode pulse --lo 0.1 --hi 300   # pin by hand

WHY A SECOND KIND OF GRID
    `scrit.adaptive_grid` puts EACH target's own S_crit at the 30th log-percentile of ITS OWN
    grid. That is right for characterising one gene -- the twist lives above S_crit, so the
    type-0 side gets 70% of the samples -- and it is exactly wrong for putting eight genes
    side by side, because every panel then carries a different dose axis and a reader cannot
    tell "resets at a low dose" from "the axis starts lower here".

    So this module derives ONE grid from the whole S_crit distribution: the geometric median
    of the finite S_crit values is placed at that same 30th log-percentile, over a span wide
    enough that most genes' transitions land inside it. Genes whose S_crit is far off the
    median fall outside, and that is a RESULT -- the statement that they reset at a dose the
    rest of the model does not care about, or do not reset at all.

WHAT IT DOES NOT DO
    It does not share a grid ACROSS MODELS. A dose is a production rate (pulse) or a
    concentration (instant) in the target species' own units, and the three models' units are
    unrelated -- see `analysis/scrit.py`'s header and `gauge/README.md`. Cross-model figures
    put each model on its own axis, or normalise by that gene's own S_crit; they never pretend
    dose 10 means the same thing in Almeida and Goldbeter.

THE VALIDITY CEILING IS NOT CLAMPED AWAY
    A shared grid runs past SOME target's ceiling almost by construction (Almeida/pulse: BMAL1
    is unstable above 22 while E4BP4's S_crit is 51). Truncating to the lowest ceiling would
    throw away the high-dose half of every other gene. So the grid is left long and the points
    beyond a target's ceiling come back INVALID -- `engine.ptc` tracks the trajectory minimum
    and `valid_mask` NaNs them, so they render as grey holes rather than as numbers.
    `report()` names those targets IN ADVANCE, so a grey region is never a surprise.

    `dt` is taken as the FINEST step any target needed in the scrit scan, because the ceiling
    is a property of the step and not of the model (REPO_MAP hazard 4) and every panel of a
    comparison must be at the same fidelity.
"""
import argparse
import glob
import os

import numpy as np

import analysis  # noqa: F401  -- repo root on sys.path
import paths

#: Fraction of the shared log-dose range that sits BELOW the anchor. Same 0.30 as
#: `scrit.BELOW_FRAC`, and for the same reason: the twist lives above the transition.
BELOW_FRAC = 0.30
#: Default span. Wider than scrit's per-target 1.8, because one grid has to serve a whole
#: model's worth of S_crit values and those spread over ~2.5 decades in all three models.
DECADES = 3.0
#: Default number of dose rows.
N_DOSE = 48

ANCHORS = ('median', 'gmean', 'min', 'max')


def _ganchor(S, how):
    """Geometric summary of the finite S_crit values -- the dose the grid is centred on."""
    s = np.asarray([x for x in np.atleast_1d(S) if np.isfinite(x) and x > 0], float)
    if s.size == 0:
        return np.nan
    if how == 'median':
        return float(np.exp(np.median(np.log(s))))
    if how == 'gmean':
        return float(np.exp(np.mean(np.log(s))))
    if how == 'min':
        return float(s.min())
    if how == 'max':
        return float(s.max())
    raise ValueError(f"anchor must be one of {ANCHORS} or a number, got {how!r}")


def shared_grid(S, n=N_DOSE, below=BELOW_FRAC, decades=DECADES, anchor='median',
                lo=None, hi=None):
    """The shared log dose grid, plus the numbers that say how well it serves each target.

    Returns (grid, info). `info['pos']` is where each target's S_crit falls as a fraction of
    the log range -- 0.30 is the design point; <0 or >1 means that gene's transition is off
    the grid entirely.
    """
    S = np.asarray(S, float)
    a = float(anchor) if not isinstance(anchor, str) else _ganchor(S, anchor)
    if lo is not None and hi is not None:
        l0, l1 = np.log10(float(lo)), np.log10(float(hi))
    elif not np.isfinite(a):
        # NOTHING resets. There is no anchor to place, so span the decades below whatever the
        # caller pinned, rather than inventing a scale.
        l1 = np.log10(float(hi)) if hi is not None else 1.0
        l0 = np.log10(float(lo)) if lo is not None else l1 - decades
    else:
        l0 = np.log10(a) - below * decades if lo is None else np.log10(float(lo))
        l1 = np.log10(a) + (1.0 - below) * decades if hi is None else np.log10(float(hi))
    if not l1 > l0:
        raise ValueError(f"empty dose range: lo={10 ** l0:g} hi={10 ** l1:g}")
    grid = np.geomspace(10 ** l0, 10 ** l1, int(n))
    with np.errstate(divide='ignore', invalid='ignore'):
        pos = (np.log10(S) - l0) / (l1 - l0)
    pos = np.asarray(pos, float)
    info = dict(anchor=a, lo=float(10 ** l0), hi=float(10 ** l1), n=int(n),
                below=float(below), decades=float(l1 - l0), pos=pos,
                n_inside=int(np.sum(np.isfinite(pos) & (pos >= 0) & (pos <= 1))),
                n_resetting=int(np.sum(np.isfinite(S))), n_targets=int(S.size))
    return grid, info


# --------------------------------------------------------------------------- #
def read_scrit(model_name, mode, tag=None):
    """Merge every scrit npz of a run into {targets, S_crit, d_max_valid, dt_used, ...}."""
    tag = tag or paths.latest_run(model_name, 'scrit')
    if tag is None:
        raise SystemExit(f"run `$PY -m analysis.scrit --model {model_name} "
                         f"--mode {mode}` first")
    d = paths.out_dir(model_name, 'scrit', tag, create=False)
    files = sorted(f for f in glob.glob(os.path.join(d, f'scrit_{mode}*.npz'))
                   if 'dosegrid' not in os.path.basename(f))
    if not files:
        raise SystemExit(f"no scrit_{mode}*.npz under {d}")
    per = ['S_crit', 'd_max_valid', 'dt_used', 'ceiling_reason', 'reentrant', 'S_converged']
    targets, acc = [], {k: [] for k in per}
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        for i, t in enumerate([str(x) for x in z['targets']]):
            if t in targets:                       # a later shard supersedes an earlier one
                j = targets.index(t)
                for k in per:
                    if k in z:
                        acc[k][j] = z[k][i]
            else:
                targets.append(t)
                for k in per:
                    acc[k].append(z[k][i] if k in z else np.nan)
    out = {k: np.array(v) for k, v in acc.items()}
    out.update(targets=np.array(targets), tag=tag,
               files=[os.path.basename(f) for f in files])
    return out


def derive(model_name, mode='pulse', tag=None, n=N_DOSE, below=BELOW_FRAC, decades=DECADES,
           anchor='median', lo=None, hi=None, dt=None, targets=None, save=True):
    """Read a scrit run, build the shared grid, print the table, save it beside the S_crit."""
    sc = read_scrit(model_name, mode, tag)
    ts = [str(t) for t in sc['targets']]
    if targets:
        keep = [i for i, t in enumerate(ts) if t in set(targets)]
        ts = [ts[i] for i in keep]
        for k in ('S_crit', 'd_max_valid', 'dt_used', 'ceiling_reason'):
            sc[k] = np.asarray(sc[k])[keep]
    S = np.asarray(sc['S_crit'], float)
    grid, info = shared_grid(S, n=n, below=below, decades=decades, anchor=anchor,
                             lo=lo, hi=hi)

    dtu = np.asarray(sc['dt_used'], float)
    dt = float(dt) if dt is not None else float(np.nanmin(dtu))
    blob = dict(model=model_name, mode=mode, scrit_tag=sc['tag'], targets=np.array(ts),
                grid=grid, dt=dt, S_crit=S, pos=info['pos'],
                d_max_valid=np.asarray(sc['d_max_valid'], float), dt_used=dtu,
                ceiling_reason=np.asarray(sc['ceiling_reason']),
                anchor=info['anchor'], lo=info['lo'], hi=info['hi'],
                below=info['below'], decades=info['decades'])
    report(blob)
    if save:
        out = paths.out_path(model_name, 'scrit', f'dosegrid_{mode}.npz', sc['tag'])
        paths.savez(out, **blob)
        print(f"[dosegrid] -> {out}", flush=True)
    return blob


def load(model_name, mode='pulse', tag=None):
    """The saved shared grid: (grid, dt, blob). Raises with the command to run if absent."""
    tag = tag or paths.latest_run(model_name, 'scrit')
    fp = None if tag is None else paths.out_path(model_name, 'scrit',
                                                 f'dosegrid_{mode}.npz', tag)
    if fp is None or not os.path.exists(fp):
        raise SystemExit(f"no shared dose grid for {model_name}/{mode}; run "
                         f"`$PY -m analysis.dosegrid --model {model_name} --mode {mode}`")
    z = dict(np.load(fp, allow_pickle=True))
    return np.asarray(z['grid'], float), float(z['dt']), z


def report(blob):
    """The table that says whether this grid is fit to compare these genes on."""
    ts = [str(t) for t in blob['targets']]
    S, pos = np.asarray(blob['S_crit'], float), np.asarray(blob['pos'], float)
    ceil = np.asarray(blob['d_max_valid'], float)
    dtu = np.asarray(blob['dt_used'], float)
    g = np.asarray(blob['grid'], float)
    print(f"\n{'=' * 86}\nSHARED dose grid -- {blob['model']} ({blob['mode']} mode)"
          f"\n{'=' * 86}")
    print(f"  anchor (geo-median S_crit) = {float(blob['anchor']):.4g}   "
          f"range {float(blob['lo']):.4g} .. {float(blob['hi']):.4g}  "
          f"({float(blob['decades']):.2f} decades, {len(g)} rows)   dt = {float(blob['dt']):g}")
    print(f"  {'target':10s} {'S_crit':>10s} {'pos':>7s} {'where':>8s} "
          f"{'ceiling':>10s} {'dt scan':>8s}  note")
    order = np.argsort([s if np.isfinite(s) else np.inf for s in S])
    for i in order:
        p = pos[i]
        where = ('--' if not np.isfinite(p) else
                 ('below' if p < 0 else ('above' if p > 1 else f'{100 * p:.0f}%')))
        note = []
        if not np.isfinite(S[i]):
            note.append('never resets')
        elif p < 0:
            note.append('transition BELOW the grid')
        elif p > 1:
            note.append('transition ABOVE the grid')
        if np.isfinite(ceil[i]) and ceil[i] < g.max():
            note.append(f'{100 * float(np.mean(g > ceil[i])):.0f}% of rows past its ceiling')
        print(f"  {ts[i]:10s} {('nan' if not np.isfinite(S[i]) else f'{S[i]:.4g}'):>10s} "
              f"{('--' if not np.isfinite(p) else f'{p:.2f}'):>7s} {where:>8s} "
              f"{('nan' if not np.isfinite(ceil[i]) else f'{ceil[i]:.4g}'):>10s} "
              f"{dtu[i]:>8g}  {'; '.join(note)}")
    inside = pos[np.isfinite(pos) & (pos >= 0) & (pos <= 1)]
    fin = int(np.sum(np.isfinite(S)))
    print(f"\n  {inside.size}/{fin} resetting targets have their S_crit INSIDE the shared "
          f"range ({len(ts)} targets total)")
    if inside.size:
        print(f"  their positions span {100 * inside.min():.0f}%..{100 * inside.max():.0f}% "
              f"of the log range (design point: {100 * float(blob['below']):.0f}%)")
    hidden = [ts[i] for i in range(len(ts))
              if np.isfinite(ceil[i]) and ceil[i] < g.max() * 0.999]
    if hidden:
        print(f"  EXPECT grey (invalid) high-dose regions for: {', '.join(hidden)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description='one shared dose grid per (model, mode)')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--mode', default=os.environ.get('MODE', 'pulse'),
                    choices=('pulse', 'instant'))
    ap.add_argument('--tag', default=None, help='scrit run tag (default: newest)')
    ap.add_argument('--targets', default=None)
    ap.add_argument('--n', type=int, default=N_DOSE)
    ap.add_argument('--below', type=float, default=BELOW_FRAC)
    ap.add_argument('--decades', type=float, default=DECADES)
    ap.add_argument('--anchor', default='median')
    ap.add_argument('--lo', type=float, default=None)
    ap.add_argument('--hi', type=float, default=None)
    ap.add_argument('--dt', type=float, default=None)
    ap.add_argument('--dry-run', action='store_true', help='print the table, save nothing')
    a = ap.parse_args(argv)
    derive(a.model, mode=a.mode, tag=a.tag, n=a.n, below=a.below, decades=a.decades,
           anchor=a.anchor, lo=a.lo, hi=a.hi, dt=a.dt,
           targets=(a.targets.split(',') if a.targets else None), save=not a.dry_run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
