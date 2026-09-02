"""
analysis/features.py
====================
THE TIDY FEATURE TABLE: one row per (model, mode, target), derived from saved surfaces only.

    $PY -m analysis.features --mode pulse [--models almeida,korencic,goldbeter]
    $PY -m analysis.features --mode pulse --char-tag zoo01 --csv

WHY IT IS A SEPARATE STEP FROM THE FIGURES
    Because a plotter that derives its own numbers is a second implementation of the analysis,
    and the two drift. Every scalar a comparison figure shows is computed HERE, saved to an
    npz (and a csv), and read back by `analysis/figures_zoo.py` without a single derivation.
    Re-styling a figure then costs nothing and cannot change a number.

    Nothing here integrates. It reads `analysis.characterize`'s surfaces, `analysis.scrit`'s
    S_crit and `analysis.dosegrid`'s shared grid, and reduces them -- so it is seconds, and can
    be re-run whenever a feature definition changes.

THE QUALITY GATE IS PART OF THE ROW (REPO_MAP hazard 12)
    `analysis.quality` exists because a topological feature extractor ALWAYS returns a number:
    a phase-scrambled CRY surface reported S_crit = 138.1 and a twist of 0.285 and nothing
    downstream could tell. So every row carries `passed`, `scramble` and the winding set, and
    the comparison figures grey out the rows that fail. A feature from a failed surface is not
    a small error, it is not a measurement at all.

WHICH TWIST NUMBER
    Three, deliberately, because they fail in different ways (see analysis/winding.py):
      total_twist        circular span of the fixed-point curve. SATURATES AT 0.5 -- Almeida's
                         BMAL1 base sits at 0.4999, pinned to the ceiling.
      accumulated_twist  unwrapped total travel. Unbounded, and the one to compare magnitudes
                         with; a value still climbing with dose resolution means aliasing.
      signed_twist       net direction. A sign flip between resolutions IS under-sampling.

DOES THE SURFACE STILL RESOLVE OLD PHASE? (hazard 17)
    `span_top` is the circular span of new phase across old phase at the highest usable dose.
    A radial/flat type-0 surface has span -> 0: it maps every old phase to one new phase, has
    zero twist BY CONSTRUCTION, and carries no phase information at all. Reading a twist number
    off such a surface is meaningless, so the span travels next to the twist everywhere.
"""
import argparse
import csv
import glob
import os

import numpy as np

import analysis  # noqa: F401  -- repo root on sys.path
import paths
from analysis import winding as W
from analysis import genemap as GM

#: Cross-model results live under this pseudo-model bucket, so `out/<model>/...` stays the one
#: layout. `paths.out_dir` deliberately does not validate the model name against a registry.
ZOO = 'zoo'

#: Phase shift below which the anchor counts as already at 0 (1e-4 cyc ~ 9 s).
PHASE_TOL = 1e-4

#: Scalar columns of the table, in report order.
COLUMNS = ('model', 'mode', 'target', 'gene', 'level',
           'S_scan', 'S_surf', 'phi_star', 'n_sing', 'pos_in_grid',
           'total_twist', 'acc_twist', 'signed_twist', 'span_top', 'span_hi_half',
           'type0_frac', 'type1_frac', 'unusable_frac', 'valid_frac', 'min_amp',
           'ceiling', 'ceiling_reason', 'dt', 'scramble', 'passed', 'phase_offset')


def _load_char(model_name, mode, tag=None):
    """{target: dict} from a characterize run (one file per target set), + the tag used."""
    tag = tag or paths.latest_run(model_name, 'characterize')
    if tag is None:
        return {}, None
    d = paths.out_dir(model_name, 'characterize', tag, create=False)
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, f'char_{mode}*.npz'))):
        z = np.load(fp, allow_pickle=True)
        ts = [str(x) for x in z['targets']]
        for i, t in enumerate(ts):
            if f'ptc__{t}' not in z:
                continue
            out[t] = dict(old=np.asarray(z[f'old__{t}']), doses=np.asarray(z[f'doses__{t}']),
                          ptc=np.asarray(z[f'ptc__{t}']), amp=np.asarray(z[f'amp__{t}']),
                          valid=np.asarray(z[f'valid__{t}']), Wv=np.asarray(z[f'W__{t}']),
                          twist=np.asarray(z[f'twist__{t}']),
                          sing_phi=np.asarray(z[f'sing_phi__{t}']),
                          sing_dose=np.asarray(z[f'sing_dose__{t}']),
                          sing_sign=np.asarray(z[f'sing_sign__{t}']),
                          S=float(z['S'][i]), phi=float(z['phi'][i]),
                          n_sing=int(z['n_sing'][i]),
                          total_twist=float(z['total_twist'][i]),
                          min_amp=float(z['min_amp'][i]), dt=float(z['dt'][i]))
    return out, tag


def _load_scan(model_name, mode, tag=None):
    """{target: dict} from THE SCREEN (`analysis.scrit`) rather than from the refined surfaces.

    The wide dose scan already IS a PTC surface -- 32 phases x 40 doses over seven decades --
    and it costs nothing extra to read. It is coarse in phase and enormous in dose, which is
    the opposite trade from `analysis.characterize`, so the two answer different questions:
    the screen shows WHERE ON THE DOSE AXIS anything happens at all, the refined surface shows
    WHAT it looks like once you are there.

    TWO THINGS ARE COARSER THAN THEY LOOK, and both are recorded rather than smoothed over:
      * validity is PER DOSE ROW, not per cell. The scan keeps one running minimum per dose
        (the min over that row's phases), so a row is either believed or discarded whole. The
        refined path tracks it per trajectory.
      * `amp` is likewise one number per dose, the row's minimum relative amplitude, so it is
        stored 1-D here where the refined path stores the full grid.
    """
    from analysis.dosegrid import _scrit_files
    from engine.ptc import valid_mask
    tag, files = _scrit_files(model_name, mode, tag)
    if not files:
        return {}, None
    out = {}
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        for i, t in enumerate([str(x) for x in z['targets']]):
            if f'ptc__{t}' not in z:
                continue
            doses = np.asarray(z[f'doses__{t}'], float)
            ptc = np.asarray(z[f'ptc__{t}'], float)              # (n_phase, n_dose)
            ymin = np.asarray(z[f'ymin__{t}'], float)            # per DOSE
            ampd = np.asarray(z[f'amp__{t}'], float)             # per DOSE
            colok = np.asarray(valid_mask(ymin), bool)
            # discard the rows the integrator was unstable on, exactly as `scrit` does before
            # taking a winding number -- otherwise the figure shows numbers scrit refused.
            ptc = np.where(colok[None, :], ptc, np.nan)
            n_phase = ptc.shape[0]
            old = np.arange(n_phase) / n_phase
            sings = W.find_singularities(old, doses, ptc)
            S, phi, nsing = W.detect_grid(old, doses, ptc)
            tw = W.twist_curve(old, doses, ptc)
            fin = np.isfinite(ampd) & colok
            out[t] = dict(
                old=old, doses=doses, ptc=ptc, amp=ampd,
                valid=np.broadcast_to(colok, ptc.shape).copy(),
                Wv=np.asarray(z[f'W__{t}'], float), twist=tw,
                sing_phi=np.array([s['phi'] for s in sings]),
                sing_dose=np.array([s['dose'] for s in sings]),
                sing_sign=np.array([s['sign'] for s in sings]),
                S=float(S), phi=float(phi), n_sing=int(nsing),
                total_twist=float(W.total_twist(tw)),
                min_amp=float(np.min(ampd[fin])) if np.any(fin) else np.nan,
                dt=float(z['dt_used'][i]) if 'dt_used' in z else np.nan)
    return out, tag


def _load_scrit(model_name, mode):
    """{target: (S_crit, ceiling, reason, pos_in_shared_grid)} -- empty dict if never scanned."""
    from analysis.dosegrid import read_scrit
    try:
        sc = read_scrit(model_name, mode)
    except SystemExit:
        return {}, None
    pos = {}
    try:
        from analysis.dosegrid import load as _load_dg
        _g, _dt, dg = _load_dg(model_name, mode, sc['tag'])
        pos = {str(t): float(p) for t, p in zip(dg['targets'], dg['pos'])}
    except SystemExit:
        pass
    out = {}
    for i, t in enumerate([str(x) for x in sc['targets']]):
        out[t] = (float(sc['S_crit'][i]), float(sc['d_max_valid'][i]),
                  str(sc['ceiling_reason'][i]), pos.get(t, np.nan))
    return out, sc['tag']


def _span_over_old(ptc_col):
    """Circular span of new phase across old phase in ONE dose row -- how much old-phase
    structure the surface still carries there. `circ_span`, not peak-to-peak: phase wraps, and
    the naive version read 4 informative dose rows for REV against a true 1 (hazard 17)."""
    return W.circ_span(np.asarray(ptc_col, float))


def row_of(model_name, mode, target, ch, sc):
    """One tidy row. `ch` is that target's characterize dict; `sc` its scrit tuple."""
    ptc = np.asarray(ch['ptc'], float)                  # (n_phase, n_dose)
    doses = np.asarray(ch['doses'], float)
    Wv = np.asarray(ch['Wv'], float)
    valid = np.asarray(ch['valid'], bool)
    fin = np.isfinite(Wv)
    n_dose = ptc.shape[1]

    usable_cols = np.where(np.isfinite(ptc).sum(axis=0) > 0.5 * ptc.shape[0])[0]
    top = int(usable_cols[-1]) if usable_cols.size else -1
    span_top = _span_over_old(ptc[:, top]) if top >= 0 else np.nan
    # the upper HALF of the dose axis, averaged: one row can be a fluke, the asymptote is not
    hi = [k for k in usable_cols if k >= n_dose // 2]
    span_hi = float(np.mean([_span_over_old(ptc[:, k]) for k in hi])) if hi else np.nan

    from analysis.quality import score
    q = score(ch['old'], doses, ptc)

    gene, level = GM.gene_of(model_name, target)
    S_scan, ceiling, reason, pos = sc if sc is not None else (np.nan, np.nan, '', np.nan)
    return dict(
        model=model_name, mode=mode, target=target, gene=gene or '', level=level or '',
        S_scan=S_scan, S_surf=ch['S'], phi_star=ch['phi'], n_sing=ch['n_sing'],
        pos_in_grid=pos,
        total_twist=ch['total_twist'],
        acc_twist=W.accumulated_twist(ch['twist']),
        signed_twist=W.signed_twist(ch['twist']),
        span_top=span_top, span_hi_half=span_hi,
        type0_frac=float(np.mean(fin & (np.abs(Wv) < 0.5))),
        type1_frac=float(np.mean(fin & (np.abs(Wv) >= 0.5))),
        unusable_frac=float(np.mean(~np.isfinite(ptc))),
        valid_frac=float(np.mean(valid)), min_amp=ch['min_amp'],
        ceiling=ceiling, ceiling_reason=reason, dt=ch['dt'],
        scramble=q['scramble'], passed=bool(q['passed']), phase_offset=0.0,
    )


#: Where a surface came from. 'scan' is the wide screen `analysis.scrit` already produced;
#: 'surface' is the refined render `analysis.characterize` makes on the derived dose grid.
SOURCES = ('scan', 'surface')


def apply_phase_offset(ch, off):
    """Move the phase ORIGIN of one surface to the common anchor (analysis.phaseref).

    Both PTC axes carry the same origin -- the readout is calibrated so dose 0 is the identity
    -- so ONE constant comes off BOTH: the old-phase labels and the new-phase values. The
    old-phase samples stay exactly where they were measured, they are just relabelled and put
    back in ascending order, so nothing is interpolated and no value is invented.

    Every difference-based feature (total / accumulated / signed twist, the old-phase span) is
    invariant under this by construction; only the ABSOLUTE phases move -- phi*, the twist
    curve's height, and the surface's colour.
    """
    off = float(off)
    # Below PHASE_TOL the "offset" is the peak-finder's own noise, not a displacement: Almeida
    # and Goldbeter come back at 1.6e-06 and -1.2e-05 cyc, i.e. already on the anchor. Shifting
    # by that much would rotate the sample grid off 0 for no reason and make two panels that
    # ARE aligned look like they start in different places.
    if abs(((off + 0.5) % 1.0) - 0.5) < PHASE_TOL:
        return ch
    ch = dict(ch)
    old = (np.asarray(ch['old'], float) - off) % 1.0
    order = np.argsort(old)                      # a pure roll: the cyclic order is preserved
    ch['old'] = old[order]
    ch['ptc'] = (np.asarray(ch['ptc'], float)[order] - off) % 1.0
    for k in ('amp', 'valid'):                   # row-permute the per-cell companions only;
        a = np.asarray(ch[k])                    # the scan stores `amp` per DOSE, 1-D
        if a.ndim == 2 and a.shape[0] == len(order):
            ch[k] = a[order]
    ch['twist'] = (np.asarray(ch['twist'], float) - off) % 1.0
    ch['sing_phi'] = (np.asarray(ch['sing_phi'], float) - off) % 1.0
    # 'phi' is present on a freshly loaded surface but not on one read back from the saved
    # table (where it lives in the row, not the surface), so shift it only if it is here.
    if 'phi' in ch and np.isfinite(ch['phi']):
        ch['phi'] = float((ch['phi'] - off) % 1.0)
    return ch


def build(mode='pulse', models=GM.MODELS, char_tags=None, source='surface',
          scope=True, phase_tag=None, verbose=True):
    """Every model's rows for one mode, plus the raw surfaces keyed (model, target)."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    from analysis.phaseref import load as _load_phaseref
    offsets, _pr = _load_phaseref(phase_tag)
    if verbose and not offsets:
        print("[features] NO COMMON PHASE ORIGIN -- run `$PY -m analysis.phaseref`. "
              "Absolute phases are model-local and NOT comparable.")
    rows, surfaces, tags = [], {}, {}
    for m in models:
        ch, ctag = (_load_scan(m, mode, (char_tags or {}).get(m)) if source == 'scan'
                    else _load_char(m, mode, (char_tags or {}).get(m)))
        if not ch:
            if verbose:
                print(f"[features] {m}/{mode}: no {source} run -- skipping")
            continue
        keep = GM.scope_targets(m) if scope else list(ch)
        skipped = [t for t in ch if t not in keep]
        if verbose and skipped:
            print(f"[features] {m}: out of comparison scope, skipping {skipped}")
        sc, stag = _load_scrit(m, mode)
        tags[m] = dict(source=ctag, scrit=stag)
        off = float(offsets.get(m, 0.0))
        for t in [x for x in keep if x in ch]:
            surf = apply_phase_offset(ch[t], off)
            r = row_of(m, mode, t, surf, sc.get(t))
            r['phase_offset'] = off
            rows.append(r)
            surfaces[(m, t)] = surf
    return rows, surfaces, tags


def _fname(mode, source):
    """Screen and refined tables never share a filename: they are different measurements of
    the same thing, and overwriting one with the other is not a merge but a loss."""
    return f"features_{mode}.npz" if source == 'surface' else f"features-{source}_{mode}.npz"


def save(rows, surfaces, tags, mode, tag=None, write_csv=True, source='surface'):
    """One npz with the scalars AND every surface, so a figure never reopens a per-model file."""
    tag = paths.run_tag(tag)
    blob = {c: np.array([r[c] for r in rows]) for c in COLUMNS}
    for (m, t), ch in surfaces.items():
        for k in ('old', 'doses', 'ptc', 'amp', 'valid', 'Wv', 'twist',
                  'sing_phi', 'sing_dose', 'sing_sign'):
            blob[f'{k}__{m}__{t}'] = np.asarray(ch[k])
    # NB no scalar `mode` key: 'mode' is already a per-row COLUMN, and writing the scalar over
    # it silently turned the column into a 0-d array that `load` could not index.
    blob['tags'] = np.array([f"{m}:{v['source']}|{v['scrit']}" for m, v in tags.items()])
    blob['source'] = source
    out = paths.out_path(ZOO, 'features', _fname(mode, source), tag)
    paths.savez(out, **blob)
    print(f"[features] -> {out}", flush=True)
    if write_csv:
        cp = out.replace('.npz', '.csv')
        with open(cp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(COLUMNS))
            w.writeheader()
            for r in rows:
                w.writerow({c: r[c] for c in COLUMNS})
        print(f"[features] -> {cp}", flush=True)
    return out


def load(mode='pulse', tag=None, source='surface'):
    """The saved feature table: (rows, surfaces, blob)."""
    tag = tag or paths.latest_run(ZOO, 'features')
    fp = None if tag is None else paths.out_path(ZOO, 'features', _fname(mode, source), tag)
    if fp is None or not os.path.exists(fp):
        raise SystemExit(f"no {source} feature table for mode {mode!r}; run "
                         f"`$PY -m analysis.features --mode {mode} --source {source}`")
    z = dict(np.load(fp, allow_pickle=True))
    n = len(z['target'])
    rows = [{c: (str(z[c][i]) if z[c].dtype.kind in 'US' else z[c][i]) for c in COLUMNS}
            for i in range(n)]
    for r in rows:                                   # numpy scalars -> python, csv-like
        r['passed'] = str(r['passed']) in ('True', 'true', '1')
        for c in ('S_scan', 'S_surf', 'phi_star', 'pos_in_grid', 'total_twist', 'acc_twist',
                  'signed_twist', 'span_top', 'span_hi_half', 'type0_frac', 'type1_frac',
                  'unusable_frac', 'valid_frac', 'min_amp', 'ceiling', 'dt', 'scramble',
                  'phase_offset'):
            r[c] = float(r[c])
        r['n_sing'] = int(float(r['n_sing']))
    surfaces = {}
    for k in z:
        if k.startswith('ptc__'):
            _p, m, t = k.split('__', 2)
            surfaces[(m, t)] = {kk: z[f'{kk}__{m}__{t}'] for kk in
                                ('old', 'doses', 'ptc', 'amp', 'valid', 'Wv', 'twist',
                                 'sing_phi', 'sing_dose', 'sing_sign')}
    return rows, surfaces, z


def print_table(rows, mode, source='surface'):
    what = ('the WIDE SCREEN (scrit: 7 decades of dose, coarse in phase)' if source == 'scan'
            else 'the REFINED surfaces (characterize)')
    print(f"\n{'=' * 128}\nZOO FEATURE TABLE -- {mode} mode, from {what}\n{'=' * 128}")
    hdr = (f"  {'model':10s} {'target':9s} {'gene':7s} {'level':8s} {'S_scan':>9s} "
           f"{'S_surf':>9s} {'phi*':>6s} {'twist':>6s} {'acc':>6s} {'sgn':>6s} "
           f"{'span_hi':>7s} {'type0':>6s} {'minamp':>7s} {'scr':>6s} {'q':>4s}")
    print(hdr)
    def _f(x, w=9, p=4):
        return f"{'--':>{w}s}" if not np.isfinite(x) else f"{x:>{w}.{p}g}"
    for g in GM.GENES:
        sub = [r for r in rows if r['gene'] == g]
        if not sub:
            continue
        print(f"  -- {g} " + '-' * 100)
        for r in sub:
            print(f"  {r['model']:10s} {r['target']:9s} {r['gene']:7s} {r['level']:8s} "
                  f"{_f(r['S_scan'])} {_f(r['S_surf'])} {_f(r['phi_star'], 6, 3)} "
                  f"{_f(r['total_twist'], 6, 3)} {_f(r['acc_twist'], 6, 3)} "
                  f"{_f(r['signed_twist'], 6, 3)} {_f(r['span_hi_half'], 7, 3)} "
                  f"{_f(r['type0_frac'], 6, 3)} {_f(r['min_amp'], 7, 3)} "
                  f"{_f(r['scramble'], 6, 3)} {'ok' if r['passed'] else 'FAIL':>4s}")
    bad = [r for r in rows if not r['passed']]
    print(f"\n  {len(rows) - len(bad)}/{len(rows)} surfaces pass the quality gate.")
    if bad:
        print("  DO NOT quote features from: "
              + ', '.join(f"{r['model']}/{r['target']}" for r in bad))
    flat = [r for r in rows if np.isfinite(r['span_hi_half']) and r['span_hi_half'] < 0.05]
    if flat:
        print("  Phase-BLIND above mid-dose (span < 0.05 cyc; its twist is meaningless): "
              + ', '.join(f"{r['model']}/{r['target']}" for r in flat))


def main(argv=None):
    ap = argparse.ArgumentParser(description='the cross-model PTC feature table')
    ap.add_argument('--mode', default='pulse', choices=('pulse', 'instant'))
    ap.add_argument('--models', default=','.join(GM.MODELS))
    ap.add_argument('--source', default='surface', choices=SOURCES,
                    help="'scan' reads the wide screen scrit already produced; "
                         "'surface' reads the refined characterize render")
    ap.add_argument('--char-tag', default=None,
                    help='source run tag (characterize or scrit), or model=tag,model=tag')
    ap.add_argument('--tag', default=None, help='output tag')
    ap.add_argument('--all-targets', action='store_true',
                    help='ignore the comparison scope and tabulate every target')
    ap.add_argument('--phase-tag', default=None, help='analysis.phaseref run tag')
    ap.add_argument('--no-csv', action='store_true')
    a = ap.parse_args(argv)
    models = [m.strip() for m in a.models.split(',') if m.strip()]
    ct = None
    if a.char_tag:
        ct = ({k: v for k, v in (p.split('=', 1) for p in a.char_tag.split(','))}
              if '=' in a.char_tag else {m: a.char_tag for m in models})
    rows, surf, tags = build(a.mode, models, ct, source=a.source,
                             scope=not a.all_targets, phase_tag=a.phase_tag)
    if not rows:
        raise SystemExit("[features] nothing to tabulate")
    print_table(rows, a.mode, a.source)
    save(rows, surf, tags, a.mode, a.tag, write_csv=not a.no_csv, source=a.source)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
