"""
analysis/figures_zoo.py
=======================
THE CROSS-MODEL FIGURES. Pure read of `analysis.features`' table -- nothing here integrates
and nothing here derives a number, so re-styling is free and cannot change a result.

    $PY -m analysis.features    --mode pulse            # <- makes the table (seconds)
    $PY -m analysis.figures_zoo --mode pulse --which all

THREE FAMILIES
  surfaces   one figure per MODEL: every gene's PTC surface on the model's SHARED dose grid,
             with its fixed-point (twist) curve beneath. Shared axis is the whole point --
             with per-target grids you cannot tell "resets at a low dose" from "this panel's
             axis starts lower".
  genes      one figure per GENE: the same gene's surface in every model that has it, side by
             side. Each model keeps its OWN dose axis, because a dose is in the target
             species' own units and those units are unrelated across models
             (analysis/dosegrid.py). The one honestly shared axis is dose/S*, and it gets the
             wide bottom panel.
  features   the scalar comparison: S_crit, phi*, the three twist measures, the old-phase
             span, the type-0 fraction and the amplitude floor, gene by gene, model by model.
             Plus a pulse-vs-instant version, since the two perturbations are the experiment's
             two arms and the interesting claim is where they DISAGREE.

CONVENTIONS (plotting.py, inherited from input_screen)
  * phase is CIRCULAR -> `cmocean.cm.phase`; a sequential map invents a discontinuity at 0/1.
  * overlays are ACHROMATIC so they read against any hue: white filled circle for a +1
    singularity, white cross for -1, dashed white for S_crit.
  * missing data is neutral GREY, not white -- a hole must look like a hole.
  * in the scalar figures, COLOUR is the model and MARKER is the expression level (mRNA /
    protein / nuclear / complex), because "Goldbeter's MP vs Almeida's PER" is a comparison
    across BOTH axes at once and the figure must not let one masquerade as the other.

A SURFACE THAT FAILED THE QUALITY GATE IS DRAWN, AND MARKED
    Not dropped: an absent panel reads as "not measured". It gets a red QC FAIL stamp, its
    scalars are drawn hollow, and REPO_MAP hazard 12 is why -- a phase-scrambled surface
    returns confident numbers, so the mark has to be on the picture.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import analysis  # noqa: F401
import paths
from analysis import genemap as GM
from analysis.features import ZOO
from plotting import phase_cmap, broken

#: One colour per model, one marker per expression level. See the header.
MODEL_COLOR = {'almeida': '#1f77b4', 'korencic': '#d95f02', 'goldbeter': '#2ca02c'}
LEVEL_MARKER = {'mrna': 'o', 'protein': 's', 'nuclear': '^', 'complex': 'D'}

NL = chr(10)

_CTX = dict(tag=None, publish=False, source='surface', anchor=None, roll=0.0)


def _phase_label():
    """What the phase axes are measured FROM. Never silently omit this: an unanchored
    old phase is model-local and two models' values are not the same time of day."""
    a = _CTX.get('anchor')
    r = float(_CTX.get('roll') or 0.0)
    base = f'old phase  (0 = {a} peak)' if a else 'old phase  (MODEL-LOCAL origin)'
    return base if abs(r) < 1e-9 else base[:-1] + f', rolled {r:+.2f} cyc for display)'


def _sfx():
    """Filename suffix marking WHICH measurement a figure is of. The screen and the refined
    render are different pictures of the same target and must not overwrite each other."""
    return '' if _CTX['source'] == 'surface' else f"_{_CTX['source']}"


def _save(fig, name, **cfg):
    p = paths.save_figure(fig, ZOO, 'figures', name, tag=_CTX['tag'],
                          publish=(name if _CTX['publish'] else None), **cfg)
    plt.close(fig)
    return p


def _fmt(x, p=3):
    return '--' if not np.isfinite(x) else f'{x:.{p}g}'


def _row(rows, model, target):
    for r in rows:
        if r['model'] == model and r['target'] == target:
            return r
    return None


# --------------------------------------------------------------------------- #
#  panels
# --------------------------------------------------------------------------- #
def _log_dose(doses):
    """Is this dose grid log-spaced or linear? Detected, not declared.

    A linear grid resolves the HIGH-dose end far better at equal row count, which is where the
    isochrons wind fastest and where a log grid aliases. Both are legitimate and the figure
    must not mislabel one as the other, so the axis follows the data."""
    d = np.asarray(doses, float)
    if len(d) < 3 or d.min() <= 0:
        return d.min() > 0
    lin = np.std(np.diff(d)) / max(abs(np.mean(np.diff(d))), 1e-300)
    lg = np.std(np.diff(np.log(d))) / max(abs(np.mean(np.diff(np.log(d)))), 1e-300)
    return lg <= lin


def _mesh_edges(old, doses):
    """Cell EDGES for pcolormesh, so the map fills its frame with no sliver of blank axis.

    `shading='nearest'` centres a cell on each sample, so the map spans only
    [x0 - dx/2, xN + dx/2] and forcing xlim to (0, 1) leaves a half-cell gap at one end. Old
    phase is PERIODIC, so the honest fix is to wrap: the column at phase 0 is also the column
    at phase 1, and with it appended the mesh covers [0, 1] exactly. Dose is log-spaced, so
    its edges are the GEOMETRIC midpoints, extrapolated by half a cell at each end.
    """
    o = np.asarray(old, float)
    d = np.asarray(doses, float)
    dx = np.diff(o).mean() if len(o) > 1 else 1.0
    # WRAP AT BOTH ENDS. One wrapped column is enough only when the samples start exactly at
    # phase 0. They do not once a model carries a phase offset: Korencic's 0.444-cycle shift
    # to the common Per anchor rolls its grid so old[0] = 0.0243, the mesh then began at
    # +0.0086, and a sliver of blank axis showed between the y-axis and the surface -- in that
    # one panel only. Prepending the last column and appending the first makes the mesh span
    # [o0 - 1.5dx, oN + 1.5dx], which contains [0, 1] whatever the offset.
    xe = np.concatenate([[o[0] - 1.5 * dx], o - dx / 2, [o[-1] + dx / 2, o[-1] + 1.5 * dx]])
    if len(d) < 2:
        return xe, np.array([d[0] * 0.9, d[0] * 1.1])
    if _log_dose(d):
        lg = np.log(d)
        mid = 0.5 * (lg[:-1] + lg[1:])
        ye = np.exp(np.concatenate([[lg[0] - (mid[0] - lg[0])], mid,
                                    [lg[-1] + (lg[-1] - mid[-1])]]))
    else:
        mid = 0.5 * (d[:-1] + d[1:])
        ye = np.concatenate([[d[0] - (mid[0] - d[0])], mid, [d[-1] + (d[-1] - mid[-1])]])
    return xe, ye


def _surface_panel(ax, ch, r, ylim=None, show_y=True, title=None):
    """One PTC surface with its singularities, its S_crit line and its QC verdict."""
    old, doses, ptc = ch['old'], ch['doses'], np.asarray(ch['ptc'], float)
    wrapped = np.vstack([ptc[-1:], ptc, ptc[:1]])   # periodic in phase -- see _mesh_edges
    xe, ye = _mesh_edges(old, doses)
    im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(wrapped.T), cmap=phase_cmap(),
                       vmin=0, vmax=1, shading='flat', rasterized=True)
    ax.set_yscale('log' if _log_dose(doses) else 'linear')
    ax.set_xlim(0.0, 1.0)                     # phase is a full circle in every panel
    ax.set_ylim(*(ylim if ylim else (ye[0], ye[-1])))
    for phi, d, sg in zip(ch['sing_phi'], ch['sing_dose'], ch['sing_sign']):
        ax.plot(phi, d, marker=('o' if sg > 0 else 'x'), ms=7, mfc='white', mec='white',
                mew=1.6, ls='none')
    # THE LINE AND THE MARKER MUST BE THE SAME MEASUREMENT. This drew `S_scan` -- the
    # transition found on the 7-decade SCREEN -- while the white marker is the singularity
    # detected on THIS surface, so the two sat apart by the difference between two dose grids
    # (Almeida/pulse: line 4.88, marker 3.71) and read as a detection failure. The line is now
    # the surface's own S*, so line and marker coincide by construction and any visible gap is
    # a real disagreement worth chasing.
    S = r['S_surf'] if r is not None and np.isfinite(r['S_surf']) else np.nan
    if np.isfinite(S):
        ax.axhline(S, color='white', ls='--', lw=1.2)
    ax.set_xlabel(_phase_label(), fontsize=8)
    ax.set_ylabel('dose' if show_y else '', fontsize=8)
    if not show_y:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=8.5)
    # NO 'QC FAIL' stamp across the surface. The gate's complaints are almost always about a
    # SUBSET of the grid -- a NaN block above the ceiling, a few scrambled cells near the
    # defect -- and a banner over the whole panel reads as "this result is void", which is a
    # stronger claim than the check makes. The verdict travels in the feature table, in the
    # printed report, and as hollow markers in the scalar figure.
    return im


def _winding_bands(ax, doses, Wv):
    """Shade the dose axis by PTC TYPE, behind the twist curve.

    The winding number is the screen's primary readout and it is one integer per dose, so it
    costs a background rather than a panel: pale blue where the PTC is type-0 (|W| < 0.5,
    i.e. resetting), unshaded where it is type-1, and hatched grey where there is no usable
    number at all -- the integrator having left its stability region or the clock having
    stopped. Those three are different claims and a single "no data" colour conflates the last
    two with the first.
    """
    d = np.asarray(doses, float)
    Wv = np.asarray(Wv, float)
    if Wv.shape != d.shape:
        return
    if _log_dose(d):
        edges = np.sqrt(d[:-1] * d[1:])                  # log-midpoints between samples
        edges = np.concatenate([[d[0] ** 2 / edges[0]], edges, [d[-1] ** 2 / edges[-1]]])
    else:
        edges = 0.5 * (d[:-1] + d[1:])
        edges = np.concatenate([[2 * d[0] - edges[0]], edges, [2 * d[-1] - edges[-1]]])
    for k in range(len(d)):
        if not np.isfinite(Wv[k]):
            ax.axvspan(edges[k], edges[k + 1], color='0.86', lw=0, zorder=0, hatch='///',
                       edgecolor='0.7')
        elif abs(Wv[k]) < 0.5:
            ax.axvspan(edges[k], edges[k + 1], color='#cfe3f5', lw=0, zorder=0)


def _twist_panel(ax, ch, r, xlim=None, show_y=True, xlabel='dose', bands=True):
    """Stable fixed-point phase vs dose -- house orientation: dose on x (log), phase on y."""
    doses, tw = np.asarray(ch['doses'], float), np.asarray(ch['twist'], float)
    if bands and 'Wv' in ch:
        _winding_bands(ax, doses, ch['Wv'])
    ax.axhline(0, color='0.75', lw=0.5, ls=':')
    ax.axhline(1, color='0.75', lw=0.5, ls=':')
    ax.plot(*broken(doses, tw), color='k', lw=1.8, zorder=5)
    S = r['S_surf'] if r is not None and np.isfinite(r['S_surf']) else np.nan
    if np.isfinite(S):
        ax.axvline(S, color='0.55', ls='--', lw=0.9, zorder=1)
        # Mark the nearest dose row THAT HAS A FIXED POINT, not the nearest row outright. The
        # row nearest S* is often NaN: below a type-1 -> type-0 transition the PTC is a degree-1
        # circle map, which need not have a fixed point at all, so there is no entrainment phase
        # to mark. Goldbeter/MB is exactly this -- S* = 9.90, the row at 9.38 is NaN and the
        # first finite one is 10.42 -- and the marker simply vanished from that panel.
        fin = np.where(np.isfinite(tw))[0]
        if len(fin):
            k = int(fin[np.argmin(np.abs(doses[fin] - S))])
            ax.plot([doses[k]], [tw[k]], 'o', ms=5, mfc='white', mec='k', mew=0.8, zorder=7)
    ax.set_xscale('log' if _log_dose(doses) else 'linear')
    ax.set_ylim(-0.03, 1.03)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(('stable FP phase' if _CTX.get('anchor') else 'stable FP phase (local)')
                  if show_y else '', fontsize=8)
    if not show_y:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=7)
    return ax


def _phase_colorbar(fig, im, label='new phase (cyc)'):
    cax = fig.add_axes([0.925, 0.30, 0.011, 0.40])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


# --------------------------------------------------------------------------- #
#  1. one figure per MODEL, one panel per gene
# --------------------------------------------------------------------------- #
def fig_model_surfaces(rows, surfaces, model, mode, ncol=5, panel_w=2.85):
    """Every perturbable target of one model, on ONE dose axis."""
    ts = [r['target'] for r in rows if r['model'] == model]
    if not ts:
        return None
    n = len(ts)
    ncol = min(ncol, n)
    nblk = int(np.ceil(n / ncol))
    dd = np.concatenate([np.asarray(surfaces[(model, t)]['doses'], float) for t in ts])
    ylim = (dd.min(), dd.max())
    shared = all(np.allclose(surfaces[(model, ts[0])]['doses'],
                             surfaces[(model, t)]['doses']) for t in ts)

    # panel_w exists so a SINGLE surface can be rendered big enough to inspect. The default
    # sizes a panel for a multi-column contact sheet, which is unreadable when the question is
    # "what does this one surface actually look like at 256 phases".
    fig, axes = plt.subplots(2 * nblk, ncol, squeeze=False,
                             figsize=(panel_w * ncol, 1.79 * panel_w * nblk),
                             gridspec_kw={'height_ratios': [1.5, 1] * nblk})
    im = None
    for k, t in enumerate(ts):
        b, c = divmod(k, ncol)
        r = _row(rows, model, t)
        ch = surfaces[(model, t)]
        g, lv = GM.gene_of(model, t)
        head = (f"{t}  ({g}, {lv})" if g else t)
        sub = (f"S*={_fmt(r['S_surf'])}  twist={_fmt(r['total_twist'], 2)}"
               f"  span={_fmt(r['span_hi_half'], 2)}")
        im = _surface_panel(axes[2 * b][c], ch, r, ylim=ylim, show_y=(c == 0),
                            title=f"{head}\n{sub}")
        _twist_panel(axes[2 * b + 1][c], ch, r, xlim=ylim, show_y=(c == 0))
    for k in range(n, nblk * ncol):
        b, c = divmod(k, ncol)
        axes[2 * b][c].axis('off')
        axes[2 * b + 1][c].axis('off')

    note = ('shared dose grid' if shared else 'PER-TARGET dose grids -- axes NOT comparable')
    fig.suptitle(f"{model} ({mode}): base PTC surface and fixed-point curve for every "
                 f"perturbable target  [{note}]{NL}"
                 f"twist panels: blue = type-0 (resetting) doses, hatched = no usable "
                 f"winding (integrator unstable or clock stopped)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.915, 0.97])
    _phase_colorbar(fig, im)
    return _save(fig, f"surfaces_{model}_{mode}{_sfx()}.png", src_model=model, mode=mode,
                 source=_CTX['source'],
                 targets=ts, shared_grid=shared)


# --------------------------------------------------------------------------- #
#  2. one figure per GENE, one column per model
# --------------------------------------------------------------------------- #
def fig_gene_across_models(rows, surfaces, gene, mode):
    """The same gene in every model that has it. Own dose axis per model; the shared axis --
    dose/S* -- is the wide bottom panel, and it is the only cross-model dose comparison this
    project will make."""
    cols = [(m, t, lv) for (m, t, lv) in GM.pairs(gene) if (m, t) in surfaces]
    if len(cols) < 1:
        return None
    ncol = len(cols)
    fig = plt.figure(figsize=(2.95 * ncol + 0.9, 8.6))
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.5, 1.0, 1.25], hspace=0.45, wspace=0.30,
                          left=0.07, right=0.90, top=0.90, bottom=0.075)
    im = None
    for c, (m, t, lv) in enumerate(cols):
        r = _row(rows, m, t)
        ch = surfaces[(m, t)]
        ax = fig.add_subplot(gs[0, c])
        im = _surface_panel(ax, ch, r, show_y=True,
                            title=f"{m}  {t}  ({lv})\nS*={_fmt(r['S_surf'])}  "
                                  f"twist={_fmt(r['total_twist'], 2)}")
        ax.title.set_color(MODEL_COLOR.get(m, 'k'))
        _twist_panel(fig.add_subplot(gs[1, c]), ch, r, show_y=True)

    ax = fig.add_subplot(gs[2, :])
    ax.axhline(0, color='0.8', lw=0.5, ls=':')
    ax.axhline(1, color='0.8', lw=0.5, ls=':')
    ax.axvline(1.0, color='0.4', ls='--', lw=1.0)
    any_norm = False
    for (m, t, lv) in cols:
        r = _row(rows, m, t)
        S = r['S_scan'] if np.isfinite(r['S_scan']) else r['S_surf']
        if not np.isfinite(S) or S <= 0:
            continue
        ch = surfaces[(m, t)]
        any_norm = True
        ax.plot(*broken(np.asarray(ch['doses'], float) / S, np.asarray(ch['twist'], float)),
                color=MODEL_COLOR.get(m, 'k'), lw=1.6,
                ls={'mrna': '-', 'protein': '--', 'nuclear': ':', 'complex': '-.'}[lv],
                label=f"{m} {t} ({lv})  S*={_fmt(S)}")
    ax.set_xscale('log')
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('dose / S*   (the only cross-model dose axis: each curve is scaled by its '
                  'own transition)', fontsize=9)
    ax.set_ylabel('stable FP phase', fontsize=9)
    ax.tick_params(labelsize=8)
    if any_norm:
        ax.legend(fontsize=7, ncol=min(3, len(cols)), loc='upper left', framealpha=0.9)
    else:
        ax.text(0.5, 0.5, 'no target of this gene has a transition to normalise by',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='0.4')

    fig.suptitle(f"{gene} ({mode}): the same gene across models "
                 f"-- {ncol} target(s) in {len(set(m for m, _t, _l in cols))} model(s)",
                 fontsize=12.5)
    _phase_colorbar(fig, im)
    return _save(fig, f"gene_{gene}_{mode}{_sfx()}.png", gene=gene, mode=mode,
                 source=_CTX['source'],
                 columns=[f'{m}/{t}' for m, t, _l in cols])


# --------------------------------------------------------------------------- #
#  3. the scalar feature comparison
# --------------------------------------------------------------------------- #
#: (column, label, log?, ylim) -- the features worth putting side by side.
FEATURES = (
    ('S_scan',       'S_crit  (scan)',                 True,  None),
    ('S_surf',       'S*  (surface)',                  True,  None),
    ('phi_star',     'phi*  (singularity old phase)',  False, (-0.03, 1.03)),
    ('total_twist',  'total twist (cyc, SATURATES 0.5)', False, (-0.02, 0.52)),
    ('acc_twist',    'accumulated twist (cyc)',        False, None),
    ('signed_twist', 'signed twist (cyc)',             False, None),
    ('span_hi_half', 'old-phase span, upper half (cyc)', False, (-0.02, 0.52)),
    ('type0_frac',   'fraction of doses type-0',       False, (-0.03, 1.03)),
    ('min_amp',      'min relative amplitude',         False, None),
    ('unusable_frac', 'fraction of cells with no phase', False, (-0.03, 1.03)),
)


def _gene_x(rows, genes):
    """x position per gene, and a small deterministic offset per (model, level) so overlapping
    points stay separable without random jitter."""
    keys = sorted({(r['model'], r['level']) for r in rows},
                  key=lambda k: (GM.MODELS.index(k[0]) if k[0] in GM.MODELS else 9,
                                 GM.LEVELS.index(k[1]) if k[1] in GM.LEVELS else 9))
    off = {k: (i - (len(keys) - 1) / 2) * (0.62 / max(len(keys), 1)) for i, k in enumerate(keys)}
    return {g: i for i, g in enumerate(genes)}, off, keys


def _scatter_feature(ax, rows, col, genes, gx, off, log=False, ylim=None):
    for r in rows:
        if r['gene'] not in gx:
            continue
        v = r[col]
        if not np.isfinite(v) or (log and v <= 0):
            continue
        x = gx[r['gene']] + off[(r['model'], r['level'])]
        ax.plot(x, v, marker=LEVEL_MARKER.get(r['level'], 'o'), ms=7, ls='none',
                mfc=(MODEL_COLOR.get(r['model'], 'k') if r['passed'] else 'none'),
                mec=MODEL_COLOR.get(r['model'], 'k'), mew=1.4,
                alpha=(1.0 if r['passed'] else 0.85))
    for i in range(len(genes) - 1):
        ax.axvline(i + 0.5, color='0.9', lw=0.7, zorder=0)
    if log:
        ax.set_yscale('log')
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, fontsize=8, rotation=30, ha='right')
    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.tick_params(labelsize=7)


def _legend_handles(rows):
    ms = [Line2D([], [], color=MODEL_COLOR[m], marker='s', ls='none', ms=7, label=m)
          for m in GM.MODELS if any(r['model'] == m for r in rows)]
    ls = [Line2D([], [], color='0.35', marker=LEVEL_MARKER[l], ls='none', ms=7, label=l)
          for l in GM.LEVELS if any(r['level'] == l for r in rows)]
    return ms + ls + [Line2D([], [], color='0.35', marker='s', ls='none', ms=7, mfc='none',
                             mew=1.4, label='hollow = QC FAIL')]


def fig_features(rows, mode, min_models=1):
    """Every scalar, gene by gene, model by model."""
    genes = [g for g in GM.GENES
             if len({r['model'] for r in rows if r['gene'] == g}) >= min_models]
    if not genes:
        return None
    gx, off, _k = _gene_x(rows, genes)
    n = len(FEATURES)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 2.9 * nrow), squeeze=False)
    for k, (col, lab, log, ylim) in enumerate(FEATURES):
        ax = axes[k // ncol][k % ncol]
        _scatter_feature(ax, rows, col, genes, gx, off, log=log, ylim=ylim)
        ax.set_title(lab, fontsize=9.5)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.legend(handles=_legend_handles(rows), loc='lower right', ncol=2, fontsize=8,
               frameon=True, bbox_to_anchor=(0.995, 0.01))
    fig.suptitle(f"PTC features across models ({mode} mode) -- colour = model, "
                 f"marker = expression level", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return _save(fig, f"features_{mode}{_sfx()}.png", mode=mode, genes=genes,
                 source=_CTX['source'],
                 n_rows=len(rows))


def fig_features_modes(rows_by_mode, min_models=1):
    """The same scalars with pulse and instant SIDE BY SIDE.

    A pulse dose is a rate and an instant dose is a concentration, so the two are not on a
    common axis and this figure never puts them on one -- each mode keeps its own panel. What
    IS comparable is the SHAPE of the answer: which genes reset, in what order, with how much
    twist. Where the two arms disagree about that, the disagreement is about the perturbation's
    duration, not about the clock."""
    modes = [m for m in ('pulse', 'instant') if m in rows_by_mode]
    if len(modes) < 2:
        return None
    allrows = [r for m in modes for r in rows_by_mode[m]]
    genes = [g for g in GM.GENES
             if len({r['model'] for r in allrows if r['gene'] == g}) >= min_models]
    show = [f for f in FEATURES if f[0] in
            ('S_scan', 'total_twist', 'acc_twist', 'span_hi_half', 'type0_frac', 'min_amp')]
    nrow = len(show)
    fig, axes = plt.subplots(nrow, 2, figsize=(11.5, 2.6 * nrow), squeeze=False)
    for i, (col, lab, log, ylim) in enumerate(show):
        for j, md in enumerate(modes):
            rws = rows_by_mode[md]
            gx, off, _k = _gene_x(rws, genes)
            _scatter_feature(axes[i][j], rws, col, genes, gx, off, log=log, ylim=ylim)
            axes[i][j].set_title(f"{lab}  --  {md}", fontsize=9.5)
        if col == 'S_scan':
            # NOT shared. A pulse dose is a rate and an instant dose is a concentration; one
            # axis across the two would assert a comparison the units do not support.
            axes[i][0].set_ylabel('dose (rate)', fontsize=8)
            axes[i][1].set_ylabel('dose (conc.)', fontsize=8)
        else:                       # everything else is a cycle count or a fraction -- share,
            lo = min(a.get_ylim()[0] for a in axes[i])      # so a difference between the two
            hi = max(a.get_ylim()[1] for a in axes[i])      # arms is a difference in the data
            for a in axes[i]:
                a.set_ylim(lo, hi)
    fig.suptitle("PTC features: 8 h pulse vs instant displacement "
                 "(dose units DIFFER between the two -- compare the pattern, not the number)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0.045, 1, 0.965])
    fig.legend(handles=_legend_handles(allrows), loc='lower center', ncol=7, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.002))
    return _save(fig, f"features_modes{_sfx()}.png", modes=modes, genes=genes,
                 source=_CTX['source'])


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description='the cross-model PTC figures (pure read)')
    ap.add_argument('--mode', default='pulse', choices=('pulse', 'instant', 'both'))
    ap.add_argument('--which', default='all',
                    help='surfaces,genes,features,modes or all')
    ap.add_argument('--models', default=','.join(GM.MODELS))
    ap.add_argument('--genes', default=None, help='default: every gene in >=1 model')
    ap.add_argument('--feat-tag', default=None, help='analysis.features run tag')
    ap.add_argument('--phase-tag', default=None, help='analysis.phaseref run tag')
    ap.add_argument('--phase-roll', type=float, default=0.0,
                    help='rotate the old-phase axis by this many cycles FOR DISPLAY, to put '
                         'a feature sitting on the 0/1 wrap into the middle of the panel')
    ap.add_argument('--source', default='surface', choices=('scan', 'surface'),
                    help="which feature table to draw: the wide screen or the refined render")
    ap.add_argument('--tag', default=None, help='output tag for the figures')
    ap.add_argument('--ncol', type=int, default=5)
    ap.add_argument('--panel-w', type=float, default=2.85,
                    help='inches per surface panel; raise it to inspect one surface closely')
    ap.add_argument('--publish', action='store_true')
    a = ap.parse_args(argv)
    from analysis.features import load as load_features

    from analysis.phaseref import load as _load_phaseref
    _off, _pr = _load_phaseref(a.phase_tag)
    _CTX.update(tag=paths.run_tag(a.tag), publish=a.publish, source=a.source,
                anchor=(str(_pr['anchor_gene']) if _pr is not None else None),
                roll=float(a.phase_roll))
    if _pr is None:
        print('[figures_zoo] no common phase origin -- absolute phases are MODEL-LOCAL; '
              'run `$PY -m analysis.phaseref`', flush=True)
    want = (['surfaces', 'genes', 'features', 'modes'] if a.which == 'all'
            else a.which.split(','))
    modes = ('pulse', 'instant') if a.mode == 'both' else (a.mode,)
    models = [m.strip() for m in a.models.split(',') if m.strip()]
    made, rows_by_mode = [], {}

    for md in modes:
        rows, surfaces, _z = load_features(md, a.feat_tag, source=a.source)
        if abs(a.phase_roll) > 1e-9:
            # Reuse the offset machinery rather than a second rotation path: it relabels both
            # phase axes together and leaves every twist measure invariant, which is exactly
            # what a display roll must also do. A feature sitting ON the 0/1 wrap is split
            # across both edges of the panel and reads as two features; rolling it to the
            # middle is the only way to see whether it is one thing or two.
            from analysis.features import apply_phase_offset
            surfaces = {k: apply_phase_offset(v, -a.phase_roll) for k, v in surfaces.items()}
            for r in rows:
                if np.isfinite(r['phi_star']):
                    r['phi_star'] = float((r['phi_star'] + a.phase_roll) % 1.0)
        rows_by_mode[md] = rows
        if 'surfaces' in want:
            for m in models:
                p = fig_model_surfaces(rows, surfaces, m, md, ncol=a.ncol,
                                       panel_w=a.panel_w)
                if p:
                    made.append(p)
        if 'genes' in want:
            genes = (a.genes.split(',') if a.genes else
                     [g for g in GM.GENES if any(r['gene'] == g for r in rows)])
            for g in genes:
                p = fig_gene_across_models(rows, surfaces, g, md)
                if p:
                    made.append(p)
        if 'features' in want:
            p = fig_features(rows, md)
            if p:
                made.append(p)
    if 'modes' in want and len(rows_by_mode) > 1:
        p = fig_features_modes(rows_by_mode)
        if p:
            made.append(p)
    print(f"[figures_zoo] {len(made)} figure(s) -> "
          f"{os.path.relpath(paths.out_dir(ZOO, 'figures', _CTX['tag']), paths.HERE)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
