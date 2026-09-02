"""
R01_sup_page/panels.py
======================
PUBLICATION PANELS for the R01 supplementary page. One page, so every millimetre is spent
deliberately.

    $PY -m R01_sup_page.panels --which bmal [--width-cm 10 --height-cm 4]

HOW THIS DIFFERS FROM analysis/figures_zoo.py, AND WHY IT IS A SEPARATE FILE
    `figures_zoo` draws WORKING figures: every annotation that helps a reader decide whether
    to believe the result -- provenance stamped into the image, quality verdicts, twist panels,
    winding bands, feature values in the titles. All of that is right for a working figure and
    wrong for a printed panel, where the caption carries the context and the stamp would be an
    artefact on the page.

    So this module is presentation only. It reads exactly the same saved arrays -- it derives
    nothing, computes nothing, and cannot disagree with the working figures about a number.

PROVENANCE WITHOUT A STAMP
    A publication figure must not carry a provenance footer, but it still has to be traceable.
    Each output therefore gets a `<name>.source.txt` naming the script, the git commit, and the
    exact data tags it was built from. Same guarantee as `paths.save_figure`'s sidecar, minus
    the ink.

SIZE
    Given in CENTIMETRES because that is how a page is measured. The default is a 3-panel row
    at 10 x 4 cm; `--width-cm/--height-cm` exist because the right aspect ratio for a row of
    square-ish phase/dose maps is found by looking, not by arithmetic.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting import phase_cmap                                       # noqa: E402
from analysis.figures_zoo import _mesh_edges, _log_dose               # noqa: E402
from analysis.features import load as load_features                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CM = 1 / 2.54

# Arial, and TEXT THAT STAYS TEXT IN THE SVG. `svg.fonttype='none'` writes glyphs as font
# references instead of outlines, so the labels remain selectable and editable in
# Illustrator/Inkscape -- which is the point of shipping an SVG alongside the PNG. The
# cost is that whoever opens it needs Arial installed; the PNG is the fallback.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'axes.unicode_minus': False,
})

#: Display name per model. The panel title is the model and nothing else.
MODEL_LABEL = {'almeida': 'Almeida', 'korencic': 'Korencic', 'goldbeter': 'Goldbeter'}

#: Type sizes, in points, for a panel a few centimetres across.
FS_TITLE, FS_LABEL, FS_TICK = 7.5, 7.0, 6.0


def _panel(ax, ch, S=None, show_y=True, title=None, mark=True, last=True):
    """One PTC surface. No feature values, no verdicts -- the caption carries those."""
    old, doses = np.asarray(ch['old'], float), np.asarray(ch['doses'], float)
    ptc = np.asarray(ch['ptc'], float)
    wrapped = np.vstack([ptc[-1:], ptc, ptc[:1]])
    xe, ye = _mesh_edges(old, doses)
    im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(wrapped.T), cmap=phase_cmap(),
                       vmin=0, vmax=1, shading='flat', rasterized=True)
    ax.set_yscale('log' if _log_dose(doses) else 'linear')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(ye[0], ye[-1])
    if mark:
        for phi, d, sg in zip(ch['sing_phi'], ch['sing_dose'], ch['sing_sign']):
            ax.plot(phi, d, marker=('o' if sg > 0 else 'x'), ms=3.2, mfc='white',
                    mec='white', mew=0.9, ls='none')
        if S is not None and np.isfinite(S):
            ax.axhline(S, color='white', ls=(0, (3, 2)), lw=0.7)
    # EVERY panel labels the full 0 / 0.5 / 1.0. Dropping the 1.0 was the wrong fix for
    # abutting labels -- the cause is that a CENTRED label at a panel edge hangs half its
    # width outside the panel, so anchoring the end labels inward (below) solves it without
    # costing the reader a tick.
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=FS_TICK, length=2, pad=1.5, width=0.6)
    # Keep every tick label INSIDE its own panel. A centred label at x=0 hangs half its width
    # into the neighbouring panel's gap, which is how abutting panels end up with labels
    # touching however few ticks each one draws. Anchoring the end labels inward is the fix
    # that survives any panel count or gap.
    lbls = ax.get_xticklabels()
    if lbls:
        lbls[0].set_horizontalalignment('left')
        lbls[-1].set_horizontalalignment('right')
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    if not show_y:
        ax.set_yticklabels([])
    if title:
        ax.set_title(title, fontsize=FS_TITLE, pad=2.5)
    return im


def fig_bmal(sources, width_cm=10.0, height_cm=4.0, cbar=True):
    """The Bmal1 cross-model row: one surface per model, shared dose axis.

    `sources` is [(model, surface_dict, S_crit)] in draw order.
    """
    n = len(sources)
    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
    # hand-placed axes: at this size tight_layout's padding is a large fraction of the panel
    left, right = 0.085, (0.885 if cbar else 0.995)
    bottom, top = 0.215, 0.875
    gap = 0.028
    w = (right - left - gap * (n - 1)) / n
    im = None
    for i, (model, ch, S) in enumerate(sources):
        ax = fig.add_axes([left + i * (w + gap), bottom, w, top - bottom])
        im = _panel(ax, ch, S=S, show_y=(i == 0), last=(i == n - 1),
                    title=MODEL_LABEL.get(model, model))
    if cbar:
        cax = fig.add_axes([right + 0.022, bottom, 0.018, top - bottom])
        cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
        cb.set_label('New Phase (cyc)', fontsize=FS_LABEL, labelpad=1.5)
        cb.ax.tick_params(labelsize=FS_TICK, length=2, pad=1.2, width=0.6)
        cb.outline.set_linewidth(0.6)
    fig.text((left + right) / 2, 0.045, 'Old Phase (cyc)', ha='center',
             fontsize=FS_LABEL)
    fig.text(0.012, (bottom + top) / 2, 'Dose (a.u.)', va='center', rotation='vertical',
             fontsize=FS_LABEL)
    return fig


# --------------------------------------------------------------------------- #
def _git(*a):
    try:
        return subprocess.check_output(['git', *a], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'nogit'


def save(fig, name, dpi=600, note=''):
    """PNG + SVG, plus a sidecar so a clean figure is still traceable."""
    os.makedirs(HERE, exist_ok=True)
    out = []
    for ext in ('png', 'svg'):
        p = os.path.join(HERE, f'{name}.{ext}')
        fig.savefig(p, dpi=dpi, format=ext,
                    bbox_inches=None)          # exact size: the layout is already explicit
        out.append(p)
    with open(os.path.join(HERE, f'{name}.source.txt'), 'w') as f:
        f.write(f"{name}.png / .svg\n"
                f"script   R01_sup_page/panels.py\n"
                f"commit   {_git('rev-parse', '--short', 'HEAD')}"
                f"{' (DIRTY)' if _git('status', '--porcelain', '-uno') else ''}\n"
                f"built    {datetime.now().isoformat(timespec='seconds')}\n"
                f"size     {fig.get_size_inches()[0] / CM:.2f} x "
                f"{fig.get_size_inches()[1] / CM:.2f} cm at {dpi} dpi\n"
                f"{note}\n")
    plt.close(fig)
    for p in out:
        print(f"[panel] -> {os.path.relpath(p, os.path.dirname(HERE))}", flush=True)
    return out


BMAL = (('almeida', 'BMAL1'), ('korencic', 'Bmalx'), ('goldbeter', 'MB'))


def build_bmal(mode='instant', tags=None, width_cm=10.0, height_cm=4.0, dpi=600,
               name=None, no_cbar=False):
    """Collect the three Bmal1 surfaces and draw them. `tags` maps model -> feature-table tag,
    so a panel can mix a re-rendered model with the rest."""
    tags = tags or {}
    cache, srcs, note = {}, [], []
    for model, target in BMAL:
        t = tags.get(model, tags.get('*'))
        if t not in cache:
            cache[t] = load_features(mode, t, source='surface')
        rows, surfaces, _z = cache[t]
        if (model, target) not in surfaces:
            print(f"[panel] {model}/{target} missing from tag {t!r} -- skipping",
                  file=sys.stderr)
            continue
        r = next((x for x in rows if x['model'] == model and x['target'] == target), None)
        srcs.append((model, surfaces[(model, target)], r['S_surf'] if r else np.nan))
        nph = np.asarray(surfaces[(model, target)]['ptc']).shape[0]
        note.append(f"  {model:10s} {target:7s} tag={t}  n_phase={nph}  "
                    f"n_dose={len(surfaces[(model, target)]['doses'])}  "
                    f"S*={r['S_surf'] if r else float('nan'):.4g}")
    fig = fig_bmal(srcs, width_cm, height_cm, cbar=not no_cbar)
    return save(fig, name or f'fig1_bmal_{mode}', dpi=dpi,
                note='data:\n' + '\n'.join(note))


def main(argv=None):
    ap = argparse.ArgumentParser(description='publication panels for the R01 sup page')
    ap.add_argument('--which', default='bmal')
    ap.add_argument('--mode', default='instant', choices=('pulse', 'instant'))
    ap.add_argument('--tag', default=None, help='feature-table tag for every model')
    ap.add_argument('--tag-map', default=None,
                    help='per-model override, e.g. korencic=korph256')
    ap.add_argument('--width-cm', type=float, default=10.0)
    ap.add_argument('--height-cm', type=float, default=4.0)
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--name', default=None)
    ap.add_argument('--no-cbar', action='store_true')
    a = ap.parse_args(argv)
    tags = {'*': a.tag}
    if a.tag_map:
        tags.update(dict(p.split('=', 1) for p in a.tag_map.split(',')))
    if a.which == 'bmal':
        build_bmal(a.mode, tags, a.width_cm, a.height_cm, a.dpi, a.name, a.no_cbar)
    else:
        raise SystemExit(f"unknown --which {a.which!r}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
