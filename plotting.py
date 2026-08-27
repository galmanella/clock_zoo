"""
plotting.py
===========
House figure conventions. Every plotter here is a PURE READ of a saved npz -- nothing
integrates, so a figure can always be regenerated without re-running a sweep.

CONVENTIONS (carried over from input_screen)
    Phase is CIRCULAR, so a phase map uses `cmocean.cm.phase`, which wraps. A sequential
    colormap on phase data puts a false discontinuity at the 0/1 wrap and invents structure
    that is not there.
    Overlays are ACHROMATIC so they read against any hue of the cyclic map: white filled
    circle for a +1 winding singularity, white cross for -1, dashed white for S_crit, black
    and bold for the base curve.
    Missing data is neutral grey, not white -- a hole must look like a hole, not like a
    legitimate light-coloured phase.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')                 # headless: save figures, never pop a window
import matplotlib.pyplot as plt

try:
    import cmocean
    PHASE_CMAP = cmocean.cm.phase
except ImportError:                   # cmocean is a soft dependency; hsv also wraps
    PHASE_CMAP = plt.cm.hsv

BAD_COLOR = '0.6'


def phase_cmap():
    """A fresh cyclic colormap with a neutral 'bad' colour (set_bad mutates, so copy)."""
    cm = PHASE_CMAP.copy()
    cm.set_bad(BAD_COLOR)
    return cm


def broken(x, y, thr=0.5):
    """Insert NaN breaks where a wrapped phase jumps the 0/1 boundary, so a line plot does not
    draw fake verticals across the panel. Keeps the natural [0, 1] y-scale.

    Same construction as input_screen/sensitivity_full.py `_broken` (commit e955873) -- the
    alternative, unwrapping onto a continuous lift, loses the [0,1] axis that makes a phase
    readable, and a dense scatter loses the sense of a continuous curve, which for a twist
    curve is the thing being read.
    """
    xo, yo = [], []
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    for k in range(len(y)):
        xo.append(x[k]); yo.append(y[k])
        if (k < len(y) - 1 and np.isfinite(y[k]) and np.isfinite(y[k + 1])
                and abs(y[k + 1] - y[k]) > thr):
            xo.append(np.nan); yo.append(np.nan)
    return np.array(xo), np.array(yo)


def phase_map(ax, old, doses, ptc, title=None, sings=(), scrit=None):
    """A PTC surface: old phase (x) vs dose (y, log), coloured by new phase.

    NO fixed-point overlay. The twist curve used to be drawn on top of the surface in black,
    and it was a bad idea twice over: a line across a cyclic colour field is hard to read at
    all, and it visually competes with the surface it is meant to annotate. It gets its own
    panel now -- see `twist_panel`, which uses the house orientation (dose on x) and so does
    NOT share an axis with this one.
    """
    im = ax.pcolormesh(old, doses, np.ma.masked_invalid(np.asarray(ptc).T),
                       cmap=phase_cmap(), vmin=0, vmax=1, shading='nearest')
    ax.set_yscale('log')
    ax.set_xlabel('old phase'); ax.set_ylabel('dose')
    for s in sings:
        ax.plot(s['phi'], s['dose'], marker=('o' if s['sign'] > 0 else 'x'),
                ms=7, mfc='white', mec='white', mew=1.6, ls='none')
    if scrit is not None and np.isfinite(scrit):
        ax.axhline(scrit, color='white', ls='--', lw=1.2)
    if title:
        ax.set_title(title, fontsize=9)
    return im


def twist_panel(ax, doses, twist, base=None, scrit=None, label=None, color='k', title=None,
                is_base=False, guides=True):
    """The stable fixed point (attracting entrainment phase) vs dose -- the TWIST curve.

    HOUSE ORIENTATION, matching input_screen (`plot_twist_movers`, `plot_screen`):
    **dose on x (log), FP phase on y, wrapped to [0, 1]** with dotted guides at 0 and 1. The
    phase axis stays literal rather than unwrapped, so a value can be read straight off it and
    compared against a PTC's old/new phase; the 0/1 wraps are handled by breaking the line
    (see `broken`) rather than by rescaling the axis.

    `scrit` is marked as a dot ON the curve at the nearest sampled dose -- again the
    input_screen convention -- plus a faint vertical guide, so the singularity dose can be
    located without reading it off a second panel.
    """
    doses = np.asarray(doses, float)
    if guides:
        ax.axhline(0, color='0.7', lw=0.5, ls=':')
        ax.axhline(1, color='0.7', lw=0.5, ls=':')
    if base is not None:
        ax.plot(*broken(doses, np.asarray(base, float)), color='k', lw=2.6, zorder=6,
                label='base')
    ax.plot(*broken(doses, np.asarray(twist, float)),
            color=('k' if is_base else color), lw=(2.6 if is_base else 1.3),
            zorder=(6 if is_base else 3), label=label)
    if scrit is not None and np.isfinite(scrit):
        ax.axvline(scrit, color='0.55', ls='--', lw=0.9, zorder=1)
        tw = np.asarray(twist, float)
        k = int(np.argmin(np.abs(doses - scrit)))
        if np.isfinite(tw[k]):
            ax.plot([doses[k]], [tw[k]], marker='o', ms=5,
                    mfc=('white' if is_base or base is not None else color),
                    mec='black', mew=0.8, zorder=7)
    ax.set_xscale('log')
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('dose'); ax.set_ylabel('stable FP phase')
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def scrit_overview(blob, path=None):
    """One panel per target: winding vs dose, with the validity ceiling marked."""
    targets = [str(t) for t in blob['targets']]
    n = len(targets)
    ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow), squeeze=False)
    for k, t in enumerate(targets):
        ax = axes[k // ncol][k % ncol]
        d = blob.get(f'doses__{t}'); W = blob.get(f'W__{t}')
        if d is None:
            ax.axis('off'); continue
        fin = np.isfinite(W)
        ax.step(np.asarray(d)[fin], np.abs(np.asarray(W)[fin]), where='mid', color='k', lw=1.4)
        S = float(blob['S_crit'][k])
        if np.isfinite(S):
            ax.axvline(S, color='tab:red', ls='--', lw=1.2)
        c = float(blob['d_max_valid'][k])
        if np.isfinite(c) and c < np.asarray(d).max():
            ax.axvspan(c, np.asarray(d).max(), color='0.85', zorder=0)
        ax.set_xscale('log'); ax.set_ylim(-0.2, 1.4); ax.set_yticks([0, 1])
        ax.set_title(f"{t}  S*={'-' if not np.isfinite(S) else f'{S:.3g}'}", fontsize=9)
        ax.set_xlabel('dose'); ax.set_ylabel('|winding|')
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(f"{blob['model']} ({blob['mode']}): PTC type vs dose "
                 f"(grey = beyond the trustworthy range)", fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig)
    return fig


def coupling_scatter(blob, path=None):
    """LC sensitivity vs PTC sensitivity, one point per parameter, with the quadrant lines.

    The point of the figure is which quadrant is populated -- specifically whether the
    low-LC / high-PTC corner (top left) has anything in it, since that is where PTC data
    would add identifiability the limit cycle cannot."""
    x = np.asarray(blob['lc_sens'], float)
    y = np.asarray(blob['ptc_sens'], float)
    names = [str(p) for p in blob['params']]
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(x[ok], y[ok], s=42, c='tab:blue', edgecolors='k', linewidths=0.5, zorder=3)
    for i in np.where(ok)[0]:
        ax.annotate(names[i], (x[i], y[i]), fontsize=6.5, xytext=(3, 3),
                    textcoords='offset points')
    xm, ym = np.median(x[ok]), np.median(y[ok])
    ax.axvline(xm, color='0.6', lw=0.8, ls=':'); ax.axhline(ym, color='0.6', lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('LC sensitivity  (phase-aligned cycle + period)')
    ax.set_ylabel(f"PTC sensitivity  ({blob['feature']})")
    ax.set_title(f"{blob['model']} / {blob['target']} ({blob['mode']}): "
                 f"log-log r = {float(blob['corr']):+.3f}", fontsize=10)
    # name the corner that matters
    ax.annotate('PTC adds what the LC cannot', xy=(0.02, 0.97), xycoords='axes fraction',
                fontsize=8, color='tab:red', va='top')
    ax.annotate('coupled', xy=(0.98, 0.97), xycoords='axes fraction', fontsize=8,
                color='0.4', ha='right', va='top')
    ax.annotate('sloppy in both', xy=(0.02, 0.03), xycoords='axes fraction', fontsize=8,
                color='0.4', va='bottom')
    ax.annotate('LC-only (useless decoupling)', xy=(0.98, 0.03), xycoords='axes fraction',
                fontsize=8, color='0.4', ha='right', va='bottom')
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig)
    return fig


def sensitivity_tornado(names, values, title, xlabel, path=None, top=30, color='steelblue'):
    """Horizontal bar chart of the top movers."""
    v = np.nan_to_num(np.asarray(values, float))
    order = np.argsort(v)[-top:]
    fig, ax = plt.subplots(figsize=(6.4, max(3.5, 0.22 * len(order))))
    ax.barh([str(names[i]) for i in order], v[order], color=color)
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig)
    return fig
