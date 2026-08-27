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


def phase_map(ax, old, doses, ptc, title=None, sings=(), scrit=None, twist=None):
    """A PTC surface: old phase (x) vs dose (y, log), coloured by new phase."""
    im = ax.pcolormesh(old, doses, np.ma.masked_invalid(np.asarray(ptc).T),
                       cmap=phase_cmap(), vmin=0, vmax=1, shading='nearest')
    ax.set_yscale('log')
    ax.set_xlabel('old phase'); ax.set_ylabel('dose')
    for s in sings:
        ax.plot(s['phi'], s['dose'], marker=('o' if s['sign'] > 0 else 'x'),
                ms=7, mfc='white', mec='white', mew=1.6, ls='none')
    if scrit is not None and np.isfinite(scrit):
        ax.axhline(scrit, color='white', ls='--', lw=1.2)
    if twist is not None:
        ax.plot(twist, doses, color='black', lw=2.0)
    if title:
        ax.set_title(title, fontsize=9)
    return im


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
