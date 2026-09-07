"""
analysis/confirm_plots.py
=========================
Inspection plots for a finite-displacement run (`analysis/confirm.py --all-dirs`).

    $PY -m analysis.confirm_plots --model almeida --target BMAL1 --mode instant

PURE READ. Everything here comes out of `confirm_<target>_<mode>.npz`; nothing integrates, so
the aesthetics can be iterated without paying for the adaptive engine again.

WHAT THESE ARE FOR
    `analysis/coupling.py` ranks directions by rho, a LINEAR ratio built from two
    finite-difference jacobians. `confirm` takes a real step of size `eps` along each direction
    and measures what actually moves. The question these plots answer is not "is the top
    direction good" but the sharper one: DOES THE MEASURED RESPONSE ORDER WITH rho AT ALL?

      scatter   dLC vs dPTC, the finite-displacement analogue of figure 3 panel A. A decoupled
                direction should sit up and to the LEFT: PTC moves, limit cycle does not.
      ordering  rho against the MEASURED ratio. If rho ranks directions correctly this trends
                upward; if it is flat, rho is sorting noise and the linear headline should be
                retracted in favour of the measured numbers.
      ptc       dPTC against old phase, one line per dose, per direction -- so a large rms can
                be checked for what it actually is. A uniform offset at every phase is a phase
                RESET (the whole PTC slides) and a shape change is something else entirely;
                the rms alone cannot tell them apart.

    Both signs of every direction are plotted. A direction whose +eps and -eps disagree
    strongly is in a non-linear regime at this step size, which is itself a reason to distrust
    the linear rho for it.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                                     # noqa: E402

CTRL = 'stiffest'          # label prefix of the positive control
FS = 7.0


def _load(model, target, mode):
    fp = paths.out_path(model, 'coupling', f'confirm_{target}_{mode}.npz')
    if not os.path.exists(fp):
        raise SystemExit(f"no confirm run at {fp}; run analysis.confirm --all-dirs first")
    return dict(np.load(fp, allow_pickle=True)), fp


def _groups(cf):
    """(labels, rho, dLC, dPTC_rms, sign, is_control) as plain arrays."""
    lab = np.array([str(x) for x in cf['labels']])
    return (lab, np.asarray(cf['rho'], float), np.asarray(cf['dLC'], float),
            np.asarray(cf['dPTC_rms'], float), np.asarray(cf['sign'], int),
            np.array([l.startswith(CTRL) for l in lab]))


def fig_scatter(cf, width=13.0, height=8.5):
    """dLC vs dPTC, coloured by the LINEAR rho. The measured version of panel A."""
    lab, rho, dlc, dptc, sgn, ctl = _groups(cf)
    fig, ax = plt.subplots(figsize=(width / 2.54, height / 2.54))
    ok = ~ctl & np.isfinite(rho)
    sc = ax.scatter(dlc[ok], dptc[ok], c=np.log10(np.maximum(rho[ok], 1e-12)),
                    cmap='viridis', s=42, edgecolor='k', lw=0.5, zorder=4)
    if ctl.any():
        ax.scatter(dlc[ctl], dptc[ctl], marker='s', s=48, facecolor='none',
                   edgecolor='tab:red', lw=1.2, zorder=5, label='stiffest LC (control)')
    cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label(r'$\log_{10}\rho$  (linear prediction)', fontsize=FS)
    cb.ax.tick_params(labelsize=FS - 1)
    # label the extremes of the LINEAR ranking, so the reader can see where rho put them
    for j in list(np.argsort(-np.where(ok, rho, -np.inf))[:2]) + \
             list(np.argsort(np.where(ok, rho, np.inf))[:1]):
        ax.annotate(f"{lab[j]} ({sgn[j]:+d})", (dlc[j], dptc[j]), fontsize=FS - 1,
                    ha='center', va='bottom', xytext=(0, 6), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('limit-cycle change  dLC  (rms, own amplitude units)', fontsize=FS)
    ax.set_ylabel('PTC change  dPTC  (rms, cycles)', fontsize=FS)
    ax.set_title(f"finite displacement, eps = {float(cf['eps']):.3g}   "
                 f"{str(cf['model'])}/{str(cf['target'])} {str(cf['mode'])}", fontsize=FS)
    ax.tick_params(labelsize=FS - 1)
    ax.legend(fontsize=FS - 1, loc='lower right')
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.4)
    return fig


def fig_ordering(cf, width=13.0, height=8.5):
    """The linear rho against the MEASURED dPTC/dLC. Does rho rank anything?"""
    lab, rho, dlc, dptc, sgn, ctl = _groups(cf)
    ok = ~ctl & np.isfinite(rho) & (dlc > 0)
    meas = dptc[ok] / dlc[ok]
    r = rho[ok]
    fig, ax = plt.subplots(figsize=(width / 2.54, height / 2.54))
    for s, mk in ((+1, 'o'), (-1, '^')):
        m = sgn[ok] == s
        ax.scatter(r[m], meas[m], marker=mk, s=40, edgecolor='k', lw=0.5,
                   label=f'{s:+d} eps', zorder=4)
    if np.sum(ok) > 2:
        lr = np.log10(r)
        lm = np.log10(np.maximum(meas, 1e-30))
        sp = float(np.corrcoef(np.argsort(np.argsort(lr)),
                               np.argsort(np.argsort(lm)))[0, 1])
        b, a = np.polyfit(lr, lm, 1)
        xs = np.linspace(lr.min(), lr.max(), 8)
        ax.plot(10 ** xs, 10 ** (a + b * xs), color='0.4', ls='--', lw=0.9, zorder=1)
        ax.set_title(f"does the linear $\\rho$ rank the measured response?   "
                     f"Spearman = {sp:.2f}, slope = {b:.2f}", fontsize=FS)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\rho$  (linear, from the jacobians)', fontsize=FS)
    ax.set_ylabel('measured  dPTC / dLC', fontsize=FS)
    ax.tick_params(labelsize=FS - 1)
    ax.legend(fontsize=FS - 1)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.4)
    return fig


def fig_ptc(cf, ncol=6, panel=3.6, sharey=True):
    """dPTC vs old phase, one line per dose, one panel per (direction, sign).

    An rms is a single number over a surface; this is the surface. A rigid vertical offset at
    every phase means the whole PTC slid (a phase reset) rather than deformed, and those are
    different claims about the isochrons."""
    lab, rho, dlc, dptc, sgn, ctl = _groups(cf)
    G = np.asarray(cf['ptc_grids'], float)          # (n_run, n_dose, n_phase)
    B = np.asarray(cf['base_new'], float)           # (n_dose, n_phase)
    doses = np.asarray(cf['doses'], float)
    n = G.shape[0]
    nrow = int(np.ceil(n / ncol))
    # sharey=True puts every panel on the control's scale, which is 10x the eigen-directions
    # and flattens all of them to a line; sharey=False shows each direction's own shape but
    # makes the panels non-comparable. Both are produced, because each hides what the other
    # shows.
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * panel / 2.54, nrow * panel / 2.54),
                             squeeze=False, sharex=True, sharey=sharey)
    old = np.arange(B.shape[1]) / B.shape[1]
    cmap = plt.get_cmap('viridis')
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i >= n:
            ax.axis('off')
            continue
        for k, d in enumerate(doses):
            # shortest signed arc: the PTC is a circular variable, so a plain difference
            # manufactures jumps of a full cycle at the wrap
            dd = (G[i, k] - B[k] + 0.5) % 1.0 - 0.5
            ax.plot(old, dd, lw=0.9, color=cmap(k / max(len(doses) - 1, 1)),
                    label=f'{d:g}' if i == 0 else None)
        ax.axhline(0, color='0.6', lw=0.5, zorder=0)
        ax.set_title(f"{lab[i]} ({sgn[i]:+d})\n"
                     + (r'$\rho$=' + f"{rho[i]:.3g}" if np.isfinite(rho[i]) else 'control')
                     + f"  dLC={dlc[i]:.3f}", fontsize=FS - 1.5)
        ax.tick_params(labelsize=FS - 2)
        ax.grid(alpha=0.2, lw=0.3)
    axes[0][0].legend(fontsize=FS - 2.5, title='dose', title_fontsize=FS - 2.5, ncol=2,
                      loc='best')
    fig.supxlabel('old phase (cyc)', fontsize=FS)
    fig.supylabel('dPTC (cyc, shortest arc)', fontsize=FS)
    fig.tight_layout(pad=0.4)
    return fig


def build(model, target, mode, dpi=200):
    cf, fp = _load(model, target, mode)
    out = []
    for name, fn in (('scatter', fig_scatter), ('ordering', fig_ordering),
                     ('ptc', fig_ptc), ('ptc_free', lambda c: fig_ptc(c, sharey=False))):
        fig = fn(cf)
        p = paths.out_path(model, 'coupling', f'confirm_{name}_{target}_{mode}.png')
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        out.append(p)
        print(f"[confirm-plots] -> {p}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='inspection plots for a confirm run')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    ap.add_argument('--mode', default='instant')
    ap.add_argument('--dpi', type=int, default=200)
    a = ap.parse_args(argv)
    build(a.model, a.target, a.mode, a.dpi)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
