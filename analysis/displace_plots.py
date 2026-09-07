"""
analysis/displace_plots.py
==========================
Draw a `displace_demo` run: PTC surfaces relative to base, and the limit cycle against base.

    $PY -m analysis.displace_plots --model almeida --target BMAL1 --mode instant --fold 2

PURE READ of `displace_<target>_<mode>_x<fold>.npz`. Nothing integrates.

THE POINT OF PUTTING THESE TWO SIDE BY SIDE
    A decoupled direction has to do two things at once, and either one alone is easy to fake.
    The PTC panel must show a large, dose-ordered change; the limit-cycle panel must show the
    trajectory sitting essentially on top of base. A direction that fails the second is just a
    big perturbation, and a direction that fails the first is not interesting. The control
    (the lowest measured-ratio direction) is included precisely so the reader can see the
    opposite pattern -- cycle visibly displaced, PTC barely moved.

CONVENTIONS
    dPTC uses the SHORTEST SIGNED ARC, not a plain subtraction: phase is circular and a raw
    difference invents jumps of a full cycle at the wrap. Diverging colour map, symmetric
    limits, so zero is the neutral colour and sign is readable.

    Cycle traces are normalised per species by its own base peak-to-trough, which is the same
    normalisation `lc_sens` uses for dLC. Without it the large-amplitude species dominate the
    panel and a species that moved a lot relative to its own swing is invisible.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                                        # noqa: E402
from plotting import phase_cmap                                     # noqa: E402

FS = 7.0


def _select(dz, signs=None, dirs=None):
    """Subset the runs. Showing both signs of four directions is eight columns of heatmap,
    which collides the titles and, worse, lets one pathological run set the shared colour
    scale for all of them."""
    keep = np.ones(len(dz['dirs']), bool)
    if signs:
        keep &= np.isin(np.asarray(dz['sign'], int), list(signs))
    if dirs:
        keep &= np.isin(np.asarray(dz['dirs'], int), list(dirs))
    # EXPLICIT whitelist, not "first axis matches the run count". Almeida has 8 states and
    # this run has 8 rows, so a shape test also sliced `base_cycle` down to 4 species and the
    # cycle panel crashed on a state index. Shape is not identity.
    out = dict(dz)
    for k in ('ptc', 'cycles', 'displacements', 'param_values', 'dirs', 'sign', 'rho',
              'eps', 'dLC', 'dT', 'dPTC_rms', 'period', 'n_bad', 'status'):
        if k in dz:
            out[k] = np.asarray(dz[k])[keep]
    return out


def _load(model, target, mode, fold):
    fp = paths.out_path(model, 'coupling', f'displace_{target}_{mode}_x{fold:g}.npz')
    if not os.path.exists(fp):
        raise SystemExit(f"no displace run at {fp}; run analysis.displace_demo first")
    return dict(np.load(fp, allow_pickle=True)), fp


def _edges(c):
    """Cell edges from centres, so pcolormesh fills the axes instead of leaving a margin."""
    c = np.asarray(c, float)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    d = np.diff(c)
    return np.concatenate([[c[0] - d[0] / 2], c[:-1] + d / 2, [c[-1] + d[-1] / 2]])


def _darc(p, p0):
    """Shortest signed arc p - p0, in cycles."""
    return (np.asarray(p, float) - np.asarray(p0, float) + 0.5) % 1.0 - 0.5


def decompose(dz):
    """Split dPTC into a RIGID SLIDE and a SHAPE change, per run.

    A large rms says the PTC moved; it does not say the isochrons deformed. If the whole
    surface shifts by a constant, the PTC has slid -- which is what a change in the phase
    REFERENCE does -- and an experiment that aligns on the reporter peak would not see it.
    Removing one offset per dose leaves only the change in the shape of the curve at fixed
    dose, which is the part that says the isochron geometry actually changed.

    Returns dict of arrays: total, uniform (the global slide), resid (after it),
    shape (after a per-dose slide), and shape_frac = shape / total."""
    P, P0 = np.asarray(dz['ptc'], float), np.asarray(dz['base_ptc'], float)
    cmean = lambda a: float(np.angle(np.nanmean(np.exp(2j * np.pi * a))) / (2 * np.pi))
    tot, uni, res, shp = [], [], [], []
    for i in range(len(P)):
        D = _darc(P[i], P0)
        tot.append(np.sqrt(np.nanmean(D ** 2)))
        u = cmean(D)
        uni.append(u)
        res.append(np.sqrt(np.nanmean((((D - u) + 0.5) % 1.0 - 0.5) ** 2)))
        pd = np.array([(((D[:, k] - cmean(D[:, k])) + 0.5) % 1.0 - 0.5)
                       for k in range(D.shape[1])]).T
        shp.append(np.sqrt(np.nanmean(pd ** 2)))
    tot, shp = np.array(tot), np.array(shp)
    return dict(total=tot, uniform=np.array(uni), resid=np.array(res), shape=shp,
                shape_frac=shp / np.maximum(tot, 1e-30))


def fig_decompose(dz, width=15.0, height=7.0):
    """Rigid slide vs shape change, against the limit-cycle cost of each direction."""
    dc = decompose(dz)
    dirs, sgn = np.asarray(dz['dirs'], int), np.asarray(dz['sign'], int)
    dlc = np.asarray(dz['dLC'], float)
    lab = [f"dir {d:02d}\n({s:+d})" for d, s in zip(dirs, sgn)]
    x = np.arange(len(dirs))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(width / 2.54, height / 2.54))
    a1.bar(x, dc['shape'], width=0.62, color='tab:blue', label='shape change (per-dose)')
    a1.bar(x, dc['total'] - dc['shape'], width=0.62, bottom=dc['shape'], color='0.8',
           label='rigid slide')
    for i in x:
        a1.text(i, dc['total'][i], f"{100 * dc['shape_frac'][i]:.0f}%", ha='center',
                va='bottom', fontsize=FS - 1.5)
    a1.set_xticks(x)
    a1.set_xticklabels(lab, fontsize=FS - 1.5)
    a1.set_ylabel('dPTC (cyc, rms)', fontsize=FS)
    a1.legend(fontsize=FS - 1.5)
    a1.tick_params(labelsize=FS - 1)
    a1.set_title('what the PTC change actually is', fontsize=FS)

    a2.scatter(dlc, dc['total'], s=34, color='0.6', label='total dPTC', zorder=3)
    a2.scatter(dlc, dc['shape'], s=44, color='tab:blue', marker='D', label='shape only',
               zorder=4)
    for i in x:
        a2.annotate(f"{dirs[i]:02d}", (dlc[i], dc['shape'][i]), fontsize=FS - 1.5,
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
    a2.set_xscale('log')
    a2.set_yscale('log')
    # default log ticks over less than a decade print every minor tick and they collide
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    for axis in (a2.xaxis, a2.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    a2.set_xlabel('dLC', fontsize=FS)
    a2.set_ylabel('dPTC (cyc, rms)', fontsize=FS)
    a2.legend(fontsize=FS - 1.5)
    a2.tick_params(labelsize=FS - 1)
    a2.grid(alpha=0.25, lw=0.4)
    a2.set_title('decoupling, before and after removing the slide', fontsize=FS)
    fig.tight_layout(pad=0.4)
    return fig


def fig_surfaces(dz, half=None, panel=4.2, absolute=False):
    """Row 1: the displaced PTC surface. Row 2: its change from base."""
    P, P0 = np.asarray(dz['ptc'], float), np.asarray(dz['base_ptc'], float)
    doses, old = np.asarray(dz['doses'], float), np.asarray(dz['old'], float)
    dirs, sgn = np.asarray(dz['dirs'], int), np.asarray(dz['sign'], int)
    rho, dlc, dptc = (np.asarray(dz[k], float) for k in ('rho', 'dLC', 'dPTC_rms'))
    n = len(P)
    D = np.array([_darc(P[i], P0) for i in range(n)])
    if half is None:
        # 99th percentile, not the max: a single blown-up run (a direction whose period
        # doubles at this step) otherwise sets the scale for every panel and flattens the
        # differences the figure exists to show.
        half = float(np.nanpercentile(np.abs(D), 99))
        half = min(max(half, 0.05), 0.5)
    xe, ye = _edges(old), _edges(doses)

    nrow = 3 if absolute else 2
    fig, axes = plt.subplots(nrow, n, figsize=(n * panel / 2.54, nrow * panel / 2.54),
                             squeeze=False)
    for i in range(n):
        ax = axes[0][i]
        im0 = ax.pcolormesh(xe, ye, np.ma.masked_invalid(P[i].T), cmap=phase_cmap(),
                            vmin=0, vmax=1, shading='flat', rasterized=True)
        ax.set_title(f"dir {dirs[i]:02d} ({sgn[i]:+d})\n"
                     + r'$\rho$=' + f"{rho[i]:.3g}\n"
                     f"dLC={dlc[i]:.4f}\ndPTC={dptc[i]:.4f}", fontsize=FS - 0.5)
        ax2 = axes[1][i]
        im1 = ax2.pcolormesh(xe, ye, np.ma.masked_invalid(D[i].T), cmap='RdBu_r',
                             vmin=-half, vmax=half, shading='flat', rasterized=True)
        for a in (ax, ax2):
            a.set_xlim(xe[0], xe[-1])
            a.set_ylim(ye[0], ye[-1])
            a.tick_params(labelsize=FS - 1)
            if i:
                a.set_yticklabels([])
        ax.set_xticklabels([])
        ax2.set_xlabel('Old Phase (cyc)', fontsize=FS)
    axes[0][0].set_ylabel('Dose (a.u.)', fontsize=FS)
    axes[1][0].set_ylabel('Dose (a.u.)', fontsize=FS)
    cb0 = fig.colorbar(im0, ax=axes[0].tolist(), pad=0.012, fraction=0.03)
    cb0.set_label('new phase (cyc)', fontsize=FS)
    cb1 = fig.colorbar(im1, ax=axes[1].tolist(), pad=0.012, fraction=0.03)
    cb1.set_label(r'$\Delta$ phase vs base (cyc)', fontsize=FS)
    for cb in (cb0, cb1):
        cb.ax.tick_params(labelsize=FS - 1)
    fig.suptitle(f"{str(dz['model'])} / {str(dz['target'])} {str(dz['mode'])}   "
                 f"largest single-parameter change = {float(dz['fold']):g}x",
                 fontsize=FS + 1, y=1.06)
    return fig


def fig_cycles(dz, panel=4.2, ncol=None):
    """Displaced limit cycle over base, every species on its own amplitude scale."""
    C, C0 = np.asarray(dz['cycles'], float), np.asarray(dz['base_cycle'], float)
    dirs, sgn = np.asarray(dz['dirs'], int), np.asarray(dz['sign'], int)
    dlc = np.asarray(dz['dLC'], float)
    dT = np.asarray(dz['dT'], float)
    names = [str(x) for x in dz['state_names']]
    n = len(C)
    ncol = ncol or n
    nrow = int(np.ceil(n / ncol))
    scale = np.maximum(C0.max(1) - C0.min(1), 1e-9)
    ph = np.arange(C0.shape[1]) / C0.shape[1]
    cmap = plt.get_cmap('tab10')

    # taller than wide per panel: eight overlaid traces in a square panel are unreadable,
    # and the point of the figure is whether solid sits on dashed
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * panel / 2.54,
                                                  nrow * panel * 1.15 / 2.54),
                             squeeze=False, sharex=True, sharey=True)
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i >= n:
            ax.axis('off')
            continue
        for j, nm in enumerate(names):
            b = (C0[j] - C0[j].mean()) / scale[j]
            d = (C[i][j] - C0[j].mean()) / scale[j]
            ax.plot(ph, b, lw=0.7, ls='--', color=cmap(j % 10), alpha=0.55)
            ax.plot(ph, d, lw=1.1, color=cmap(j % 10), label=nm if i == 0 else None)
        ax.set_title(f"dir {dirs[i]:02d} ({sgn[i]:+d})\n"
                     f"dLC={dlc[i]:.4f}  dT/T={dT[i]:+.3f}", fontsize=FS)
        ax.tick_params(labelsize=FS - 1)
        ax.grid(alpha=0.2, lw=0.3)
    axes[0][0].legend(fontsize=FS - 2.5, ncol=2, loc='upper right', framealpha=0.85)
    fig.supxlabel('phase (cyc)', fontsize=FS)
    fig.supylabel('state, centred and scaled by its own base amplitude', fontsize=FS)
    fig.suptitle('dashed = base,  solid = displaced', fontsize=FS + 1)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.97))
    return fig


def build(model, target, mode, fold, dpi=220, half=None, signs=(+1,)):
    dz, fp = _load(model, target, mode, fold)
    sel = _select(dz, signs=signs)
    out = []
    for name, fig in (('surfaces', fig_surfaces(sel, half=half)),
                      ('cycles', fig_cycles(sel)),
                      ('surfaces_both', fig_surfaces(dz, half=half, panel=3.6)),
                      ('cycles_both', fig_cycles(dz, panel=3.6)),
                      ('decompose', fig_decompose(sel)),
                      ('decompose_both', fig_decompose(dz))):
        p = paths.out_path(model, 'coupling',
                           f'displace_{name}_{target}_{mode}_x{fold:g}.png')
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        out.append(p)
        print(f"[displace-plots] -> {p}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='plots for a displace_demo run')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    ap.add_argument('--mode', default='instant')
    ap.add_argument('--fold', type=float, default=2.0)
    ap.add_argument('--half', type=float, default=None,
                    help='fix the delta colour range to +/- this (default: from the data)')
    ap.add_argument('--dpi', type=int, default=220)
    ap.add_argument('--signs', default='1',
                    help='which displacement signs in the main figure')
    a = ap.parse_args(argv)
    build(a.model, a.target, a.mode, a.fold, a.dpi, a.half,
          signs=tuple(int(x) for x in a.signs.split(',')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
