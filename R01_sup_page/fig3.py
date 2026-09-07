"""
R01_sup_page/fig3.py
====================
FIGURE 3, PANEL A: which directions in parameter space move the PTC without moving the
limit cycle. Four styles, because the right one depends on a judgement about scale that the
data cannot settle on its own.

    $PY -m R01_sup_page.fig3 --npz out/almeida/coupling/coupling_BMAL1_instant_twist.npz

WHY rho = 1 IS MEANINGFUL HERE (corrected)
    rho(v) = ||J_PTC v||^2 / ||J_LC v||^2 with BOTH JACOBIANS SCALED TO UNIT SPECTRAL NORM
    first -- `analysis/coupling.decoupling_spectrum` does this before solving the pencil. That
    scaling is the answer to "can rescaling fix the units": each experiment is measured
    against its OWN best-resolved direction, so rho = 1 means "as visible to the PTC as to the
    limit cycle, each relative to the most that experiment can see". It is a real reference,
    not a convention, and the diagonal is drawn.

    The inputs are already dimensionless and gauge-clean before that: `lc_sens` divides every
    species by its own peak-to-trough on the base cycle and stores the period as dT/T, a PTC
    is measured in cycles, and `coupling` projects OUT the gauge before the eigenproblem
    (without which a pure gauge motion -- exactly LC-null while still moving a
    fixed-absolute-dose PTC -- would read as perfect decoupling; REPO_MAP hazard 6). The
    `J_LC`/`J_PTC` stored in the npz are ALREADY gauge-projected; only the spectral scaling is
    applied here, and `_normalised()` is checked against the stored `dir_lc_resp`,
    `dir_ptc_resp` and `rho` so this file cannot silently drift from the analysis.

    The strongest claims are still ratios of ratios, where even the spectral choice cancels:
        best combination / best single parameter    = 122.3 / 1.64 = 75x   (linear, pulse)
        decoupled / coupled, by finite displacement = 18.1  / 1.64 ~ 11x   (measured)
    Those disagree by ~7x -- the linear rho overstates, and PROJECT_SUMMARY 3.6 says to quote
    the finite-displacement number. So rho ranks directions; it does not size the effect.

    'cloud'      spectral-normalised axes; both the rho = 1 diagonal and the best single
                 parameter are drawn.
    'raw'        absolute ||J v||, log-log, no reference line -- shows the unscaled geometry.
    'spectrum'   one axis: rho sorted, against rho = 1 and the best single parameter.
    'frobenius'  the same plot under FROBENIUS rather than spectral scaling, as a robustness
                 check: if the ranking survives the choice of norm, it is not an artefact
                 of it.

THE CLOUD IS A SAMPLE OF DIRECTIONS, NOT OF PARAMETER SETS
    It is drawn uniformly on the unit sphere of the retained subspace, so its density is a
    statement about the geometry of the two jacobians -- not about which parameter sets are
    biologically plausible. It is there to show that decoupling is EXTREMAL rather than
    typical, which is the whole reason an eigenproblem is needed to find it.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R01_sup_page.panels import CM, FS_LABEL, FS_TICK, save        # noqa: E402

RHO1 = '$' + chr(92) + 'rho=1$'
RHOEQ = '$' + chr(92) + 'rho$='

#: Random unit directions in the background cloud. Free: two matrix-vector products each.
N_CLOUD = 4000


def _normalised(cp, norm=2):
    """(An, Bn) -- the PTC and LC jacobians on the scaling the analysis actually used.

    `coupling.py` stores J_LC/J_PTC ALREADY gauge-projected, then scales each to unit spectral
    norm inside `decoupling_spectrum`. Reproducing that scaling here is what makes the axes of
    this figure the same quantity as the stored rho; getting it wrong silently rescales rho by
    (||J_LC||/||J_PTC||)^2, which for Almeida/BMAL1 is a factor of 3.2 -- large enough to
    change the headline and small enough to look plausible. Hence `check_against_stored`.

    norm=2 is the analysis default; norm='fro' is offered only as a robustness check."""
    A, B = np.nan_to_num(np.asarray(cp['J_PTC'], float)), np.nan_to_num(np.asarray(cp['J_LC'], float))
    return (A / max(np.linalg.norm(A, norm), 1e-300),
            B / max(np.linalg.norm(B, norm), 1e-300))


def check_against_stored(cp, rtol=1e-9):
    """Reproduce the stored per-direction responses and rho, or raise.

    Cheap insurance that this figure is rendering the analysis rather than a near-miss of it."""
    An, Bn = _normalised(cp)
    V = np.asarray(cp['V'])
    lc, ptc = np.linalg.norm(Bn @ V, axis=0), np.linalg.norm(An @ V, axis=0)
    for nm, got, ref in (('dir_lc_resp', lc, cp['dir_lc_resp']),
                         ('dir_ptc_resp', ptc, cp['dir_ptc_resp']),
                         ('rho', ptc ** 2 / np.maximum(lc ** 2, 1e-300), cp['rho'])):
        d = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))
        if not d < rtol:
            raise AssertionError(f'{nm} disagrees with the stored analysis: max rel diff {d:.3e}')
    return True


def decoupling_points(cp, n_cloud=N_CLOUD, seed=0, norm=2):
    """(cloud, eigen-directions, single parameters), each as lc / ptc / rho arrays.

    Pure read -- nothing integrates. `rho` and `axis_rho` are taken from the npz when the
    scaling is the analysis default, so the figure quotes the analysis rather than a
    re-derivation of it."""
    An, Bn = _normalised(cp, norm)
    V = np.asarray(cp['V'])                       # (n_param, k), unit-norm columns
    stock = (norm == 2)
    if stock:
        check_against_stored(cp)

    def resp(M):
        return np.linalg.norm(Bn @ M, axis=0), np.linalg.norm(An @ M, axis=0)

    rng = np.random.default_rng(seed)
    # Sample INSIDE the retained subspace. Drawing in the full parameter space would mix in
    # the directions J_LC cannot resolve -- exactly what `lc_floor` excludes -- and those land
    # in the decoupled corner for a numerical reason rather than a physical one, manufacturing
    # the result the figure exists to test.
    G = rng.normal(size=(V.shape[1], n_cloud))
    C = V @ (G / np.linalg.norm(G, axis=0, keepdims=True))
    C /= np.linalg.norm(C, axis=0, keepdims=True)
    cl, dv, pr = resp(C), resp(V), resp(np.eye(An.shape[1]))
    ratio = lambda t: (t[1] ** 2) / np.maximum(t[0] ** 2, 1e-300)
    return (dict(lc=cl[0], ptc=cl[1], rho=ratio(cl)),
            dict(lc=dv[0], ptc=dv[1], rho=np.asarray(cp['rho']) if stock else ratio(dv)),
            dict(lc=pr[0], ptc=pr[1],
                 rho=np.asarray(cp['axis_rho']) if stock else ratio(pr),
                 names=[str(x) for x in cp['params']]))


def fig_decoupling(cp, style='cloud', width_cm=10.0, height_cm=6.0, n_cloud=N_CLOUD,
                   n_label=3, cloud=True):
    norm = 'fro' if style == 'frobenius' else 2
    show_cloud = cloud
    cloud, dirs, pars = decoupling_points(cp, n_cloud, norm=norm)
    best_par = float(np.max(pars['rho']))
    best_nm = pars['names'][int(np.argmax(pars['rho']))]
    fig, ax = plt.subplots(figsize=(width_cm * CM, height_cm * CM))

    if style == 'spectrum':
        r = np.sort(dirs['rho'])[::-1]
        ax.plot(np.arange(1, len(r) + 1), r, 'o-', ms=4, lw=1.2, color='0.25',
                label='eigen-directions')
        ax.axhline(best_par, color='tab:blue', ls='--', lw=1.0)
        ax.axhline(1.0, color='0.45', ls=':', lw=1.0)
        # opposite corners: stacked at the same x the two labels touch, and the dashed line
        # runs straight through the lower one
        ax.text(len(r), best_par * 1.2, f'best single parameter ({best_nm})',
                fontsize=FS_TICK, color='tab:blue', va='bottom', ha='right')
        ax.text(1, 1 / 1.2, 'equally visible to both', fontsize=FS_TICK, color='0.45',
                va='top', ha='left')
        ax.set_yscale('log')
        ax.set_xlabel(r'direction (sorted by $\rho$)', fontsize=FS_LABEL)
        ax.set_ylabel(r'$\rho$  =  PTC response$^2$ / LC response$^2$', fontsize=FS_LABEL)
        ax.legend(fontsize=FS_TICK, loc='upper right', framealpha=0.9)
    else:
        if style == 'raw':
            # undo the scaling: absolute ||J v||, so the unscaled geometry is visible
            nl = float(np.linalg.norm(np.asarray(cp['J_LC']), 2))
            npt = float(np.linalg.norm(np.asarray(cp['J_PTC']), 2))
            xl, yl = r'LC response  $\|J_{LC}v\|$', r'PTC response  $\|J_{PTC}v\|$'
        else:
            nl = npt = 1.0
            tag = 'Frobenius' if style == 'frobenius' else 'spectral'
            xl = rf'LC response  (/ $\|J_{{LC}}\|$, {tag})'
            yl = rf'PTC response  (/ $\|J_{{PTC}}\|$, {tag})'
        X, Y = lambda d: d['lc'] * nl, lambda d: d['ptc'] * npt
        if show_cloud:
            ax.scatter(X(cloud), Y(cloud), s=1.5, c='0.78', lw=0, rasterized=True,
                       label=f'random directions (n={n_cloud})')
        ax.scatter(X(pars), Y(pars), s=16, marker='o', facecolor='none',
                   edgecolor='tab:blue', lw=0.9, label='single parameters')
        sc = ax.scatter(X(dirs), Y(dirs), s=30, marker='D',
                        c=np.log10(np.maximum(dirs['rho'], 1e-12)), cmap='viridis',
                        edgecolor='k', lw=0.5, zorder=5, label='eigen-directions')
        sc.set_label(None)
        from matplotlib.lines import Line2D
        handles = ([Line2D([], [], ls='', marker='.', ms=3, color='0.7',
                           label=f'random directions (n={n_cloud})')] if show_cloud else []) +                   [Line2D([], [], ls='', marker='o', mfc='none', mec='tab:blue', ms=4,
                          label='single parameters'),
                   Line2D([], [], ls='', marker='D', mfc='0.55', mec='k', mew=0.4, ms=4,
                          label='eigen-directions')]
        cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.045)
        cb.set_label(r'$\log_{10}\rho$', fontsize=FS_LABEL)
        cb.ax.tick_params(labelsize=FS_TICK)
        # Limits from the DATA, per axis. Deriving a single [lo, hi] for both axes and
        # forcing it on x collapses the x range onto the (much wider) y range and the whole
        # cloud lands on the frame edge.
        shown = ([cloud] if show_cloud else []) + [pars, dirs]
        xs = np.concatenate([X(d) for d in shown])
        ys = np.concatenate([Y(d) for d in shown])
        x0, x1 = xs.min() / 1.7, xs.max() * 1.7
        y0, y1 = ys.min() / 1.7, ys.max() * 2.2

        if style != 'raw':
            # rho = 1: equally visible to both experiments, each against its own best-resolved
            # direction. Meaningful only because both jacobians carry the SAME scaling.
            d = np.array([min(x0, y0), max(x1, y1)])
            for m, c, ls, txt, fr in ((1.0, '0.45', ':', RHO1, 0.45),
                                      (np.sqrt(best_par), 'tab:blue', '--',
                                       f'best single ({best_nm})', 0.97)):
                ax.plot(d, d * m, color=c, ls=ls, lw=0.9, zorder=1)
                # Anchor each label a different fraction along its own line. Both lines exit
                # through the same corner, so a shared anchor stacks the two labels on top of
                # each other and they read as one string.
                xe = min(x1, y1 / m)
                xa = x0 * (xe / x0) ** fr
                ax.annotate(txt, (xa, xa * m), fontsize=FS_TICK, color=c, ha='right',
                            va='bottom', rotation=45, rotation_mode='anchor',
                            xytext=(0, 2), textcoords='offset points')
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        # Label the most decoupled directions ABOVE their own marker. Side offsets put the
        # text over whichever neighbouring diamond happened to be there -- with 14 directions
        # the left group is dense, and the third label landed on an unlabelled point.
        for j in np.argsort(dirs['rho'])[::-1][:n_label]:
            ax.annotate(RHOEQ + f"{dirs['rho'][j]:.3g}", (X(dirs)[j], Y(dirs)[j]),
                        fontsize=FS_TICK, ha='center', va='bottom',
                        xytext=(0, 7), textcoords='offset points')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xl, fontsize=FS_LABEL)
        ax.set_ylabel(yl, fontsize=FS_LABEL)
        ax.legend(handles=handles, fontsize=FS_TICK, loc='upper left', framealpha=0.9,
                  handletextpad=0.4, borderpad=0.35, labelspacing=0.3)
    ax.tick_params(labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    fig.tight_layout(pad=0.3)
    return fig


STYLES = ('cloud', 'raw', 'spectrum', 'frobenius')


def build(npz, styles=STYLES, width_cm=10.0, height_cm=6.0, dpi=600, prefix='fig3A',
          cloud=True):
    cp = dict(np.load(npz, allow_pickle=True))
    out = []
    for st in styles:
        fig = fig_decoupling(cp, style=st, width_cm=width_cm, height_cm=height_cm,
                             cloud=cloud)
        out += save(fig, f'{prefix}_{st}', dpi=dpi,
                    note=(f"data:\n  {npz}\n"
                          f"  {str(cp['target'])} / {str(cp['mode'])}, "
                          f"feature {str(cp['feature'])}\n"
                          f"  {int(cp['n_kept_lc'])} directions at lc_floor="
                          f"{float(cp['lc_floor']):.0e}, rho "
                          f"{cp['rho'].min():.4g}..{cp['rho'].max():.4g}\n"
                          f"  best single parameter rho = "
                          f"{float(np.max(decoupling_points(cp)[2]['rho'])):.4g}"),
                    script='R01_sup_page/fig3.py')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='figure 3 panel A, in four styles')
    ap.add_argument('--npz', required=True)
    ap.add_argument('--styles', default=','.join(STYLES))
    ap.add_argument('--width-cm', type=float, default=10.0)
    ap.add_argument('--height-cm', type=float, default=6.0)
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--prefix', default='fig3A')
    ap.add_argument('--no-cloud', action='store_true',
                    help='drop the random-direction background')
    a = ap.parse_args(argv)
    build(a.npz, tuple(a.styles.split(',')), a.width_cm, a.height_cm, a.dpi,
          a.prefix, cloud=not a.no_cloud)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
