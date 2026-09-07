"""
R01_sup_page/fig3_final.py
==========================
FIGURE 3, publication layout: one row, three panels.

    (a) every decoupling direction, the top few highlighted
    (b) the PTC change from base along one direction
    (c) that direction's limit cycle, over base

    $PY -m R01_sup_page.fig3_final --direction 2 --sign 1

PURE READ of the coupling and displace npz files. Nothing integrates.

SPACE
    Panel (a) drops its colourbar: with the top directions highlighted in a single colour the
    bar carries no information the markers do not, and it costs a whole column of width on a
    three-panel row. Panel (b) keeps one, because a signed delta is unreadable without a
    scale. The delta range is fixed to the same +/-0.5 as figure 2 so the two delta panels on
    the page are directly comparable.

WHAT PANEL (b) IS NOT
    It is the TOTAL change, and for this direction most of it is a RIGID SLIDE of the whole
    surface rather than a deformation of the isochrons -- measured at 95% slide / 34% shape
    for dir 02 (`analysis/displace_plots.decompose`). A rigid offset moves with the reporter
    peak, so an experiment aligning on that reporter would not see most of what this panel
    shows. The honest caption number is the shape component, not the rms of this panel.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                                          # noqa: E402
from R01_sup_page.panels import CM, save                               # noqa: E402
from R01_sup_page.fig3 import decoupling_points                        # noqa: E402
from analysis.displace_plots import decompose                          # noqa: E402

#: Type sizes. LOCAL, not panels.py's -- raising the shared constants would silently restyle
#: figures 1 and 2, which are already settled. The old values (7.5 / 7.0 / 6.0, with legends
#: at 5.0-5.5) were unreadable at print size; nothing here goes below 7 pt.
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = 9.0, 8.5, 7.5, 7.0

#: Delta colour range, matched to figure 2 so both delta panels on the page read alike.
DELTA_HALF = 0.5

#: How many directions to highlight in panel (a).
N_TOP = 3


def _edges(old, doses):
    """Cell edges for pcolormesh: phase wrapped at BOTH ends, dose on LINEAR midpoints.

    `figures_zoo._mesh_edges` does the phase wrap correctly but takes GEOMETRIC midpoints in
    dose, which is right for the log screens and wrong here -- this grid is linear and starts
    at dose 0, where a geometric midpoint is undefined. The both-ends phase wrap is kept for
    the reason it was added there: it makes the mesh span [0, 1] whatever the phase offset,
    so no sliver of blank axis shows between the y-axis and the surface."""
    o, d = np.asarray(old, float), np.asarray(doses, float)
    dx = np.diff(o).mean() if len(o) > 1 else 1.0
    xe = np.concatenate([[o[0] - 1.5 * dx], o - dx / 2,
                         [o[-1] + dx / 2, o[-1] + 1.5 * dx]])
    if len(d) > 1:
        dd = np.diff(d)
        ye = np.concatenate([[d[0] - dd[0] / 2], d[:-1] + dd / 2, [d[-1] + dd[-1] / 2]])
    else:
        ye = np.array([d[0] - 0.5, d[0] + 0.5])
    return xe, ye


def _dz_row(dz, d, sign=1):
    m = (np.asarray(dz['dirs'], int) == d) & (np.asarray(dz['sign'], int) == sign)
    if not m.any():
        raise SystemExit(f'direction {d} sign {sign:+d} is not in this displace run')
    return int(np.where(m)[0][0])


def _panel_scatter(ax, cp, show_dir=None):
    _cloud, dirs, pars = decoupling_points(cp, n_cloud=1)
    best = float(np.max(pars['rho']))
    order = np.argsort(dirs['rho'])[::-1]
    top = set(int(i) for i in order[:N_TOP])
    rest = [i for i in range(len(dirs['rho'])) if i not in top]
    t = sorted(top)

    ax.scatter(pars['lc'], pars['ptc'], s=11, marker='o', facecolor='none',
               edgecolor='0.55', lw=0.7, label='single params')
    ax.scatter(dirs['lc'][rest], dirs['ptc'][rest], s=17, marker='D', facecolor='0.78',
               edgecolor='0.35', lw=0.4, label='directions')
    ax.scatter(dirs['lc'][t], dirs['ptc'][t], s=27, marker='D', facecolor='tab:red',
               edgecolor='k', lw=0.5, zorder=5, label='top ' + str(N_TOP))
    if show_dir is not None and show_dir < len(dirs['rho']):
        ax.annotate('dir ' + format(show_dir, '02d'),
                    (dirs['lc'][show_dir], dirs['ptc'][show_dir]), fontsize=FS_LEG,
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

    xs = np.concatenate([dirs['lc'], pars['lc']])
    ys = np.concatenate([dirs['ptc'], pars['ptc']])
    x0, x1 = xs.min() / 1.8, xs.max() * 1.8
    y0, y1 = ys.min() / 1.8, ys.max() * 2.2
    d = np.array([min(x0, y0), max(x1, y1)])
    ax.plot(d, d, color='0.5', ls=':', lw=0.8, zorder=1)                   # rho = 1
    ax.plot(d, d * np.sqrt(best), color='tab:blue', ls='--', lw=0.8, zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('LC response', fontsize=FS_LABEL, labelpad=1.5)
    ax.set_ylabel('PTC response', fontsize=FS_LABEL, labelpad=1.5)
    ax.legend(fontsize=FS_LEG, loc='upper left', framealpha=0.85,
              handletextpad=0.25, borderpad=0.22, labelspacing=0.16, borderaxespad=0.15)


def _panel_dptc(ax, dz, row, half=DELTA_HALF):
    P = np.asarray(dz['ptc'], float)[row]
    P0 = np.asarray(dz['base_ptc'], float)
    old, doses = np.asarray(dz['old'], float), np.asarray(dz['doses'], float)
    D = (P - P0 + 0.5) % 1.0 - 0.5
    # wrap the periodic phase column at BOTH ends, so the surface meets the y-axis instead of
    # leaving the thin gap figure 1 had
    Dw = np.vstack([D[-1:], D, D[:1]])
    xe, ye = _edges(old, doses)
    im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(Dw.T), cmap='RdBu_r',
                       vmin=-half, vmax=half, shading='flat', rasterized=True)
    # the wrapped columns exist so the mesh COVERS [0, 1]; the frame is still [0, 1], not the
    # padded edges, or the panel shows a sliver of the duplicated column at each side
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(float(doses.min()), float(doses.max()))
    ax.set_xlabel('Old Phase (cyc)', fontsize=FS_LABEL, labelpad=1.5)
    ax.set_ylabel('Dose (a.u.)', fontsize=FS_LABEL, labelpad=1.5)
    return im


def _panel_cycle(ax, dz, row):
    C = np.asarray(dz['cycles'], float)[row]
    C0 = np.asarray(dz['base_cycle'], float)
    names = [str(x) for x in dz['state_names']]
    scale = np.maximum(C0.max(1) - C0.min(1), 1e-9)
    ph = np.append(np.arange(C0.shape[1]) / C0.shape[1], 1.0)
    cmap = plt.get_cmap('tab10')
    for j, nm in enumerate(names):
        b = (C0[j] - C0[j].mean()) / scale[j]
        d = (C[j] - C0[j].mean()) / scale[j]
        ax.plot(ph, np.append(b, b[0]), lw=0.6, ls='--', color=cmap(j % 10), alpha=0.55)
        ax.plot(ph, np.append(d, d[0]), lw=1.0, color=cmap(j % 10), label=nm)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Phase (cyc)', fontsize=FS_LABEL, labelpad=1.5)
    ax.set_ylabel('State (norm.)', fontsize=FS_LABEL, labelpad=1.5)
    ax.legend(fontsize=FS_LEG, ncol=4, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), frameon=False, handlelength=0.9,
              handletextpad=0.3, labelspacing=0.25, columnspacing=0.8)


def fig3_final(cp, dz, direction=2, sign=1, width_cm=17.4, height_cm=5.2,
               half=DELTA_HALF):
    row = _dz_row(dz, direction, sign)
    # LAYOUT IN CENTIMETRES, converted to figure fractions at the end. Fractions hide what
    # a gap is actually for; in cm it is obvious that the a-b gap only has to hold one y-axis
    # label while the bar-c gap holds colourbar ticks plus (c)'s label. Widths are also then
    # stable if the figure is rescaled.
    L_MARGIN = 1.20        # (a)'s y-axis label + ticks
    W_A = 4.20
    G_AB = 1.10            # just (b)'s "Dose (a.u.)" and its ticks
    W_B = 3.20
    G_BCB = 0.14           # heatmap to colourbar
    W_CB = 0.20
    G_CBC = 1.55           # colourbar ticks + (c)'s y-axis label
    W_C = 4.60
    R_MARGIN = 0.20
    width_cm = (L_MARGIN + W_A + G_AB + W_B + G_BCB + W_CB + G_CBC + W_C + R_MARGIN)

    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
    f = lambda cm: cm / width_cm                    # cm -> figure fraction
    TOP, BOT = 0.875, 0.255
    # (c) is shorter than the others on purpose: its legend goes UNDERNEATH, which costs
    # height that was empty anyway and saves the width an in-panel legend would take.
    C_BOT = 0.455
    x_a = f(L_MARGIN)
    x_b = f(L_MARGIN + W_A + G_AB)
    x_cb = f(L_MARGIN + W_A + G_AB + W_B + G_BCB)
    x_c = f(L_MARGIN + W_A + G_AB + W_B + G_BCB + W_CB + G_CBC)
    axa = fig.add_axes([x_a, BOT, f(W_A), TOP - BOT])
    axb = fig.add_axes([x_b, BOT, f(W_B), TOP - BOT])
    cax = fig.add_axes([x_cb, BOT, f(W_CB), TOP - BOT])
    axc = fig.add_axes([x_c, C_BOT, f(W_C), TOP - C_BOT])

    _panel_scatter(axa, cp, show_dir=direction)
    im = _panel_dptc(axb, dz, row, half)
    _panel_cycle(axc, dz, row)

    cb = fig.colorbar(im, cax=cax)
    cax.set_title(r'$\Delta$ phase' + chr(10) + '(cyc)', fontsize=FS_TICK, pad=3)
    cb.set_ticks([-half, 0, half])
    cb.ax.tick_params(labelsize=FS_TICK, pad=1)

    for ax in (axa, axb, axc):
        ax.tick_params(labelsize=FS_TICK, pad=1.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    # One shared y in FIGURE coordinates. Placing these at a fixed offset in each axes'
    # own coordinates drops (c)'s letter below the other two, because (c) is shorter.
    for x, letter in ((x_a, 'a'), (x_b, 'b'), (x_c, 'c')):
        fig.text(x - 0.014, 0.975, letter, fontsize=FS_TITLE + 1, fontweight='bold',
                 ha='left', va='top')
    fold = float(dz['fold'])
    axb.set_title('dir ' + format(direction, '02d') + ', ' + format(fold, 'g') + 'x change',
                  fontsize=FS_TITLE, pad=2)
    axc.set_title('dashed: base', fontsize=FS_TITLE, pad=2)
    return fig


def build(coupling_npz, displace_npz, direction=2, sign=1, width_cm=17.4, height_cm=5.2,
          dpi=600, name='fig3', half=DELTA_HALF):
    cp = dict(np.load(coupling_npz, allow_pickle=True))
    dz = dict(np.load(displace_npz, allow_pickle=True))
    row = _dz_row(dz, direction, sign)
    dc = decompose(dz)
    fig = fig3_final(cp, dz, direction, sign, width_cm, height_cm, half)
    nl = chr(10)
    note = ('data:' + nl
            + '  ' + str(coupling_npz) + nl
            + '  ' + str(displace_npz) + nl
            + 'panel a: ' + str(int(cp['n_kept_lc'])) + ' directions at lc_floor='
            + format(float(cp['lc_floor']), '.0e') + '; rho '
            + format(float(cp['rho'].min()), '.3g') + '..'
            + format(float(cp['rho'].max()), '.4g') + '; best single parameter rho = '
            + format(float(cp['axis_rho'].max()), '.4g') + nl
            + 'panels b,c: dir ' + format(direction, '02d') + ' sign '
            + format(sign, '+d') + ', largest single-parameter change '
            + format(float(dz['fold']), 'g') + 'x' + nl
            + '  dLC=' + format(float(dz['dLC'][row]), '.4f')
            + '  dT/T=' + format(float(dz['dT'][row]), '+.4f')
            + '  dPTC_rms=' + format(float(dz['dPTC_rms'][row]), '.4f') + nl
            + '  of which rigid slide=' + format(float(dc['uniform'][row]), '+.4f')
            + ', shape only=' + format(float(dc['shape'][row]), '.4f')
            + ' (' + format(100 * float(dc['shape_frac'][row]), '.0f') + '% of total)' + nl
            + '  delta colour range +/-' + format(half, 'g'))
    return save(fig, name, dpi=dpi, note=note, script='R01_sup_page/fig3_final.py')


def main(argv=None):
    ap = argparse.ArgumentParser(description='figure 3, final layout')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    ap.add_argument('--mode', default='instant')
    ap.add_argument('--feature', default='twist')
    ap.add_argument('--fold', type=float, default=2.0)
    ap.add_argument('--direction', type=int, default=2)
    ap.add_argument('--sign', type=int, default=1)
    ap.add_argument('--width-cm', type=float, default=17.4)
    ap.add_argument('--height-cm', type=float, default=5.2)
    ap.add_argument('--half', type=float, default=DELTA_HALF)
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--name', default='fig3')
    a = ap.parse_args(argv)
    cpz = paths.out_path(a.model, 'coupling',
                         'coupling_' + a.target + '_' + a.mode + '_' + a.feature + '.npz')
    dzz = paths.out_path(a.model, 'coupling',
                         'displace_' + a.target + '_' + a.mode + '_x'
                         + format(a.fold, 'g') + '.npz')
    build(cpz, dzz, a.direction, a.sign, a.width_cm, a.height_cm, a.dpi, a.name, a.half)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
#  SIMPLIFIED figure 3: (a) limit cycle, (b) and (c) two PTC surfaces
# --------------------------------------------------------------------------- #
def _panel_ptc(ax, ptc, old, doses, show_y=True):
    """One PTC surface on the shared 0..1 phase colour scale."""
    from plotting import phase_cmap
    P = np.asarray(ptc, float)
    Pw = np.vstack([P[-1:], P, P[:1]])          # wrap the periodic column at BOTH ends
    xe, ye = _edges(old, doses)
    im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(Pw.T), cmap=phase_cmap(),
                       vmin=0.0, vmax=1.0, shading='flat', rasterized=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(float(doses.min()), float(doses.max()))
    ax.set_xlabel('Old Phase (cyc)', fontsize=FS_LABEL, labelpad=1.5)
    if show_y:
        ax.set_ylabel('Dose (a.u.)', fontsize=FS_LABEL, labelpad=1.5)
    else:
        ax.set_yticklabels([])
    return im


def _panel_cycle_multi(ax, dz, rows, styles, style_labels):
    """Base plus one or two displaced cycles, each species on its own base amplitude.

    TWO legends: colour = species (solid swatches), line style = which parameter set. Putting
    the style meaning in the panel title instead only works while there are two styles; with
    base, 2x and 0.5x on one axis a title cannot say which is which."""
    from matplotlib.lines import Line2D
    C0 = np.asarray(dz['base_cycle'], float)
    C = np.asarray(dz['cycles'], float)
    names = [str(x) for x in dz['state_names']]
    scale = np.maximum(C0.max(1) - C0.min(1), 1e-9)
    ph = np.append(np.arange(C0.shape[1]) / C0.shape[1], 1.0)
    cmap = plt.get_cmap('tab10')
    close = lambda y: np.append(y, y[0])
    for j, nm in enumerate(names):
        b = (C0[j] - C0[j].mean()) / scale[j]
        ax.plot(ph, close(b), lw=0.6, ls='--', color=cmap(j % 10), alpha=0.55)
        for i, (r, st) in enumerate(zip(rows, styles)):
            d = (C[r][j] - C0[j].mean()) / scale[j]
            # label the SOLID line so the colour legend shows solid swatches
            ax.plot(ph, close(d), color=cmap(j % 10),
                    label=(nm if i == 0 else None), **st)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Phase (cyc)', fontsize=FS_LABEL, labelpad=1.5)
    ax.set_ylabel('State (norm.)', fontsize=FS_LABEL, labelpad=1.5)

    col = ax.legend(fontsize=FS_LEG, ncol=4, loc='upper center',
                    bbox_to_anchor=(0.5, -0.235), frameon=False, handlelength=1.1,
                    handletextpad=0.35, labelspacing=0.22, columnspacing=0.8)
    ax.add_artist(col)
    hs = [Line2D([], [], color='k', lw=0.8, ls='--', alpha=0.7, label='base')]
    for st, lab in zip(styles, style_labels):
        hs.append(Line2D([], [], color='k', label=lab,
                         **{k: v for k, v in st.items() if k in ('lw', 'ls')}))
    ax.legend(handles=hs, fontsize=FS_LEG + 2.0, ncol=len(hs), loc='upper center',
              bbox_to_anchor=(0.5, -0.68), frameon=False, handlelength=1.8,
              handletextpad=0.4, columnspacing=1.3)


def fig3_simple(dz, direction=2, sign=1, left='base', height_cm=6.0):
    """(a) the limit cycle, (b) and (c) two PTC surfaces -- no scatter, no delta panel.

    `left='base'`   -> (b) is the unperturbed PTC.
    `left='invert'` -> (b) is the displacement the OTHER way. Because the step is scaled so the
                       largest single parameter changes by exactly `fold`, the opposite sign
                       changes that same parameter by 1/fold -- so the two panels are honestly
                       titled "2x" and "0.5x" rather than "opposite".
    """
    r_pos = _dz_row(dz, direction, sign)
    r_neg = _dz_row(dz, direction, -sign) if left == 'invert' else None
    old, doses = np.asarray(dz['old'], float), np.asarray(dz['doses'], float)
    fold = float(dz['fold'])

    # PTC panels narrower than tall: they were near-square and the row was wider than it
    # needed to be. G_AB carries (b)'s "Dose (a.u.)" label and was crowding panel (a).
    L_MARGIN, W_A, G_AB, W_B, G_BC, W_C, G_CB, W_CB, R_MARGIN =         1.20, 4.60, 1.60, 2.85, 0.75, 2.85, 0.14, 0.20, 1.30
    width_cm = (L_MARGIN + W_A + G_AB + W_B + G_BC + W_C + G_CB + W_CB + R_MARGIN)
    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
    f = lambda cm: cm / width_cm
    TOP, BOT, A_BOT = 0.885, 0.165, 0.470        # (a) is shorter: two legends sit underneath
    x_a = f(L_MARGIN)
    x_b = f(L_MARGIN + W_A + G_AB)
    x_c = f(L_MARGIN + W_A + G_AB + W_B + G_BC)
    x_cb = f(L_MARGIN + W_A + G_AB + W_B + G_BC + W_C + G_CB)
    axa = fig.add_axes([x_a, A_BOT, f(W_A), TOP - A_BOT])
    axb = fig.add_axes([x_b, BOT, f(W_B), TOP - BOT])
    axc = fig.add_axes([x_c, BOT, f(W_C), TOP - BOT])
    cax = fig.add_axes([x_cb, BOT, f(W_CB), TOP - BOT])

    fs, inv = format(fold, 'g'), format(1.0 / fold, 'g')
    if left == 'invert':
        rows = [r_pos, r_neg]
        styles = [dict(lw=1.0, ls='-'), dict(lw=0.9, ls=':')]
        style_labels = [f'{fs}x', f'{inv}x']
        t_b = f'{inv}x'
    else:
        rows, styles, style_labels = [r_pos], [dict(lw=1.0, ls='-')], [f'{fs}x']
        t_b = 'base'
    _panel_cycle_multi(axa, dz, rows, styles, style_labels)

    P = np.asarray(dz['ptc'], float)
    left_ptc = np.asarray(dz['base_ptc'], float) if left == 'base' else P[r_neg]
    _panel_ptc(axb, left_ptc, old, doses, show_y=True)
    im = _panel_ptc(axc, P[r_pos], old, doses, show_y=False)
    # 0.0 / 0.5 / 1.0 on both, as elsewhere; G_BC above is sized so the adjacent '1.0' and
    # '0.0' do not overlap
    for ax in (axb, axc):
        ax.set_xticks([0.0, 0.5, 1.0])

    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=FS_TICK, pad=1)
    cb.set_label('New Phase (cyc)', fontsize=FS_LABEL, labelpad=2)

    axb.set_title(t_b, fontsize=FS_TITLE, pad=2)
    axc.set_title(f'{fs}x', fontsize=FS_TITLE, pad=2)
    for ax in (axa, axb, axc):
        ax.tick_params(labelsize=FS_TICK, pad=1.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    for x, letter in ((x_a, 'a'), (x_b, 'b'), (x_c, 'c')):
        fig.text(x - 0.013, 0.975, letter, fontsize=FS_TITLE + 1, fontweight='bold',
                 ha='left', va='top')
    return fig


def build_simple(displace_npz, direction=2, sign=1, left='base', height_cm=6.0, dpi=600,
                 name=None):
    dz = dict(np.load(displace_npz, allow_pickle=True))
    r = _dz_row(dz, direction, sign)
    fig = fig3_simple(dz, direction, sign, left, height_cm)
    nl = chr(10)
    note = ('data:' + nl + '  ' + str(displace_npz) + nl
            + '(a) limit cycle, (b) ' + ('base PTC' if left == 'base'
                                         else 'PTC at 1/fold, the inverse step')
            + ', (c) PTC displaced' + nl
            + '  dir ' + format(direction, '02d') + ' sign ' + format(sign, '+d')
            + ', largest single-parameter change ' + format(float(dz['fold']), 'g') + 'x' + nl
            + '  dLC=' + format(float(dz['dLC'][r]), '.4f')
            + '  dT/T=' + format(float(dz['dT'][r]), '+.4f')
            + '  dPTC_rms=' + format(float(dz['dPTC_rms'][r]), '.4f'))
    return save(fig, name or 'fig3_simple', dpi=dpi, note=note,
                script='R01_sup_page/fig3_final.py')
