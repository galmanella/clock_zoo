"""
tools/decoupling_intuition.py
=============================
TEACHING FIGURE for the decoupling analysis: why the directions are not the principal
components of either response.

    $PY -m tools.decoupling_intuition

Draws a 2-parameter toy, so the geometry that is real but invisible in 16 dimensions can
actually be seen. Nothing here reads project data -- the matrices are invented to make the
point clearly.

WHAT THE THREE PANELS SAY
    (a) Two response "roses" in parameter space: for every direction, how large is the LC
        response and how large is the PTC response. Each has its own principal axes, and they
        are NOT the same axes -- that is the intuition most people arrive with, and it is
        correct as far as it goes.

    (b) The RATIO rho(theta) = (PTC response / LC response)^2. The generalized directions are
        exactly the stationary points of this curve. That is the whole analysis in one line:
        it does not maximise either response, it maximises their ratio, so it cannot be the
        PCA of either matrix.

    (c) The same picture after WHITENING by the LC response. The LC rose becomes a circle --
        every direction now costs the same limit-cycle motion -- and a circle has no preferred
        axes, so the frame is free to diagonalise the PTC instead. In these coordinates the
        generalized directions ARE ordinary principal axes, and they ARE perpendicular. Map
        back to parameter space and the stretch destroys the right angle, which is why they
        look oblique in (a).

The real analysis is this, in 16 gauge-quotiented log-parameter dimensions, with the LC
response measured over the whole phase-aligned cycle plus the period, and the PTC response
over the whole (phase x dose) surface.
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh, cholesky
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FS_T, FS_L, FS_K = 11.0, 10.0, 9.0
C_LC, C_PT, C_GEN = '#1f77b4', '#d62728', '#111111'


def toy():
    """(J_LC, J_PTC) for the toy. The PTC is tilted IN PARAMETER SPACE.

    Rotating the OUTPUT instead leaves J^T J diagonal and every basis trivially coincides --
    a version of this figure built that way showed all three frames at 0/90 deg and quietly
    demonstrated nothing."""
    B = np.diag([3.0, 0.6])                      # LC: stiff along p0, sloppy along p1
    th = np.radians(35.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = np.diag([2.0, 1.2]) @ R                  # PTC: its own shape, its own orientation
    return B, A


def _rose(M, th):
    """Response magnitude ||M v(theta)|| for unit v at each angle."""
    V = np.vstack([np.cos(th), np.sin(th)])
    return np.linalg.norm(M @ V, axis=0)


def _iso(M, th, scale=1.0):
    """The ISO-RESPONSE ellipse {v : ||M v|| = const}, semi-axis 1/sigma_i.

    Plotted instead of a polar plot of ||M v(theta)||: that "response rose" is correct but
    pinches to a waist along the insensitive direction and reads as a peanut, not an ellipse.
    The iso-response contour is a true ellipse and is the standard sloppiness picture -- with
    the convention that the LONG axis is the direction the observable barely notices."""
    _u, sig, vt = np.linalg.svd(M)
    semi = scale / sig
    return (vt.T * semi) @ np.vstack([np.cos(th), np.sin(th)])


def _axes_of(M):
    """Right-singular directions (the principal axes of that response) and their magnitudes."""
    _u, s, vt = np.linalg.svd(M)
    return vt, s


def _line(ax, v, L, **kw):
    ax.plot([-L * v[0], L * v[0]], [-L * v[1], L * v[1]], **kw)


def build(out_dir=None, dpi=220):
    B, A = toy()
    th = np.linspace(0, 2 * np.pi, 721)
    rB, rA = _rose(B, th), _rose(A, th)
    rho_th = (rA / rB) ** 2

    vtB, sB = _axes_of(B)
    vtA, sA = _axes_of(A)
    rho, W = eigh(A.T @ A, B.T @ B)
    o = np.argsort(-rho)
    rho, W = rho[o], W[:, o]
    W = W / np.linalg.norm(W, axis=0)

    # whitening: L L^T = B^T B, u = L^T v  =>  ||B v|| = |u|, so the LC rose becomes a circle
    L = cholesky(B.T @ B, lower=True)
    Winv = np.linalg.inv(L.T)
    Uw = L.T @ W
    Uw = Uw / np.linalg.norm(Uw, axis=0)
    dot = float(abs(Uw[:, 0] @ Uw[:, 1]))
    assert dot < 1e-10, f'whitened directions should be orthogonal, got dot={dot:.2e}'

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5 / 2.54 * 1.35, 5.6 / 2.54 * 1.35))

    # ---- (a) parameter space -------------------------------------------------------- #
    # each ellipse scaled to its own largest semi-axis: what matters here is orientation and
    # shape, not the absolute size of the two responses
    eB, eA = _iso(B, th), _iso(A, th)
    eB, eA = eB / np.abs(eB).max(), eA / np.abs(eA).max()
    a1.plot(eB[0], eB[1], color=C_LC, lw=2.0, label='LC iso-response')
    a1.plot(eA[0], eA[1], color=C_PT, lw=2.0, label='PTC iso-response')
    for i, v in enumerate(vtB):
        _line(a1, v, 3.0, color=C_LC, ls=':', lw=1.2,
              label='PCA of LC' if i == 0 else None)
    for i, v in enumerate(vtA):
        _line(a1, v, 3.0, color=C_PT, ls=':', lw=1.2,
              label='PCA of PTC' if i == 0 else None)
    for i in range(2):
        _line(a1, W[:, i], 3.0, color=C_GEN, ls='-', lw=1.6,
              label='generalized' if i == 0 else None)
    a1.set_title('(a) parameter space', fontsize=FS_T, pad=8)
    a1.set_xlabel('parameter 1', fontsize=FS_L)
    a1.set_ylabel('parameter 2', fontsize=FS_L)
    a1.set_aspect('equal')
    lim1 = 1.15 * max(np.abs(eB).max(), np.abs(eA).max(), 1.0)
    a1.set_xlim(-lim1, lim1)
    a1.set_ylim(-lim1, lim1)


    # ---- (b) the ratio, and its stationary points ------------------------------------ #
    deg = np.degrees(th)
    a2.plot(deg, rho_th, color=C_GEN, lw=1.8)
    for i in range(2):
        d = np.degrees(np.arctan2(W[1, i], W[0, i])) % 180
        for dd in (d, d + 180):
            a2.axvline(dd, color=C_GEN, ls='--', lw=1.0, alpha=0.7)
        a2.annotate(f'$\\rho$={rho[i]:.2f}', (d, rho[i]), fontsize=FS_K,
                    ha='center', va='bottom', xytext=(0, 6),
                    textcoords='offset points')
    a2.axhline(1.0, color='0.55', ls=':', lw=1.0)
    a2.set_yscale('log')
    a2.set_ylim(rho_th.min() / 2.2, rho_th.max() * 2.8)   # headroom for the labels
    a2.set_xlim(0, 360)
    a2.set_xticks([0, 90, 180, 270, 360])
    a2.set_title('(b) the ratio is what is optimised', fontsize=FS_T, pad=8)
    a2.set_xlabel('direction (deg)', fontsize=FS_L)
    a2.set_ylabel(r'$\rho=(\mathrm{PTC}/\mathrm{LC})^2$', fontsize=FS_L)
    a2.grid(alpha=0.25, lw=0.4)

    # ---- (c) after whitening --------------------------------------------------------- #
    eBw, eAw = _iso(B @ Winv, th), _iso(A @ Winv, th)
    sc = np.abs(eBw).max()
    eBw, eAw = eBw / sc, eAw / sc                  # one common scale: the circle sets it
    a3.plot(eBw[0], eBw[1], color=C_LC, lw=2.0, label='LC (now a circle)')
    a3.plot(eAw[0], eAw[1], color=C_PT, lw=2.0, label='PTC iso-response')
    for i in range(2):
        _line(a3, Uw[:, i], 3.0, color=C_GEN, ls='-', lw=1.6,
              label='generalized' if i == 0 else None)
    a3.set_title('(c) whitened by the LC response', fontsize=FS_T, pad=8)
    a3.set_xlabel('whitened 1', fontsize=FS_L)
    a3.set_ylabel('whitened 2', fontsize=FS_L)
    a3.set_aspect('equal')
    lim3 = 1.15 * max(np.abs(eBw).max(), np.abs(eAw).max())
    a3.set_xlim(-lim3, lim3)
    a3.set_ylim(-lim3, lim3)


    for ax in (a1, a2, a3):
        ax.tick_params(labelsize=FS_K)
    h1, l1 = a1.get_legend_handles_labels()
    h3, l3 = a3.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in list(zip(h1, l1)) + list(zip(h3, l3)):
        if l not in seen:
            seen.add(l); H.append(h); L.append(l)
    fig.legend(H, L, fontsize=FS_K, ncol=len(L), loc='lower center',
               bbox_to_anchor=(0.5, -0.02), frameon=False, handlelength=1.8,
               columnspacing=1.6)
    fig.tight_layout(pad=0.9, w_pad=2.0, rect=(0, 0.075, 1, 1))

    out_dir = out_dir or os.path.join('out', 'zoo', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, 'decoupling_intuition.png')
    fig.savefig(p, dpi=dpi, bbox_inches='tight')
    fig.savefig(p.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close(fig)

    ang = lambda v: np.degrees(np.arctan2(v[1], v[0])) % 180
    with open(p + '.source.txt', 'w') as f:
        f.write('decoupling_intuition.png / .svg\n'
                'script   tools/decoupling_intuition.py\n'
                'A TOY, not project data: 2 invented parameters, so the geometry is visible.\n'
                f'  J_LC  = diag(3.0, 0.6)\n'
                f'  J_PTC = diag(2.0, 1.2) @ rot(35 deg)   (tilted in PARAMETER space)\n'
                f'  PCA of LC   axes at {ang(vtB[0]):.1f} and {ang(vtB[1]):.1f} deg\n'
                f'  PCA of PTC  axes at {ang(vtA[0]):.1f} and {ang(vtA[1]):.1f} deg\n'
                f'  generalized axes at {ang(W[:, 0]):.1f} and {ang(W[:, 1]):.1f} deg '
                f'(rho {rho[0]:.3f}, {rho[1]:.3f})\n'
                f'  angle between generalized directions: '
                f'{abs(ang(W[:, 0]) - ang(W[:, 1])):.1f} deg in parameter space, '
                f'90.0 deg after whitening\n')
    print(f'[intuition] -> {p}', flush=True)
    print(f'  PCA(LC)  {ang(vtB[0]):.1f}, {ang(vtB[1]):.1f} deg   '
          f'PCA(PTC) {ang(vtA[0]):.1f}, {ang(vtA[1]):.1f} deg   '
          f'generalized {ang(W[:, 0]):.1f}, {ang(W[:, 1]):.1f} deg', flush=True)
    return p


if __name__ == '__main__':
    raise SystemExit(0 if build() else 1)
