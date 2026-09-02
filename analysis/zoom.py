"""
analysis/zoom.py
================
A HIGH-RESOLUTION PATCH of the (old phase, dose) plane, and the one test that says whether a
phase singularity is inside it.

    $PY -m analysis.zoom --model korencic --target Bmalx --mode instant \\
        --phase 0.9,0.1 --dose 15,30 --n-phase 128 --n-dose 96

WHY A PATCH AND NOT A FINER FULL SURFACE
    Because the rest of the surface is already resolved. `analysis.characterize` at 256 phases
    over the whole circle costs 256 x 97 points to answer a question that lives in a few
    percent of the plane; the same budget spent on a window around the defect buys ~20x the
    linear resolution where it matters.

THE TEST: WINDING AROUND A CLOSED LOOP
    A phase singularity is a topological defect, and the definition is not "the colours look
    like a pinwheel" -- it is that the new phase advances by a NON-ZERO INTEGER when you walk
    once around any closed loop enclosing it. So this module walks the rectangular boundary of
    the patch, sums the shortest-arc phase differences between consecutive boundary samples,
    and reports the total.

        +/-1  a genuine singularity is enclosed
           0  there is none, however steep the phase gradient inside looks

    That distinction is exactly what a steep-but-continuous phase ridge cannot fake, and it is
    why this is the honest way to settle "is that a defect or a contour crossing the 0/1 wrap".
    The loop sum is also self-checking: if any consecutive pair on the boundary differs by more
    than ~0.25 cyc the loop is under-sampled and the integer is not to be believed, so
    `max_step` is reported next to it.

PHASE WINDOWS WRAP
    `--phase 0.9,0.1` means the arc from 0.9 up through 1.0 == 0.0 to 0.1, which is the case
    that matters when the feature of interest is sitting ON the wrap. The window is given in
    the ANCHORED frame (the one the figures show, 0 = the anchor gene's peak) and converted to
    the model's raw phase internally, so the numbers you read off a figure are the numbers you
    type here.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analysis  # noqa: F401  -- repo root on sys.path
import paths
from analysis.features import ZOO
from plotting import phase_cmap

#: Above this, consecutive boundary samples are too far apart for the loop sum to mean anything.
LOOP_STEP_WARN = 0.25


def _arc(lo, hi, n):
    """`n` samples along the phase arc lo -> hi, going upward and wrapping if hi < lo."""
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi += 1.0
    return np.linspace(lo, hi, int(n)) % 1.0, (hi - lo)


def loop_winding(ptc):
    """Total phase advance around the boundary of the patch, in cycles, plus the largest step.

    `ptc` is [n_phase, n_dose] over a RECTANGLE. The walk is bottom edge, right edge, top edge
    reversed, left edge reversed -- one closed circuit."""
    p = np.asarray(ptc, float)
    edge = np.concatenate([p[:, 0], p[-1, :], p[::-1, -1], p[0, ::-1]])
    if not np.all(np.isfinite(edge)):
        return np.nan, np.nan
    st = ((np.roll(edge, -1) - edge + 0.5) % 1.0) - 0.5
    return float(st.sum()), float(np.abs(st).max())


def run(model_name, target, mode='instant', phase=(0.0, 1.0), dose=(1.0, 100.0),
        n_phase=128, n_dose=96, dt=None, skip_p=None, pulse=8.0, anchored=True,
        phase_tag=None, tag=None, chunk=2048):
    import jax
    import jax.numpy as jnp
    from models import get_model
    from engine.ptc import make_ptc, phase_or_nan, valid_mask, recommended_skip
    from analysis.phaseref import load as load_phaseref

    model = get_model(model_name)
    off = 0.0
    if anchored:
        offs, _pr = load_phaseref(phase_tag)
        off = float(offs.get(model_name, 0.0))
    # the window is quoted in the ANCHORED frame the figures show; the engine wants raw phase
    ph_disp, span = _arc(phase[0], phase[1], n_phase)
    ph_raw = (ph_disp + off) % 1.0
    doses = np.linspace(float(dose[0]), float(dose[1]), int(n_dose))
    if skip_p is None:
        skip_p, _mu, _r = recommended_skip(model, tol=1e-2, verbose=False)
    dt = float(dt or 0.005)

    f, solver = make_ptc(model, target, mode=mode, readout='raw', skip_p=int(skip_p), dt=dt,
                         pulse=pulse, track_min=True)
    P = model.jax_params()
    x0 = solver.guess(P)
    fj = jax.jit(f)
    PH = jnp.asarray(np.tile(ph_raw, len(doses)))
    DZ = jnp.asarray(np.repeat(doses, len(ph_raw)))
    n = int(PH.shape[0])
    zs, ms = [], []
    print(f"[zoom] {model_name}/{target}/{mode}: {n_phase} x {n_dose} = {n} points, "
          f"dt={dt:g}, skip_p={skip_p}", flush=True)
    for a in range(0, n, chunk):
        z_, m_ = fj(P, x0, PH[a:min(a + chunk, n)], DZ[a:min(a + chunk, n)])
        zs.append(np.asarray(z_))
        ms.append(np.asarray(m_))
        print(f"[zoom]   {min(a + chunk, n)}/{n}", flush=True)
    z, mn = np.concatenate(zs), np.concatenate(ms)
    p, amp = phase_or_nan(z)
    v = valid_mask(mn)
    p = np.where(v, p, np.nan)
    ptc = p.reshape(len(doses), n_phase).T           # [phase, dose]
    ampg = amp.reshape(len(doses), n_phase).T
    # report in the anchored frame, exactly as the figures do
    ptc_disp = (ptc - off) % 1.0

    w, mx = loop_winding(ptc_disp)
    blob = dict(model=model_name, target=target, mode=mode, old=ph_disp, old_raw=ph_raw,
                doses=doses, ptc=ptc_disp, amp=ampg, valid=v.reshape(len(doses), n_phase).T,
                dt=dt, skip_p=int(skip_p), offset=off, span=span,
                loop_winding=w, loop_max_step=mx)
    report(blob)
    out = paths.out_path(ZOO, 'zoom',
                         f'zoom_{model_name}_{target}_{mode}.npz', paths.run_tag(tag))
    paths.savez(out, **blob)
    print(f"[zoom] -> {out}", flush=True)
    return blob


def report(b):
    w, mx = b['loop_winding'], b['loop_max_step']
    print(f"\n{'=' * 84}\nZOOM -- {b['model']}/{b['target']}/{b['mode']}   "
          f"phase {b['old'][0]:.3f}..{b['old'][-1]:.3f} (arc {b['span']:.3f} cyc), "
          f"dose {b['doses'][0]:g}..{b['doses'][-1]:g}\n{'=' * 84}")
    print(f"  WINDING AROUND THE PATCH BOUNDARY = {w:+.4f}   (largest boundary step "
          f"{mx:.4f} cyc)")
    if not np.isfinite(w):
        print("  -- the boundary has unusable cells; the loop sum is undefined.")
    elif mx > LOOP_STEP_WARN:
        print(f"  -- WARNING: a boundary step exceeds {LOOP_STEP_WARN}, so the loop is "
              f"under-sampled and this integer is NOT trustworthy. Enlarge the patch or "
              f"raise the resolution.")
    elif abs(w) > 0.5:
        print("  -- A GENUINE PHASE SINGULARITY IS ENCLOSED. New phase advances by a nonzero")
        print("     integer around a closed loop, which no continuous phase field can do.")
    else:
        print("  -- NO SINGULARITY IS ENCLOSED. Whatever is inside is a steep but CONTINUOUS")
        print("     phase gradient; a contour crossing the 0/1 wrap looks like a defect on a")
        print("     cyclic colour map and is not one.")
    fin = np.isfinite(b['ptc'])
    print(f"  patch: {fin.sum()}/{fin.size} cells usable, min relative amplitude "
          f"{np.nanmin(b['amp']):.4g}")


def figure(b, tag=None, panel_w=7.0):
    old, doses, ptc = b['old'], b['doses'], np.asarray(b['ptc'], float)
    # the arc may wrap; plot against the UNWRAPPED arc coordinate so the panel is contiguous
    x = b['old'][0] + np.linspace(0, float(b['span']), len(old))
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2.1, panel_w),
                             gridspec_kw={'width_ratios': [1.25, 1]})
    im = axes[0].pcolormesh(x, doses, np.ma.masked_invalid(ptc.T), cmap=phase_cmap(),
                            vmin=0, vmax=1, shading='nearest', rasterized=True)
    axes[0].set_xlabel('old phase (anchored frame; arc may cross the 0/1 wrap)', fontsize=9)
    axes[0].set_ylabel('dose', fontsize=9)
    w, mx = b['loop_winding'], b['loop_max_step']
    verdict = ('UNDEFINED' if not np.isfinite(w)
               else ('UNDER-SAMPLED' if mx > LOOP_STEP_WARN
                     else ('SINGULARITY ENCLOSED' if abs(w) > 0.5 else 'NO SINGULARITY')))
    axes[0].set_title(f"{b['model']} {b['target']} ({b['mode']})\n"
                      f"loop winding = {w:+.3f}  ->  {verdict}", fontsize=10)
    fig.colorbar(im, ax=axes[0], label='new phase (cyc)', fraction=0.046)

    # the boundary walk itself, which is what the number is computed from
    p = ptc
    edge = np.concatenate([p[:, 0], p[-1, :], p[::-1, -1], p[0, ::-1]])
    st = ((np.roll(edge, -1) - edge + 0.5) % 1.0) - 0.5
    axes[1].plot(np.cumsum(st), color='k', lw=1.6)
    n1, n2, n3 = p.shape[0], p.shape[0] + p.shape[1], 2 * p.shape[0] + p.shape[1]
    for xx, lab in ((0, 'bottom'), (n1, 'right'), (n2, 'top'), (n3, 'left')):
        axes[1].axvline(xx, color='0.75', lw=0.8, ls=':')
        axes[1].text(xx, 1.01, lab, fontsize=7, color='0.4', transform=
                     axes[1].get_xaxis_transform(), ha='left')
    axes[1].axhline(0, color='0.6', lw=0.8)
    axes[1].axhline(w, color='tab:red', lw=1.0, ls='--')
    axes[1].set_xlabel('step along the patch boundary', fontsize=9)
    axes[1].set_ylabel('cumulative phase advance (cyc)', fontsize=9)
    axes[1].set_title('the loop integral: it must end on an INTEGER', fontsize=10)
    fig.tight_layout()
    return paths.save_figure(fig, ZOO, 'zoom', f"zoom_{b['model']}_{b['target']}_{b['mode']}",
                             tag=paths.run_tag(tag), loop_winding=float(w),
                             phase=[float(b['old'][0]), float(b['old'][-1])],
                             dose=[float(doses[0]), float(doses[-1])])


def main(argv=None):
    ap = argparse.ArgumentParser(description='a fine patch of the PTC, and the loop-winding '
                                             'test for a singularity inside it')
    ap.add_argument('--model', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--mode', default='instant', choices=('pulse', 'instant'))
    ap.add_argument('--phase', default='0,1', help='lo,hi in the ANCHORED frame; wraps if hi<lo')
    ap.add_argument('--dose', default='1,100', help='lo,hi (linear)')
    ap.add_argument('--n-phase', type=int, default=128)
    ap.add_argument('--n-dose', type=int, default=96)
    ap.add_argument('--dt', type=float, default=None)
    ap.add_argument('--skip', type=int, default=None)
    ap.add_argument('--raw-phase', dest='anchored', action='store_false',
                    help="window is in the model's own phase frame, not the anchored one")
    ap.add_argument('--phase-tag', default=None)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    ph = tuple(float(x) for x in a.phase.split(','))
    dz = tuple(float(x) for x in a.dose.split(','))
    b = run(a.model, a.target, a.mode, ph, dz, a.n_phase, a.n_dose, a.dt, a.skip,
            anchored=a.anchored, phase_tag=a.phase_tag, tag=a.tag)
    figure(b, a.tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
