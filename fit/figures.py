"""
fit/figures.py
==============
Figures for the fitting runs. PURE READ -- every panel comes from a saved npz, so replotting
never re-runs a fit.

    python -m fit.figures --which recover [--tag T]
    python -m fit.figures --which radial  [--tag T]

WHAT EACH FIGURE IS FOR
    recover  Did the optimizer get the PARAMETERS back, not just the cost down? The surfaces
             will look nearly identical whenever the cost is low -- that is what a low cost
             MEANS -- so the informative panels are the parameter comparison and the cost
             trace, not the pretty ones.
    radial   Did the isochrons actually become radial, and was the oscillation still alive
             when they did? The amplitude and Floquet annotations are the point: the Mirsky
             radialization run produced a beautiful twist improvement by killing the clock.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import paths
from plotting import phase_cmap, twist_panel

_CTX = {}


def _save(fig, name, **cfg):
    return paths.save_figure(fig, _CTX['model'], _CTX['analysis'], name, _CTX['tag'], **cfg)


def _surf(ax, old, doses, ptc, title):
    m = ax.pcolormesh(old, doses, np.asarray(ptc).T, cmap=phase_cmap(), vmin=0, vmax=1,
                      shading='nearest')
    ax.set_yscale('log')
    ax.set_xlabel('old phase (cyc)')
    ax.set_title(title, fontsize=9)
    return m


def _circd(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 1.0
    return np.minimum(d, 1 - d)


def fig_recover(z):
    """Truth vs fit: surfaces, per-parameter recovery, and the descent."""
    old, doses = np.asarray(z['old']), np.asarray(z['doses'])
    zt = np.asarray(z['target_surface_re']) + 1j * np.asarray(z['target_surface_im'])
    ptc_t = (np.angle(zt) / (2 * np.pi)) % 1.0
    names = [str(s) for s in z['names']]
    th_t, th_f = np.asarray(z['theta_true']), np.asarray(z['theta_fit'])

    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.42, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    m = _surf(ax, old, doses, ptc_t, 'TRUTH (displaced parameters)')
    ax.set_ylabel('dose')
    fig.colorbar(m, ax=ax, label='new phase (cyc)')

    # The fit surface is not stored directly (the target is), so show the residual instead --
    # which is the informative half anyway.
    ax = fig.add_subplot(gs[0, 1])
    err = float(z['rms_log']); err0 = float(z['rms_log0'])
    ax.axis('off')
    ok = bool(z['recovered'])
    ax.text(0.02, 0.95,
            f"{'RECOVERED' if ok else 'NOT RECOVERED'}",
            fontsize=15, fontweight='bold', va='top',
            color='#1a7f37' if ok else '#b31d28', transform=ax.transAxes)
    ax.text(0.02, 0.78,
            f"displacement eps = {float(z['eps']):.2f}\n"
            f"parameter distance to truth\n"
            f"    start   {err0:.4f}\n"
            f"    end     {err:.4f}   ({1 - err / err0:+.1%})\n\n"
            f"cost\n"
            f"    nominal {float(z['cost_nominal']):.5f}\n"
            f"    fitted  {float(z['cost_fit']):.5f}\n"
            f"    truth   {float(z['cost_truth']):.5f}\n\n"
            f"{int(z['n_within_10pct'])}/{len(names)} parameters within 10%\n"
            f"{float(z['seconds']):.0f}s, {int(z['n_starts'])} start(s)",
            fontsize=9.5, va='top', family='monospace', transform=ax.transAxes)

    ax = fig.add_subplot(gs[0, 2])
    tf = np.asarray(z['trace_f'])
    ax.plot(tf, lw=1.4, color='#333')
    ax.axhline(float(z['cost_truth']), ls='--', color='#1a7f37', lw=1.2, label='truth (floor)')
    ax.set_yscale('log' if np.all(tf > 0) else 'linear')
    ax.set_xlabel('cost evaluation'); ax.set_ylabel('cost')
    ax.set_title('descent', fontsize=9); ax.legend(fontsize=8)

    # per-parameter: log error before (nominal->truth) and after (fit->truth)
    ax = fig.add_subplot(gs[1, :])
    th_nom = th_t / np.exp(np.asarray(z['B']) @ np.asarray(z['v_true']))
    before = np.log(th_nom / th_t)
    after = np.log(th_f / th_t)
    i = np.argsort(-np.abs(before))
    x = np.arange(len(names))
    ax.bar(x - 0.2, np.abs(before[i]), 0.4, label='nominal vs truth (start)', color='#c8c8c8')
    ax.bar(x + 0.2, np.abs(after[i]), 0.4, label='fitted vs truth (end)', color='#2b6cb0')
    ax.axhline(0.1, ls=':', color='k', lw=1, label='10% band')
    ax.set_xticks(x); ax.set_xticklabels([names[j] for j in i], rotation=60, ha='right',
                                         fontsize=8)
    ax.set_ylabel('|log ratio to truth|'); ax.legend(fontsize=8)
    ax.set_title('per-parameter recovery (sorted by how far the start was)', fontsize=9)

    fig.suptitle(f"Self-recovery control -- {z['model']}/{z['target']} ({z['mode']}), "
                 f"eps={float(z['eps']):.2f}", fontsize=11)
    return _save(fig, f"recover_{z['target']}_{z['mode']}_eps{float(z['eps']):g}",
                 model=str(z['model']), target=str(z['target']), eps=float(z['eps']),
                 backend=str(z['backend']), recovered=bool(z['recovered']))


def fig_radial(z):
    """Base vs fitted surface and twist, with the degeneracy diagnostics on the figure."""
    old, doses = np.asarray(z['old']), np.asarray(z['doses'])
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.1),
                             gridspec_kw={'width_ratios': [1, 1, 1, 1.25], 'wspace': 0.34})

    m = _surf(axes[0], old, doses, z['ptc_base'], 'base')
    axes[0].set_ylabel('dose')
    _surf(axes[1], old, doses, z['ptc_fit'], 'fitted')
    fig.colorbar(m, ax=axes[1], label='new phase (cyc)')

    d = _circd(z['ptc_fit'], z['ptc_base'])
    mm = axes[2].pcolormesh(old, doses, d.T, cmap='magma', shading='nearest')
    axes[2].set_yscale('log'); axes[2].set_xlabel('old phase (cyc)')
    axes[2].set_title('|change|', fontsize=9)
    fig.colorbar(mm, ax=axes[2], label='|d phase| (cyc)')

    twist_panel(axes[3], doses, np.asarray(z['twist_fit']),
                base=np.asarray(z['twist_base']), label='fitted',
                title='isochron twist (flat = radial)')

    tag = ('DEGENERATE - the clock collapsed' if bool(z['collapsed'])
           else 'unusable surface' if not bool(z['quality_fit'])
           else 'radialized' if (bool(z['improved']) and bool(z['less_twist']))
           else 'partial' if bool(z['improved']) else 'no progress')
    fig.suptitle(
        f"{z['model']}/{z['target']} ({z['mode']}) -- {tag}   |   "
        f"residual {float(z['base__parts_c_ptc']):.4f} -> {float(z['fit__parts_c_ptc']):.4f}   "
        f"twist {float(z['base__total_twist']):.4f} -> {float(z['fit__total_twist']):.4f}   "
        f"LC amp {float(z['base__amp_lc']):.3f} -> {float(z['fit__amp_lc']):.3f}   "
        f"mu {float(z['base__mu']):.3f} -> {float(z['fit__mu']):.3f}", fontsize=10)
    return _save(fig, f"radial_{z['target']}_{z['mode']}", model=str(z['model']),
                 target=str(z['target']), collapsed=bool(z['collapsed']))


def _load(model, analysis, pattern, tag=None):
    tag = tag or paths.latest_run(model, analysis)
    if tag is None:
        return None, None
    d = paths.out_dir(model, analysis, tag, create=False)
    fs = sorted(glob.glob(os.path.join(d, pattern)))
    if not fs:
        return None, tag
    return dict(np.load(fs[-1], allow_pickle=True)), tag


def main(argv=None):
    ap = argparse.ArgumentParser(description='figures for the fitting runs')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--which', default='recover', choices=('recover', 'radial'))
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    analysis = 'fit_recover' if a.which == 'recover' else 'fit_radial'
    z, tag = _load(a.model, analysis, '*.npz', a.tag)
    if z is None:
        raise SystemExit(f"no {analysis} run found for {a.model}"
                         + (f" (tag {tag})" if tag else ""))
    z['model'] = str(z['model']); z['target'] = str(z['target']); z['mode'] = str(z['mode'])
    _CTX.update(model=a.model, analysis=analysis, tag=tag)
    p = fig_recover(z) if a.which == 'recover' else fig_radial(z)
    print(f"[fit.figures] -> {p}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
