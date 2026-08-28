"""
fit/figures.py
==============
Figures for the fitting runs. PURE READ -- every panel comes from a saved npz, so replotting
never re-runs a fit.

    python -m fit.figures --which radial  [--tag T]
    python -m fit.figures --which recover [--tag T]

WHY THE LIMIT-CYCLE PANELS ARE HERE
    The first radial run returned VERDICT: RADIALIZED. Residual 0.258 -> 0.011, twist
    0.459 -> 0.213, both moving the right way. It was a false positive: the fit had escaped to a
    period of 0.159 h with 45x the amplitude and a Floquet multiplier of exactly 1 -- not a
    circadian oscillator, not even an attracting cycle. Those three numbers sat in one line of
    text under a headline that said success.

    Plotted, it is unmissable. So every fit figure now shows the limit cycle in REAL TIME
    alongside the PTC, because that is the axis on which this class of failure is obvious and
    the PTC panels alone look fine.
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


def _circd(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 1.0
    return np.minimum(d, 1 - d)


def _cycles(model_name, names, theta_base, theta_fit, m=256):
    """Re-solve both limit cycles from the saved parameter vectors.

    radial.py does not store the cycles, so they are recomputed here from `theta_*`, which it
    does store. Returns (t, y, T) per point in REAL TIME, since the period difference is the
    whole diagnostic."""
    from models import get_model
    from engine.orbit import OrbitSolver
    import jax
    model = get_model(model_name)
    solver = OrbitSolver(model)
    ref = int(model.var_index(model.reference_variable))
    out = []
    for th in (theta_base, theta_fit):
        P = model.jax_apply(np.asarray(th), list(names))
        y0, T, _r = jax.jit(solver.solve)(P, solver.guess(P))
        cyc = np.asarray(solver.cycle(P, y0, T, m))[:, ref]
        T = float(T)
        out.append((np.linspace(0, T, m), cyc, T))
    return out


def fig_radial(z):
    """PTC before/after, the twist, and -- decisively -- the limit cycle in real time."""
    old, doses = np.asarray(z['old']), np.asarray(z['doses'])
    names = [str(s) for s in z['names']]
    (tb, cb, Tb), (tf, cf, Tf) = _cycles(str(z['model']), names,
                                         z['theta_base'], z['theta_fit'])

    fig = plt.figure(figsize=(16.5, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9], hspace=0.42, wspace=0.32)

    for i, (key, lab) in enumerate((('ptc_base', 'base'), ('ptc_fit', 'fitted'))):
        ax = fig.add_subplot(gs[0, i])
        m = ax.pcolormesh(old, doses, np.asarray(z[key]).T, cmap=phase_cmap(), vmin=0, vmax=1,
                          shading='nearest')
        ax.set_yscale('log'); ax.set_xlabel('old phase (cyc)')
        if i == 0:
            ax.set_ylabel('dose')
        S = float(z[f"{'base' if i == 0 else 'fit'}__S_crit"])
        if np.isfinite(S):
            ax.axhline(S, color='w', ls='--', lw=1.1, alpha=0.85)
        pref = 'base' if i == 0 else 'fit'
        ax.set_title(f"{lab}   c_ptc={float(z[pref + '__parts_c_ptc']):.4f}", fontsize=9)
        if i == 1:
            fig.colorbar(m, ax=ax, label='new phase (cyc)')

    ax = fig.add_subplot(gs[0, 2])
    d = _circd(z['ptc_fit'], z['ptc_base'])
    mm = ax.pcolormesh(old, doses, d.T, cmap='magma', shading='nearest')
    ax.set_yscale('log'); ax.set_xlabel('old phase (cyc)')
    ax.set_title(f'|fitted - base|   rms {np.sqrt(np.nanmean(d ** 2)):.4f} cyc', fontsize=9)
    fig.colorbar(mm, ax=ax, label='|d phase| (cyc)')

    ax = fig.add_subplot(gs[0, 3])
    twist_panel(ax, doses, np.asarray(z['twist_fit']), base=np.asarray(z['twist_base']),
                label='fitted', title='isochron twist (flat = radial)')

    # --- the panels that expose the failure -------------------------------------- #
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(tb, cb, lw=1.8, label=f'base (T={Tb:.2f} h)')
    ax.plot(tf, cf, lw=1.8, label=f'fitted (T={Tf:.3f} h)')
    ax.set_xlabel('time (h)'); ax.set_ylabel(f'{"reference species"}')
    ax.set_title('limit cycle in REAL TIME -- period and amplitude', fontsize=9)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(np.linspace(0, 1, len(cb)), cb / max(np.mean(cb), 1e-12), lw=1.8, label='base')
    ax.plot(np.linspace(0, 1, len(cf)), cf / max(np.mean(cf), 1e-12), lw=1.8, label='fitted')
    ax.set_xlabel('phase (cyc)'); ax.set_ylabel('level / mean')
    ax.set_title('same cycles, normalized -- waveform only', fontsize=9)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    tr = np.asarray(z['trace_f']) if z.get('trace_f') is not None \
        and np.asarray(z['trace_f']).size else None
    if tr is not None:
        ax.plot(tr, lw=1.2, color='#333')
        ax.set_yscale('log' if np.all(tr > 0) else 'linear')
    ax.set_xlabel('evaluation'); ax.set_ylabel('cost')
    ax.set_title('search trace', fontsize=9)

    ax = fig.add_subplot(gs[1, 3]); ax.axis('off')
    ar = float(z['fit__amp_lc']) / max(float(z['base__amp_lc']), 1e-12)
    pr = Tf / max(Tb, 1e-12)
    bad = (ar < 0.5 or ar > 2.0 or pr < 0.5 or pr > 2.0
           or not (0.0 < float(z['fit__mu']) < 0.99))
    lines = [f"{'':14s}{'base':>12s}{'fitted':>12s}",
             '-' * 38,
             f"{'c_ptc':14s}{float(z['base__parts_c_ptc']):12.4f}"
             f"{float(z['fit__parts_c_ptc']):12.4f}",
             f"{'twist':14s}{float(z['base__total_twist']):12.4f}"
             f"{float(z['fit__total_twist']):12.4f}",
             f"{'period (h)':14s}{Tb:12.3f}{Tf:12.3f}",
             f"{'amp_lc':14s}{float(z['base__amp_lc']):12.3f}"
             f"{float(z['fit__amp_lc']):12.3f}",
             f"{'Floquet mu':14s}{float(z['base__mu']):12.4f}{float(z['fit__mu']):12.4f}",
             '',
             f"amplitude x{ar:.3g}   period x{pr:.3g}"]
    if bad:
        lines += ['', 'DEGENERATE: the fitted object is not a',
                  'circadian oscillator. Residual and twist',
                  '"improvements" are artefacts of fitting a',
                  'different dynamical object.']
    ax.text(0.0, 1.0, '\n'.join(lines), family='monospace', fontsize=9.5, va='top',
            color=('#b31d28' if bad else '#1a7f37'), transform=ax.transAxes)

    tag = 'DEGENERATE' if bad else ('radialized' if bool(z['improved']) else 'no progress')
    fig.suptitle(f"{z['model']}/{z['target']} ({z['mode']}) radial fit [{z['optimizer']}] "
                 f"-- {tag}", fontsize=12,
                 color=('#b31d28' if bad else 'black'))
    return _save(fig, f"radial_{z['target']}_{z['mode']}_{z['optimizer']}",
                 target=str(z['target']), degenerate=bool(bad))


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
    m = ax.pcolormesh(old, doses, ptc_t.T, cmap=phase_cmap(), vmin=0, vmax=1,
                      shading='nearest')
    ax.set_yscale('log'); ax.set_xlabel('old phase (cyc)'); ax.set_ylabel('dose')
    ax.set_title('TRUTH (displaced parameters)', fontsize=9)
    fig.colorbar(m, ax=ax, label='new phase (cyc)')

    ax = fig.add_subplot(gs[0, 1]); ax.axis('off')
    err, err0 = float(z['rms_log']), float(z['rms_log0'])
    ok = bool(z['recovered'])
    ax.text(0.02, 0.95, 'RECOVERED' if ok else 'NOT RECOVERED', fontsize=15,
            fontweight='bold', va='top', color='#1a7f37' if ok else '#b31d28',
            transform=ax.transAxes)
    ax.text(0.02, 0.78,
            f"eps = {float(z['eps']):.2f}\n"
            f"parameter distance to truth\n    start {err0:.4f}\n    end   {err:.4f}\n\n"
            f"cost\n    nominal {float(z['cost_nominal']):.5f}\n"
            f"    fitted  {float(z['cost_fit']):.5f}\n"
            f"    truth   {float(z['cost_truth']):.5f}\n\n"
            f"{int(z['n_within_10pct'])}/{len(names)} parameters within 10%",
            fontsize=9.5, va='top', family='monospace', transform=ax.transAxes)

    ax = fig.add_subplot(gs[0, 2])
    tf = np.asarray(z['trace_f'])
    ax.plot(tf, lw=1.4, color='#333')
    ax.axhline(float(z['cost_truth']), ls='--', color='#1a7f37', lw=1.2, label='truth')
    ax.set_yscale('log' if np.all(tf > 0) else 'linear')
    ax.set_xlabel('evaluation'); ax.set_ylabel('cost'); ax.legend(fontsize=8)
    ax.set_title('descent', fontsize=9)

    ax = fig.add_subplot(gs[1, :])
    th_nom = th_t / np.exp(np.asarray(z['B']) @ np.asarray(z['v_true']))
    before = np.log(th_nom / th_t)
    after = np.log(th_f / th_t)
    i = np.argsort(-np.abs(before))
    x = np.arange(len(names))
    ax.bar(x - 0.2, np.abs(before[i]), 0.4, label='nominal vs truth', color='#c8c8c8')
    ax.bar(x + 0.2, np.abs(after[i]), 0.4, label='fitted vs truth', color='#2b6cb0')
    ax.axhline(0.1, ls=':', color='k', lw=1, label='10%')
    ax.set_xticks(x); ax.set_xticklabels([names[j] for j in i], rotation=60, ha='right',
                                         fontsize=8)
    ax.set_ylabel('|log ratio to truth|'); ax.legend(fontsize=8)
    fig.suptitle(f"Self-recovery -- {z['model']}/{z['target']} ({z['mode']}), "
                 f"eps={float(z['eps']):.2f}", fontsize=11)
    return _save(fig, f"recover_{z['target']}_{z['mode']}_eps{float(z['eps']):g}",
                 target=str(z['target']), recovered=bool(z['recovered']))


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
    ap.add_argument('--which', default='radial', choices=('recover', 'radial'))
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    analysis = 'fit_recover' if a.which == 'recover' else 'fit_radial'
    z, tag = _load(a.model, analysis, '*.npz', a.tag)
    if z is None:
        raise SystemExit(f"no {analysis} run for {a.model}" + (f" (tag {tag})" if tag else ""))
    for k in ('model', 'target', 'mode', 'optimizer'):
        if k in z:
            z[k] = str(z[k])
    _CTX.update(model=a.model, analysis=analysis, tag=tag)
    p = fig_recover(z) if a.which == 'recover' else fig_radial(z)
    print(f"[fit.figures] -> {p}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
