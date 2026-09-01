"""
analysis/compare_points.py
==========================
Put the actual surfaces side by side: what do the SEED, the TRUTH, the two optimizers' local
minima, and the BARRIER PEAKS between them actually look like?

    $PY -m analysis.compare_points --model almeida --target BMAL1

Everything in this project so far has been scalars -- ranks, condition numbers, barrier heights,
costs. Those are inferences about surfaces nobody has looked at. This renders the surfaces.

THE POINTS
    seed        nominal parameters, v = 0. Where every fit starts.
    truth       the displaced parameter set the synthetic target was generated from.
    cma         where CMA-ES stopped (cost 0.0286, ~1.3 h typical phase error).
    lbfgs       where L-BFGS stopped (cost 0.0607, ~1.9 h).
    barrier-c   the peak of the cost along the straight segment cma -> truth (t = 0.75).
    barrier-l   the peak along lbfgs -> truth (t = 0.80).

    The two barrier points are the least interesting parameter sets in the figure and the most
    interesting panels: they are the states a local optimizer would have to pass THROUGH, and
    the whole "distinct basins" claim rests on what happens there. The measured answer is that
    the clock stays alive (alive_frac = 1.000, healthy amplitude), so the barrier is genuine
    phase disagreement rather than the oscillation dying -- this shows what that looks like.

WHAT IS PLOTTED, AND IN WHAT UNITS
    PTC surfaces on a common colour scale, so the panels are directly comparable.
    Twist and winding against dose, overlaid, since those are the features the analysis quotes.
    LC traces against NORMALIZED phase, not time. The period is the time-gauge generator and can
    be rescaled to anything, so comparing traces on a clock-time axis would show a difference
    that is pure choice of units. Normalized phase is the gauge-invariant comparison; the
    periods are reported in the table instead.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analysis  # noqa: F401
import paths
from analysis import winding as W
from analysis import quality as Q
from plotting import phase_cmap, twist_panel


def gather(model_name='almeida', target='BMAL1', mode='instant', n_phase=24, n_dose=16,
           max_factor=6.0, tag_cma='T1cma', tag_lbfgs='T1d'):
    """Evaluate every point of interest on a common, figure-quality grid."""
    from models import get_model
    from engine.orbit import OrbitSolver
    from fit.cost import make_cost, FixedTarget, RadialTarget
    from fit.doses import fit_dose_grid
    import jax.numpy as jnp

    model = get_model(model_name)
    doses, s_crit = fit_dose_grid(model_name, target, mode, max_factor, n_dose)
    C0 = make_cost(model, target, doses, RadialTarget(), n_phase=n_phase, mode=mode,
                   backend='diffrax', dt=0.02, w_osc=0.0, w_amp=0.0)

    def load(tag):
        d = paths.out_dir(model_name, 'fit_recover', tag, create=False)
        fs = sorted(glob.glob(os.path.join(d, '*.npz')))
        if not fs:
            raise SystemExit(f"no fit_recover run under tag {tag}")
        return np.load(fs[-1], allow_pickle=True)

    z_c, z_l = load(tag_cma), load(tag_lbfgs)
    v_true = np.asarray(z_c['v_true'])
    v_cma, v_lbfgs = np.asarray(z_c['v_fit']), np.asarray(z_l['v_fit'])

    zt, _al, _a = C0['surface'](v_true)
    C = make_cost(model, target, doses, FixedTarget(zt), n_phase=n_phase, mode=mode,
                  backend='diffrax', dt=0.02, w_osc=0.0, w_amp=0.0)

    # locate the barrier peaks on the two straight segments
    def peak(v_from):
        ts = np.linspace(0, 1, 21)
        fs = np.array([C['total'](v_from + t * (v_true - v_from)) for t in ts])
        i = int(np.argmax(fs))
        return v_from + ts[i] * (v_true - v_from), float(ts[i]), float(fs[i])

    v_bc, t_bc, f_bc = peak(v_cma)
    v_bl, t_bl, f_bl = peak(v_lbfgs)

    pts = [('seed', np.zeros(C['n_free'])), ('truth', v_true), ('cma', v_cma),
           ('lbfgs', v_lbfgs), (f'barrier-c (t={t_bc:.2f})', v_bc),
           (f'barrier-l (t={t_bl:.2f})', v_bl)]

    solver = OrbitSolver(model)
    names = C['names']
    obs = list(model.observable_states())
    oidx = [int(model.var_index(s)) for s in obs]
    old = np.asarray(C['old'])

    out = []
    for label, v in pts:
        zu, alive, amp = C['surface'](v)
        ptc = np.where(alive, (np.angle(zu) / (2 * np.pi)) % 1.0, np.nan)
        p = C['parts'](v)
        P = model.jax_apply(C['theta'](v), names)
        y0, T, _r = solver.solve(P, solver.guess(P))
        cyc = np.asarray(solver.cycle(P, y0, T, 128))[:, oidx]
        tw = W.twist_curve(old, doses, ptc)
        S, phi, nsing = W.detect_grid(old, doses, ptc)
        q = Q.score(old, doses, ptc)
        out.append(dict(label=label, v=v, ptc=ptc, amp=amp, alive=alive, cyc=cyc,
                        period=float(T), twist=tw, total_twist=W.total_twist(tw),
                        S=S, phi=phi, n_sing=int(nsing),
                        W=W.winding_vs_dose(W.impute(ptc)),
                        c_ptc=p['c_ptc'], amp_lc=p['amp_lc'],
                        alive_frac=p['alive_frac'], quality=q['passed'],
                        scramble=q['scramble'], theta=C['theta'](v)))
    return dict(points=out, doses=np.asarray(doses), old=old, obs=obs, names=names,
                s_crit=s_crit, model=model_name, target=target, mode=mode)


def figure(g, path=None):
    pts, doses, old, obs = g['points'], g['doses'], g['old'], g['obs']
    n = len(pts)
    truth = next(p for p in pts if p['label'] == 'truth')

    # A dedicated narrow column for the colorbars. Attaching a colorbar to the last panel
    # steals width from it alone, so the rightmost surface renders narrower than the five it is
    # meant to be compared with.
    fig = plt.figure(figsize=(3.05 * n + 1.1, 12.4))
    gs = fig.add_gridspec(4, n + 1, height_ratios=[1.25, 0.95, 0.95, 0.85],
                          width_ratios=[1.0] * n + [0.075], hspace=0.52, wspace=0.30)

    # --- row 0: PTC surfaces, common colour scale ------------------------------ #
    for i, p in enumerate(pts):
        ax = fig.add_subplot(gs[0, i])
        m = ax.pcolormesh(old, doses, p['ptc'].T, cmap=phase_cmap(), vmin=0, vmax=1,
                          shading='nearest')
        ax.set_yscale('log')
        ax.set_xlabel('old phase (cyc)')
        if i == 0:
            ax.set_ylabel('dose')
        # this panel's OWN S_crit -- the previous version drew one global value on every
        # panel, which put the line in the wrong place everywhere except by coincidence
        if np.isfinite(p['S']):
            ax.axhline(p['S'], color='w', ls='--', lw=1.1, alpha=0.85)
            ax.plot([p['phi']], [p['S']], 'o', mfc='none', mec='w', mew=1.8, ms=9)
        bad = '' if p['quality'] else '  [GATE FAIL]'
        ax.set_title(f"{p['label']}{bad}\nc_ptc={p['c_ptc']:.4f}  T={p['period']:.1f} h",
                     fontsize=8.5)
        if i == n - 1:
            fig.colorbar(m, cax=fig.add_subplot(gs[0, n]), label='new phase (cyc)')

    # --- row 1: PTC difference from the truth ---------------------------------- #
    for i, p in enumerate(pts):
        ax = fig.add_subplot(gs[1, i])
        d = np.abs(p['ptc'] - truth['ptc']) % 1.0
        d = np.minimum(d, 1 - d)
        mm = ax.pcolormesh(old, doses, d.T, cmap='magma', vmin=0, vmax=0.5,
                           shading='nearest')
        ax.set_yscale('log'); ax.set_xlabel('old phase (cyc)')
        if i == 0:
            ax.set_ylabel('dose')
        ax.set_title(f"|PTC - truth|   rms {np.sqrt(np.nanmean(d ** 2)):.4f} cyc",
                     fontsize=8.5)
        if i == n - 1:
            fig.colorbar(mm, cax=fig.add_subplot(gs[1, n]), label='|d phase| (cyc)')

    # --- row 2: features -- twist, winding, LC traces --------------------------- #
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    ax = fig.add_subplot(gs[2, 0:2])
    for i, p in enumerate(pts):
        twist_panel(ax, doses, p['twist'], label=p['label'], color=cols[i],
                    scrit=g['s_crit'] if i == 0 else None,
                    title='isochron twist (flat = radial)')
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[2, 2:4])
    for i, p in enumerate(pts):
        ax.step(doses, p['W'], where='mid', color=cols[i], lw=1.6, alpha=0.85,
                label=p['label'])
    ax.set_xscale('log'); ax.set_xlabel('dose'); ax.set_ylabel('winding number')
    ax.axvline(g['s_crit'], color='k', ls='--', lw=1.0)
    ax.set_yticks([-1, 0, 1, 2]); ax.set_title('winding vs dose', fontsize=9)
    ax.legend(fontsize=7, ncol=2)

    # LC traces of the reference species, against NORMALIZED phase (see module docstring)
    ph = np.arange(pts[0]['cyc'].shape[0]) / pts[0]['cyc'].shape[0]
    ref = 0
    ax = fig.add_subplot(gs[2, 4:n])
    for i, p in enumerate(pts):
        c = p['cyc'][:, ref]
        ax.plot(ph, c / max(np.mean(c), 1e-12), color=cols[i], lw=1.6, alpha=0.85,
                label=f"{p['label']} (T={p['period']:.1f})")
    ax.set_xlabel('phase (cyc)'); ax.set_ylabel(f'{obs[ref]} / mean')
    ax.set_title(f'limit cycle, {obs[ref]} (normalized phase and level)', fontsize=9)
    ax.legend(fontsize=7)

    # --- row 3: the numbers ------------------------------------------------------ #
    ax = fig.add_subplot(gs[3, 0:n])
    ax.axis('off')
    hdr = (f"{'point':22s} {'c_ptc':>8s} {'phase err':>10s} {'S_crit':>9s} {'phi*':>7s} "
           f"{'n_sing':>6s} {'twist':>8s} {'T (h)':>7s} {'amp_lc':>8s} {'alive':>7s} "
           f"{'scramble':>9s} {'gate':>6s}")
    lines = [hdr, '-' * len(hdr)]
    for p in pts:
        x = float(np.clip(1 - 2 * p['c_ptc'], -1, 1))
        cyc_err = float(np.arccos(x) / (2 * np.pi))
        lines.append(
            f"{p['label']:22s} {p['c_ptc']:8.4f} {cyc_err * 24:9.2f}h {p['S']:9.4g} "
            f"{p['phi']:7.3f} {p['n_sing']:6d} {p['total_twist']:8.4f} {p['period']:7.2f} "
            f"{p['amp_lc']:8.3f} {p['alive_frac']:7.3f} {p['scramble']:9.4f} "
            f"{'ok' if p['quality'] else 'FAIL':>6s}")
    ax.text(0.0, 1.0, '\n'.join(lines), family='monospace', fontsize=8.5, va='top',
            transform=ax.transAxes)

    fig.suptitle(f"{g['model']}/{g['target']} ({g['mode']}) -- surfaces at the seed, the truth, "
                 f"both optimizers' minima, and the barrier peaks between them", fontsize=11)
    if path:
        fig.savefig(path, dpi=140, bbox_inches='tight')
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description='render the surfaces at the interesting points')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    ap.add_argument('--mode', default='instant', choices=('pulse', 'instant'))
    ap.add_argument('--n-phase', type=int, default=24)
    ap.add_argument('--n-dose', type=int, default=16)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)

    g = gather(a.model, a.target, a.mode, a.n_phase, a.n_dose)
    tag = paths.run_tag(a.tag)
    out = paths.out_path(a.model, 'compare_points',
                         f'points_{a.target}_{a.mode}.npz', tag)
    blob = {'model': a.model, 'target': a.target, 'mode': a.mode, 'doses': g['doses'],
            'old': g['old'], 's_crit': g['s_crit'],
            'labels': np.array([p['label'] for p in g['points']])}
    for p in g['points']:
        k = p['label'].split(' ')[0]
        for f in ('ptc', 'amp', 'alive', 'cyc', 'twist', 'W', 'v', 'theta'):
            blob[f'{f}__{k}'] = np.asarray(p[f])
        for f in ('period', 'total_twist', 'S', 'phi', 'n_sing', 'c_ptc', 'amp_lc',
                  'alive_frac', 'scramble', 'quality'):
            blob[f'{f}__{k}'] = np.asarray(p[f])
    paths.savez(out, **blob)

    png = out.replace('.npz', '.png')
    figure(g, png)
    print(f"[compare_points] -> {out}")
    print(f"[compare_points] -> {png}")
    for p in g['points']:
        print(f"  {p['label']:22s} c_ptc={p['c_ptc']:.4f} T={p['period']:6.2f} "
              f"amp_lc={p['amp_lc']:6.3f} alive={p['alive_frac']:.3f} "
              f"S={p['S']:.4g} twist={p['total_twist']:.4f} "
              f"gate={'ok' if p['quality'] else 'FAIL'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
