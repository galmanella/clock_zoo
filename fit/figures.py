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




def _backfill(z, old, doses, names):
    """Recompute panels that older runs did not store, from what they DID store.

    A figure should never require re-fitting. Runs made before radial.py saved the target
    surface and the multi-species cycles still hold `theta_base` / `theta_fit`, which is enough
    to reconstruct both: re-solve the orbit for the cycles, and re-profile (k, psi) against the
    stored PTC for the target. Costs a couple of orbit solves, not a fit.
    """
    import jax
    import jax.numpy as jnp
    from models import get_model
    from engine.orbit import OrbitSolver, make_guess_fn
    from fit.target import profile as _profile, radial_z as _radial_z

    # RECONSTRUCT THE PINNED TARGET for runs made before it was stored. Pinning is the
    # default (`--profile-target` opts out) and `from_singularity` is exactly k = 1/S_crit,
    # psi = phi* - 0.5 applied to the SEED's measured singularity -- both of which the run
    # already saved as base__S_crit / base__phi_sing. So this is a re-derivation, not a guess.
    if 'ptc_target_used' not in z and {'base__S_crit', 'base__phi_sing'} <= set(z):
        from fit.target import radial_z as _rz
        S0, p0 = float(z['base__S_crit']), float(z['base__phi_sing'])
        if np.isfinite(S0) and S0 > 0 and np.isfinite(p0):
            z['k_used'] = 1.0 / S0
            z['psi_used'] = (p0 - 0.5) % 1.0
            z['ptc_target_used'] = (np.angle(np.asarray(
                _rz(old, doses, z['k_used'], z['psi_used']))) / (2 * np.pi)) % 1.0
            z['target_pinned'] = True

    need_t = 'ptc_target_fit' not in z
    need_c = 'cyc_fit' not in z
    need_r = 'fit__orbit_res' not in z
    if not (need_t or need_c or need_r):
        return
    model = get_model(str(z['model']))
    y_seed = jnp.asarray(model.get_initial_state(), jnp.float64)

    # SOLVE WITH THE SECTION THE RUN USED, not the one the model defaults to now.
    #
    # If the run recorded it, use it. If it predates that (RAD01/RAD03/RAD05), recover it by
    # trying the candidates and keeping the one that actually converges -- a wrong section does
    # not merely shift the phase origin, it fails to solve, so the residual identifies it
    # unambiguously. The choice is reported rather than assumed silently.
    cands = [str(z['section'])] if 'section' in z else         [str(model.reference_variable)] + [s_ for s_ in model.state_names
                                           if s_ != model.reference_variable]
    best = None
    for sec in cands:
        try:
            sv, gs = OrbitSolver(model, ref=sec), make_guess_fn(model, ref=sec)
            P0 = model.jax_apply(np.asarray(z['theta_fit']), names)
            _y, _T, r = jax.jit(sv.solve)(P0, gs(P0, y_seed))
            r = float(r)
        except Exception:
            continue
        if best is None or r < best[0]:
            best = (r, sec, sv, gs)
        if r < 1e-8:
            break
    _res, sec_used, solver, guess = best
    z['section_used'] = sec_used
    if 'section' not in z:
        print(f"[fig] run did not record its section; recovered '{sec_used}' "
              f"(fitted BVP residual {_res:.2e})")
    obs = list(model.observable_states())
    oidx = [int(model.var_index(sn)) for sn in obs]
    z['obs'] = np.array(obs)

    for pref, thkey, ptckey in (('base', 'theta_base', 'ptc_base'),
                                ('fit', 'theta_fit', 'ptc_fit')):
        P = model.jax_apply(np.asarray(z[thkey]), names)

        # SOLVE THE ORBIT THE WAY THE COST DID, AND KEEP THE RESIDUAL.
        #
        # This line used to read `solver.solve(P, solver.guess(P))` and throw the residual away
        # as `_r`. Both halves were wrong, and together they produced a figure that showed an
        # orbit the fit had never evaluated:
        #
        #   * `solver.guess` is the numpy peak-hunt; `fit.cost._surface` uses `make_guess_fn`,
        #     the jittable relaxation. On a pathological parameter set the two land in
        #     DIFFERENT places. On the RAD01 optimum, `solver.guess` converged to a fixed point
        #     (every species constant, all negative) while `make_guess_fn` did not converge at
        #     all. Diagnosing the fitted PTC against the first of those was diagnosing the
        #     wrong object.
        #
        #   * discarding the residual hid which case this was. `solve` returns the BVP residual
        #     precisely so a caller can ask whether the thing it just got back IS a periodic
        #     orbit. At the RAD01 optimum it is 3.87, not ~1e-13: there is no cycle there, and
        #     `cycle()` just integrates a transient forward for T hours. That transient wanders,
        #     which is why the "fitted PTC" varied with old phase and looked plausible while
        #     the underlying state was garbage.
        #
        # The residual is now stored and PLOTTED, so a non-orbit can never again be presented
        # as a limit cycle.
        y0, T, res = jax.jit(solver.solve)(P, guess(P, y_seed))
        z[f'{pref}__orbit_res'] = float(res)
        if need_c:
            z[f'cyc_{pref}'] = np.asarray(solver.cycle(P, y0, T, 256))[:, oidx]
            z[f'{pref}__period_solved'] = float(T)
        if need_t:
            ptc = np.asarray(z[ptckey])
            alive = np.isfinite(ptc)
            zu = np.exp(2j * np.pi * np.nan_to_num(ptc))
            k, psi, _c = _profile(jnp.asarray(zu), jnp.asarray(alive), old, doses)
            z[f'ptc_target_{pref}'] = (np.angle(np.asarray(_radial_z(old, doses, k, psi)))
                                       / (2 * np.pi)) % 1.0
            z[f'k_target_{pref}'] = float(k)
            z[f'psi_target_{pref}'] = float(psi)


def fig_radial(z):
    """Target, base and fitted PTCs; both limit cycles on their OWN axes; the numbers.

    FOUR THINGS THIS LAYOUT FIXES, ALL OF WHICH THE PREVIOUS VERSION GOT WRONG:

    1. THE TARGET IS SHOWN. The earlier figure plotted base, fitted and their difference but
       never the radial surface being fitted TO -- the one panel that says what "success" would
       even look like. RadialTarget profiles (k, psi) per evaluation, so the target is not a
       fixed picture; radial.py now stores the registration that actually scored each surface.

    2. EACH LIMIT CYCLE GETS ITS OWN AXES. Plotting a 24.8 h cycle and a 0.159 h cycle on shared
       axes rendered the second as a dot at the origin -- the failure was visible but its SHAPE
       was not, and the shape is what says whether the thing still oscillates.

    3. NO NORMALIZATION BY MEAN. The old "level / mean" panel divided by a mean near zero, put
       everything on a 1e11 scale, and flattened both curves to lines. Absolute concentration
       against real time, per panel, is what can actually be read.

    4. SEVERAL SPECIES, not just the reference. One trace cannot distinguish "the clock stopped"
       from "this particular species stopped".
    """
    old, doses = np.asarray(z['old']), np.asarray(z['doses'])
    names = [str(s_) for s_ in z['names']]
    _backfill(z, old, doses, names)
    obs = [str(s_) for s_ in z['obs']] if 'obs' in z else None

    def _sc(key):
        return float(z[key]) if key in z and np.isfinite(z[key]) else None

    # PREFER THE TARGET THE COST USED. `ptc_target_*` are per-surface PROFILED registrations
    # -- each surface's own best alignment to a radial pattern -- which is a descriptive
    # statistic, not the thing the optimizer minimised against. When the run pinned its target,
    # that pinned surface is the only honest thing to label "target" and the only honest thing
    # to difference against.
    pinned = bool(z['target_pinned']) if 'target_pinned' in z else False
    if pinned and 'ptc_target_used' in z:
        t_ptc = np.asarray(z['ptc_target_used'])
        k_t = float(z['k_used'])
        t_lab = f'radial target (PINNED, S={1.0 / k_t:.4g})'
        t_base = t_fit = t_ptc
    else:
        t_ptc = np.asarray(z['ptc_target_fit'])
        k_t = float(z['k_target_fit']) if 'k_target_fit' in z else np.nan
        t_lab = 'radial target (profiled per surface)'
        t_base, t_fit = np.asarray(z['ptc_target_base']), np.asarray(z['ptc_target_fit'])

    panels = [(t_lab, t_ptc, (1.0 / k_t) if np.isfinite(k_t) and k_t > 0 else None),
              ('base', np.asarray(z['ptc_base']), _sc('base__S_crit')),
              ('fitted', np.asarray(z['ptc_fit']), _sc('fit__S_crit'))]
    diffs = [('|base - target|', np.asarray(z['ptc_base']), t_base),
             ('|fitted - target|', np.asarray(z['ptc_fit']), t_fit)]

    # dedicated colorbar columns so no panel is narrower than its neighbours
    fig = plt.figure(figsize=(20.5, 9.0))
    gs = fig.add_gridspec(2, 7, height_ratios=[1.05, 0.95],
                          width_ratios=[1, 1, 1, 0.07, 1, 1, 0.07],
                          hspace=0.44, wspace=0.42)

    for i, (lab, ptc, sc) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        m = ax.pcolormesh(old, doses, ptc.T, cmap=phase_cmap(), vmin=0, vmax=1,
                          shading='nearest')
        ax.set_yscale('log'); ax.set_xlabel('old phase (cyc)')
        if i == 0:
            ax.set_ylabel('dose')

        # EACH PANEL'S OWN S_crit, NOT ONE GLOBAL VALUE.
        #
        # The dashed line marks where THAT surface turns over, so a line drawn at the same
        # height in all three panels is not merely an approximation -- it is a false claim.
        # The whole point of the fit is that the singularity MOVES, and a line that cannot
        # move cannot show it. (Fixed once before in analysis/compare_points for the same
        # reason; it was still wrong here.) The target's S_crit is exact rather than measured:
        # radial_z has |k * dose| = 1 at its singularity, so it sits at 1/k.
        if sc is not None and np.isfinite(sc) and sc > 0:
            ax.axhline(sc, color='w', ls=':', lw=1.2, alpha=0.85)
            ax.text(0.02, sc, f'S={sc:.3g}', color='w', fontsize=6.5, va='bottom')
        ax.set_title(lab, fontsize=9.5)
    fig.colorbar(m, cax=fig.add_subplot(gs[0, 3]), label='new phase (cyc)')

    for j, (lab, A, B) in enumerate(diffs):
        ax = fig.add_subplot(gs[0, 4 + j])
        d = np.abs(A - B) % 1.0
        d = np.minimum(d, 1 - d)
        mm = ax.pcolormesh(old, doses, d.T, cmap='magma', vmin=0, vmax=0.5, shading='nearest')
        ax.set_yscale('log'); ax.set_xlabel('old phase (cyc)')
        ax.set_title(f'{lab}   rms {np.sqrt(np.nanmean(d ** 2)):.4f} cyc', fontsize=9.5)
    fig.colorbar(mm, cax=fig.add_subplot(gs[0, 6]), label='|d phase| (cyc)')

    # --- limit cycles, one panel each, own axes, several species ------------------- #
    for j, (pref, lab) in enumerate((('base', 'base'), ('fit', 'fitted'))):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        key = f'cyc_{pref}'
        if key not in z:
            ax.axis('off')
            ax.text(0.5, 0.5, lab + ' cycle not stored -- re-run fit.radial', ha='center')
            continue
        cyc = np.asarray(z[key])
        T = float(z[f'{pref}__period_solved'] if f'{pref}__period_solved' in z
                  else z[f'{pref}__period'])
        res = float(z[f'{pref}__orbit_res']) if f'{pref}__orbit_res' in z else float('nan')
        t = np.linspace(0, T, cyc.shape[0])

        # EVERY SPECIES ON ITS OWN SCALE, WITH ITS ABSOLUTE RANGE IN THE LEGEND.
        #
        # Almeida's species differ by ELEVEN orders of magnitude on a bad parameter set. Shared
        # axes render all but the largest as a flat line at zero, which reads as "the clock
        # stopped" whether or not it did -- the exact artifact that made a diverging transient
        # look like a dead oscillator here. Scaling each trace to its own [min, max] makes the
        # SHAPE readable at any magnitude, and the legend carries the absolute numbers so the
        # normalisation hides nothing: a species with zero range plots flat at 0 AND shows
        # range 0 in its label.
        for k in range(cyc.shape[1]):
            y = cyc[:, k]
            lo, hi = float(np.min(y)), float(np.max(y))
            span = hi - lo
            yn = (y - lo) / span if span > 0 else np.zeros_like(y)
            nm = obs[k] if obs and k < len(obs) else 'y%d' % k
            ax.plot(t, yn, lw=1.4, label=f'{nm}  [{lo:.3g}, {hi:.3g}]')

        ok = np.isfinite(res) and res < 1e-4
        neg = float(np.min(cyc)) < 0.0
        flag = 'orbit' if ok else 'NOT AN ORBIT'
        ax.set_title(f'{lab}:  T = {T:.2f} h   BVP res {res:.1e}   [{flag}]',
                     fontsize=8.5, color='k' if ok else '#b3261e')
        ax.set_xlabel('time (h)')
        ax.set_ylabel('each species scaled to its own [min, max]')
        ax.set_ylim(-0.05, 1.05)
        if neg:
            ax.text(0.99, 0.02, 'negative concentrations present', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=7.5, color='#b3261e')
        ax.legend(fontsize=6.0, ncol=2, loc='upper right', framealpha=0.85)

    ax = fig.add_subplot(gs[1, 4])
    twist_panel(ax, doses, np.asarray(z['twist_fit']), base=np.asarray(z['twist_base']),
                label='fitted', scrit=_sc('fit__S_crit'),
                title='isochron twist (flat = radial)')

    ax = fig.add_subplot(gs[1, 5:]); ax.axis('off')
    Tb, Tf = float(z['base__period']), float(z['fit__period'])
    ar = float(z['fit__amp_lc']) / max(float(z['base__amp_lc']), 1e-12)
    pr = Tf / max(Tb, 1e-12)
    bad = (ar < 0.5 or ar > 2.0 or pr < 0.5 or pr > 2.0
           or not (0.0 < float(z['fit__mu']) < 0.99))
    lines = [f"{'':12s}{'base':>11s}{'fitted':>11s}", '-' * 34,
             f"{'c_ptc':12s}{float(z['base__parts_c_ptc']):11.4f}"
             f"{float(z['fit__parts_c_ptc']):11.4f}",
             f"{'twist':12s}{float(z['base__total_twist']):11.4f}"
             f"{float(z['fit__total_twist']):11.4f}",
             f"{'S_crit':12s}{float(z['base__S_crit']):11.4g}"
             f"{float(z['fit__S_crit']):11.4g}",
             f"{'period (h)':12s}{Tb:11.3f}{Tf:11.3f}",
             f"{'amp_lc':12s}{float(z['base__amp_lc']):11.3f}"
             f"{float(z['fit__amp_lc']):11.3f}",
             f"{'Floquet mu':12s}{float(z['base__mu']):11.4f}{float(z['fit__mu']):11.4f}",
             '', f"amplitude x{ar:.3g}   period x{pr:.3g}",
             f"target k={float(z['k_target_fit']):.4g} psi={float(z['psi_target_fit']):.3f}"]
    if bad:
        lines += ['', 'DEGENERATE -- not a circadian', 'oscillator; residual and twist',
                  '"gains" are artefacts.']
    ax.text(0.0, 1.0, "\n".join(lines), family='monospace', fontsize=9, va='top',
            color=('#b31d28' if bad else '#1a7f37'), transform=ax.transAxes)

    # THE TITLE MUST AGREE WITH THE VERDICT. It used to read `improved` alone, so RAD05 was
    # captioned "radialized" while radial.py's own verdict was UNUSABLE: twist had moved
    # 0.4972 -> 0.4818, technically "less", on a surface with three singularities and a winding
    # set of [0, 1, 2]. A twist number read off a surface that fails the quality gate is not a
    # measurement, so a figure must never promote it to a headline.
    qual_ok = bool(z['quality_fit']) if 'quality_fit' in z else True
    if bad:
        lab, col = 'DEGENERATE', '#b31d28'
    elif not qual_ok:
        lab, col = 'UNUSABLE -- fitted surface fails the PTC quality gate', '#b31d28'
    elif bool(z['improved']):
        lab, col = 'radialized', 'black'
    else:
        lab, col = 'no progress', 'black'
    fig.suptitle(f"{z['model']}/{z['target']} ({z['mode']}) radial fit [{z['optimizer']}] "
                 f"-- {lab}", fontsize=12, color=col)
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
