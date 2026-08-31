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
from plotting import broken, phase_cmap, twist_panel

_CTX = {}


def _save(fig, name, **cfg):
    # the published copy carries the model prefix: docs/figures/ is shared across models, and
    # publishing is EXPLICIT (hazard 13) -- never a side effect of drawing something
    pub = f"{_CTX['model']}_{name}.png" if _CTX.get('publish') else None
    return paths.save_figure(fig, _CTX['model'], _CTX['analysis'], name, _CTX['tag'],
                             publish=pub, **cfg)


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
    # ALSO RE-DERIVE A STORED-BUT-NaN CYCLE, which is not the same case as a missing one.
    #
    # `fit.radial._diagnose` solves the orbit with `solver.guess` -- the numpy peak-hunt --
    # while `fit.cost._surface` and this function use `make_guess_fn`, the jittable relaxation.
    # That is hazard 15's first bullet, and it is still live in the diagnosis path: three runs
    # of the Aug-30 campaign came back with `period` and `mu` NaN and an all-NaN `cyc_fit`,
    # were flagged DEGENERATE on the NaN mu, and re-solve HERE at a BVP residual of 6e-14. The
    # fits were fine; only the diagnostic's initial guess was not.
    #
    # A figure that plots the stored NaNs shows an empty panel and a "not a circadian
    # oscillator" caption for a perfectly good orbit, which is precisely the
    # solver-limit-rendered-as-divergence failure of hazard 15's second bullet. So when the
    # stored cycle is entirely non-finite, re-derive it and SAY SO on the panel.
    stored_bad = ('cyc_fit' in z and np.asarray(z['cyc_fit']).size
                  and not np.isfinite(np.asarray(z['cyc_fit'], float)).any())
    need_c = 'cyc_fit' not in z or stored_bad
    if stored_bad:
        z['cycle_backfilled'] = True
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

    # PREFER THE RE-SOLVED PERIOD when the run stored a NaN one -- and keep the two claims
    # apart. A stored NaN means the DIAGNOSTIC did not converge (`solver.guess`, hazard 15),
    # not that the fit has no period; `_backfill` re-solves along the production path and
    # reports its BVP residual, so if that residual is tiny there IS an orbit and its period
    # is a measurement. What stays unknown in that case is the Floquet multiplier, because the
    # run never got one -- so the verdict must say UNKNOWN rather than DEGENERATE. Reading a
    # failed diagnostic as a dead clock is how three good fits in the Aug-30 campaign were
    # captioned "not a circadian oscillator".
    def _per(pref):
        v = float(z[f'{pref}__period'])
        if np.isfinite(v):
            return v, False
        s = float(z.get(f'{pref}__period_solved', np.nan))
        return s, np.isfinite(s)

    Tb, _bb = _per('base')
    Tf, T_recovered = _per('fit')
    res_f = float(z.get('fit__orbit_res', np.nan))
    mu_f = float(z['fit__mu'])
    mu_unknown = not np.isfinite(mu_f)
    ar = float(z['fit__amp_lc']) / max(float(z['base__amp_lc']), 1e-12)
    pr = Tf / max(Tb, 1e-12)
    bad = (ar < 0.5 or ar > 2.0
           or (np.isfinite(pr) and (pr < 0.5 or pr > 2.0))
           or (not mu_unknown and not (0.0 < mu_f < 0.99)))
    # an orbit that will not re-solve is a different failure again, and a real one
    if np.isfinite(res_f) and res_f > 1e-4:
        bad = True
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
             f"fitted BVP residual {res_f:.2e}",
             f"target k={float(z['k_target_fit']):.4g} psi={float(z['psi_target_fit']):.3f}"]
    if T_recovered:
        lines += ['', "the run's own diagnosis did NOT",
                  "converge (period/mu stored NaN).",
                  f"Re-solved here along the production",
                  f"path: residual {res_f:.1e}, T={Tf:.3f} h.",
                  "So there IS an orbit; only mu is",
                  "unmeasured."]
    if bad:
        lines += ['', 'DEGENERATE -- not a circadian', 'oscillator; residual and twist',
                  '"gains" are artefacts.']
    elif mu_unknown:
        lines += ['', 'Floquet mu UNMEASURED -- stability', 'is not established either way.']
    col = '#b31d28' if bad else ('#8250df' if mu_unknown else '#1a7f37')
    ax.text(0.0, 1.0, "\n".join(lines), family='monospace', fontsize=9, va='top',
            color=col, transform=ax.transAxes)

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
    elif mu_unknown:
        lab, col = ('radialized, but STABILITY UNMEASURED (Floquet mu not available)',
                    '#8250df')
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


# --------------------------------------------------------------------------- #
#  CAMPAIGN figures: many runs at once, from fit/aggregate.py's joined npz
#  ---------------------------------------------------------------------
#  A per-run figure (fig_radial) answers "did THIS search work". A campaign asks a different
#  question that no per-run figure can show: whether independent searches land in the SAME
#  place. That is a statement about the set, so it needs a plot of the set.
#
#  ONE RULE RUNS THROUGH ALL OF THESE: a run whose verdict is not a measurement is never drawn
#  as though it were. DEGENERATE / UNUSABLE / NO DIAGNOSIS runs keep their own colour
#  everywhere, are excluded from every summary statistic, and are excluded from the twist and
#  parameter panels entirely -- a twist curve read off a phase-scrambled surface is a picture of
#  noise, and averaging it in is how the broken CRY probe once produced the most attractive
#  number in the project (PROJECT_SUMMARY 3.7, 5.5).
# --------------------------------------------------------------------------- #
from analysis.winding import circ_span
from fit.aggregate import VERDICT_COLOR, VERDICTS, USABLE, best_index
from fit.rescan import ID_TOL

#: a dose row whose old-phase span reaches this is sweeping the whole phase circle -- type-1,
#: and the part of a radial target that actually constrains anything. Below S* the measured
#: spans sit at 0.49-0.50 and the first row above it drops to 0.36, so the threshold separates
#: them with room to spare rather than splitting a continuum.
INFORMATIVE_SPAN = 0.45
#: below this a PTC does not resolve old phase at all -- a phaseless surface (PROJECT_SUMMARY
#: 5.10d), which has zero twist by construction and therefore satisfies `less_twist` for free
FLAT_SPAN = 0.05


def _span_vs_dose(ptc):
    """Old-phase span of each dose row, in cycles. CIRCULAR, which is not optional.

    The first version of this subtracted an ARITHMETIC mean before taking the peak-to-peak, and
    that is wrong the moment a row straddles the 0/1 wrap: REV's target sits at psi = 0.975 and
    read 4 informative dose rows against a true 1, which would have gone straight into a table.
    `circ_span` is the house function for exactly this and saturates at 0.5 -- here that ceiling
    is the meaning wanted, since 0.5 IS "sweeps the whole circle".
    """
    p = np.asarray(ptc, float)
    return np.array([circ_span(p[:, j]) for j in range(p.shape[1])])


def _vcolors(z):
    return [VERDICT_COLOR.get(str(v), '0.5') for v in z['verdict']]


def _usable(z):
    return np.array([str(v) in USABLE for v in z['verdict']])


def _verdict_legend(ax, z, **kw):
    present = [v for v in VERDICTS if np.any(z['verdict'] == v)]
    ax.legend(handles=[matplotlib.patches.Patch(facecolor=VERDICT_COLOR[v], label=v)
                       for v in present], **kw)


def fig_seeds(z):
    """ONE OPTIMUM OR SEVERAL? -- the whole seed campaign on one page.

    The question this exists for cannot be asked of one run and cannot be asked within one
    SLURM task either: four tasks of four seeds give four four-seed comparisons, and sixteen
    seeds sitting in four separate clusters look exactly like one cluster when the only
    distances ever measured are within a task. `fit.aggregate` joins them; this reads the join.

    THE TWO PANELS THAT CARRY THE ANSWER are (b) and (c), and they have to be read together.
    (b) plots cost against distance-to-the-best-solution: a single funnel converging on one
    point is a unimodal landscape found repeatedly, while a horizontal band -- equally good
    solutions at every distance -- is a multimodal one. (c) shows the same information as the
    full pairwise matrix, which is what distinguishes 'spread out around one optimum' from
    'several tight clusters'.

    Distances are Euclidean in `v`, which lives in the GAUGE QUOTIENT, so two parameter sets
    that differ only by a change of units are at distance 0 (REPO_MAP hazard 6). Without that,
    a gauge motion would read as a distinct basin and every campaign would look multimodal.
    """
    # A PER-TASK `seeds_*.npz` -- the one `run_seeds._compare_seeds` writes -- has the same
    # name and the same first four keys as an aggregated one, so `_load` will happily hand one
    # over. It carries only (seeds, v, cost, dist): no verdicts and no surfaces, which is the
    # whole reason fit/aggregate.py exists. Say that, rather than dying on a KeyError six
    # panels in.
    missing = [k for k in ('verdict', 'ptc_fit', 'labels', 'ptc_target_used') if k not in z]
    if missing:
        raise SystemExit(
            f"this looks like a PER-TASK seeds npz, not an aggregated one (missing "
            f"{', '.join(missing)}). It compares only the seeds that shared one array task. "
            f"Run `python -m fit.aggregate --model {_CTX['model']} --tag <campaign>` and plot "
            f"the tag it writes.")
    lab, cost = z['labels'], np.asarray(z['cost'], float)
    D, ok = np.asarray(z['dist'], float), _usable(z)
    n = len(lab)
    bi = best_index(z)
    order = np.argsort(cost)
    cols = _vcolors(z)
    # every seed shares one grid (`collect` refuses to join runs that do not), so
    # row 0 is the campaign's dose axis
    doses, old = np.asarray(z['doses'])[0], np.asarray(z['old'])
    base_cost = float(np.nanmedian(z['base__parts_total']))

    fig = plt.figure(figsize=(19.5, 15.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95], hspace=0.46, wspace=0.30)

    # (a) cost per run, sorted --------------------------------------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(n), cost[order], color=[cols[i] for i in order], edgecolor='0.25', lw=0.5)
    ax.axhline(base_cost, color='k', ls='--', lw=1.2)
    ax.text(-0.4, base_cost, f'base {base_cost:.3f} ', va='bottom', ha='left', fontsize=8,
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(lab[i]) for i in order], fontsize=7.5)
    ax.set_yscale('log')
    ax.set_xlabel(f"{z['key']} (sorted by cost)"); ax.set_ylabel('final cost')
    ax.set_title('(a) every search, ranked', fontsize=10)
    _verdict_legend(ax, z, fontsize=7, loc='upper left', framealpha=0.9)

    # (b) THE MULTIMODALITY PANEL ------------------------------------------------------ #
    ax = fig.add_subplot(gs[0, 1])
    for i in range(n):
        ax.plot(D[i, bi], cost[i], 'o', ms=9, mfc=cols[i], mec='k', mew=0.6, zorder=3)
        ax.annotate(str(lab[i]), (D[i, bi], cost[i]), fontsize=7, xytext=(4, 4),
                    textcoords='offset points')
    if ok.sum() >= 2:
        cu = cost[ok]
        ax.axhspan(cu.min(), cu.max(), color='#1a7f37', alpha=0.09, zorder=0)
        ax.text(0.02, 0.11, f'usable band {cu.min():.3f}-{cu.max():.3f} '
                            f'({cu.max() / cu.min():.2f}x)',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=8, color='#1a7f37')
    ax.set_yscale('log')
    ax.set_xlabel('quotient distance to the best usable solution')
    ax.set_ylabel('final cost')
    ax.set_title('(b) a funnel = one optimum;  a flat band = MULTIMODAL', fontsize=10)

    # (c) the full pairwise matrix ----------------------------------------------------- #
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(D[np.ix_(order, order)], cmap='viridis', origin='upper')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([str(lab[i]) for i in order], fontsize=6.5, rotation=90)
    ax.set_yticklabels([str(lab[i]) for i in order], fontsize=6.5)
    for k, i in enumerate(order):
        c = VERDICT_COLOR.get(str(z['verdict'][i]), '0.5')
        ax.plot([-0.85], [k], marker='s', ms=5, color=c, clip_on=False)
    fig.colorbar(im, ax=ax, fraction=0.046, label='quotient distance')
    ax.set_title('(c) pairwise distance (cost order)', fontsize=10)

    # (d) descent traces --------------------------------------------------------------- #
    ax = fig.add_subplot(gs[1, 0])
    tr = np.asarray(z['trace_f'], float)
    for i in range(n):
        t = tr[i][np.isfinite(tr[i])]
        if t.size:
            # RUNNING MINIMUM, not the raw evaluations. CMA samples a population, so the raw
            # trace is dominated by the spread of the population rather than by progress.
            ax.plot(np.arange(t.size), np.minimum.accumulate(t), lw=1.0, color=cols[i],
                    alpha=0.85)
    ax.axhline(base_cost, color='k', ls='--', lw=1.0)
    ax.set_xlabel('evaluation'); ax.set_ylabel('best cost so far')
    ax.set_yscale('log')
    ax.set_title('(d) descent -- all searches stopped on the budget, none converged',
                 fontsize=10)

    # (e) twist curves, USABLE ONLY ---------------------------------------------------- #
    ax = fig.add_subplot(gs[1, 1])
    twist_panel(ax, doses, np.asarray(z['twist_base'])[0], is_base=True, label='base',
                scrit=float(np.nanmedian(z['base__S_crit'])))
    for i in np.where(ok)[0]:
        ax.plot(*broken(doses, np.asarray(z['twist_fit'])[i]), lw=1.1, alpha=0.8,
                color=VERDICT_COLOR.get(str(z['verdict'][i]), '0.5'))
    kk = float(np.nanmedian(z['k_used']))
    if np.isfinite(kk) and kk > 0:
        ax.axvline(1.0 / kk, color='#0969da', ls='-.', lw=1.2)
        ax.text(1.0 / kk, 1.01, ' target defect', color='#0969da', fontsize=7, va='bottom')
    ab = float(np.nanmedian(z['base__accum_twist']))
    au = np.asarray(z['fit__accum_twist'])[ok]
    ax.legend(fontsize=7, loc='lower left', framealpha=0.9)
    ax.set_title(f"(e) isochron twist -- flat is the goal\n"
                 f"accumulated: base {ab:.2f} cyc -> fits "
                 f"{np.nanmin(au):.2f}-{np.nanmax(au):.2f} cyc", fontsize=9.5)

    # (f) WHERE IN DOSE THE RESIDUAL ACTUALLY LIVES ------------------------------------ #
    #
    # THE PANEL THAT SAYS WHAT THE COST BOUGHT, and the one that stops a low number from being
    # read as a good fit. The radial target is only INFORMATIVE below its own singularity: a
    # Poincare surface at dose >> S* resets to nearly the same phase whatever the old phase
    # was, so its old-phase structure decays away (grey curve). Above S* almost any strongly
    # resetting surface scores well, and on a log-spaced window most of the CELLS live there.
    #
    # So a fit can drive the total residual down without ever matching the part of the target
    # that carries isochron geometry. Split by dose, that is immediately visible; pooled into
    # one c_ptc it is invisible. Read this panel before quoting a cost.
    ax = fig.add_subplot(gs[1, 2])
    tgt = np.asarray(z['ptc_target_used'])[0]
    S_t = 1.0 / kk if (np.isfinite(kk) and kk > 0) else np.nan

    def _rms_vs_dose(p):
        d = np.abs(((np.asarray(p, float) - tgt) + 0.5) % 1.0 - 0.5)
        return np.sqrt(np.nanmean(d ** 2, axis=0))

    for i in np.where(ok)[0]:
        ax.plot(doses, _rms_vs_dose(np.asarray(z['ptc_fit'])[i]), lw=1.1, alpha=0.8,
                color=VERDICT_COLOR.get(str(z['verdict'][i]), '0.5'))
    ax.plot(doses, _rms_vs_dose(np.asarray(z['ptc_base'])[0]), color='k', lw=2.6, zorder=6,
            label='base')
    # how much old-phase structure the TARGET still has at each dose -- its information content
    ax.plot(doses, _span_vs_dose(tgt), color='0.45', ls='--', lw=1.6, zorder=5,
            label="target's own old-phase span")
    if np.isfinite(S_t):
        ax.axvline(S_t, color='#0969da', ls='-.', lw=1.2)
        ax.axvspan(doses.min(), S_t, color='#0969da', alpha=0.07)
        n_lo = int(np.sum(doses <= S_t))
        ax.text(0.02, 0.02, f'{n_lo} of {len(doses)} doses below S*:\nthe only ones where the'
                            f'\ntarget still has structure',
                transform=ax.transAxes, va='bottom', fontsize=7.5, color='#0969da')
    ax.set_xscale('log'); ax.set_xlabel('dose'); ax.set_ylabel('rms |fit - target| (cyc)')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('(f) the residual is NOT reduced where the target is informative',
                 fontsize=9.5)

    # (h) the numbers ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[2, 2]); ax.axis('off')
    lines = [f"campaign  {z['campaign_tag']}",
             f"probe     {z['target']} / {z['mode']}   readout {z['readout']}, "
             f"section {z['section']}",
             f"grid      {len(old)} phase x {len(doses)} dose  "
             f"[{doses.min():.4g}, {doses.max():.4g}]", '']
    for v in VERDICTS:
        c = int(np.sum(z['verdict'] == v))
        if c:
            lines.append(f"{v:<14s} {c:2d} / {n}")
    lines.append('')
    if ok.sum() >= 2:
        cu = cost[ok]
        off = D[np.ix_(ok, ok)][np.triu_indices(int(ok.sum()), 1)]
        lines += [f"usable cost    {cu.min():.4f} - {cu.max():.4f}  ({cu.max() / cu.min():.2f}x)",
                  f"base cost      {base_cost:.4f}",
                  f"quotient dist  min {off.min():.2f}  med {np.median(off):.2f}  "
                  f"max {off.max():.2f}",
                  f"search box     |v| <= 3 per axis in {z['v'].shape[1]}D", '',
                  ('VERDICT: MULTIMODAL -- equally good' if np.median(off) > 1.0 else
                   'VERDICT: ONE CLUSTER'),
                  ('solutions far apart in parameter space.' if np.median(off) > 1.0 else
                   'the searches agree.')]
    if np.isfinite(S_t):
        lo = doses <= S_t
        rl = np.array([np.nanmean(_rms_vs_dose(np.asarray(z['ptc_fit'])[i])[lo])
                       for i in np.where(ok)[0]])
        rh = np.array([np.nanmean(_rms_vs_dose(np.asarray(z['ptc_fit'])[i])[~lo])
                       for i in np.where(ok)[0]])
        rb = _rms_vs_dose(np.asarray(z['ptc_base'])[0])
        lines += ['', f"residual below S*  base {np.nanmean(rb[lo]):.3f} -> "
                      f"{np.nanmean(rl):.3f}",
                  f"residual above S*  base {np.nanmean(rb[~lo]):.3f} -> "
                  f"{np.nanmean(rh):.3f}"]
    nz = int(np.sum(np.asarray(z['fit__n_sing'])[ok] == 0))
    if nz:
        lines += ['', f"CAVEAT: {nz}/{int(ok.sum())} usable fits have NO",
                  "singularity left in the window. The target",
                  "HAS one by construction, so those fits do",
                  "not match its topology -- 'flat' and 'the",
                  "transition moved out' score alike here."]
    # A PTC that does not depend on old phase carries NO phase information at all: the
    # perturbation resets the clock to the same place whenever it is applied. That surface has
    # zero twist by construction and passes every guard the cost has -- the oscillator is alive,
    # the amplitude is fine, the Floquet multiplier is fine -- so it must be counted explicitly.
    fl = np.array([np.nanmean(_span_vs_dose(p)) for p in np.asarray(z['ptc_fit'])[ok]])
    nfl = int(np.sum(fl < FLAT_SPAN))
    if nfl:
        lines += ['', f"CAVEAT: {nfl}/{int(ok.sum())} usable fits have a PTC",
                  f"essentially INDEPENDENT of old phase",
                  f"(mean span < {FLAT_SPAN} cyc). Such a surface has",
                  "no phase information and zero twist by",
                  "construction, yet passes every liveness guard."]
    nf = int(np.sum(z['twist_sign_flip']))
    if nf:
        lines += ['', f"CAVEAT: {nf} run(s) flip the twist SIGN.",
                  f"On {len(doses)} dose samples that is the",
                  "aliasing signature of 5.9c, not direction."]
    ax.text(0.0, 1.0, "\n".join(lines), family='monospace', fontsize=8.4, va='top',
            transform=ax.transAxes)

    # (g) which parameters are pinned across seeds? ------------------------------------- #
    #
    # THE IDENTIFIABILITY PANEL, and the reason the campaign is worth its CPU-hours. If the
    # searches disagree everywhere, "multimodal" is all there is to say; if they agree on SOME
    # parameters while disagreeing on others, then those are the combinations the PTC actually
    # determines, and the disagreement measures the width of the flat floor (5.1). Values are
    # gauge-fixed representatives -- theta is reconstructed from `v` in the quotient, so equal
    # `v` means equal theta and there is no unit freedom left to confound the spread.
    ax = fig.add_subplot(gs[2, 0:2])
    th_b = np.asarray(z['theta_base'], float)[0]
    th_f = np.asarray(z['theta_fit'], float)[ok]
    names = [str(s) for s in z['names']]
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.log10(th_f / th_b[None, :])
    spread = np.nanmax(lr, 0) - np.nanmin(lr, 0)
    po = np.argsort(spread)
    for j, p in enumerate(po):
        ax.plot([j, j], [np.nanmin(lr[:, p]), np.nanmax(lr[:, p])], color='0.75', lw=6,
                solid_capstyle='round', zorder=1)
        ax.plot(np.full(lr.shape[0], j), lr[:, p], 'o', ms=4.5, mfc='#0969da', mec='k',
                mew=0.4, alpha=0.85, zorder=3)
    ax.axhline(0, color='k', lw=1.0)
    # THE SCALE THAT MAKES THE PANEL READABLE. PROJECT_SUMMARY 5.1 measured that a +-26%
    # Almeida is distinguishable from nominal by one gene's PTC (rank 16/16, condition 65) --
    # so this band is the resolution the observable is supposed to have. Solutions scattered
    # over one to two DECADES while costing within 1.6x of each other are far outside it.
    ax.axhspan(-np.log10(1.26), np.log10(1.26), color='#0969da', alpha=0.12, zorder=0)
    ax.text(0.995, 0.985, "+-26% band -- the scale 5.1 says one gene's PTC can resolve",
            transform=ax.transAxes, fontsize=8.5, color='#0969da', ha='right', va='top')
    ax.set_xticks(range(len(po)))
    ax.set_xticklabels([names[p] for p in po], rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('log10(fitted / base)')
    ax.set_title(f"(g) where the {int(ok.sum())} usable solutions AGREE and where they do not "
                 f"-- narrow = the PTC pins it, wide = the flat floor "
                 f"(left-to-right: increasing disagreement)", fontsize=10)

    fig.suptitle(f"{z['model']}/{z['target']} ({z['mode']}) radial fit -- {n} independent "
                 f"seeds  [{z['campaign_tag']}]", fontsize=13)
    return _save(fig, f"seeds_{z['target']}_{z['mode']}", target=str(z['target']),
                 n_runs=int(n), usable=int(ok.sum()))


def _surface_grid(fig, gs, panels, old, doses, ncol):
    """A grid of PTC surfaces sharing one colour scale. `panels` is [(title, ptc, color), ...]."""
    im = None
    for j, (title, ptc, tc) in enumerate(panels):
        ax = fig.add_subplot(gs[j // ncol, j % ncol])
        im = ax.pcolormesh(old, doses, np.ma.masked_invalid(np.asarray(ptc).T),
                           cmap=phase_cmap(), vmin=0, vmax=1, shading='nearest')
        ax.set_yscale('log')
        ax.set_title(title, fontsize=8.5, color=tc)
        ax.tick_params(labelsize=6.5)
        if j % ncol:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('dose', fontsize=8)
        if j // ncol == (len(panels) - 1) // ncol:
            ax.set_xlabel('old phase', fontsize=8)
    return im


def fig_seed_surfaces(z):
    """Every fitted PTC in the campaign, beside the base and the target it was fitted TO.

    The distances in `fig_seeds` say the solutions are far apart in PARAMETER space. This says
    whether they are far apart as PHASE RESPONSES, which is the physical question and need not
    have the same answer: a flat cost floor can be a set of genuinely different surfaces or one
    surface reached by different parameter sets, and only the picture distinguishes them.

    Ordered by cost, and captioned with the verdict, so a good-looking surface with a failing
    verdict cannot be read as a result.
    """
    old, doses = np.asarray(z['old']), np.asarray(z['doses'])[0]
    cost = np.asarray(z['cost'], float)
    order = np.argsort(cost)
    n = len(order)
    panels = [('base   c=%.3f' % np.nanmedian(z['base__parts_total']),
               np.asarray(z['ptc_base'])[0], 'k'),
              ('RADIAL TARGET (pinned)', np.asarray(z['ptc_target_used'])[0], '#0969da')]
    for i in order:
        panels.append((f"{z['key']} {z['labels'][i]}   c={cost[i]:.3f}\n{z['verdict'][i]}",
                       np.asarray(z['ptc_fit'])[i],
                       VERDICT_COLOR.get(str(z['verdict'][i]), '0.5')))
    ncol = 6
    nrow = int(np.ceil(len(panels) / ncol))
    fig = plt.figure(figsize=(3.05 * ncol, 3.15 * nrow))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.55, wspace=0.18)
    im = _surface_grid(fig, gs, panels, old, doses, ncol)
    fig.colorbar(im, ax=fig.axes, fraction=0.014, pad=0.012, label='new phase (cyc)')
    fig.suptitle(f"{z['model']}/{z['target']} ({z['mode']}) -- {n} independent seeds, every "
                 f"fitted PTC surface  [{z['campaign_tag']}]", fontsize=12)
    return _save(fig, f"seeds_{z['target']}_{z['mode']}_surfaces",
                 target=str(z['target']), n_runs=int(n))


def fig_genes(z):
    """One gene per column: can each probe's PTC be radialized, and at what cost to the clock?

    A per-gene campaign varies the TARGET, so unlike a seed campaign there is no distance
    matrix to draw -- two runs scored against two different pinned targets have incomparable
    costs and no meaningful separation. What IS comparable is the shape of the answer: how far
    each surface moved, what happened to its twist, and what the fit did to the oscillator.

    THE BOTTOM ROW IS THE CONTROL AND IT IS NOT DECORATION. A radial-looking PTC bought by
    stretching the period to 46 h or by leaving the attracting cycle altogether is not a
    radialized clock, and the surface panels alone cannot show that -- which is the failure
    mode this whole pipeline was built against (PROJECT_SUMMARY 5.2, 5.7).
    """
    # PER-GENE dose axes: `fit_dose_grid` derives the window from each target's own
    # S_crit, so these differ by up to 60x and MUST NOT share one axis
    old, DOSES = np.asarray(z['old']), np.asarray(z['doses'])
    tg = [str(t) for t in np.atleast_1d(z['target'])]
    n = len(tg)
    order = np.argsort(np.asarray(z['cost'], float))

    fig = plt.figure(figsize=(4.3 * n + 1.0, 15.5))
    gs = fig.add_gridspec(4, n + 1, height_ratios=[1, 1, 1, 1.05],
                          width_ratios=[1] * n + [0.05], hspace=0.40, wspace=0.30)
    im = None
    for c, i in enumerate(order):
        vc = VERDICT_COLOR.get(str(z['verdict'][i]), '0.5')
        doses = DOSES[i]                          # THIS gene's window, not the campaign's first
        for r, (lab, ptc) in enumerate((('base', np.asarray(z['ptc_base'])[i]),
                                        ('radial target (pinned)',
                                         np.asarray(z['ptc_target_used'])[i]),
                                        ('fitted', np.asarray(z['ptc_fit'])[i]))):
            ax = fig.add_subplot(gs[r, c])
            im = ax.pcolormesh(old, doses, np.ma.masked_invalid(ptc.T), cmap=phase_cmap(),
                               vmin=0, vmax=1, shading='nearest')
            ax.set_yscale('log'); ax.tick_params(labelsize=7)
            # each panel's OWN S_crit -- one line at one height across three panels would be a
            # false claim, and the whole point is that the singularity MOVES (fig_radial's own
            # note, for the same reason)
            s = (float(z['base__S_crit'][i]) if r == 0 else
                 (1.0 / float(z['k_used'][i]) if r == 1 else float(z['fit__S_crit'][i])))
            if np.isfinite(s) and s > 0:
                ax.axhline(s, color='w', ls=':', lw=1.2)
                ax.text(0.02, s, f'S={s:.3g}', color='w', fontsize=6.5, va='bottom')
            elif r == 2:
                ax.text(0.5, 0.06, 'no singularity in window', transform=ax.transAxes,
                        ha='center', fontsize=7.5, color='w')
            # EVERY panel gets a dose label, because every COLUMN has its own dose axis --
            # labelling only the left column would invite the reader to assume a shared one
            ax.set_ylabel(f'{lab}\ndose' if c == 0 else 'dose', fontsize=8.5)
            if r == 0:
                ax.set_title(f"{tg[i]}   dose {doses.min():.4g}-{doses.max():.4g}\n"
                             f"c {z['base__parts_c_ptc'][i]:.3f} -> "
                             f"{z['fit__parts_c_ptc'][i]:.3f}   {z['verdict'][i]}",
                             fontsize=10, color=vc)
            if r == 2:
                ax.set_xlabel('old phase', fontsize=8.5)

        ax = fig.add_subplot(gs[3, c])
        twist_panel(ax, doses, np.asarray(z['twist_fit'])[i],
                    base=np.asarray(z['twist_base'])[i], color=vc, label='fitted',
                    scrit=float(z['base__S_crit'][i]))
        ax.legend(fontsize=7, loc='upper right')
        tb, tf = float(z['base__accum_twist'][i]), float(z['fit__accum_twist'][i])
        flip = '  SIGN FLIP (aliased?)' if bool(z['twist_sign_flip'][i]) else ''
        # HOW MUCH OF THIS GENE'S WINDOW THE TARGET ACTUALLY CONSTRAINS.
        #
        # The window comes from the fixture S_crit and the target is pinned to the base
        # surface's MEASURED singularity, and the two do not have to agree -- for REV they
        # differ by 1.8x, which puts 13 of 14 doses above the defect, where a Poincare target
        # has almost no old-phase structure left. Costs from different genes are therefore NOT
        # comparable without this number: a low residual against a nearly featureless target
        # is close to vacuous, and REV's 0.007 is the cheapest number in the campaign for
        # exactly that reason.
        sp = _span_vs_dose(np.asarray(z['ptc_target_used'])[i])
        n_inf = int(np.sum(sp > INFORMATIVE_SPAN))
        ax.set_title(f"twist {tb:.3f} -> {tf:.3f} cyc accumulated{flip}\n"
                     f"T {float(z['base__parts_period'][i]):.1f} -> "
                     f"{float(z['fit__parts_period'][i]):.1f} h    "
                     f"mu {float(z['fit__mu'][i]):.3g}    "
                     f"Re(lam) {float(z['fit__parts_re_lambda'][i]):+.4f}\n"
                     f"target informative on {n_inf}/{len(sp)} doses"
                     + ('\nNEARLY FEATURELESS TARGET -- cost is cheap here'
                        if n_inf <= len(sp) // 4 else ''),
                     fontsize=8, color=vc)
    fig.colorbar(im, cax=fig.add_subplot(gs[0:3, n]), label='new phase (cyc)')
    fig.suptitle(f"{z['model']} ({z['mode']}) radial fit, one seed per gene -- which probes "
                 f"can be radialized  [{z['campaign_tag']}]", fontsize=13)
    return _save(fig, f"genes_{z['mode']}", n_runs=int(n))


def _load(model, analysis, pattern, tag=None):
    tag = tag or paths.latest_run(model, analysis)
    if tag is None:
        return None, None
    d = paths.out_dir(model, analysis, tag, create=False)
    fs = sorted(glob.glob(os.path.join(d, pattern)))
    if not fs:
        return None, tag
    return dict(np.load(fs[-1], allow_pickle=True)), tag


def _spans_for(z):
    """The fitted PTCs' old-phase spans, matched to a stability audit BY LABEL.

    Returns None when the aggregated seeds npz is not beside the stability one, so the panel is
    simply omitted rather than the figure failing. Matched by label and not by position: the
    two files are written by different drivers and there is no guarantee they order their runs
    the same way -- silently zipping them would mislabel every point on the panel.
    """
    d = paths.out_dir(_CTX['model'], _CTX['analysis'], _CTX['tag'], create=False)
    fs = sorted(glob.glob(os.path.join(d, 'seeds_*.npz')))
    if not fs:
        return None
    a = dict(np.load(fs[-1], allow_pickle=True))
    if 'ptc_fit' not in a or 'labels' not in a:
        return None
    by = {str(l): i for i, l in enumerate(a['labels'])}
    out = np.full(len(z['labels']), np.nan)
    for k, l in enumerate(z['labels']):
        j = by.get(str(l))
        if j is not None:
            out[k] = float(np.nanmean(_span_vs_dose(np.asarray(a['ptc_fit'][j], float))))
    return out


def fig_cycles(z):
    """THE LIMIT CYCLE PER RUN, plus what happens when you step off it.

    A PTC is a statement about asymptotic phase, and asymptotic phase only exists on an
    ATTRACTING orbit. So the cycle panels alone are not the diagnostic -- one period of a
    repelling orbit, of an orbit that decays to a fixed point, and of a healthy clock look
    identical, all three with a circadian period, a full amplitude and positive concentrations
    (fit/stability, THE FAILURE THIS EXISTS TO CATCH). Panel (b) is what separates them: the
    transverse distance from a perturbed start back to the cycle, period by period, on a log
    axis. Decaying lines attract; flat or rising lines do not.

    Every trace is scaled to its own [min, max] with the absolute range in the legend, for the
    reason fig_radial gives: Almeida's species differ by up to fifty decades at an optimizer's
    parameter set, and shared axes render all but the largest as a flat line at zero, which
    reads as "the clock stopped" whether or not it did.
    """
    from fit.stability import VERDICT_COLOR as SC, VERDICTS as SV, FLOOR_FRAGILE
    lab = [str(x) for x in z['labels']]
    obs = [str(x) for x in z['obs']]
    n = len(lab)
    T, growth, verd = z['period'], z['growth'], z['verdict']
    order = np.argsort(np.where(np.isfinite(T), T, np.inf))
    cols = [SC.get(str(v), '0.5') for v in verd]

    ncol = 4
    nrow_c = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(4.6 * ncol, 3.0 * nrow_c + 13.0))
    gs = fig.add_gridspec(nrow_c + 4, ncol, hspace=0.78, wspace=0.28,
                          height_ratios=[1] * nrow_c + [1.25, 1.25, 1.15, 1.25])

    # --- (a) one period of every fitted limit cycle ---------------------------------- #
    for j, i in enumerate(order):
        ax = fig.add_subplot(gs[j // ncol, j % ncol])
        cyc = np.asarray(z['cyc'][i], float)
        if not np.isfinite(cyc).all() or not np.isfinite(T[i]):
            ax.axis('off')
            ax.text(0.5, 0.5, f"{z['key']} {lab[i]}\n{verd[i]}", ha='center', va='center',
                    color=cols[i], fontsize=9)
            continue
        t = np.linspace(0, T[i], cyc.shape[0])
        for k in range(cyc.shape[1]):
            y = cyc[:, k]
            lo, hi = float(y.min()), float(y.max())
            sp = hi - lo
            ax.plot(t, (y - lo) / sp if sp > 0 else np.zeros_like(y), lw=1.1,
                    label=f'{obs[k]} [{lo:.2g}, {hi:.2g}]')
        # MEDIAN relative amplitude across species, not the max. `amp_lc` is range/|mean| on
        # ONE species, and a species whose baseline falls toward zero inflates it without the
        # oscillation growing at all -- the measurement that wrongly declared RAD03 degenerate
        # (PROJECT_SUMMARY 5.7). The max over species inherits that defect; the median does not,
        # so both are shown and the median is the one to read.
        rr = np.asarray(z['relamp_species'][i], float)
        rr = rr[np.isfinite(rr)]
        rmed = float(np.median(rr)) if rr.size else np.nan
        rmax = float(np.max(rr)) if rr.size else np.nan
        ax.set_title(f"{z['key']} {lab[i]}   T = {T[i]:.2f} h\n"
                     f"diam {z['amp_cycle'][i]:.3g}   rel amp {rmed:.2f} (max {rmax:.1f})\n"
                     f"growth {growth[i]:.3f}   {verd[i]}", fontsize=8, color=cols[i])
        ax.set_xlabel('time (h)', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        if j % ncol == 0:
            ax.set_ylabel('each species on its own scale', fontsize=8)
        if j == 0:
            ax.legend(fontsize=5.2, ncol=2, loc='upper right', framealpha=0.85)

    # --- (b) THE STABILITY PANEL ------------------------------------------------------ #
    ax = fig.add_subplot(gs[nrow_c, 0:2])
    for i in range(n):
        d = np.asarray(z['walk_dist'][i], float) / max(float(z['diam'][i]), 1e-30)
        d = d[np.isfinite(d)]
        if len(d):
            ax.plot(np.arange(1, len(d) + 1), d, lw=1.3, color=cols[i], alpha=0.9)
            ax.annotate(lab[i], (len(d), d[-1]), fontsize=6.5, color=cols[i],
                        xytext=(3, 0), textcoords='offset points', va='center')
        f = np.asarray(z['floor_dist'][i], float)
        f = f[np.isfinite(f)]
        if len(f):
            ax.plot(np.arange(1, len(f) + 1), f, lw=0.8, ls=':', color=cols[i], alpha=0.55)
    ax.set_yscale('log')
    ax.set_xlabel('periods after a small displacement off the cycle')
    ax.set_ylabel('distance back to the cycle / cycle diameter')
    ax.set_title('(b) STEP OFF THE ORBIT AND WATCH. Falling = attracting; flat or rising = not.\n'
                 'Dotted: the same walk with NO displacement -- that run\'s numerical floor.',
                 fontsize=9.5)
    ax.legend(handles=[matplotlib.patches.Patch(facecolor=SC[v], label=v)
                       for v in SV if np.any(verd == v)], fontsize=7, loc='lower left',
              ncol=2, framealpha=0.9)

    # --- (c) period vs amplitude, the two measures asked for -------------------------- #
    ax = fig.add_subplot(gs[nrow_c, 2])
    for i in range(n):
        if np.isfinite(T[i]):
            ax.plot(T[i], z['amp_cycle'][i], 'o', ms=9, mfc=cols[i], mec='k', mew=0.6)
            ax.annotate(lab[i], (T[i], z['amp_cycle'][i]), fontsize=7, xytext=(4, 3),
                        textcoords='offset points')
    ax.axvspan(24.83 * 0.7, 24.83 * 1.4, color='#0969da', alpha=0.08)
    ax.axvline(24.83, color='#0969da', ls='-.', lw=1.2)
    ax.text(24.83, 1.0, ' base 24.83 h', color='#0969da', fontsize=7.5, rotation=90,
            va='bottom', transform=ax.get_xaxis_transform())
    ax.set_yscale('log')
    ax.set_xlabel('period (h)'); ax.set_ylabel('cycle amplitude (state-space diameter)')
    ax.set_title('(c) period vs amplitude\nband = the circadian range', fontsize=9.5)

    # --- (d) the numerical floor, which is a result in its own right ------------------ #
    ax = fig.add_subplot(gs[nrow_c, 3])
    o2 = np.argsort(np.where(np.isfinite(z['floor']), z['floor'], -1))
    ax.barh(range(n), np.maximum(np.asarray(z['floor'])[o2], 1e-12),
            color=[cols[i] for i in o2], edgecolor='0.25', lw=0.4)
    ax.axvline(FLOOR_FRAGILE, color='#b31d28', ls='--', lw=1.2)
    ax.set_yticks(range(n)); ax.set_yticklabels([lab[i] for i in o2], fontsize=7)
    ax.set_xscale('log'); ax.set_xlabel('numerical floor (cycle diameters)')
    ax.set_title('(d) how cleanly each orbit\ncan be integrated AT ALL', fontsize=9.5)

    # --- (e) per-species amplitude: which species actually oscillate ------------------- #
    ax = fig.add_subplot(gs[nrow_c + 1, 0:3])
    A = np.asarray(z['amp_species'], float)
    x = np.arange(len(obs))
    for j, i in enumerate(order):
        if np.isfinite(A[i]).all():
            ax.plot(x + (j - n / 2) * 0.012, np.maximum(A[i], 1e-60), 'o', ms=5,
                    mfc=cols[i], mec='none', alpha=0.85)
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(obs, rotation=30, ha='right')
    ax.set_ylabel('absolute amplitude on the cycle')
    ax.set_title('(e) per-species amplitude. A species pinned at 1e-20 is the hazard-14 '
                 'collapse -- the orbit is real but the state spans tens of decades',
                 fontsize=9.5)

    # --- (f) the numbers -------------------------------------------------------------- #
    ax = fig.add_subplot(gs[nrow_c + 1, 3]); ax.axis('off')
    lines = [f"{z['campaign_tag']}  ({n} runs)", f"walk: {int(z['n_blocks'])} periods, "
             f"{int(z['ndir'])} directions", '']
    for v in SV:
        c = int(np.sum(verd == v))
        if c:
            lines.append(f"{v:<14s} {c:2d}/{n}")
    good = np.array([str(v) == 'ATTRACTING' for v in verd])
    if good.any():
        lines += ['', f"attracting: growth {np.nanmin(growth[good]):.3f}-"
                      f"{np.nanmax(growth[good]):.3f} /period",
                  f"periods {np.nanmin(T[good]):.1f}-{np.nanmax(T[good]):.1f} h"]
    nf = int(np.sum(np.asarray(z['floor']) > FLOOR_FRAGILE))
    if nf:
        lines += ['', f"{nf} orbit(s) have a floor above",
                  f"{FLOOR_FRAGILE:.0e} diameters -- nothing",
                  "measured on them is precise."]
    ax.text(0.0, 1.0, "\n".join(lines), family='monospace', fontsize=8.5, va='top',
            transform=ax.transAxes)

    # --- (g) amplitude over the walk: does the oscillation survive? -------------------- #
    ax = fig.add_subplot(gs[nrow_c + 2, :])
    for i in range(n):
        a = np.asarray(z['walk_amp'][i], float)
        a = a[np.isfinite(a)]
        if len(a) and np.isfinite(z['amp_cycle'][i]) and z['amp_cycle'][i] > 0:
            ax.plot(np.arange(1, len(a) + 1), a / z['amp_cycle'][i], lw=1.3, color=cols[i])
            ax.annotate(lab[i], (len(a), a[-1] / z['amp_cycle'][i]), fontsize=6.5,
                        color=cols[i], xytext=(3, 0), textcoords='offset points', va='center')
    ax.axhline(1.0, color='k', lw=0.8, ls=':')
    ax.set_yscale('log'); ax.set_xlabel('periods after the displacement')
    ax.set_ylabel('oscillation amplitude / cycle amplitude')
    ax.set_title('(g) DOES IT STILL OSCILLATE? A line diving to 1e-20 is a clock that stopped: '
                 'the solved orbit is real, and the system left it for a fixed point.',
                 fontsize=9.5)

    # --- (h) IS A FLAT PTC AN UNSTABLE ORBIT? ----------------------------------------- #
    #
    # The question the whole audit was run to answer. The campaign's fits are nearly all type-0
    # with almost no old-phase structure (PROJECT_SUMMARY 5.10d); if that were a stability
    # artefact, the flattest surfaces would be the unstable ones. Joined here from the
    # aggregated seeds npz sitting beside this one, so both axes come from saved data and
    # nothing is recomputed.
    sp = _spans_for(z)
    if sp is not None:
        ax = fig.add_subplot(gs[nrow_c + 3, :])
        for i in range(n):
            if not np.isfinite(sp[i]):
                continue
            known = np.isfinite(growth[i])
            gx = growth[i] if known else 1.06
            ax.plot(sp[i], gx, 'o', ms=11, mec='k', mew=0.7,
                    mfc=(cols[i] if known else 'white'))
            ax.annotate(lab[i], (sp[i], gx), fontsize=8, xytext=(6, 5),
                        textcoords='offset points')
        ax.axhline(1.0, color='#b31d28', ls='--', lw=1.2)
        ax.text(0.995, 1.005, 'growth = 1: the stability boundary ', color='#b31d28',
                fontsize=8.5, ha='right', va='bottom', transform=ax.get_yaxis_transform())
        ax.axvline(FLAT_SPAN, color='#0969da', ls=':', lw=1.4)
        ax.text(FLAT_SPAN, 0.93, '  <- phaseless PTC', color='#0969da', fontsize=8.5,
                transform=ax.get_xaxis_transform(), va='top')
        ax.set_xlabel("the fitted PTC's mean old-phase span (cyc):  0 = carries no phase "
                      "information at all,  0.5 = sweeps the full circle")
        ax.set_ylabel('per-period growth')
        ax.set_title('(h) THE QUESTION: is the flat type-0 a stability artefact?   '
                     'Open circles = growth not measurable (drawn at 1.06).', fontsize=10)

    fig.suptitle(f"{z['model']}/{z['target']} ({z['mode']}) -- fitted limit cycles and their "
                 f"stability  [{z['campaign_tag']}]", fontsize=13)
    return _save(fig, f"cycles_{z['target']}_{z['mode']}", n_runs=int(n),
                 attracting=int(np.sum(verd == 'ATTRACTING')))


def fig_rescan(z):
    """Every fitted surface again, FINER and reaching down to dose 0.

    WHAT THIS SETTLES. In the fit window nearly every fit looked flat and singularity-free, and
    that has two completely different explanations which the window cannot separate: the
    transition moved BELOW the window (an ordinary clock with a small S_crit) or the surface
    carries no phase information at all (a degeneracy -- PROJECT_SUMMARY 5.10d). Extending down
    decides it, because as dose -> 0 the PTC must approach the identity: a healthy clock has to
    become type-1 somewhere, wherever its S_crit sits.

    THE BOTTOM ROW IS THE CONTROL. The dose-0 row is the identity by construction (engine.ptc
    calibrates the phase origin there), so its deviation measures whether the readout was
    calibrated AT ALL at that parameter set. Where it is not, nothing else on that surface is a
    measurement -- and it is not a coincidence which runs fail it.

    The fit window is shaded on every panel, so what the optimizer actually saw is never
    confused with what the clock actually does.
    """
    old = np.asarray(z['old'])
    lab = [str(x) for x in z['labels']]
    n = len(lab)
    order = np.argsort([float(x) if str(x).lstrip('-').isdigit() else 0.0 for x in lab])
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(3.25 * ncol, 3.5 * nrow + 8.0))
    gs = fig.add_gridspec(nrow + 2, ncol, hspace=0.62, wspace=0.24,
                          height_ratios=[1] * nrow + [1.35, 1.35])

    im = None
    for j, i in enumerate(order):
        ax = fig.add_subplot(gs[j // ncol, j % ncol])
        d = np.asarray(z['doses'])[i]
        p = np.asarray(z['ptc'])[i]
        # dose 0 cannot go on a log axis; it is drawn as the bottom row at the grid's own floor
        pos = d > 0
        im = ax.pcolormesh(old, d[pos], np.ma.masked_invalid(p[:, pos].T), cmap=phase_cmap(),
                           vmin=0, vmax=1, shading='nearest')
        ax.set_yscale('log')
        fd = np.asarray(z['fit_doses'])[i]
        ax.axhspan(float(np.min(fd)), float(np.max(fd)), color='w', alpha=0.0)
        for yv in (float(np.min(fd)), float(np.max(fd))):
            ax.axhline(yv, color='w', ls='-', lw=1.6, alpha=0.9)
        s = float(z['S_crit'][i])
        if np.isfinite(s) and s > 0:
            ax.axhline(s, color='w', ls=':', lw=1.4)
        ie = float(z['identity_err'][i])
        col = '#b31d28' if (not np.isfinite(ie) or ie > ID_TOL) else 'k'
        ax.set_title(f"{z['key']} {lab[i]}   S*={s:.3g}   n_sing={int(z['n_sing'][i])}\n"
                     f"accum twist {z['accum'][i]:.2f} cyc   id-err {ie:.1e}",
                     fontsize=8, color=col)
        ax.tick_params(labelsize=6.5)
        if j % ncol == 0:
            ax.set_ylabel('dose', fontsize=8)
        if j // ncol == nrow - 1:
            ax.set_xlabel('old phase', fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=[fig.axes[k] for k in range(min(n, len(fig.axes)))],
                     fraction=0.012, pad=0.01, label='new phase (cyc)')

    # --- where the singularity actually is, against the window that was fitted --------- #
    ax = fig.add_subplot(gs[nrow, 0:3])
    for j, i in enumerate(order):
        fd = np.asarray(z['fit_doses'])[i]
        ax.plot([j, j], [float(np.min(fd)), float(np.max(fd))], lw=7, color='#0969da',
                alpha=0.30, solid_capstyle='butt')
        s = float(z['S_crit'][i])
        if np.isfinite(s) and s > 0:
            ax.plot(j, s, 'o', ms=9, mfc='#1a7f37', mec='k', mew=0.6, zorder=4)
        else:
            ax.plot(j, float(np.min(np.asarray(z['doses'])[i][np.asarray(z['doses'])[i] > 0])),
                    'x', ms=10, mec='#b31d28', mew=2.2, zorder=4)
    ax.set_yscale('log')
    ax.set_xticks(range(n)); ax.set_xticklabels([lab[i] for i in order], fontsize=7.5)
    ax.set_xlabel(str(z['key'])); ax.set_ylabel('dose')
    ax.set_title('the singularity vs the window it was fitted in\n'
                 'blue band = the fit window;  dot = S* found on the extended grid;  '
                 'x = still none', fontsize=9.5)

    # --- the dose-0 identity control --------------------------------------------------- #
    ax = fig.add_subplot(gs[nrow, 3:])
    ie = np.asarray(z['identity_err'], float)[order]
    ax.bar(range(n), np.where(np.isfinite(ie), np.maximum(ie, 1e-12), 1.0),
           color=['#b31d28' if (not np.isfinite(v) or v > ID_TOL) else '#1a7f37' for v in ie],
           edgecolor='0.25', lw=0.5)
    ax.axhline(ID_TOL, color='#b31d28', ls='--', lw=1.2)
    ax.set_yscale('log')
    ax.set_xticks(range(n)); ax.set_xticklabels([lab[i] for i in order], fontsize=7.5)
    ax.set_xlabel(str(z['key'])); ax.set_ylabel('|new phase - old phase| at dose 0 (cyc)')
    ax.set_title('THE CONTROL: dose 0 must return the identity.\n'
                 'Red = the phase readout is not calibrated at that parameter set',
                 fontsize=9.5)

    # --- twist, now on a grid fine enough to measure it -------------------------------- #
    ax = fig.add_subplot(gs[nrow + 1, 0:3])
    for i in order:
        d = np.asarray(z['doses'])[i]
        pos = d > 0
        ax.plot(*broken(d[pos], np.asarray(z['twist'])[i][pos]), lw=1.1, alpha=0.85)
    ax.set_xscale('log'); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('dose'); ax.set_ylabel('stable FP phase')
    ax.set_title(f"twist on {int(z['n_dose'])} dose samples over {float(z['decades']):g} "
                 f"decades -- the campaign measured it on 14 over 1.2", fontsize=9.5)

    ax = fig.add_subplot(gs[nrow + 1, 3:]); ax.axis('off')
    ns = np.asarray(z['n_sing'])
    okid = np.isfinite(ie) & (ie <= ID_TOL)
    lines = [f"{z['campaign_tag']}   {n} runs",
             f"grid  {int(z['n_phase'])} phase x {int(z['n_dose'])} dose, "
             f"{float(z['decades']):g} decades + dose 0", '',
             f"has a singularity on the extended grid : {int(np.sum(ns > 0)):2d}/{n}",
             f"still none, even down to dose 0        : {int(np.sum(ns == 0)):2d}/{n}", '',
             f"dose-0 identity holds (<{ID_TOL:g} cyc)       : {int(np.sum(okid)):2d}/{n}",
             f"readout NOT calibrated there           : {int(np.sum(~okid)):2d}/{n}", '',
             'A surface with no singularity even at',
             'dose 0 is not a clock with a small S*.',
             'It carries no phase information at all.']
    ax.text(0.0, 1.0, "\n".join(lines), family='monospace', fontsize=9, va='top',
            transform=ax.transAxes)

    fig.suptitle(f"{z['model']}/{z['target']} ({z['mode']}) -- fitted PTCs re-rendered finer "
                 f"and down to dose 0  [{z['campaign_tag']}]", fontsize=13)
    return _save(fig, f"rescan_{z['target']}_{z['mode']}", n_runs=int(n),
                 n_phase=int(z['n_phase']), n_dose=int(z['n_dose']))


#: --which -> (analysis dir, npz glob, plotter(s)). The GLOB MATTERS: a campaign tag directory
#: holds more than one npz (seeds_*, viability_*), and `_load` takes the LAST match, so a bare
#: '*.npz' would hand fig_radial a viability record and fail somewhere unhelpful.
_WHICH = {
    'radial': ('fit_radial', 'radial_*.npz', (lambda z: [fig_radial(z)])),
    'recover': ('fit_recover', '*.npz', (lambda z: [fig_recover(z)])),
    'seeds': ('fit_radial', 'seeds_*.npz', (lambda z: [fig_seeds(z), fig_seed_surfaces(z)])),
    'genes': ('fit_radial', 'genes_*.npz', (lambda z: [fig_genes(z)])),
    'cycles': ('fit_radial', 'stability_*.npz', (lambda z: [fig_cycles(z)])),
    'rescan': ('fit_radial', 'rescan_*.npz', (lambda z: [fig_rescan(z)])),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description='figures for the fitting runs')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--which', default='radial', choices=tuple(_WHICH),
                    help="'radial'/'recover': one run. 'seeds'/'genes': a CAMPAIGN, from the "
                         "npz written by `python -m fit.aggregate`.")
    ap.add_argument('--tag', default=None)
    ap.add_argument('--publish', action='store_true',
                    help='ALSO copy to docs/figures/<model>_<name>.png with a .source.txt. '
                         'Only for figures PROJECT_SUMMARY actually cites (hazard 13).')
    a = ap.parse_args(argv)
    analysis, pattern, plot = _WHICH[a.which]
    z, tag = _load(a.model, analysis, pattern, a.tag)
    if z is None:
        raise SystemExit(
            f"no {a.which} npz ({pattern}) for {a.model}/{analysis}"
            + (f" under tag {tag!r}" if tag else "")
            + (f"\nRun `python -m fit.aggregate --model {a.model} --tag <campaign> "
               f"--kind {a.which}` first." if a.which in ('seeds', 'genes') else ""))
    for k in ('model', 'target', 'mode', 'optimizer', 'campaign_tag', 'key',
              'section', 'readout'):
        if k in z and np.asarray(z[k]).ndim == 0:
            z[k] = str(z[k])
    _CTX.update(model=a.model, analysis=analysis, tag=tag, publish=a.publish)
    for p in plot(z):
        print(f"[fit.figures] -> {p}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
