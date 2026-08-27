"""
analysis/figures.py
===================
Every figure, regenerated from saved npz. PURE READ -- nothing here integrates, so a figure
can always be rebuilt without re-running a sweep, and a change to how something is drawn never
costs compute.

    python -m analysis.figures --model almeida --target BMAL1 [--which all]

Figures produced (into docs/figures/):

  scrit        winding vs dose for every target, with the validity ceiling shaded.
  surfaces     the base PTC surfaces, with singularities, S_crit and the twist curve.
  tornado      LC and PTC sensitivity per parameter, side by side on a shared ordering.
  lc_examples  what an LC sensitivity NUMBER actually looks like: base vs perturbed cycle
               profiles for the strongest, a middling and the weakest parameter.
  ptc_examples the same for the PTC, with a Delta new-phase panel. The Delta panel is the
               point: raw before/after surfaces make a small isochron change nearly impossible
               to see, while the difference localises it immediately (in input_screen this is
               what showed a stiff direction moving the singularity as a red/blue dipole while
               a sloppy one was provably flat).
  directions   the COMBINATORIAL analysis: per-direction LC vs PTC response from the
               generalized eigenproblem, next to the per-parameter scatter, plus the
               decoupling spectrum and the loadings of the top directions.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analysis  # noqa: F401
import paths
from plotting import phase_map, phase_cmap, scrit_overview

#: figures live in their run directory, not one flat folder -- see paths.save_figure.
_CTX = dict(model=None, analysis='figures', tag=None, publish=False)


def _save(fig, name, **cfg):
    # the published copy carries the model prefix: docs/figures/ is shared across models, so
    # `surfaces_pulse.png` would collide the moment Korencic is run.
    pub = f"{_CTX['model']}_{name}" if _CTX['publish'] else None
    p = paths.save_figure(fig, _CTX['model'], _CTX['analysis'], name, tag=_CTX['tag'],
                          publish=pub, **cfg)
    plt.close(fig)
    return p


def _pick_examples(names, values, n=3, valid=None):
    """Strongest, median and weakest by `values` -- so a figure shows the whole range rather
    than only the headline case.

    `valid` restricts the choice to parameters that actually HAVE data at the factor being
    displayed. Without it the picker happily chooses a setting whose orbit was rejected and
    the panel comes out blank, which reads as "this parameter does nothing" when it means
    "this parameter was not measured here"."""
    v = np.nan_to_num(np.asarray(values, float))
    idx = np.arange(len(v)) if valid is None else np.where(np.asarray(valid))[0]
    if len(idx) == 0:
        idx = np.arange(len(v))
    order = idx[np.argsort(-v[idx])]
    return [int(order[0]), int(order[len(order) // 2]), int(order[-1])][:n]


# --------------------------------------------------------------------------- #
def fig_tornado(lc, pt, model, target, mode):
    """LC and PTC sensitivity per parameter, on ONE shared ordering so the two can be read
    against each other. Sorting each panel independently would hide exactly the thing of
    interest: a parameter high in one and low in the other."""
    lp = [str(x) for x in lc['params']]
    pp = [str(x) for x in pt['params']]
    common = [p for p in lp if p in set(pp)]
    x = np.array([lc['lc_sens'][lp.index(p)] for p in common])
    y = np.array([pt['twist_span'][pp.index(p)] for p in common])
    order = np.argsort(x)
    names = [common[i] for i in order]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, max(4, 0.32 * len(names))),
                             gridspec_kw={'width_ratios': [1, 1, 0.9]}, sharey=True)
    ypos = np.arange(len(names))
    axes[0].barh(ypos, x[order], color='steelblue')
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(names, fontsize=7)
    axes[0].set_xlabel('LC sensitivity'); axes[0].set_title('limit cycle', fontsize=10)
    axes[1].barh(ypos, y[order], color='indianred')
    axes[1].set_xlabel('PTC sensitivity (twist)'); axes[1].set_title('PTC', fontsize=10)
    ratio = y[order] / np.maximum(x[order], 1e-12)
    axes[2].barh(ypos, ratio, color='0.45')
    axes[2].axvline(np.median(ratio), color='k', ls=':', lw=0.9)
    axes[2].set_xlabel('PTC / LC'); axes[2].set_title('ratio (per parameter)', fontsize=10)
    for a in axes:
        a.tick_params(labelsize=7)
    fig.suptitle(f"{model} / {target} ({mode}): parameter sensitivity, sorted by LC",
                 fontsize=12)
    fig.tight_layout()
    return _save(fig, f"tornado_{target}_{mode}.png", target=target, mode=mode,
                 n_params=len(names))


def fig_lc_examples(lc, model, n_show=4):
    """Base vs perturbed cycle profiles -- what an LC sensitivity number means concretely."""
    prof = np.asarray(lc['profiles'])            # (n_param, n_fac, n_states, m)
    base = np.asarray(lc['base_profiles'])       # (n_states, m)
    fac = np.asarray(lc['factors'], float)
    names = [str(p) for p in lc['params']]
    obs = [str(s) for s in lc['observables']][:n_show]
    obs_idx = np.asarray(lc['obs_idx'])[:n_show]
    lo, hi = 3, 5                                # x0.79 and x1.26, symmetric in log
    have = [np.isfinite(prof[i, lo]).all() and np.isfinite(prof[i, hi]).all()
            for i in range(prof.shape[0])]
    ex = _pick_examples(names, lc['lc_sens'], valid=have)
    ph = np.arange(base.shape[1]) / base.shape[1]

    fig, axes = plt.subplots(len(ex), len(obs_idx),
                             figsize=(2.9 * len(obs_idx), 2.4 * len(ex)), squeeze=False)
    for r, i in enumerate(ex):
        for c, (si, sn) in enumerate(zip(obs_idx, obs)):
            ax = axes[r][c]
            ax.plot(ph, base[si], color='k', lw=2.0, label='base')
            for k, col, ls in ((lo, 'tab:blue', '--'), (hi, 'tab:red', '--')):
                if np.isfinite(prof[i, k, si]).all():
                    ax.plot(ph, prof[i, k, si], color=col, lw=1.3, ls=ls,
                            label=f'x{fac[k]:g}')
            if c == 0:
                ax.set_ylabel(f"{names[i]}\nlc_sens={lc['lc_sens'][i]:.3f}", fontsize=8)
            if r == 0:
                ax.set_title(sn, fontsize=9)
            if r == len(ex) - 1:
                ax.set_xlabel('phase')
            ax.tick_params(labelsize=7)
    axes[0][-1].legend(fontsize=7, loc='best')
    fig.suptitle(f"{model}: limit cycle under a +/-26% parameter change "
                 f"(strongest / median / weakest)", fontsize=11)
    fig.tight_layout()
    return _save(fig, "lc_examples.png", factors=[float(fac[lo]), float(fac[hi])],
                 states=obs)


def fig_ptc_examples(pt, model, target, mode):
    """Base, perturbed, and the DELTA panel -- the last one is why this figure exists."""
    grids = np.asarray(pt['ptc_grids'])          # (n_param, n_fac, n_phase, n_dose)
    base = np.asarray(pt['base_ptc'])
    doses = np.asarray(pt['doses'])
    old = np.asarray(pt['old'])
    fac = np.asarray(pt['factors'], float)
    names = [str(p) for p in pt['params']]
    hi = 5                                       # x1.26
    have = [np.isfinite(grids[i, hi]).any() for i in range(grids.shape[0])]
    ex = _pick_examples(names, pt['twist_span'], valid=have)

    fig, axes = plt.subplots(len(ex), 3, figsize=(11.0, 3.0 * len(ex)), squeeze=False)
    # a SHARED symmetric scale across every Delta panel, so panels are comparable to each
    # other rather than each being autoscaled to its own noise
    dmax = 0.0
    for i in ex:
        d = ((grids[i, hi] - base + 0.5) % 1.0) - 0.5
        if np.isfinite(d).any():
            dmax = max(dmax, float(np.nanmax(np.abs(d))))
    dmax = max(dmax, 1e-3)

    for r, i in enumerate(ex):
        g = grids[i, hi]
        phase_map(axes[r][0], old, doses, base, title='base' if r == 0 else None)
        phase_map(axes[r][1], old, doses, g,
                  title=f'x{fac[hi]:g}' if r == 0 else None)
        d = ((g - base + 0.5) % 1.0) - 0.5
        im = axes[r][2].pcolormesh(old, doses, np.ma.masked_invalid(d.T), cmap='RdBu_r',
                                   vmin=-dmax, vmax=dmax, shading='nearest')
        axes[r][2].set_yscale('log'); axes[r][2].set_xlabel('old phase')
        if r == 0:
            axes[r][2].set_title('$\\Delta$ new phase (shared scale)', fontsize=9)
        plt.colorbar(im, ax=axes[r][2], fraction=0.046)
        axes[r][0].set_ylabel(f"{names[i]}\ntwist resp={pt['twist_span'][i]:.3f}", fontsize=8)
    fig.suptitle(f"{model} / {target} ({mode}): PTC under a +26% parameter change "
                 f"(strongest / median / weakest)", fontsize=11)
    fig.tight_layout()
    return _save(fig, f"ptc_examples_{target}_{mode}.png", target=target, mode=mode,
                 factor=float(fac[hi]), n_dose=len(doses))


def fig_directions(cp, model, target, mode):
    """The COMBINATORIAL picture, which the per-parameter scatter cannot show."""
    rho = np.asarray(cp['rho'])
    V = np.asarray(cp['V'])
    axis_rho = np.asarray(cp['axis_rho'])
    names = [str(p) for p in cp['params']]
    lcs, pts = np.asarray(cp['lc_sens']), np.asarray(cp['ptc_sens'])

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.34, wspace=0.28)

    # (a) the decoupling spectrum
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(np.arange(len(rho)), np.maximum(rho, 1e-6), 'o-', color='tab:purple')
    ax.axhline(np.nanmax(axis_rho), color='tab:orange', ls='--', lw=1.4,
               label=f'best single parameter ({np.nanmax(axis_rho):.2f})')
    ax.axhline(1.0, color='0.6', ls=':', lw=1.0)
    ax.set_xlabel('direction (sorted)'); ax.set_ylabel(r'$\rho=\|J_{PTC}v\|^2/\|J_{LC}v\|^2$')
    ax.set_title('decoupling spectrum', fontsize=10)
    ax.legend(fontsize=7)

    # (b) per-parameter vs per-direction, on the same axes
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(lcs / np.nanmax(lcs), pts / np.nanmax(pts), s=34, c='tab:blue',
               edgecolors='k', linewidths=0.4, label='parameters (axis-aligned)', zorder=3)
    lr, pr = np.asarray(cp['dir_lc_resp']), np.asarray(cp['dir_ptc_resp'])
    ax.scatter(lr / np.nanmax(lr), pr / np.nanmax(pr), s=44, marker='D', c='tab:purple',
               edgecolors='k', linewidths=0.4, label='directions (combinations)', zorder=4)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('LC response (normalised)'); ax.set_ylabel('PTC response (normalised)')
    ax.set_title('why combinations matter', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')

    # (c) loadings of the top decoupled directions
    ax = fig.add_subplot(gs[0, 2])
    k = min(3, V.shape[1])
    w = 0.8 / k
    ypos = np.arange(len(names))
    for i in range(k):
        ax.barh(ypos + (i - (k - 1) / 2) * w, V[:, i], height=w,
                label=f'dir {i} ($\\rho$={rho[i]:.1f})')
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=6.5)
    ax.axvline(0, color='k', lw=0.7)
    ax.set_xlabel('loading'); ax.set_title('composition of the top directions', fontsize=10)
    ax.legend(fontsize=7)

    # (d) sigma_LC vs PTC response, the input_screen-style sloppy-tail view
    ax = fig.add_subplot(gs[1, 0])
    s_lc = np.asarray(cp['sigma_lc']); resp = np.asarray(cp['ptc_response'])
    m = s_lc / s_lc.max() > 1e-12
    ax.loglog(s_lc[m] / s_lc.max(), resp[m] / resp.max(), 'o', color='tab:green')
    ax.set_xlabel(r'$\sigma_{LC}/\sigma_{max}$ (LC singular direction)')
    ax.set_ylabel(r'$\|J_{PTC}v\|$ (normalised)')
    ax.set_title('sloppy-tail view', fontsize=10)

    # (e) top-k subspace angles
    ax = fig.add_subplot(gs[1, 1])
    ks, angs = np.asarray(cp['top_angle_ks']), np.asarray(cp['top_angles'])
    ax.bar(ks.astype(str), angs, color='tab:red')
    ax.axhline(90, color='0.6', ls=':')
    ax.set_ylim(0, 95)
    ax.set_xlabel('k (leading subspace size)'); ax.set_ylabel('mean principal angle (deg)')
    ax.set_title('do the two pin the same combinations?', fontsize=10)

    # (f) the headline numbers, as text
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    top = np.argsort(-np.abs(V[:, 0]))[:6]
    lines = [
        f"model      {model} / {target} ({mode})",
        f"parameters {len(names)} (gauge quotiented)",
        "",
        f"best single parameter   rho = {np.nanmax(axis_rho):.2f}",
        f"best combination        rho = {rho[0]:.1f}",
        f"                        = {rho[0] / max(np.nanmax(axis_rho), 1e-9):.0f}x better",
        "",
        f"top direction moves the LC by {float(cp['dir_lc_resp'][0]):.1e}",
        f"and the PTC by                {float(cp['dir_ptc_resp'][0]):.1e}",
        "(both relative to that jacobian's norm)",
        "",
        "composition:",
    ] + [f"   {names[j]:>12s}  {V[j, 0]:+.3f}" for j in top]
    ax.text(0.0, 1.0, "\n".join(lines), va='top', ha='left', family='monospace', fontsize=8.5)

    fig.suptitle(f"{model} / {target} ({mode}): combinatorial LC-vs-PTC decoupling",
                 fontsize=12)
    return _save(fig, f"directions_{target}_{mode}.png", target=target, mode=mode,
                 n_params=len(names))


def fig_direction_examples(cf, model, target, mode, n_states=3):
    """THE side-by-side: for the SAME nudge along a direction, what moved in the limit cycle
    and what moved in the PTC.

    Pure read of `analysis/confirm.py`'s raw arrays -- `lc_profiles` and `ptc_grids` are the
    actual integrated output, so this figure never re-integrates anything.

    Each row is one direction. Left: the observable cycle profiles, base in black and the
    +/-eps displacements dashed. Right: the base PTC, the displaced PTC, and their difference.
    Every Delta panel shares one symmetric colour scale so the rows are comparable to each
    other rather than each being autoscaled to its own noise -- which is the whole point, since
    the claim is about the RATIO of movements between rows.
    """
    labels = [str(x) for x in cf['labels']]
    sign = np.asarray(cf['sign'])
    LC = np.asarray(cf['lc_profiles'])          # (n_run, n_states, m)
    baseC = np.asarray(cf['base_profiles'])
    # Prefer the DENSE JAX render for the surfaces: the adaptive verification grid is
    # necessarily coarse (it is slow), and a 4x16 surface cannot be read. The two engines
    # agree to 1.6e-05 on this model, so the picture and the verification are the same claim.
    if 'dense_ptc_grids' in cf:
        PT = np.asarray(cf['dense_ptc_grids'])          # (n_run, n_phase, n_dose)
        baseP = np.asarray(cf['dense_base'])            # (n_phase, n_dose)
        doses = np.asarray(cf['dense_doses'])
        old = np.asarray(cf['dense_old'])
        engine_note = f"surfaces: JAX, {len(old)}x{len(doses)}"
    else:
        PT = np.asarray(cf['ptc_grids']).transpose(0, 2, 1)
        baseP = np.asarray(cf['base_new']).T
        doses = np.asarray(cf['doses'])
        old = np.asarray(cf['old'])
        engine_note = f"surfaces: adaptive, {len(old)}x{len(doses)}"
    states = [str(s) for s in cf['state_names']]
    obs = [str(s) for s in cf['observables']][:n_states]
    oidx = [states.index(s) for s in obs]
    dLC, dP = np.asarray(cf['dLC']), np.asarray(cf['dPTC_rms'])
    eps = float(cf['eps'])

    uniq = []
    for l in labels:
        if l not in uniq:
            uniq.append(l)
    ph = np.arange(baseC.shape[1]) / baseC.shape[1]

    dmax = 0.0
    for r in range(PT.shape[0]):
        d = ((PT[r] - baseP + 0.5) % 1.0) - 0.5
        if np.isfinite(d).any():
            dmax = max(dmax, float(np.nanmax(np.abs(d))))
    dmax = max(dmax, 1e-3)

    ncol = n_states + 3
    fig, axes = plt.subplots(len(uniq), ncol, figsize=(2.75 * ncol, 2.65 * len(uniq)),
                             squeeze=False)
    for r, lab in enumerate(uniq):
        runs = [i for i, l in enumerate(labels) if l == lab]
        rp = runs[0]                                        # the +eps run, for the PTC panels
        for c, (si, sn) in enumerate(zip(oidx, obs)):
            ax = axes[r][c]
            ax.plot(ph, baseC[si], color='k', lw=2.2, label='base', zorder=3)
            for i in runs:
                ax.plot(ph, LC[i][si], lw=1.3, ls='--',
                        color=('tab:red' if sign[i] > 0 else 'tab:blue'),
                        label=f"{'+' if sign[i] > 0 else '-'}{eps:g}")
            if r == 0:
                ax.set_title(f"LC: {sn}", fontsize=9)
            if r == len(uniq) - 1:
                ax.set_xlabel('phase')
            if c == 0:
                ax.set_ylabel(f"{lab}\ndLC={dLC[rp]:.4f}  dPTC={dP[rp]:.4f}", fontsize=8)
            ax.tick_params(labelsize=7)
        phase_map(axes[r][n_states], old, doses, baseP,
                  title='PTC base' if r == 0 else None)
        phase_map(axes[r][n_states + 1], old, doses, PT[rp],
                  title=f'PTC displaced (+{eps:g})' if r == 0 else None)
        d = ((PT[rp] - baseP + 0.5) % 1.0) - 0.5
        im = axes[r][n_states + 2].pcolormesh(old, doses, np.ma.masked_invalid(d.T),
                                              cmap='RdBu_r', vmin=-dmax, vmax=dmax,
                                              shading='nearest')
        axes[r][n_states + 2].set_yscale('log')
        axes[r][n_states + 2].set_xlabel('old phase')
        if r == 0:
            axes[r][n_states + 2].set_title(r'$\Delta$ PTC (shared scale)', fontsize=9)
        plt.colorbar(im, ax=axes[r][n_states + 2], fraction=0.046)
    axes[0][0].legend(fontsize=7, loc='best')
    fig.suptitle(f"{model} / {target} ({mode}): the SAME nudge (eps={eps:g}) seen in the "
                 f"limit cycle and in the PTC\n"
                 f"dLC / dPTC in the row labels are measured on the ADAPTIVE engine; "
                 f"{engine_note}", fontsize=11)
    fig.tight_layout()
    return _save(fig, f"direction_examples_{target}_{mode}.png", target=target,
                 mode=mode, eps=eps, engine=engine_note)


def fig_surfaces(ch, model, mode):
    """PTC surfaces at base, each with its twist curve in a DEDICATED panel beneath it.

    The twist used to be drawn as a black line over the surface; it is its own quantity and
    gets its own axes, on a shared dose axis so a reader can carry a dose between them."""
    from plotting import twist_panel
    ts = [str(t) for t in ch['targets']]
    ncol = len(ts)
    fig, axes = plt.subplots(2, ncol, figsize=(3.5 * ncol, 6.2), squeeze=False,
                             gridspec_kw={'height_ratios': [1.35, 1]})
    for k, t in enumerate(ts):
        sings = [{'phi': p, 'dose': d, 'sign': sg} for p, d, sg in
                 zip(ch[f'sing_phi__{t}'], ch[f'sing_dose__{t}'], ch[f'sing_sign__{t}'])]
        phase_map(axes[0][k], ch[f'old__{t}'], ch[f'doses__{t}'], ch[f'ptc__{t}'],
                  title=f"{t}   S*={ch['S'][k]:.3g}", sings=sings, scrit=ch['S'][k])
        twist_panel(axes[1][k], ch[f'doses__{t}'], ch[f'twist__{t}'], scrit=ch['S'][k],
                    title=f"twist = {ch['total_twist'][k]:.3f} cyc")
    fig.suptitle(f"{model} ({mode}): PTC surfaces at base, with the fixed-point (twist) "
                 f"curve below each", fontsize=12)
    fig.tight_layout()
    return _save(fig, f"surfaces_{mode}.png", mode=mode, targets=ts)


# --------------------------------------------------------------------------- #
def fig_sweep(sw, model):
    """A RANGE of displacements along one direction: what moves, and how much.

    Four things on one page, all against the same eps axis:
      * the limit cycle at every eps, coloured by eps -- how much the cycle actually deforms;
      * the PTC surface at the extremes and at base;
      * the twist curve at every eps, in its own panel;
      * the PTC FEATURES (total twist, S_crit, phi*) as functions of eps, which is what makes
        "it barely moves" or "it moves a lot" quantitative rather than an impression.
    """
    from plotting import twist_panel
    eps = np.asarray(sw['eps'])
    LC = np.asarray(sw['lc_profiles'])
    PT = np.asarray(sw['ptc_grids'])
    TW = np.asarray(sw['twist_curves'])
    doses, old = np.asarray(sw['doses']), np.asarray(sw['old'])
    states = [str(x) for x in sw['state_names']]
    obs = [str(x) for x in sw['observables']][:3]
    oidx = [states.index(x) for x in obs]
    baseC, baseP = np.asarray(sw['base_profiles']), np.asarray(sw['base_ptc'])
    label, rho = str(sw['label']), float(sw['rho'])
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(eps.min(), eps.max())
    ph = np.arange(baseC.shape[1]) / baseC.shape[1]
    i0 = int(np.argmin(np.abs(eps)))
    ends = [0, i0, len(eps) - 1]

    fig = plt.figure(figsize=(16.5, 9.2))
    gs = fig.add_gridspec(3, 6, hspace=0.42, wspace=0.38)

    for c, (si, sn) in enumerate(zip(oidx, obs)):          # row 0: the limit cycle
        ax = fig.add_subplot(gs[0, c])
        for i in range(len(eps)):
            if np.isfinite(LC[i]).all():
                ax.plot(ph, LC[i][si], color=cmap(norm(eps[i])), lw=1.1)
        ax.plot(ph, baseC[si], color='k', lw=2.0, zorder=5)
        ax.set_title(f"LC: {sn}", fontsize=9); ax.set_xlabel('phase')
        ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[0, 3])                          # the twist curve at every eps
    for i in range(len(eps)):
        if np.isfinite(TW[i]).any():
            twist_panel(ax, doses, TW[i], color=cmap(norm(eps[i])))
    twist_panel(ax, doses, np.asarray(sw['base_twist']), color='k')
    ax.set_title('twist curve vs dose', fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.05, label='eps')

    for j, i in enumerate(ends):                            # row 1: surfaces at the extremes
        ax = fig.add_subplot(gs[1, j * 2:j * 2 + 2])
        phase_map(ax, old, doses, PT[i], title=f"PTC at eps = {eps[i]:+.2f}")

    panels = [('total twist (cyc)', sw['total_twist'], 'tab:purple'),
              ('S_crit', sw['S_crit'], 'tab:orange'),
              ('phi* (singularity phase)', sw['phi_sing'], 'tab:green'),
              ('period T (h)', sw['period'], 'tab:brown'),
              ('dLC vs base', sw['dLC'], 'tab:blue'),
              ('dPTC vs base', sw['dPTC'], 'tab:red')]
    for c, (ttl, y, col) in enumerate(panels):              # row 2: features vs eps
        ax = fig.add_subplot(gs[2, c])
        y = np.asarray(y, float)
        ax.plot(eps, y, 'o-', color=col, ms=4)
        ax.axvline(0, color='0.7', lw=0.8, ls=':')
        if ttl.startswith('S_crit'):
            ax.set_yscale('log')
        ax.set_xlabel('eps'); ax.set_title(ttl, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{model} / {sw['target']} ({sw['mode']}): sweep along the {label} direction "
                 f"(rho = {rho:.3g})", fontsize=13)
    return _save(fig, f"sweep_{sw['target']}_{sw['mode']}_{sw['direction']}.png",
                 direction=str(sw['direction']), rho=rho,
                 eps_range=[float(eps.min()), float(eps.max())], n_eps=len(eps))


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description='regenerate every figure from saved npz')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--target', default=os.environ.get('TARGET', 'BMAL1'))
    ap.add_argument('--mode', default='pulse')
    ap.add_argument('--which', default='all')
    ap.add_argument('--tag', default=None,
                    help='run tag the figures are written under (default: a timestamp)')
    ap.add_argument('--publish', action='store_true',
                    help='also copy each figure into docs/figures/ for PROJECT_SUMMARY')
    ap.add_argument('--direction', default='decoupled')
    a = ap.parse_args(argv)
    _CTX.update(model=a.model, tag=paths.run_tag(a.tag), publish=a.publish)
    from analysis.coupling import _load
    want = a.which.split(',') if a.which != 'all' else \
        ['scrit', 'surfaces', 'tornado', 'lc_examples', 'ptc_examples', 'directions',
         'direction_examples', 'sweep']
    made = []

    if 'scrit' in want:
        sc, _t = _load(a.model, 'scrit', f'scrit_{a.mode}*.npz')
        sc['model'] = str(sc['model']); sc['mode'] = str(sc['mode'])
        made.append(_save(scrit_overview(sc), f"scrit_{a.mode}.png", mode=a.mode,
                          n_targets=len(sc["targets"])))
    if 'surfaces' in want:
        ch, _t = _load(a.model, 'characterize', f'char_{a.mode}*.npz')
        ch['model'] = str(ch['model']); ch['mode'] = str(ch['mode'])
        made.append(fig_surfaces(ch, a.model, a.mode))
    lc = pt = None
    if {'tornado', 'lc_examples', 'ptc_examples', 'directions'} & set(want):
        lc, _t = _load(a.model, 'lc_sens', 'lc_sens*.npz')
        pt, _t = _load(a.model, 'ptc_sens', f'ptc_sens_{a.target}_{a.mode}*.npz')
    if 'tornado' in want:
        made.append(fig_tornado(lc, pt, a.model, a.target, a.mode))
    if 'lc_examples' in want:
        made.append(fig_lc_examples(lc, a.model))
    if 'ptc_examples' in want:
        made.append(fig_ptc_examples(pt, a.model, a.target, a.mode))
    if 'directions' in want:
        fp = paths.out_path(a.model, 'coupling',
                            f'coupling_{a.target}_{a.mode}_twist.npz')
        if not os.path.exists(fp):
            print(f"[figures] run `python -m analysis.coupling --model {a.model} "
                  f"--target {a.target}` first", file=sys.stderr)
        else:
            cp = dict(np.load(fp, allow_pickle=True))
            made.append(fig_directions(cp, a.model, a.target, a.mode))
    if 'direction_examples' in want:
        fp = paths.out_path(a.model, 'coupling', f'confirm_{a.target}_{a.mode}.npz')
        if not os.path.exists(fp):
            print(f"[figures] run `python -m analysis.confirm --model {a.model} "
                  f"--target {a.target}` first", file=sys.stderr)
        else:
            cf = dict(np.load(fp, allow_pickle=True))
            if 'lc_profiles' not in cf:
                print("[figures] the confirm npz predates raw-array saving; re-run "
                      "analysis.confirm", file=sys.stderr)
            else:
                made.append(fig_direction_examples(cf, a.model, a.target, a.mode))
    if 'sweep' in want:
        import glob as _g
        st = paths.latest_run(a.model, 'sweep')
        hits = (_g.glob(os.path.join(paths.out_dir(a.model, 'sweep', st, create=False),
                                     f'sweep_{a.target}_{a.mode}_{a.direction}.npz'))
                if st else [])
        if not hits:
            print(f"[figures] run `python -m analysis.sweep --model {a.model} "
                  f"--target {a.target} --direction {a.direction}` first", file=sys.stderr)
        else:
            sw = dict(np.load(hits[0], allow_pickle=True))
            for k in ('model', 'target', 'mode', 'direction', 'label'):
                sw[k] = str(sw[k])
            made.append(fig_sweep(sw, a.model))
    print(f"[figures] {len(made)} figure(s) -> "
          f"{os.path.relpath(paths.out_dir(a.model, 'figures', _CTX['tag']), paths.HERE)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
