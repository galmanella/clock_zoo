"""
fit/figures_arms.py
===================
Figures for an ARM-COMPARISON campaign -- several tags meant to differ in one cost setting each.

    $PY -m fit.arms_rescore --model almeida --arms arm0_ctl arm1_ramp arm2_brack \
        arm3_inform arm4_floor --control arm0_ctl --tag arms_batch1 --check
    $PY -m fit.figures_arms --model almeida --arms arm0_ctl arm1_ramp arm2_brack \
        arm3_inform arm4_floor --tag arms_batch1

PURE READ. Every panel comes from a saved npz plus the one JSON `fit.arms_rescore` writes, so
replotting never re-runs a fit and never re-scores an endpoint.

WHY THE FIRST FIGURE IS AN AUDIT AND NOT A RESULT
    An arm campaign's premise is that the arms differ. That is a CLAIM ABOUT THE PLUMBING, not
    about the model, and it is cheap to check: two runs that minimised the same function follow
    the same path, so their evaluation traces are equal element by element. Batch 1 failed that
    check -- four of five arms have traces that agree on all 8000 population evaluations and
    differ only at index 0, the single evaluation the parent makes -- and every downstream
    figure would otherwise have read as "the arms made no difference", which is a statement
    about the objective rather than about the wiring.

    So `fig_collapse` runs first and groups the runs by what they ACTUALLY optimised. The rest
    of the figures then plot those groups, not the declared tags.

    The grouping is DERIVED, never assumed: runs are keyed by a hash of `trace_f[1:]`, which is
    the pool's output alone. Nothing here hard-codes which arms collapsed.
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import paths
from plotting import broken, phase_cmap, twist_panel

_CTX = {}

#: one colour per arm, stable across every figure in the set
ARM_COLORS = ('#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#17becf')
#: `_harden`'s failure sentinel -- plotted as a capped marker rather than a spike to 1e6
SENTINEL = 1e6


def _save(fig, name, **cfg):
    pub = f"{_CTX['model']}_{name}.png" if _CTX.get('publish') else None
    return paths.save_figure(fig, _CTX['model'], _CTX['analysis'], name, _CTX['tag'],
                             publish=pub, **cfg)


def _start_of(dirname):
    for part in dirname.split('__')[1:]:
        for token in part.split('_'):
            if token.startswith('start-'):
                return token[len('start-'):]
    return '?'


def load_runs(model, arms, analysis='fit_radial'):
    """Every run directory of every named arm, with its npz and its search signature.

    `sig` HASHES `trace_f[1:]`, NOT the whole trace. Index 0 is the parent's own evaluation of
    the start point under the arm's declared cost; entries 1.. are the worker pool's. Hashing
    only the pool's output answers the question that matters -- did these two runs minimise the
    same function -- and does not let a declared-but-unused option split a group.
    """
    root = os.path.join(paths.OUT, str(model), analysis)
    runs = []
    for ai, arm in enumerate(arms):
        for d in sorted(glob.glob(os.path.join(root, arm + '__*'))):
            fs = sorted(glob.glob(os.path.join(d, 'radial_*.npz')))
            if not fs:
                continue
            z = dict(np.load(fs[-1], allow_pickle=True))
            tf = np.asarray(z['trace_f'], float)
            runs.append(dict(
                arm=arm, color=ARM_COLORS[ai % len(ARM_COLORS)], dirname=os.path.basename(d),
                start=_start_of(os.path.basename(d)), seed=int(z['seed_used']), z=z,
                tf=tf, best=np.minimum.accumulate(tf),
                sig=hashlib.sha1(np.ascontiguousarray(tf[1:]).tobytes()).hexdigest()[:10],
                doses=np.asarray(z['doses'], float)))
    if not runs:
        raise SystemExit(f"no runs under {root} for arms {arms}")
    return runs


def objective_groups(runs):
    """Group runs by the function they actually minimised.

    Two runs from DIFFERENT (start, seed) cells never share a trace, so the signature is
    compared only within a cell; a group is then the set of arms that agree in EVERY cell they
    share. Anything less than every cell is reported rather than merged -- a partial agreement
    is a finding, not a grouping.
    """
    cells = sorted({(r['start'], r['seed']) for r in runs})
    arms = sorted({r['arm'] for r in runs})
    sig = {(r['arm'], r['start'], r['seed']): r['sig'] for r in runs}
    groups, seen = [], set()
    for arm in arms:
        if arm in seen:
            continue
        same = [a for a in arms
                if all(sig.get((a, s, d)) == sig.get((arm, s, d)) for s, d in cells)]
        groups.append(same)
        seen.update(same)
    return groups, cells


def _group_label(group, runs):
    """Name a group by the ONE setting its members' dose grids agree on, since that is the only
    arm option that reached the pool. Falls back to listing the arms."""
    lo = {round(float(r['doses'][0]), 6) for r in runs if r['arm'] in group}
    return (f"lo dose {sorted(lo)[0]:g}" if len(lo) == 1 else '+'.join(group))


# --------------------------------------------------------------------------- #
#  1. the audit
# --------------------------------------------------------------------------- #
def fig_collapse(runs, yard=None):
    """DID THE ARMS DIFFER? Convergence per cell, and the trace difference against the control.

    Read the BOTTOM row first. It plots |f_arm(i) - f_control(i)| against evaluation index on a
    log axis, so a run that minimised a different function shows a cloud of nonzero differences
    and a run that did not shows nothing at all except, possibly, a single point at i = 0.

    That single point is the tell, and it is why index 0 is drawn as a ringed marker rather than
    left in the line: it is the parent's own evaluation of the start under the arm's DECLARED
    cost, the one place an unused option is still visible. An arm whose only nonzero difference
    sits there declared a cost it never searched.
    """
    groups, cells = objective_groups(runs)
    arms = sorted({r['arm'] for r in runs})
    ctl = arms[0]
    col = {r['arm']: r['color'] for r in runs}
    R = {(r['arm'], r['start'], r['seed']): r for r in runs}

    fig = plt.figure(figsize=(4.1 * len(cells), 9.4))
    gs = fig.add_gridspec(3, len(cells), height_ratios=[1.0, 1.0, 0.62], hspace=0.42,
                          wspace=0.30)

    for j, (start, seed) in enumerate(cells):
        # --- best-so-far ------------------------------------------------------------- #
        ax = fig.add_subplot(gs[0, j])
        for k, arm in enumerate(arms):
            r = R.get((arm, start, seed))
            if r is None:
                continue
            # WIDTH DESCENDING WITH ARM ORDER, so a curve hidden underneath another is still
            # visible as a halo. Identical curves are the finding here; a legend cannot show it
            # and equal linewidths would render four runs as one and look like a plotting bug.
            ax.plot(np.arange(len(r['best'])), np.clip(r['best'], None, SENTINEL),
                    color=col[arm], lw=4.4 - 0.75 * k, alpha=0.95, label=arm, zorder=3 + k)
        ax.set_yscale('log'); ax.set_xlabel('evaluation'); ax.set_title(
            f"start {start}   seed {seed}", fontsize=9.5)
        if j == 0:
            ax.set_ylabel('best objective so far\n(each arm in ITS OWN units)')
            ax.legend(fontsize=7, frameon=False, loc='upper right')

        # --- the trace itself, difference against the control ------------------------ #
        ax = fig.add_subplot(gs[1, j])
        c = R.get((ctl, start, seed))
        for k, arm in enumerate(arms):
            if arm == ctl or c is None:
                continue
            r = R.get((arm, start, seed))
            if r is None:
                continue
            n = min(len(r['tf']), len(c['tf']))
            d = np.abs(r['tf'][:n] - c['tf'][:n])
            nz = np.nonzero(d)[0]
            ax.plot(nz[nz > 0], d[nz[nz > 0]], ls='none', marker='.', ms=2.0,
                    color=col[arm], alpha=0.55, label=f"{arm}  ({len(nz)} differ)")
            if len(nz) and nz[0] == 0:
                # white halo under the ring: index 0 is the whole point of the panel and it
                # must stay legible on top of a dense cloud from a genuinely different arm
                ax.plot([0], [d[0]], marker='o', ms=12, mfc='none', mec='w', mew=3.6,
                        zorder=6, clip_on=False)
                ax.plot([0], [d[0]], marker='o', ms=12, mfc='none', mec=col[arm], mew=2.0,
                        zorder=7, clip_on=False)
        ax.set_yscale('log'); ax.set_xlabel('evaluation index')
        # The band at 1e6 is `_harden`'s failure sentinel, not a large disagreement: an arm
        # whose window reaches lower doses has evaluations that fail where the control's do
        # not, and |sentinel - ordinary| is ~1e6 by construction. Named here so it is not
        # read as a cost difference.
        ax.set_title('|f - f(control)| per evaluation\n(band at 1e6 = failure sentinel on one '
                     'side only)', fontsize=8.2)
        ax.legend(fontsize=6.5, frameon=False, loc='upper right')
        if j == 0:
            ax.set_ylabel(f'difference from {ctl}')

        # --- what each arm said about its own start point ---------------------------- #
        ax = fig.add_subplot(gs[2, j])
        xs, dec, sea = [], [], []
        for arm in arms:
            r = R.get((arm, start, seed))
            if r is None:
                continue
            xs.append(arm)
            dec.append(float(r['tf'][0]))
            sea.append(np.nan if yard is None else yard.get(r['dirname'], {}).get('searched_f0',
                                                                                 np.nan))
        x = np.arange(len(xs))
        ax.bar(x - 0.2, dec, 0.4, color=[col[a] for a in xs], label='declared (arm own cost)')
        if yard is not None and np.isfinite(np.asarray(sea, float)).any():
            ax.bar(x + 0.2, sea, 0.4, color='0.35', hatch='///', edgecolor='w',
                   label='SEARCHED (worker pool)')
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('_', '\n') for a in xs], fontsize=6.5)
        ax.set_title('objective at the start point', fontsize=9)
        hi = np.nanmax([v for v in list(dec) + list(sea) if np.isfinite(v)] or [1.0])
        ax.set_ylim(0, hi * 1.38)          # headroom, so the legend never sits on a bar
        if j == 0:
            ax.set_ylabel('f(v0)')
            ax.legend(fontsize=6.5, frameon=False, loc='upper left')

    txt = ("SEARCH GROUPS, derived from the pool's own output (a hash of trace_f[1:], the "
           "parent's start evaluation excluded):\n   "
           + "\n   ".join(f"{_group_label(g, runs):<18s}  {' = '.join(g)}" for g in groups))
    if len(groups) < len(arms):
        txt += ("\n\nARMS THAT SHARE A GROUP MINIMISED THE SAME FUNCTION. Their tags name "
                "settings the search never saw, and\ntheir endpoints are replicates, not "
                "arms. Only differences BETWEEN groups are attributable to a setting.")
    fig.text(0.005, -0.035, txt, fontsize=8.2, family='monospace', va='top')
    fig.suptitle("Arm audit: did the arms minimise different functions?", fontsize=12.5, y=0.985)
    return _save(fig, 'arms_collapse', arms=arms, cells=len(cells),
                 groups=[' '.join(g) for g in groups])


# --------------------------------------------------------------------------- #
#  2. the one contrast that survived
# --------------------------------------------------------------------------- #
def fig_groups(runs, yard):
    """The surviving between-group contrast: convergence, endpoint yardstick, and bracketing.

    Panel 3 is the one that answers P2. It puts every endpoint's measured S_crit on the dose
    axis against the window that scored it, so "did the transition stay inside the window"
    is read off the picture rather than off a boolean. A run with no detectable singularity has
    nothing to plot and is drawn at the axis floor as an open marker -- absent, not zero.
    """
    groups, cells = objective_groups(runs)
    starts = sorted({r['start'] for r in runs})
    lab = {tuple(g): _group_label(g, runs) for g in groups}
    gcol = dict(zip((tuple(g) for g in groups), ('#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd')))
    rep = {}                      # one representative run per (group, start, seed)
    for r in runs:
        g = next(tuple(g) for g in groups if r['arm'] in g)
        rep.setdefault((g, r['start'], r['seed']), r)

    fig = plt.figure(figsize=(16.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.28)

    # --- convergence, own units --------------------------------------------------- #
    for i, start in enumerate(starts):
        ax = fig.add_subplot(gs[i, 0])
        for (g, st, sd), r in sorted(rep.items()):
            if st != start:
                continue
            ax.plot(np.arange(len(r['best'])), np.clip(r['best'], None, SENTINEL),
                    color=gcol[g], lw=1.5, alpha=0.85,
                    label=lab[g] if sd == 0 else None)
        ax.set_yscale('log'); ax.set_xlabel('evaluation')
        ax.set_ylabel('best so far')
        ax.set_title(f"convergence, start {start}\n(EACH GROUP IN ITS OWN UNITS -- not "
                     f"comparable)", fontsize=8.8)
        ax.legend(fontsize=7.5, frameon=False)

    # --- endpoint on the COMMON yardstick ----------------------------------------- #
    ax = fig.add_subplot(gs[:, 1])
    ticks, y = [], 0
    for start in starts:
        for g in groups:
            key = tuple(g)
            vals, seeds, fails = [], [], []
            for sd in sorted({r['seed'] for r in runs}):
                r = rep.get((key, start, sd))
                if r is None:
                    continue
                row = yard.get(r['dirname'], {})
                vals.append(row.get('yard', np.nan))
                seeds.append(sd)
                fails.append(row.get('n_fail'))
            if not vals:
                continue
            # EACH SEED ON ITS OWN SUB-ROW. Seeds within a cell routinely land within a few
            # thousandths of each other, and on one shared row both the markers and their
            # labels overprint -- which hides the very thing the panel is for, the SPREAD
            # inside an arm against the gap between arms.
            off = (np.arange(len(vals)) - (len(vals) - 1) / 2.0) * 0.17
            ax.plot(vals, y + off, 'o', color=gcol[key], ms=8, alpha=0.85)
            for v, dy, sd, nf in zip(vals, off, seeds, fails):
                ax.annotate(f"s{sd}" + (f" ({nf}✗)" if nf else ""), (v, y + dy),
                            textcoords='offset points', xytext=(8, -2), ha='left',
                            fontsize=6.5, va='center')
            ticks.append((y, f"{lab[key]}\nstart {start}"))
            y += 1
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([t[1] for t in ticks], fontsize=7.5)
    ax.set_ylim(-0.6, y - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel('c_ptc under the CONTROL objective, on the CONTROL grid')
    ax.set_title("endpoints on ONE yardstick\n(the only column comparable across groups;\n"
                 "'n✗' = contract checks failed)", fontsize=9)
    ax.grid(axis='x', alpha=0.25)

    # --- bracketing: where the transition ended up, against the window ------------- #
    ax = fig.add_subplot(gs[:, 2])
    y = 0
    for start in starts:
        for g in groups:
            key = tuple(g)
            for sd in sorted({r['seed'] for r in runs}):
                r = rep.get((key, start, sd))
                if r is None:
                    continue
                d = r['doses']
                ax.plot([d[0], d[-1]], [y, y], color=gcol[key], lw=5, alpha=0.28,
                        solid_capstyle='butt')
                sb = float(r['z']['base__S_crit'])
                sf = float(r['z']['fit__S_crit'])
                if np.isfinite(sb):
                    ax.plot([sb], [y], marker='|', ms=13, color='k', mew=1.6, zorder=5)
                if np.isfinite(sf):
                    ax.plot([sf], [y], marker='o', ms=7, color=gcol[key], mec='k', mew=0.8,
                            zorder=6)
                else:
                    ax.plot([d[0] * 0.55], [y], marker='o', ms=7, mfc='none', mec='r', mew=1.6,
                            zorder=6)
                ax.annotate(f"{lab[key]}  {start} s{sd}", (d[-1], y),
                            textcoords='offset points', xytext=(6, -3), fontsize=6.4)
                y += 1
    ax.set_xscale('log'); ax.set_xlabel('dose')
    ax.set_yticks([]); ax.invert_yaxis()
    ax.set_xlim(None, ax.get_xlim()[1] * 9)
    ax.set_title("C6, drawn: the scored window (bar), the base singularity (|),\n"
                 "the fitted one (●).  Open red ● at the left = NO singularity\n"
                 "found anywhere in the window", fontsize=9)

    fig.suptitle("What the batch can still answer: the between-group contrast", fontsize=12.5,
                 y=0.98)
    return _save(fig, 'arms_groups', groups=[' '.join(g) for g in groups])


# --------------------------------------------------------------------------- #
#  3. surfaces, twist, cycles -- per search group
# --------------------------------------------------------------------------- #
def fig_surfaces(runs, group, tag):
    """Target, base and every distinct endpoint surface of one search group.

    Each panel carries ITS OWN S_crit as a dotted white line, never one global value: the fit
    is supposed to move the singularity, and a line that cannot move cannot show that it did
    (the same mistake fixed twice before, in analysis/compare_points and in fit.figures).
    """
    sel = [r for r in runs if r['arm'] == group[0]]
    z0 = sel[0]['z']
    old, doses = np.asarray(z0['old']), np.asarray(z0['doses'])
    starts = sorted({r['start'] for r in sel})
    seeds = sorted({r['seed'] for r in sel})

    ncol = 1 + len(seeds)
    fig = plt.figure(figsize=(3.5 * ncol + 1.0, 3.5 * len(starts) + 0.9))
    gs = fig.add_gridspec(len(starts), ncol + 1, wspace=0.34, hspace=0.42,
                          width_ratios=[1] * ncol + [0.055])

    def _panel(ax, ptc, title, sc):
        m = ax.pcolormesh(old, doses, np.ma.masked_invalid(np.asarray(ptc)).T,
                          cmap=phase_cmap(), vmin=0, vmax=1, shading='nearest')
        ax.set_yscale('log'); ax.set_xlabel('old phase (cyc)')
        if sc is not None and np.isfinite(sc) and sc > 0:
            ax.axhline(sc, color='w', ls=':', lw=1.3, alpha=0.9)
            ax.text(0.02, sc, f"S={sc:.3g}", color='w', fontsize=6.5, va='bottom')
        ax.set_title(title, fontsize=8.8)
        return m

    for i, start in enumerate(starts):
        ref = [r for r in sel if r['start'] == start][0]
        left = (('radial target (PINNED, S=%.4g)' % (1.0 / float(ref['z']['k_used'])),
                 np.asarray(ref['z']['ptc_target_used']), 1.0 / float(ref['z']['k_used']))
                if i == 0 else
                ('base (nominal parameters)', np.asarray(ref['z']['ptc_base']),
                 float(ref['z']['base__S_crit'])))
        _panel(fig.add_subplot(gs[i, 0]), left[1], left[0], left[2])
        for j, sd in enumerate(seeds):
            r = [x for x in sel if x['start'] == start and x['seed'] == sd]
            ax = fig.add_subplot(gs[i, 1 + j])
            if not r:
                ax.axis('off'); continue
            r = r[0]
            sc = float(r['z']['fit__S_crit'])
            m = _panel(ax, np.asarray(r['z']['ptc_fit']),
                       f"fitted  start {start}  seed {sd}"
                       + ("" if np.isfinite(sc) else "\nNO SINGULARITY IN WINDOW"), sc)
        fig.colorbar(m, cax=fig.add_subplot(gs[i, -1]), label='new phase (cyc)')

    fig.suptitle(f"PTC surfaces -- search group '{tag}'  ({' = '.join(group)})", fontsize=12,
                 y=0.995)
    return _save(fig, f'arms_surfaces_{tag.replace(" ", "").replace(".", "p")}',
                 group=' '.join(group))


def fig_twist(runs):
    """Twist curves, one panel per (group, start), base in bold black under the endpoints.

    The twist curve is the stable fixed point against dose. Flattening it is what
    'radialization' means, so a fit that lowers `c_ptc` while leaving this curve alone has not
    done the thing the objective is a proxy for.
    """
    groups, _cells = objective_groups(runs)
    starts = sorted({r['start'] for r in runs})
    seeds = sorted({r['seed'] for r in runs})
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(seeds), 2)))

    fig, axes = plt.subplots(len(groups), len(starts),
                             figsize=(6.2 * len(starts), 4.0 * len(groups)), squeeze=False)
    for i, g in enumerate(groups):
        sel = [r for r in runs if r['arm'] == g[0]]
        for j, start in enumerate(starts):
            ax = axes[i][j]
            ss = [r for r in sel if r['start'] == start]
            if not ss:
                ax.axis('off'); continue
            z0 = ss[0]['z']
            twist_panel(ax, np.asarray(z0['doses']), np.asarray(z0['twist_base']),
                        scrit=float(z0['base__S_crit']), is_base=True, label='base')
            for r in sorted(ss, key=lambda x: x['seed']):
                ax.plot(*broken(np.asarray(r['z']['doses']),
                                np.asarray(r['z']['twist_fit'])),
                        color=cols[seeds.index(r['seed'])], lw=1.5,
                        label=f"seed {r['seed']}  twist {float(r['z']['fit__total_twist']):.3f}")
            ax.set_title(f"{_group_label(g, runs)}   start {start}   "
                         f"(base twist {float(z0['base__total_twist']):.3f})", fontsize=9)
            ax.legend(fontsize=6.8, frameon=False, loc='best')
    fig.suptitle("Twist: stable fixed-point phase vs dose, base (black) against the endpoints",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, 'arms_twist')


def fig_cycles(runs):
    """The limit cycle at every distinct endpoint, in real time, every species on its own scale.

    THIS PANEL EXISTS BECAUSE A PTC FIGURE CANNOT FAIL THIS WAY. The first radial run reported
    RADIALIZED from a fit whose period was 0.159 h with 45x the amplitude -- not a circadian
    oscillator. On the PTC panels it looked fine. Period, amplitude and the Floquet multiplier
    are printed in every title for the same reason.
    """
    groups, _cells = objective_groups(runs)
    panels = []
    for g in groups:
        sel = sorted([r for r in runs if r['arm'] == g[0]],
                     key=lambda x: (x['start'], x['seed']))
        panels.append((f"BASE -- {_group_label(g, runs)}", sel[0]['z'], 'base'))
        panels += [(f"{_group_label(g, runs)}  {r['start']}  seed {r['seed']}", r['z'], 'fit')
                   for r in sel]

    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.7 * nrow), squeeze=False)
    for k, (title, z, pref) in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        cyc = np.asarray(z[f'cyc_{pref}'])
        T = float(z[f'{pref}__period'])
        mu = float(z[f'{pref}__mu'])
        amp = float(z[f'{pref}__amp_lc'])
        t = np.linspace(0, max(T, 1e-9), cyc.shape[0])
        obs = [str(s) for s in z['obs']] if 'obs' in z else [f'x{i}' for i in range(cyc.shape[1])]
        for c in range(cyc.shape[1]):
            y = cyc[:, c]
            rng = float(np.nanmax(y) - np.nanmin(y))
            # EACH SPECIES SCALED TO ITS OWN RANGE. Almeida's species span eleven orders of
            # magnitude on a bad parameter set; on shared axes all but the largest render as a
            # flat line at zero, which reads as "the clock stopped" whether or not it did.
            ax.plot(t, (y - np.nanmin(y)) / rng if rng > 0 else np.zeros_like(y),
                    lw=1.0, label=obs[c] if k == 0 else None)
        bad = (not np.isfinite(T)) or T < 6 or T > 96 or not np.isfinite(mu) or mu > 1.0
        ax.set_title(f"{title}\nT={T:.3g} h  amp={amp:.3g}  mu={mu:.3g}", fontsize=7.2,
                     color=('firebrick' if bad else 'black'))
        ax.set_xlabel('time (h)', fontsize=7); ax.tick_params(labelsize=6)
        ax.set_ylabel('per-species scaled', fontsize=6.5)
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    axes[0][0].legend(fontsize=5.5, frameon=False, ncol=2)
    fig.suptitle("Limit cycle at each endpoint. Red title = period, amplitude or Floquet "
                 "multiplier out of range", fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _save(fig, 'arms_cycles')


def fig_contract(runs, yard):
    """The contract, per endpoint -- the PRIMARY outcome, ahead of any cost value.

    A lower yardstick cost on a surface that fails C5 or C6 is not an improvement, so this is
    the figure to read before the cost ones. C2 is post hoc (`fit.stability`) and shows as '-'.
    """
    from fit.contract import CHECKS
    order = sorted(runs, key=lambda r: (r['start'], r['arm'], r['seed']))
    M = np.full((len(order), len(CHECKS)), np.nan)
    labs = []
    for i, r in enumerate(order):
        row = yard.get(r['dirname'], {})
        for j, c in enumerate(CHECKS):
            v = row.get(c)
            M[i, j] = np.nan if v is None else (1.0 if v else 0.0)
        labs.append(f"{r['arm']}  {r['start']}  s{r['seed']}"
                    + ("  !UNSTABLE" if row.get('unstable') else ""))

    fig, ax = plt.subplots(figsize=(8.0, 0.30 * len(order) + 2.0))
    cmap = matplotlib.colors.ListedColormap(['#c0392b', '#27ae60'])
    cmap.set_bad('0.85')
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(CHECKS)))
    ax.set_xticklabels([c.replace('_', '\n') for c in CHECKS], fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labs, fontsize=6.8, family='monospace')
    for i in range(len(order)):
        for j in range(len(CHECKS)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, '.' if M[i, j] else 'X', ha='center', va='center',
                        color='w', fontsize=7)
    ax.set_title("Surface validity contract at every endpoint\n"
                 "green '.' passes, red 'X' fails, grey '-' not run (C2 is post hoc)",
                 fontsize=10)
    return _save(fig, 'arms_contract')


# --------------------------------------------------------------------------- #
def _load_yard(model, analysis, tag):
    p = os.path.join(paths.out_dir(model, analysis, tag, create=False), 'yardstick.json')
    if not os.path.exists(p):
        raise SystemExit(
            f"no {os.path.relpath(p, paths.HERE)}. The yardstick and the contract are the "
            f"expensive part and are deliberately NOT recomputed by a figure script; run\n"
            f"    $PY -m fit.arms_rescore --model {model} --arms <arms> --control <arm> "
            f"--tag {tag} --check")
    blob = json.load(open(p))
    return {r['dirname']: r for r in blob['rows']}


_WHICH = ('collapse', 'groups', 'surfaces', 'twist', 'cycles', 'contract')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('PURE READ')[0].strip())
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--analysis', default='fit_radial')
    ap.add_argument('--tag', default='arms_batch1')
    ap.add_argument('--which', nargs='+', default=['all'], choices=('all',) + _WHICH)
    ap.add_argument('--publish', action='store_true')
    a = ap.parse_args(argv)

    _CTX.update(model=a.model, analysis=a.analysis, tag=a.tag, publish=a.publish)
    runs = load_runs(a.model, a.arms, a.analysis)
    yard = _load_yard(a.model, a.analysis, a.tag)
    which = _WHICH if 'all' in a.which else a.which
    groups, _cells = objective_groups(runs)

    print(f"[figures_arms] {len(runs)} runs, {len(groups)} distinct search group(s):")
    for g in groups:
        print(f"               {_group_label(g, runs):<18s} {' = '.join(g)}")

    if 'collapse' in which:
        fig_collapse(runs, yard)
    if 'groups' in which:
        fig_groups(runs, yard)
    if 'surfaces' in which:
        for g in groups:
            fig_surfaces(runs, g, _group_label(g, runs))
    if 'twist' in which:
        fig_twist(runs)
    if 'cycles' in which:
        fig_cycles(runs)
    if 'contract' in which:
        fig_contract(runs, yard)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
