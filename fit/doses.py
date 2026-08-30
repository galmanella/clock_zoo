"""
fit/doses.py
============
The dose window a FIT should use, which is not the same as the window a CHARACTERIZATION
should use.

WHY THEY DIFFER
    `analysis/scrit.py` builds a deliberately wide adaptive grid -- S_crit at roughly the 30th
    log-percentile, so the type-0 side spans ~70% of the range -- because the twist lives ABOVE
    S_crit and a characterization must resolve it. For BMAL1/instant that grid runs to 454,
    about 18x S_crit = 24.97.

    A fit cannot use the top of that range. MEASURED (Almeida/BMAL1/instant, adaptive Tsit5
    backend, n_phase=8, gradient of c_ptc at nominal, S_crit = 24.97):

        dose 7.2 alone          (0.3 x S)     |grad| = 2.1e-01
        doses 7.2, 14.8, 36.5   (0.3-1.5 x S) |grad| = 4.8e-01
        doses 10-21             (0.4-0.9 x S) |grad| = 6.9e-01
        doses 75-129            (3-5 x S)     |grad| = 2.4e+00
        dose 154 alone          (6 x S)       |grad| = 2.9e-01
        all of 7.2 ... 454.4    (to 18 x S)   |grad| = 1.2e+73    <-- and the value is FINE

    Every sub-range is well behaved, so the blow-up comes from one or two cells at the very top
    (184.6 and/or 454.4 -- the only doses in the failing set not individually cleared).

    THREE EXPLANATIONS RULED OUT BY MEASUREMENT, so this note does not repeat a guess:
      * not the oscillation or amplitude terms -- the Hopf barrier's own gradient is 1.4e-03;
      * not amplitude collapse near the singularity, and not a runaway -- across the whole
        failing grid the relative amplitude is between 0.999 and 1.000475 and every one of the
        48 cells is alive, i.e. every trajectory returns cleanly to the limit cycle;
      * not reverse-mode instability over the long transient skip -- shortening skip_p from 8
        to 1 leaves the gradient at 1.2e+73, unchanged.

    What is left is that the map really is non-smooth in PARAMETERS at those doses. Almeida has
    a stable equilibrium coexisting with its limit cycle (see analysis/lc_sens: continuation had
    to be seeded through the relaxation for exactly this reason), and an instant kick of +454
    into a species whose cycle spans ~4 lands the state near that basin boundary. The
    trajectory still returns -- hence a perfectly healthy amplitude -- but WHERE it returns in
    phase depends on which side of the boundary it passed, so the derivative is enormous while
    the value is bounded and correct. The cost cannot see this: (1-cos)/2 is bounded in [0,1]
    by construction, so nothing in the objective looks wrong.

    That is a property of the model, not of the solver -- it survives the switch to an adaptive
    integrator, which is what fixed the analogous fixed-step RK4 problem (REPO_MAP hazard 1).

    So the fit window is capped at `max_factor * S_crit`. The default 6.0 keeps the entire
    informative structure -- the type-1 side, the transition, the singularity, and a stretch of
    developed type-0 -- while staying well inside the smooth region. Doses far above S_crit are
    not merely dangerous, they are also uninformative: both model and target are featureless
    type-0 there, so they contribute almost nothing about isochron geometry.

THIS IS A REAL LIMITATION AND IT IS REPORTED, NOT HIDDEN
    Capping the window means the fit never sees the far-supercritical regime. If a model
    matched the radial target on [0, 6*S] and diverged from it at 20*S, this fit would not
    know. `fit_dose_grid` therefore returns S_crit alongside the grid and every driver records
    both, so the restriction is visible in the output rather than implicit in a default.
"""
import glob
import os

import numpy as np

import paths


def load_scrit(model_name, target, mode='instant'):
    """(S_crit, full_dose_grid, dt) for one target, read from fixtures/scrit/<model>/.

    NO out/ IS CONSULTED. See the note in the body: a result in out/ silently overriding the
    tracked value is how two machines end up fitting on different dose windows.
    """
    # OUT/ IS OUTPUT. IT IS NEVER AN INPUT. S_crit is read ONLY from fixtures/.
    #
    # An earlier version searched out/ first and fell back to fixtures. That was wrong, and it
    # contradicted its own justification: if a local run in out/ overrides the tracked value,
    # then two machines silently disagree about the dose window and their fits stop being
    # comparable -- the exact failure the fixture was introduced to prevent. It was also
    # fragile in a duller way: an out/ directory that merely EXISTS but is empty, or holds the
    # other perturbation mode, changes which branch runs.
    #
    # So the rule is absolute. `analysis.scrit` PRODUCES S_crit into out/; promoting it to
    # fixtures/ is a separate, deliberate, reviewable act:
    #
    #     python -m fit.doses --promote --model almeida --mode instant [--tag TAG]
    #
    # which makes "the dose window changed" a tracked diff rather than a property of whichever
    # machine happened to run last.
    fixture_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'fixtures', 'scrit', model_name)
    search = [(fixture_dir, 'fixtures')]
    seen = []
    for d, tg in search:
        for fp in sorted(glob.glob(os.path.join(d, f'scrit_{mode}*.npz'))):
            z = np.load(fp, allow_pickle=True)
            names = [str(t) for t in z['targets']]
            if target in names:
                i = names.index(target)
                grid = np.asarray(z[f'grid__{target}']) if f'grid__{target}' in z else None
                dt = float(z['dt_used'][i]) if 'dt_used' in z.files else 0.02
                return float(z['S_crit'][i]), grid, dt
        seen.append(tg)
    raise SystemExit(
        f"no S_crit for {target} ({mode}) in fixtures/scrit/{model_name}/." \
        + chr(10) + f"  1. produce it:  python -m analysis.scrit --model {model_name} "
        f"--mode {mode}" + chr(10)
        + f"  2. promote it:  python -m fit.doses --promote --model {model_name} "
        f"--mode {mode}" + chr(10)
        + f"  3. commit fixtures/scrit/{model_name}/ so every machine agrees.")


def fit_dose_grid(model_name, target, mode='instant', max_factor=8.0, n=10,
                  lo_factor=0.5):
    """(doses, S_crit) -- a log-spaced fit window from lo_factor*S_crit to max_factor*S_crit.

    Log-spaced rather than a subset of the scrit grid so the sampling is under this module's
    control and reproducible from (S_crit, max_factor, n) alone, without depending on which
    adaptive grid happened to be saved.

    THE PLACEMENT OF S_crit IN THE RANGE IS THE POINT, NOT THE ENDPOINTS.
        TWIST LIVES ABOVE S_crit. PROJECT_SUMMARY 3.4 sets the convention -- put S_crit at
        roughly the 30th log-percentile so the type-0 side spans ~70% of the sampled range --
        precisely so the isochron twist is resolved rather than crammed into the top of the
        grid.

        The first version of this function used 0.15 * S_crit to 6 * S_crit, which puts S_crit
        at the 51st percentile: barely half the range is type-0, and the radial fit is entirely
        about twist in that half. 0.5 to 8.0 puts it at the 25th percentile, so type-0 spans
        75%.

            lo=0.15 hi=6    S_crit at 51.4 pct   type-0  48.6%   <- was
            lo=0.5  hi=8    S_crit at 25.0 pct   type-0  75.0%   <- now
            lo=0.4  hi=10   S_crit at 28.5 pct   type-0  71.5%

        8 * S_crit is still well inside the smooth regime: the gradient pathology sets in around
        18 * S_crit, and 3-5 * S_crit measured |grad| = 2.4 against 0.7 near S_crit.
    """
    S, _grid, dt = load_scrit(model_name, target, mode)
    if not np.isfinite(S):
        raise SystemExit(f"{target} ({mode}) has no S_crit -- it does not reset, so there is "
                         f"no transition to fit")
    return np.geomspace(lo_factor * S, max_factor * S, int(n)), S


def promote(model_name, mode='instant', tag=None, dry_run=False):
    """Copy an S_crit result out of out/ into fixtures/ -- the ONLY way it becomes an input.

    Deliberately a separate command rather than something a fit does implicitly. Promoting
    changes the dose window every future fit on every machine will use, so it should be an act
    someone performs and commits, visible as a diff, not a side effect of whoever ran
    `analysis.scrit` most recently.
    """
    import shutil
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = os.path.join(root, 'fixtures', 'scrit', model_name)

    d = paths.model_dir(model_name, 'scrit') if hasattr(paths, 'model_dir') else None
    src_root = d or os.path.join(paths.OUT, model_name, 'scrit')
    tags = [tag] if tag else sorted(
        (t for t in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, t))),
        reverse=True) if os.path.isdir(src_root) else []
    for tg in tags:
        hits = sorted(glob.glob(os.path.join(src_root, tg, f'scrit_{mode}*.npz')))
        if not hits:
            continue
        src = hits[-1]
        z = np.load(src, allow_pickle=True)
        names = [str(t) for t in z['targets']]
        S = np.asarray(z['S_crit'])
        print(f"promoting {src}")
        print(f"  -> {os.path.join(dest, os.path.basename(src))}")
        for nm, sv in zip(names, S):
            print(f"     {nm:9s} S_crit {sv:12.4f}" + ("   (no reset)" if not np.isfinite(sv)
                                                       else ""))
        if dry_run:
            print("  --dry-run: nothing copied")
            return src
        os.makedirs(dest, exist_ok=True)
        for ext in ('', '.meta.json'):
            if os.path.exists(src + ext):
                shutil.copy2(src + ext, os.path.join(dest, os.path.basename(src) + ext))
        print(f"  copied. NOW COMMIT fixtures/scrit/{model_name}/ so the cluster sees it.")
        return src
    raise SystemExit(f"no scrit_{mode}*.npz under {src_root} (tags tried: {tags or 'none'}) -- "
                     f"run `python -m analysis.scrit --model {model_name} --mode {mode}` first")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description='S_crit is read from fixtures/ only; this promotes an out/ result into it.')
    ap.add_argument('--promote', action='store_true', required=True)
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--mode', default='instant', choices=('instant', 'pulse'))
    ap.add_argument('--tag', default=None, help='which out/ run (default: newest that has it)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    promote(a.model, a.mode, a.tag, a.dry_run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

