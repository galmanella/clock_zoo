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


def load_scrit(model_name, target, mode='instant', tag=None):
    """(S_crit, full_dose_grid, dt) for one target from the newest scrit run."""
    tag = tag or paths.latest_run(model_name, 'scrit')
    if tag is None:
        raise SystemExit(f"run `python -m analysis.scrit --model {model_name} "
                         f"--mode {mode}` first")
    d = paths.out_dir(model_name, 'scrit', tag, create=False)
    for fp in sorted(glob.glob(os.path.join(d, f'scrit_{mode}*.npz'))):
        z = np.load(fp, allow_pickle=True)
        names = [str(t) for t in z['targets']]
        if target in names:
            i = names.index(target)
            grid = np.asarray(z[f'grid__{target}']) if f'grid__{target}' in z else None
            dt = float(z['dt_used'][i]) if 'dt_used' in z.files else 0.02
            return float(z['S_crit'][i]), grid, dt
    raise SystemExit(f"no scrit entry for {target} ({mode}) in {d}")


def fit_dose_grid(model_name, target, mode='instant', max_factor=6.0, n=10, tag=None,
                  lo_factor=0.15):
    """(doses, S_crit) -- a log-spaced fit window from lo_factor*S_crit to max_factor*S_crit.

    Log-spaced rather than a subset of the scrit grid so the sampling is under this module's
    control and reproducible from (S_crit, max_factor, n) alone, without depending on which
    adaptive grid happened to be saved.

    The low end matters as much as the high end: the type-1 region is where the map is closest
    to smooth, and it is what pins the target's dose scale `k`. Starting at 0.15*S_crit gives
    roughly a decade below the transition.
    """
    S, _grid, dt = load_scrit(model_name, target, mode, tag)
    if not np.isfinite(S):
        raise SystemExit(f"{target} ({mode}) has no S_crit -- it does not reset, so there is "
                         f"no transition to fit")
    return np.geomspace(lo_factor * S, max_factor * S, int(n)), S
