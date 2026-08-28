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
    backend, gradient of c_ptc at nominal):

        doses 10-21    (0.4-0.9 x S_crit)   |grad| = 6.9e-01
        doses 12-21    (0.5-0.9 x S_crit)   |grad| = 7.7e-01
        doses 75-129   (3-5 x S_crit)       |grad| = 2.4e+00
        full grid to 454                    |grad| = 2.0e+75

    The cost VALUE stays bounded in [0,1] throughout -- (1-cos)/2 cannot do otherwise -- so
    nothing in the objective looks wrong. Only the derivative explodes, which is the signature
    of a surface that is genuinely non-smooth in PARAMETERS at those doses: an instant kick of
    +454 into a species whose limit cycle spans ~4 sends the trajectory far off the attractor,
    and where it lands is exquisitely sensitive. That is a real property of the model, not a
    solver artifact -- note it survives the switch to an adaptive integrator, which is what
    fixed the analogous RK4 problem.

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
