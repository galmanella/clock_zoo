"""
fit/contract.py
===============
IS THIS SURFACE SCOREABLE? Six checks, computed from the production surface.

    python -m fit.contract --model almeida --tag bmal1seeds     # audit a campaign
    python -m fit.contract --selftest                           # the regression fixture

WHY A CONTRACT AND NOT MORE PENALTY TERMS
    The Aug-30 campaign's pathologies look like the optimizer finding cheap corners. Measured,
    they are not: correlations of cost against killed-cell fraction, phase under-resolution,
    cycle diameter and period are -0.14, -0.17, +0.09, +0.04 over 15 runs, and the three
    cheapest fits have healthy readout amplitude (PROJECT_SUMMARY 5.12a). For a large minority
    of candidates `c_ptc` is simply computed on something that is not a PTC -- the optimizer is
    handed noise and asked to minimise it.

    So the first move is not to reshape the objective. It is to be able to SAY, per candidate,
    whether its number means anything. `docs/FIT_VALIDITY.md` is the plan; this is its P0, and
    it changes no objective at all.

THE CHECKS -- all model-agnostic, target-agnostic, task-agnostic
    C1 orbit        the BVP converged to something that is a limit cycle       (hazard 2)
    C2 attractor    ... and a trajectory actually stays on it                  (hazard 19)
    C3 calibration  dose 0 returns the identity, so the phase readout works    (hazard 20)
    C4 resolution   the phase grid resolves the cycle                          (5.12)
    C5 aliveness    the perturbation did not extinguish the oscillation        (hazard 3)
    C6 bracketing   the dose window contains the type-1 -> type-0 transition   (hazard 17)

FIVE OF THE SIX ARE FREE, AND THAT IS THE POINT
    C1, C4, C5 and C6 are functions of things the cost already computes -- the orbit, the cycle,
    the per-cell amplitude, the surface -- and C3 costs one extra dose row. Only C2 is expensive
    (~7 s: a 24-period walk over an epsilon ladder, `fit/stability.py`), and it is deliberately
    NOT computed here: C3 flags the same runs for one element each way, at about a thousandth of
    the cost, so the cheap check screens in the loop and the expensive one confirms post hoc.

EVERYTHING RUNS THROUGH THE PRODUCTION PATH
    The orbit is re-solved with `make_guess_fn` and `C['solver']` -- the same pair `_surface`
    uses -- and the surface comes from `C['surface']`. A diagnostic that reimplements the thing
    it checks cannot check it (hazard 15), and that mistake has already been made twice here.
"""
import argparse
import glob
import json
import os

import numpy as np

import paths

#: BVP residual above which the solve did not converge to a periodic orbit
RES_TOL = 1e-4
#: dose-0 identity residual, in cycles. Two orders clear of both populations -- calibrated runs
#: land at 1e-4..9e-4 (the Fourier readout's own resolution), broken ones at 4e-2..5e-1.
ID_TOL = 1e-2
#: max arc-length between adjacent phase samples, as a fraction of the cycle diameter.
#: A DESIGN CHOICE, not a measured gap: on the campaign's 20-phase grid this quantity is a
#: continuum from 0.18 to 0.87 with no break, so 0.25 is "a quarter of the cycle between
#: neighbouring columns" and it flags 12 of 15. Raise `n_phase` until it passes.
ARC_TOL = 0.25
#: median per-cell |z| below which the perturbation has extinguished the oscillation over most
#: of the window. `engine.ptc.DEAD_AMP = 0.01` means "not literally zero"; it does not mean
#: "carries phase information", and at |z| = 0.05 the Fourier phase is noise.
ALIVE_TOL = 0.2

CHECKS = ('C1_orbit', 'C2_attractor', 'C3_calibration', 'C4_resolution',
          'C5_aliveness', 'C6_bracketing')


def evaluate(C, model, v, with_c2=False):
    """The contract for one candidate. Returns a dict of numbers plus per-check booleans.

    `with_c2=True` additionally runs the stability walk (~7 s). Off by default -- see the module
    docstring on why C3 is the in-loop proxy.
    """
    import jax
    import jax.numpy as jnp
    from engine.orbit import make_guess_fn
    from engine.ptc import DEAD_AMP
    from analysis import winding as W

    old = np.asarray(C['old'], float)
    doses = np.asarray(C['doses'], float)
    n_phase = len(old)
    solver = C['solver']
    P = model.jax_apply(np.asarray(C['theta'](v)), list(C['names']))
    guess = make_guess_fn(model)
    y_seed = jnp.asarray(model.get_initial_state(), jnp.float64)

    out = {}

    # --- C1: is it an orbit at all? ---------------------------------------------------- #
    y0, T, res = jax.jit(solver.solve)(P, guess(P, y_seed))
    T, res = float(T), float(res)
    T_nom = float(getattr(model, 'approx_period', None) or 24.0)
    cyc = np.asarray(solver.cycle(P, y0, T, max(n_phase, 64)))
    finite = bool(np.isfinite(cyc).all()) and np.isfinite(T)
    ref = int(model.var_index(model.reference_variable))
    c = cyc[:, ref] if finite else np.zeros(2)
    amp_lc = float((c.max() - c.min()) / max(abs(c.mean()), 1e-12)) if finite else 0.0
    out['orbit_res'] = res
    out['period'] = T
    out['amp_lc'] = amp_lc
    out['C1_orbit'] = bool(finite and res < RES_TOL and 0.25 * T_nom < T < 4.0 * T_nom
                           and cyc.min() >= -1e-8 and amp_lc > 1e-3)

    # --- C4: does the phase grid resolve the cycle? ------------------------------------- #
    #
    # Sampled at the GRID's n_phase, not at some convenient resolution: the question is whether
    # the columns the cost actually compares are neighbours in state space.
    if out['C1_orbit']:
        cg = np.asarray(solver.cycle(P, y0, T, n_phase))
        step = np.linalg.norm(np.diff(np.vstack([cg, cg[:1]]), axis=0), axis=1)
        diam = np.linalg.norm(cg.max(0) - cg.min(0))
        out['arc_step'] = float(step.max() / max(diam, 1e-30))
    else:
        out['arc_step'] = np.nan
    out['C4_resolution'] = bool(np.isfinite(out['arc_step']) and out['arc_step'] < ARC_TOL)

    # --- the surface, then C5 and C6 ---------------------------------------------------- #
    zu, alive, amp = C['surface'](v)
    alive = np.asarray(alive)
    amp = np.asarray(amp, float)
    ptc = np.where(alive, (np.angle(np.asarray(zu)) / (2 * np.pi)) % 1.0, np.nan)
    out['alive_frac'] = float(alive.mean())
    out['amp_median'] = float(np.median(amp))
    out['amp_frac_below'] = float(np.mean(amp < ALIVE_TOL))
    out['C5_aliveness'] = bool(out['amp_median'] >= ALIVE_TOL)

    w = W.winding_vs_dose(ptc)
    wf = np.asarray(w, float)
    ok = np.isfinite(wf)
    wset = sorted(set(int(x) for x in wf[ok])) if ok.any() else []
    out['winding_set'] = wset
    # a transition INSIDE the window, and not merely at an edge row -- an edge change is as
    # likely to mean the transition is just outside as just inside
    changed = np.where(ok[:-1] & ok[1:] & (wf[:-1] != wf[1:]))[0] if len(wf) > 1 else np.array([])
    interior = [int(i) for i in changed if 0 < i < len(wf) - 2]
    out['n_winding_changes'] = int(len(changed))
    out['C6_bracketing'] = bool(len(interior) > 0)
    # topologically impossible windings are a separate tell (aliasing or a tumbling readout)
    out['impossible_winding'] = bool(any(x < 0 or x > 1 for x in wset))

    # --- C3: dose 0 must return the identity -------------------------------------------- #
    #
    # If the grid already carries a dose-0 row, use it -- free. Otherwise evaluate one, which is
    # the cheapest cell on any grid: no perturbation, no stiffness.
    j0 = int(np.argmin(doses))
    if doses[j0] == 0.0:
        row = ptc[:, j0]
    else:
        from engine.ptc import grid_points
        ph, dz = grid_points(n_phase, np.array([0.0]))
        zr, ymin = C['ptc_fn'](P, jnp.concatenate([y0, jnp.asarray([T])]), ph, dz)
        zr = np.asarray(zr).ravel()
        good = np.isfinite(zr.real) & np.isfinite(zr.imag) & (np.abs(zr) > DEAD_AMP)
        row = np.where(good, (np.angle(zr) / (2 * np.pi)) % 1.0, np.nan)
    d0 = np.abs(((row - old) + 0.5) % 1.0 - 0.5)
    out['identity_err'] = float(np.nanmax(d0)) if np.isfinite(d0).any() else np.nan
    out['C3_calibration'] = bool(np.isfinite(out['identity_err'])
                                 and out['identity_err'] <= ID_TOL)

    # --- C2: post hoc only -------------------------------------------------------------- #
    if with_c2:
        from fit.stability import make_walker, measure, classify
        walk = make_walker(model, backend=str(C.get('backend', 'diffrax')))
        mres = measure(walk, P, np.asarray(y0), T, model.n_states)
        verdict, why = classify(mres, out['C1_orbit'])
        out['growth'] = mres['growth']
        out['floor'] = mres['floor']
        out['stability_verdict'] = verdict
        out['C2_attractor'] = bool(verdict == 'ATTRACTING')
    else:
        out['growth'] = np.nan
        out['floor'] = np.nan
        out['stability_verdict'] = 'NOT RUN'
        out['C2_attractor'] = None

    out['scoreable'] = bool(all(out[k] for k in CHECKS if out[k] is not None))
    return out


def fmt(r):
    """One line per candidate, with a dash where a check was not run."""
    def m(k):
        v = r[k]
        return '-' if v is None else ('.' if v else 'X')
    return (f"{m('C1_orbit')}{m('C2_attractor')}{m('C3_calibration')}{m('C4_resolution')}"
            f"{m('C5_aliveness')}{m('C6_bracketing')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('WHY A CONTRACT')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--tag', default=None, help='aggregated campaign tag')
    ap.add_argument('--analysis', default='fit_radial')
    ap.add_argument('--with-c2', action='store_true',
                    help='also run the stability walk (~7 s per candidate)')
    ap.add_argument('--selftest', action='store_true',
                    help='reproduce the measured regression fixture')
    a = ap.parse_args(argv)

    from models import get_model
    from fit.cost import make_cost, RadialTarget

    tag = a.tag or 'bmal1seeds'
    d = paths.out_dir(a.model, a.analysis, tag, create=False)
    fs = sorted(glob.glob(os.path.join(d, 'seeds_*.npz')))
    if not fs:
        raise SystemExit(f"no aggregated seeds_*.npz under {d}; run `python -m fit.aggregate` "
                         f"--model {a.model} --tag {tag}` first.")
    z = dict(np.load(fs[-1], allow_pickle=True))
    cfg = json.loads(str(z['cfg_json']))
    model = get_model(a.model)
    model.reference_variable = str(z['section'])
    labels = [str(x) for x in z['labels']]

    C = make_cost(model, str(z['target']), np.asarray(z['doses'][0]),
                  RadialTarget.from_singularity(1.0 / float(z['k_used'][0]),
                                                (float(z['psi_used'][0]) + 0.5) % 1.0),
                  n_phase=int(cfg['n_phase']), mode=str(z['mode']), backend='diffrax',
                  dt=float(cfg['dt']), w_osc=float(cfg['w_osc']), w_amp=float(cfg['w_amp']),
                  pulse=float(cfg['pulse']), skip_p=cfg['skip_p'],
                  readout_ref=str(z['readout']), basis=np.asarray(z['B']))

    print(f"contract on {tag}: {len(labels)} candidates"
          + ("  (+ the stability walk)" if a.with_c2 else "  (C2 not run -- post hoc)"))
    print(f"  {'seed':>5s} {'C1..C6':>7s} {'res':>9s} {'arc':>7s} {'|z|med':>7s} "
          f"{'id_err':>9s} {'wind':>14s}  scoreable")
    fails = {k: [] for k in CHECKS}
    for i, L in enumerate(labels):
        r = evaluate(C, model, np.asarray(z['v'][i]), with_c2=a.with_c2)
        for k in CHECKS:
            if r[k] is False:
                fails[k].append(L)
        print(f"  {L:>5s} {fmt(r):>7s} {r['orbit_res']:9.1e} {r['arc_step']:7.3f} "
              f"{r['amp_median']:7.3f} {r['identity_err']:9.1e} "
              f"{str(r['winding_set']):>14s}  {'yes' if r['scoreable'] else 'NO'}")
    print("\n  ('.' passes, 'X' fails, '-' not run;  order C1 C2 C3 C4 C5 C6)")
    for k in CHECKS:
        if fails[k]:
            print(f"  {k:<16s} fails: {sorted(fails[k], key=int)}")

    if a.selftest:
        # THE REGRESSION FIXTURE. Measured on bmal1seeds and pinned here, because these
        # thresholds are only defensible as long as they keep reproducing the sets they were
        # derived from. C2 is excluded: it is not run in the loop.
        want = {'C1_orbit': ['3'],
                'C3_calibration': ['3', '4', '5', '11', '14'],
                'C5_aliveness': ['2', '5', '11', '14']}
        ok = True
        for k, exp in want.items():
            got = sorted(fails[k], key=int)
            good = got == exp
            ok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] {k}: {got}" + ('' if good else f"  expected {exp}"))
        # TWO EXPECTATIONS MOVED WHEN THE CHECKS WERE IMPLEMENTED, both because the loose
        # definition used to derive them was the wrong one. Recorded rather than quietly
        # updated, since silently moving a fixture to match new code defeats the fixture.
        #
        #   C4: 12 -> 13. The 12 came from the 15 candidates with a finite cycle; seed 3 has
        #       none, so it fails C4 as well as C1. 13 of 16 is the honest count.
        #   C6: 15 -> 16. The 15 came from `detect_grid`'s n_sing, which finds seed 2's
        #       singularity. But seed 2's winding changes between dose ROW 0 AND ROW 1 -- one
        #       row of type-1 at the very bottom of the window - and the extended rescan shows
        #       its real structure lives from 0.02 up to ~8, i.e. mostly below the window.
        #       "A transition sitting on the boundary row" is not a bracketed window, so C6
        #       requires an INTERIOR change and seed 2 fails it. All 16 fail C6.
        n4 = len(fails['C4_resolution'])
        print(f"  [{'PASS' if n4 == 13 else 'FAIL'}] C4_resolution flags {n4}, expected 13")
        n6 = len(fails['C6_bracketing'])
        print(f"  [{'PASS' if n6 == 16 else 'FAIL'}] C6_bracketing flags {n6}, expected 16")
        ok &= (n4 == 13) and (n6 == 16)
        print("  selftest", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
