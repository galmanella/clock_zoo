"""
fit/multires.py
===============
Multi-resolution continuation: fit on a coarse (phase x dose) grid, then refine, warm-starting
each stage from the last.

WHY THIS IS THE CHEAPEST SMOOTHING OF ALL
    The ruggedness in this landscape comes from the spiral geometry around the phase singularity:
    matching two surfaces is a cross-correlation of two spirals, so every half-turn of relative
    rotation is a local minimum (PROJECT_SUMMARY 5.4c). Crucially the ruggedness gets DENSER as
    resolution improves -- a finer grid resolves more turns and therefore more minima.

    Read the other way, that is a gift: a COARSE grid cannot resolve the fine turns at all, so
    the cost on it is smooth by construction. No kernel, no averaging, no extra evaluations --
    the smoothing is free, and a coarse evaluation is also cheaper than a fine one.

    So the schedule does double duty. Early stages are both smoother AND faster; the expensive
    fine grid is only ever visited from a starting point that is already close.

RELATION TO THE OTHER TWO SMOOTHERS
    `search.smoothed_trust_region` averages over a ball in PARAMETER space; `search.cma(mode=
    'anneal')` does the same implicitly through its population width. This averages in DATA
    space instead, and composes with either -- the natural pairing is a coarse-grid global stage
    followed by a fine-grid local one.

A CAVEAT WORTH STATING
    Coarse and fine costs are different functions, not the same function at different accuracy.
    A coarse grid can miss the singularity entirely (its amplitude dip falls between cells), and
    its S_crit is quantized to a handful of dose values. So the coarse stage locates a basin; it
    does not produce a number worth quoting. Only the final stage's fit is a result, and the
    per-stage costs below are NOT comparable to each other.
"""
import argparse
import os
import time

import numpy as np

import paths
from fit import search


#: (n_phase, n_dose) stages, coarse to fine. The last should match whatever resolution the
#: result is meant to be quoted at.
DEFAULT_SCHEDULE = ((8, 6), (12, 8), (16, 10), (24, 16))


def run_schedule(build_cost, v0, schedule=DEFAULT_SCHEDULE, optimizer='cma', bound=3.0,
                 maxfev=1200, maxiter=60, seed=0, verbose=True):
    """Walk the schedule, warm-starting each stage from the previous stage's solution.

    `build_cost(n_phase, n_dose) -> cost dict` is a factory rather than a cost, because each
    stage is a different grid and therefore a different objective.
    """
    v = np.asarray(v0, float)
    stages = []
    t0 = time.time()
    for si, (nph, nd) in enumerate(schedule):
        C = build_cost(nph, nd)
        f_start = C['total'](v)
        if verbose:
            print(f"\n  [stage {si}] {nph} phases x {nd} doses ({nph * nd} cells)  "
                  f"start f={f_start:.6f}", flush=True)
        if optimizer == 'bobyqa':
            r = search.bobyqa(C, v, bound=bound, maxfev=maxfev, seek_global=True,
                              verbose=verbose)
        elif optimizer == 'cma':
            # anneal, not ipop: the point of the coarse stage is a smooth envelope, and
            # restarting from random points would throw away the warm start
            r = search.cma(C, v0=v, bound=bound, maxfev=maxfev, seed=seed + si,
                           mode='anneal', verbose=verbose)
        elif optimizer == 'lm':
            r = search.levenberg_marquardt(C, v, bound=bound, maxiter=maxiter,
                                           verbose=verbose, label=f'stage{si}')
        else:
            r = search.lbfgs(C, v, bound=bound, maxiter=maxiter, verbose=verbose,
                             label=f'stage{si}')
        v = r['v']
        p = C['parts'](v)
        stages.append(dict(stage=si, n_phase=nph, n_dose=nd, f_start=float(f_start),
                           f_end=float(r['f']), v=v.copy(), nev=r.get('nev', -1),
                           seconds=r.get('seconds', np.nan), c_ptc=p['c_ptc'],
                           amp_lc=p['amp_lc'], alive_frac=p['alive_frac']))
        if verbose:
            print(f"  [stage {si}] f {f_start:.6f} -> {r['f']:.6f}  "
                  f"c_ptc={p['c_ptc']:.6f}  ({r.get('nev', -1)} evals)", flush=True)

    if verbose:
        print(f"\n  {'stage':>6s} {'grid':>10s} {'f start':>11s} {'f end':>11s} "
              f"{'c_ptc':>10s} {'evals':>7s}")
        for st in stages:
            print(f"  {st['stage']:6d} {st['n_phase']:4d}x{st['n_dose']:<5d} "
                  f"{st['f_start']:11.6f} {st['f_end']:11.6f} {st['c_ptc']:10.6f} "
                  f"{st['nev']:7d}")
        print(f"  total {time.time() - t0:.0f}s")
        print("  NOTE: per-stage costs are NOT comparable -- each stage is a different grid, "
              "hence a different objective. Only the last stage is a result.")
    return v, stages


def build_radial(model_name='almeida', target='BMAL1', mode='instant', max_factor=6.0,
                 backend='diffrax', dt=0.02, w_osc=0.2, w_amp=1.0):
    """A `build_cost(n_phase, n_dose)` factory for the radial-isochron target."""
    from models import get_model
    from fit.cost import make_cost, RadialTarget
    from fit.doses import fit_dose_grid
    model = get_model(model_name)

    def build(n_phase, n_dose):
        doses, _s = fit_dose_grid(model_name, target, mode, max_factor, n_dose)
        return make_cost(model, target, doses, RadialTarget(), n_phase=n_phase, mode=mode,
                         backend=backend, dt=dt, w_osc=w_osc, w_amp=w_amp)
    return build


def main(argv=None):
    ap = argparse.ArgumentParser(description='multi-resolution continuation fit')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    ap.add_argument('--mode', default='instant', choices=('pulse', 'instant'))
    ap.add_argument('--optimizer', default='cma', choices=('cma', 'bobyqa', 'lm', 'lbfgs'))
    ap.add_argument('--schedule', default='8x6,12x8,16x10,24x16')
    ap.add_argument('--maxfev', type=int, default=1200)
    ap.add_argument('--maxiter', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)

    sched = tuple(tuple(int(x) for x in s.split('x')) for s in a.schedule.split(','))
    build = build_radial(a.model, a.target, a.mode)
    tag = paths.run_tag(a.tag)

    from parallel import announce
    announce(analysis='fit.multires', model=a.model, target=a.target, mode=a.mode,
             optimizer=a.optimizer, schedule=a.schedule, maxfev=a.maxfev, tag=tag)

    C0 = build(*sched[0])
    v, stages = run_schedule(build, np.zeros(C0['n_free']), sched, a.optimizer,
                             maxfev=a.maxfev, maxiter=a.maxiter, seed=a.seed)

    blob = dict(model=a.model, target=a.target, mode=a.mode, optimizer=a.optimizer,
                schedule=np.array(sched), v_final=v,
                stage=np.array([s['stage'] for s in stages]),
                n_phase=np.array([s['n_phase'] for s in stages]),
                n_dose=np.array([s['n_dose'] for s in stages]),
                f_start=np.array([s['f_start'] for s in stages]),
                f_end=np.array([s['f_end'] for s in stages]),
                c_ptc=np.array([s['c_ptc'] for s in stages]),
                amp_lc=np.array([s['amp_lc'] for s in stages]),
                alive_frac=np.array([s['alive_frac'] for s in stages]),
                nev=np.array([s['nev'] for s in stages]),
                v_stages=np.array([s['v'] for s in stages]))
    out = paths.out_path(a.model, 'fit_multires',
                         f'multires_{a.target}_{a.mode}_{a.optimizer}.npz', tag)
    paths.savez(out, **blob)
    print(f"\n[multires] -> {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
