"""
fit/search.py
=============
Optimizers over the gauge quotient. L-BFGS is primary; CMA-ES is an opt-in cross-check.

WHY GRADIENT-PRIMARY
    REPO_MAP hazard 1 records that autodiffing the FIXED-STEP RK4 PTC blew up (|J| ~ 3.95e83
    against a true response of 0.006). That indicts fixed-step RK4, not gradients: input_screen
    resolved it with the adaptive Tsit5 backend, which is now ported (engine/flow.py), and on
    Mirsky the gradient optimizer recovered 132/132 parameters from the truth while CMA-ES did
    worse. At 16 free directions a quasi-Newton method should also simply cost less: a few
    hundred gradient evaluations against CMA's popsize-12 x 300 generations ~ 3600 forward
    evaluations.

    That said, the gradient is only usable where `fit.cost --gradcheck` passes. It does NOT pass
    on an unrestricted dose grid -- see fit/cost.make_cost's `max_dose_factor`. Run the check,
    do not assume.

WHY MULTISTART RATHER THAN CMA FOR GLOBALITY
    The question is whether distinct basins exist, which wants many INDEPENDENT converged
    solutions to cluster -- what multistart produces directly. CMA-ES answers a different
    question (find me one good point) and spends its budget concentrating a population.
    `--optimizer cma` is kept for the case where the gradient is unusable or the landscape
    turns out to be barrier-dominated; it ports the IPOP-restart structure from
    input_screen/fit_jax.py:run_cma_points, whose `nan_to_num(..., 1e6)` infeasible-handling
    convention is reused here too.

EVERY RUN SAVES ITS RAW INTERMEDIATES
    The full iterate trace, all per-start solutions, and the model surface at each optimum --
    not just the winning parameter vector. Analysis is expensive, replotting is cheap.
"""
import time

import numpy as np

#: What an unusable candidate scores. Large but FINITE: an inf or NaN makes L-BFGS abandon the
#: line search and CMA rank the whole population by chance.
BIG = 1e6


def _harden(cost):
    """(f, g) wrappers that never return NaN/inf, and that record every evaluation."""
    trace = {'v': [], 'f': []}

    def f(v):
        try:
            val = cost['total'](v)
        except Exception:
            val = np.nan
        val = float(val) if np.isfinite(val) else BIG
        trace['v'].append(np.asarray(v, float).copy())
        trace['f'].append(val)
        return val

    def g(v):
        try:
            gr = cost['grad'](v)
        except Exception:
            gr = None
        if gr is None or not np.all(np.isfinite(gr)):
            return np.zeros(len(v))
        return np.asarray(gr, float)

    return f, g, trace


def lbfgs(cost, v0, bound=3.0, maxiter=300, verbose=True, label=''):
    """One L-BFGS-B descent from `v0`. Returns a result dict with the full trace."""
    from scipy.optimize import minimize
    f, g, trace = _harden(cost)
    n = len(v0)
    t0 = time.time()
    res = minimize(f, np.asarray(v0, float), jac=g, method='L-BFGS-B',
                   bounds=[(-bound, bound)] * n,
                   options={'maxiter': maxiter, 'maxfun': maxiter * 2})
    dt = time.time() - t0
    p = cost['parts'](res.x)
    if verbose:
        print(f"    {label:14s} f {trace['f'][0]:.4f} -> {res.fun:.4f} in {res.nit} its "
              f"({len(trace['f'])} evals, {dt:.0f}s)  c_ptc={p['c_ptc']:.4f} "
              f"amp_lc={p['amp_lc']:.2f} alive={p['alive_frac']:.2f}", flush=True)
    return dict(v=np.asarray(res.x), f=float(res.fun), f0=float(trace['f'][0]),
                nit=int(res.nit), nev=len(trace['f']), seconds=dt,
                success=bool(res.success), message=str(res.message),
                trace_v=np.array(trace['v']), trace_f=np.array(trace['f']), parts=p)


def multistart(cost, n_starts=8, bound=3.0, sigma=0.8, maxiter=300, seed=0,
               include_base=True, verbose=True):
    """Independent L-BFGS descents from random points, for basin structure.

    `include_base` keeps start 0 at the base parameter set, so a run always contains the
    from-nominal descent that the single-start experiments report.
    """
    rng = np.random.default_rng(seed)
    n = cost['n_free']
    starts = []
    if include_base:
        starts.append(np.zeros(n))
    while len(starts) < n_starts:
        starts.append(np.clip(rng.normal(0, sigma, n), -bound, bound))
    out = []
    for i, s in enumerate(starts):
        out.append(lbfgs(cost, s, bound, maxiter, verbose, label=f'start {i}'))
        out[-1]['v0'] = s
    out.sort(key=lambda r: r['f'])
    return out


def cma(cost, v0=None, bound=3.0, sigma0=0.5, maxfev=4000, popsize=None, seed=0,
        restarts=2, popsize_factor=2.0, gens_per_run=None, verbose=True):
    """CMA-ES with IPOP-style restarts. The cross-check, not the default.

    Ported in structure from input_screen/fit_jax.py:run_cma_points -- growing-popsize restarts
    from fresh points, which was described there as the cure for a barrier-separated PTC
    landscape that a single diffuse run cannot cross. `maxfev` is the TOTAL budget across
    restarts.
    """
    import cma as _cma
    n = cost['n_free']
    v0 = np.zeros(n) if v0 is None else np.asarray(v0, float)
    if popsize is None:
        popsize = int(4 + 3 * np.log(n))
    rng = np.random.default_rng(seed + 7)
    f, _g, trace = _harden(cost)

    best_v, best_f, used, ps, runs = None, np.inf, 0, popsize, []
    for r in range(restarts + 1):
        if r == 0:
            u0 = v0
        else:
            u0 = np.clip(rng.uniform(-bound, bound, n), -bound, bound)
        es = _cma.CMAEvolutionStrategy(
            list(u0), sigma0,
            {'bounds': [[-bound] * n, [bound] * n], 'popsize': int(ps),
             'maxfevals': int(maxfev - used), 'verbose': -9, 'seed': seed + 1 + r * 997})
        gen = 0
        while not es.stop():
            U = es.ask()
            vals = [f(u) for u in U]
            es.tell(U, vals)
            used += len(U); gen += 1
            if gens_per_run and gen >= gens_per_run:
                break
        if es.result.fbest < best_f:
            best_f, best_v = float(es.result.fbest), np.asarray(es.result.xbest)
        runs.append(dict(restart=r, popsize=int(ps), gens=gen, fbest=float(es.result.fbest)))
        if verbose:
            print(f"    [restart {r} pop {int(ps)}] {gen} gens, best {es.result.fbest:.4f} "
                  f"(overall {best_f:.4f})", flush=True)
        ps *= popsize_factor
        if used >= maxfev:
            break
    return dict(v=best_v, f=best_f, runs=runs, nev=len(trace['f']),
                trace_v=np.array(trace['v']), trace_f=np.array(trace['f']),
                parts=cost['parts'](best_v))
