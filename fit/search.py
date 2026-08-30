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

#: Gradient norm above which the value is the basin-boundary pathology rather than a slope.
#:
#: The cost is bounded in [0, 1] and the search box is |v| <= ~3, so an honest gradient cannot
#: exceed O(10). MEASURED on Almeida/BMAL1/instant: legitimate gradients run 0.2-2.4, while a
#: cell whose trajectory passes near the coexisting equilibrium's basin boundary produces 1e73
#: with a perfectly healthy forward value (see fit/doses.py). `fit/doses.py` keeps those doses
#: out of the fit window, but a DISPLACED parameter set can put a boundary crossing at a dose
#: that is ordinarily safe, so the guard has to be here too.
#:
#: Rescaled, not zeroed: the direction is still meaningful even when the magnitude is not, and
#: zeroing would tell L-BFGS it had converged. Every trigger is COUNTED and reported, because a
#: run where this fires constantly is a run whose result should not be trusted.
GRAD_MAX = 1e3


def _harden(cost):
    """(f, g) wrappers that never return NaN/inf, and that record every evaluation."""
    trace = {'v': [], 'f': [], 'gclip': 0, 'gbad': 0, 'gdead': 0, 'gmax': 0.0}

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
        if gr is None:
            trace['gdead'] += 1
            return np.zeros(len(v))
        gr = np.asarray(gr, float)
        bad = ~np.isfinite(gr)
        if bad.any():
            # COMPONENT-WISE, not all-or-nothing. An all-zero gradient is not a neutral
            # fallback -- it is the convergence signal, and L-BFGS stops on it. In the first T1
            # run 7 of 108 evaluations had a non-finite component (a defective eigenvalue in
            # the Hopf barrier, since fixed at source in fit/cost._leading_np), the whole vector
            # was zeroed, and the search halted at 58 of 150 iterations and reported a landscape
            # verdict that was really this substitution. Keeping the finite components loses
            # only the directions that are genuinely undefined.
            trace['gbad'] += 1
            gr = np.where(bad, 0.0, gr)
            if not gr.any():
                trace['gdead'] += 1
        nrm = float(np.linalg.norm(gr))
        trace['gmax'] = max(trace['gmax'], nrm)
        if nrm > GRAD_MAX:
            trace['gclip'] += 1
            gr = gr * (GRAD_MAX / nrm)
        return gr

    return f, g, trace


def _grad_health(trace, n_eval):
    """One line on how trustworthy the gradients were, or '' if they were clean."""
    if not (trace['gclip'] or trace['gbad'] or trace['gdead']):
        return ''
    msg = (f"  [grad] {trace['gclip']} clipped (>{GRAD_MAX:g}), {trace['gbad']} had a "
           f"non-finite component, {trace['gdead']} were fully zero, "
           f"max |g| = {trace['gmax']:.3e} over {n_eval} evals")
    if trace['gdead']:
        msg += ("\n  [grad] a FULLY ZERO gradient reads as convergence to L-BFGS -- treat any"
                " early stop in this run as suspect")
    return msg


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
        h = _grad_health(trace, len(trace['f']))
        if h:
            print(h, flush=True)
    return dict(v=np.asarray(res.x), f=float(res.fun), f0=float(trace['f'][0]),
                nit=int(res.nit), nev=len(trace['f']), seconds=dt,
                success=bool(res.success), message=str(res.message),
                n_grad_clipped=trace['gclip'], n_grad_bad=trace['gbad'],
                n_grad_dead=trace['gdead'],
                grad_max=trace['gmax'],
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
        restarts=2, popsize_factor=2.0, gens_per_run=None, verbose=True, log_every=10,
        mode='ipop', sigma_factor=0.45, evaluator=None):
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

    # mode='ipop'    restarts from RANDOM points with a growing population -- the standard
    #                cure for a multimodal landscape, and what to use when distinct basins are
    #                the problem.
    # mode='anneal'  restarts from the BEST-SO-FAR with a shrinking sigma -- graduated
    #                optimization. CMA's population is already a smoothing kernel of width
    #                sigma, so a large sigma sees only the smooth envelope of a rugged landscape
    #                and successive contractions resolve finer structure.
    #
    #                That is the right mode HERE. The ruggedness in this problem is the spiral
    #                cross-correlation effect (PROJECT_SUMMARY 5.4c): the surface is smooth at
    #                large scale and rugged at small, and the conditioning of the map improves
    #                30x when probed at radius 0.23 rather than 1e-3. IPOP does the opposite of
    #                what that calls for -- it grows the population while resetting sigma, which
    #                spends evaluations re-exploring rather than sharpening.
    t_start = time.time()
    # EVALUATE THE START. CMA samples a population around v0 but never evaluates v0 itself, so
    # without this the routine can return a point WORSE than the one it was given -- and it did:
    # in a warm-started multi-resolution run, stage 1 went 0.157 -> 0.219, discarding the
    # previous stage's progress. An optimizer that can move backwards is unusable as a
    # continuation step.
    best_v, best_f = np.asarray(v0, float).copy(), f(np.asarray(v0, float))
    used, ps, runs = 1, popsize, []
    sig = sigma0
    for r in range(restarts + 1):
        if r == 0:
            u0 = v0
        elif mode == 'anneal':
            u0 = best_v if best_v is not None else v0
            sig = sig * sigma_factor
        else:
            u0 = np.clip(rng.uniform(-bound, bound, n), -bound, bound)
        es = _cma.CMAEvolutionStrategy(
            list(u0), sig,
            {'bounds': [[-bound] * n, [bound] * n], 'popsize': int(ps),
             'maxfevals': int(maxfev - used), 'verbose': -9, 'seed': seed + 1 + r * 997})
        gen = 0
        t_gen = time.time()
        while not es.stop():
            U = es.ask()
            if evaluator is None:
                vals = [f(u) for u in U]
            else:
                # THE ONE PARALLEL LINE. Members of a generation are mutually independent by
                # construction, so this is identical to the serial comprehension above --
                # asserted bit-for-bit in fit/parallel's self-test, not assumed.
                vals = [float(x) for x in evaluator(U)]
                for _u, _val in zip(U, vals):     # keep the trace the serial path would build
                    trace['v'].append(np.asarray(_u, float).copy())
                    trace['f'].append(_val)
            es.tell(U, vals)
            used += len(U); gen += 1
            # Progress EVERY `log_every` generations. Without this a CMA run is silent until its
            # restart ends, and since the first restart gets the whole budget that can be hours.
            # A long job with no output is indistinguishable from a wedged one -- this cost real
            # time twice in this project before it was added.
            if verbose and log_every and gen % log_every == 0:
                now = time.time()
                print(f"      [restart {r} pop {int(ps)}] gen {gen:4d}  "
                      f"{used}/{maxfev} evals  best {es.result.fbest:.6f}  "
                      f"{(now - t_gen) / log_every:.2f}s/gen", flush=True)
                t_gen = now
            if gens_per_run and gen >= gens_per_run:
                break
        if es.result.fbest < best_f:
            best_f, best_v = float(es.result.fbest), np.asarray(es.result.xbest)
        runs.append(dict(restart=r, popsize=int(ps), gens=gen, fbest=float(es.result.fbest)))
        if verbose:
            print(f"    [restart {r} pop {int(ps)} sigma {sig:.3f}] {gen} gens, "
                  f"best {es.result.fbest:.6f} (overall {best_f:.6f})", flush=True)
        if mode == 'ipop':
            ps *= popsize_factor
        if used >= maxfev:
            break
    return dict(v=best_v, f=best_f, runs=runs, nev=len(trace['f']),
                seconds=time.time() - t_start,
                trace_v=np.array(trace['v']), trace_f=np.array(trace['f']),
                parts=cost['parts'](best_v))


# --------------------------------------------------------------------------- #
#  Levenberg-Marquardt with geodesic acceleration
# --------------------------------------------------------------------------- #
def levenberg_marquardt(cost, v0, bound=3.0, maxiter=100, lam0=1e-2, lam_up=10.0,
                        lam_down=3.0, accel=True, alpha=0.75, h_accel=0.1,
                        tol=1e-10, verbose=True, label='lm'):
    """LM on the residual vector, optionally with geodesic acceleration.

    WHY THIS ALGORITHM FOR THIS LANDSCAPE
        The measured geometry is the sloppy-model one: conditioning that depends on the scale
        you probe at (rank 16/16 with condition 65 at a +-26% step, rank 8/16 with condition
        2000 locally), fits that hold at the truth and fail from a distance, and the same
        failure under a completely different cost (LC-only fitting on Mirsky). That is a long,
        narrow, CURVED valley.

        Three properties matter here, and general-purpose methods have none of them:

        1. J^T J is rebuilt from scratch every step. L-BFGS accumulates curvature from gradient
           history, so the 1e25 spikes this landscape produces where a trajectory grazes the
           coexisting equilibrium's basin boundary corrupt its Hessian estimate for the rest of
           the run. Here a bad point costs one rejected step.
        2. The damping lambda makes the ~8 near-null directions harmless. An undamped Newton
           step divides by singular values of 1e-4 and flies off; LM interpolates smoothly
           toward gradient descent in exactly those directions.
        3. Geodesic acceleration adds the second-order term that lets a step follow the
           valley's CURVATURE rather than its tangent. In a narrow curved canyon the tangent
           step leaves the valley immediately and the line search collapses, which is the
           classic reason first-order methods crawl here.

    The acceleration is the directional second derivative of the residual along the proposed
    step, taken by finite difference (two extra residual evaluations, no second-order autodiff).
    It is applied only when |a| / |delta| < alpha, the standard guard: a large acceleration
    means the quadratic model is not trustworthy and the plain LM step is safer.

    Reference for the method and for why sloppy fits fail from a distance: Transtrum, Machta &
    Sethna, "Geometry of nonlinear least squares with applications to sloppy models and
    optimization" (PRE 83, 036701, 2011).
    """
    v = np.clip(np.asarray(v0, float), -bound, bound)
    n = len(v)
    lam = lam0
    r = cost['residual'](v)
    f = float(r @ r)
    hist = {'f': [f], 'v': [v.copy()], 'lam': [lam], 'accel_used': 0, 'rejected': 0}
    t0 = time.time()
    nev = 1

    for it in range(maxiter):
        J = cost['jac'](v)
        if not np.all(np.isfinite(J)):
            J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
        JtJ = J.T @ J
        Jtr = J.T @ r
        D = np.diag(np.maximum(np.diag(JtJ), 1e-12))     # Marquardt scaling
        step_taken = False
        for _try in range(12):
            try:
                delta = -np.linalg.solve(JtJ + lam * D, Jtr)
            except np.linalg.LinAlgError:
                lam *= lam_up
                continue
            if accel:
                # directional second derivative of r along delta, by central difference
                vp = np.clip(v + h_accel * delta, -bound, bound)
                vm = np.clip(v - h_accel * delta, -bound, bound)
                rp, rm = cost['residual'](vp), cost['residual'](vm)
                nev += 2
                d2 = (rp - 2.0 * r + rm) / (h_accel ** 2)
                a = -np.linalg.solve(JtJ + lam * D, J.T @ d2)
                ratio = np.linalg.norm(a) / max(np.linalg.norm(delta), 1e-300)
                if ratio < alpha:
                    delta = delta + 0.5 * a
                    hist['accel_used'] += 1
            v_new = np.clip(v + delta, -bound, bound)
            r_new = cost['residual'](v_new)
            nev += 1
            f_new = float(r_new @ r_new)
            if np.isfinite(f_new) and f_new < f:
                v, r, f = v_new, r_new, f_new
                lam = max(lam / lam_down, 1e-12)
                step_taken = True
                break
            lam *= lam_up
            hist['rejected'] += 1
        hist['f'].append(f); hist['v'].append(v.copy()); hist['lam'].append(lam)
        if not step_taken or (len(hist['f']) > 1 and
                              abs(hist['f'][-2] - f) < tol * max(f, 1e-12)):
            break

    dt = time.time() - t0
    p = cost['parts'](v)
    if verbose:
        print(f"    {label:14s} f {hist['f'][0]:.4f} -> {f:.4f} in {it + 1} its "
              f"({nev} evals, {dt:.0f}s)  c_ptc={p['c_ptc']:.4f} amp_lc={p['amp_lc']:.2f} "
              f"alive={p['alive_frac']:.2f}", flush=True)
        print(f"    {'':14s} lambda {lam:.2e}, {hist['accel_used']} accelerated steps, "
              f"{hist['rejected']} rejected", flush=True)
    return dict(v=v, f=f, f0=hist['f'][0], nit=it + 1, nev=nev, seconds=dt,
                success=True, message='lm', n_grad_clipped=0, n_grad_bad=0, n_grad_dead=0,
                grad_max=float('nan'), accel_used=hist['accel_used'],
                rejected=hist['rejected'], lam=lam,
                trace_v=np.array(hist['v']), trace_f=np.array(hist['f']), parts=p)




def basin_hopping(cost, v0, bound=3.0, n_hops=20, T=0.02, step=1.0, maxiter=60,
                  local='lm', seed=0, verbose=True, label='bh'):
    """Basin hopping: local minimization, then a stochastic jump, Metropolis accept/reject.

    WHY THIS, AND WHY ONLY NOW. Every earlier optimizer here stopped away from the truth, and I
    reported that as failure. It was not. `analysis/minimum.py` established that CMA-ES's
    stopping point is a GENUINE local minimum -- 0 of 24 random directions descend at radii
    0.01, 0.05, 0.15 and 0.40 -- and that a BARRIER of +0.0363 separates it from the truth, with
    the cost climbing monotonically over three quarters of the way there before falling. No
    local method can cross that, so the correct response is a method built to escape minima
    rather than a better local one.

    The measured barrier also sets the parameters. It is only ~6% of the range between nominal
    (0.592) and the floor (0.0004), and the truth sits |dv| = 2.53 away, so:

        T     0.02   Metropolis temperature, of order the barrier height -- large enough to
                     accept a crossing, small enough not to random-walk.
        step  1.0    jump size, a substantial fraction of the distance to a distinct basin.
                     Too small and every hop lands in the same basin.

    Requires a local minimizer at each hop; 'lm' (Levenberg-Marquardt on the residual vector)
    is the default because it rebuilds J^T J each step and so is unharmed by the gradient spikes
    this landscape produces, but 'lbfgs' and 'nm' also work.
    """
    rng = np.random.default_rng(seed)
    _local = {'lm': levenberg_marquardt, 'lbfgs': lbfgs, 'nm': nelder_mead}[local]
    v = np.clip(np.asarray(v0, float), -bound, bound)

    r = _local(cost, v, bound=bound, maxiter=maxiter, verbose=False, label=label)
    v_cur, f_cur = r['v'], r['f']
    v_best, f_best, best_r = v_cur.copy(), f_cur, r
    hops = [dict(hop=0, f=f_cur, f_best=f_best, accepted=True, v=v_cur.copy())]
    n_acc, nev = 1, r.get('nev', 0)
    t0 = time.time()
    if verbose:
        print(f"    {label}: start f={f_cur:.6f}", flush=True)

    for k in range(1, n_hops + 1):
        v_try = np.clip(v_cur + rng.normal(0, step, len(v)), -bound, bound)
        rk = _local(cost, v_try, bound=bound, maxiter=maxiter, verbose=False, label=label)
        nev += rk.get('nev', 0)
        # Metropolis: always accept downhill, accept uphill with exp(-df/T) so a barrier can be
        # crossed but the walk still concentrates on good regions.
        df = rk['f'] - f_cur
        accept = df < 0 or rng.random() < np.exp(-df / max(T, 1e-12))
        if accept:
            v_cur, f_cur, n_acc = rk['v'], rk['f'], n_acc + 1
        if rk['f'] < f_best:
            v_best, f_best, best_r = rk['v'].copy(), rk['f'], rk
        hops.append(dict(hop=k, f=rk['f'], f_best=f_best, accepted=bool(accept),
                         v=rk['v'].copy()))
        if verbose:
            mark = 'accept' if accept else '  reject'
            print(f"    {label}: hop {k:3d}  f={rk['f']:.6f}  {mark}   best={f_best:.6f}",
                  flush=True)

    dt = time.time() - t0
    p = cost['parts'](v_best)
    if verbose:
        print(f"    {label:14s} best {f_best:.6f} over {n_hops} hops "
              f"({n_acc}/{n_hops + 1} accepted, {nev} evals, {dt:.0f}s)  "
              f"c_ptc={p['c_ptc']:.4f}", flush=True)
    return dict(v=v_best, f=f_best, f0=hops[0]['f'], nit=n_hops, nev=nev, seconds=dt,
                success=True, message=f'basin-hopping, {n_acc} accepted',
                n_grad_clipped=best_r.get('n_grad_clipped', 0),
                n_grad_bad=best_r.get('n_grad_bad', 0), n_grad_dead=best_r.get('n_grad_dead', 0),
                grad_max=best_r.get('grad_max', float('nan')),
                hops_f=np.array([h['f'] for h in hops]),
                hops_v=np.array([h['v'] for h in hops]),
                hops_accepted=np.array([h['accepted'] for h in hops]),
                trace_v=np.array([h['v'] for h in hops]),
                trace_f=np.array([h['f'] for h in hops]), parts=p)




def bobyqa(cost, v0, bound=3.0, maxfev=1500, rhobeg=0.25, rhoend=1e-3, seek_global=False,
           npt=None, verbose=True, label='bobyqa'):
    """Py-BOBYQA: model-based derivative-free trust region. The purpose-built "last push".

    WHY THIS RATHER THAN THE HAND-ROLLED VERSION
        `smoothed_trust_region` implements the right principle -- average the objective over a
        ball and anneal the radius -- with the wrong step. It uses steepest descent inside the
        trust region, and on a landscape whose smoothed conditioning is ~65 that zigzags:
        MEASURED on the benchmark, two rejected steps for every accepted one and f = 0.348 after
        859 evaluations, against CMA's 0.0198 in 1512.

        BOBYQA does what that was reaching for. It maintains an interpolation set spread over
        the trust radius and fits a QUADRATIC model to it, so it gets curvature (which fixes the
        zigzag) and the model is a least-squares fit over a region of radius rho (which is the
        smoothing). Then rho contracts. Same principle, correct step.

    rhobeg = 0.25 is the initial trust radius, chosen from the same measurement that motivates
    all the smoothing here: the map is rank 16/16 with condition 65 when probed at radius ~0.23,
    against rank 8/16 and condition 2000 at 1e-3. Starting inside the well-conditioned regime
    and contracting is the whole idea.

    `seek_global=True` enables Py-BOBYQA's own multiple-restart heuristic for noisy/rugged
    objectives -- appropriate here, since the ruggedness is exactly what it is designed for.
    """
    import pybobyqa
    f, _g, trace = _harden(cost)
    v = np.clip(np.asarray(v0, float), -bound, bound)
    n = len(v)
    lo = np.full(n, -bound)
    hi = np.full(n, bound)
    t0 = time.time()
    soln = pybobyqa.solve(f, v, bounds=(lo, hi), maxfun=int(maxfev), rhobeg=rhobeg,
                          rhoend=rhoend, npt=npt, seek_global_minimum=bool(seek_global),
                          objfun_has_noise=True, print_progress=False)
    dt = time.time() - t0
    v_best = np.clip(np.asarray(soln.x, float), -bound, bound)
    p = cost['parts'](v_best)
    if verbose:
        print(f"    {label:14s} f {trace['f'][0]:.6f} -> {float(soln.f):.6f} "
              f"({len(trace['f'])} evals, {dt:.0f}s)  c_ptc={p['c_ptc']:.4f}  "
              f"flag={soln.flag} ({soln.msg})", flush=True)
    return dict(v=v_best, f=float(soln.f), f0=float(trace['f'][0]),
                nit=int(getattr(soln, 'nruns', 1)), nev=len(trace['f']), seconds=dt,
                success=soln.flag == soln.EXIT_SUCCESS, message=str(soln.msg),
                n_grad_clipped=0, n_grad_bad=0, n_grad_dead=0, grad_max=float('nan'),
                trace_v=np.array(trace['v']), trace_f=np.array(trace['f']), parts=p)
