"""
fit/parallel.py
===============
Evaluate a CMA population across processes.

WHY THE POPULATION AND NOT THE CELLS
    A cost evaluation is one orbit solve plus n_phase * n_dose independent perturbation
    integrations, and BOTH are parallel axes. Cells are the finer one and the only one that can
    shorten a single evaluation, but the population is the one that is free: each member is
    already an ordinary call to `cost['total']` on its own parameter vector, so nothing about
    the numerics changes and the result is identical to the serial loop by construction.

    MEASURED, per-evaluation wall time on the 20 x 14 grid (40 samples at CMA's own sigma=0.5):

        p50 0.76 s   p75 3.67 s   p90 11.75 s   max 13.90 s   mean 3.38 s

    The distribution is heavy-tailed -- max/median is 18x -- and a CMA generation is a
    SYNCHRONOUS BARRIER, so a generation costs max(t_i), not mean(t_i). Two consequences that
    shaped this module:

      * OVERSUBSCRIBE. With popsize > n_workers the fast members (median under a second) fill
        the gaps while one straggler runs, so the generation costs about max(t_max,
        popsize * t_mean / n_workers) instead of ceil(popsize / n_workers) * t_max. At 16
        workers, popsize 64 costs almost exactly what popsize 16 costs.
      * A generation can never go below t_max ~ 14 s however many workers there are. Cell-level
        parallelism is the only thing that lowers that floor; it is deliberately NOT built here
        (see PROJECT_SUMMARY 5.8) because it also forfeits autodiff, and the data matters more
        than the last factor of two.

WHAT THIS ACTUALLY BUYS, MEASURED -- AND WHY THE TEST MACHINE UNDERSTATES IT
    Verified first: parallel evaluation is BIT-IDENTICAL to the serial loop at 4 and 8 workers.
    That is the property worth having, and it is why this route was chosen over vmap batching,
    whose values came back 1e-3 apart and were never explained.

    The SPEEDUP on the development laptop is poor, and the reason is the hardware:

        33 members, 20 x 14 grid
        1 worker : wall 153.7s   per-eval as seen by the worker  4.66s   sum  153.7s
        8 workers: wall 126.0s   per-eval as seen by the worker 26.58s   sum  877.1s

    Each evaluation runs 5.7x SLOWER when eight run at once -- the same work costs 877 core-
    seconds instead of 154. Workers time themselves, so this is not queueing and not the
    straggler (the slowest member is 10% of the total). Four other explanations were measured
    and rejected: XLA's Eigen pool (41.5s vs 42.1s with it disabled -- no effect), XLA silently
    parallelising one evaluation (it does not), memory pressure (31.5 GB, 12.4 free), and pool
    overhead (the queue bound at 8 workers is 19.3s).

    What is left is the CPU: an Intel Core Ultra 7 268V, a HYBRID part whose eight "cores" are
    four P-cores plus four low-power E-cores sharing 12 MB of L3 at a 2.2 GHz base. At eight
    workers half the population lands on E-cores while all-core clocks fall away from
    single-core boost.

    SO DO NOT SIZE A CLUSTER JOB FROM THIS NUMBER. A homogeneous node with real cores, more L3
    and sustained clocks should scale far better; measure it there with the same test before
    committing a long run, rather than assuming either this figure or a linear one.

WHY A COST SPEC AND NOT THE COST OBJECT
    `make_cost` returns jitted closures. Those are not picklable, so the cost cannot be shipped
    to a worker. Instead each worker receives a small dict of PRIMITIVES and rebuilds the cost
    itself, once, in an initializer. That costs one JAX compile per worker (tens of seconds,
    paid once for a persistent pool) and buys exact reproduction of the serial code path.

WINDOWS: SPAWN, ONE THREAD PER WORKER, AND EVERY CALLER NEEDS `if __name__ == '__main__':`
    Processes are spawned, not forked, so the initializer runs in a fresh interpreter and every
    worker imports JAX independently. Each worker is pinned to ONE thread: MEASURED, a single
    evaluation is FASTER at one thread than at four (1.298 s vs 1.552 s), because each cell is
    a sequential adaptive integration that XLA's CPU backend does not parallelise, so letting
    workers grab threads only makes them contend.

    Spawn also re-imports the caller's main module inside each child. A driver that builds a pool at
    import scope therefore has each child build its own pool, recursively, and multiprocessing
    stops it with

        RuntimeError: An attempt has been made to start a new process before the current
        process has finished its bootstrapping phase.

    `fit/radial.py` is safe because it already guards `main()`. Any NEW entry point that uses
    PoolEvaluator must do the same -- this is not optional on Windows, and it fails at pool
    construction rather than anywhere near the code that looks responsible.
"""
import multiprocessing as mp
import os
import time

import numpy as np

#: rebuilt once per worker process by `_init_worker`
_W = {}



# --------------------------------------------------------------------------- #
#  What a worker must be told, and the audit that keeps that list honest
# --------------------------------------------------------------------------- #
#
# ONE LIST PER FACTORY, USED BOTH TO CARRY AND TO FORWARD.
#
# The August-31 arm campaign lost three of its five arms here. `cost_spec` enumerated its
# fields by hand and `build_cost` re-enumerated them by hand into the `make_cost` call, so the
# two lists could drift -- and they did: `amp_ramp` was carried but never forwarded, and
# `row_weight` / `w_brack` were neither. `make_cost`'s own defaults then filled the gap
# (`amp_ramp=(0.05, 0.20)`, i.e. the ramp ON), so every worker in arms 0-3 minimised ONE
# function while four different tags claimed otherwise. The traces are bit-identical over all
# 8000 population evaluations; only the parent's single evaluation of the start point differs.
# PROJECT_SUMMARY 5.15.
#
# So the keywords are named ONCE, the spec is a dict keyed by exactly those names, and
# `build_cost` splats it. Carrying and forwarding are now the same act and cannot disagree.
_ABS_COST_KW = ('n_phase', 'mode', 'backend', 'dt', 'skip_p', 'w_osc', 'w_amp', 'pulse',
                'readout_ref', 'w_stab', 'r_max', 'basis', 'amp_ramp', 'row_weight',
                'row_weight_floor', 'w_brack', 'brack_lo', 'brack_hi')

_REL_COST_KW = ('n_phase', 'n_dose', 'n_probe', 'span', 'probe', 's_crit_base', 'mode',
                'backend', 'dt', 'pulse', 'skip_p', 'readout_ref', 'basis', 'amp_ramp',
                'w_osc', 'w_amp', 'profile_psi', 'w_anchor', 'disp_floor')

#: Factory parameters DELIBERATELY not shipped, each with the reason. Anything a factory grows
#: that is in neither this map nor the lists above stops `cost_spec` -- see `_audit_cost_kw`.
_NOT_SHIPPED = {
    'model': 'rebuilt from `model_name`',
    'target_state': 'shipped as `target`',
    'doses': 'shipped as `doses` (relative mode builds its own)',
    'tgt': 'shipped as `target_k` / `target_psi`',
    'param_names': 'always None; the quotient basis is shipped instead, as `basis`',
    'amp_frac': 'not exposed by RunConfig -- add it to _ABS_COST_KW if that changes',
    'm_amp': 'not exposed by RunConfig -- add it to _ABS_COST_KW if that changes',
    'eps': 'not exposed by RunConfig -- add it to _ABS_COST_KW if that changes',
    'grad_mode': 'gradient construction only; the pool evaluates values, never grad',
    'ridge': 'gradient construction only; the pool evaluates values, never grad',
}


def _audit_cost_kw():
    """FAIL IF A COST FACTORY HAS GROWN AN OPTION NOTHING SHIPS.

    This is the check that makes the fix permanent rather than a one-off repair. Adding a term
    to `make_cost` and wiring it to a RunConfig field is a two-file change today; without this
    the third file -- the one that ships it to the workers -- is easy to forget, and forgetting
    it is SILENT, because `make_cost` supplies a default and the pool happily minimises it.

    Called from `cost_spec`, so it runs once per campaign, in the parent, before any worker
    starts and before any cluster time is spent.
    """
    import inspect
    from fit.cost import make_cost
    from fit.relcost import make_relative_cost
    for fn, shipped in ((make_cost, _ABS_COST_KW), (make_relative_cost, _REL_COST_KW)):
        unknown = [p for p in inspect.signature(fn).parameters
                   if p not in shipped and p not in _NOT_SHIPPED]
        if unknown:
            raise RuntimeError(
                f"{fn.__module__}.{fn.__name__} has parameter(s) {unknown} that "
                f"fit.parallel neither ships to workers nor declares unshippable. A worker "
                f"would silently build the cost with the DEFAULT value while the parent used "
                f"the configured one -- the failure that cost the Aug-31 arm campaign three of "
                f"its five arms. Add each name to the appropriate _*_COST_KW tuple (and to the "
                f"RunConfig plumbing), or to _NOT_SHIPPED with the reason it cannot matter.")


def cost_spec(model_name, target, doses, n_phase, mode='instant', backend='diffrax', dt=0.02,
              w_osc=0.2, w_amp=1.0, target_k=None, target_psi=None, section=None,
              readout=None, pulse=8.0, skip_p=None, window_mode='absolute',
              span=None, w_anchor=1.0, s_crit_base=None, amp_ramp=None, basis=None,
              n_dose=None, row_weight=False, row_weight_floor=0.1,
              w_brack=0.0, brack_lo=0.25, brack_hi=0.65, w_stab=0.0, r_max=0.98,
              n_probe=None, probe=None, profile_psi=None, disp_floor=None):
    """A picklable description of a cost, sufficient to rebuild it in a worker.

    Everything here is a primitive or an array. `section`/`readout` are carried explicitly
    rather than left to the model default, so a worker cannot silently build a cost against a
    different phase observable than the parent -- the failure mode that made RAD03 re-plot as
    "NOT AN ORBIT" (REPO_MAP hazard 14).

    THE OBJECTIVE OPTIONS GO INTO ONE `cost_kw` DICT, keyed by the factory's own parameter
    names, and `build_cost` splats it. That is deliberate and it is the fix for the Aug-31 arm
    campaign: a hand-written forwarding list can omit an option that the carrying list includes,
    and the omission is invisible because the factory has a default.

    `basis` IS REQUIRED, in both window modes. The gauge quotient is an SVD of a projector whose
    nonzero singular values are all 1, so a worker re-deriving it gets a DIFFERENT valid basis
    and the same `v` then denotes different parameters -- measured at 3.0 decades (hazard 18).
    Absolute mode used to let it default to None and re-derive; parent and worker happened to
    agree because they ran the same LAPACK on the same host, which is an accident of deployment
    and not a property of the code.
    """
    _audit_cost_kw()
    if basis is None:
        raise ValueError(
            "cost_spec requires `basis`: a worker that re-derives the gauge quotient gets a "
            "different valid basis and the same `v` then means different parameters "
            "(hazard 18, measured at 3.0 decades). Pass the parent cost's C['B'].")

    if str(window_mode) == 'relative':
        # OPTIONS THE RELATIVE COST CANNOT HONOUR ARE AN ERROR, NOT A SILENT DROP. Dropping
        # them is precisely the bug this module is being fixed for.
        unusable = {k: v for k, v in (('row_weight', bool(row_weight)),
                                      ('w_brack', float(w_brack)),
                                      ('w_stab', float(w_stab))) if v}
        if unusable:
            raise ValueError(
                f"window_mode='relative' cannot honour {sorted(unusable)}: "
                f"fit.relcost.make_relative_cost has no such term. Set them to their off "
                f"values, or use window_mode='absolute'.")
        kw = dict(n_phase=int(n_phase), n_dose=int(n_dose or 14),
                  span=None if span is None else (float(span[0]), float(span[1])),
                  s_crit_base=None if s_crit_base is None else float(s_crit_base),
                  mode=str(mode), backend=str(backend), dt=float(dt), pulse=float(pulse),
                  skip_p=None if skip_p is None else int(skip_p), readout_ref=readout,
                  basis=np.asarray(basis, float),
                  amp_ramp=None if amp_ramp is None else (float(amp_ramp[0]),
                                                          float(amp_ramp[1])),
                  w_osc=float(w_osc), w_amp=float(w_amp), w_anchor=float(w_anchor))
        for k, v in (('n_probe', n_probe), ('probe', probe), ('profile_psi', profile_psi),
                     ('disp_floor', disp_floor)):
            if v is not None:                      # otherwise let the factory's default stand
                kw[k] = v
        allowed = _REL_COST_KW
    else:
        kw = dict(n_phase=int(n_phase), mode=str(mode), backend=str(backend), dt=float(dt),
                  skip_p=None if skip_p is None else int(skip_p),
                  w_osc=float(w_osc), w_amp=float(w_amp), pulse=float(pulse),
                  readout_ref=readout, w_stab=float(w_stab), r_max=float(r_max),
                  basis=np.asarray(basis, float),
                  amp_ramp=None if amp_ramp is None else (float(amp_ramp[0]),
                                                          float(amp_ramp[1])),
                  row_weight=bool(row_weight), row_weight_floor=float(row_weight_floor),
                  w_brack=float(w_brack), brack_lo=float(brack_lo), brack_hi=float(brack_hi))
        allowed = _ABS_COST_KW

    bad = [k for k in kw if k not in allowed]
    if bad:
        raise RuntimeError(f"cost_spec built key(s) {bad} that window_mode={window_mode!r} "
                           f"does not ship; the two lists have drifted apart again.")
    return dict(model_name=str(model_name), target=str(target),
                doses=np.asarray(doses, float), section=section,
                target_k=None if target_k is None else float(target_k),
                target_psi=None if target_psi is None else float(target_psi),
                window_mode=str(window_mode), cost_kw=kw)


def build_cost(spec):
    """Rebuild the cost described by `spec`. Used by workers AND by the serial fallback, so
    there is exactly one construction path and no chance of the two diverging.

    The factory call is `f(model, target, ..., **spec['cost_kw'])` -- a splat, never a
    hand-written argument list. A splat cannot forget a key.
    """
    from models import get_model

    model = get_model(spec['model_name'])
    if spec.get('section'):
        model.reference_variable = spec['section']
    kw = dict(spec['cost_kw'])
    if kw.get('readout_ref'):
        model.readout_variable = kw['readout_ref']
    if kw.get('basis') is None:
        raise ValueError("a cost spec must carry `basis` (hazard 18); see cost_spec.")

    if spec.get('window_mode') == 'relative':
        from fit.relcost import make_relative_cost
        return make_relative_cost(model, spec['target'], **kw)

    from fit.cost import make_cost, RadialTarget
    tgt = (RadialTarget(k=spec['target_k'], psi=spec['target_psi'])
           if spec['target_k'] is not None else RadialTarget())
    return make_cost(model, spec['target'], spec['doses'], tgt, **kw)


def probe_points(v0, n=4, radius=0.35, seed=20260901):
    """Deterministic points for the parent/worker parity check: the start, plus displacements.

    A FIXED SEED, because a parity failure has to be reproducible and a parity pass must not be
    a lucky draw. The radius is large enough that the terms which are zero at a healthy base
    point -- the bracketing barrier, the aliveness ramp -- are actually exercised; a check made
    only at v0 would have PASSED for every batch-1 arm from the base start, where all of them
    agree at 0.428466.
    """
    v0 = np.asarray(v0, float)
    rng = np.random.default_rng(int(seed))
    return [v0.copy()] + [v0 + radius * rng.normal(size=v0.shape) for _ in range(max(n - 1, 0))]


def _init_worker(spec, barrier=None):
    # one thread per worker: measured faster than four, and prevents oversubscription when
    # n_workers threads each try to grab the machine
    # ONE THREAD PER WORKER, AND OMP/MKL ALONE DO NOT ACHIEVE THAT.
    #
    # JAX's CPU backend runs its own Eigen thread pool, sized to the machine's cores and
    # untouched by OMP_NUM_THREADS. Capping only OMP left every worker spinning up a full-width
    # pool, so n workers each tried to use all 8 cores and spent their time contending: the
    # measured speedup barely improved from 4 workers to 8 (2.12x -> 2.55x), the signature of a
    # shared bottleneck rather than a straggler. XLA_FLAGS must be set BEFORE jax is imported,
    # which is why this sits at the top of the worker initializer and not in the module body.
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '')
                               + ' --xla_cpu_multi_thread_eigen=false'
                               + ' --xla_force_host_platform_device_count=1').strip()
    os.environ.setdefault('JAX_ENABLE_X64', '1')
    from fit.search import _harden
    cost = build_cost(spec)
    f, _g, _t = _harden(cost)
    _W['f'] = f
    f(np.zeros(cost['n_free']))            # compile here, not inside the first generation
    if barrier is not None:
        # A REAL BARRIER, because a warm-up TASK does not work. Submitting n no-op tasks and
        # counting distinct pids reported "1 distinct processes compiled": the tasks are so
        # cheap that one worker drained the queue before its siblings had finished starting,
        # so the others still compiled inside the first timed generation. A barrier cannot be
        # short-circuited that way -- nobody passes until everybody arrives.
        barrier.wait(timeout=900)


def _eval(v):
    return _W['f'](np.asarray(v, float))


def _ready(_i):
    """No-op whose only purpose is to force a worker through `_init_worker`."""
    return os.getpid()


class PoolEvaluator:
    """`ev(U) -> [f(u) for u in U]`, evaluated across processes.

    Deliberately the same signature as the serial list comprehension it replaces, so the CMA
    loop does not need to know which one it has.
    """

    def __init__(self, spec, n_workers, verbose=True, verify=None, verify_at=None,
                 verify_tol=1e-9):
        """`verify` is the PARENT's own scalar objective, and passing it is strongly advised.

        THE POOL'S CONTRACT IS THAT IT COMPUTES THE SAME FUNCTION THE PARENT IS OPTIMISING, and
        until August 31 nothing checked it. `cost_spec`/`build_cost` dropped three arm options,
        so four campaigns minimised one objective under four names and the only evidence was
        that their traces came back identical -- discovered afterwards, from the saved npz.

        A structural fix (one keyword list, splatted) closes that particular hole; this closes
        the CLASS. Any future divergence between what the parent asks for and what a worker
        builds -- a stale module on a compute node, a model default set in one process and not
        the other, a basis re-derived by different LAPACK -- shows up here, in seconds, before
        the campaign spends a node-hour.

        `verify_tol` is not zero because these are floating-point reductions in two processes
        with different XLA thread settings; it is 1e-9 because a genuine objective mismatch is
        never that small. The measured disagreement is PRINTED either way, so bit-identity is
        visible in the log rather than merely hoped for.
        """
        self.spec, self.n = spec, int(n_workers)
        self.pool = None
        if self.n > 1:
            ctx = mp.get_context('spawn')
            barrier = ctx.Barrier(self.n + 1)          # +1: the parent waits too
            self.pool = ctx.Pool(self.n, initializer=_init_worker,
                                 initargs=(spec, barrier))
            # BLOCK UNTIL THE WORKERS HAVE ACTUALLY COMPILED.
            #
            # `Pool(initializer=...)` returns immediately and initializes workers LAZILY, so
            # without this the per-worker JAX compile (tens of seconds) is paid inside the
            # first generation. It measured as a terrible speedup -- 1.32x on 4 workers -- and
            # in a real run it would look like a mysteriously slow first generation. Forcing
            # it here makes the cost visible, reported, and paid once.
            t0 = time.time()
            try:
                barrier.wait(timeout=900)              # returns only when all n have compiled
            except Exception as exc:
                self.pool.terminate()
                raise RuntimeError(
                    f"only some of {self.n} workers finished building the cost "
                    f"({type(exc).__name__}). A worker died during initialisation -- run with "
                    f"--workers 1 to see its traceback.") from exc
            if verbose:
                print(f"      [parallel] {self.n} workers ready in {time.time() - t0:.0f}s "
                      f"(all compiled before the first generation)", flush=True)
        else:
            from fit.search import _harden
            self._f, _g, _t = _harden(build_cost(spec))
        if verify is not None:
            self.check_parity(verify, verify_at, verify_tol, verbose)

    def check_parity(self, parent_f, points=None, tol=1e-9, verbose=True):
        """Does this pool compute the parent's function? Raises if not.

        Refuses to run on a single point by default: at the base start every batch-1 arm agreed
        at 0.428466, because the terms that distinguished them are exactly zero on a healthy
        surface. `probe_points` displaces off it for that reason.
        """
        if points is None:
            raise ValueError("check_parity needs probe points -- see fit.parallel.probe_points")
        pts = [np.asarray(p, float) for p in points]
        mine = np.asarray(self(pts), float)
        theirs = np.asarray([float(parent_f(p)) for p in pts], float)
        # A sentinel on BOTH sides is agreement, not a 0.0 difference to be trusted: it means
        # the point is unevaluable for both, which says nothing about whether they agree.
        finite = np.isfinite(mine) & np.isfinite(theirs) & (mine < 1e5) & (theirs < 1e5)
        d = np.abs(mine - theirs)
        worst = float(np.max(d)) if len(d) else 0.0
        if verbose:
            print(f"      [parallel] parent/worker parity over {len(pts)} probe points: "
                  f"max |delta| = {worst:.3e}"
                  f"{'  (bit-identical)' if worst == 0.0 else ''}"
                  f"{'' if finite.all() else f'  [{int((~finite).sum())} sentinel]'}",
                  flush=True)
        if worst > tol:
            i = int(np.argmax(d))
            raise RuntimeError(
                f"THE POOL IS NOT MINIMISING THE PARENT'S OBJECTIVE. At probe point {i} the "
                f"parent scores {theirs[i]:.10g} and a worker scores {mine[i]:.10g} "
                f"(|delta| {d[i]:.3e} > tol {tol:g}). The run would optimise one function "
                f"while every label, config and figure named another -- see PROJECT_SUMMARY "
                f"5.15. Check that every objective option is in fit.parallel's _ABS_COST_KW / "
                f"_REL_COST_KW and that the workers are running this same source tree.")
        return worst

    def __call__(self, U):
        if self.pool is None:
            return [self._f(np.asarray(u, float)) for u in U]
        # chunksize=1 is REQUIRED, not a default. Evaluation time spans 0.76 s to 13.9 s, so
        # any static chunking hands one worker several slow members while others idle; with
        # chunksize=1 the pool acts as a work queue and the fast members fill the gaps.
        return list(self.pool.map(_eval, [np.asarray(u, float) for u in U], chunksize=1))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def recommend_popsize(n_free, n_workers):
    """Population large enough to keep `n_workers` busy through a straggler.

    A generation costs about max(t_max, popsize * t_mean / n_workers). Setting popsize so the
    second term reaches the first -- popsize ~ n_workers * t_max / t_mean ~ 4.1 * n_workers --
    fills the machine at no extra wall-clock. Below CMA's own default there is nothing to gain,
    so the default is the floor.
    """
    default = int(4 + 3 * np.log(int(n_free)))
    return max(default, int(round(4.1 * int(n_workers))))


# --------------------------------------------------------------------------- #
#  The regression test for the failure this module has already had
# --------------------------------------------------------------------------- #
def selftest(model_name='almeida', target='BMAL1', verbose=True):
    """Does a spec round-trip carry a NON-DEFAULT objective?

    EVERY OPTION IS SET AWAY FROM ITS DEFAULT, and that is the whole test. The Aug-31 arm
    campaign was lost to a `build_cost` that forwarded none of the arm options while
    `make_cost`'s own defaults filled the gap, so a round-trip built from DEFAULTS would have
    agreed perfectly and passed. Only a spec that asks for something unusual can tell the two
    apart.

    The reference is `make_cost` called directly -- the parent's construction path -- and the
    comparison is at displaced points, because the ramp and the bracketing barrier are exactly
    zero on a healthy surface and a check at the base point alone cannot see them.
    """
    import numpy as _np
    from models import get_model
    from fit.cost import make_cost, RadialTarget
    from fit.doses import fit_dose_grid

    m = get_model(model_name)
    doses, s_crit = fit_dose_grid(model_name, target, 'instant', 8.0, 6, lo_factor=0.5)
    tgt = RadialTarget.from_singularity(s_crit, 0.25)
    # deliberately unusual, and every one of them a term that reaches the surface
    opts = dict(n_phase=8, mode='instant', backend='diffrax', dt=0.02, w_osc=0.31, w_amp=1.7,
                pulse=8.0, skip_p=None, w_stab=0.0, r_max=0.98,
                amp_ramp=(0.11, 0.33), row_weight=True, row_weight_floor=0.23,
                w_brack=0.7, brack_lo=0.3, brack_hi=0.6)
    parent = make_cost(m, target, doses, tgt, **opts)
    spec = cost_spec(model_name, target, doses, opts['n_phase'], mode=opts['mode'],
                     backend=opts['backend'], dt=opts['dt'], w_osc=opts['w_osc'],
                     w_amp=opts['w_amp'], target_k=tgt.k, target_psi=tgt.psi,
                     section=m.reference_variable, readout=getattr(m, 'readout_variable', None),
                     pulse=opts['pulse'], skip_p=opts['skip_p'], amp_ramp=opts['amp_ramp'],
                     row_weight=opts['row_weight'], row_weight_floor=opts['row_weight_floor'],
                     w_brack=opts['w_brack'], brack_lo=opts['brack_lo'],
                     brack_hi=opts['brack_hi'], w_stab=opts['w_stab'], r_max=opts['r_max'],
                     basis=parent['B'])
    worker = build_cost(spec)

    pts = probe_points(parent['v0'], n=4, radius=0.35)
    a = _np.array([float(parent['total'](p)) for p in pts])
    b = _np.array([float(worker['total'](p)) for p in pts])
    worst = float(_np.max(_np.abs(a - b)))
    ok = worst == 0.0
    if verbose:
        print(f"  round-trip of a NON-DEFAULT objective, {len(pts)} points: "
              f"max |delta| = {worst:.3e}   {'PASS' if ok else 'FAIL'}")
        for p, x, y in zip(range(len(pts)), a, b):
            print(f"    point {p}   parent {x:.10f}   worker {y:.10f}")

    # AND THE NEGATIVE CONTROL. If the round-trip silently dropped the options, the worker
    # would equal a DEFAULT-built cost instead -- which is exactly what happened in batch 1.
    # Requiring the two to DIFFER stops a future regression passing by reproducing the bug on
    # both sides.
    naive = make_cost(m, target, doses, tgt, n_phase=opts['n_phase'], mode=opts['mode'],
                      backend=opts['backend'], dt=opts['dt'], basis=parent['B'])
    c = _np.array([float(naive['total'](p)) for p in pts])
    sep = float(_np.max(_np.abs(a - c)))
    ok2 = sep > 1e-6
    if verbose:
        print(f"  negative control -- the SAME cost built from defaults must NOT match: "
              f"max |delta| = {sep:.3e}   {'PASS' if ok2 else 'FAIL'}")

    # the audit itself, and the two refusals
    ok3 = True
    try:
        _audit_cost_kw()
    except RuntimeError as exc:
        ok3 = False
        print(f"  keyword audit FAIL: {exc}")
    if verbose and ok3:
        print("  keyword audit: every make_cost / make_relative_cost option is shipped or "
              "declared unshippable   PASS")

    ok4 = True
    try:
        cost_spec(model_name, target, doses, 8, basis=None)
        ok4 = False
    except ValueError:
        pass
    try:
        cost_spec(model_name, target, doses, 8, basis=parent['B'],
                  window_mode='relative', w_brack=1.0)
        ok4 = False
    except ValueError:
        pass
    if verbose:
        print(f"  refuses a spec with no basis, and refuses an absolute-only term in relative "
              f"mode   {'PASS' if ok4 else 'FAIL'}")

    print("  selftest", "PASS" if (ok and ok2 and ok3 and ok4) else "FAIL")
    return ok and ok2 and ok3 and ok4


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="fit.parallel checks")
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest(a.model, a.target) else 1
    ap.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
