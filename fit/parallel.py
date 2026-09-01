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


def cost_spec(model_name, target, doses, n_phase, mode='instant', backend='diffrax', dt=0.02,
              w_osc=0.2, w_amp=1.0, target_k=None, target_psi=None, section=None,
              readout=None, pulse=8.0, skip_p=None, window_mode='absolute',
              span=None, w_anchor=1.0, s_crit_base=None, amp_ramp=None, basis=None,
              n_dose=None):
    """A picklable description of a cost, sufficient to rebuild it in a worker.

    Everything here is a primitive or an array. `section`/`readout` are carried explicitly
    rather than left to the model default, so a worker cannot silently build a cost against a
    different phase observable than the parent -- the failure mode that made RAD03 re-plot as
    "NOT AN ORBIT" (REPO_MAP hazard 14).
    """
    return dict(model_name=str(model_name), target=str(target),
                doses=np.asarray(doses, float), n_phase=int(n_phase), mode=str(mode),
                backend=str(backend), dt=float(dt), w_osc=float(w_osc), w_amp=float(w_amp),
                target_k=None if target_k is None else float(target_k),
                target_psi=None if target_psi is None else float(target_psi),
                section=section, readout=readout, pulse=float(pulse),
                skip_p=None if skip_p is None else int(skip_p),
                # RELATIVE-WINDOW FIELDS. `basis` is carried EXPLICITLY and is not optional:
                # the gauge quotient is an SVD of a projector whose nonzero singular values are
                # all 1, so a worker re-deriving it gets a different valid basis and the same
                # `v` then means different parameters -- measured at 3.0 DECADES (hazard 18).
                # A worker that rebuilt its own basis would score a different model than the
                # parent asked about, silently.
                window_mode=str(window_mode),
                span=None if span is None else (float(span[0]), float(span[1])),
                w_anchor=float(w_anchor),
                s_crit_base=None if s_crit_base is None else float(s_crit_base),
                amp_ramp=None if amp_ramp is None else (float(amp_ramp[0]), float(amp_ramp[1])),
                basis=None if basis is None else np.asarray(basis, float),
                n_dose=None if n_dose is None else int(n_dose))


def build_cost(spec):
    """Rebuild the cost described by `spec`. Used by workers AND by the serial fallback, so
    there is exactly one construction path and no chance of the two diverging."""
    from models import get_model
    from fit.cost import make_cost, RadialTarget

    model = get_model(spec['model_name'])
    if spec.get('section'):
        model.reference_variable = spec['section']
    if spec.get('readout'):
        model.readout_variable = spec['readout']
    if spec.get('window_mode') == 'relative':
        from fit.relcost import make_relative_cost
        if spec.get('basis') is None:
            raise ValueError("a relative-window spec must carry `basis`: a worker that "
                             "re-derives the gauge quotient gets a different valid basis and "
                             "the same v then means different parameters (hazard 18).")
        return make_relative_cost(
            model, spec['target'], n_phase=spec['n_phase'], n_dose=spec.get('n_dose') or 14,
            span=spec.get('span') or (1.2, 8.0), s_crit_base=spec.get('s_crit_base'),
            mode=spec['mode'], backend=spec['backend'], dt=spec['dt'],
            pulse=spec.get('pulse', 8.0), skip_p=spec.get('skip_p'),
            readout_ref=spec.get('readout'), amp_ramp=spec.get('amp_ramp'),
            w_osc=spec['w_osc'], w_amp=spec['w_amp'], w_anchor=spec.get('w_anchor', 1.0),
            basis=spec['basis'])
    tgt = (RadialTarget(k=spec['target_k'], psi=spec['target_psi'])
           if spec['target_k'] is not None else RadialTarget())
    return make_cost(model, spec['target'], spec['doses'], tgt, n_phase=spec['n_phase'],
                     mode=spec['mode'], backend=spec['backend'], dt=spec['dt'],
                     w_osc=spec['w_osc'], w_amp=spec['w_amp'],
                     pulse=spec.get('pulse', 8.0), skip_p=spec.get('skip_p'),
                     readout_ref=spec.get('readout'))


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

    def __init__(self, spec, n_workers, verbose=True):
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
