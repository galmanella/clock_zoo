"""
fit/viability.py
================
Find parameter sets where a HEALTHY CIRCADIAN CLOCK exists, by rejection sampling, and use them
as CMA starting points.

WHY: dispersed starts, done properly
    A multi-seed sweep only probes multimodality if the seeds start somewhere different.
    Isotropic dispersion cannot deliver that here -- MEASURED, viability falls off so fast
    (alive fraction 0.67 / 0.50 / 0.17 / 0.00 at radius 0.25 / 0.5 / 1.0 / 1.5) that any radius
    big enough to separate seeds is mostly dead, and at a viable radius the dispersion is
    NARROWER per coordinate than CMA's own sigma0.

    Rejection sampling inverts the problem: draw from the WHOLE box and keep only what is
    viable. The accepted points are as far apart as the viable set allows, by construction, and
    the rejected draws are free viability data.

ACCEPTANCE IS THREE-PART, AND ALL THREE WERE FORCED BY A MEASUREMENT
    1. A LIMIT CYCLE EXISTS (`ok_orbit`, INCLUDING the amplitude gate). REPO_MAP hazard 2: an
       equilibrium satisfies y(T) = y(0) for ANY T and so converges the BVP perfectly. Without
       the amplitude gate, 26% of random draws were accepted as "viable" while having a
       relative amplitude of exactly 0.000 -- they were fixed points. With it, ~0.9%.
    2. THE PERIOD IS CIRCADIAN. Of 12 gated hits, periods ran 6.3 h to 25.9 h and only 4 were
       within 0.7-1.4 x 24.83 h. Seeding a fit at a 6 h clock produces another RAD05: a genuine
       oscillator that is not the object of study.
    3. NO SPECIES HAS COLLAPSED. All 4 of those circadian hits had a species at 1e-6 or below
       (relative amplitudes 12-16.5, the baseline-collapse signature met three times in this
       project). Base's tightest species sits at 1.4% of its own mean, so `min_ratio=1e-3`
       admits base comfortably while rejecting a collapsed corner.

    ZERO of 12 far viable points passed all three. That is itself a result: oscillators exist
    throughout the box at ~1%, but HEALTHY CIRCADIAN ones appear to be localized near base.

WHY NO CHEAP FORWARD-INTEGRATION PRE-FILTER
    Rejecting most draws with a short forward integration looks attractive. It is not reliable:
    a slow spiral into a fixed point and a genuine oscillation are indistinguishable over any
    fixed window, and the window needed depends on the unknown period AND the unknown
    convergence rate. The gate would be either wrong or not actually cheap. Speed comes from
    parallelism instead.

PARALLELISM, AND WHY IT IS THE EASY CASE
    This is the only workload in the pipeline with NO synchronisation: each worker draws, tests
    and reports, and never waits for a sibling. Unlike a CMA generation -- a barrier, where 8
    workers bought only ~1.4x -- this should scale close to linearly.

    Parallelised ACROSS SEEDS, not by pooling hits. Seed s draws from its own RNG stream, so
    its start is the same point whether it ran on worker 0 or worker 30, and whether or not the
    other seeds ever ran. Pooling hits from a shared queue would make the start depend on
    scheduling order and the run would stop being reproducible.
"""
import multiprocessing as mp
import os
import time

import numpy as np

_W = {}


def make_probe(model_name, section=None, readout=None, period_lo=0.7, period_hi=1.4,
               min_ratio=1e-3, amp_floor=1e-3):
    """`probe(v) -> dict | None` -- the three-part acceptance test on one parameter set.

    Returns the orbit's properties when accepted, and None otherwise, so a caller can record
    what it found without re-solving.
    """
    import jax
    import jax.numpy as jnp
    from models import get_model
    from engine.orbit import OrbitSolver, make_guess_fn
    from fit.cost import quotient_basis

    model = get_model(model_name)
    if section:
        model.reference_variable = section
    if readout:
        model.readout_variable = readout
    names, zb, B, _g = quotient_basis(model)
    solver, guess = OrbitSolver(model), make_guess_fn(model)
    y_seed = jnp.asarray(model.get_initial_state(), jnp.float64)
    solve = jax.jit(solver.solve)
    T_nom = float(getattr(model, 'approx_period', None) or 24.0)
    n_free = B.shape[1]

    def probe(v):
        v = np.asarray(v, float)
        P = model.jax_apply(np.exp(zb + B @ v), names)
        try:
            y0, T, res = solve(P, guess(P, y_seed))
            T, res = float(T), float(res)
            # 1a. a converged orbit with a sane period
            if not (np.isfinite(T) and res < 1e-4 and 0.25 * T_nom < T < 4.0 * T_nom):
                return None
            cyc = np.asarray(solver.cycle(P, y0, T, 128))
            if not (np.isfinite(cyc).all() and cyc.min() >= -1e-9):
                return None
            rng_ = cyc.max(0) - cyc.min(0)
            mean = np.abs(cyc.mean(0))
            rel = rng_ / np.maximum(mean, 1e-30)
            # 1b. AMPLITUDE -- rejects equilibria, which pass every other test perfectly
            if float(rel.max()) < amp_floor:
                return None
            # 2. circadian
            if not (period_lo * T_nom < T < period_hi * T_nom):
                return None
            # 3. no species collapsed toward zero
            ratio = float(np.min(cyc.min(0) / np.maximum(mean, 1e-30)))
            if ratio < min_ratio:
                return None
            return dict(v=v, period=T, res=res, rel_amp=float(rel.max()),
                        min_ratio=ratio, norm=float(np.linalg.norm(v)))
        except Exception:
            return None

    probe.n_free = n_free
    probe.T_nom = T_nom
    return probe


def search_one(probe, seed, bound=3.0, max_draws=20000, rng_offset=10_000, keep_probes=0):
    """Rejection-sample until a healthy circadian clock is found, deterministically for `seed`.

    Returns (hit_or_None, n_draws, sampled_norms). The RNG stream is a function of the SEED
    alone, so this is reproducible independently of scheduling or of which other seeds ran.
    """
    rng = np.random.default_rng(rng_offset + int(seed))
    n = probe.n_free
    kept = []
    for i in range(1, int(max_draws) + 1):
        v = rng.uniform(-bound, bound, n)
        h = probe(v)
        if keep_probes and len(kept) < keep_probes:
            kept.append((float(np.linalg.norm(v)), h is not None))
        if h is not None:
            h['draws'] = i
            h['seed'] = int(seed)
            return h, i, kept
    return None, int(max_draws), kept


# --------------------------------------------------------------------------- #
#  parallel driver: one seed per task, no synchronisation between them
# --------------------------------------------------------------------------- #
def _init(spec):
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[var] = '1'
    os.environ.setdefault('JAX_ENABLE_X64', '1')
    _W['probe'] = make_probe(**spec['probe'])
    _W['opts'] = spec['opts']


def _work(seed):
    h, n, kept = search_one(_W['probe'], seed, **_W['opts'])
    return int(seed), h, n, kept


def find_starts(model_name, seeds, workers=1, section=None, readout=None, bound=3.0,
                period_lo=0.7, period_hi=1.4, min_ratio=1e-3, max_draws=20000,
                keep_probes=200, verbose=True):
    """Starting points for `seeds`, each a healthy circadian clock. Returns (starts, report).

    `starts` maps seed -> vector, with None where the search hit `max_draws`; the caller must
    decide what to do about that rather than being handed a silent fallback.
    """
    spec = dict(probe=dict(model_name=model_name, section=section, readout=readout,
                           period_lo=period_lo, period_hi=period_hi, min_ratio=min_ratio),
                opts=dict(bound=bound, max_draws=max_draws, keep_probes=keep_probes))
    seeds = [int(s) for s in seeds]
    t0 = time.time()
    if verbose:
        print(f"[viability] searching for {len(seeds)} healthy circadian start(s): "
              f"period in {period_lo}-{period_hi} x T_nom, no species below "
              f"{min_ratio:g} of its mean, |v| <= {bound} in every coordinate", flush=True)
    if workers and workers > 1:
        ctx = mp.get_context('spawn')
        with ctx.Pool(int(workers), initializer=_init, initargs=(spec,)) as pool:
            res = pool.map(_work, seeds, chunksize=1)
    else:
        _init(spec)
        res = [_work(s) for s in seeds]

    starts, report = {}, []
    for sd, h, n, kept in res:
        starts[sd] = None if h is None else np.asarray(h['v'])
        report.append(dict(seed=sd, found=h is not None, draws=n,
                           period=None if h is None else h['period'],
                           min_ratio=None if h is None else h['min_ratio'],
                           norm=None if h is None else h['norm'], probes=kept))
    if verbose:
        ok = [r for r in report if r['found']]
        tot = sum(r['draws'] for r in report)
        print(f"[viability] {len(ok)}/{len(seeds)} found in {time.time() - t0:.0f}s "
              f"({tot} draws, hit rate {len(ok) / max(tot, 1):.2%})", flush=True)
        for r in report:
            if r['found']:
                print(f"    seed {r['seed']:3d}  {r['draws']:6d} draws  period "
                      f"{r['period']:6.2f} h  min/mean {r['min_ratio']:.2e}  "
                      f"|v| {r['norm']:.2f}", flush=True)
            else:
                print(f"    seed {r['seed']:3d}  NO healthy circadian clock in {r['draws']} "
                      f"draws", flush=True)
        if len(ok) > 1:
            V = np.array([starts[r['seed']] for r in ok])
            D = np.linalg.norm(V[:, None] - V[None, :], axis=-1)
            off = D[np.triu_indices(len(ok), 1)]
            print(f"[viability] starts are {off.min():.2f}-{off.max():.2f} apart "
                  f"(median {np.median(off):.2f}). For scale: CMA's initial population spans "
                  f"~2.9 and two random points in the box ~9.8.", flush=True)
    return starts, report
