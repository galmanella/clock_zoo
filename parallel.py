"""
parallel.py
===========
One dispatcher for "run this function over a work list", plus the array-job shard helper.
Every driver in `analysis/` goes through `pmap`, so switching serial <-> joblib <-> ray is an
env var and never a code change.

    CLOCKZOO_BACKEND = serial | joblib | ray        (default: joblib)
    NJOBS            = worker count                 (default: -1 = all cores)

`ray` stays OPTIONAL on purpose -- it is imported lazily and only if requested, so it is not a
cluster dependency. The required runtime set is numpy/scipy/jax/matplotlib/cmocean/joblib.

SHARDING
    A SLURM array job splits the work list by `shard_of`; each task writes its own npz into the
    shared run directory (see paths.shard_filename) and a later `--merge` pass concatenates.
    Sharding is CONTIGUOUS-STRIDED (`items[i::n]`), not blocked, so a systematic cost gradient
    along the work list (e.g. parameters ordered by how slow they are) spreads evenly over
    tasks instead of piling onto one.
"""
import os
import sys
import time

from paths import shard_id


def backend():
    return os.environ.get('CLOCKZOO_BACKEND', 'joblib').lower()


def njobs(default=-1):
    return int(os.environ.get('NJOBS', default))


def shard_of(items, idx=None, n=None):
    """This task's slice of `items` (strided). Returns (subset, idx, n)."""
    i0, n0 = shard_id()
    idx = i0 if idx is None else int(idx)
    n = n0 if n is None else int(n)
    if n <= 1:
        return list(items), 0, 1
    if not (0 <= idx < n):
        raise ValueError(f"shard index {idx} out of range for {n} shards")
    return list(items)[idx::n], idx, n


def pmap(fn, items, desc='', backend_name=None, n_jobs=None):
    """Map `fn` over `items`, in parallel per the selected backend. Order is preserved.

    Deliberately NOT fault-tolerant: an exception in a worker propagates. Swallowing worker
    errors is how input_screen got a run of all-NaN results with no indication anything had
    gone wrong (PROJECT_SUMMARY 2.4). If a specific item is allowed to fail, the CALLER
    catches it inside `fn` and returns a sentinel -- explicitly, per item.
    """
    items = list(items)
    be = (backend_name or backend()).lower()
    nj = njobs() if n_jobs is None else n_jobs
    t0 = time.time()
    if desc:
        print(f"[pmap] {desc}: {len(items)} items | backend={be} n_jobs={nj}", flush=True)

    if be == 'serial' or len(items) <= 1 or nj == 1:
        out = [fn(x) for x in items]
    elif be == 'joblib':
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=nj)(delayed(fn)(x) for x in items)
    elif be == 'ray':
        import ray                                    # optional dependency, imported lazily
        if not ray.is_initialized():
            ray.init(num_cpus=(None if nj <= 0 else nj), ignore_reinit_error=True,
                     log_to_driver=True)              # surface worker errors
        remote = ray.remote(fn)
        out = ray.get([remote.remote(x) for x in items])
    else:
        raise ValueError(f"unknown CLOCKZOO_BACKEND {be!r}; use serial|joblib|ray")

    if desc:
        print(f"[pmap] {desc}: done in {time.time() - t0:.1f}s", flush=True)
    return out


def limit_blas_threads(n=1):
    """Pin BLAS/OpenMP to `n` threads. Call BEFORE importing numpy/jax in a worker-parallel
    driver: otherwise each of NJOBS processes spawns its own full thread pool and the node
    thrashes (measured badly on the cluster in input_screen; see fast_ectopic's header)."""
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(v, str(n))


def announce(**kw):
    """One-line, greppable run header for a SLURM log."""
    print("[run] " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)
    sys.stdout.flush()
