"""
analysis/lc_sens.py
===================
OBJECTIVE (b), part 1: how much does each parameter move the LIMIT CYCLE?

    $PY -m analysis.lc_sens --model almeida [--shard i --nshards n] [--merge]

For every parameter x every factor in a fixed multiplicative grid, re-solve the periodic orbit
and measure how far the cycle moved from base:

    shape   RMS change of the phase-aligned state profiles, each normalised by its own base
            amplitude -- so a species is compared against how much it itself oscillates, and a
            large-but-flat species cannot dominate.
    period  relative change of T.
    lc_sens sqrt(shape^2 + period^2), both dimensionless.

Reported for both the OBSERVABLE states (what an experiment could actually constrain) and ALL
states (the full theoretical cycle). The gap between those two is itself informative: a
parameter that moves hidden states but not observable ones is invisible to data.

TARGET-INDEPENDENT, so this runs once per model and pairs with every ptc_sens run.

WHY FINITE DIFFERENCES AND NOT AUTODIFF
    The orbit solver IS differentiable and its jacobian is trustworthy. But the PTC side of
    this comparison must be finite-difference (autodiffing a fixed-step RK4 PTC at nontrivial
    dose is catastrophically wrong -- see engine/ptc.py), and the two halves have to be
    measured the same way or the comparison in analysis/coupling.py is meaningless. A factor
    sweep also probes a FINITE displacement, which is what "how much does this parameter
    matter" actually asks; a derivative at a point can be tiny while the finite response is
    large, and vice versa.

PHASE ALIGNMENT IS FREE HERE
    Both cycles come from the BVP solver anchored at the same physical event (a turning point
    of the reference species), on an exact uniform phase grid. So the profiles are already
    aligned and comparing them is a plain subtraction -- no cross-correlation, no estimated
    offset. That is the whole reason engine/orbit.py exists.
"""
import argparse
import os
import sys
import time

import numpy as np

import analysis  # noqa: F401
import paths
from parallel import announce, shard_of

#: The multiplicative factor grid. Same as input_screen's, so numbers are comparable across
#: the two projects: roughly log-uniform over x0.25 .. x4 with 1.0 in the middle.
FACTORS = np.array([0.25, 0.5, 0.62, 0.79, 1.0, 1.26, 1.6, 2.0, 4.0])
BASE_IDX = 4                      # index of factor 1.0


def make_profiles_fn(model, m=64, n_steps=1024):
    """cycle_of(params, y_seed) -> (profiles[n_states, m], T, status, y0).

    A thin wrapper over engine.orbit.make_orbit_finder, which owns every hard-won detail of
    getting a trustworthy orbit at a displaced parameter set (a small residual is not enough;
    continue through the relaxation, not into Newton; multi-start over the period). Both
    sensitivity drivers go through it so the two halves of the coupling analysis cannot drift
    apart in how they decide an orbit is real.
    """
    from engine.orbit import make_orbit_finder
    find, _solver = make_orbit_finder(model, m=m, n_steps=n_steps)

    def cycle_of(params, y_seed=None):
        x0, T, C, status = find(params, y_seed)
        if x0 is None:
            return None, T, status, None
        return C, T, status, np.asarray(x0)[:-1]

    return cycle_of


def run_param(cycle_of, name, base, obs_idx, C0, T0, scale, factors=FACTORS, on_factor=None):
    """Per-factor LC displacement for one parameter, plus a per-factor status string.

    A factor whose orbit is rejected (dead / non-converged / absurd period) leaves NaN in the
    metrics and records WHY. That is information, not a hole: a parameter that abolishes the
    oscillation at x4 has told you something important about the model, and it must not be
    conflated with a parameter that merely does nothing.

    `on_factor(k, status, seconds)` fires after EVERY factor solve. One parameter is nine
    orbit solves and several minutes on the adaptive backend, so a per-parameter line is not a
    heartbeat -- it cannot distinguish slow from stuck, which is what cost 80 minutes of
    silence here once already."""
    shape_all = np.full(len(factors), np.nan)
    shape_obs = np.full(len(factors), np.nan)
    dT = np.full(len(factors), np.nan)
    status = np.array(['?'] * len(factors), dtype=object)
    prof = np.full((len(factors),) + C0.shape, np.nan, np.float32)   # raw cycles, for coupling

    def one(k, y_seed):
        """Solve at factor k, continuing from the previous cycle point; if that fails, retry
        from the model's own initial condition."""
        tk = time.time()
        pd = dict(base)
        pd[name] = base[name] * float(factors[k])
        try:
            C, T, st, xs = cycle_of(pd, y_seed)
            if C is None and y_seed is not None:   # continuation missed: try a fresh start
                C, T, st, xs = cycle_of(pd, None)
        except Exception as e:                     # never let one parameter kill the sweep
            print(f"    [{name} x{factors[k]}] orbit FAILED {type(e).__name__}: {e}",
                  flush=True)
            status[k] = 'error'
            if on_factor:
                on_factor(k, 'error', time.time() - tk)
            return None
        status[k] = st
        if C is not None:
            prof[k] = C.astype(np.float32)     # raw cycle saved BEFORE any summarising
            d = (C - C0) / scale[:, None]
            shape_all[k] = float(np.sqrt(np.mean(d ** 2)))
            shape_obs[k] = float(np.sqrt(np.mean(d[obs_idx] ** 2)))
            dT[k] = (T - T0) / T0
        if on_factor:
            on_factor(k, st, time.time() - tk)
        return None if C is None else xs

    # walk OUTWARD from the base factor in both directions, continuing from the previous
    # solution: adjacent factors have nearby orbits, so each solve starts inside Newton's basin
    xb = one(BASE_IDX, None)
    for rng in (range(BASE_IDX + 1, len(factors)), range(BASE_IDX - 1, -1, -1)):
        x = xb
        for k in rng:
            xn = one(k, x)
            if xn is not None:
                x = xn                             # keep the last good x across a failure
    return shape_all, shape_obs, dT, status, prof


def run(model_name, shard=None, nshards=None, tag=None, m=64, factors=FACTORS,
        resume=True):
    from models import get_model
    model = get_model(model_name)
    base = model.get_parameters()
    names = [p for p in model.parameter_names if base[p] > 0]
    mine, idx, n = shard_of(names, shard, nshards)
    tag = paths.run_tag(tag)
    announce(analysis='lc_sens', model=model_name, params=len(names),
             shard=f'{idx}/{n}', doing=len(mine), tag=tag)

    cycle_of = make_profiles_fn(model, m=m)
    C0, T0, st0, _x0 = cycle_of(base)
    if C0 is None:
        raise RuntimeError(f"{model_name}: the BASE orbit is not usable ({st0})")
    # per-species normaliser: its own peak-to-trough on the base cycle. A species that barely
    # oscillates should not be able to dominate the metric just because it is large.
    scale = np.maximum(C0.max(1) - C0.min(1), 1e-9)
    obs = list(model.observable_states())
    obs_idx = np.array([model.var_index(s) for s in obs])
    print(f"[lc-sens] base T = {T0:.4f} h; observables {obs}", flush=True)

    def assemble(SA, SO, DT, ST, PR, done):
        """Build the output blob from however many parameters are finished."""
        A, O, D = np.array(SA), np.array(SO), np.array(DT)
        S = np.array(ST, dtype=object)
        span = lambda M: np.nanmax(np.abs(M), axis=1) if M.size else M
        b = dict(model=model_name, params=np.array(done), factors=factors,
                 shape_all=A, shape_obs=O, dT=D, status=S.astype(str),
                 profiles=np.array(PR, np.float32),
                 base_profiles=C0.astype(np.float32), scale=scale, obs_idx=obs_idx,
                 shape_all_span=span(A), shape_obs_span=span(O), dT_span=span(D),
                 T0=T0, observables=np.array(obs), base_idx=BASE_IDX,
                 n_params_planned=len(mine), complete=len(done) == len(mine))
        b['lc_sens'] = np.sqrt(b['shape_all_span'] ** 2 + b['dT_span'] ** 2)
        b['lc_sens_obs'] = np.sqrt(b['shape_obs_span'] ** 2 + b['dT_span'] ** 2)
        return b

    out = paths.out_path(model_name, 'lc_sens',
                         paths.shard_filename('lc_sens', idx if n > 1 else None), tag)
    ckpt = out.replace('.npz', '.partial.npz')

    # RESUME. A killed run used to lose everything -- 18 parameters are ~2 h and the blob was
    # written only at the very end, so an interrupt at 80 minutes left nothing on disk at all.
    SA, SO, DT, ST, PR, done = [], [], [], [], [], []
    if resume and os.path.exists(ckpt):
        c = dict(np.load(ckpt, allow_pickle=True))
        done = [str(x) for x in c['params']]
        SA, SO = list(c['shape_all']), list(c['shape_obs'])
        DT, ST, PR = list(c['dT']), list(c['status']), list(c['profiles'])
        print(f"[lc-sens] resuming from {ckpt}: {len(done)} parameter(s) already done "
              f"({', '.join(done)})", flush=True)
    todo = [q for q in mine if q not in set(done)]
    n_done0 = len(done)          # fixed: `done` grows during the loop, so using it
                                 # live numbered the second parameter '3/2'

    t0 = time.time()
    nfac = len(factors)
    total = len(todo) * nfac
    seen = [0]

    for i, q in enumerate(todo):
        def on_factor(k, st, secs, _q=q, _i=i):
            seen[0] += 1
            el = time.time() - t0
            rate = el / max(seen[0], 1)
            print(f"[lc-sens] {n_done0 + _i + 1:2d}/{len(mine)} {_q:>10s}  "
                  f"x{factors[k]:<5g} ({k + 1}/{nfac})  {str(st):<18s} {secs:6.1f}s  "
                  f"| {seen[0]:3d}/{total} solves, {el / 60:5.1f} min elapsed, "
                  f"eta {rate * (total - seen[0]) / 60:5.1f} min", flush=True)
        a, o, d, st, pr = run_param(cycle_of, q, base, obs_idx, C0, T0, scale, factors,
                                    on_factor=on_factor)
        SA.append(a); SO.append(o); DT.append(d); ST.append(st); PR.append(pr)
        done.append(q)
        # checkpoint after EVERY parameter: an interrupt now costs one parameter, not the run
        paths.savez(ckpt, **assemble(SA, SO, DT, ST, PR, done))

    blob = assemble(SA, SO, DT, ST, PR, done)
    # restore the declared parameter order: a resumed run finishes in a different order than
    # it started, and a sweep should not depend on where it was interrupted
    order = [done.index(q) for q in mine if q in set(done)]
    if order != list(range(len(done))):
        for k in ('params', 'shape_all', 'shape_obs', 'dT', 'status', 'profiles',
                  'shape_all_span', 'shape_obs_span', 'dT_span', 'lc_sens', 'lc_sens_obs'):
            blob[k] = np.asarray(blob[k])[order]
    paths.savez(out, **blob)
    print(f"[lc-sens] -> {out}  ({len(done)}/{len(mine)} parameters, "
          f"{(time.time() - t0) / 60:.1f} min)", flush=True)
    for stale in (ckpt, ckpt + '.meta.json'):    # savez writes a provenance sidecar too
        if os.path.exists(stale):
            os.remove(stale)                    # the real output supersedes the checkpoint
    if n == 1:
        report(blob)
    return blob


def report(blob, top=15):
    ls = blob['lc_sens']
    names = blob['params']
    dead = int(np.sum(~np.isfinite(ls)))
    print(f"\n{'=' * 78}\nLC sensitivity -- {blob['model']} "
          f"({len(names)} parameters, T0 = {float(blob['T0']):.4f} h)\n{'=' * 78}")
    print(f"  {'parameter':16s} {'lc_sens':>9s} {'shape(all)':>11s} {'shape(obs)':>11s} "
          f"{'|dT|/T':>9s}")
    for i in np.argsort(-np.nan_to_num(ls))[:top]:
        print(f"  {str(names[i]):16s} {ls[i]:9.4f} {blob['shape_all_span'][i]:11.4f} "
              f"{blob['shape_obs_span'][i]:11.4f} {blob['dT_span'][i]:9.4f}")
    flat = np.argsort(np.nan_to_num(ls, nan=np.inf))[:5]
    print(f"  ... least sensitive: "
          f"{', '.join(f'{names[i]}={ls[i]:.2e}' for i in flat)}")
    if dead:
        print(f"  {dead} parameter(s) had no usable orbit at any factor")
    st = blob.get('status')
    if st is not None:
        st = np.asarray(st).astype(str)
        bad = {k: int((st == k).sum()) for k in np.unique(st) if k != 'ok'}
        if bad:
            print(f"  rejected settings (of {st.size}): {bad}")
            for k in bad:
                who = sorted({str(names[i]) for i in np.where((st == k).any(axis=1))[0]})
                print(f"      {k:20s} {', '.join(who[:8])}"
                      f"{' ...' if len(who) > 8 else ''}")


def merge(model_name, tag=None):
    tag = tag or paths.latest_run(model_name, 'lc_sens')
    shards = paths.load_shards(model_name, 'lc_sens', tag, prefix='lc_sens')
    if not shards:
        print(f"[lc-sens] nothing to merge for {model_name} tag={tag}", file=sys.stderr)
        return None
    blob = dict(model=model_name, factors=shards[0]['factors'], T0=shards[0]['T0'],
                observables=shards[0]['observables'], base_idx=shards[0]['base_idx'],
                base_profiles=shards[0]['base_profiles'], scale=shards[0]['scale'],
                obs_idx=shards[0]['obs_idx'])
    for k in ('params', 'shape_all_span', 'shape_obs_span', 'dT_span', 'lc_sens',
              'lc_sens_obs'):
        blob[k] = np.concatenate([np.atleast_1d(s[k]) for s in shards])
    for k in ('shape_all', 'shape_obs', 'dT', 'status', 'profiles'):
        blob[k] = np.concatenate([np.atleast_2d(s[k]) for s in shards], axis=0)
    out = paths.out_path(model_name, 'lc_sens', 'lc_sens_merged.npz', tag)
    paths.savez(out, **blob)
    report(blob)
    print(f"[lc-sens] merged {len(shards)} shard(s) -> {out}", flush=True)
    return blob


def main(argv=None):
    ap = argparse.ArgumentParser(description='per-parameter limit-cycle sensitivity')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--m', type=int, default=64, help='cycle sample points')
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--nshards', type=int, default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--no-resume', action='store_true',
                    help='ignore any checkpoint and start over')
    a = ap.parse_args(argv)
    if a.merge:
        return 0 if merge(a.model, a.tag) is not None else 1
    run(a.model, shard=a.shard, nshards=a.nshards, tag=a.tag, m=a.m,
        resume=not a.no_resume)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
