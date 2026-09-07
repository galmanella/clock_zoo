"""
analysis/ptc_sens.py
====================
OBJECTIVE (b), part 2: how much does each parameter move the PTC?

    $PY -m analysis.ptc_sens --model almeida --target BMAL1 [--shard i --nshards n] [--merge]

For every parameter x every factor (the SAME grid as analysis/lc_sens.py, so the two pair up),
recompute the PTC on the target's FIXED base dose grid and measure three features against the
base surface:

    pointwise   RMS circular difference in new_phase over the whole grid. The most direct
                "did the response change", but it mixes a rigid shift with a shape change.
    twist       RMS difference of the stable-fixed-point-vs-dose curve. SMOOTH and
                gauge-invariant, and therefore the feature to read isochron change from.
    singularity shift of (S*, phi*). Reported, but it is a TOPOLOGICAL DEFECT: discontinuous
                and quantised by the grid, so it is a poor sensitivity measure and is flagged
                as such wherever it is plotted.

THE DOSE GRID IS FIXED AT THE BASE VALUE, ON PURPOSE
    Re-deriving S_crit per parameter setting and re-centring the grid would compare each
    perturbed model to itself and hide the very effect being measured. A single-parameter
    perturbation is generically NOT a gauge motion, so its effect on a fixed absolute dose
    grid is a real physical effect, not a units artifact. See gauge/README.md.

RAW GRIDS ARE SAVED
    Every PTC grid for every (parameter, factor) goes into the npz. Plotting and re-analysis
    are then pure reads -- no recompute, and a change to the feature extractor can be applied
    retrospectively without re-running the sweep.

ERROR DISCIPLINE
    PTC generation and feature detection sit in SEPARATE try blocks, so a detection failure
    can never destroy the raw grid it was reading. That is copied deliberately from
    input_screen, where a swallowed exception plus a missing dependency produced a run of
    all-NaN results with nothing indicating anything had gone wrong.
"""
import argparse
import os
import sys
import time

import numpy as np

import analysis  # noqa: F401
import paths
from analysis.lc_sens import FACTORS, BASE_IDX
from parallel import announce, shard_of


def load_grid(model_name, target, mode='pulse', tag=None):
    """The target's adaptive dose grid from a previous `analysis.scrit` run."""
    tag = tag or paths.latest_run(model_name, 'scrit')
    if tag is None:
        return None, None
    import glob
    d = paths.out_dir(model_name, 'scrit', tag, create=False)
    for f in sorted(glob.glob(os.path.join(d, f'scrit_{mode}*.npz'))):
        z = np.load(f, allow_pickle=True)
        if f'grid__{target}' in z:
            return np.asarray(z[f'grid__{target}']), (float(z['dt_used'][
                list(z['targets']).index(target)]) if 'dt_used' in z else 0.02)
    return None, None


def make_ptc_at(model, target, mode, doses, n_phase, dt, skip_p, readout_ref=None):
    """(ptc_of(params, y_seed) -> (grid[n_phase, n_dose], amp, valid, y0, status), doses).

    `readout_ref` overrides which species carries the phase; None uses the model's own
    `readout_variable`. Exposed for `analysis/observable.py`, which reads the SAME surface
    through several candidate species to check they differ only by the calibrated origin
    (REPO_MAP hazard 14). Every other caller wants the default and is unaffected."""
    import jax
    import jax.numpy as jnp
    from engine.orbit import make_orbit_finder
    from engine.ptc import make_ptc, grid_points, phase_or_nan, valid_mask

    f, _solver = make_ptc(model, target, mode=mode, readout='raw', skip_p=skip_p, dt=dt,
                          track_min=True, readout_ref=readout_ref)
    fj = jax.jit(f)
    find, _s = make_orbit_finder(model)
    ph, dz = grid_points(n_phase, doses)
    nd = len(doses)

    def ptc_of(params, y_seed=None):
        x0, T, cyc, status = find(params, y_seed)
        if x0 is None:
            return None, None, None, None, status
        P = model.jax_params(params) if isinstance(params, dict) else params
        z, mn = fj(P, x0, ph, dz)
        p, a = phase_or_nan(np.asarray(z))
        v = valid_mask(np.asarray(mn))
        p = np.where(v, p, np.nan)          # unstable points are not data
        y0 = np.asarray(x0)[:-1]
        return (p.reshape(nd, n_phase).T, a.reshape(nd, n_phase).T,
                v.reshape(nd, n_phase).T, y0, 'ok')

    return ptc_of


def features(old, doses, ptc, base):
    """(pointwise, twist_rms, dS, dphi, total_twist) of `ptc` against the base surface."""
    from analysis import winding as W
    pt = W.circ_rms(ptc, base['ptc'])
    tw = W.twist_curve(old, doses, ptc)
    tw_rms = W.circ_rms(tw, base['twist'])
    S, phi, _n = W.detect_grid(old, doses, ptc)
    dS = (np.log(S / base['S']) if (np.isfinite(S) and np.isfinite(base['S']) and S > 0
                                    and base['S'] > 0) else np.nan)
    dphi = (abs(((phi - base['phi'] + 0.5) % 1.0) - 0.5)
            if np.isfinite(phi) and np.isfinite(base['phi']) else np.nan)
    return pt, tw_rms, dS, dphi, W.total_twist(tw), S, phi


def run(model_name, target, mode='pulse', n_phase=32, factors=FACTORS, shard=None,
        nshards=None, tag=None, scrit_tag=None, doses=None, dt=None, save_grids=True,
        resume=True):
    from models import get_model
    from engine.ptc import recommended_skip
    from analysis import winding as W

    model = get_model(model_name)
    if doses is None:
        doses, dt_grid = load_grid(model_name, target, mode, scrit_tag)
        if doses is None:
            raise SystemExit(f"no dose grid for {model_name}/{target}/{mode}; run "
                             f"`python -m analysis.scrit --model {model_name}` first")
        dt = dt or dt_grid
    dt = dt or 0.02
    skip_p, mu, _r = recommended_skip(model, tol=1e-2, verbose=False)

    base_params = model.get_parameters()
    names = [p for p in model.parameter_names if base_params[p] > 0]
    mine, idx, n = shard_of(names, shard, nshards)
    tag = paths.run_tag(tag)
    announce(analysis='ptc_sens', model=model_name, target=target, mode=mode,
             params=len(names), shard=f'{idx}/{n}', doing=len(mine),
             doses=f'{doses.min():.3g}..{doses.max():.3g}', n_dose=len(doses), dt=dt, tag=tag)

    ptc_of = make_ptc_at(model, target, mode, doses, n_phase, dt, skip_p)
    old = np.arange(n_phase) / n_phase

    p0, a0, v0, y00, st0 = ptc_of(base_params)
    if p0 is None:
        raise RuntimeError(f"{model_name}/{target}: the BASE PTC failed ({st0})")
    S0, phi0, _n0 = W.detect_grid(old, doses, p0)
    base = dict(ptc=p0, twist=W.twist_curve(old, doses, p0), S=S0, phi=phi0)
    print(f"[ptc-sens] base: S_crit={S0:.4g} phi*={phi0:.3f} "
          f"total twist={W.total_twist(base['twist']):.4f} cyc, "
          f"{int(np.sum(~np.isfinite(p0)))}/{p0.size} unusable points", flush=True)

    nf = len(factors)
    PT = np.full((len(mine), nf), np.nan)
    TW = np.full((len(mine), nf), np.nan)
    DS = np.full((len(mine), nf), np.nan)
    DP = np.full((len(mine), nf), np.nan)
    TT = np.full((len(mine), nf), np.nan)
    SS = np.full((len(mine), nf), np.nan)
    PH = np.full((len(mine), nf), np.nan)
    ST = np.empty((len(mine), nf), dtype=object)
    grids = np.full((len(mine), nf, n_phase, len(doses)), np.nan, np.float32)
    # amplitude and validity are RAW ENGINE OUTPUT too. They were being computed and thrown
    # away, which meant "was this point dead or was the integrator unstable?" could only be
    # answered by re-running the sweep. They cost the same as the phase grid to store.
    amps = np.full((len(mine), nf, n_phase, len(doses)), np.nan, np.float32)
    vals = np.zeros((len(mine), nf, n_phase, len(doses)), bool)

    def build(done):
        span = lambda A: np.nanmax(np.abs(A), axis=1) if A.size else A
        b = dict(model=model_name, target=target, mode=mode, params=np.array(mine),
                 factors=factors, doses=doses, old=old, dt=dt, skip_p=skip_p, mu=mu,
                 base_ptc=p0.astype(np.float32), base_twist=base['twist'],
                 base_S=S0, base_phi=phi0, status=ST.astype(str),
                 pointwise=PT, twist=TW, dS=DS, dphi=DP, total_twist=TT,
                 S=SS, phi=PH,
                 pointwise_span=span(PT), twist_span=span(TW),
                 dS_span=span(DS), dphi_span=span(DP),
                 done=np.asarray(done, bool), complete=bool(np.all(done)))
        if save_grids:
            b['ptc_grids'] = grids
            b['amp_grids'] = amps
            b['valid_grids'] = vals
            b['base_amp'] = a0.astype(np.float32)
            b['base_valid'] = v0
        return b

    out = paths.out_path(model_name, 'ptc_sens',
                         paths.shard_filename(f'ptc_sens_{target}_{mode}',
                                              idx if n > 1 else None), tag)
    ckpt = out.replace('.npz', '.partial.npz')
    done = np.zeros(len(mine), bool)

    # RESUME. This sweep is ~2 h and wrote nothing until the very end, so a kill at 90 minutes
    # left no output at all -- the same all-or-nothing failure lc_sens had.
    if resume and os.path.exists(ckpt):
        c = np.load(ckpt, allow_pickle=True)
        if list(c['params']) == list(mine) and len(c['doses']) == len(doses):
            done = np.asarray(c['done'], bool)
            for dst, key in ((PT, 'pointwise'), (TW, 'twist'), (DS, 'dS'), (DP, 'dphi'),
                             (TT, 'total_twist'), (SS, 'S'), (PH, 'phi')):
                dst[...] = c[key]
            ST[...] = c['status'].astype(object)
            if save_grids and 'ptc_grids' in c:
                grids[...], amps[...], vals[...] = (c['ptc_grids'], c['amp_grids'],
                                                    c['valid_grids'])
            print(f"[ptc-sens] resuming from {ckpt}: {int(done.sum())}/{len(mine)} "
                  f"parameter(s) already done", flush=True)
        else:
            print(f"[ptc-sens] checkpoint {ckpt} does not match this run "
                  f"(different parameters or dose grid) -- starting over", flush=True)

    t0 = time.time()
    nsolve = [0]
    total = int((~done).sum()) * nf
    for i, name in enumerate(mine):
        if done[i]:
            continue
        def one(k, y_seed):
            tk = time.time()
            pd = dict(base_params)
            pd[name] = base_params[name] * float(factors[k])
            try:                                     # (1) PTC generation
                p, a, v, y0, st = ptc_of(pd, y_seed)
            except Exception as e:
                print(f"    [{name} x{factors[k]}] PTC FAILED {type(e).__name__}: {e}",
                      flush=True)
                ST[i, k] = 'error'
                tick(k, 'error', time.time() - tk)
                return None
            ST[i, k] = st
            if p is None:
                tick(k, st, time.time() - tk)
                return None
            grids[i, k] = p.astype(np.float32)       # raw grids saved BEFORE any detection
            amps[i, k] = a.astype(np.float32)
            vals[i, k] = v
            try:                                     # (2) feature detection, separately, so a
                pt, tw, dS, dphi, tt, S, phi = features(old, doses, p, base)   # detector bug
                PT[i, k], TW[i, k] = pt, tw          # can never destroy the raw grid
                DS[i, k], DP[i, k], TT[i, k] = dS, dphi, tt
                SS[i, k], PH[i, k] = S, phi
            except Exception as e:
                print(f"    [{name} x{factors[k]}] DETECT FAILED {type(e).__name__}: {e}",
                      flush=True)
            tick(k, st, time.time() - tk)
            return y0

        def tick(k, st, secs, _i=i, _n=name):
            """One line per FACTOR. A per-parameter line is ~6 min apart here, which cannot
            tell a slow sweep from a hung one."""
            nsolve[0] += 1
            el = time.time() - t0
            rate = el / max(nsolve[0], 1)
            print(f"[ptc-sens] {_i + 1:2d}/{len(mine)} {_n:>10s}  x{factors[k]:<5g} "
                  f"({k + 1}/{nf})  {str(st):<18s} {secs:6.1f}s  | {nsolve[0]:3d}/{total} "
                  f"grids, {el / 60:5.1f} min elapsed, "
                  f"eta {rate * (total - nsolve[0]) / 60:5.1f} min", flush=True)

        yb = one(BASE_IDX, None)
        for rng in (range(BASE_IDX + 1, nf), range(BASE_IDX - 1, -1, -1)):
            y = yb
            for k in rng:
                yn = one(k, y)
                if yn is not None:
                    y = yn
        done[i] = True
        paths.savez(ckpt, **build(done))     # an interrupt now costs one parameter

    blob = build(done)
    if not save_grids:
        print("[ptc-sens] WARNING --no-grids: the raw surfaces are NOT being saved, so every "
              "plot or re-analysis will need the whole sweep re-run", file=sys.stderr)
    paths.savez(out, **blob)
    print(f"[ptc-sens] -> {out}  ({int(done.sum())}/{len(mine)} parameters, "
          f"{(time.time() - t0) / 60:.1f} min)", flush=True)
    for stale in (ckpt, ckpt + '.meta.json'):    # savez writes a provenance sidecar too
        if os.path.exists(stale):
            os.remove(stale)                      # the real output supersedes the checkpoint
    if n == 1:
        report(blob)
    return blob


def report(blob, top=15):
    names = blob['params']
    tw, pt = blob['twist_span'], blob['pointwise_span']
    print(f"\n{'=' * 78}\nPTC sensitivity -- {blob['model']} / {blob['target']} "
          f"({blob['mode']})\n{'=' * 78}")
    print(f"  base S_crit = {float(blob['base_S']):.4g}, phi* = {float(blob['base_phi']):.3f}, "
          f"dose {blob['doses'].min():.3g}..{blob['doses'].max():.3g}, dt = {float(blob['dt']):g}")
    print(f"\n  ranked by TWIST response (the smooth, gauge-invariant feature):")
    print(f"  {'parameter':16s} {'twist':>9s} {'pointwise':>10s} {'|dlog S*|':>10s} "
          f"{'|dphi*|':>9s}")
    for i in np.argsort(-np.nan_to_num(tw))[:top]:
        print(f"  {str(names[i]):16s} {tw[i]:9.4f} {pt[i]:10.4f} "
              f"{blob['dS_span'][i]:10.4f} {blob['dphi_span'][i]:9.4f}")
    st = np.asarray(blob['status']).astype(str)
    bad = {k: int((st == k).sum()) for k in np.unique(st) if k not in ('ok', 'None')}
    if bad:
        print(f"  rejected settings (of {st.size}): {bad}")


def merge(model_name, target, mode='pulse', tag=None):
    tag = tag or paths.latest_run(model_name, 'ptc_sens')
    shards = paths.load_shards(model_name, 'ptc_sens', tag,
                               prefix=f'ptc_sens_{target}_{mode}')
    if not shards:
        print(f"[ptc-sens] nothing to merge", file=sys.stderr)
        return None
    blob = {k: shards[0][k] for k in ('model', 'target', 'mode', 'factors', 'doses', 'old',
                                      'dt', 'skip_p', 'mu', 'base_ptc', 'base_twist',
                                      'base_S', 'base_phi')}
    for k in ('params', 'pointwise_span', 'twist_span', 'dS_span', 'dphi_span'):
        blob[k] = np.concatenate([np.atleast_1d(s[k]) for s in shards])
    for k in ('pointwise', 'twist', 'dS', 'dphi', 'total_twist', 'S', 'phi', 'status'):
        blob[k] = np.concatenate([np.atleast_2d(s[k]) for s in shards], axis=0)
    for k in ('ptc_grids', 'amp_grids', 'valid_grids'):
        if all(k in s for s in shards):
            blob[k] = np.concatenate([s[k] for s in shards], axis=0)
    for k in ('base_amp', 'base_valid'):
        if k in shards[0]:
            blob[k] = shards[0][k]
    out = paths.out_path(model_name, 'ptc_sens', f'ptc_sens_{target}_{mode}_merged.npz', tag)
    paths.savez(out, **blob)
    report(blob)
    print(f"[ptc-sens] merged {len(shards)} shard(s) -> {out}", flush=True)
    return blob


def main(argv=None):
    ap = argparse.ArgumentParser(description='per-parameter PTC sensitivity')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--target', default=os.environ.get('TARGET'))
    ap.add_argument('--mode', default=os.environ.get('MODE', 'pulse'),
                    choices=('pulse', 'instant'))
    ap.add_argument('--n-phase', type=int, default=32)
    ap.add_argument('--dt', type=float, default=None)
    ap.add_argument('--scrit-tag', default=None)
    ap.add_argument('--doses', default=None,
                    help="explicit grid 'lo,hi,n' instead of scrit's adaptive one -- for "
                         'when the result must share a dose axis with another figure')
    ap.add_argument('--scale', default='linear', choices=('linear', 'log'),
                    help='spacing for --doses (default linear, as the R01 figures use)')
    ap.add_argument('--no-grids', action='store_true', help='do not save the raw PTC grids')
    ap.add_argument('--no-resume', action='store_true',
                    help='ignore any checkpoint and start over')
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--nshards', type=int, default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args(argv)
    if not a.target:
        raise SystemExit('--target is required (pick a resetting one from analysis.scrit)')
    if a.merge:
        return 0 if merge(a.model, a.target, a.mode, a.tag) is not None else 1
    dz = None
    if a.doses:
        lo, hi, nd = a.doses.split(',')
        lo, hi, nd = float(lo), float(hi), int(nd)
        # A linear grid starts at `lo` INCLUSIVE, so '0,100,49' puts a genuine dose-0 row
        # on the grid -- the readout-calibration control (REPO_MAP hazard 20).
        dz = (np.linspace(lo, hi, nd) if a.scale == 'linear'
              else np.geomspace(max(lo, 1e-12), hi, nd))
    run(a.model, a.target, mode=a.mode, n_phase=a.n_phase, dt=a.dt, shard=a.shard,
        nshards=a.nshards, tag=a.tag, scrit_tag=a.scrit_tag, doses=dz,
        save_grids=not a.no_grids, resume=not a.no_resume)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
