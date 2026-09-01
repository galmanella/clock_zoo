"""
analysis/characterize.py
========================
OBJECTIVE (a), step 2: the full PTC surface of every target at the BASE parameter set, on the
adaptive dose grid `analysis.scrit` derived, with its features extracted.

    $PY -m analysis.characterize --model almeida [--mode pulse|instant] [--targets A,B]

Per target it saves the raw surface plus:
    winding vs dose        type-1 / type-0 / unusable, per dose
    (S*, phi*)             dipole-filtered singularities; the LOWEST-dose one is the
                           type-1 -> type-0 transition
    twist curve            stable fixed point vs dose, and its total span
    min relative amplitude how close the perturbation comes to killing the clock

and prints the canonical per-model table that PROJECT_SUMMARY should quote, so nothing
downstream ever recomputes it.

WHAT THE COMPARISON IS FOR
    Mirsky answered "which components reset the clock" with a sharp structural rule: only the
    repressor arm resets, the positive arm is dose-flat at any dose. Whether that is a general
    feature of mammalian clock models or a fact about Mirsky's topology is exactly what three
    models of different structure -- transcription-factor level (Almeida), delay chain
    (Korencic), phosphorylation cascade (Goldbeter) -- can answer.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

import analysis  # noqa: F401
import paths
from parallel import announce, shard_of


def characterize_target(model, target, doses, mode='pulse', n_phase=32, dt=0.02,
                        skip_p=None, verbose=True):
    """Full PTC surface + features at base for one target."""
    import jax
    import jax.numpy as jnp
    from engine.ptc import (make_ptc, grid_points, phase_or_nan, valid_mask,
                            recommended_skip)
    from analysis import winding as W

    if skip_p is None:
        skip_p, _mu, _r = recommended_skip(model, tol=1e-2, verbose=False)
    f, solver = make_ptc(model, target, mode=mode, readout='raw', skip_p=skip_p, dt=dt,
                         track_min=True)
    P = model.jax_params()
    x0 = solver.guess(P)
    ph, dz = grid_points(n_phase, doses)
    z, mn = jax.jit(f)(P, x0, ph, dz)
    p, a = phase_or_nan(np.asarray(z))
    v = valid_mask(np.asarray(mn))
    p = np.where(v, p, np.nan)
    nd, M = len(doses), n_phase
    ptc = p.reshape(nd, M).T
    amp = a.reshape(nd, M).T
    val = v.reshape(nd, M).T

    old = np.arange(M) / M
    Wv = W.winding_vs_dose(W.impute(ptc))
    sings = W.find_singularities(old, doses, ptc)
    S, phi, nsing = W.detect_grid(old, doses, ptc)
    tw = W.twist_curve(old, doses, ptc)
    res = dict(target=target, mode=mode, doses=doses, old=old, ptc=ptc, amp=amp, valid=val,
               W=Wv, S=S, phi=phi, n_sing=nsing, twist=tw,
               total_twist=W.total_twist(tw), min_amp=float(np.nanmin(amp)),
               n_unusable=int(np.sum(~np.isfinite(ptc))), dt=dt, skip_p=skip_p,
               sing_phi=np.array([s['phi'] for s in sings]),
               sing_dose=np.array([s['dose'] for s in sings]),
               sing_sign=np.array([s['sign'] for s in sings]))
    if verbose:
        t0 = int(np.sum(np.isfinite(Wv) & (np.abs(Wv) < 0.5)))
        print(f"  {target:9s} S*={_f(S):>9s} phi*={_f(phi):>6s} twist={res['total_twist']:.4f} "
              f"n_sing={int(nsing)} type0={t0}/{nd} min_amp={res['min_amp']:.3f} "
              f"unusable={res['n_unusable']}", flush=True)
    return res


def _f(x):
    return 'nan' if not np.isfinite(x) else f'{x:.4g}'


def load_grids(model_name, mode, tag=None):
    """{target: (grid, dt)} from the newest scrit run."""
    tag = tag or paths.latest_run(model_name, 'scrit')
    if tag is None:
        raise SystemExit(f"run `python -m analysis.scrit --model {model_name}` first")
    d = paths.out_dir(model_name, 'scrit', tag, create=False)
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, f'scrit_{mode}*.npz'))):
        z = np.load(fp, allow_pickle=True)
        tg = [str(t) for t in z['targets']]
        for i, t in enumerate(tg):
            if f'grid__{t}' in z:
                out[t] = (np.asarray(z[f'grid__{t}']),
                          float(z['dt_used'][i]) if 'dt_used' in z else 0.02)
    return out, tag


def run(model_name, mode='pulse', targets=None, n_phase=32, shard=None, nshards=None,
        tag=None, scrit_tag=None):
    from models import get_model
    model = get_model(model_name)
    grids, stag = load_grids(model_name, mode, scrit_tag)
    all_t = [t for t in (targets or model.perturbable_targets()) if t in grids]
    missing = [t for t in (targets or model.perturbable_targets()) if t not in grids]
    if missing:
        print(f"[characterize] no dose grid for {missing} -- skipping", file=sys.stderr)
    mine, idx, n = shard_of(all_t, shard, nshards)
    tag = paths.run_tag(tag)
    announce(analysis='characterize', model=model_name, mode=mode, targets=len(all_t),
             shard=f'{idx}/{n}', doing=len(mine), scrit_tag=stag, tag=tag)

    t0 = time.time()
    rows = []
    for t in mine:
        g, dt = grids[t]
        rows.append(characterize_target(model, t, g, mode=mode, n_phase=n_phase, dt=dt))
    print(f"[characterize] {len(rows)} target(s) in {time.time() - t0:.0f}s", flush=True)

    blob = dict(model=model_name, mode=mode,
                targets=np.array([r['target'] for r in rows]),
                S=np.array([r['S'] for r in rows]),
                phi=np.array([r['phi'] for r in rows]),
                n_sing=np.array([r['n_sing'] for r in rows]),
                total_twist=np.array([r['total_twist'] for r in rows]),
                min_amp=np.array([r['min_amp'] for r in rows]),
                n_unusable=np.array([r['n_unusable'] for r in rows]),
                dt=np.array([r['dt'] for r in rows]))
    for r in rows:
        for k in ('doses', 'old', 'ptc', 'amp', 'valid', 'W', 'twist',
                  'sing_phi', 'sing_dose', 'sing_sign'):
            blob[f'{k}__{r["target"]}'] = np.asarray(r[k])
    # ONE FILE PER TARGET. A single `char_<mode>.npz` per run tag silently clobbered itself
    # whenever the driver was invoked twice with different --targets in the same tag: the
    # BMAL1/DBP run, then CRY, then PER each overwrote the previous, and the only survivor
    # looked like a complete result. Encoding the target in the name makes repeat invocations
    # accumulate instead of destroy, and readers glob and merge them.
    subset = '-'.join(str(t) for t in blob['targets'])
    out = paths.out_path(model_name, 'characterize', f'char_{mode}__{subset}.npz', tag)
    paths.savez(out, **blob)
    print(f"[characterize] -> {out}", flush=True)
    if n == 1:
        print_table(blob)
        print(f"[characterize] figures: python -m analysis.figures --model {model_name} "
              f"--which surfaces --tag <tag>", flush=True)
    return blob


def print_table(blob):
    print(f"\n{'=' * 78}\nPTC characterization -- {blob['model']} ({blob['mode']} mode)"
          f"\n{'=' * 78}")
    print(f"  {'target':10s} {'S_crit':>10s} {'phi*':>7s} {'twist':>8s} {'n_sing':>7s} "
          f"{'min amp':>8s} {'type':>7s}")
    order = np.argsort([s if np.isfinite(s) else np.inf for s in blob['S']])
    for i in order:
        t = str(blob['targets'][i])
        Wv = blob.get(f'W__{t}')
        reaches0 = bool(np.any(np.isfinite(Wv) & (np.abs(Wv) < 0.5))) if Wv is not None else False
        print(f"  {t:10s} {_f(blob['S'][i]):>10s} {_f(blob['phi'][i]):>7s} "
              f"{blob['total_twist'][i]:8.4f} {int(blob['n_sing'][i]):7d} "
              f"{blob['min_amp'][i]:8.3f} {'0' if reaches0 else '1 only':>7s}")
    n0 = int(np.sum(np.isfinite(blob['S'])))
    print(f"\n  {n0}/{len(blob['targets'])} targets show a phase singularity on their grid")
    tw = blob['total_twist']
    if np.any(np.isfinite(tw)):
        j = int(np.nanargmax(tw))
        print(f"  most twisted: {blob['targets'][j]} ({tw[j]:.4f} cyc); "
              f"least: {blob['targets'][int(np.nanargmin(tw))]} "
              f"({np.nanmin(tw):.4f} cyc)")


# NOTE: this module used to carry its own `_plot`. It has been removed rather than repaired.
# It was a second implementation of the surfaces figure, it wrote outside the
# out/<model>/<analysis>/<tag>/ convention, and when `phase_map` dropped its `twist=` overlay
# this copy silently broke while the canonical one in analysis/figures.py kept working. One
# plotter per figure; `python -m analysis.figures --which surfaces` is it.


def main(argv=None):
    ap = argparse.ArgumentParser(description='base PTC surfaces and their features')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--mode', default=os.environ.get('MODE', 'pulse'),
                    choices=('pulse', 'instant'))
    ap.add_argument('--targets', default=os.environ.get('TARGETS'))
    ap.add_argument('--n-phase', type=int, default=32)
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--nshards', type=int, default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--scrit-tag', default=None)
    a = ap.parse_args(argv)
    run(a.model, mode=a.mode, targets=(a.targets.split(',') if a.targets else None),
        n_phase=a.n_phase, shard=a.shard, nshards=a.nshards, tag=a.tag,
        scrit_tag=a.scrit_tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
