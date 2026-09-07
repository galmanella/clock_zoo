"""
analysis/displace_demo.py
=========================
Show what a LARGE displacement along a chosen decoupling direction actually does: the full PTC
surface relative to base, and the limit cycle next to base.

    $PY -m analysis.displace_demo --model almeida --target BMAL1 --mode instant --dirs 1,2,3,6

COMPUTE ONLY. Writes one npz with the raw surfaces and cycles; `analysis/displace_plots.py`
draws them. Iterating on the figure must not re-run the engine.

WHY THE STEP IS NORMALISED BY THE LARGEST PARAMETER CHANGE
    The eigen-directions are UNIT vectors in log-parameter space, so a fixed `eps` moves each
    one by a different amount in the only units a reader cares about -- how much did a
    parameter actually change. A direction spread over many parameters gets a small fold change
    per parameter; one concentrated on two gets a large one. Comparing them at fixed `eps`
    therefore compares different-sized perturbations and the comparison is not fair.

    So each direction is scaled to make its LARGEST single-parameter change exactly `fold`
    (default 2x): eps = log(fold) / max|v_i|. Every direction shown is then "the same size
    perturbation" in the strongest parameter it touches, and the panels are comparable.

WHY BOTH SIGNS
    A 2x displacement is far outside the linear regime the directions were derived in -- the
    eps = 0.15 verification already showed the top-rho direction going 40% asymmetric. +v and
    -v are both computed so the asymmetry is visible rather than assumed away.
"""
import argparse
import os
import sys
import time

import numpy as np

import analysis  # noqa: F401
import paths


def _scaled_eps(v, fold):
    """eps such that the largest |parameter fold change| along v is exactly `fold`."""
    return float(np.log(fold) / max(np.max(np.abs(v)), 1e-12))


def run(model_name, target, mode='instant', dirs=(1, 2, 3, 6), fold=2.0, feature='twist',
        n_phase=32, doses=None, dt=0.02, tag=None):
    from models import get_model
    from engine.orbit import make_orbit_finder
    from engine.ptc import recommended_skip
    from analysis.ptc_sens import make_ptc_at

    model = get_model(model_name)
    fp = paths.out_path(model_name, 'coupling',
                        f'coupling_{target}_{mode}_{feature}.npz')
    if not os.path.exists(fp):
        raise SystemExit(f"no coupling run at {fp}")
    cp = dict(np.load(fp, allow_pickle=True))
    names = [str(p) for p in cp['params']]
    V, rho = np.asarray(cp['V']), np.asarray(cp['rho'])

    if doses is None:
        doses = np.linspace(0.0, 100.0, 49)
    doses = np.asarray(doses, float)
    skip_p, _mu, _r = recommended_skip(model, tol=1e-2, verbose=False)
    ptc_of = make_ptc_at(model, target, mode, doses, n_phase, dt, skip_p)
    find, _s = make_orbit_finder(model)

    base = model.get_parameters()
    _x0, T0, C0, st0 = find(base)
    if C0 is None:
        raise SystemExit(f"base orbit not usable ({st0})")
    scale = np.maximum(C0.max(1) - C0.min(1), 1e-9)
    p0, a0, v0, _y0, stp = ptc_of(base)
    if p0 is None:
        raise SystemExit(f"base PTC failed ({stp})")
    print(f"[displace] base T = {T0:.4f} h, {int(np.sum(~np.isfinite(p0)))}/{p0.size} "
          f"unusable base PTC points", flush=True)
    print(f"[displace] {n_phase} phases x {len(doses)} doses, "
          f"dose {doses.min():g}..{doses.max():g} linear\n", flush=True)

    rows, PTC, CYC, VEC, PV = [], [], [], [], []
    t0 = time.time()
    todo = [(int(k), s) for k in dirs for s in (+1, -1)]
    print(f"  {'direction':>10} {'sign':>5} {'rho':>9} {'eps':>7} {'max fold':>9} "
          f"{'dLC':>8} {'|dT|/T':>8} {'dPTC rms':>9} {'status':>12}")
    for n, (k, sgn) in enumerate(todo):
        v = np.asarray(V[:, k], float)
        v = v / np.linalg.norm(v)
        eps = _scaled_eps(v, fold)
        d = sgn * eps * v                                  # log-space displacement
        pd = dict(base)
        for nm, di in zip(names, d):
            pd[nm] = base[nm] * float(np.exp(di))
        _x, T, C, st = find(pd)
        if C is None:
            print(f"  {('dir %02d' % k):>10} {sgn:+5d} {rho[k]:9.4g} {eps:7.3f} "
                  f"{fold:9.2f} {'':8} {'':8} {'':9} {'orbit ' + st:>12}", flush=True)
            continue
        dlc = float(np.sqrt(np.mean(((C - C0) / scale[:, None]) ** 2)))
        dT = (T - T0) / T0
        p, a, vv, _y, stp = ptc_of(pd)
        if p is None:
            print(f"  {('dir %02d' % k):>10} {sgn:+5d} {rho[k]:9.4g} {eps:7.3f} "
                  f"{fold:9.2f} {dlc:8.4f} {abs(dT):8.4f} {'':9} {'ptc ' + stp:>12}",
                  flush=True)
            continue
        # shortest signed arc: the PTC is circular, a plain difference invents full-cycle jumps
        dd = (p - p0 + 0.5) % 1.0 - 0.5
        good = np.isfinite(dd)
        dptc = float(np.sqrt(np.mean(dd[good] ** 2))) if good.any() else np.nan
        print(f"  {('dir %02d' % k):>10} {sgn:+5d} {rho[k]:9.4g} {eps:7.3f} {fold:9.2f} "
              f"{dlc:8.4f} {abs(dT):8.4f} {dptc:9.4f} {stp:>12}"
              f"   [{n + 1}/{len(todo)}, {time.time() - t0:.0f}s]", flush=True)
        rows.append(dict(dir=k, sign=sgn, rho=float(rho[k]), eps=eps, dLC=dlc, dT=float(dT),
                         dPTC_rms=dptc, period=float(T), status=str(stp),
                         n_bad=int(np.sum(~good))))
        PTC.append(p.astype(np.float32))
        CYC.append(C.astype(np.float32))
        VEC.append(d)
        PV.append([pd[nm] for nm in names])

    if not rows:
        raise SystemExit('every displacement failed; nothing to save')
    blob = dict(model=model_name, target=target, mode=mode, feature=feature, fold=fold,
                doses=doses, old=np.arange(n_phase) / n_phase, dt=dt, n_phase=n_phase,
                params=np.array(names), base_params=np.array([base[n] for n in names]),
                base_ptc=p0.astype(np.float32), base_cycle=C0.astype(np.float32),
                base_period=T0, base_amp=a0.astype(np.float32), base_valid=v0,
                state_names=np.array(list(model.state_names)),
                ptc=np.array(PTC), cycles=np.array(CYC),
                displacements=np.array(VEC), param_values=np.array(PV),
                dirs=np.array([r['dir'] for r in rows]),
                sign=np.array([r['sign'] for r in rows]),
                rho=np.array([r['rho'] for r in rows]),
                eps=np.array([r['eps'] for r in rows]),
                dLC=np.array([r['dLC'] for r in rows]),
                dT=np.array([r['dT'] for r in rows]),
                dPTC_rms=np.array([r['dPTC_rms'] for r in rows]),
                period=np.array([r['period'] for r in rows]),
                n_bad=np.array([r['n_bad'] for r in rows]),
                status=np.array([r['status'] for r in rows]))
    out = paths.out_path(model_name, 'coupling',
                         f'displace_{target}_{mode}_x{fold:g}.npz', tag)
    paths.savez(out, **blob)
    print(f"\n[displace] -> {out}", flush=True)
    return blob


def main(argv=None):
    ap = argparse.ArgumentParser(description='large displacement along chosen directions')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    ap.add_argument('--mode', default='instant')
    ap.add_argument('--dirs', default='1,2,3,6',
                    help='eigen-direction indices, comma separated')
    ap.add_argument('--fold', type=float, default=2.0,
                    help='largest single-parameter fold change (default 2x)')
    ap.add_argument('--n-phase', type=int, default=32)
    ap.add_argument('--doses', default='0,100,49', help="linear dose grid 'lo,hi,n'")
    ap.add_argument('--dt', type=float, default=0.02)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    lo, hi, nd = a.doses.split(',')
    run(a.model, a.target, a.mode, tuple(int(x) for x in a.dirs.split(',')), a.fold,
        n_phase=a.n_phase, doses=np.linspace(float(lo), float(hi), int(nd)), dt=a.dt,
        tag=a.tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
