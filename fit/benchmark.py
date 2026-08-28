"""
fit/benchmark.py
================
Head-to-head on ONE problem, with a matched evaluation budget.

    python -m fit.benchmark --evals 1500

The problem is the T1 self-recovery control at eps = 0.3, seed 0, 12 phases x 8 doses -- exactly
the setting in which the existing runs scored:

    CMA-ES (IPOP)   0.0286   1512 evals    <- the number to beat
    L-BFGS          0.0607    187 evals
    Nelder-Mead     0.4449   4000 evals
    truth (floor)   0.000392

Any new method has to be measured against that or it is a putative solution, not a solution.
Costs here are directly comparable: same target, same grid, same start, budget matched.

WHAT IS BEING TESTED
    cma-ipop   the incumbent, re-run here so the comparison is like-for-like rather than quoted
    cma-anneal restarts from best-so-far with contracting sigma instead of random restarts with
               growing population
    str        trust-region descent on a finite-difference gradient at radius delta, annealed
    multires   coarse-to-fine grid continuation; the target is re-rendered from the known truth
               at each resolution, which is legitimate because the target is synthetic

REPORTED
    final cost, evaluations, wall time, and -- separately -- the distance to the truth in the
    IDENTIFIED subspace. The full-space distance charges an optimizer for directions the data
    does not constrain, which is why 5.4b/5.1 exist; cost alone is the fair ranking, and the
    subspace distance says whether a low cost also means the right model.
"""
import argparse
import glob
import os
import time

import numpy as np

import paths
from fit import search


def build(model, target, mode, doses, n_phase, zt, w_osc=0.2, w_amp=1.0):
    from fit.cost import make_cost, FixedTarget
    return make_cost(model, target, doses, FixedTarget(zt), n_phase=n_phase, mode=mode,
                     backend='diffrax', dt=0.02, w_osc=w_osc, w_amp=w_amp)


def run(evals=1500, model_name='almeida', target='BMAL1', mode='instant', eps=0.3, seed=0,
        n_phase=12, n_dose=8, which='all', tag=None):
    from models import get_model
    from fit.cost import make_cost, RadialTarget
    from fit.doses import fit_dose_grid
    from fit.recover import displaced_truth

    model = get_model(model_name)
    doses, s_crit = fit_dose_grid(model_name, target, mode, 6.0, n_dose)
    C0 = make_cost(model, target, doses, RadialTarget(), n_phase=n_phase, mode=mode,
                   backend='diffrax', dt=0.02, w_osc=0.0, w_amp=0.0)
    v_true = displaced_truth(C0, eps, seed)
    zt, alive_t, _a = C0['surface'](v_true)
    C = build(model, target, mode, doses, n_phase, zt)
    v0 = C['v0']
    n = C['n_free']

    f_true = C['total'](v_true)
    f_nom = C['total'](v0)
    print(f"[benchmark] {model_name}/{target} {mode}, eps={eps}, {n_phase}x{n_dose}, "
          f"{n} free directions")
    print(f"[benchmark] cost at truth {f_true:.6f}, at nominal {f_nom:.6f}, "
          f"budget ~{evals} evals each", flush=True)

    # identified subspace at the truth, for the "is a low cost also the right model" column
    h = 1e-3
    def surf(v):
        zu, al, am = C['surface'](v)
        return np.concatenate([np.real(zu).ravel(), np.imag(zu).ravel()])
    J = np.zeros((surf(v_true).size, n))
    for i in range(n):
        vp = v_true.copy(); vp[i] += h
        vm = v_true.copy(); vm[i] -= h
        J[:, i] = (surf(vp) - surf(vm)) / (2 * h)
    _U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    k_det = int(np.sum(sv / sv[0] > 1e-2))
    print(f"[benchmark] {k_det} of {n} directions determined at the truth", flush=True)

    results = {}

    def record(name, r):
        dv = r['v'] - v_true
        c = Vt @ dv
        results[name] = dict(f=r['f'], nev=r.get('nev', -1), secs=r.get('seconds', np.nan),
                             v=r['v'], det=float(np.linalg.norm(c[:k_det])),
                             tot=float(np.linalg.norm(dv)), parts=r['parts'])
        print(f"  -> {name}: f={r['f']:.6f}  {r.get('nev', -1)} evals  "
              f"{r.get('seconds', float('nan')):.0f}s  "
              f"|err| det={results[name]['det']:.3f} total={results[name]['tot']:.3f}",
              flush=True)

    want = [w.strip() for w in which.split(',')] if which != 'all' else \
        ['cma-ipop', 'cma-anneal', 'bobyqa', 'lm', 'bh', 'lbfgs', 'multires']

    if 'cma-ipop' in want:
        print(f"\n=== cma-ipop (incumbent) ===", flush=True)
        # same restart structure as the anneal arm, so the ONLY difference between them is
        # where a restart starts from and what happens to sigma
        _ps = int(4 + 3 * np.log(n))
        _gpr = max(10, evals // (4 * _ps))
        record('cma-ipop', search.cma(C, bound=3.0, seed=seed, maxfev=evals, mode='ipop',
                                      restarts=3, gens_per_run=_gpr))
    if 'cma-anneal' in want:
        print(f"\n=== cma-anneal ===", flush=True)
        # gens_per_run is REQUIRED for this comparison to mean anything. `mode` only takes
        # effect on RESTARTS, and with maxfev=1500 at popsize 12 restart 0 consumes the entire
        # budget -- so the first version of this benchmark ran ipop and anneal as literally the
        # same computation and reported identical numbers for both.
        _ps = int(4 + 3 * np.log(n))
        _gpr = max(10, evals // (4 * _ps))                 # ~4 restarts inside the budget
        record('cma-anneal', search.cma(C, bound=3.0, seed=seed, maxfev=evals, mode='anneal',
                                        restarts=3, gens_per_run=_gpr))
    if 'bobyqa' in want:
        print("\n=== Py-BOBYQA (model-based trust region) ===", flush=True)
        record('bobyqa', search.bobyqa(C, v0, bound=3.0, maxfev=evals, seek_global=True))
    if 'lm' in want:
        print("\n=== Levenberg-Marquardt (residual vector) ===", flush=True)
        from fit.cost import make_cost as _mc, FixedTarget as _FT
        Cf = _mc(model, target, mode=mode, doses=doses, tgt=_FT(zt), n_phase=n_phase,
                 backend='diffrax', dt=0.02, grad_mode='fwd')
        record('lm', search.levenberg_marquardt(Cf, v0, bound=3.0,
                                                maxiter=max(4, evals // (2 * n))))
    if 'bh' in want:
        print("\n=== basin hopping ===", flush=True)
        record('bh', search.basin_hopping(C, v0, bound=3.0, n_hops=max(3, evals // 300),
                                          maxiter=25, local='lbfgs', seed=seed))
    if 'lbfgs' in want:
        print("\n=== L-BFGS ===", flush=True)
        record('lbfgs', search.lbfgs(C, v0, bound=3.0, maxiter=evals // 3, label='lbfgs'))
    if 'str' in want:
        print(f"\n=== smoothed trust region ===", flush=True)
        # 2n evals per iteration -> iterations sized to the same budget
        record('str', search.smoothed_trust_region(C, v0, bound=3.0,
                                                   maxiter=max(4, evals // (2 * n)), seed=seed))
    if 'multires' in want:
        print(f"\n=== multires (coarse -> fine) ===", flush=True)
        # the target is re-rendered from the KNOWN truth at each resolution
        sched = ((6, 5), (8, 6), (12, 8))
        per = max(200, evals // len(sched))

        def build_stage(nph, nd):
            dz, _s = fit_dose_grid(model_name, target, mode, 6.0, nd)
            Cs0 = make_cost(model, target, dz, RadialTarget(), n_phase=nph, mode=mode,
                            backend='diffrax', dt=0.02, w_osc=0.0, w_amp=0.0)
            zts, _al, _am = Cs0['surface'](v_true)
            return build(model, target, mode, dz, nph, zts)

        t0 = time.time()
        from fit.multires import run_schedule
        v_mr, stages = run_schedule(build_stage, v0, sched, optimizer='cma', maxfev=per,
                                    seed=seed)
        # score the multires answer on the SAME final grid as everyone else
        r = dict(v=v_mr, f=C['total'](v_mr), nev=int(sum(s['nev'] for s in stages)),
                 seconds=time.time() - t0, parts=C['parts'](v_mr))
        record('multires', r)

    print(f"\n{'=' * 78}\nBENCHMARK -- lower cost is better; floor is {f_true:.6f}\n{'=' * 78}")
    print(f"  {'method':14s} {'cost':>10s} {'vs incumbent':>13s} {'phase err':>10s} "
          f"{'evals':>7s} {'time':>7s} {'|err| det':>10s} {'|err| all':>10s}")
    base = results.get('cma-ipop', {}).get('f', np.nan)
    for k, r in sorted(results.items(), key=lambda kv: kv[1]['f']):
        x = float(np.clip(1 - 2 * r['parts']['c_ptc'], -1, 1))
        herr = float(np.arccos(x) / (2 * np.pi)) * 24
        rel = '' if not np.isfinite(base) else f"{(r['f'] / base - 1) * 100:+.1f}%"
        print(f"  {k:14s} {r['f']:10.6f} {rel:>13s} {herr:9.2f}h {r['nev']:7d} "
              f"{r['secs']:6.0f}s {r['det']:10.3f} {r['tot']:10.3f}")

    tag = paths.run_tag(tag)
    blob = dict(model=model_name, target=target, mode=mode, eps=eps, seed=seed,
                n_phase=n_phase, n_dose=n_dose, evals=evals, f_true=f_true, f_nom=f_nom,
                k_det=k_det, methods=np.array(list(results)), v_true=v_true)
    for k, r in results.items():
        for f in ('f', 'nev', 'secs', 'det', 'tot'):
            blob[f'{f}__{k}'] = np.asarray(r[f])
        blob[f'v__{k}'] = r['v']
    out = paths.out_path(model_name, 'fit_benchmark', f'bench_{target}_{mode}.npz', tag)
    paths.savez(out, **blob)
    print(f"\n[benchmark] -> {out}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description='optimizer head-to-head, matched budget')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    ap.add_argument('--mode', default='instant', choices=('pulse', 'instant'))
    ap.add_argument('--eps', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--evals', type=int, default=1500)
    ap.add_argument('--n-phase', type=int, default=12)
    ap.add_argument('--n-dose', type=int, default=8)
    ap.add_argument('--which', default='all')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    run(a.evals, a.model, a.target, a.mode, a.eps, a.seed, a.n_phase, a.n_dose, a.which, a.tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
