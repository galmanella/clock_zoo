"""
analysis/minimum.py
===================
When a fit stops away from the truth, WHY did it stop? Two very different answers:

    a TRUE LOCAL MINIMUM   the cost rises in every direction. A local optimizer is then
                           behaving CORRECTLY -- it cannot be expected to escape a genuine
                           minimum just because a better one exists elsewhere. Blaming the
                           optimizer is a category error; the landscape is multimodal and a
                           method that escapes shallow minima is required.

    a STALL                the cost still descends somewhere nearby, but the search stopped:
                           line-search collapse, a corrupted quasi-Newton Hessian, a
                           convergence test tripped by a zeroed gradient. That IS an optimizer
                           defect and is fixable.

`fit/recover.py` reported both L-BFGS and CMA-ES as "NOT RECOVERED" without distinguishing
these, which is not a conclusion about anything.

    $PY -m analysis.minimum --tags T1d,T1cma

TWO PROBES
    1. RANDOM. Sample directions around the stopping point at several radii and count how many
       descend. Zero descent directions at small radius is the signature of a genuine minimum.
    2. TOWARD THE TRUTH. Evaluate the cost along the straight segment from the stopping point to
       the known truth. This is the decisive one, and it is only available because the target is
       synthetic:

         cost rises then falls  -> a BARRIER separates the two. Genuinely distinct basins, and
                                   no local method could have crossed it.
         cost falls monotonically -> there was a downhill path the whole way and the optimizer
                                   simply stopped. A stall.

IS THE MINIMUM ANY GOOD?
    Separately from why it stopped: a local minimum can still be scientifically acceptable. The
    per-cell residual translates back into a phase error via

        c_ptc = mean (1 - cos 2*pi*d) / 2   =>   typical |d| = arccos(1 - 2*c_ptc) / 2*pi

    so the report converts each cost into cycles and hours. That is the number to compare
    against experimental phase precision: with real data the floor is set by measurement noise,
    not by zero, and any parameter set inside that band is data-consistent no matter how far it
    sits from the "true" one.
"""
import argparse
import glob
import os

import numpy as np

import analysis  # noqa: F401
import paths


def phase_error(c_ptc, period_h=24.0):
    """(cycles, hours) of typical per-cell phase error corresponding to a pointwise cost."""
    x = float(np.clip(1.0 - 2.0 * c_ptc, -1.0, 1.0))
    cyc = float(np.arccos(x) / (2 * np.pi))
    return cyc, cyc * period_h


def probe(cost, v, n_dir=24, radii=(0.01, 0.05, 0.15, 0.4), seed=0, verbose=True):
    """Random directions around `v`: how many descend, and how far down?"""
    rng = np.random.default_rng(seed)
    f0 = cost['total'](v)
    n = len(v)
    out = []
    for r in radii:
        best, n_down = f0, 0
        for _ in range(n_dir):
            d = rng.normal(size=n)
            d /= np.linalg.norm(d)
            f = cost['total'](v + r * d)
            if np.isfinite(f) and f < f0:
                n_down += 1
                best = min(best, f)
        out.append(dict(radius=r, n_down=n_down, n_dir=n_dir, best=best, f0=f0))
        if verbose:
            print(f"    radius {r:5.2f}: {n_down:3d}/{n_dir} descend, best {best:.6f} "
                  f"({'DOWN ' + f'{f0 - best:.2e}' if n_down else 'none'})", flush=True)
    return out


def segment(cost, v_from, v_to, n=21, verbose=True):
    """Cost along the straight line from the stopping point to the truth."""
    ts = np.linspace(0.0, 1.0, n)
    fs = np.array([cost['total'](v_from + t * (v_to - v_from)) for t in ts])
    f0, f1 = fs[0], fs[-1]
    peak = float(np.max(fs))
    barrier = peak - f0
    if verbose:
        print(f"    {'t':>5s} {'cost':>12s}")
        for t, f in zip(ts, fs):
            bar = '#' * int(np.clip(60 * (f - fs.min()) / max(peak - fs.min(), 1e-12), 0, 60))
            print(f"    {t:5.2f} {f:12.6f} {bar}", flush=True)
    return ts, fs, float(barrier)


def run(tags=('T1d', 'T1cma'), model_name='almeida', target='BMAL1', mode='instant',
        n_phase=12, n_dose=8, seed=0):
    from models import get_model
    from fit.cost import make_cost, FixedTarget, RadialTarget
    from fit.doses import fit_dose_grid

    model = get_model(model_name)
    doses, s_crit = fit_dose_grid(model_name, target, mode, 6.0, n_dose)
    C0 = make_cost(model, target, doses, RadialTarget(), n_phase=n_phase, mode=mode,
                   backend='diffrax', dt=0.02, w_osc=0.0, w_amp=0.0)

    results = {}
    for tg in tags:
        d = paths.out_dir(model_name, 'fit_recover', tg, create=False)
        fs = sorted(glob.glob(os.path.join(d, '*.npz')))
        if not fs:
            print(f"  {tg}: no run found"); continue
        z = np.load(fs[-1], allow_pickle=True)
        v_fit, v_true = np.asarray(z['v_fit']), np.asarray(z['v_true'])
        zt, alive_t, _a = C0['surface'](v_true)
        C = make_cost(model, target, doses, FixedTarget(zt), n_phase=n_phase, mode=mode,
                      backend='diffrax', dt=0.02)

        f_fit = C['total'](v_fit)
        f_true = C['total'](v_true)
        p_fit = C['parts'](v_fit)
        g = C['grad'](v_fit)
        gn = float(np.linalg.norm(g))
        cyc, hrs = phase_error(p_fit['c_ptc'])

        print(f"\n{'=' * 78}\n{tg}: is the stopping point a real minimum?\n{'=' * 78}")
        print(f"  cost here {f_fit:.6f}   at truth {f_true:.6f}   "
              f"|v_fit - v_true| = {np.linalg.norm(v_fit - v_true):.3f}")
        print(f"  |grad| here = {gn:.3e}")
        print(f"  residual c_ptc = {p_fit['c_ptc']:.4f}  ->  typical phase error "
              f"{cyc:.4f} cyc = {hrs:.2f} h on a 24 h clock")
        print(f"\n  [1] random directions:")
        pr = probe(C, v_fit, seed=seed)
        print(f"\n  [2] straight line toward the truth (t=0 here, t=1 truth):")
        ts, fs_seg, barrier = segment(C, v_fit, v_true)

        n_down_small = pr[0]['n_down'] + pr[1]['n_down']
        verdict = ('STALL -- descent directions exist nearby; the search stopped early'
                   if n_down_small > 0 else
                   'TRUE LOCAL MINIMUM -- no descent found nearby')
        sep = ('BARRIER of +%.4f between here and the truth: genuinely distinct basins, and no '
               'local method could cross it' % barrier if barrier > 1e-4 else
               'NO barrier on the direct line -- the cost descends monotonically toward the '
               'truth, so the stop was premature')
        print(f"\n  VERDICT: {verdict}")
        print(f"  SEPARATION: {sep}")
        results[tg] = dict(f_fit=f_fit, f_true=f_true, grad=gn, probe=pr, ts=ts,
                           seg=fs_seg, barrier=barrier, cyc=cyc, hrs=hrs,
                           v_fit=v_fit, v_true=v_true)

    if results:
        tag = paths.run_tag(None)
        blob = {'model': model_name, 'target': target, 'mode': mode,
                'tags': np.array(list(results))}
        for k, r in results.items():
            for f in ('f_fit', 'f_true', 'grad', 'barrier', 'cyc', 'hrs'):
                blob[f'{f}__{k}'] = np.asarray(r[f])
            blob[f'seg__{k}'] = r['seg']
            blob[f'ts__{k}'] = r['ts']
            blob[f'v_fit__{k}'] = r['v_fit']
        out = paths.out_path(model_name, 'minimum', f'minimum_{target}_{mode}.npz', tag)
        paths.savez(out, **blob)
        print(f"\n[minimum] -> {out}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description='local minimum or stall?')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    ap.add_argument('--mode', default='instant', choices=('pulse', 'instant'))
    ap.add_argument('--tags', default='T1d,T1cma')
    ap.add_argument('--n-phase', type=int, default=12)
    ap.add_argument('--n-dose', type=int, default=8)
    a = ap.parse_args(argv)
    run(tuple(a.tags.split(',')), a.model, a.target, a.mode, a.n_phase, a.n_dose)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
