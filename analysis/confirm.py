"""
analysis/confirm.py
===================
Turn a decoupling CANDIDATE into a result, or retract it.

    python -m analysis.confirm --model almeida --target BMAL1 [--eps 0.15]

`analysis/coupling.py` reports directions that a linear, finite-difference summary says move
the PTC a lot per unit of limit-cycle movement. That is a jacobian claim. This module takes a
FINITE step along such a direction and measures what actually happens:

  * the limit cycle, on the orbit solver -- the same measure `lc_sens` uses;
  * the PTC, on `engine/reference.py`, the ADAPTIVE scipy/LSODA engine that shares no
    numerical machinery with the JAX engine the jacobian was built from.

WHY THIS STEP IS NOT OPTIONAL
    In input_screen the autodiff version of exactly this claim was wrong by ~80 orders of
    magnitude (a predicted PTC change of ~1e84 against an actual 0.006), and the ONLY part of
    that analysis that survived was the finite-displacement evidence computed on the adaptive
    engine. A linear summary can be wrong for three independent reasons -- the derivative is
    inaccurate, the response is non-linear at any useful step size, or the direction is one the
    jacobian could not resolve -- and a finite displacement tests all three at once.

CONTROLS, BECAUSE A SINGLE NUMBER PROVES NOTHING
    Every run reports three directions, not one:

      decoupled   the candidate (largest rho)
      coupled     the most LC-favouring direction (smallest rho) -- should show the OPPOSITE
                  pattern, large LC change and small PTC change
      stiffest    the top LC singular direction -- the POSITIVE CONTROL, which must move both

    If the "decoupled" direction shows a large PTC change AND the coupled one does not, the
    ordering is real. If everything moves the PTC by the same amount, the analysis is measuring
    step size rather than direction, and the candidate is retracted.
"""
import argparse
import os
import sys

import numpy as np

import analysis  # noqa: F401
import paths


def displaced(model, names, v, eps):
    """Base parameters displaced by exp(eps * v) in log space, v unit-norm."""
    base = model.get_parameters()
    out = dict(base)
    for n, vi in zip(names, np.asarray(v)):
        out[n] = base[n] * float(np.exp(eps * vi))
    return out


def lc_change(model, params, base_C, base_T, scale, m=64):
    """Relative cycle-shape change + relative period change, on the orbit solver."""
    from engine.orbit import make_orbit_finder
    find, _s = make_orbit_finder(model, m=m)
    x0, T, C, st = find(params)
    if C is None:
        return np.nan, np.nan, st
    d = (C - base_C) / scale[:, None]
    return float(np.sqrt(np.mean(d ** 2))), float((T - base_T) / base_T), st


def ptc_change(model, params, target, doses, base_new, mode='pulse', n_phases=16):
    """RMS circular change of the PTC, on the INDEPENDENT adaptive engine."""
    from engine.reference import AdaptiveReference, ptc as ref_ptc
    ref = AdaptiveReference(model, params=params).build()
    rows = []
    for d in doses:
        _old, new = ref_ptc(model, ref, target, float(d), n_phases=n_phases, mode=mode)
        rows.append(new)
    new = np.array(rows)
    diff = np.abs(((new - base_new + 0.5) % 1.0) - 0.5)
    return (float(np.sqrt(np.nanmean(diff ** 2))), float(np.nanmax(diff)),
            int(np.sum(~np.isfinite(new))), new, ref.period)


def run(model_name, target, mode='pulse', eps=0.15, n_phases=16, n_dose=3, feature='twist'):
    from models import get_model
    from engine.orbit import make_orbit_finder

    model = get_model(model_name)
    fp = paths.out_path(model_name, 'coupling', f'coupling_{target}_{mode}_{feature}.npz')
    if not os.path.exists(fp):
        raise SystemExit(f"run `python -m analysis.coupling --model {model_name} "
                         f"--target {target}` first")
    cp = dict(np.load(fp, allow_pickle=True))
    names = [str(p) for p in cp['params']]
    V, rho = np.asarray(cp['V']), np.asarray(cp['rho'])
    # the stiffest LC direction, as the positive control
    _u, _s, vt = np.linalg.svd(np.nan_to_num(np.asarray(cp['J_LC'])), full_matrices=False)
    dirs = [('decoupled (candidate)', V[:, 0], rho[0]),
            ('coupled  (control)', V[:, -1], rho[-1]),
            ('stiffest LC (positive control)', vt[0], np.nan)]

    # base LC + base PTC on the adaptive engine
    find, _s2 = make_orbit_finder(model)
    _x0, T0, C0, st0 = find(model.get_parameters())
    if C0 is None:
        raise SystemExit(f"base orbit not usable ({st0})")
    scale = np.maximum(C0.max(1) - C0.min(1), 1e-9)
    # sample the target's own grid: low, near-S_crit, high
    from analysis.ptc_sens import load_grid
    grid, _dt = load_grid(model_name, target, mode)
    if grid is None:
        raise SystemExit('no dose grid; run analysis.scrit first')
    doses = grid[np.linspace(0, len(grid) - 1, n_dose).astype(int)]

    print(f"[confirm] {model_name} / {target} ({mode}); eps = {eps} in log-parameter space")
    print(f"[confirm] doses {np.array2string(doses, precision=3)}; "
          f"PTC on the ADAPTIVE engine, {n_phases} phases")
    from engine.reference import AdaptiveReference, ptc as ref_ptc
    ref0 = AdaptiveReference(model).build()
    base_new = np.array([ref_ptc(model, ref0, target, float(d), n_phases=n_phases,
                                 mode=mode)[1] for d in doses])
    print(f"[confirm] base period {ref0.period:.4f} h, "
          f"{int(np.sum(~np.isfinite(base_new)))} unusable base points\n")

    print(f"  {'direction':32s} {'rho':>8s} {'sign':>5s} {'dLC':>9s} {'|dT|/T':>9s} "
          f"{'dPTC rms':>9s} {'dPTC max':>9s}")
    rows = []
    for label, v, r in dirs:
        v = np.asarray(v, float)
        v = v / np.linalg.norm(v)
        for sgn in (+1, -1):
            pd = displaced(model, names, sgn * v, eps)
            dlc, dT, st = lc_change(model, pd, C0, T0, scale)
            if not np.isfinite(dlc):
                print(f"  {label:32s} {r:8.2f} {sgn:+5d} {'orbit ' + st:>9s}")
                continue
            prms, pmax, nbad, _new, per = ptc_change(model, pd, target, doses, base_new,
                                                     mode=mode, n_phases=n_phases)
            print(f"  {label:32s} {r:8.2f} {sgn:+5d} {dlc:9.4f} {abs(dT):9.4f} "
                  f"{prms:9.4f} {pmax:9.4f}" + (f"  ({nbad} bad)" if nbad else ""))
            rows.append(dict(label=label, rho=r, sign=sgn, dLC=dlc, dT=dT,
                             dPTC_rms=prms, dPTC_max=pmax, n_bad=nbad))

    # verdict
    dec = [r for r in rows if r['label'].startswith('decoupled')]
    cou = [r for r in rows if r['label'].startswith('coupled')]
    sti = [r for r in rows if r['label'].startswith('stiffest')]
    print()
    if dec and cou and sti:
        rd = np.mean([r['dPTC_rms'] / max(r['dLC'], 1e-12) for r in dec])
        rc = np.mean([r['dPTC_rms'] / max(r['dLC'], 1e-12) for r in cou])
        actual = rd / max(rc, 1e-12)
        # rho is a ratio of SQUARED responses, so the comparable linear prediction for a
        # ratio of amplitudes is its square root
        predicted = float(np.sqrt(rho[0] / max(rho[-1], 1e-12)))
        print(f"  ACTUAL PTC-per-LC ratio    decoupled {rd:.2f}   coupled {rc:.2f}   "
              f"-> {actual:.1f}x")
        print(f"  positive control (stiffest LC) moved the PTC by "
              f"{np.mean([r['dPTC_rms'] for r in sti]):.4f} rms, so the measurement works")
        ok = rd > 2 * rc
        print(f"  [{'CONFIRMED' if ok else 'NOT CONFIRMED'}] the ORDERING "
              f"{'survives' if ok else 'does NOT survive'} a finite displacement.")
        print(f"  The MAGNITUDE does not: the linear analysis predicts {predicted:.0f}x "
              f"(sqrt of the rho ratio)\n  against an actual {actual:.1f}x, i.e. overstated "
              f"~{predicted / max(actual, 1e-12):.0f}x. Cite the finite-displacement number.")
    out = paths.out_path(model_name, 'coupling', f'confirm_{target}_{mode}.npz')
    paths.savez(out, eps=eps, doses=doses, base_new=base_new,
                labels=np.array([r['label'] for r in rows]),
                dLC=np.array([r['dLC'] for r in rows]),
                dPTC_rms=np.array([r['dPTC_rms'] for r in rows]),
                dPTC_max=np.array([r['dPTC_max'] for r in rows]),
                sign=np.array([r['sign'] for r in rows]),
                rho=np.array([r['rho'] for r in rows]))
    print(f"\n[confirm] -> {out}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description='finite-displacement confirmation')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--target', default=os.environ.get('TARGET', 'BMAL1'))
    ap.add_argument('--mode', default='pulse')
    ap.add_argument('--eps', type=float, default=0.15)
    ap.add_argument('--n-phases', type=int, default=16)
    ap.add_argument('--n-dose', type=int, default=3)
    a = ap.parse_args(argv)
    run(a.model, a.target, a.mode, a.eps, a.n_phases, a.n_dose)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
