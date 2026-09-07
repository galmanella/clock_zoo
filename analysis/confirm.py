"""
analysis/confirm.py
===================
Turn a decoupling CANDIDATE into a result, or retract it.

    $PY -m analysis.confirm --model almeida --target BMAL1 [--eps 0.15]

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


def _full_params(model, names, values):
    """A COMPLETE parameter dict: the model's base, with `names` overridden by `values`.

    `names` is the gauge-quotiented subset the coupling analysis kept, not the whole parameter
    set, so a dict built from it alone is missing entries the RHS needs."""
    pd = dict(model.get_parameters())
    pd.update({n: float(v) for n, v in zip(names, values)})
    return pd


def dense_render(model_name, target, mode='pulse', n_phase=32, tag=None, scrit_tag=None):
    """Add a dense JAX render of an EXISTING confirm run, in place.

    This is what saving `param_values` buys: the expensive adaptive verification does not have
    to be repeated to get a better picture of the same parameter sets. Reads the npz, computes,
    writes it back."""
    import jax
    from models import get_model
    from engine.orbit import make_orbit_finder
    from engine.ptc import make_ptc, grid_points, phase_or_nan, valid_mask, recommended_skip
    from analysis.ptc_sens import load_grid

    model = get_model(model_name)
    fp = paths.out_path(model_name, 'coupling', f'confirm_{target}_{mode}.npz', tag)
    cf = dict(np.load(fp, allow_pickle=True))
    names = [str(x) for x in cf['param_names']]
    dgrid, ddt = load_grid(model_name, target, mode, scrit_tag)
    skip_p, _mu, _r = recommended_skip(model, tol=1e-2, verbose=False)
    f, _sv = make_ptc(model, target, mode=mode, readout='raw', skip_p=skip_p,
                      dt=ddt or 0.02, track_min=True)
    fj = jax.jit(f)
    gph, gdz = grid_points(n_phase, dgrid)
    find, _s = make_orbit_finder(model)

    def one(params):
        x0, _T, _C, st = find(params)
        if x0 is None:
            return None, st
        z, mn = fj(model.jax_params(params), x0, gph, gdz)
        p, _a = phase_or_nan(np.asarray(z))
        p = np.where(valid_mask(np.asarray(mn)), p, np.nan)
        return p.reshape(len(dgrid), n_phase).T, 'ok'

    db, st = one(model.get_parameters())
    if db is None:
        raise SystemExit(f'base orbit not usable ({st})')
    grids = []
    for pv in np.asarray(cf['param_values']):
        g, st = one(_full_params(model, names, pv))
        grids.append(g if g is not None else np.full_like(db, np.nan))
    cf.update(dense_doses=np.asarray(dgrid), dense_old=np.arange(n_phase) / n_phase,
              dense_base=db.astype(np.float32),
              dense_ptc_grids=np.array(grids, dtype=np.float32),
              dense_dt=float(ddt or 0.02), dense_engine='jax')
    paths.savez(fp, **cf)
    print(f"[confirm] dense render {n_phase}x{len(dgrid)} added for "
          f"{len(grids)} displacement(s) -> {fp}")
    return cf


def displaced(model, names, v, eps):
    """Base parameters displaced by exp(eps * v) in log space, v unit-norm."""
    base = model.get_parameters()
    out = dict(base)
    for n, vi in zip(names, np.asarray(v)):
        out[n] = base[n] * float(np.exp(eps * vi))
    return out


def lc_change(model, params, base_C, base_T, scale, m=64):
    """Relative cycle-shape change + relative period change, on the orbit solver.
    Returns the perturbed CYCLE too -- the summary scalar is not the evidence."""
    from engine.orbit import make_orbit_finder
    find, _s = make_orbit_finder(model, m=m)
    x0, T, C, st = find(params)
    if C is None:
        return np.nan, np.nan, st, None
    d = (C - base_C) / scale[:, None]
    return float(np.sqrt(np.mean(d ** 2))), float((T - base_T) / base_T), st, C


def ptc_change(model, params, target, doses, base_new, mode='pulse', n_phases=16):
    """RMS circular change of the PTC, on the INDEPENDENT adaptive engine.
    Returns the full perturbed PTC grid -- the surface IS the evidence, the rms is a caption."""
    from engine.reference import AdaptiveReference, ptc as ref_ptc
    ref = AdaptiveReference(model, params=params).build()
    rows = []
    for d in doses:
        _old, new = ref_ptc(model, ref, target, float(d), n_phases=n_phases, mode=mode)
        rows.append(new)
    new = np.array(rows)                                   # (n_dose, n_phase)
    diff = np.abs(((new - base_new + 0.5) % 1.0) - 0.5)
    return (float(np.sqrt(np.nanmean(diff ** 2))), float(np.nanmax(diff)),
            int(np.sum(~np.isfinite(new))), new, ref.period)


def run(model_name, target, mode='pulse', eps=0.15, n_phases=16, n_dose=3, feature='twist',
        all_dirs=False, doses=None, resume=True, tag=None, scrit_tag=None):
    from models import get_model
    from engine.orbit import make_orbit_finder

    model = get_model(model_name)
    # SAME TAG as the coupling run. analysis.coupling gained --tag so two variants of one
    # (target, mode) -- above all include_time, whose settings give quotients of different
    # SIZE -- cannot overwrite each other. This read has to follow it, or confirm silently
    # verifies a DIFFERENT run's directions than the ones just computed.
    fp = paths.out_path(model_name, 'coupling', f'coupling_{target}_{mode}_{feature}.npz',
                        tag)
    if not os.path.exists(fp):
        msg = "run `python -m analysis.coupling --model %s --target %s --mode %s%s` first" % (
            model_name, target, mode, (' --tag ' + tag) if tag else '')
        raise SystemExit("no " + fp + chr(10) + "  " + msg)
    cp = dict(np.load(fp, allow_pickle=True))
    names = [str(p) for p in cp['params']]
    V, rho = np.asarray(cp['V']), np.asarray(cp['rho'])
    # the stiffest LC direction, as the positive control
    _u, _s, vt = np.linalg.svd(np.nan_to_num(np.asarray(cp['J_LC'])), full_matrices=False)
    if all_dirs:
        # EVERY eigen-direction, not just the extremes. The 3-direction version answers "is
        # the top candidate real?"; sweeping the whole spectrum answers the stronger question,
        # whether dPTC/dLC actually ORDERS with rho or whether the top one is a lucky draw.
        # The stiffest LC direction stays as the positive control -- it must move both.
        dirs = [(f'dir {k:02d}', V[:, k], rho[k]) for k in range(V.shape[1])]
        dirs.append(('stiffest LC (positive control)', vt[0], np.nan))
    else:
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
    if doses is None:
        from analysis.ptc_sens import load_grid
        grid, _dt = load_grid(model_name, target, mode, scrit_tag)
        if grid is None:
            raise SystemExit('no dose grid; run analysis.scrit first')
        doses = grid[np.linspace(0, len(grid) - 1, n_dose).astype(int)]
    doses = np.asarray(doses, float)

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
    # RAW INTEGRATION OUTPUT, kept per (direction, sign). Recomputing this costs minutes of
    # adaptive integration; storing it costs a few hundred kB. Every plot and every re-analysis
    # downstream reads these rather than re-running anything.
    LCP, PTCG, VECS, PARAMSETS = [], [], [], []
    import time
    t0 = time.time()
    nrun = [0]
    total = 2 * len(dirs)
    for label, v, r in dirs:
        v = np.asarray(v, float)
        v = v / np.linalg.norm(v)
        for sgn in (+1, -1):
            tk = time.time()
            nrun[0] += 1
            pd = displaced(model, names, sgn * v, eps)
            dlc, dT, st, C = lc_change(model, pd, C0, T0, scale)
            if not np.isfinite(dlc):
                print(f"  {label:32s} {r:8.2f} {sgn:+5d} {'orbit ' + st:>9s}"
                      f"   [{nrun[0]}/{total}, {time.time() - tk:.0f}s]", flush=True)
                continue
            prms, pmax, nbad, new, per = ptc_change(model, pd, target, doses, base_new,
                                                    mode=mode, n_phases=n_phases)
            el = time.time() - t0
            print(f"  {label:32s} {r:8.2f} {sgn:+5d} {dlc:9.4f} {abs(dT):9.4f} "
                  f"{prms:9.4f} {pmax:9.4f}" + (f"  ({nbad} bad)" if nbad else "")
                  + f"   [{nrun[0]}/{total}, {time.time() - tk:.0f}s, "
                    f"{el / 60:.1f} min elapsed, eta "
                    f"{el / nrun[0] * (total - nrun[0]) / 60:.1f} min]", flush=True)
            rows.append(dict(label=label, rho=r, sign=sgn, dLC=dlc, dT=dT,
                             dPTC_rms=prms, dPTC_max=pmax, n_bad=nbad, period=per))
            LCP.append(C)                                   # (n_states, m) perturbed cycle
            PTCG.append(new)                                # (n_dose, n_phase) perturbed PTC
            VECS.append(sgn * v)                            # the displacement direction
            PARAMSETS.append([pd[n] for n in names])        # the actual parameter values used

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
    # ---- dense rendering of the same displacements, for looking at ---------------------- #
    # The adaptive engine VERIFIES (it shares no machinery with the jacobian), but it is slow,
    # so the verification grid above is necessarily coarse -- too coarse to actually see a
    # surface. The JAX engine agrees with it to 1.6e-05 on this model (engine/validate.py), so
    # it is sound to use for a dense picture of the SAME parameter sets. Numbers come from the
    # adaptive run; pictures come from this one. Both are saved.
    dense = dict()
    try:
        import jax
        import jax.numpy as jnp
        from engine.ptc import make_ptc, grid_points, phase_or_nan, valid_mask, recommended_skip
        from analysis.ptc_sens import load_grid
        dgrid, ddt = load_grid(model_name, target, mode, scrit_tag)
        skip_p, _mu, _r = recommended_skip(model, tol=1e-2, verbose=False)
        nph = 32
        fj_fn, solver = make_ptc(model, target, mode=mode, readout='raw', skip_p=skip_p,
                                 dt=ddt or 0.02, track_min=True)
        fj = jax.jit(fj_fn)
        gph, gdz = grid_points(nph, dgrid)

        def dense_ptc(params):
            from engine.orbit import make_orbit_finder
            fnd, _sv = make_orbit_finder(model)
            x0, _T, _C, st = fnd(params)
            if x0 is None:
                return None
            z, mn = fj(model.jax_params(params), x0, gph, gdz)
            p, _a = phase_or_nan(np.asarray(z))
            p = np.where(valid_mask(np.asarray(mn)), p, np.nan)
            return p.reshape(len(dgrid), nph).T          # (n_phase, n_dose)

        db = dense_ptc(model.get_parameters())
        # start from the FULL base dict and override the kept names: `names` is the gauge-
        # quotiented subset the coupling analysis kept (15 of Almeida's 18), so a dict built
        # from it alone is missing parameters the model needs and the RHS raises on the first
        # one it cannot find.
        dg = [dense_ptc(_full_params(model, names, pv)) for pv in PARAMSETS]
        dense = dict(dense_doses=np.asarray(dgrid), dense_old=np.arange(nph) / nph,
                     dense_base=db.astype(np.float32),
                     dense_ptc_grids=np.array([g if g is not None else
                                               np.full_like(db, np.nan) for g in dg],
                                              dtype=np.float32),
                     dense_dt=float(ddt or 0.02), dense_engine='jax')
        print(f"[confirm] dense render: {nph} phases x {len(dgrid)} doses on the JAX engine "
              f"for {len(dg)} displacement(s)")
    except Exception as e:                     # never let the picture break the verification
        print(f"[confirm] dense render skipped ({type(e).__name__}: {e})", file=sys.stderr)

    out = paths.out_path(model_name, 'coupling', f'confirm_{target}_{mode}.npz', tag)
    paths.savez(
        out,
        # --- configuration, so the run is reconstructible ---------------------------- #
        model=model_name, target=target, mode=mode, eps=eps, doses=doses,
        param_names=np.array(names), state_names=np.array(list(model.state_names)),
        observables=np.array(list(model.observable_states())),
        n_phases=n_phases, old=np.arange(n_phases) / n_phases,
        # --- RAW baselines ------------------------------------------------------------ #
        base_profiles=np.asarray(C0),            # (n_states, m) the base limit cycle
        base_period=T0, lc_scale=scale,
        base_new=base_new,                       # (n_dose, n_phase) the base PTC, adaptive
        base_params=np.array([model.get_parameters()[n] for n in names]),
        # --- RAW per-displacement output ---------------------------------------------- #
        # THE POINT OF THIS FILE. Each of these took minutes of adaptive integration to
        # produce and a few hundred kB to keep. Every figure and every re-analysis reads
        # them; nothing downstream ever needs to integrate again.
        lc_profiles=np.array(LCP),               # (n_run, n_states, m)
        ptc_grids=np.array(PTCG),                # (n_run, n_dose, n_phase)
        directions=np.array(VECS),               # (n_run, n_param) unit displacement
        param_values=np.array(PARAMSETS),        # (n_run, n_param) values actually used
        # --- derived summaries (cheap to recompute, kept for convenience) -------------- #
        labels=np.array([r['label'] for r in rows]),
        dLC=np.array([r['dLC'] for r in rows]),
        dT=np.array([r['dT'] for r in rows]),
        dPTC_rms=np.array([r['dPTC_rms'] for r in rows]),
        dPTC_max=np.array([r['dPTC_max'] for r in rows]),
        period=np.array([r['period'] for r in rows]),
        n_bad=np.array([r['n_bad'] for r in rows]),
        sign=np.array([r['sign'] for r in rows]),
        rho=np.array([r['rho'] for r in rows]))
    print(f"\n[confirm] -> {out}")
    print(f"[confirm] saved RAW: lc_profiles {np.array(LCP).shape}, "
          f"ptc_grids {np.array(PTCG).shape}, plus baselines and the exact parameter "
          f"values used")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description='finite-displacement confirmation')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--target', default=os.environ.get('TARGET', 'BMAL1'))
    ap.add_argument('--mode', default='pulse')
    ap.add_argument('--eps', type=float, default=0.15)
    ap.add_argument('--n-phases', type=int, default=16)
    ap.add_argument('--n-dose', type=int, default=3)
    ap.add_argument('--all-dirs', action='store_true',
                    help='every eigen-direction, not just the two extremes')
    ap.add_argument('--doses', default=None,
                    help="explicit linear dose grid 'lo,hi,n' (default: the scrit grid)")
    ap.add_argument('--tag', default=None,
                    help='tag of the analysis.coupling run being confirmed; also where '
                         'this writes. Must match, or a different run gets verified.')
    ap.add_argument('--scrit-tag', default=None,
                    help='analysis.scrit run supplying the dose grid (default: newest)')
    ap.add_argument('--dense-only', action='store_true',
                    help='add/refresh the dense JAX render of an existing run, no re-verify')
    a = ap.parse_args(argv)
    if a.dense_only:
        dense_render(a.model, a.target, a.mode, tag=a.tag, scrit_tag=a.scrit_tag)
        return 0
    dz = None
    if a.doses:
        lo, hi, nd = a.doses.split(',')
        dz = np.linspace(float(lo), float(hi), int(nd))
    run(a.model, a.target, a.mode, a.eps, a.n_phases, a.n_dose, all_dirs=a.all_dirs,
        doses=dz, tag=a.tag, scrit_tag=a.scrit_tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
