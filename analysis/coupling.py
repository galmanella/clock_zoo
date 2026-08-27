"""
analysis/coupling.py
====================
OBJECTIVE (b), part 3: HOW COUPLED are the limit cycle and the PTC, locally at base?

    python -m analysis.coupling --model almeida --target BMAL1

Pure read: it consumes the npz files written by `analysis.lc_sens` and `analysis.ptc_sens`
and adds no new integration. Three levels, cheapest first.

1. PER-PARAMETER SCATTER
   LC sensitivity (x) against PTC sensitivity (y), one point per parameter. The question is
   which QUADRANT is populated:

       high LC / high PTC   the boring, expected correlation -- move one, move the other.
       high LC / low  PTC   moves the cycle without moving the response. Decoupled, but the
                            USELESS direction: it does not let PTC data add anything.
       low  LC / high PTC   THE prize. A parameter the limit cycle barely sees but the PTC
                            does is a direction where PTC data adds identifiability that LC
                            data cannot.
       low  LC / low  PTC   sloppy in both; invisible to either experiment.

   On Mirsky that top-left corner was EMPTY, and the only decoupling found was the useless
   kind (activators moved the LC without moving the PTC). Whether a smaller model opens it is
   the whole reason this repo exists.

2. LOCAL COUPLING MATRIX
   Stack the per-factor responses into J_LC and J_PTC (settings x parameters, in log-parameter
   space), PROJECT OUT THE GAUGE, and compare their row spaces by PRINCIPAL ANGLES. Small
   angles mean the two measurements constrain the same combinations; a large angle is a
   direction one sees and the other does not. Also reports, for each right-singular direction
   of J_LC ordered by LC-sensitivity, how strongly J_PTC responds -- the sloppy-tail question.

   THE GAUGE PROJECTION IS NOT OPTIONAL. A gauge direction is exactly LC-null by construction
   while moving a fixed-absolute-dose PTC (the dose axis rescales), so leaving it in
   manufactures precisely the "low LC / high PTC" signal this analysis is looking for. In
   input_screen that was a real, published-adjacent false positive. Almeida has 2 such
   directions of 18, Korencic 1 of 34, Goldbeter 5 of 52.

3. WHAT TO CONFIRM NEXT
   Any direction that looks decoupled here is a CANDIDATE, not a result. The linear-algebra
   step is built from finite differences, but it is still a linear summary; the claim only
   becomes trustworthy after a finite displacement along that direction is pushed through the
   independent adaptive engine (engine/reference.py). This module prints the shortlist to
   confirm; it does not confirm it for you.

READ THE TWIST, NOT THE SINGULARITY
   The singularity location is a topological defect: discontinuous and grid-quantised. Twist
   (the fixed-point-vs-dose curve) is smooth AND gauge-invariant, so it is the default PTC
   feature here. The others are available via --feature.
"""
import argparse
import glob
import os
import sys

import numpy as np

import analysis  # noqa: F401
import paths

FEATURES = ('twist', 'pointwise', 'dS', 'dphi')


def _load(model_name, kind, pattern, tag=None):
    tag = tag or paths.latest_run(model_name, kind)
    if tag is None:
        raise SystemExit(f"no {kind} output for {model_name}; run `python -m analysis.{kind}`")
    d = paths.out_dir(model_name, kind, tag, create=False)
    files = sorted(glob.glob(os.path.join(d, pattern)))
    merged = [f for f in files if 'merged' in f]
    if merged:
        return dict(np.load(merged[0], allow_pickle=True)), tag
    if not files:
        raise SystemExit(f"no files matching {pattern} in {d}")
    if len(files) == 1:
        return dict(np.load(files[0], allow_pickle=True)), tag
    raise SystemExit(f"{len(files)} shards in {d}; run with --merge first")


def align(lc, pt):
    """Common parameter ordering between the two sweeps."""
    lp = [str(x) for x in lc['params']]
    pp = [str(x) for x in pt['params']]
    common = [p for p in lp if p in set(pp)]
    li = [lp.index(p) for p in common]
    pi = [pp.index(p) for p in common]
    return common, np.array(li), np.array(pi)


def principal_angles(A, B, tol=1e-10):
    """Principal angles (degrees) between the ROW spaces of A and B, ascending.

    Computed from the singular values of Qa^T Qb with Qa, Qb orthonormal bases of the two row
    spaces. 0 deg = a shared direction; 90 deg = a direction in one space orthogonal to the
    whole of the other."""
    def basis(M):
        M = np.nan_to_num(np.asarray(M, float))
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        r = int(np.sum(s > tol * (s.max() if s.size else 1)))
        return vt[:r].T                                     # columns span the row space
    Qa, Qb = basis(A), basis(B)
    if Qa.size == 0 or Qb.size == 0:
        return np.array([]), Qa, Qb
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1, 1))), Qa, Qb


def run(model_name, target, mode='pulse', feature='twist', lc_tag=None, pt_tag=None,
        plot=True):
    from gauge.gauge import Gauge
    from models import get_model

    lc, lct = _load(model_name, 'lc_sens', 'lc_sens*.npz', lc_tag)
    pt, ptt = _load(model_name, 'ptc_sens', f'ptc_sens_{target}_{mode}*.npz', pt_tag)
    names, li, pi = align(lc, pt)
    print(f"[coupling] {model_name} / {target} ({mode}); {len(names)} shared parameters; "
          f"lc tag {lct}, ptc tag {ptt}")

    # ---- 1. per-parameter scatter -------------------------------------------------- #
    x = np.asarray(lc['lc_sens'])[li]                     # LC sensitivity per parameter
    y = np.asarray(pt[f'{feature}_span'])[pi]             # PTC sensitivity per parameter
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        raise SystemExit('too few parameters with both sensitivities finite')
    r = np.corrcoef(np.log10(x[ok] + 1e-12), np.log10(y[ok] + 1e-12))[0, 1]
    xm, ym = np.nanmedian(x[ok]), np.nanmedian(y[ok])
    quad = dict(
        both=int(np.sum((x[ok] > xm) & (y[ok] > ym))),
        lc_only=int(np.sum((x[ok] > xm) & (y[ok] <= ym))),
        ptc_only=int(np.sum((x[ok] <= xm) & (y[ok] > ym))),
        neither=int(np.sum((x[ok] <= xm) & (y[ok] <= ym))))
    print(f"\n1. PER-PARAMETER SCATTER  (PTC feature: {feature})")
    print(f"   log-log correlation r = {r:+.3f}")
    print(f"   quadrants about the medians (LC {xm:.3g}, PTC {ym:.3g}):")
    print(f"     high LC & high PTC : {quad['both']:3d}   (coupled, expected)")
    print(f"     high LC & low  PTC : {quad['lc_only']:3d}   (decoupled the USELESS way)")
    print(f"     low  LC & high PTC : {quad['ptc_only']:3d}   <- the identifiability prize")
    print(f"     low  LC & low  PTC : {quad['neither']:3d}   (sloppy in both)")
    ratio = np.where(ok, y / np.maximum(x, 1e-12), np.nan)
    top = [i for i in np.argsort(-np.nan_to_num(ratio))[:5] if ok[i]]
    print(f"   best PTC-per-LC parameters: "
          f"{', '.join(f'{names[i]} ({ratio[i]:.2f})' for i in top)}")

    # ---- 2. local coupling matrix ---------------------------------------------------- #
    # rows = factor settings, cols = parameters, in LOG-parameter displacement
    fac = np.asarray(lc['factors'], float)
    dlog = np.log(fac)[None, :]                            # displacement per setting
    with np.errstate(invalid='ignore', divide='ignore'):
        J_LC = (np.asarray(lc['shape_all'])[li] / dlog).T          # (n_fac, n_param)
        J_PT = (np.asarray(pt[feature])[pi] / dlog).T
    keep = np.array([i for i in range(len(names)) if ok[i]])
    J_LC, J_PT = J_LC[:, keep], J_PT[:, keep]
    kept = [names[i] for i in keep]

    g = Gauge(get_model(model_name))
    gi = [g.names.index(p) for p in kept if p in g.names]
    Gsub = g.G[gi] if len(gi) == len(kept) else None
    if Gsub is not None and Gsub.size:
        Q, _ = np.linalg.qr(Gsub)
        Pperp = np.eye(len(kept)) - Q @ Q.T                 # project OUT the gauge
        J_LCq, J_PTq = np.nan_to_num(J_LC) @ Pperp, np.nan_to_num(J_PT) @ Pperp
        print(f"\n2. LOCAL COUPLING  (gauge quotient: removed {Gsub.shape[1]} of "
              f"{len(kept)} directions)")
    else:
        J_LCq, J_PTq = np.nan_to_num(J_LC), np.nan_to_num(J_PT)
        print(f"\n2. LOCAL COUPLING  (gauge NOT projected -- parameter sets differ)")

    ang, _Qa, _Qb = principal_angles(J_LCq, J_PTq)
    if ang.size:
        print(f"   principal angles between row(J_LC) and row(J_PTC), degrees:")
        print(f"     {np.array2string(np.round(ang, 1), max_line_width=70)}")
        print(f"     smallest {ang.min():.1f} deg (shared), largest {ang.max():.1f} deg "
              f"(seen by one only)")

    # for each LC singular direction, how much does the PTC move?
    u, s_lc, vt = np.linalg.svd(J_LCq, full_matrices=False)
    resp = np.linalg.norm(J_PTq @ vt.T, axis=0)
    s_n = s_lc / (s_lc.max() if s_lc.max() > 0 else 1)
    r_n = resp / (resp.max() if resp.max() > 0 else 1)
    print(f"   LC singular directions, from stiffest to sloppiest:")
    print(f"     {'sigma_LC/max':>13s} {'|J_PTC v|/max':>14s}")
    for i in range(len(s_lc)):
        mark = '  <- LC-sloppy but PTC-responsive' if (s_n[i] < 0.1 and r_n[i] > 0.3) else ''
        print(f"     {s_n[i]:13.3e} {r_n[i]:14.3e}{mark}")
    tail = s_n < 0.1
    frac = (np.sum(resp[tail] ** 2) / np.sum(resp ** 2)) if np.any(tail) and resp.any() else 0.0
    print(f"   the LC-sloppiest directions (sigma < 0.1 max, {int(tail.sum())} of "
          f"{len(s_lc)}) carry {100 * frac:.1f}% of the total PTC response")

    # ---- 3. what to confirm ----------------------------------------------------------- #
    cand = [i for i in np.argsort(-r_n) if s_n[i] < 0.1][:3]
    print(f"\n3. TO CONFIRM (candidates only -- a linear summary is not a result)")
    if cand:
        for i in cand:
            heavy = np.argsort(-np.abs(vt[i]))[:4]
            print(f"   direction {i}: sigma_LC/max = {s_n[i]:.2e}, PTC response "
                  f"{r_n[i]:.2e}; dominated by "
                  f"{', '.join(f'{kept[j]}({vt[i][j]:+.2f})' for j in heavy)}")
        print(f"   Next: displace along these by a finite amount and recompute the PTC on the "
              f"ADAPTIVE\n   engine (engine/reference.py). input_screen's equivalent claim "
              f"survived only where that\n   was done -- the jacobian-derived version was "
              f"wrong by ~80 orders of magnitude.")
    else:
        print(f"   none: every LC-sloppy direction is also PTC-quiet, so at base this model's "
              f"PTC\n   adds nothing the limit cycle does not already constrain "
              f"(the input_screen result).")

    blob = dict(model=model_name, target=target, mode=mode, feature=feature,
                params=np.array(kept), lc_sens=x[keep], ptc_sens=y[keep],
                corr=r, quadrants=np.array(list(quad.values())),
                quadrant_names=np.array(list(quad)),
                principal_angles=ang, sigma_lc=s_lc, ptc_response=resp,
                sloppy_ptc_fraction=frac, J_LC=J_LCq, J_PTC=J_PTq)
    out = paths.out_path(model_name, 'coupling', f'coupling_{target}_{mode}_{feature}.npz')
    paths.savez(out, **blob)
    print(f"\n[coupling] -> {out}")
    if plot:
        try:
            from plotting import coupling_scatter
            fig = coupling_scatter(blob)
            fp = out.replace('.npz', '.png')
            fig.savefig(fp, dpi=140, bbox_inches='tight')
            print(f"[coupling] -> {fp}")
        except Exception as e:
            print(f"[coupling] plot skipped ({type(e).__name__}: {e})", file=sys.stderr)
    return blob


def main(argv=None):
    ap = argparse.ArgumentParser(description='LC vs PTC coupling, locally at base')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--target', default=os.environ.get('TARGET'))
    ap.add_argument('--mode', default=os.environ.get('MODE', 'pulse'))
    ap.add_argument('--feature', default='twist', choices=FEATURES)
    ap.add_argument('--lc-tag', default=None)
    ap.add_argument('--pt-tag', default=None)
    ap.add_argument('--no-plot', action='store_true')
    a = ap.parse_args(argv)
    if not a.target:
        raise SystemExit('--target is required')
    run(a.model, a.target, a.mode, a.feature, a.lc_tag, a.pt_tag, plot=not a.no_plot)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
