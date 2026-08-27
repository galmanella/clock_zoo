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
   Central-difference jacobians of the FULL observables with respect to log-parameters --
   J_LC over the phase-aligned cycle profiles plus the period, J_PTC over the PTC surface --
   with the gauge PROJECTED OUT, compared by PRINCIPAL ANGLES between their row spaces. Small
   angles mean the two measurements constrain the same parameter combinations; a large angle
   is a direction one sees and the other does not. Also reports, for each right-singular
   direction of J_LC ordered by LC-sensitivity, how strongly J_PTC responds.

   THE JACOBIANS COME FROM THE RAW GRIDS, not from the per-factor sensitivity scalars. That
   shortcut gives a matrix of rank <= n_factors, so most parameter directions are invisible to
   it by under-determination alone, and they then read as "LC-sloppy but PTC-responsive" --
   a fabricated identifiability prize. See `build_jacobians`.

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


def build_jacobians(lc, pt, names, li, pi, ok, hi=5, lo=3):
    """Central-difference jacobians of the FULL observables w.r.t. log-parameters.

        J_LC  : (n_states * m + 1) x n_param   phase-aligned cycle profiles, plus the period
        J_PTC : (n_phase * n_dose) x n_param   the PTC surface itself

    BUILT FROM THE RAW GRIDS, NOT FROM THE SCALAR SUMMARIES. The obvious shortcut is to stack
    the per-factor sensitivity SCALARS into a (n_factor x n_param) matrix -- and it is wrong in
    a way that manufactures exactly the result this analysis is looking for. With 9 factors and
    18 parameters such a matrix has rank <= 9, so at least 9 parameter directions are invisible
    to it PURELY BY UNDER-DETERMINATION. Its SVD then reports singular values of ~1e-17 for
    them, and since J_PTC responds to those arbitrary null vectors, they read as
    "LC-sloppy but PTC-responsive" -- a fake identifiability prize. That is the same class of
    error as input_screen's null(J_LC), which reported 7.2e-5 for a provably-zero direction.

    With one column per parameter and thousands of rows, the rank is limited by the physics
    rather than by the experiment design, which is the only way the null space means anything.

    `hi`/`lo` index the factor grid symmetrically about 1.0 (default x1.26 and x0.79, i.e.
    +/-0.235 in log), so the difference is a genuine central difference in log-parameter.
    """
    fac = np.asarray(lc['factors'], float)
    dlog = np.log(fac[hi]) - np.log(fac[lo])
    prof = np.asarray(lc['profiles'])                      # (n_param, n_fac, n_states, m)
    scale = np.asarray(lc['scale'])[None, :, None]
    dT = np.asarray(lc['dT'])
    grids = np.asarray(pt['ptc_grids'])                    # (n_param, n_fac, n_phase, n_dose)

    cols_lc, cols_pt, kept, keep = [], [], [], []
    for j, i in enumerate(range(len(names))):
        if not ok[i]:
            continue
        a, b = prof[li[i], hi], prof[li[i], lo]
        ga, gb = grids[pi[i], hi], grids[pi[i], lo]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue                                       # a rejected setting: no derivative
        dtemp = (dT[li[i], hi] - dT[li[i], lo])
        if not np.isfinite(dtemp):
            continue
        cols_lc.append(np.concatenate([((a - b) / scale[0]).ravel() / dlog, [dtemp / dlog]]))
        # circular difference for phases, and NaN (unusable points) contributes nothing
        d = ((ga - gb + 0.5) % 1.0) - 0.5
        cols_pt.append(np.nan_to_num(d).ravel() / dlog)
        kept.append(names[i]); keep.append(i)
    if not kept:
        raise SystemExit('no parameter has finite raw grids at both difference factors')
    return (np.array(cols_lc).T, np.array(cols_pt).T, kept, np.array(keep))


def decoupling_spectrum(J_LC, J_PT, ridge=1e-10):
    """The maximally-decoupled parameter DIRECTIONS, by generalized eigendecomposition.

    Answers directly: which unit direction `v` in log-parameter space maximises

        rho(v)  =  ||J_PTC v||^2 / ||J_LC v||^2

    i.e. moves the PTC as much as possible per unit of limit-cycle movement. That is the
    stationary problem

        (J_PTC^T J_PTC) v  =  rho (J_LC^T J_LC) v

    whose eigenpairs give the whole spectrum at once, ordered from most PTC-favouring to most
    LC-favouring.

    WHY THIS AND NOT THE PER-PARAMETER SCATTER. A decoupled direction is generically a
    COMBINATION of parameters, and nothing forces it to align with a coordinate axis. Ranking
    single parameters by their sensitivity ratio can only find the decoupling that happens to
    be axis-aligned, so it systematically UNDER-reports how much freedom there is. Scanning
    J_LC's singular directions (the input_screen approach) is better but still indirect: those
    directions are chosen to diagonalise the LC alone and need not be extremal for the ratio.

    Both matrices are scaled to unit spectral norm first, so rho is a dimensionless ratio of
    RELATIVE responses and rho = 1 means "equally visible to both experiments".

    A RATIO IS NOT ENOUGH, AND THIS IS WHERE IT GOES WRONG. rho blows up wherever the
    DENOMINATOR is small, so the top eigenvector will happily be a direction the LC jacobian
    cannot resolve rather than one it genuinely does not move. Those are different claims and
    only the second is a result. The finite-difference LC jacobian has a noise floor -- the
    orbit is solved to |F| ~ 1e-13 but the cycle itself is accurate to ~5e-6 relative
    (engine/orbit's 8x self-convergence on Almeida) -- so a relative LC response below roughly
    1e-5 carries no information at all.

    So the pencil is solved inside the subspace where J_LC is numerically resolvable:
    `lc_floor` is a relative singular-value cut on J_LC, and everything below it is dropped
    rather than inverted. Sweeping `lc_floor` and watching rho is the honest diagnostic --
    if the top rho collapses as the floor rises, it was living on unresolvable directions.

    Returns (rho[k], V[n_param, k] unit-norm, info) with `info` carrying the per-direction
    ABSOLUTE relative responses `lc_resp` and `ptc_resp`, which is what actually has to be
    checked before believing any of it.
    """
    A = np.nan_to_num(np.asarray(J_PT, float))
    B = np.nan_to_num(np.asarray(J_LC, float))
    A = A / max(np.linalg.norm(A, 2), 1e-300)
    B = B / max(np.linalg.norm(B, 2), 1e-300)

    # restrict to the LC-resolvable subspace
    _u, s, vt = np.linalg.svd(B, full_matrices=False)
    keep = s / (s.max() if s.size else 1) > ridge
    Q = vt[keep].T                                     # (n_param, k) orthonormal
    Ar, Br = A @ Q, B @ Q
    GA, GB = Ar.T @ Ar, Br.T @ Br
    from scipy.linalg import eigh
    rho, W = eigh(GA, GB)
    order = np.argsort(-rho)
    rho, W = rho[order], W[:, order]
    V = Q @ W
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    info = dict(lc_resp=np.linalg.norm(B @ V, axis=0),   # relative, since |B|_2 = 1
                ptc_resp=np.linalg.norm(A @ V, axis=0),
                n_kept=int(keep.sum()), n_total=len(s),
                cond=float(np.linalg.cond(GB)))
    return rho, V, info


def top_subspace_angles(A, B, ks=(1, 2, 3, 5)):
    """{k: principal angles (deg) between the leading-k right-singular subspaces of A and B}.

    "Do the two experiments determine the SAME best-determined parameter combinations?" --
    which is the question the full-row-space angle cannot answer once both jacobians are full
    rank, since two rank-r subspaces of a p-dimensional space are then forced to intersect in
    at least 2r - p dimensions regardless of any physics."""
    _ua, _sa, vta = np.linalg.svd(np.nan_to_num(A), full_matrices=False)
    _ub, _sb, vtb = np.linalg.svd(np.nan_to_num(B), full_matrices=False)
    out = {}
    for k in ks:
        if k > min(vta.shape[0], vtb.shape[0]):
            continue
        sv = np.linalg.svd(vta[:k] @ vtb[:k].T, compute_uv=False)
        out[k] = np.degrees(np.arccos(np.clip(sv, -1, 1)))
    return out


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
        plot=True, lc_floor=1e-3):
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
    J_LC, J_PT, kept, keep = build_jacobians(lc, pt, names, li, pi, ok)

    g = Gauge(get_model(model_name))
    gi = [g.names.index(p) for p in kept if p in g.names]
    n_obs_lc, n_obs_pt = J_LC.shape[0], J_PT.shape[0]
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

    # Angles between the DOMINANT subspaces, not between the full row spaces. Both jacobians
    # are essentially full rank here, and two rank-r subspaces of a p-dimensional space must
    # intersect in at least 2r-p dimensions -- so the full-row-space angles come out as a row
    # of exact zeros that says nothing except "both are full rank". The informative question
    # is whether the two experiments determine the same BEST-determined combinations, which is
    # the angle between their leading k directions.
    ang = top_subspace_angles(J_LCq, J_PTq)
    print(f"   angle between the top-k best-determined subspaces of each (degrees):")
    print(f"     {'k':>3s} {'angles':>34s}")
    for k, a in ang.items():
        print(f"     {k:3d} {np.array2string(np.round(a, 1), max_line_width=60):>34s}")
    print(f"   (0 deg = the two experiments pin the same combination; 90 deg = one sees a "
          f"direction\n    the other is blind to)")

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
    phys = s_n > 1e-12          # drop the directions the gauge projection annihilated
    tail = (s_n < 0.1) & phys
    frac = ((np.sum(resp[tail] ** 2) / np.sum(resp[phys] ** 2))
            if np.any(tail) and resp[phys].any() else 0.0)
    print(f"   the LC-sloppiest directions (sigma < 0.1 max, {int(tail.sum())} of "
          f"{int(phys.sum())} physical) carry {100 * frac:.1f}% of the total PTC response")
    n_num_null = int(np.sum(s_n < 1e-12))
    if n_num_null:
        print(f"   NOTE {n_num_null} direction(s) are numerically null in J_LC. With "
              f"{n_obs_lc} observables against")
        print(f"        {len(kept)} parameters that is a genuine flat direction rather than "
              f"a rank artifact --")
        print(f"        but confirm it by finite displacement before believing it (see 3).")

    # ---- 2b. the maximally-decoupled DIRECTIONS ---------------------------------------- #
    # the per-parameter best ratio, on the SAME normalisation, so the two are comparable
    An = np.nan_to_num(J_PTq) / max(np.linalg.norm(np.nan_to_num(J_PTq), 2), 1e-300)
    Bn = np.nan_to_num(J_LCq) / max(np.linalg.norm(np.nan_to_num(J_LCq), 2), 1e-300)
    axis_rho = (np.sum(An ** 2, axis=0) / np.maximum(np.sum(Bn ** 2, axis=0), 1e-300))
    best_axis = int(np.argmax(axis_rho))
    print(f"\n2b. MAXIMALLY-DECOUPLED DIRECTIONS (generalized eigenproblem)")
    print(f"   rho = ||J_PTC v||^2 / ||J_LC v||^2, both jacobians normalised to unit spectral "
          f"norm.")
    print(f"   A per-parameter scatter can only find AXIS-ALIGNED decoupling; a decoupled "
          f"direction\n   is generically a combination, so this is the question that scatter "
          f"cannot answer.")
    print(f"\n   {'LC floor':>9s} {'kept':>5s} {'rho_max':>10s} {'|J_LC v|':>10s} "
          f"{'|J_PTC v|':>10s}   dominant parameters")
    for fl in (1e-10, 1e-4, 1e-3, 1e-2, 3e-2):
        r, Vv, inf = decoupling_spectrum(J_LCq, J_PTq, ridge=fl)
        if not len(r):
            continue
        heavy = np.argsort(-np.abs(Vv[:, 0]))[:3]
        print(f"   {fl:9.0e} {inf['n_kept']:2d}/{inf['n_total']:<2d} {r[0]:10.2f} "
              f"{inf['lc_resp'][0]:10.2e} {inf['ptc_resp'][0]:10.2e}   "
              f"{', '.join(f'{kept[j]}({Vv[j, 0]:+.2f})' for j in heavy)}")
    print(f"   ^ if rho_max collapses as the floor rises, the big values lived on LC "
          f"directions the\n     finite-difference jacobian cannot resolve (its noise floor "
          f"is ~1e-5 relative), not on\n     directions the limit cycle genuinely does not "
          f"move.")

    rho, V, info = decoupling_spectrum(J_LCq, J_PTq, ridge=lc_floor)
    print(f"\n   AT lc_floor = {lc_floor:.0e}  ({info['n_kept']}/{info['n_total']} LC "
          f"directions resolvable)")
    print(f"   spectrum: {np.array2string(np.round(rho, 2), max_line_width=72)}")
    print(f"   best single PARAMETER : rho = {axis_rho[best_axis]:.3f}  ({kept[best_axis]})")
    print(f"   best COMBINATION      : rho = {rho[0]:.3f}   "
          f"-- {rho[0] / max(axis_rho[best_axis], 1e-30):.1f}x the best single parameter")
    for i in range(min(3, V.shape[1])):
        heavy = np.argsort(-np.abs(V[:, i]))[:5]
        print(f"   dir {i}: rho = {rho[i]:8.2f}  LC {info['lc_resp'][i]:.2e}  "
              f"PTC {info['ptc_resp'][i]:.2e}  "
              f"{', '.join(f'{kept[j]}({V[j, i]:+.2f})' for j in heavy)}")
    print(f"   most LC-favouring (rho = {rho[-1]:.3g}): "
          f"{', '.join(f'{kept[j]}({V[j, -1]:+.2f})' for j in np.argsort(-np.abs(V[:, -1]))[:4])}")
    condB = info['cond']

    # ---- 3. what to confirm ----------------------------------------------------------- #
    cand = [i for i in np.argsort(-r_n) if (s_n[i] < 0.1 and s_n[i] > 1e-12)][:3]
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
                rho=rho, V=V, axis_rho=axis_rho, cond_B=condB,
                dir_lc_resp=info['lc_resp'], dir_ptc_resp=info['ptc_resp'],
                lc_floor=lc_floor, n_kept_lc=info['n_kept'],
                top_angles=np.array([np.mean(a) for a in ang.values()]),
                top_angle_ks=np.array(list(ang)), sigma_lc=s_lc, ptc_response=resp,
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
    ap.add_argument('--lc-floor', type=float, default=1e-3,
                    help='relative singular-value cut on J_LC for the decoupling pencil')
    a = ap.parse_args(argv)
    if not a.target:
        raise SystemExit('--target is required')
    run(a.model, a.target, a.mode, a.feature, a.lc_tag, a.pt_tag, plot=not a.no_plot,
        lc_floor=a.lc_floor)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
