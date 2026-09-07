"""
analysis/identifiability.py
===========================
THE project question, as a rank comparison.

    $PY -m analysis.identifiability --model korencic --mode instant \
        --targets Bmalx,Perx,Reverbx,Cryx

"How identifiable are mechanistic models from limit-cycle data, and from PTC data?" is a
question about the SIZE of the set of parameter sets consistent with the data -- not about
recovering a particular parameter vector. Non-uniqueness is the premise, so the quantity of
interest is the DIMENSION of the solution family:

    dim(solution set) = n_free - rank(J)

for each observable, measured LOCALLY (the coarse-scale rank saturates for everything, which is
why analysis/coupling.py's +-26% comparison cannot separate them).

    LC     the phase-aligned limit-cycle profiles of all observable species -- what a
           trajectory experiment measures, with the phase origin fixed by the orbit solver so
           the comparison is not confounded by a phase shift.
    PTC    the single-gene phase-transition surface (unit phase, real and imaginary parts).
    both   the two stacked, each block normalized so neither dominates by sheer scale.

If PTC adds directions LC cannot see, rank(both) > rank(LC) and the difference is exactly the
identifiability the perturbation experiment buys. On Almeida, matched single-gene: LC alone
3/16, PTC alone 7/16 -- a PTC is worth ~3-4 trajectories, per gene (PROJECT_SUMMARY 5.4b).

WHY THE MATCHED COMPARISON IS THE ONE TO QUOTE
    The unmatched block pits an all-species, 64-point time course against ONE gene's PTC. Those
    are not the same size of experiment, so whatever it shows cannot support an
    experimental-design claim. It is kept as a warning, printed second, and labelled.

MULTI-GENE: ARE THE PROBES REDUNDANT?
    `--targets` stacks per-gene PTC jacobians, each block scale-normalized so no probe dominates
    by magnitude, and reports the rank of every subset. On Almeida this INVERTED the Mirsky
    design rule: one gene 7/16, all four 13/16, condition 4x better -- genes are the informative
    axis there, where for Mirsky the advice was "vary dose, not gene" (PROJECT_SUMMARY 5.5).
    Optimal PTC experimental design is model-dependent and does not transfer, which is exactly
    what the complexity ladder was built to be able to say.

    GATE THE SURFACES FIRST. The first pass of Almeida's table reported the best single probe at
    10/16 and four probes at 14/16; E4BP4 and REV then FAILED analysis/quality, and a jacobian
    of a phase-scrambled surface is a jacobian of NOISE -- which INFLATES numerical rank. The
    broken probe produced the most attractive number. Run `analysis.quality` before believing
    any row here (REPO_MAP hazard 12).

THE TIME GAUGE
    `--no-include-time` drops the time rescale from the quotient, growing it by one direction
    (korencic 33 -> 34). Use it for PULSE mode: a pulse of duration fixed in HOURS is itself a
    clock, so pulse data can see the time rescale and quotienting it out projects away a
    direction the data determines (FIT_VALIDITY 3c). Instant-mode PTCs are invariant to 1.1e-07
    cyc under a pure time rescale and want the default.
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                                          # noqa: E402

#: Finite-difference step in log-parameter space. 1e-3 is the LOCAL regime; PROJECT_SUMMARY 5.1
#: measures the rank rising to full at h ~ 0.23 (x1.26), and records that both numbers are right
#: and answer different questions. Local is the one that speaks to a solution-set dimension.
H = 1e-3

#: Relative singular-value thresholds a direction counts as "determined" at.
TOLS = (1e-2, 1e-3)


def _ranks(J, n, tols=TOLS):
    s = np.linalg.svd(J, compute_uv=False)
    s = s / s[0] if s[0] > 0 else s
    return [int(np.sum(s > t)) for t in tols], (1 / s[-1] if s[-1] > 0 else np.inf), s


def report(label, J, n, tols=TOLS):
    (r2, r3), cond, s = _ranks(J, n, tols)
    print(f"  {label:26s} rank@{tols[0]:.0e} {r2:2d}/{n}  rank@{tols[1]:.0e} {r3:2d}/{n}  "
          f"cond {cond:8.2e}   dim(solution set) = {n - r2}", flush=True)
    return s, r2


def build(model_name, mode, targets, n_phase=12, n_dose=8, cap=6.0, include_time=True,
          m_cycle=64, h=H):
    """Finite-difference jacobians of the LC and of each target's PTC, in quotient coordinates.

    Returns (info, J_lc_all, {target: J_ptc}, {target: J_lc_one}). Each is normalized by its own
    Frobenius norm, which is what makes ranks across observables comparable at a shared
    threshold -- a raw scale difference would otherwise decide the count."""
    import jax.numpy as jnp
    from fit.cost import make_cost, RadialTarget, quotient_basis
    from fit.doses import fit_dose_grid
    from models import get_model
    from engine.orbit import OrbitSolver

    model = get_model(model_name)
    names, z_base, B, _g = quotient_basis(model, include_time=include_time)
    n = B.shape[1]
    obs = list(model.observable_states())
    oidx = [int(model.var_index(s)) for s in obs]
    solver = OrbitSolver(model)
    v0 = np.zeros(n)

    print(f"[identifiability] {model_name} ({mode}), {n} free directions"
          f"{'' if include_time else '  [include_time=False: the time rescale is NOT quotiented]'}"
          f", {len(obs)} observable species, h={h}", flush=True)

    def lc_vec_of(species):
        """LC profile restricted to a subset of species -- for the MATCHED comparison."""
        idx = jnp.asarray([int(model.var_index(t)) for t in species])

        def f(v):
            P = model.jax_apply(jnp.exp(jnp.asarray(z_base) + jnp.asarray(B) @ jnp.asarray(v)),
                                names)
            y0, T, _r = solver.solve(P, solver.guess(P))
            return np.asarray(solver.cycle(P, y0, T, m_cycle)[:, idx]).ravel()
        return f

    def jac_of(fn, v0_, label):
        f0 = fn(v0_)
        J = np.zeros((np.asarray(f0).size, n))
        t0 = time.time()
        for i in range(n):
            vp = v0_.copy(); vp[i] += h
            vm = v0_.copy(); vm[i] -= h
            J[:, i] = (fn(vp) - fn(vm)) / (2 * h)
            if (i + 1) % 8 == 0 or i + 1 == n:
                el = time.time() - t0
                print(f"    {label:>18s} {i + 1:3d}/{n} columns, {el / 60:4.1f} min, "
                      f"eta {el / (i + 1) * (n - i - 1) / 60:4.1f} min", flush=True)
        return J

    J_lc = jac_of(lc_vec_of(obs), v0, 'LC (all species)')
    J_lc = J_lc / np.linalg.norm(J_lc)

    J_ptc, J_lc1, doses_of = {}, {}, {}
    for t in targets:
        doses, S = fit_dose_grid(model_name, t, mode, cap, n_dose)
        doses_of[t] = (np.asarray(doses), float(S))
        C = make_cost(model, t, doses, RadialTarget(), n_phase=n_phase, mode=mode,
                      backend='diffrax', dt=0.02, w_osc=0.0, w_amp=0.0)

        def ptc_vec(v, _C=C):
            zu, _alive, _amp = _C['surface'](v)
            return np.concatenate([np.real(zu).ravel(), np.imag(zu).ravel()])

        Jp = jac_of(ptc_vec, v0, f'PTC {t}')
        J_ptc[t] = Jp / np.linalg.norm(Jp)
        Jl = jac_of(lc_vec_of([t]), v0, f'LC {t} only')
        J_lc1[t] = Jl / np.linalg.norm(Jl)

    info = dict(model=model_name, mode=mode, n_free=n, targets=np.array(list(targets)),
                observables=np.array(obs), include_time=include_time, h=h,
                n_phase=n_phase, n_dose=n_dose, cap=cap, obs_idx=np.array(oidx))
    return info, J_lc, J_ptc, J_lc1, doses_of


def analyse(info, J_lc, J_ptc, J_lc1, doses_of):
    """Print every table and return the numbers worth storing."""
    n = info['n_free']
    targets = [str(t) for t in info['targets']]
    out = {}

    print(f"\n{'=' * 84}\nMATCHED -- one gene's PTC against the SAME species' trajectory"
          f"\n{'=' * 84}")
    print("  This is the row to quote: both sides are one species, so it is a statement about\n"
          "  experimental design rather than about how much data each side happened to carry.\n")
    for t in targets:
        d, S = doses_of[t]
        print(f"  --- {t}   S_crit = {S:.4g}, doses {d.min():.3g} .. {d.max():.3g}")
        _s, r_lc1 = report(f'LC ({t} only)', J_lc1[t], n)
        _s, r_p = report(f'PTC ({t})', J_ptc[t], n)
        _s, r_b = report(f'LC+PTC ({t})', np.vstack([J_lc1[t], J_ptc[t]]), n)
        print(f"    PTC adds {r_b - r_lc1:2d} beyond one-species LC ({r_lc1} -> {r_b} of {n});"
              f"  PTC alone is worth {r_p / max(r_lc1, 1):.1f}x the trajectory\n")
        out[f'matched_{t}'] = np.array([r_lc1, r_p, r_b])

    print(f"{'=' * 84}\nUNMATCHED -- kept as a WARNING, not as a result\n{'=' * 84}")
    print("  All observable species x a full time course against ONE gene's PTC. The two\n"
          "  experiments are not the same size, so this cannot support a design claim.\n")
    _s, r_lc = report('LC (all species)', J_lc, n)
    for t in targets:
        report(f'LC(all)+PTC({t})', np.vstack([J_lc, J_ptc[t]]), n)
    out['rank_lc_all'] = r_lc

    if len(targets) > 1:
        print(f"\n{'=' * 84}\nMULTI-GENE -- are the probes redundant?\n{'=' * 84}")
        print("  Per-gene PTC jacobians stacked, each block already unit-Frobenius so no probe\n"
              "  dominates by magnitude. Gate the surfaces with analysis.quality first: a\n"
              "  jacobian of a scrambled surface is a jacobian of noise, and noise INFLATES\n"
              "  rank (PROJECT_SUMMARY 5.5).\n")
        rows = []
        for k in range(1, len(targets) + 1):
            best = None
            for combo in itertools.combinations(targets, k):
                J = np.vstack([J_ptc[t] for t in combo])
                (r2, r3), cond, _s = _ranks(J, n)
                rows.append((combo, r2, r3, cond))
                if best is None or r2 > best[1] or (r2 == best[1] and cond < best[3]):
                    best = (combo, r2, r3, cond)
            lo = min(r for c, r, _3, _c in rows if len(c) == k)
            hi = max(r for c, r, _3, _c in rows if len(c) == k)
            cl = min(c_ for cb, _r, _3, c_ in rows if len(cb) == k)
            ch = max(c_ for cb, _r, _3, c_ in rows if len(cb) == k)
            print(f"  {k} probe(s):  rank@1e-2 {lo}-{hi}/{n}   cond {cl:.2e}-{ch:.2e}"
                  f"   best {'+'.join(best[0])} at {best[1]}/{n}")
        print()
        for combo, r2, r3, cond in rows:
            print(f"    {'+'.join(combo):<40s} rank@1e-2 {r2:2d}/{n}  rank@1e-3 {r3:2d}/{n}  "
                  f"cond {cond:8.2e}")
        out['multi'] = np.array([(('+'.join(c)), r2, r3, cond) for c, r2, r3, cond in rows],
                                dtype=object)

    # Which directions does each MISS? Principal angles between the well-determined subspaces.
    t0 = targets[0]
    _u, _s, Vt_lc = np.linalg.svd(J_lc, full_matrices=False)
    _u, _s, Vt_p = np.linalg.svd(J_ptc[t0], full_matrices=False)
    (r_lc2, _), _c, _s = _ranks(J_lc, n)
    (r_p2, _), _c, _s = _ranks(J_ptc[t0], n)
    k = min(r_lc2, r_p2)
    if k > 0:
        sv = np.linalg.svd(Vt_lc[:k] @ Vt_p[:k].T, compute_uv=False)
        ang = np.degrees(np.arccos(np.clip(sv, -1, 1)))
        print(f"\n  principal angles between the top-{k} determined subspaces, "
              f"LC(all) vs PTC({t0}) (deg):")
        print("   " + "  ".join(f"{a:.1f}" for a in ang))
        print(f"  largest {ang.max():.1f} deg -- 0 means they see the SAME combinations, "
              f"90 fully complementary")
        out['principal_angles'] = ang
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='rank(J) / dim(solution set) per observable')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--mode', default='instant', choices=('instant', 'pulse'))
    ap.add_argument('--targets', default=None,
                    help='comma-separated; default: the model in-scope targets that reset')
    ap.add_argument('--n-phase', type=int, default=12)
    ap.add_argument('--n-dose', type=int, default=8)
    ap.add_argument('--cap', type=float, default=6.0, help='max_factor for fit_dose_grid')
    ap.add_argument('--h', type=float, default=H)
    ap.add_argument('--no-include-time', dest='include_time', action='store_false',
                    help='drop the time rescale from the quotient -- USE FOR PULSE MODE')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)

    if a.targets:
        targets = [t.strip() for t in a.targets.split(',') if t.strip()]
    else:
        from analysis import genemap as GM
        targets = GM.scope_targets(a.model)
        if not targets:
            raise SystemExit(f'no in-scope targets for {a.model}; pass --targets')
    if a.mode == 'pulse' and a.include_time:
        print("[identifiability] NOTE: pulse mode with include_time=True quotients out a\n"
              "  direction the data CAN see -- an 8 h pulse is itself a clock (FIT_VALIDITY\n"
              "  3c). Pass --no-include-time unless you mean to reproduce an older count.",
              flush=True)

    info, J_lc, J_ptc, J_lc1, doses_of = build(
        a.model, a.mode, targets, n_phase=a.n_phase, n_dose=a.n_dose, cap=a.cap,
        include_time=a.include_time, h=a.h)
    out = analyse(info, J_lc, J_ptc, J_lc1, doses_of)

    blob = dict(info)
    blob['J_lc_all'] = J_lc                       # SAVE THE RAW JACOBIANS -- hazard 9
    for t in targets:
        blob[f'J_ptc__{t}'] = J_ptc[t]
        blob[f'J_lc1__{t}'] = J_lc1[t]
        blob[f'doses__{t}'], blob[f'scrit__{t}'] = doses_of[t]
    for k, v in out.items():
        blob[k] = v
    p = paths.out_path(a.model, 'identifiability',
                       f'identifiability_{a.mode}.npz', paths.run_tag(a.tag))
    paths.savez(p, **blob)
    print(f"\n[identifiability] -> {p}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
