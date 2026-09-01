"""
fit/relcost.py
==============
DIRECTION 2: score the type-0 region in units of the candidate's OWN transition.

    $PY -m fit.relcost --selftest        # the invariants, incl. "same as make_cost" fidelity

THE IDEA
    A fixed absolute dose window lets the optimizer walk its transition out of view: 16 of 16
    Aug-30 fits ended with S* outside the window, most of them 3-4 decades below it, and the
    surface they were scored on had no transition in it at all (PROJECT_SUMMARY 5.10c).
    Widening the window costs resolution and does not close the escape (5.13a).

    So stop fixing the window in absolute units. Locate the candidate's own transition S*(v) and
    score on `[lo, hi] x S*(v)`. Moving S* then relabels the dose axis and changes nothing about
    what is measured, so the escape route is not merely penalised, IT NO LONGER EXISTS -- and
    what is scored is always the type-0 region, which is where isochron twist lives and what a
    radialization is about.

    THIS IS FOR SHAPE QUESTIONS ONLY, and that limit is not a technicality. Declaring the dose
    scale a nuisance parameter is legitimate when the question is "are these isochrons radial";
    it is wrong when fitting real data, where dose is a measured physical quantity and its
    calibration is information. It also makes S* itself unidentifiable by construction: you can
    no longer ask this cost where the transition is. `fit/cost.py` remains the general PTC
    objective.

LOCATING S* -- NOT WITH A SINGULARITY FINDER
    `analysis.winding.detect_grid` is quantised to the dose grid, a staircase. `soft_singularity`
    reads the amplitude dip, so it cannot see a defect that has already left the probe range --
    which is the state most candidates are in. Both would make the window jump.

    Instead: the dose at which per-row circular DISPERSION (fit.cost.row_dispersion) crosses
    0.5, interpolated in log dose. Smooth, defined everywhere, and the same observable the
    bracketing barrier uses. MEASURED along a path from base to a fitted optimum, over the
    contiguous stretch where an orbit exists, it slides 29.93 -> 9.93 with per-step
    |dlog10 S*| median 0.026 and max 0.085 -- a ratio of 3.3, i.e. smooth (5.13b).

    IT IS A PROXY, NOT S_crit. At the base point it reads ~72 where the singularity-based
    S_crit is 32.79, a consistent factor ~2.2. That is harmless for an anchor -- both model and
    target are expressed in the same units -- and must never be reported as S_crit.

THE GRADIENT IS TAKEN AT FIXED WINDOW, AND THAT IS AN APPROXIMATION
    S*(v) is a feature location, not the argmin of anything, so holding it fixed is NOT Danskin
    -- it drops the `dC/dS* . dS*/dv` term. The design makes that term small on purpose: the
    window and the target move together, so the cost is close to invariant under a change of
    S*. `--selftest` MEASURES the resulting bias against finite differences that DO move the
    window, rather than assuming it away.
"""
import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from engine.ptc import grid_points, DEAD_AMP, NEG_TOL
from fit.cost import make_cost, row_dispersion, RadialTarget
from fit.target import circ_cost, radial_z

#: dose window, in units of the candidate's own located transition. Starts ABOVE it: the point
#: is to score the type-0 side. 8.0 matches fit/doses' ceiling, staying inside the smooth regime
#: (hazard 11's blow-up sets in around 18 x S_crit).
SPAN = (1.2, 8.0)
#: probe range, in units of the BASE model's S_crit, and it deliberately stops at 12x -- an
#: earlier probe ran to 100x and returned uniformly dead surfaces, which is hazard 11's
#: invalid-cell regime rather than a property of the candidates.
PROBE = (1e-5, 12.0)  # x base S_crit. MEASURED bounds, see `locate` and ANCHOR VALIDITY.
DISP_FLOOR = 0.95     # a real type-1 side must reach this dispersion somewhere below S*
DEAD_PEN = 4.0        # charged when no dose in the whole probe range has a live orbit


def _surface_with(C, P, v, doses, n_phase):
    """`C['surface']`, but on an ARBITRARY dose array -- no rebuild, no recompile.

    `ptc_fn` takes its evaluation points as RUNTIME arguments, so one cost object can score any
    dose grid. That is what makes a per-candidate window affordable: rebuilding `make_cost` per
    evaluation recompiles and is orders of magnitude slower.

    This duplicates `fit.cost._surface`'s alive mask, which is a hazard-15 risk -- a diagnostic
    that reimplements the thing it checks cannot check it. `selftest` therefore asserts this
    reproduces `C['surface']` EXACTLY on C's own grid, turning the duplication into a checked
    invariant.
    """
    solver = C['solver']
    y0, T, res = solver.solve(P, C['guess'](P, C['y_seed']))
    full = solver.cycle(P, y0, T, 64)
    ref = C['ref_idx']
    cyc = full[:, ref]
    amp_lc = (jnp.max(cyc) - jnp.min(cyc)) / jnp.maximum(jnp.abs(jnp.mean(cyc)), 1e-12)
    T_nom = C['T_nom']
    ok_orbit = ((res < 1e-4) & jnp.isfinite(T) & (T > 0.25 * T_nom) & (T < 4.0 * T_nom)
                & (jnp.min(full) >= NEG_TOL) & jnp.all(jnp.isfinite(full)) & (amp_lc > 1e-3))
    ph, dz = grid_points(n_phase, doses)
    z, ymin = C['ptc_fn'](P, jnp.concatenate([y0, T[None]]), ph, dz)
    nd = len(doses)
    z = z.reshape(nd, n_phase).T
    ymin = ymin.reshape(nd, n_phase).T
    fin = jnp.isfinite(z.real) & jnp.isfinite(z.imag) & jnp.isfinite(ymin)
    zs = jnp.where(fin, z, 1.0 + 0j)
    amp = jnp.abs(zs)
    alive = fin & (amp > DEAD_AMP) & (ymin >= NEG_TOL) & ok_orbit
    return zs / (amp + 1e-12), alive, amp, amp_lc, T


def make_relative_cost(model, target_state, n_phase=20, n_dose=14, n_probe=24, span=SPAN,
                       probe=PROBE, s_crit_base=None, mode='instant', backend='diffrax',
                       dt=0.02, pulse=8.0, skip_p=None, readout_ref=None, basis=None,
                       amp_ramp=(0.05, 0.20), w_osc=0.2, w_amp=1.0, profile_psi=True,
                       w_anchor=1.0, disp_floor=DISP_FLOOR):
    """A cost whose dose window follows the candidate's own transition.

    Same interface as `fit.cost.make_cost` where it overlaps -- `v0`, `n_free`, `theta`,
    `parts`, `total`, `grad`, `surface` -- so drivers can dispatch on it.
    """
    from fit.doses import load_scrit
    if s_crit_base is None:
        s_crit_base, _g, _dt = load_scrit(getattr(model, 'name', None) or model.__class__.__name__.lower(),
                                          target_state, mode)
    probe_doses = np.geomspace(probe[0] * s_crit_base, probe[1] * s_crit_base, int(n_probe))
    # `shape` is filled in below, once the proxy has been CALIBRATED against the real S_crit.
    shape = None

    # ONE cost object, built on the probe grid, reused for every evaluation. It supplies the
    # production PTC, solver and quotient; the dose array is a runtime argument from here on.
    C0 = make_cost(model, target_state, probe_doses, RadialTarget(), n_phase=n_phase, mode=mode,
                   backend=backend, dt=dt, w_osc=w_osc, w_amp=w_amp, pulse=pulse,
                   skip_p=skip_p, readout_ref=readout_ref, basis=basis, amp_ramp=amp_ramp)
    from engine.orbit import make_guess_fn
    C0['guess'] = make_guess_fn(model)
    C0['y_seed'] = jnp.asarray(model.get_initial_state(), jnp.float64)
    C0['ref_idx'] = int(model.var_index(model.reference_variable))
    C0['T_nom'] = float(getattr(model, 'approx_period', None) or 24.0)
    C0['model'] = model
    old = np.asarray(C0['old'], float)
    ld_probe = np.log(probe_doses)

    # `C0['theta']` returns NUMPY, which cannot be traced. Rebuild it in jnp from the same
    # (z_base, B) the cost object uses, so the traced path and the reported path agree by
    # construction rather than by coincidence.
    _zb = jnp.asarray(C0['z_base'])
    _B = jnp.asarray(C0['B'])

    def _theta_j(v):
        return jnp.exp(_zb + _B @ jnp.asarray(v, jnp.float64))

    def theta(v):
        return np.asarray(_theta_j(v))

    def locate(v):
        """S*(v) and the evidence that it MEANS anything. numpy, deliberately not differentiated.

        Returns (S, where, q). `where` is 'in' only when the dispersion crossing is bracketed by
        rows that actually have a limit cycle -- dead rows read resultant 0, i.e. dispersion 1.0,
        which is indistinguishable from perfect type-1 and would otherwise report a dead surface
        as "still transitioning above the ceiling".
        """
        P = model.jax_apply(_theta_j(v), list(C0['names']))
        zu, alive, _a, _al, _T = _surface_with(C0, P, v, probe_doses, n_phase)
        d = np.asarray(row_dispersion(zu, alive), float)
        af = np.asarray(alive, float).mean(axis=0)
        idx = np.where(af > 0.5)[0]
        if len(idx) == 0:
            return float(probe_doses[0]), 'dead', dict(dmax=1.0, d_top=1.0, d_bot=1.0,
                                                       alive=0.0, dead=True)
        dl, ll = d[idx], ld_probe[idx]
        q = dict(dmax=float(dl.max()), d_top=float(dl[-1]), d_bot=float(dl[0]),
                 alive=float(af.mean()), dead=False)
        a = np.where(dl > 0.5)[0]
        if len(a) == 0:
            return float(np.exp(ll[0])), 'below', q     # transition under the probe floor
        if a[-1] >= len(dl) - 1:
            return float(np.exp(ll[-1])), 'above', q    # not reached before the orbit dies
        i = int(a[-1])
        f = (dl[i] - 0.5) / max(dl[i] - dl[i + 1], 1e-12)
        return float(np.exp(ll[i] + f * (ll[i + 1] - ll[i]))), 'in', q

    def anchor_penalty(where, q):
        """0 for an anchor that means what it says; positive and graded otherwise.

        ANCHOR VALIDITY -- why this is not optional. A relative window removes the escape
        "walk S* out of the window" by construction, but it opens two new ones, and BOTH were
        measured on real Aug-30 endpoints rather than imagined:

          (i)  COLLAPSE. seeds 7 and 15 put their crossing 3.5 and 2.6 decades BELOW S_crit and
               score 0.250 / 0.267 against base's 0.481 -- i.e. they look better. But their
               dispersion never exceeds 0.74 anywhere in 6 decades: there is no type-1 side, so
               the "transition" is a crossing between two type-0 regimes and the window sits on
               a surface that was already flat before the anchor. Charged via `dmax`.
          (ii) ESCAPE UP. seed 2 holds dispersion 1.000 to 12 x S_crit and its orbit then dies;
               seed 0 has no orbit at any of 6 decades. Neither has a reachable type-0 region.
               Charged via `d_top` / the dead branch.

        The scale is set so any invalid candidate scores worse than any valid one: `circ_cost`
        is bounded in [0,1], and each clause here reaches >= 1 when fully violated.
        """
        if q['dead']:
            return DEAD_PEN
        pen = max(0.0, disp_floor - q['dmax']) / max(1.0 - disp_floor, 1e-3)
        if where == 'above':
            pen += max(0.0, q['d_top'] - 0.5) / 0.5
        elif where == 'below':
            pen += max(0.0, 0.5 - q['d_bot']) / 0.5
        return float(pen)

    def _score(v, doses):
        """The scored surface and its cost, at a FIXED window. Jittable in v."""
        P = model.jax_apply(_theta_j(v), list(C0['names']))
        zu, alive, amp, amp_lc, T = _surface_with(C0, P, v, doses, n_phase)
        k = span[0] / doses[0]          # = 1/S*, the target's singularity at the anchor
        soft = None
        if amp_ramp is not None:
            lo, hi = amp_ramp
            t = jnp.clip((amp - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
            soft = t * t * (3.0 - 2.0 * t)

        def at(psi):
            return circ_cost(zu, radial_z(old, doses, k, psi), alive, soft=soft)

        if profile_psi:
            # psi -- the kick direction, which places the target's singular phase -- is a
            # NUISANCE: a Poincare oscillator has no distinguished phase, so pinning it would
            # ask "is the model radial AND oriented just so" (fit/target's own argument).
            #
            # Scanned inline in jnp rather than via `fit.target.profile`, which calls
            # `np.asarray(doses)` and so cannot be traced when the window is a runtime
            # argument. Free either way: the target is closed form, so this is n_psi array
            # evaluations and NOT ONE extra ODE solve.
            #
            # stop_gradient on the argmin is Danskin: the gradient w.r.t. the model is the
            # partial at the inner optimum, with no differentiation through the argmin.
            psis = jnp.linspace(0.0, 1.0, 64, endpoint=False)
            psi = lax.stop_gradient(psis[jnp.argmin(jax.vmap(at)(psis))])
        else:
            psi = 0.0
        # THE DEGENERACY GUARDS ARE NOT OPTIONAL HERE EITHER.
        #
        # MEASURED, and the reason this block exists: with the shape term alone, 8 CMA
        # evaluations took the relative cost 0.481 -> 0.310 while the fixed-grid diagnosis
        # came back "DEGENERATE -- Floquet mu = nan". A window that follows the candidate
        # removes one escape; it does nothing about the candidate ceasing to oscillate. These
        # are `fit.cost`'s own `osc` and `amp_floor`, reused rather than restated, so the two
        # objectives cannot drift apart in what they consider alive.
        a = jnp.maximum(0.0, 1.0 - amp_lc / C0['amp_floor']) ** 2
        if C0['osc_fn'] is not None and w_osc:
            b = C0['osc_fn'](P, C0['y_fp_seed'])[0]
        else:
            b = jnp.array(0.0)
        return at(psi) + w_osc * b + w_amp * a, zu, alive, amp_lc, T, at(psi), b, a

    _score_j = jax.jit(lambda v, d: _score(v, d)[0])
    _grad_j = jax.jit(jax.grad(lambda v, d: _score(v, d)[0]))

    # CALIBRATE THE ANCHOR, ONCE, AT THE BASE POINT.
    #
    # The dispersion crossing is a PROXY for the transition and sits systematically above the
    # singularity -- measured 65.47 against a true S_crit of 24.97, a factor 2.62. Left
    # uncalibrated, a span of (1.2, 8) "x S*" put the window top at 20.97 x the REAL S_crit,
    # past the ~18x where hazard 11's gradient pathology begins: the cost stayed smooth and
    # finite (0.4324 -> 0.4331 over the sweep) while |grad| read 1.3e+93. The finite differences
    # were correct the whole time; only autodiff was destroyed.
    #
    # So `span` is quoted in units of the model's ACTUAL S_crit -- the same convention as
    # fit/doses -- and converted here. The ratio is a property of this (model, target, mode) and
    # is reported, not hidden.
    _proxy_base, _where0, _q0 = locate(np.zeros(C0['n_free']))
    _cal = float(s_crit_base) / float(_proxy_base)
    shape = np.geomspace(span[0], span[1], int(n_dose)) * _cal

    def window(v):
        S, where, q = locate(v)
        return shape * S, S, where, q

    def parts(v):
        v = jnp.asarray(v, jnp.float64)
        doses, S, where, q = window(v)
        c, zu, alive, amp_lc, T, cp, b, a = _score(v, jnp.asarray(doses))
        pen = w_anchor * anchor_penalty(where, q)
        return dict(total=float(c) + pen, c_ptc=float(cp), osc=float(b),
                    amp_pen=float(a), anchor_pen=pen,
                    disp_max=float(q['dmax']), s_star=float(S), where=where,
                    alive_frac=float(jnp.mean(alive.astype(jnp.float64))),
                    amp_lc=float(amp_lc), period=float(T),
                    dose_lo=float(doses[0]), dose_hi=float(doses[-1]))

    def total(v):
        v = jnp.asarray(v, jnp.float64)
        doses, _S, _w, _q = window(v)
        return float(_score_j(v, jnp.asarray(doses))) + w_anchor * anchor_penalty(_w, _q)

    def grad(v):
        v = jnp.asarray(v, jnp.float64)
        doses, _S, _w, _q = window(v)
        return np.asarray(_grad_j(v, jnp.asarray(doses)))

    def surface(v):
        v = jnp.asarray(v, jnp.float64)
        doses, _S, _w, _q = window(v)
        _c, zu, alive, _a, _T, _cp, _b, _ap = _score(v, jnp.asarray(doses))
        return np.asarray(zu), np.asarray(alive), np.asarray(doses)

    return dict(proxy_base=_proxy_base, calibration=_cal,
                v0=C0['v0'], n_free=C0['n_free'], names=C0['names'], B=C0['B'],
                z_base=C0['z_base'], theta=theta, parts=parts, total=total, grad=grad,
                surface=surface, window=window, locate=locate, probe_doses=probe_doses,
                anchor_penalty=anchor_penalty, w_anchor=w_anchor,
                shape=shape, old=old, s_crit_base=s_crit_base, base_cost=C0)


def selftest(model_name='almeida', target='BMAL1'):
    from models import get_model
    m = get_model(model_name)
    m.reference_variable = 'PER'
    ok = True

    # --- 1. FIDELITY: _surface_at must reproduce make_cost's own surface exactly ---------- #
    from fit.doses import fit_dose_grid
    doses, S = fit_dose_grid(model_name, target, 'instant', 8.0, 14, lo_factor=0.5)
    C = make_cost(m, target, doses, RadialTarget(), n_phase=20, mode='instant',
                  backend='diffrax', dt=0.02, w_osc=0.0, w_amp=0.0, readout_ref='REV',
                  amp_ramp=None)
    from engine.orbit import make_guess_fn
    C['guess'] = make_guess_fn(m); C['y_seed'] = jnp.asarray(m.get_initial_state(), jnp.float64)
    C['ref_idx'] = int(m.var_index(m.reference_variable))
    C['T_nom'] = float(getattr(m, 'approx_period', None) or 24.0); C['model'] = m
    v = C['v0']
    zu_ref, al_ref, _amp = C['surface'](v)
    P = m.jax_apply(jnp.asarray(C['theta'](v)), list(C['names']))
    zu, al, _a, _al, _T = _surface_with(C, P, v, doses, 20)
    d_z = float(np.max(np.abs(np.asarray(zu) - np.asarray(zu_ref))))
    d_a = int(np.sum(np.asarray(al) != np.asarray(al_ref)))
    # 1e-8, not machine epsilon: the two paths are the same computation under DIFFERENT jit
    # contexts, so floating-point reassociation alone puts them ~1e-10 apart. What must be
    # exact is the alive mask, since that is what decides whether a cell is scored at all.
    print(f"  [fidelity]   max|dz| {d_z:.2e}   alive mismatches {d_a}   "
          f"{'PASS' if d_z < 1e-8 and d_a == 0 else 'FAIL'}")
    ok &= (d_z < 1e-8 and d_a == 0)

    # --- 2. the anchor, and the cost at base --------------------------------------------- #
    R = make_relative_cost(m, target, n_phase=20, n_dose=14, n_probe=14, s_crit_base=S,
                           readout_ref='REV', amp_ramp=None)
    p = R['parts'](R['v0'])
    print(f"  [calib]      proxy/S_crit = {R['calibration']:.4g}^-1  "
          f"(proxy {R['proxy_base']:.4g} vs S_crit {S:.4g})")
    print(f"  [window]     {p['dose_lo']/S:.2f} .. {p['dose_hi']/S:.2f} x TRUE S_crit "
          f"{'PASS' if p['dose_hi']/S < 18 else 'FAIL -- inside hazard 11'}")
    ok &= (p['dose_hi'] / S < 18)
    print(f"  [base]       S* {p['s_star']:.4g} ({p['where']})  window "
          f"{p['dose_lo']:.4g}..{p['dose_hi']:.4g}  c_ptc {p['c_ptc']:.5f}  "
          f"alive {p['alive_frac']:.3f}")
    ok &= p['alive_frac'] > 0.99

    # --- 3. CONTINUITY -- the criterion that actually gates this design ------------------ #
    #
    # The stated risk was that a per-candidate window makes the cost jump. That, not the
    # gradient, is what decides whether CMA can optimise it, and CMA is what the campaign uses.
    g = R['grad'](R['v0'])
    i = int(np.argmax(np.abs(g)))
    ds = np.array([-2e-3, -1e-3, -5e-4, 0.0, 5e-4, 1e-3, 2e-3])
    Ss, Cs = [], []
    for d in ds:
        vv = np.array(R['v0'], float); vv[i] += d
        Ss.append(R['locate'](vv)[0]); Cs.append(R['total'](vv))
    Ss, Cs = np.array(Ss), np.array(Cs)
    stepC = np.abs(np.diff(Cs)); stepS = np.abs(np.diff(np.log10(Ss)))
    smooth = (stepC.max() / max(np.median(stepC), 1e-30) < 4.0
              and stepS.max() / max(np.median(stepS), 1e-30) < 4.0)
    print(f"  [continuity] along v[{i}]: S* {Ss.min():.4f}..{Ss.max():.4f}, cost "
          f"{Cs.min():.6f}..{Cs.max():.6f}")
    print(f"               step ratios max/median -- S* {stepS.max()/np.median(stepS):.2f}, "
          f"cost {stepC.max()/np.median(stepC):.2f}   {'PASS -- no jumps' if smooth else 'FAIL'}")
    ok &= smooth

    # --- 4. THE GRADIENT BIAS, measured rather than assumed ------------------------------ #
    #
    # autodiff holds the window fixed; these finite differences MOVE it, because `total`
    # re-locates S*. The gap is exactly the dropped dC/dS* . dS*/dv term. It is REPORTED, not
    # asserted on: a bias here disqualifies L-BFGS and LM, and leaves CMA and BOBYQA -- which
    # never call `grad` -- completely unaffected.
    h, worst = 1e-3, 0.0
    for k in np.argsort(-np.abs(g))[:4]:
        vp = np.array(R['v0'], float); vp[k] += h
        vm = np.array(R['v0'], float); vm[k] -= h
        fd = (R['total'](vp) - R['total'](vm)) / (2 * h)
        worst = max(worst, abs(fd - g[k]) / max(abs(fd), abs(g[k]), 1e-12))
    print(f"  [grad bias]  |grad| {np.linalg.norm(g):.3e}; worst gap between fixed-window "
          f"autodiff and window-moving FD: {worst:.2f}")
    print(f"               -> GRADIENT-FREE OPTIMIZERS ONLY (cma, bobyqa). `total` is correct "
          f"and smooth; `grad` is biased by the dropped term.")
    print("  selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('THE IDEA')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.model, a.target)
    ap.error('nothing to do -- pass --selftest')


if __name__ == '__main__':
    raise SystemExit(main())
