"""
analysis/observable.py
======================
WHICH SPECIES SHOULD CARRY THE POINCARE SECTION, AND WHICH SHOULD CARRY THE PHASE?

    $PY -m analysis.observable --model korencic [--mode instant] [--target Bmalx]

These are TWO roles with DIFFERENT criteria, and conflating them cost this project a
retraction (PROJECT_SUMMARY 5.7, REPO_MAP hazard 14). Almeida used `reference_variable =
'BMAL1'` for both, chosen on its BASE-POINT numbers -- largest relative amplitude, cleanly
unimodal. Neither property survived the parameter sets an optimizer actually visited:

    at the RAD01 optimum   BMAL1 min 1.2e-27   ROR 4.1e-28   E4BP4 3.8e-17   REV max 1.2e+03

Thirty decades inside one state vector. The section stopped being unimodal, so Newton landed
wherever the guess pointed it and the SAME parameter set returned a period of 0.159 h from one
guess and 43.1 h from another. The dynamics were fine the whole time.

THE MEASUREMENT THIS MODULE MAKES, AND WHY IT IS NOT THE BASE POINT
    A base-point ranking is what produced the wrong answer. What matters is behaviour over the
    region a SEARCH moves through, so the sample is drawn from the gauge-quotient box at a
    LADDER OF RADII -- |v| = 0.25 .. 3.0 -- rather than from the base point or from
    `fit/viability`'s accept-set. Viability is deliberately the wrong sample here: it keeps only
    healthy circadian clocks, and the section has to survive the points a CMA population VISITS,
    most of which viability would reject. Its `ok_orbit` gate is reused, its acceptance is not.

    Per species, per sampled point:
      rel_amp     range / |mean|      -- is there an oscillation to read at all
      base_ratio  min / |mean|        -- THE COLLAPSE SIGNATURE. `fit/viability` rejects a
                                         whole parameter set on this at 1e-3; here it is
                                         reported per species, because the question is WHICH
                                         species collapses, not whether one did.
      turns       sign changes of dy/dt around the closed cycle. 2 = unimodal. A section needs
                                         this to stay 2, or `rhs(y0)[ref] = 0` stops having a
                                         unique solution and Newton's answer depends on the
                                         guess.
      spread      decades between the smallest and largest state in the vector -- hazard 14's
                                         actual tell, and a property of the POINT, not a species

    SECTION wants robust unimodality. READOUT wants a baseline that never approaches zero.
    They are ranked separately and the module refuses to collapse them into one recommendation.

WHY SWITCHING THE READOUT IS FREE, AND WHY THAT IS CHECKED RATHER THAN ASSERTED
    Asymptotic phase is a property of the STATE, not of the species you watch, so with a long
    enough transient skip every observable reports the same PTC up to a constant offset -- and
    `engine/ptc` calibrates that offset away against an unperturbed cell. So a readout change
    should move the phase ORIGIN and nothing else. Almeida's three candidates agreed to 5.6e-4
    cyc, which is what licensed the switch and kept the earlier BMAL1-readout results standing.

    The readout check reruns the base PTC under each candidate and reports the worst
    disagreement. A candidate that does NOT agree is not offering a different origin -- its
    Fourier fundamental is not resolving the oscillation, which `phase_or_nan` guards against,
    and that is a disqualification rather than a preference.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                                          # noqa: E402

#: |v| ladder. Chosen to bracket `fit/viability`'s own measured alive fractions
#: (0.67 / 0.50 / 0.17 / 0.00 at 0.25 / 0.5 / 1.0 / 1.5), so the scan spans "almost always
#: alive" through "almost never" rather than sampling one regime N times.
RADII = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

#: A cycle is unimodal when dy/dt changes sign exactly twice around the closed loop.
UNIMODAL = 2

#: A readout is "safe" if its minimum never falls below this fraction of its own mean.
#: 1e-2 is an order above `fit/viability`'s whole-parameter-set rejection at 1e-3, because a
#: readout should have margin over the threshold at which a point stops being usable at all.
SAFE_BASE = 1e-2


def turns(profile):
    """Sign changes of the derivative around the CLOSED cycle, per species.

    `profile` is (n_states, m) sampled over one period, so the loop wraps and the difference
    must wrap with it -- np.diff alone would miss a turning point sitting on the seam and
    report an odd number, which is topologically impossible on a closed curve."""
    d = np.diff(np.concatenate([profile, profile[:, :1]], axis=1), axis=1)
    s = np.sign(d)
    out = np.zeros(profile.shape[0], int)
    for i in range(profile.shape[0]):
        nz = s[i][s[i] != 0]          # drop flat runs: a plateau between two samples is not
        if nz.size < 2:               # two crossings, and a constant species has no turns
            continue
        out[i] = int(np.sum(nz[1:] != nz[:-1]) + (nz[-1] != nz[0]))
    return out


def scan(model_name, n_per_radius=6, radii=RADII, m=128, seed=0):
    """Sample the quotient box at a ladder of radii and measure every species at every point.

    Points whose orbit is rejected are recorded with their status rather than dropped -- that a
    region kills the orbit is information, and REPO_MAP hazard 2 is why `make_orbit_finder` is
    used rather than a bare BVP residual (an equilibrium satisfies the BVP perfectly)."""
    from models import get_model
    from engine.orbit import make_orbit_finder
    from fit.cost import quotient_basis

    model = get_model(model_name)
    names, zb, B, _g = quotient_basis(model)
    n_free = B.shape[1]
    states = list(model.state_names)
    find, _s = make_orbit_finder(model, m=m)
    rng = np.random.default_rng(seed)

    print(f"[observable] {model_name}: {len(states)} states, {len(names)} parameters, "
          f"{n_free} free directions after the gauge quotient", flush=True)

    todo = [(0.0, np.zeros(n_free))]
    for r in radii:
        for _k in range(n_per_radius):
            u = rng.normal(size=n_free)
            todo.append((r, r * u / np.linalg.norm(u)))
    print(f"[observable] {len(radii)} radii x {n_per_radius} draws + base = {len(todo)} points",
          flush=True)

    rows, prof, t0 = [], [], time.time()
    for i, (r, v) in enumerate(todo):
        tk = time.time()
        P = model.jax_apply(np.exp(zb + B @ v), names)
        try:
            _x0, T, C, st = find(P)
        except Exception as e:                       # report it, never swallow it -- hazard 7
            print(f"    |v|={r:4.2f} orbit RAISED {type(e).__name__}: {e}", flush=True)
            T, C, st = np.nan, None, 'error'
        Tf = float(T) if T is not None and np.isfinite(T) else np.nan
        row = dict(radius=float(r), status=str(st), period=Tf)
        if C is not None:
            C = np.asarray(C, float)
            mean = np.abs(C.mean(1))
            row['rel_amp'] = (C.max(1) - C.min(1)) / np.maximum(mean, 1e-30)
            row['base_ratio'] = C.min(1) / np.maximum(mean, 1e-30)
            row['turns'] = turns(C)
            nzero = np.abs(C)[np.abs(C) > 0]
            row['spread'] = (np.log10(float(np.max(np.abs(C))) / float(nzero.min()))
                             if nzero.size else np.nan)
            prof.append(C.astype(np.float32))        # SAVE THE RAW ARRAYS -- hazard 9
        else:
            row['rel_amp'] = np.full(len(states), np.nan)
            row['base_ratio'] = np.full(len(states), np.nan)
            row['turns'] = np.full(len(states), -1)
            row['spread'] = np.nan
            prof.append(np.full((len(states), m), np.nan, np.float32))
        rows.append(row)
        el = time.time() - t0
        print(f"  |v|={r:4.2f} [{i + 1:3d}/{len(todo)}] {str(st):>12s}  T={Tf:8.3f}  "
              f"spread={row['spread']:5.2f} dec  {time.time() - tk:5.1f}s  "
              f"| {el / 60:4.1f} min elapsed, eta "
              f"{el / (i + 1) * (len(todo) - i - 1) / 60:4.1f} min", flush=True)

    return dict(model=model_name, states=np.array(states), radii=np.array(radii),
                n_per_radius=n_per_radius, seed=seed, m=m,
                radius=np.array([r['radius'] for r in rows]),
                status=np.array([r['status'] for r in rows]),
                period=np.array([r['period'] for r in rows]),
                spread=np.array([r['spread'] for r in rows]),
                rel_amp=np.array([r['rel_amp'] for r in rows]),
                base_ratio=np.array([r['base_ratio'] for r in rows]),
                turns=np.array([r['turns'] for r in rows]),
                profiles=np.array(prof)), model


def rank(blob):
    """Print the two rankings and return (section_candidates, readout_candidates).

    'ok' points only. A rejected orbit has no cycle to measure a species on, and averaging a
    NaN into a ranking is how a species that never oscillates wins one."""
    states = [str(s) for s in blob['states']]
    ok = blob['status'] == 'ok'
    n_ok = int(ok.sum())
    print(f"\n{'=' * 78}\nOBSERVABLE SCAN -- {blob['model']}\n{'=' * 78}")
    print(f"  {n_ok} of {len(ok)} sampled points have a usable orbit")
    if not n_ok:
        print("  nothing to rank")
        return [], []
    for r in blob['radii']:
        sel = np.isclose(blob['radius'], r)
        print(f"    |v| = {r:4.2f}   {int((sel & ok).sum())}/{int(sel.sum())} alive")
    sp = blob['spread'][ok]
    print(f"  state-vector spread over live points: median {np.nanmedian(sp):.1f} decades, "
          f"worst {np.nanmax(sp):.1f}")

    ta, ba, ra = blob['turns'][ok], blob['base_ratio'][ok], blob['rel_amp'][ok]
    uni = (ta == UNIMODAL).mean(0)
    worst_base = np.nanmin(ba, axis=0)
    med_amp = np.nanmedian(ra, axis=0)

    print(f"\n  {'species':<12s} {'unimodal':>9s} {'worst base/mean':>16s} "
          f"{'median rel amp':>15s}   role")
    for i in sorted(range(len(states)), key=lambda k: -uni[k]):
        note = []
        if uni[i] >= 0.95:
            note.append('SECTION ok')
        if worst_base[i] >= SAFE_BASE:
            note.append('READOUT ok')
        print(f"  {states[i]:<12s} {uni[i] * 100:8.0f}% {worst_base[i]:16.3e} "
              f"{med_amp[i]:15.3f}   {', '.join(note) or '--'}")

    # SECTION: unimodality first, amplitude as the tie-break -- a flat unimodal species gives a
    # well-posed section that is numerically useless.
    sec = sorted(range(len(states)), key=lambda i: (-uni[i], -med_amp[i]))
    # READOUT: worst-case baseline first. This is the quantity that reached 1e-27 on Almeida.
    rdo = sorted(range(len(states)), key=lambda i: (-worst_base[i], -med_amp[i]))
    print(f"\n  SECTION candidates (robust unimodality, then amplitude): "
          f"{', '.join(states[i] for i in sec[:4])}")
    print(f"  READOUT candidates (safest baseline, then amplitude):     "
          f"{', '.join(states[i] for i in rdo[:4])}")
    print("  These are SEPARATE roles -- see the header. Do not collapse them.")
    return [states[i] for i in sec[:4]], [states[i] for i in rdo[:4]]


def _circ(d):
    """Shortest signed arc of a phase difference, in cycles. Phase wraps; a plain subtraction
    invents jumps of a whole cycle at the seam."""
    return np.angle(np.exp(2j * np.pi * np.asarray(d))) / (2 * np.pi)


def check_readout(model_name, target, mode, candidates, n_phase=16, n_dose=4, scrit_tag=None):
    """Do the candidate readouts report the SAME PTC at base, up to the calibrated origin?"""
    from models import get_model
    from engine.ptc import recommended_skip
    from analysis.ptc_sens import load_grid, make_ptc_at

    model = get_model(model_name)
    grid, dt = load_grid(model_name, target, mode, scrit_tag)
    if grid is None:
        raise SystemExit(f"no dose grid for {model_name}/{target}/{mode}; "
                         f"run `analysis.scrit --model {model_name} --mode {mode}` first")
    doses = np.asarray(grid)[np.linspace(0, len(grid) - 1, n_dose).astype(int)]
    skip_p, mu, _r = recommended_skip(model, tol=1e-2, verbose=False)
    print(f"\n{'=' * 78}\nREADOUT EQUIVALENCE -- {model_name} / {target} ({mode})\n{'=' * 78}")
    print(f"  mu = {mu:.4f} -> skip_p = {skip_p}; dt = {dt}; {n_phase} phases; doses "
          f"{np.array2string(doses, precision=3)}", flush=True)

    base_params = model.get_parameters()
    surf = {}
    for c in candidates:
        t0 = time.time()
        ptc_of = make_ptc_at(model, target, mode, doses, n_phase, dt, skip_p, readout_ref=c)
        p, _a, _v, _y0, st = ptc_of(base_params)
        surf[c] = p
        print(f"  {c:<12s} {str(st):>6s}  {time.time() - t0:5.1f}s", flush=True)

    ref = candidates[0]
    print(f"\n  {'candidate':<12s} {'rms vs ' + ref:>16s} {'max':>10s}   origin offset")
    worst = 0.0
    for c in candidates[1:]:
        a, b = surf[ref], surf[c]
        if a is None or b is None:
            print(f"  {c:<12s} {'no surface':>16s}")
            continue
        good = np.isfinite(a) & np.isfinite(b)
        if not good.any():
            print(f"  {c:<12s} {'no live cells':>16s}")
            continue
        # A constant offset between two readouts is EXPECTED, not a disagreement: each is
        # calibrated to its own dose-0 identity. Compare after removing the circular mean.
        d = _circ(a[good] - b[good])
        off = np.angle(np.mean(np.exp(2j * np.pi * d))) / (2 * np.pi)
        e = _circ(d - off)
        rms, mx = float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))
        worst = max(worst, mx)
        print(f"  {c:<12s} {rms:16.2e} {mx:10.2e}   {off:+.4f} cyc")
    print(f"\n  worst disagreement {worst:.2e} cyc. Almeida's three candidates agreed to "
          f"5.6e-04 cyc,\n  which is what made switching its readout free and kept the "
          f"earlier results standing.")
    return surf, worst


def main(argv=None):
    ap = argparse.ArgumentParser(description='section vs readout: which species for which role')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'korencic'))
    ap.add_argument('--mode', default='instant', choices=('instant', 'pulse'))
    ap.add_argument('--target', default=None,
                    help='target whose dose grid the readout check borrows '
                         '(default: the first in-scope target)')
    ap.add_argument('--n', type=int, default=6, help='draws per radius')
    ap.add_argument('--m', type=int, default=128, help='cycle sample points')
    ap.add_argument('--n-phase', type=int, default=16)
    ap.add_argument('--n-dose', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scrit-tag', default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--no-readout-check', action='store_true')
    a = ap.parse_args(argv)

    blob, model = scan(a.model, n_per_radius=a.n, m=a.m, seed=a.seed)
    sec, rdo = rank(blob)
    blob['section_candidates'] = np.array(sec)
    blob['readout_candidates'] = np.array(rdo)

    if not a.no_readout_check and rdo:
        from analysis import genemap as GM
        target = a.target or (GM.scope_targets(a.model) or list(model.state_names))[0]
        _s, worst = check_readout(a.model, target, a.mode, rdo,
                                  n_phase=a.n_phase, n_dose=a.n_dose, scrit_tag=a.scrit_tag)
        blob['readout_check_target'] = target
        blob['readout_check_worst'] = worst

    out = paths.out_path(a.model, 'observable', f'observable_{a.mode}.npz',
                         paths.run_tag(a.tag))
    paths.savez(out, **blob)
    print(f"\n[observable] -> {out}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
