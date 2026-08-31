"""
fit/stability.py
================
IS THE THING THE BVP RETURNED ACTUALLY WHERE THE SYSTEM LIVES?

    python -m fit.stability --model almeida --tag bmal1seeds          # audit a campaign
    python -m fit.stability --model almeida --tag genes --kind genes
    python -m fit.stability --selftest                                # known answers

THE FAILURE THIS EXISTS TO CATCH
    A converged periodic-orbit solve says `phi_T(y0) = y0` to machine precision. It says
    NOTHING about whether a real trajectory ever goes there. Three distinct objects satisfy it,
    and one period of the solved cycle looks the same in all three:

      1. an ATTRACTING limit cycle -- what every PTC in this project assumes;
      2. a REPELLING limit cycle -- a genuine periodic orbit that nothing stays on. Asymptotic
         phase does not exist on it, so the entire PTC construction is measuring the transient
         to wherever the trajectory actually goes;
      3. a periodic orbit coexisting with a STABLE FIXED POINT that wins. The plotted cycle is
         real and the system collapses to a point anyway.

    REPO_MAP hazard 2 catches the crude version (an equilibrium has no amplitude; a
    negative-orthant runaway has no positivity). It does not catch these: all three have a
    healthy amplitude, a circadian period, positive concentrations and a residual of 1e-13.

WHY NOT THE FLOQUET MULTIPLIER
    Because it is measured and untrustworthy here. `solver.floquet` forms the monodromy, whose
    entries grow like exp(lambda*T): on the Aug-30 campaign it returned a NON-FINITE matrix for
    three runs (which then read as DEGENERATE, PROJECT_SUMMARY 5.10f), 2.35e6 for the PER
    optimum, and 0.5146 for that SAME point on recompute. A diagnostic that disagrees with
    itself cannot be the guard.

    But it is trustworthy AT THE BASE POINT, where the monodromy is well conditioned -- so that
    is where this module is cross-checked. `--selftest` asserts the walk recovers the published
    0.546 to within 10%; it measures 0.549. Agreement there is what licenses using the walk at
    the parameter sets where Floquet cannot be computed at all.

WHAT THIS DOES INSTEAD: IT INTEGRATES AND WATCHES
    Perturb the cycle, integrate forward for many periods IN BLOCKS OF ONE PERIOD, and track
    the transverse distance back to the cycle, the amplitude, and |rhs|. The verdict is read off
    where the trajectory ended up:

        ATTRACTING     the deviation decays geometrically
        REPELLING      it grows
        LEAVES ORBIT   still oscillating, but far from the solved orbit -- another attractor
        DECAYS TO FP   amplitude collapses and |rhs| -> 0: it settled on a POINT
        DIVERGES       the state runs away
        UNRESOLVED     no usable perturbation size exists (see below) -- NOT a verdict
        SOLVER LIMIT   the integrator gave up; nothing was measured

    Blocks matter: a diffrax MAX_STEPS failure NaNs a whole call, so one long integration
    renders identically to a blow-up -- which is how a 517 h solve was once read as "the model
    diverges" (hazard 15). Blocks localise it, and it is reported as a solver limit, never as a
    dynamical fact.

THE THING THAT MAKES THIS HARD, AND THAT THE FIRST VERSION GOT WRONG
    **Every orbit has its own numerical floor, and it spans five orders of magnitude.** Walk
    from y0 with NO perturbation at all: the trajectory starts exactly on the cycle, so any
    distance it accumulates is pure integration error. On Almeida that is 8e-9 of the cycle
    diameter at the base point and **7.6e-2 at the seed-5 optimum**. A perturbation below the
    floor measures nothing but noise.

    `fit.cost.make_growth_fn` has exactly this bug at its default `eps = 1e-4`, and it is not
    academic -- it drove the previous conclusion. Its "growth" fell monotonically as eps rose,
    which a dynamical quantity cannot do:

        eps            1e-4    1e-3    1e-2    3e-2    1e-1
        base           0.857   0.857   0.871   0.824   0.695     <- a plateau: real
        seed 4         1.714   1.431   1.127   0.968   0.839     <- no plateau: noise
        PER optimum    1.386   1.063   0.808   0.729   0.659     <- no plateau: noise

    Read at eps=1e-4 that says two thirds of the campaign repels. Measured above each orbit's
    own floor, 10 of 16 attract cleanly and the PER optimum decays at 0.53 per period. The
    module therefore MEASURES the floor first, climbs an epsilon ladder until the perturbation
    clears it by 30x, and returns UNRESOLVED rather than a number when no rung does.

    A second, subtler floor sits above it: the reference cycle is a polyline, so distance to
    its VERTICES cannot fall below half a sample spacing (5e-4 diameters even at n_cyc=2048).
    `_dist_to_cycle` measures to the SEGMENTS, and the fit is restricted to the leading two
    decades so the residual chordal error cannot flatten the slope.

USE IT AS A GATE, NOT AS A REPORT
    `classify()` is the reusable predicate. `fit.cost.make_cost` already takes `w_stab` and
    `r_max` for the in-loop version -- but DO NOT enable it until `make_growth_fn`'s epsilon is
    fixed, or the fit will be penalising orbits for the integrator's error.
"""
import argparse
import json
import os

import numpy as np

import paths

#: per-period growth above this and the orbit is treated as REPELLING. Not 1.0: the measurement
#: has noise, and a marginally-attracting cycle is still a cycle. Measured spread on known-good
#: Almeida points is well inside this (base 0.816, REV optimum 0.552).
R_REPEL = 0.98
#: the walked amplitude may fall to this fraction of the cycle's before it counts as collapsed
AMP_COLLAPSE = 0.10
#: |rhs| / |rhs| on the cycle below this at the end of the walk means it stopped moving
RHS_SETTLED = 1e-3
#: distance to the cycle above this multiple of the cycle's own diameter is "ran away"
D_DIVERGE = 10.0
#: and above THIS it has demonstrably left the solved orbit, even while still oscillating --
#: 5% of the cycle's own diameter is far outside where a perturbation of eps=1e-3 belongs
D_LEAVE = 0.05

#: a numerical floor above this fraction of the cycle diameter means the orbit cannot be
#: integrated cleanly at all: starting exactly ON it, the trajectory wanders this far within a
#: few periods. Not a stability verdict -- a warning that nothing measured here is precise.
FLOOR_FRAGILE = 1e-3
#: fit the decay only over the leading two decades -- past that the polyline chordal error
#: flattens the sequence and biases the slope toward 1 (see `measure`)
TAIL_FRAC = 1e-2
#: and never over more than this many periods, so a slow second mode cannot dominate the fit
FIT_CAP = 12

VERDICTS = ('ATTRACTING', 'REPELLING', 'LEAVES ORBIT', 'DECAYS TO FP', 'DIVERGES',
            'UNRESOLVED', 'SOLVER LIMIT', 'NOT AN ORBIT')
VERDICT_COLOR = {'ATTRACTING': '#1a7f37', 'REPELLING': '#b31d28', 'LEAVES ORBIT': '#bc4c00',
                 'DECAYS TO FP': '#8250df', 'DIVERGES': '#a40e26', 'UNRESOLVED': '#8250df',
                 'SOLVER LIMIT': '#57606a', 'NOT AN ORBIT': '#57606a'}


def _dist_to_cycle(pts, cyc):
    """Distance from each point to the closed POLYLINE through `cyc` -- segments, not vertices.

    Measuring to the vertices instead puts a hard floor of about half a sample spacing under
    every reading: perimeter/(2*n_cyc), which is 5e-4 of the cycle diameter even at n_cyc=2048.
    A decaying deviation then stops decaying when it reaches that floor, and a straight-line fit
    through the flat tail returns a slope near zero -- i.e. growth ~1.0 for an orbit whose true
    multiplier is 0.55. That is not a small error: it is the difference between "attracts" and
    "marginal", on every orbit in the campaign.

    Point-to-segment removes it, because a point sitting ON the cycle between two samples is at
    distance ~0 from the segment joining them.
    """
    a = cyc                                   # (n, k)
    b = np.roll(cyc, -1, axis=0)              # closed: the last segment wraps
    ab = b - a                                # (n, k)
    denom = np.einsum('nk,nk->n', ab, ab)
    denom = np.where(denom > 0, denom, 1.0)
    pa = pts[:, None, :] - a[None, :, :]      # (m, n, k)
    t = np.clip(np.einsum('mnk,nk->mn', pa, ab) / denom, 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.min(np.linalg.norm(pts[:, None, :] - proj, axis=-1), axis=1)


def make_walker(model, backend='diffrax', m=64):
    """`walk(P, y0, T, u, eps, n_blocks) -> dict` -- integrate and watch, one period per block.

    Returns the per-period distance to the cycle, amplitude and |rhs|, plus how far the walk
    actually got before the integrator (not the dynamics) stopped it.
    """
    import jax
    import jax.numpy as jnp
    from engine.flow import make_window_sampler
    from engine.orbit import OrbitSolver

    sampler = make_window_sampler(model, backend=backend)
    solver = OrbitSolver(model)
    rhs = model.jax_rhs

    @jax.jit
    def _block(y, T, P):
        """One period forward. Returns the states and the end state."""
        ys = sampler(y, m, T / m, P)
        return ys, ys[-1]

    def walk(P, y0, T, u=None, eps=1e-3, n_blocks=24, n_cyc=2048):
        # 2048, not 256: the reference cycle is a POLYLINE and the distance is measured to its
        # vertices, so the discretisation puts a floor of about half a sample spacing under
        # every measurement. At 256 that floor is 4e-3 of the cycle diameter -- larger than the
        # deviations being measured -- and it is one of the two things that made the first
        # version of this module report repulsion for orbits that attract.
        cyc = np.asarray(solver.cycle(P, y0, T, n_cyc))
        scale = np.maximum(cyc.max(0) - cyc.min(0), 1e-12)
        diam = float(np.linalg.norm(scale))
        rhs_cyc = float(np.median([np.linalg.norm(np.asarray(rhs(jnp.asarray(c), P)))
                                   for c in cyc[::16]]))
        y = np.asarray(y0, float) + (eps * scale * np.asarray(u) if u is not None else 0.0)
        d, amp, rn, blocks = [], [], [], 0
        limit_at = -1
        for b in range(int(n_blocks)):
            ys, ye = _block(jnp.asarray(y), T, P)
            ys = np.asarray(ys)
            if not np.isfinite(ys).all():
                # THE INTEGRATOR STOPPED, NOT THE SYSTEM. Recorded as such and the walk ends;
                # never folded into a dynamical verdict.
                limit_at = b
                break
            # TRANSVERSE distance to the cycle, at the END of this period.
            #
            # Per sampled state, the min over CYCLE POINTS -- that quotients out the phase
            # direction, which is neutral and would otherwise floor any growth ratio at 1.
            # Then the LAST sample, so `dist[b]` is the deviation after b+1 periods and the
            # sequence is the power iteration.
            #
            # Taking the min over the block as well (the first version of this line) measures
            # "did the trajectory pass near the cycle at any point", which is ~0 for anything
            # in the neighbourhood and reported 1e-4 diameters for orbits whose growth was
            # 1.7 per period -- a contradiction that is how the bug was caught.
            d.append(float(_dist_to_cycle(ys[-1:], cyc)[0]))
            amp.append(float(np.linalg.norm(ys.max(0) - ys.min(0))))
            rn.append(float(np.linalg.norm(np.asarray(rhs(jnp.asarray(ys[-1]), P)))))
            y = np.asarray(ye, float)
            blocks = b + 1
            if not np.isfinite(y).all() or np.max(np.abs(y)) > 1e12:
                limit_at = -2                      # genuine runaway, distinct from a solver cap
                break
        return dict(dist=np.array(d), amp=np.array(amp), rhs=np.array(rn),
                    blocks=blocks, limit_at=limit_at, diam=diam,
                    amp_cycle=float(np.linalg.norm(cyc.max(0) - cyc.min(0))),
                    rhs_cycle=rhs_cyc)

    return walk


def measure(walk, P, y0, T, n_states, n_blocks=24, ndir=3, seed=0,
            eps_ladder=(1e-2, 3e-2, 1e-1, 3e-1), floor_margin=30.0, fit_margin=5.0):
    """Per-period growth, measured ABOVE THE NUMERICAL FLOOR OF THIS PARAMETER SET.

    THE FLOOR IS NOT A CONSTANT AND IT MUST BE MEASURED, NOT ASSUMED. Walk the orbit from y0
    with NO perturbation at all: the trajectory starts exactly on the cycle, so whatever
    distance it accumulates is pure integration error. On Almeida that is 4e-8 of the cycle
    diameter at the base point and **8.7e-3 at the seed-4 optimum** -- five orders apart, on
    the same integrator, at the same tolerances. A fixed epsilon cannot serve both.

    THIS IS THE BUG THAT MADE THE FIRST VERSION OF THIS MODULE WRONG, and `fit.cost`'s
    `make_growth_fn` has it too. Its default `eps = 1e-4` is BELOW seed 4's floor, so its
    "growth" was the ratio of two noise measurements. Symptom: growth fell monotonically as eps
    rose, which a dynamical quantity cannot do --

        eps            1e-4    1e-3    1e-2    3e-2    1e-1
        base           0.857   0.857   0.871   0.824   0.695     <- a plateau: real
        seed 4         1.714   1.431   1.127   0.968   0.839     <- no plateau: noise
        PER optimum    1.386   1.063   0.808   0.729   0.659     <- no plateau: noise

    -- and at an epsilon that clears the floor, both "repelling" orbits decay geometrically at
    about 0.52-0.57 per period. They attract. The base point's own decay, measured this way, is
    0.57 against the published Floquet multiplier of 0.546.

    So: climb an epsilon ladder until the initial deviation clears `floor_margin` x floor, then
    fit the slope of log(distance) over the periods where it is still above `fit_margin` x
    floor. If no rung clears the floor, return NaN and let the caller say UNRESOLVED. A guard
    that cannot measure must say so, not guess.
    """
    g = np.random.default_rng(1234 + int(seed))
    U = g.normal(size=(int(ndir), int(n_states)))
    U /= np.linalg.norm(U, axis=1, keepdims=True)

    base = walk(P, y0, T, u=None, n_blocks=max(8, n_blocks // 3))
    floor = float(np.max(base['dist']) / base['diam']) if len(base['dist']) else np.inf
    if base['limit_at'] >= 0:
        return dict(growth=np.nan, floor=np.inf, eps=np.nan, walk=base, floor_walk=base,
                    n_fit=0)

    best = None
    for eps in eps_ladder:
        cand = []
        for u in U:
            w = walk(P, y0, T, u=u, eps=eps, n_blocks=n_blocks)
            if not len(w['dist']):
                continue
            d = w['dist'] / w['diam']
            cand.append((d[0], w, d))
        if not cand:
            continue
        # the direction that got FURTHEST from the orbit -- one escaping direction is enough to
        # make an orbit unusable, and averaging over directions would hide it
        d0, w, d = max(cand, key=lambda c: c[2][-1])
        cleared = d0 > floor_margin * floor
        # keep the LARGEST epsilon tried, so an UNRESOLVED report names the strongest
        # perturbation that still failed to clear the floor rather than the weakest
        best = (eps, w, d)
        if cleared:
            break
    if best is None:
        return dict(growth=np.nan, floor=floor, eps=np.nan, walk=base, floor_walk=base,
                    n_fit=0)
    eps, w, d = best
    # TWO CEILINGS ON THE FIT WINDOW, and both are needed.
    #
    #   * `fit_margin * floor` -- below the orbit's own integration noise the sequence is not a
    #     measurement at all.
    #   * `d[0] * TAIL_FRAC`   -- and ABOVE the floor there is a second, subtler plateau: the
    #     reference cycle is a polyline, so once the deviation is comparable to its chordal
    #     error the distance stops shrinking. Measured on the Almeida base at n_cyc=2048 that
    #     happens near 1e-6 diameters, where the per-period ratio drifts from the true 0.55 up
    #     to ~0.87. Fitting the whole 24-period run through that tail returned 0.727 for an
    #     orbit whose published Floquet multiplier is 0.546; restricted to the leading two
    #     decades it returns 0.55.
    keep = (d > fit_margin * floor) & (d > d[0] * TAIL_FRAC)
    k = int(np.argmax(~keep)) if (~keep).any() else len(d)
    k = min(k, FIT_CAP)
    if k < 3 or d[0] <= floor_margin * floor:
        return dict(growth=np.nan, floor=floor, eps=eps, walk=w, floor_walk=base, n_fit=k)
    sl = np.polyfit(np.arange(k), np.log(d[:k]), 1)[0]
    return dict(growth=float(np.exp(sl)), floor=floor, eps=eps, walk=w, floor_walk=base,
                n_fit=k)


def classify(mres, orbit_ok=True):
    """THE GUARD. `mres` is a `measure()` result. -> (verdict, why).

    Ordered so a MEASUREMENT FAILURE can never be reported as a dynamical fact. The first three
    branches are all "we did not measure this", and each is a different reason:

        NOT AN ORBIT   the BVP never converged
        SOLVER LIMIT   the integrator gave up part-way
        UNRESOLVED     no perturbation on the ladder cleared this orbit's own numerical floor,
                       so any growth number would be the ratio of two noise measurements --
                       which is exactly what the first version of this module reported as
                       repulsion for three orbits that in fact attract.

    Only then are the dynamical verdicts asked, and they come from the WALK (where the
    trajectory actually ended up over many periods), not from an eigenvalue.
    """
    res = mres['walk']
    growth, floor = mres['growth'], mres['floor']
    if not orbit_ok:
        return 'NOT AN ORBIT', 'the BVP did not converge to a periodic orbit'
    if res['limit_at'] >= 0:
        return ('SOLVER LIMIT',
                f"the integrator gave up after {res['limit_at']} period(s); nothing about "
                f"stability was measured here")
    d, a, diam = res['dist'], res['amp'], res['diam']
    d_end = float(d[-1] / diam) if len(d) and np.isfinite(diam) and diam > 0 else np.nan
    if res['limit_at'] == -2 or (np.isfinite(d_end) and d_end > D_DIVERGE):
        return 'DIVERGES', f"the state ran away ({d_end:.1f} cycle diameters from the orbit)"
    if len(a) and a[-1] < AMP_COLLAPSE * res['amp_cycle']             and res['rhs'][-1] < RHS_SETTLED * max(res['rhs_cycle'], 1e-30):
        return ('DECAYS TO FP',
                f"amplitude fell to {a[-1] / res['amp_cycle']:.1%} of the cycle's and |rhs| to "
                f"{res['rhs'][-1] / max(res['rhs_cycle'], 1e-30):.1e} of it -- it settled on a "
                f"POINT. The solved orbit is real; nothing stays on it.")
    if not np.isfinite(growth):
        return ('UNRESOLVED',
                f"no perturbation up to the top of the ladder cleared this orbit's numerical "
                f"floor of {floor:.1e} diameters, so growth is not measurable here")
    if np.isfinite(d_end) and d_end > D_LEAVE:
        return ('LEAVES ORBIT',
                f"still oscillating after {len(d)} periods but {d_end:.2g} cycle diameters away "
                f"from the solved orbit -- it went to a DIFFERENT attractor, and the PTC was "
                f"computed on this one")
    if growth > R_REPEL:
        return ('REPELLING',
                f"deviation grows {growth:.4f} per period, fitted over {mres['n_fit']} periods "
                f"at eps={mres['eps']:.0e} against a floor of {floor:.1e}")
    return ('ATTRACTING',
            f"deviation decays {growth:.4f} per period (fitted over {mres['n_fit']} periods at "
            f"eps={mres['eps']:.0e}, floor {floor:.1e}); after {len(d)} periods it is "
            f"{d_end:.2g} diameters out")


def audit_one(model, theta, names, section, n_blocks=24, ndir=3, eps=1e-3, backend='diffrax',
              seed=0):
    """Re-solve one parameter set ON THE PRODUCTION PATH, then integrate and watch."""
    import jax
    import jax.numpy as jnp
    from engine.orbit import OrbitSolver, make_guess_fn
    from fit.cost import make_growth_fn

    m = model
    if section:
        m.reference_variable = section
    solver, guess = OrbitSolver(m, ref=section or None), make_guess_fn(m, ref=section or None)
    y_seed = jnp.asarray(m.get_initial_state(), jnp.float64)
    P = m.jax_apply(np.asarray(theta), list(names))
    y0, T, r = jax.jit(solver.solve)(P, guess(P, y_seed))
    y0, T, r = np.asarray(y0), float(T), float(r)
    T_nom = float(getattr(m, 'approx_period', None) or 24.0)
    cyc = np.asarray(solver.cycle(P, jnp.asarray(y0), T, 256))
    orbit_ok = bool(r < 1e-4 and np.isfinite(T) and 0.25 * T_nom < T < 4.0 * T_nom
                    and np.isfinite(cyc).all() and cyc.min() >= -1e-8)

    out = dict(period=T, orbit_res=r, orbit_ok=orbit_ok, y0=y0, cyc=cyc)
    if not orbit_ok:
        out.update(growth=np.nan, growth_cost=np.nan, floor=np.nan, eps_used=np.nan, n_fit=0,
                   verdict='NOT AN ORBIT', why='BVP did not converge',
                   dist=np.zeros(0), amp=np.zeros(0), rhs=np.zeros(0), floor_dist=np.zeros(0),
                   blocks=0, limit_at=-3, diam=np.nan, amp_cycle=np.nan, rhs_cycle=np.nan)
        return out

    walk = make_walker(m, backend=backend)
    mres = measure(walk, P, y0, T, m.n_states, n_blocks=n_blocks, ndir=ndir, seed=seed)
    v, why = classify(mres, orbit_ok)
    w = mres['walk']
    # `make_growth_fn`'s own number, at ITS default eps, kept ONLY so the two can be compared:
    # this module exists partly because that one is contaminated below the floor, and the
    # comparison is the evidence (see `measure`). Never used for the verdict.
    growth_cost = float(np.asarray(make_growth_fn(m, backend=backend)(P, jnp.asarray(y0), T)))
    out.update(growth=mres['growth'], growth_cost=growth_cost, floor=mres['floor'],
               eps_used=mres['eps'], n_fit=mres['n_fit'], verdict=v, why=why,
               floor_dist=mres['floor_walk']['dist'] / max(mres['floor_walk']['diam'], 1e-30),
               **{k: w[k] for k in
                  ('dist', 'amp', 'rhs', 'blocks', 'limit_at', 'diam', 'amp_cycle',
                   'rhs_cycle')})
    return out


def audit(model_name, tag, analysis='fit_radial', kind='seeds', n_blocks=24, ndir=3,
          eps=1e-3, backend='diffrax', verbose=True):
    """Audit every run of an aggregated campaign. Returns the blob that gets saved."""
    from models import get_model
    import glob
    d = paths.out_dir(model_name, analysis, tag, create=False)
    pat = 'seeds_*.npz' if kind == 'seeds' else 'genes_*.npz'
    fs = sorted(glob.glob(os.path.join(d, pat)))
    if not fs:
        raise SystemExit(f"no aggregated {pat} under {d}. Run `python -m fit.aggregate "
                         f"--model {model_name} --tag {tag}"
                         + ('' if kind == 'seeds' else ' --kind genes') + "` first.")
    z = dict(np.load(fs[-1], allow_pickle=True))
    labels = [str(x) for x in z['labels']]
    names = [str(x) for x in z['names']]
    section = str(z['section']) if 'section' in z else ''
    model = get_model(model_name)
    n = len(labels)
    rows, curves = [], []
    print(f"[stability] auditing {n} run(s) of {tag}: {n_blocks} periods x {ndir} directions "
          f"each, backend={backend}", flush=True)
    import time
    t0 = time.time()
    for i, lab in enumerate(labels):
        r = audit_one(model, z['theta_fit'][i], names, section, n_blocks=n_blocks,
                      ndir=ndir, eps=eps, backend=backend, seed=i)
        rows.append(r)
        curves.append(r)
        if verbose:
            el = time.time() - t0
            print(f"  [{i + 1:2d}/{n}] {str(z['key']):>6s} {lab:>5s}  T={r['period']:7.3f} h  "
                  f"res={r['orbit_res']:.1e}  growth={r['growth']:.4f}  "
                  f"{r['verdict']:<12s}  ({el:.0f}s elapsed, "
                  f"~{el / (i + 1) * (n - i - 1):.0f}s left)", flush=True)

    L = max([len(r['dist']) for r in rows] + [len(r['floor_dist']) for r in rows] + [1])

    def pad(key):
        a = np.full((n, L), np.nan)
        for i, r in enumerate(rows):
            a[i, :len(r[key])] = r[key]
        return a

    blob = dict(labels=np.array(labels), key=np.asarray(str(z['key'])),
                target=np.asarray(z['target']), model=np.asarray(model_name),
                mode=np.asarray(z['mode']), campaign_tag=np.asarray(str(tag)),
                section=np.asarray(section), readout=np.asarray(str(z.get('readout', ''))),
                names=np.array(names), theta_fit=np.asarray(z['theta_fit']),
                # --- scalars ------------------------------------------------------------- #
                period=np.array([r['period'] for r in rows]),
                orbit_res=np.array([r['orbit_res'] for r in rows]),
                orbit_ok=np.array([r['orbit_ok'] for r in rows]),
                growth=np.array([r['growth'] for r in rows]),
                growth_cost=np.array([r['growth_cost'] for r in rows]),
                floor=np.array([r['floor'] for r in rows]),
                eps_used=np.array([r['eps_used'] for r in rows]),
                n_fit=np.array([r['n_fit'] for r in rows]),
                verdict=np.array([r['verdict'] for r in rows]),
                why=np.array([r['why'] for r in rows]),
                blocks=np.array([r['blocks'] for r in rows]),
                limit_at=np.array([r['limit_at'] for r in rows]),
                diam=np.array([r['diam'] for r in rows]),
                amp_cycle=np.array([r['amp_cycle'] for r in rows]),
                rhs_cycle=np.array([r['rhs_cycle'] for r in rows]),
                n_blocks=np.asarray(n_blocks), eps=np.asarray(eps), ndir=np.asarray(ndir),
                # --- RAW: the cycles and the walks (hazard 9) ---------------------------- #
                cyc=np.array([r['cyc'] for r in rows]),
                y0=np.array([r['y0'] for r in rows]),
                walk_dist=pad('dist'), walk_amp=pad('amp'), walk_rhs=pad('rhs'),
                floor_dist=pad('floor_dist'),
                obs=np.array(list(model.observable_states())),
                )
    # amplitude measures per species, on the cycle itself, in absolute units
    cyc = blob['cyc']
    blob['amp_species'] = cyc.max(1) - cyc.min(1)
    blob['mean_species'] = cyc.mean(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        blob['relamp_species'] = blob['amp_species'] / np.abs(blob['mean_species'])
    return blob


def report(b):
    n = len(b['labels'])
    print(f"\n{'=' * 96}\nSTABILITY AUDIT -- {b['campaign_tag']} ({n} runs)\n{'=' * 96}")
    print(f"  {str(b['key']):>6s} {'T (h)':>8s} {'growth':>8s} {'(cost)':>8s} "
          f"{'floor':>10s} {'eps':>7s} {'amp_lc':>9s} {'d_end':>10s} {'amp_end':>9s}  verdict")
    for i in np.argsort(b['period']):
        de = b['walk_dist'][i]
        de = de[np.isfinite(de)]
        r = (de[-1] / b['diam'][i]) if len(de) and np.isfinite(b['diam'][i]) else np.nan
        aa = b['walk_amp'][i]; aa = aa[np.isfinite(aa)]
        ar = (aa[-1] / b['amp_cycle'][i]) if len(aa) and b['amp_cycle'][i] else np.nan
        fr = '*' if b['floor'][i] > FLOOR_FRAGILE else ' '
        print(f"  {str(b['labels'][i]):>6s} {b['period'][i]:8.3f} "
              f"{b['growth'][i]:8.4f} {b['growth_cost'][i]:8.4f} {b['floor'][i]:10.1e}{fr}"
              f"{b['eps_used'][i]:6.0e} {b['amp_cycle'][i]:9.3g} {r:10.3g} {ar:9.3g}  "
              f"{b['verdict'][i]}")
    print()
    for v in VERDICTS:
        c = int(np.sum(b['verdict'] == v))
        if c:
            print(f"  {v:<14s} {c:2d}/{n}")
    bad = [i for i in range(n) if b['verdict'][i] not in ('ATTRACTING',)]
    for i in bad:
        print(f"\n  {str(b['labels'][i])}: {b['verdict'][i]} -- {b['why'][i]}")
    if not bad:
        print("\n  Every orbit attracts.")
    nf = int(np.sum(b['floor'] > FLOOR_FRAGILE))
    if nf:
        print(f"\n  {nf} orbit(s) marked * have a numerical floor above {FLOOR_FRAGILE:.0e} "
              f"diameters: started EXACTLY on the orbit, with no perturbation at all, the "
              f"trajectory still wanders that far within a few periods. Nothing measured on "
              f"them is precise, and it is the same fragility that makes `solver.guess` fail "
              f"on them (PROJECT_SUMMARY 5.10f).")
    dis = int(np.sum(np.isfinite(b['growth']) & np.isfinite(b['growth_cost'])
                     & ((b['growth'] > R_REPEL) != (b['growth_cost'] > R_REPEL))))
    if dis:
        print(f"\n  {dis} orbit(s) land on the OTHER SIDE of {R_REPEL} under "
              f"`fit.cost.make_growth_fn` at its default eps=1e-4 (the '(cost)' column). Below "
              f"an orbit's floor that number is the ratio of two noise measurements -- see "
              f"`measure`. Do not adopt w_stab until that default is fixed.")
    return b


def selftest():
    """The measurement against a case whose answer is known INDEPENDENTLY of this module.

    THE ANCHOR IS THE BASE POINT'S FLOQUET MULTIPLIER, 0.546, computed by `solver.floquet` and
    published in PROJECT_SUMMARY 1. At the base point the monodromy is well conditioned and that
    number is trustworthy; it is only at the optimizer's pathological parameter sets that it
    overflows. So the base point is exactly where the two methods CAN be cross-checked, and
    agreement there is what licenses using the walk where Floquet cannot be computed at all.

    The test asserts the measured decay is within 10% of 0.546 -- not merely "< 1". A guard that
    only had to get the SIDE right would have passed with the tail-contaminated fit that
    returned 0.727, and that bias is large enough to flip a marginal orbit.

    NOT ASSERTED, DELIBERATELY: that any particular fitted optimum repels. An earlier version of
    this selftest asserted the PER optimum was REPELLING on the strength of
    `make_growth_fn`'s 1.437, and that number turned out to be the module's own noise floor --
    so the test was pinning a bug in place. Whether a given optimum attracts is a RESULT, and
    results do not belong in a selftest.
    """
    from models import get_model
    from fit.cost import quotient_basis
    m = get_model('almeida')
    names, z_base, _B, _g = quotient_basis(m)
    MU_PUBLISHED = 0.546

    base = audit_one(m, np.exp(z_base), names, 'PER', n_blocks=24, ndir=3)
    g = base['growth']
    err = abs(g - MU_PUBLISHED) / MU_PUBLISHED
    print(f"  base point:  walk gives {g:.4f} per period, Floquet gives {MU_PUBLISHED:.3f} "
          f"-> {err:.1%} apart   [{base['verdict']}]")
    print(f"               numerical floor {base['floor']:.1e} diam, eps used "
          f"{base['eps_used']:.0e}, fitted over {base['n_fit']} periods")
    ok = base['verdict'] == 'ATTRACTING' and err < 0.10
    if not ok:
        print("  The walk and the Floquet multiplier disagree at a point where BOTH are "
              "trustworthy. Fix that before believing anything this module says elsewhere.")
    print("  selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('THE FAILURE')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--tag', default=None, help='aggregated campaign tag')
    ap.add_argument('--kind', default='seeds', choices=('seeds', 'genes'))
    ap.add_argument('--analysis', default='fit_radial')
    ap.add_argument('--periods', type=int, default=24, help='periods to walk (default 24)')
    ap.add_argument('--ndir', type=int, default=3, help='perturbation directions')
    ap.add_argument('--eps', type=float, default=1e-3,
                    help='perturbation size, as a fraction of each species range')
    ap.add_argument('--backend', default='diffrax')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.tag:
        raise SystemExit('--tag is required (or use --selftest)')
    b = audit(a.model, a.tag, a.analysis, a.kind, n_blocks=a.periods, ndir=a.ndir,
              eps=a.eps, backend=a.backend)
    report(b)
    out = paths.out_path(a.model, a.analysis, f"stability_{b['mode']}.npz", a.tag)
    paths.savez(out, **b)
    print(f"\n[stability] -> {os.path.relpath(out, paths.HERE)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
