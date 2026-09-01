# The surface validity contract

*How to stop a PTC fit from reporting a number that does not mean anything. Model-agnostic,
target-agnostic, task-agnostic by construction — none of the checks below know what
"radialization" is, and none contains a constant tuned to Almeida or to BMAL1.*

Companion to [`PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md) §5.10–5.12 (what was measured) and
[`REPO_MAP.md`](../REPO_MAP.md) hazards 17–20 (why each rule exists).

---

## 0. The distinction that organises everything: VALIDITY before OPTIMALITY

It is tempting to read the Aug-30 campaign's pathologies as the optimizer *cheating* — finding
cheap corners of the cost. Measured across the 16 seeds, that is **not** what happened:

    corr(cost, fraction of cells where the clock was killed)   -0.14
    corr(cost, phase under-resolution)                         -0.17
    corr(cost, cycle diameter)                                 +0.09
    corr(cost, period)                                         +0.04

n = 15, so none of these drives the cost ranking. The cheapest three fits have healthy
readout amplitude and one of them has the *best* phase resolution in the set.

So the failures are not (demonstrably) a reward-hacking problem. They are a **measurement
validity** problem: for a large minority of candidates the number `c_ptc` is computed on a
surface that is not a PTC at all, and the optimizer is neither helped nor hindered by it — it is
simply being handed noise and asked to minimise it.

That changes the fix. Adding penalty terms alone would be treating a symptom. What is needed
first is a **contract**: a candidate's surface is either valid — in which case its cost means
what it says — or it is not, in which case it is rejected and *says so*, rather than
contributing a plausible-looking number. Penalties come second, and only for the failures that
are genuinely dynamical rather than numerical.

---

## 1. The measured failure catalogue

Every row was measured on `out/almeida/fit_radial/bmal1seeds` (16 seeds) and every one
generalises: none depends on the model, the target gene, or the fitting task.

| # | failure | how it shows | measured incidence |
|---|---|---|---|
| **F1** | **The dose window does not BRACKET the transition** | winding is constant across the whole window; `n_sing = 0` | 11/12 usable. S\* at **0.01–4** vs a window floor of **12.5** — *and seed 0 the other way*, S\* = **363** vs a ceiling of 199.8 |
| **F2** | **The perturbation KILLS the clock, and the readout still returns a phase** | `\|z\|` (post-perturbation amplitude relative to the intact clock) collapses while cells stay "alive" | 4/16. Median `\|z\|` **0.053–0.093**, minimum **0.003**, against a `DEAD_AMP` gate of **0.01** |
| **F3** | **The old-phase axis does not resolve the cycle** | consecutive PTC columns start from states far apart | seeds 1, 4: **0.82–0.87 cycle diameters** between adjacent columns on the 20-phase grid the fit used; 47% of the cycle's arc length inside 10% of the phase axis |
| **F4** | **The solved orbit is not an attractor** | walk off it and it does not come back | 3/16 (2 decay to a fixed point, 1 diverges), 2/16 unmeasurable |
| **F5** | **The phase readout is not calibrated at that parameter set** | dose-0 row is not the identity | 5/16 at **4e-2 – 5e-1 cyc**, against 1e-4 – 9e-4 for the rest |
| **F6** | **Absolute dose vs candidate-dependent state scale** | the same dose is a wildly different relative kick | cycle diameters span **16 → 4930 (308×)** across the population, all scored on one fixed absolute window |
| **F7** | **The quality gate is post hoc** | `scramble`, winding set never enter the cost | winding sets `[-4,-1,0,1,2]` and `[-1,0,1]` reached; negative winding is topologically impossible for a real PTC |
| **F8** | **The time unit is unfixed, so a reported period belongs to the REPRESENTATIVE, not the model** | periods scatter with nothing pinning them | **8.4 – 38.9 h** from a base of 24.8 — but see §3c: in `instant` mode this is not a defect to penalise |

**F1 and F6 are the same mechanism seen twice.** Because dose is absolute and the state scale
is not fixed, moving the clock's scale moves its S\_crit relative to a fixed window. F6 is why
F1 is always available.

**F2 explains three of the six surface oddities directly** (the "really flat type-0" of seed 2,
the noise near dose 10 of seed 5, the circular contours of seed 11): all three are the phase of
an oscillator the perturbation has extinguished. **F3 explains the sharp old-phase
discontinuity** in seeds 1 and 4 — those columns are not neighbours in state space.

---

## 2. The contract

A surface is **scoreable** only if all six hold. Each is model-agnostic, and each returns a
number rather than a boolean so it can also be *reported*.

**Five of the six are free.** C1, C4, C5 and C6 are functions of things `fit.cost._surface`
already computes — the orbit, the cycle, `amp`, the surface — and C3 costs one extra dose row.
Only **C2 is expensive** (~7 s: a 24-period walk over an epsilon ladder), and §6 shows C3
predicts it well enough that it does not belong in the inner loop at all.

| check | quantity | rule | catches |
|---|---|---|---|
| **C1 orbit** | BVP residual, period band, positivity, amplitude | already in `fit.cost._surface` | hazard 2 |
| **C2 attractor** | per-period growth measured ABOVE the orbit's own numerical floor | `growth < 1`; UNRESOLVED if no perturbation clears the floor | F4 — **post hoc only**, C3 is its in-loop proxy (§6) |
| **C3 calibration** | dose-0 identity residual | `< 1e-2 cyc` | F5 |
| **C4 phase resolution** | max arc-length between adjacent phase samples / cycle diameter | `< 0.25` | F3 |
| **C5 aliveness** | readout amplitude `\|z\|` per cell | see §3 — a weight, not a gate | F2 |
| **C6 bracketing** | winding change inside the dose window | at least one, and not at an edge row | F1 |

### Why each threshold is a measurement, not a preference

* **C2** — the floor spans 8e-9 to 7.6e-2 cycle diameters *within one campaign*, so it must be
  measured per orbit (REPO_MAP hazard 19). `fit/stability.measure` already does this.
* **C3** — the two populations separate by two orders (1e-4…9e-4 vs 4e-2…5e-1), so 1e-2 sits in
  a gap, not on a slope.
* **C4** — this one is a DESIGN CHOICE, not a measured gap, and it should be labelled as such.
  On the 20-phase grid the campaign actually used, adjacent-column separation is a continuum
  with no break:

        6:0.18  2:0.21  10:0.23  8:0.28  13:0.29  7:0.32  12:0.33  0:0.35
        15:0.59  9:0.70  11:0.72  14:0.77  4:0.82  5:0.85  1:0.87

  So the honest reading is not "two seeds are under-resolved" but **the phase grid was too
  coarse for almost the whole population** — seeds 1 and 4 are merely where it becomes visible
  as a discontinuity. 0.25 (a quarter of the cycle's extent between neighbours) flags 13 of 16;
  pick it, or another value, deliberately — but then refine `n_phase` until it passes at the
  search box corners, and do **not** resample the phase axis non-uniformly (see §4).
* **C5** — `DEAD_AMP = 0.01` was chosen to mean "not literally zero". It does not mean "carries
  phase information": at `|z| = 0.05` the post-perturbation oscillation is 5% of the intact
  clock and its Fourier phase is noise.
* **C6** — a window that does not bracket the transition cannot see the feature the whole
  experiment is about.

---

## 3. Cost changes, after pruning

The first draft listed eight. Four survive. What was cut and why is §3d — the cuts matter more
than the additions, because three of them were two implementations of one idea.

### 3a. No change to the objective — reporting and gating only

1. **Emit the contract vector with every evaluation and store it with every result.** `c_ptc`
   alone is not a reportable number; `(c_ptc, C1…C6)` is. **Everything C1–C6 needs is already
   computed by `fit.cost._surface`** — the orbit, the cycle, `amp`, the surface — so this is
   bookkeeping, not new computation. Pinned by the regression fixture in §6.
2. **`fit.aggregate` refuses to rank runs that fail the contract**, the way it already refuses
   to join runs with different gauge bases.

### 3b. Changes the objective — one commit, one re-run, one control

3. **Raise the aliveness threshold and make it a ramp.** `DEAD_AMP = 0.01` means "not literally
   zero", not "carries phase information": at `|z| = 0.05` the post-perturbation oscillation is
   5% of the intact clock and its Fourier phase is noise. Weight each cell's phase residual by a
   smooth ramp in `|z|` instead of the current hard mask.

   This is **one constant and a ramp**, not a restructuring — the hard mask already scores dead
   cells at the maximum, so the behaviour is unchanged in kind. It removes the 4 of 16 surfaces
   that are currently scored as PTCs and are not.

4. **Weight each dose row by the TARGET's own old-phase span.** A Poincaré target goes
   phase-blind above its S\*: span 0.5 below, 0.05 at 6×S\* (hazard 17). On a log window most
   rows are up there, and measured on the campaign the residual below S\* went 0.199 → 0.181–0.221
   (no better) while above it went 0.276 → 0.035–0.057 (all of the gain).

   **This is the change most likely to move the optimum**, and it also subsumes the bracketing
   barrier — see §3d.

### 3c. NOT a cost term: the period

An earlier draft of this document proposed "add a period term when the period is data". **That
is wrong, and `fit/radial.py` already says so** — *"Period is pure GAUGE in parameter space —
freely rescalable"*. Time rescaling is one of the gauge generators, so every physical model here
sits on a gauge orbit containing a one-parameter family of periods: `T` is a property of the
REPRESENTATIVE the quotient happens to pick, not of the model. A period term would constrain the
representative and identify nothing.

MEASURED, on Almeida under a pure time-rescale gauge motion of `rho = 1.4191`:

    mode         period                  max |PTC difference|
    instant      24.83 -> 17.49 h         1.1e-07 cyc      INVARIANT
    pulse (8 h)  24.83 -> 17.49 h         3.8e-01 cyc      NOT invariant

The period scales exactly as `1/rho` in both, and **an instant-mode PTC cannot see it at all**.
For the mode this campaign used, period is unidentifiable by construction and the scatter in F8
is not a pathology to penalise.

**What period data does is BREAK the gauge, not add a constraint inside it.** Set
`include_time=False` and the quotient grows by exactly one direction —

    almeida 16 -> 17     korencic 33 -> 34     goldbeter 47 -> 48     goodwin 7 -> 8

— and the measured period pins that new direction. `gauge.random_gauge` already carries the
right note (*"the time rescale changes the period, so it is not a symmetry of a
period-constrained cost"*); what is missing is that no driver ever passes `include_time=False`.

**A corollary that is not about periods at all.** A pulse whose duration is fixed in HOURS is
itself a clock, so pulse-mode data breaks the time gauge whether or not the period is
recorded — that is the 0.38 cyc above. Every pulse-mode analysis in this project quotients with
`include_time=True`, i.e. projects out a direction its own data can see. The instant-mode
results (§5.4b, §5.10) are unaffected; the pulse-mode ones (§3.x) sit on a quotient one
dimension smaller than the data can determine, and their identifiability counts should be
re-derived with `include_time=False`. In REPO_MAP Open.

---

### 3d. What was cut, and why

| cut | why |
|---|---|
| **a period term** | period is GAUGE — §3c. Constrains the representative, identifies nothing. |
| **a bracketing / singularity-location barrier** | **subsumed by 3b.4.** Weighting by the target's own span already punishes a model whose S\* has left the window: it is flat exactly where the target has structure, so it mismatches exactly where the weight is. The barrier would add a `detect_grid` call — quantised, non-differentiable — to do the same job worse, and §5.4c already measured that a singularity-location term saturates and stops contributing gradient. Keep C6 as a REPORTED check, not a term. |
| **a stability cost term** | **subsumed by C3 at ~1/1000 the cost.** C3 (one dose row) and C2 (a 24-period integration) flag the same runs but for one element each way — see §6. Screen with C3 in the loop; confirm with `fit/stability.py` post hoc, where it already exists. |
| **fixing `make_growth_fn`'s epsilon** | with no stability term there is no caller. `fit.stability.measure` supersedes it and is cross-checked against the published Floquet multiplier. Two implementations of one idea, one known broken, is how REPO_MAP says a directory becomes a museum — **delete `make_growth_fn` and `w_stab`** rather than repair them. |

## 4. Two constraints that rule out the obvious fixes

**You cannot adapt the evaluation grid per candidate.** The (phase, dose) grid is *the
experiment*. If each candidate is scored on its own grid, two candidates' costs are no longer
comparable and the optimizer is minimising a moving target — the same objection §5.4c raised
against re-profiling `(k, psi)` every evaluation, and the reason the target is pinned. So the
grid must be chosen ONCE, from the target/data, wide and fine enough for the whole search
region — and a candidate whose transition leaves it must be *penalised* (item 6), not
re-gridded.

**You cannot resample the old-phase axis non-uniformly either.** Old phase is the experimental
variable — *when* the pulse was applied. Uniform-in-phase is what an experiment does and what
the target is defined on. So F3 is a **resolution requirement** (refine `n_phase` until C4
passes), never an arc-length reparametrisation.

The practical consequence of both: **the grid must be commissioned before the fit**, and it
must be wide enough to bracket every candidate the search can reach — which is what §5 is for.

---

## 5. Commissioning: one tool, run once per (model, target, mode)

The first draft listed five steps. Four of them are `fit/rescan.py` with different arguments, so
this is one command run at the corners of the search box rather than a procedure:

    $PY -m fit.rescan --model M --tag <commissioning> --n-phase P --n-dose D --decades A

Rescan the base point AND the box corners, and read off: does the window bracket the transition
(C6)? does the twist converge (§5.9b — per gene, and it varies 16×: BMAL1 needs ≥48 dose
samples, REV converges at 12)? does C4 pass? Then **record the resulting grid as a fixture**
(hazard 16) so every fit of that (model, target, mode) uses the same one and costs stay
comparable.

Re-derive it per model and per target. That is the repo's most-repeated lesson (§5.1, §5.3), and
§5.9d is the direct evidence: there is no globally safe resolution.

---

## 6. Priorities

### The constraint that sets them: a contract-compliant grid costs ~10× per evaluation

Measured, same parameter set, `total(v)` wall time:

    20 x 14   [0.5, 8] S*      2.5 s      today
    32 x 24   [0.5, 8] S*     11.7 s      x4.7
    32 x 48   [0.5, 8] S*     24.7 s      x9.8
    40 x 64   [0.01, 8] S*    35.3 s      x14

Two things fall out, and they drive the whole ordering:

* **Dose ROWS are the expensive axis, and roughly linear** — 24 → 48 rows doubled the time.
* **Extending the window DOWN is free.** 32×48 over `[0.5, 8]` took 24.7 s; the same grid over
  `[0.01, 8]` took **20.9 s**. Low doses integrate quickly, so the fix for F1/C6 in the
  direction that actually failed costs nothing.

*(The phase-axis scaling came out non-monotonic over three repetitions and is NOT established —
measure it before budgeting on it.)*

So the campaign as run (8000 evals × 16 seeds ≈ 16 CPU-hours per seed) does **not** survive a
naive grid refinement: it becomes ~160. Either the evaluation budget drops ~10×, or the search
grid and the scoring grid are decoupled. `fit/multires.py` already implements the second, but
it **lost its benchmark** — 0.0968 against cma-anneal's 0.0198 at equal evaluations — with a
finest stage of 24×16, coarser than the contract wants. That is an open question, not a
solution.

### The order

| P | do | changes objective? | cost | why here |
|---|---|---|---|---|
| **P0** | contract vector C1–C6 computed, stored, reported (3a.1–2) | no | ~0 — all inputs already computed | Free, and it stops wrong numbers being quoted while everything else is decided. Pinned by the fixture below. |
| **P1** | aliveness threshold + ramp (3b.3) | **yes** | one constant | Removes the 4 of 16 surfaces that are noise. Highest value per line changed. |
| **P2** | extend the dose window down (C6) | **yes** (grid) | **free — measured** | The failure that hit 15 of 16, fixed at no compute cost. |
| **P3** | informativeness weighting (3b.4) | **yes** | ~0 per eval | The one most likely to move the optimum, and it subsumes the bracketing barrier. |
| **P4** | dose resolution per gene (§5) | **yes** (grid) | **~linear, the real bill** | Needs the budget decision above. Do it last because it is the only expensive item. |
| **P5** | stability + rescan per campaign | no | already built | Post hoc, unchanged: `fit.stability`, `fit.rescan`. |

**P1–P4 all change the objective, so they land as ONE commit and ONE re-run**, with the Aug-30
campaign as the labelled baseline. P0 and P5 are independent and can land immediately.

**A separate thread, not on this list:** `include_time=False` for the pulse-mode identifiability
counts (§3c). It touches `analysis/`, not the fit, and blocks nothing here.

### The regression fixture for P0

Measured on `bmal1seeds`; a correct implementation must reproduce these exactly. It is the
cheapest test in the plan and it pins every threshold at once:

    C1 orbit          {3}
    C2 attractor      {1, 4, 5, 11, 14}                             post hoc
    C3 dose-0 ident   {3, 4, 5, 11, 14}
    C4 phase res      {0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15}  13 of 16
    C5 aliveness      {2, 5, 11, 14}
    C6 bracketing     ALL                                           16 of 16

**Two of these moved when the checks were implemented, and both times the loose definition
used to derive the original number was the wrong one.** Recorded rather than quietly updated,
because silently moving a fixture to match new code defeats the fixture.

* **C4: 12 → 13.** The 12 counted only the 15 candidates with a finite cycle. Seed 3 has none,
  so it fails C4 as well as C1.
* **C6: 15 → 16.** The 15 came from `detect_grid`'s `n_sing`, which does find seed 2's
  singularity. But seed 2's winding changes between dose ROW 0 and ROW 1 — a single row of
  type-1 at the very bottom of the window — and the extended rescan shows its real structure
  runs from 0.02 up to ~8, i.e. mostly *below* the window. A transition sitting on the boundary
  row is not a bracketed window, so C6 requires an INTERIOR change. **Every one of the 16 fails
  it**, which strengthens rather than weakens the point below.

Two things to read off it before writing code.

**C6 fails everywhere — all 16.** Not one candidate kept a transition properly inside its own
fit window. That is why P2 is high and why the bracketing *barrier* was cut: the problem is the
window, not a missing penalty. A penalty term cannot help when no candidate satisfies it.

**C3 predicts C2 at ~1/1000 the cost.** They differ by one element each way: one dose-0 row
anticipates a 24-period integration on 4 of 5 failures. That is the redundancy that keeps the
stability measurement out of the inner loop entirely.

---

## 7. What this does not fix

* **Multimodality (§5.10a).** Twelve solutions within 1.6× in cost and a median of 9.89 apart
  in the quotient is a property of the landscape, not of the measurement. A validity contract
  removes the *invalid* optima; it does not merge the valid ones.
* **From-distance locality (§5.4).** Unchanged.
* **The non-reproducible gauge basis (hazard 18).** Independent, and still open.
