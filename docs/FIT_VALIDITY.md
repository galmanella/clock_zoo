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

A surface is **scoreable** only if all six hold. Each is model-agnostic, each is cheap relative
to the surface itself, and each returns a number rather than a boolean so it can also be
*reported*.

| check | quantity | rule | catches |
|---|---|---|---|
| **C1 orbit** | BVP residual, period band, positivity, amplitude | already in `fit.cost._surface` | hazard 2 |
| **C2 attractor** | per-period growth measured ABOVE the orbit's own numerical floor | `growth < 1`; UNRESOLVED if no perturbation clears the floor | F4 |
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
  as a discontinuity. 0.25 (a quarter of the cycle's extent between neighbours) flags 12 of 15;
  pick it, or another value, deliberately — but then refine `n_phase` until it passes at the
  search box corners, and do **not** resample the phase axis non-uniformly (see §4).
* **C5** — `DEAD_AMP = 0.01` was chosen to mean "not literally zero". It does not mean "carries
  phase information": at `|z| = 0.05` the post-perturbation oscillation is 5% of the intact
  clock and its Fourier phase is noise.
* **C6** — a window that does not bracket the transition cannot see the feature the whole
  experiment is about.

---

## 3. Cost changes

Split deliberately into changes that **do not** alter the objective (so the Aug-30 results stay
comparable) and changes that **do** (so they must be a separate, deliberate commit and a
re-run).

### 3a. No change to the objective — reporting and gating only

1. **Emit the contract vector with every evaluation and store it with every result.** `c_ptc`
   alone is not a reportable number; `(c_ptc, C1…C6)` is.
2. **`fit.aggregate` refuses to rank runs that fail the contract**, the way it already refuses
   to join runs with different bases.
3. **Fix `make_growth_fn`'s epsilon** (climb the ladder as `fit.stability.measure` does). This
   is a bug fix, not a design change: at `eps=1e-4` it misclassifies 6 of 16.

### 3b. Changes the objective — one commit, one re-run, one control

4. **Weight the pointwise phase residual by aliveness.** Replace the hard `alive` mask with a
   smooth weight `w = min(|z_model|, |z_target|)` clipped into `[0, 1]`, and charge the
   amplitude deficit as its own explicit term. Rationale: a cell where the clock is dead
   carries no *phase* information, but "you killed the clock" is real information and belongs
   in a term that has a gradient. The current hard mask conflates the two and, at
   `DEAD_AMP = 0.01`, admits 4 of 16 surfaces that are essentially noise.
5. **Weight by target informativeness.** A Poincaré target's old-phase span falls from 0.5
   below S\* to 0.05 at 6×S\* (hazard 17), and on a log window most cells are up there. Weight
   each dose row by the *target's own* old-phase span so the cost measures agreement where the
   target has something to say. This is the single change most likely to move the answer.
6. **A bracketing term, not a fine-alignment term.** §5.4c showed a singularity-location term
   saturates once the defect is on target and contributes no gradient there. Its useful job is
   different: keep S\* *inside the window*. Use `fit.target.soft_singularity` as a one-sided
   barrier on `log(S*/S_window_lo)` and `log(S_window_hi/S*)`, active only near the edges.
7. **A stability term**, once (3) is fixed: penalise `growth` above ~0.9, one-sided.
**None of 4–7 mentions the target, the gene, or the model.** They are properties of "a PTC
surface scored against another PTC surface".

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

## 5. Commissioning: what to run once per (model, target, mode)

Before any fit, and re-derived rather than inherited — the repo's most-repeated lesson is that
scale-bearing constants do not transfer (§5.1, §5.3).

1. `analysis.scrit` → S\_crit and the integrator step. *(exists)*
2. **Dose window**: `[S_crit / 10^a, S_crit * 10^b]` with `a, b` chosen so the window still
   brackets the transition at the extremes of the search box, plus an explicit dose-0 row.
   Measure `a, b` by rescanning the base point displaced to the box corners — do not guess.
3. **Dose resolution**: refine `n_dose` until `accumulated_twist` converges (§5.9b: BMAL1 needs
   ≥48; REV converges at 12 — it is per gene and varies 16×).
4. **Phase resolution**: refine `n_phase` until C4 passes at the box corners, not just at base.
5. **Record the commissioned grid as a fixture** (hazard 16), so every fit of that
   (model, target, mode) uses the same one and results stay comparable.

`fit/rescan.py` already does 2–4 for a finished campaign; the same code run on displaced base
points is the commissioning tool.

---

## 6. Order of work

Ordered by (information gained) / (risk of invalidating what exists).

| # | step | changes objective? | how you know it worked |
|---|---|---|---|
| 1 | `make_growth_fn` epsilon ladder | no | its verdict matches `fit.stability` on all 16 seeds |
| 2 | contract vector C1–C6 computed and stored per evaluation | no | re-scoring the 16 seeds reproduces the sets below exactly |
| 3 | `fit.aggregate` / `fit.figures` report the contract beside the cost | no | no run is ranked on a surface that fails it |
| 4 | commission the grid for Almeida/BMAL1/instant (§5) | no (new fixture) | base point brackets at the box corners |
| 5 | aliveness weighting + amplitude term (3b.4) | **yes** | `--selftest`: a surface with `\|z\| = 0.05` everywhere scores ≈ 1.0, not ≈ 0.1 |
| 6 | informativeness weighting (3b.5) | **yes** | the residual split at S\* stops being 0.18/0.05 and becomes comparable |
| 7 | bracketing barrier (3b.6) | **yes** | no fit ends with `n_sing = 0` in its own window |
| 8 | stability term (3b.7) | **yes** | T1 self-recovery still passes from the truth |
| 9 | re-run T1 per target, then the campaign | — | 3 of 16 pathological orbits should not recur |

Steps 5–8 change the objective, so they land as **one** commit with **one** re-run, and the
Aug-30 campaign becomes the labelled baseline rather than something to compare against
piecemeal.

### The regression fixture for step 2

Measured on `bmal1seeds`, these are the exact sets a correct implementation must reproduce.
They are the cheapest test in the whole plan and they pin every threshold at once:

    C1 orbit          {3}
    C2 attractor      {1, 4, 5, 11, 14}
    C3 dose-0 ident   {3, 4, 5, 11, 14}
    C4 phase res      {0, 1, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15}     12 of 15
    C5 aliveness      {2, 5, 11, 14}
    C6 bracketing     all but {2}                                   15 of 16

Two things are worth reading off this table before writing any code.

**C6 fails almost everywhere.** Only seed 2 kept a transition inside its own fit window. That is
§5.10c restated as a gate, and it means the bracketing barrier (3b.6) is not a corner case —
it is the common case.

**C3 predicts C2 at a fraction of the cost.** They differ by one element each way ({3} vs {1}):
a single dose-0 row anticipates a 24-period integration on 4 of 5 failures. Compute C3 first and
short-circuit.

---

## 7. What this does not fix

* **Multimodality (§5.10a).** Twelve solutions within 1.6× in cost and a median of 9.89 apart
  in the quotient is a property of the landscape, not of the measurement. A validity contract
  removes the *invalid* optima; it does not merge the valid ones.
* **From-distance locality (§5.4).** Unchanged.
* **The non-reproducible gauge basis (hazard 18).** Independent, and still open.
