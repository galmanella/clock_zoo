# `clock_zoo` — LC vs PTC identifiability across simpler clock models

*Semi-README scientific reference. Update it with every result. Figures live in
`docs/figures/`, copied out of the git-ignored `out/`, so they travel with this document.*

---

## 0. TL;DR

**Question (unchanged from `../input_screen/`).** Does phase-transition-curve (PTC) data make a
mechanistic circadian model more identifiable than limit-cycle (LC) data alone?

**Why a new repo.** Three weeks on **Mirsky 2009** (21 states, 132 params) established the
machinery and several robust phenomenological results, but the optimisation never worked:
gradient descent recovers 132/132 parameters *from the truth* and stalls at ~26/132 *from a
distance*. That "from-distance locality" blocked the actual scientific question. So: change
the model, not the optimizer, and re-ask it on models small enough to optimise.

**Status.** The stack is built and gated. Objectives (a) and (b) are answered for Almeida,
across three perturbation targets (BMAL1, CRY, PER).

**The headline scientific result so far:** for Almeida at base, there IS a parameter
combination that the PTC constrains and the limit cycle leaves ~11x freer -- confirmed by
finite displacement on an independent adaptive solver, with controls (§3.6). No single
parameter shows this (best ratio 1.04); it only appears in a COMBINATION, which is why a
per-parameter sensitivity scatter misses it entirely. This is the opposite of the Mirsky
result, where no direction decoupled. The clean probes (BMAL1, DBP) find the SAME direction
(`ker`/`gamma_CP`/`ve`/`ke`), so it looks like a property of the model rather than of the probe
-- but see 3.7: the CRY and PER surfaces turned out to be phase-scrambled and their numbers are
retracted, so this rests on two probes, not four.

The most consequential methodological findings, each of which changed a scientific answer:

- **Almeida's PTC type-transition dose is a property of the integrator step until you refine
  it.** At `dt=0.02`, 3 of 8 targets appeared to reset; refined, **7 of 8 do**.
- **A converged periodic-orbit solve is not necessarily a limit cycle.** Equilibria and
  negative-orthant runaways both satisfy the BVP, and both produced spectacular-looking
  "sensitivities" (5e98, 1e11) before being caught.
- **Almeida is far less gauge-degenerate than Mirsky** — 2 flat directions of 18, versus 13 of
  132 — so its identifiability question is essentially unconfounded by units.
- **A jacobian ratio overstates decoupling.** The linear analysis predicted ~140x for the
  direction above; a finite displacement gives 11x. The ordering survives, the magnitude does
  not — so the finite-displacement number is the one to quote.

---

## 1. The models

All three were taken from `../circadian_modeling.ipynb`, ported unchanged (verified against
the notebook's own `derivatives`), and all oscillate self-sustainedly at their published
parameters, so **no re-fit was needed** before characterization.

| model | states | params | period | structure | Floquet mu | gauge |
|---|---|---|---|---|---|---|
| **Goodwin/Gonze** | 3 | 11 | 23.540 h | single negative loop, Michaelian degradation | 0.951 | 4 |
| **Almeida 2020** | 8 | 18 | 24.826 h | transcription-factor level; EBOX/RRE/DBOX; one reversible complex | 0.546 | 2 |
| **Korencic/Grabe** | 15 | 34 | 25.809 h | 5 genes x a 3-stage delay chain; no complexes | 0.505 | 1 |
| **Leloup–Goldbeter 2003** | 16 | 52 | 23.849 h | phosphorylation + nuclear transport + MM degradation | 0.334 | 5 |
| *(Mirsky 2009, for scale)* | *21* | *132* | *23.7 h* | *mass-action dimers* | — | *13* |

Goodwin is a **smoke test, not a science target**: it is the smallest thing that oscillates and
its gauge is known-good from `input_screen`, so it validates every tool before that tool
touches a model whose answer we do not already know.

**Four Goldbeter parameters were dropped.** `V1PCP`, `V1BP`, `V2PCP`, `V2BP` appear in the
notebook's parameter dict but in none of its 16 equations — verified numerically: perturbing
each leaves the RHS bit-identical. Keeping them would contribute four guaranteed-flat
directions to every identifiability count, inflating precisely the quantity being measured.

### 1.1 The perturbation

Two generic modes, defined for any state of any model: **pulse** (an additive production rate
`+dose` for 8 h, then off) and **instant** (a one-shot displacement, clamped at 0). No ectopic
protein, no degron, no complex bookkeeping. This is the notebook's own
`AdditiveRatePerturbation`, so PRCs stay comparable with what is already there, and it is the
only definition that maps onto Korencic, which has no protein species at all.

---

## 2. The gauge: how much of each model is even meaningful

Rescaling a species — measuring it in different units — is an exact symmetry absorbed into the
parameters, so any observable that does not fix the concentration units is blind to it. The
generators are **derived** from each model's declared dimensions and then **verified**, not
asserted: the defining identity `f(S y; theta_gauged) = rho . S . f(y; theta)` is checked
pointwise, with no integration to hide behind.

| model | generators | of params | physically meaningful | identity |
|---|---|---|---|---|
| almeida | 2 | 18 | **16** | 1.3e-13 |
| korencic | 1 | 34 | **33** | 6.3e-15 |
| goldbeter | 5 | 52 | **47** | 1.6e-15 |
| goodwin | 4 | 11 | 7 | 3.6e-16 |

**Almeida's 2 generators are forced by its shared promoter terms.** `EBOX` is added to five
different species' equations, `RRE` to four, and the PER/CRY binding and BMAL1 titration terms
chain the rest — so *all eight species share one concentration unit*, leaving one scale factor
plus time. There is no per-species freedom at all.

**Korencic needed a new piece of contract.** Its transcription is a bare product of Hill
factors with **no maximal-rate parameter**, so nothing can absorb a rescaling of a transcript.
That is not expressible as "these species share a unit", so `models/api.py` gained
`scale_constraints()`: linear relations the log scales must satisfy. Korencic declares three
per gene, which pin every species scale to the single time rescale (its derived factors come
out as exactly `rho^-1, rho^-2, rho^-3` along each chain). **One generator for 34 parameters.**

The practical consequence: unlike Mirsky, these models' identifiability questions are almost
entirely unconfounded by units — but the gauge must still be projected out before any
parameter-space comparison, because a gauge direction is exactly LC-null while moving a
fixed-absolute-dose PTC, which is precisely the false signal `analysis/coupling.py` hunts for.

---

## 3. Results

### 3.1 Objective (a): which components reset the clock, and where

`analysis/scrit.py`, Almeida, pulse mode. **S_crit is the dose of the type-1 → type-0 winding
transition.** This is the canonical table — do not recompute it.

| target | S_crit | dt-converged | re-entrant | valid to | dt needed |
|---|---|---|---|---|---|
| DBP | **0.196** | yes | yes | 1e4 | 0.005 |
| BMAL1 | **3.16** | yes | no | 22.1 (unstable) | 0.005 |
| PER_CRY | **5.51** | yes | no | 117 (unstable) | 0.00125 |
| PER | **16.8** | yes | no | 1e4 | 0.00125 |
| CRY | **29.2** | yes | yes | 1e4 | 0.00125 |
| REV | **29.2** | yes | yes | 1e4 | 0.00125 |
| E4BP4 | **50.9** | yes | no | 1e4 | 0.00125 |
| ROR | — | — | — | 1e4 | 0.005 |

**Seven of eight Almeida targets reset.** ROR is the sole non-resetter, and credibly so: it
was scanned to the top of the range (1e4) without the integrator going unstable, so this is a
statement about the model rather than about where the numerics gave up.

> **This table required refining `dt`, and that changed the answer.** The dose at which
> fixed-step RK4 goes unstable is a property of the STEP, not the model, and it moves ~8x per
> 4x smaller step. At `dt=0.02` the ceiling for CRY is 6.2 and CRY appears not to reset at
> all; at `dt=0.005` the ceiling is 48.6 and CRY resets at 32.2; `dt=0.00125` confirms 32.2.
> A single default step would have produced the clean, plausible, **wrong** claim that 5 of
> Almeida's 8 targets are non-resetting.

**Contrast with Mirsky.** There the rule was sharp and structural: only the repressor arm
resets; BMAL1/CLK/RORa are dose-flat with no singularity even at 500x peak. Almeida shows no
such arm asymmetry — BMAL1 is among its *strongest* resetters (S_crit 3.16). Whether that is a
real difference in clock architecture or a consequence of Almeida's single shared scale group
is an open question, and Korencic/Goldbeter are the next data points.

### 3.2 The base PTC surfaces

`analysis/characterize.py` on the adaptive grids.

![Almeida PTC surfaces](docs/figures/almeida_surfaces_pulse.png)

| target | S* | phi* | total twist | singularities | min rel. amplitude |
|---|---|---|---|---|---|
| DBP | 0.452 | 0.771 | **0.424 cyc** | 1 | 1.000 |
| BMAL1 | 4.503 | 0.521 | **0.489 cyc** | 1 | 1.000 |
| PER | 15.65 | 0.854 | **0.159 cyc** | 1 | 1.000 |

Each surface shows exactly **one** dipole-filtered singularity — a clean type-1 → type-0
transition, textbook pinwheel structure, the S_crit line passing through the defect. Two
things worth noting:

- **The clock is never even weakened** (minimum relative amplitude 1.000 across every grid
  point). Almeida's oscillation is robust to these perturbations in a way Goodwin's is not —
  an 8 h pulse of dose 0.5 into Goodwin's X drives Z high enough that its n=4 Hill shuts
  transcription down ~1e-5 and the oscillator stops **permanently**.
- **Isochron twist is substantial** — 0.42–0.49 cyc for DBP and BMAL1, i.e. the entrainment
  phase shears by nearly half a cycle across the dose range. Twist is the smooth,
  gauge-invariant feature and is what any isochron claim should be read from; `S*`/`phi*` are
  topological defects and are grid-quantised.

*(S* here is 4.503 for BMAL1 against `scrit`'s 3.16 — the two use different dose grids, so
this is grid resolution, not a discrepancy.)*

### 3.3 Objective (b), part 1: LC sensitivity

`analysis/lc_sens.py`, Almeida, 18 parameters x 9 factors (x0.25 – x4).

Top movers: `vd` 0.659, `vr` 0.605, `gamma_DB` 0.604, `gamma_REV` 0.589, `krr` 0.462.
Least: `ker` 0.196, `gamma_BP` 0.219, `gamma_CP` 0.269.

**The striking feature is the RANGE: 0.20 to 0.66, a factor of 3.3.** For Mirsky the
equivalent span was ~50x, with a long sloppy tail. Almeida has **no sloppy parameters in the
single-parameter sense** — every one of its 18 parameters moves the limit cycle by a
comparable amount. That is a genuinely favourable property for identifiability and a direct
consequence of choosing a small model.

*Caveat, stated plainly:* 51 of 162 settings were rejected (6 dead, 11 negative-orthant, 34
non-converged). The rejections concentrate at large displacements, so the reported spans are
mild **under**-estimates. The 34 non-converged are a known limitation of the orbit guess, not
a claim that no orbit exists there.

### 3.4 Objective (b), parts 2–3: PTC sensitivity and the coupling

`analysis/ptc_sens.py` on Almeida / BMAL1 (pulse), 18 parameters x 9 factors on the fixed base
dose grid 0.912–22.1. Ranked by twist response: `vd` 0.375, `gamma_c` 0.340, `vr` 0.321,
`ke` 0.295, `gamma_P` 0.288. As with the LC, the range is narrow — 0.24 to 0.37.

`analysis/coupling.py`, the gauge quotiented out (2 of 15 usable parameters), J_LC = 513
observables x 15 parameters, J_PTC = 576 x 15.

![LC vs PTC coupling](docs/figures/almeida_coupling_BMAL1.png)

**Preliminary, and it points the opposite way to Mirsky — but read the caveats.**

| | Mirsky (input_screen) | Almeida (here) |
|---|---|---|
| per-parameter log-log correlation | tight | **+0.660** (moderate) |
| the low-LC / high-PTC quadrant | **empty** | 4 of 18 parameters |
| angle between the single best-determined LC and PTC combination | — | **79.1 deg** |
| LC-sloppiest directions' share of PTC response | 3.4% (pointwise) / 0.5% (twist) | **7.7%** |

The clearest signal is the **79.1 deg** between the leading LC direction and the leading PTC
direction: the combination the limit cycle pins hardest and the one the PTC pins hardest are
nearly orthogonal. The next few are more shared (k=3: 14, 39, 84 deg; k=5: 9, 22, 45, 77, 81),
so the two experiments overlap substantially but each retains directions the other is close to
blind to. Consistently, the PTC response is **not monotone in LC sensitivity** — the 2nd LC
direction carries the largest PTC response (1.000) while the 1st carries 0.375.

**What this does not yet establish.** Three things, stated plainly:

1. **The quadrant count is weak evidence.** Every parameter sits within a ~3x band on both
   axes (see the figure), so the quadrant split is a median cut through a tight cluster, not
   an order-of-magnitude separation like Mirsky's. "4 in the prize quadrant" should not be
   read as "4 decoupled parameters".
2. **The candidate direction has now been confirmed — see §3.6.**
3. **3 of 18 parameters were dropped**, lacking a converged orbit at both difference factors.

**One thing that IS solid**: after the gauge projection, the two exactly-null directions of
J_LC are also exactly null in J_PTC (3.5e-16). That is the correct, self-consistent behaviour
and confirms the projection is doing what it should rather than leaking a false signal.

#### What the sensitivity numbers actually look like

![LC and PTC tornado](docs/figures/almeida_tornado_BMAL1_pulse.png)

LC sensitivity decays smoothly from 0.66 (`vd`) to 0.20 (`ker`); the PTC response is FLATTER,
0.37 to 0.14. The third panel is the per-parameter ratio, and its ceiling is the point: the
best single parameter reaches only **1.04**, so no individual parameter is meaningfully
decoupled.

![LC examples](docs/figures/almeida_lc_examples.png)
![PTC examples](docs/figures/almeida_ptc_examples_BMAL1_pulse.png)

A `lc_sens` of 0.66 (`vd`) is a visible ~40% change in BMAL1's peak and ~50% in DBP's; 0.20
(`ker`) barely moves the cycle at all. On the PTC side the **Δ new-phase** panel is what makes
a change legible — raw before/after surfaces look nearly identical, while the difference shows
`vd` displacing the singularity as a red/blue dipole plus a broad high-dose band, and the
weakest parameter as a faint local smudge.

### 3.5 The decoupling is COMBINATORIAL, and only combinations show it

A per-parameter scatter can only find decoupling that happens to be **axis-aligned**. A
decoupled direction is generically a combination, so the right question is: which unit
direction `v` in log-parameter space maximises

    rho(v) = ||J_PTC v||^2 / ||J_LC v||^2

which is the generalized eigenproblem `(J_PTC^T J_PTC) v = rho (J_LC^T J_LC) v`.

![Combinatorial decoupling](docs/figures/almeida_directions_BMAL1_pulse.png)

| | rho |
|---|---|
| best single parameter (`ve`) | **1.64** |
| best combination | **122.3** |
| gain | **75x** |

The top direction is `0.65 ker − 0.53 gamma_CP + 0.49 ve + 0.21 ke + 0.11 gamma_DB`. It moves
the limit cycle by **1.6e-02** and the PTC by **1.8e-01** of their respective jacobian norms.

**This is not a ridge artifact, and it was checked rather than assumed.** rho blows up wherever
the denominator is small, so the top eigenvector will happily be a direction the LC jacobian
cannot RESOLVE rather than one the limit cycle genuinely does not move — a different claim, and
only the second is a result. Two checks:

- The LC jacobian's finite-difference noise floor is ~1e-5 relative (the orbit solves to
  |F| ~ 1e-13 but the cycle is accurate to ~5e-6). The top direction's LC response is
  **1.6e-02**, three orders above it.
- Sweeping the LC singular-value floor from 1e-10 to 1e-2, rho_max stays at **122.3** with the
  same composition, and only falls to 37 at 3e-2 when a 13th direction is cut. A ridge artifact
  would collapse as the floor rose.

So Almeida does have a genuinely decoupled direction — **and no single parameter comes close to
it**. This is the concrete reason the per-parameter scatter in §3.4 under-reports, and it
sharpens the contrast with Mirsky, where the equivalent search was done over `J_LC`'s singular
directions rather than as an extremal problem.

### 3.6 CONFIRMED by finite displacement — and the magnitude is NOT

`analysis/confirm.py`: step a finite `eps = 0.15` in log-parameter space along each direction,
recompute the limit cycle on the orbit solver and the PTC on **`engine/reference.py`, the
independent adaptive engine** that shares no numerical machinery with the JAX engine the
jacobian came from. Both signs, three directions, 12 phases x 3 doses.

| direction | rho | dLC | dPTC (rms) | dPTC (max) | **actual PTC per LC** |
|---|---|---|---|---|---|
| **decoupled** (candidate) | 122.3 | **0.0017** | **0.0303** | 0.052 | **18.1** |
| **coupled** (control) | 0.01 | 0.0725 | 0.1182 | 0.42 | **1.64** |
| **stiffest LC** (positive control) | — | 0.1077 | 0.1325 | 0.46 | 1.23 |

**The ordering is real.** The candidate direction moves the limit cycle by 0.17% while moving
the PTC by 3.0% rms — an actual PTC-per-LC ratio **11x** that of the coupled control. The
positive control moves both, so the measurement is working rather than reporting noise; and the
two signs agree closely (0.0300 / 0.0305), so this is not a one-sided artifact of the step.

![The same nudge in LC and in PTC](docs/figures/almeida_direction_examples_BMAL1_pulse.png)

**Read that figure carefully, because it does not say what a quick glance suggests.** In row 1
the dashed +/-eps cycles are invisible under the black base curve, and the PTC Delta is a small
dipole localised at the singularity. In rows 2-3 the cycles visibly separate and the PTC Delta
is broad and strong. So the decoupled direction does **not** move the PTC more in absolute
terms -- it moves it about 3.5x LESS (0.028 vs 0.098). What makes it decoupled is that it moves
the limit cycle 38-57x less than that. The claim is a RATIO, and the figure shows the ratio,
not a large PTC excursion.

The localisation is itself the expected signature: a red/blue dipole at the singularity is the
phase singularity being displaced, which is exactly what an isochron change looks like.

**The magnitude is overstated by the linear analysis, and that matters.** rho is a ratio of
squared responses, so the comparable linear prediction for the amplitude ratio is
`sqrt(122.3/0.01) ~ 140x` against an actual **11x** — the jacobian overstates by roughly 13x.
The direction is genuinely decoupled; it is **not** decoupled by two orders of magnitude. Quote
the finite-displacement number.

This is precisely why the confirmation step is mandatory rather than a formality. It is also a
much milder version of the `input_screen` failure, where the linear step was wrong by ~80
orders of magnitude — the difference being that here the jacobian is finite-difference on an
adaptive-verified engine rather than autodiff through fixed-step RK4.

**So, for Almeida at base: PTC data does constrain a parameter combination that the limit cycle
leaves ~11x freer.** That is a positive answer to the project's question for this model, at this
operating point, for this one probe — and the opposite of what Mirsky gave.

### 3.7 Three targets -- and a RETRACTION: two of the three surfaces were unusable

The objective-(b) chain -- `ptc_sens`, `coupling`, `sweep` -- was repeated for **CRY** and
**PER** alongside BMAL1. The table below was published here in an earlier revision. **The CRY
and PER rows are now retracted.** They are kept, struck, because the retraction is the point.

| target | S_crit | phi* | twist (cyc) | best single param | best combination | gain | status |
|---|---|---|---|---|---|---|---|
| **BMAL1** | 4.50 | 0.521 | **0.489** | `ve` 1.64 | **122.3** | 75x | **stands** |
| ~~CRY~~ | ~~138.1~~ | ~~0.938~~ | ~~0.285~~ | ~~`ker` 11.27~~ | ~~1923.5~~ | ~~171x~~ | **RETRACTED** |
| ~~PER~~ | ~~15.65~~ | ~~0.854~~ | ~~0.159~~ | ~~`ve` 2.86~~ | ~~108.4~~ | ~~38x~~ | **RETRACTED** |

**What went wrong.** The user noticed from a figure that the CRY surface did not look like a
PTC, and said so ("CRY is not radial, this is an artifact -- its surface a complete mess"). It
checks out. The CRY pulse surface is phase-scrambled:

- its winding sequence over dose is `+1 x8, 0, 0, +1, +2, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, +1, 0`
  -- a PTC has winding +1 or 0 and nothing else, so `-1` and `+2` are not resetting behaviour,
  they are evidence the phase readout is noise;
- its phase changes by up to **0.499 cyc between ADJACENT phase samples**, against a
  theoretical maximum of 0.5 -- i.e. neighbouring samples are uncorrelated;
- it carries **13 raw winding plaquettes** which the dipole filter reduces to a single
  survivor, and the reported `S_crit = 138.1` is just which one happened to survive;
- its twist reads 0.285 at `n_phase = 24` and 0.046 at `n_phase = 32` -- a 6x swing from phase
  resolution alone, so the quantity is not well defined.

PER is milder but also fails: its winding set includes `-1` at the transition.

**So `rho = 1924` for CRY was never a measurement of anything**, and the two claims built on it
are withdrawn: that "CRY is nearly radial at base" (it has no usable isochron structure to be
radial or otherwise) and that "CRY is the most promising probe" (it is the least usable).

**What survives.** The BMAL1 result is unaffected and is the one to quote: a direction exists
that is quiet in the LC and loud in the PTC, `rho = 122`, confirmed by finite displacement on
the independent adaptive solver (dLC 0.0017 vs dPTC 0.0303 at eps = 0.15, an 18x ratio against
1.2-1.6x for both control directions). DBP also passes the gate. The qualitative claim that the
decoupled direction is a property of the model rather than of the probe was supported by the
agreement of three eigenvectors, and now rests on **two** clean probes (BMAL1, DBP) rather than
three -- weaker evidence, honestly labelled, and worth re-testing on Korencic and Goldbeter.

**What changed in the pipeline.** `analysis/quality.py` now gates every surface before any
feature is quoted from it, and reproduces this exact verdict mechanically:

```
pulse    BMAL1 PASS   DBP PASS   CRY FAIL   PER FAIL
instant  BMAL1 PASS   PER PASS   CRY FAIL
```

Its hard checks are the winding set and the "scramble fraction" (the share of cells whose phase
jumps more than 0.25 cyc to their neighbour: BMAL1 0.019, CRY 0.149). The lesson is not that
CRY is a bad gene -- it is that **a topological feature extractor will always return a number**,
and the dipole filter in particular is designed to clean up exactly the artifact that here was
the entire signal. Nothing downstream could tell the difference, so the check had to be added
upstream.

**Instant mode is cleaner than pulse.** Re-running the characterization with the `instant`
perturbation gives BMAL1 a surface with `plaq = 1 -> 1`, i.e. a single raw winding plaquette and
no dipole filtering at all -- the cleanest surface in the project, and the one the fitting work
in section 5 uses.

### 3.8 Different directions change the PTC in different WAYS

![Direction gallery](docs/figures/almeida_direction_gallery_BMAL1_pulse.png)

Sweeping five directions spanning the rho spectrum (122 -> 24.6 -> 9.59 -> 1.07 -> 0.006) at
eps up to +/-1.5 gives qualitatively distinct modifications, not merely different magnitudes:

| direction | rho | what it does to the PTC |
|---|---|---|
| decoupled | 122 | a broad, smooth, single-signed offset at high dose, plus a dipole at the singularity |
| eigendirection 1 | 24.6 | the same broad offset with the OPPOSITE sign, and a sharper dipole |
| eigendirection 2 | 9.59 | almost purely a singularity displacement -- nearly zero change elsewhere |
| eigendirection 6 | 1.07 | reshapes the type-0 boundary as a diagonal band |
| coupled | 0.006 | global high-frequency reorganisation; the orbit is lost entirely for eps < -0.3 |

So the eigenbasis is not just an ordering by magnitude -- it separates *kinds* of isochron
change. Direction 2 in particular is close to a pure "move the black hole" knob, which is what
a radialization program would want to steer with.

---

## 4. Method notes that changed an answer

Each of these was found by an independent cross-check, not by inspection, and each had already
produced a plausible-looking wrong number.

1. **A small BVP residual does not mean you have a limit cycle.** `phi_T(y0) - y0 = 0` holds at
   any equilibrium for any `T`, and the phase condition `f(y0)[ref] = 0` holds there too
   because the whole field vanishes. Almeida at `vr x4`: residual ~1e-15, `T = 4.6e5 h`, and
   an "LC sensitivity" of **5e98**.
2. **Negative-orthant runaways do too.** The models clamp the state at 0 inside the RHS, which
   makes the field degenerate for negative states, so a linear runaway closes on itself. At
   `krr x2`: a "cycle" reaching **-2.6e12** that passed both the amplitude and period tests
   and read as a sensitivity of 1e11.
3. **Continue through the relaxation, not into Newton.** Almeida admits a **stable equilibrium
   alongside its limit cycle**, and undamped Newton handed the previous solution jumps into
   it: `vr x1.26`, one step from base, was reported "dead" while an LSODA relaxation shows a
   healthy oscillation of relative amplitude 6.8. Of 8 sampled rejections, **7 genuinely
   oscillated**.
4. **The transient skip must come from the Floquet multiplier.** There is no safe constant.
   Almeida (mu = 0.546) needs 12 periods for a transient to fall to 1e-3; Goodwin (mu = 0.951)
   needs **138**. A fixed 6 leaves 2.6% of Almeida's transient and 74% of Goodwin's — and with
   the two PTC engines skipping different amounts they disagreed by 0.33 cyc while each looked
   perfectly plausible alone.
5. **The Fourier window must span an integer number of TRUE periods.** Windowing on the
   nominal period leaks harmonics into the fundamental and biases the phase, so the dose-0 PTC
   stops being the identity: Goodwin (nominal 24.0 vs true 23.540) was off by 3.6e-3 cyc, 93x
   worse than after the fix.
6. **A phase is only meaningful if there is an oscillation.** `arg` of a near-zero Fourier
   coefficient is noise, and the phase readout returns it without complaint — reporting a
   confident type-0 PTC for a Goodwin clock that had stopped permanently.

### 4.1 Validation status

`python -m engine.validate` — eight checks, all four models PASS:

| check | Almeida | Korencic | Goldbeter | Goodwin |
|---|---|---|---|---|
| jax RHS vs numpy RHS | 0 | 4.5e-16 | 0 | 0 |
| orbit residual | 1.3e-13 | 8.7e-15 | 5.8e-15 | 7.8e-16 |
| period vs adaptive relaxation | 2.8e-09 | 1.3e-11 | 1.5e-10 | 5.1e-11 |
| 8x self-convergence | 2.9e-09 | 1.3e-11 | 4.4e-11 | 6.7e-12 |
| gauge identity | 1.3e-13 | 6.3e-15 | 1.6e-15 | 3.6e-16 |
| gauge covariance of the orbit | 1.4e-14 | 3.9e-15 | 2.6e-15 | 1.5e-14 |
| **two PTC engines agree** | **1.6e-05** | **7.8e-04** | — | **2.6e-04** |

The last row is the one that matters: `engine/ptc.py` (JAX, fixed-step RK4, Fourier phase)
against `engine/reference.py` (scipy LSODA, peak-matching phase) — no shared numerical
machinery, **no offset fitting**, and agreement required on which cells are dead as well as on
the live phases.

---

## 5. Fitting: can a single-gene PTC identify the model?

This is the deviation from the batch-1 plan, taken because the CRY/PER surfaces turned out to
be unusable (3.7) and the more interesting question is whether a model this small can be
RE-FITTED at all -- something that never worked for Mirsky.

**The experiment.** Fit Almeida so that its single-gene PTC matches a perfect radial-isochron
(Poincare) target, with no constraint on limit-cycle shape beyond "it still oscillates". Probe:
**BMAL1**, `instant` mode -- the cleanest surface in the project (`plaq 1 -> 1`, i.e. one raw
winding plaquette and nothing for the dipole filter to do).

### 5.1 Identifiability is SCALE-DEPENDENT (a claim corrected twice)

**First claim.** That Almeida is "not a sloppy model": `J_PTC` rank 13 of 13, condition ~18, so
a single gene's PTC constrains every physically meaningful combination. Offered as the main
reason to expect the fit to work.

**First correction, itself an over-correction.** Those numbers are from the batch-1 `coupling`
jacobians -- pulse mode, wide dose grid, 15 parameters -- while the fit runs instant mode on the
capped window over 16 quotient directions. Measured there: rank **7/16**, condition **2.0e3**.
That prompted "it IS sloppy", which is also not right.

**What is actually true.** The two measurements differ in the FINITE-DIFFERENCE STEP, and the
answer depends on it. `analysis/coupling.py` differences on the factor grid (x1.26 / x0.79, i.e.
h = log 1.26 = 0.231 in log-parameters); the conditioning study used h = 1e-4. Sweeping h on one
fixed configuration (pulse, 12 phases x 8 doses, cap 6 S_crit):

| h | parameter change | rank @1e-2 | condition |
|---|---|---|---|
| 1e-4 | 1.000x | 8/16 | 2.00e3 |
| 1e-3 | 1.001x | 8/16 | 2.00e3 |
| 1e-2 | 1.010x | 8/16 | 1.17e3 |
| 0.05 | 1.051x | 10/16 | 4.33e2 |
| **0.231** | **1.26x** | **16/16** | **6.46e1** |

h = 1e-4 and 1e-3 agree exactly, so the small-step values are the converged local derivative, not
a differencing artifact.

**So both numbers are right and they answer different questions.**

- **At the +-26% scale the model IS identifiable from one gene's PTC**: all 16 directions
  produce a distinguishable change, condition ~65. A 26%-different Almeida is tellable from
  nominal.
- **Locally it is not**: rank 8 of 16, condition 2000. About eight directions are flat at the
  bottom.

That is a broad, well-shaped basin with a flat floor -- and it predicts exactly what T1 did:
the cost fell 10x (the basin is real and findable) while the parameters drifted further away
(the floor carries no information).

**The consequence for fitting.** A PTC fit can place Almeida within roughly tens of percent and
cannot pin it further. Any recovered parameter set must be reported as a basin, not a point, and
"the cost went down" is not evidence that the parameters are right -- T1 is the counterexample.

**And what does NOT explain it.** Four factors were varied one at a time and none moved the local
rank: phase/dose resolution (4x more cells: 7/16 -> 7/16), dose ceiling (6 -> 18 x S_crit:
7/16 -> 7/16), perturbation mode (instant 7/16 vs pulse 8/16), and step size within the
small-step regime. The flat directions are a property of the single-gene PTC map itself.

**Methodological note, now four errors deep.** Mirsky's dose grid, the amplitude floor, the Hopf
barrier scale, and this conditioning number were all inherited across a change of setup, and all
four were wrong in the new one while looking authoritative. Each was caught only by testing
against a case with a known answer. Nothing scale-bearing should be carried into Korencic or
Goldbeter without re-measuring it.

### 5.2 The failure this is designed against

`input_screen/radialize.py` already ran this experiment on Mirsky and **converged to a
degenerate optimum**: cost 0.89 -> 0.159 in 11 L-BFGS iterations, achieved by annihilating the
singularity and collapsing the limit-cycle amplitude. The optimizer walked the clock to a Hopf
bifurcation; the twist "improved" because the oscillation was dying. Parameters moved less than
4%.

Both terms of that cost (twist span, singularity position) are *minimized* by a dying
oscillator. The cost here inverts that: it is a **pointwise match over the whole surface**, and
an unusable cell scores the **maximum** rather than being dropped. A dead oscillator therefore
becomes the worst point in the space instead of the best. Asserted by
`python -m fit.cost --selftest`: base **0.292**, Hopf-collapsed **2.748**.

### 5.3 Four things the plan got wrong, each caught by a measurement

Recorded because each was a plausible assumption that a check refuted.

1. **"Hazard 1 is about RK4, so the adaptive backend fixes the gradient."** Half right. The
   blow-up (|grad| = 2.0e75) *survived* the switch to Tsit5. Three further hypotheses were
   refuted too -- not the penalty terms (the Hopf barrier's own gradient is 1.4e-03), not
   amplitude collapse or runaway (relative amplitude stays in [0.999, 1.000475] with every cell
   alive), and not reverse-mode instability over the transient skip (shortening `skip_p` 8 -> 2
   -> 1 leaves it at 1.2e73, flat). What remains is genuine non-smoothness in PARAMETERS: a
   +454 kick into a species whose cycle spans ~4 lands near the basin boundary of the
   equilibrium that coexists with the cycle. The trajectory still returns -- hence a healthy
   amplitude and a bounded cost -- but *where* in phase it returns depends on which side it
   passed. Fix: `fit/doses.py` caps the fit window at `6 * S_crit`; |grad| 2.0e75 -> **1.13**,
   and autodiff then matches finite differences to 1e-4.
2. **The characterization dose grid is not the fit dose grid.** They differ by design and now
   have separate code paths.
3. **Two constants inherited from input_screen were charging a healthy ground truth.** The
   amplitude floor (0.5 of nominal) charged 0.153 to a displaced-but-healthy truth whose PTC
   residual was 1e-12; the Hopf barrier (margin 0.015, k 200 -- Mirsky's scale) charged 0.766
   to a truth at `Re(lambda) = 0.0143`, which is POSITIVE and therefore genuinely
   self-sustained. Either would have moved the global minimum off the truth and made the
   control measure the penalty rather than the landscape. Both are now derived from the model
   (`amp_frac = 0.05`, `margin = 0.02 * re_nominal`, `k = 30 / re_nominal`).
4. **RK4 at dt=0.02 is the wrong answer on this dose range, not merely the slower one.** It
   disagrees with the adaptive backend by 8.2e-03 cyc and *converges to it* on refinement
   (5.97e-04 at dt=0.01, 7.65e-05 at dt=0.005) -- which is independently what `analysis/scrit`
   concluded when it chose dt=0.00125 for BMAL1/instant.

The general lesson, now three times over: **constants and grids inherited from the Mirsky work
are not transferable to a smaller model**, and they fail silently. Expect the same moving to
Korencic and Goldbeter.

### 5.4 THE RESULT: from-distance locality is not a large-model artifact

This is what the model switch was made to find out, and it is a negative answer.

`input_screen` PROJECT_SUMMARY names "from-distance locality" as the framework's central
weakness: on Mirsky, L-BFGS recovered **132/132** parameters when started AT the truth and
stalled near **26/132** when started away from it. The natural hypothesis -- the one that
justified moving to smaller models -- was that this is a symptom of 132 parameters with ~100
flat directions, and would go away at 16.

**It does not.** The self-recovery control on Almeida, in-class target, truth reachable by
construction:

| start | optimizer | cost (floor 3.9e-4) | RMS log-distance to truth | params within 10% |
|---|---|---|---|---|
| **at the truth** | L-BFGS | 0.000392 -> **0.000392** | 0.283 -> **0.0000** | **18/18** |
| from nominal (26% away) | L-BFGS | 0.592 -> 0.0607 | 0.283 -> 0.685 | **0/18** |
| from nominal (26% away) | CMA-ES | 0.592 -> **0.0286** | 0.283 -> 0.597 | **2/18** |

The from-truth run settles the objective: started at the answer, L-BFGS stays there in two
iterations and returns all 18 parameters exactly. **So the truth is a genuine isolated minimum
and the cost is sound** -- the failure from nominal is about REACHING it, not about what is
being minimized. That is the same signature as Mirsky, reproduced at 16 dimensions on an
unrelated model with a well-conditioned coarse structure.

Two optimizers, one gradient-based and one gradient-free, land 0.6 away in parameters from a
start 0.28 away. Neither reaches the floor: CMA's best is 0.0286, still **73x** above the
truth's cost, so this is not a wide flat plateau being wandered -- the search never gets down
to the floor at all.

**Why the landscape is so hostile.** Gradient norms during these runs reached **1.98e25** from
nominal and **1.58e42** starting at the truth (39 of 49 evaluations clipped). The cause is
identified: Almeida has a stable equilibrium coexisting with its cycle, and a perturbed
trajectory that passes near that basin boundary returns to the cycle at a phase that depends on
which side it passed. The forward value stays bounded and healthy -- relative amplitude in
[0.999, 1.0005], every cell alive -- so the cost looks perfectly well behaved while its
derivative does not exist in any useful sense. `fit/doses.py` keeps the worst doses out of the
window and `fit/search.GRAD_MAX` rescales what is left, but the spikes are intrinsic to the map.

**What this means for the programme.** The model switch was the right experiment and it
answered its question: the obstacle to fitting PTCs is not model size and not
gauge-degeneracy. It is a property of PTC landscapes -- a well-defined minimum surrounded by a
region in which no local method can navigate. Going to still smaller models will not help.
The productive directions are (a) better targets: more probes, if genes are non-redundant
(5.5), and (b) accepting basin-level answers, since 5.1 shows the model IS identifiable to
within tens of percent even where it cannot be pinned.

**T3 (the radial fit) is therefore NOT worth running yet**, and has not been run. Fitting an
out-of-class target through a map that cannot recover an in-class one would produce a number
with no interpretation.

### 5.5 Genes are NOT redundant -- the opposite of the Mirsky prescription

Since dose variation was ruled out (5.1: widening the window 3x moves nothing), the remaining
lever is the probe. Rank of stacked per-gene surface jacobians, each block scale-normalized so
no probe dominates, **quality-gated surfaces only**:

| probes | rank @1e-2 | rank @1e-3 | condition |
|---|---|---|---|
| BMAL1 alone | 7/16 | 14/16 | 2.02e3 |
| any single (BMAL1, PER, PER_CRY, DBP) | 7-8/16 | 13-15/16 | 2.0e3 - 9.8e3 |
| any two | 9-11/16 | 15-16/16 | 8.2e2 - 1.9e3 |
| any three | 11-13/16 | 16/16 | 5.2e2 - 8.3e2 |
| **all four** | **13/16** | **16/16** | **5.08e2** |

Four probes nearly double the determined rank and improve conditioning 4x. Three directions
remain poorly determined even then.

**This inverts the Mirsky design rule.** There the finding was that single-gene PTCs are highly
redundant, and the advice was *vary dose, not gene*. In Almeida dose does nothing and gene is
the informative axis. **Optimal PTC experimental design is model-dependent and does not
transfer** -- which is a result the complexity ladder was built to be able to state.

Two details worth keeping:

- **BMAL1 is the WEAKEST probe** (7/16) despite having the cleanest surface in the project
  (`plaq 1 -> 1`). `PER+PER_CRY+DBP` reaches 13/16 without it. A clean surface is not an
  informative one, and the two were being conflated when BMAL1 was chosen as the fit probe.
- **The quality gate overturned the headline number.** The first pass of this table included
  E4BP4 and reported the best single probe at 10/16 (condition 609) and four probes at 14/16
  (condition 290). E4BP4 then FAILED the gate (winding set {-1, 0, +1}), as did REV. A jacobian
  of a phase-scrambled surface is a jacobian of noise, and **noise inflates numerical rank** --
  so the broken probe produced the most attractive number, exactly as CRY's broken surface had
  produced the largest S_crit/twist/rho in 3.7. Same trap, opposite sign, caught only because
  the gate now runs before anything is quoted.

### 5.5 Status

`fit/` is built and self-tested: `target.py`, `cost.py`, `search.py` (L-BFGS, multistart,
CMA-ES), `doses.py`, `recover.py` (T1, with `--start truth` and `--optimizer`), `radial.py`
(T3, unrun), `figures.py`. T0 and T0.5 pass; T1 fails from a distance and passes from the truth.

## 5b. Next

1. **Map recovery against displacement.** T1 is done at eps = 0.3 (fails) and eps = 0 (passes).
   The distance at which it breaks is the quantitative version of "from-distance locality",
   which was never measured on Mirsky, and `fit/recover.py --eps` sweeps it directly.
2. **Do genes carry independent information?** Rank of stacked multi-gene PTC jacobians. On
   Mirsky the lesson was "vary dose, not gene", but dose variation is already ruled out here
   (5.1), so if Almeida's genes are non-redundant the two models differ in a way that changes
   experimental design.
3. **T2, basin structure** -- multistart clustered in quotient coordinates. Note the from-truth
   control means any cluster found is about reachability, not about the cost having many
   minima.
4. **T3 (radial fit) stays parked** until something above changes the picture.
4. **Run (a) and (b) for Korencic and Goldbeter.** The complexity ladder 11 -> 18 -> 34 -> 52 ->
   (132) makes "identifiability vs model size" measurable rather than anecdotal. Re-derive, do
   not inherit, every scale-bearing constant.
5. **Re-test the "same decoupled direction" claim** (3.7) now that it rests on two clean probes
   rather than four.
6. **Reduce the 34 non-converged LC settings** -- better orbit continuation, or a
   predictor-corrector along the factor grid.

---

## 6. Figures

### Where they live

`out/<model>/<analysis>/<tag>/<name>.png`, next to the npz that produced them -- **not** in one
flat folder. Each carries a `.png.meta.json` sidecar (entry script, git commit, host, time, and
the config that produced it) and a footer stamp inside the image, so a PNG pasted into a slide
is still traceable to its run. Re-running an analysis with different settings writes a
distinguishable batch instead of silently overwriting.

`docs/figures/` holds only the curated copies this document cites, promoted explicitly with
`--publish` and prefixed by model name. Regenerate everything with:

```bash
python -m analysis.figures --model almeida --target BMAL1 --tag batch1 --publish
```

Pure read of the saved npz -- no figure costs compute to rebuild.

### What each one answers

| figure | what it answers |
|---|---|
| `scrit_pulse` | winding vs dose per target; where each changes type, and where the numerics stop being trustworthy (shaded) |
| `surfaces_pulse` | the base PTC surfaces, each with its twist curve in a panel below |
| `tornado_BMAL1_pulse` | LC and PTC sensitivity per parameter on a shared ordering, plus their ratio |
| `lc_examples` | what an LC sensitivity number looks like as an actual cycle change |
| `ptc_examples_BMAL1_pulse` | the same for the PTC, with the Delta new-phase panel |
| `directions_BMAL1_pulse` | the combinatorial decoupling: spectrum, parameters vs directions, loadings |
| `direction_examples_BMAL1_pulse` | one nudge, shown in the LC and in the PTC side by side |
| `sweep_BMAL1_pulse_<direction>` | a RANGE of nudges along one direction, with the PTC features tracked along it |

### Reading a PTC surface

On a **PTC surface**: x is old phase, y is dose (log), colour is new phase; the dashed white
line is `S_crit` and the white dot is the phase singularity.

On a **twist panel**: x is dose (log), y is the stable fixed-point phase wrapped to [0, 1],
with dotted guides at 0 and 1, a vertical guide at `S_crit` and a dot where the curve crosses
it. This is the input_screen house orientation (`plot_twist_movers`), and 0/1 wraps are handled
by breaking the line with NaN rather than by unwrapping -- keeping the phase axis literal so a
value can be read straight off it. Note the two panel types therefore do NOT share an axis.

**The fixed-point curve is no longer drawn over the surface.** A line across a cyclic colour
field is hard to read and competes with the surface it annotates.

## 7. Glossary

- **PTC** — phase transition curve: new asymptotic phase vs old phase, as a function of dose.
- **Type-1 / type-0** — weak (winding 1) vs strong (winding 0) resetting. The transition passes
  through a **phase singularity** where the perturbed state is phaseless.
- **S\***, **phi\*** — dose and old-phase of the singularity. Topological defects: discontinuous
  and grid-quantised.
- **Twist** — how the isochrons shear with distance from the cycle, measured as the stable
  fixed point's phase vs dose. Smooth and gauge-invariant; radial isochrons give zero twist.
- **Floquet multiplier** — the factor by which an off-cycle perturbation shrinks per period.
  Sets how long you must wait before an "asymptotic" phase reading is actually asymptotic.
- **Gauge** — the exact unit-rescaling symmetry of the parameters. Flat for any observable
  that does not fix the concentration units.
