# Repository map

One place to answer "which module is canonical for X?". Companion to
[`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) (the scientific narrative); this is the *code*
index.

Everything runs as a module from the repo root, **with the interpreter called by its full
path** -- see [README](README.md#running); `$PY` below is that path:

    export PY=/home/galmanel/miniconda3/envs/mirsky/bin/python     # cluster
    export PY=/c/Users/galma/anaconda3/python.exe                  # local

so `$PY -m engine.orbit`, `$PY -m analysis.scrit --model almeida`. A bare `python` picks up
whatever is on PATH, which is how a dry-run and the array job it validated end up running
different code.

## Layout

| dir | what lives there |
|---|---|
| `models/` | the zoo + the capability contract. A model declares what it is; downstream code asks the model instead of importing its private tables. |
| `engine/` | limit-cycle solver, perturbations, the PTC, and an INDEPENDENT adaptive-solver reference. Model-agnostic throughout. |
| `analysis/` | the drivers that produce results. Headless, shardable, provenance-stamped. |
| `gauge/` | the exact unit-rescaling symmetry, derived per model. **Read before comparing any two parameter sets.** |
| `fit/` | the radialization pipeline: target, cost, search, dose window, campaigns, aggregation, figures. |
| `campaigns/` | one JSON per campaign. List-valued fields expand to a SLURM array; the file IS the experiment's record. |
| `fixtures/` | anything a later run DEPENDS on, S_crit above all. Tracked, promoted deliberately -- see hazard 16. |
| `slurm/` | cluster templates. Thin: the drivers already shard. |
| `out/` | results, `out/<model>/<analysis>/<tag>/`. Git-ignored; each npz carries a `.meta.json` provenance sidecar. |
| `docs/` | `FIT_VALIDITY.md` -- the surface validity contract every fit must satisfy, and the plan to enforce it. Read before changing `fit/cost.py`. |
| `docs/figures/` | ONLY figures explicitly cited by PROJECT_SUMMARY. Not a gallery, not a dumping ground -- see the figure rule under Hazards. |

## "What do I run for…?"

| Task | Run |
|---|---|
| Check a new model is wired up correctly | `$PY -m models.api <name>` then `$PY -m engine.validate <name>` |
| **The full gate** (nothing in `analysis/` is trusted until this passes) | `$PY -m engine.validate` |
| Where does each target change PTC type, and on what dose grid? | `$PY -m analysis.scrit --model M` |
| The base PTC surfaces and their features | `$PY -m analysis.characterize --model M` |
| How much does each parameter move the limit cycle? | `$PY -m analysis.lc_sens --model M` |
| How much does each parameter move the PTC? | `$PY -m analysis.ptc_sens --model M --target T` |
| **Are the LC and the PTC coupled?** (the batch-1 question) | `$PY -m analysis.coupling --model M --target T` |
| Is the gauge declaration right? | `$PY -m gauge.validate [identity\|algebra\|invariance]` |
| Launch a campaign (after `--dry-run`) | `sbatch --array=0-N slurm/campaign_cpu.sh campaigns/<c>.json` |
| **Join a finished campaign's tasks into one result** | `$PY -m fit.aggregate --model M --tag <c> [--kind genes]` |
| The campaign figures (pure read of that npz) | `$PY -m fit.figures --which seeds\|genes --tag <c>` |
| One run's figure | `$PY -m fit.figures --which radial --tag <run-tag>` |
| **Do the fitted orbits actually ATTRACT?** | `$PY -m fit.stability --model M --tag <c>` (`--selftest` first) |
| Are the fitted surfaces flat, or is the transition just below the window? | `$PY -m fit.rescan --model M --tag <c>` |
| Their figures | `$PY -m fit.figures --which cycles\|rescan --tag <c>` |
| **Compare campaigns with different objectives** | `$PY -m fit.compare_arms --model M --arms A B C --control A` |

Order matters: `scrit` derives the dose grid and the integrator step that `characterize` and
`ptc_sens` consume, and `coupling` is a pure read of `lc_sens` + `ptc_sens`.

Order matters for a campaign too: **aggregate before plotting**. `fit.figures --which seeds`
reads the JOINED npz, not the per-task ones, because the comparison it draws does not exist
until the tasks are joined.

## Index

### `models/`
| File | Role |
|---|---|
| `api.py` | **The replaceability contract.** Five capability tiers (CORE / DIMENSIONS / JAX / PERTURBATION / OBSERVABLES); `require()` fails by name, `check_conformance()` validates a new model. |
| `base.py` | `ClockModel` ABC, `JaxDictParams` mixin, `MODEL_REGISTRY`. |
| `almeida.py` | Almeida 2020, 8 states / 18 params. Transcription-factor level, one reversible complex. |
| `korencic.py` | Korencic/Grabe, 15 states / 34 params. Five genes x a 3-stage delay chain; no complexes. |
| `goldbeter.py` | Leloup & Goldbeter 2003, 16 states / 52 params. Phosphorylation + transport + MM degradation. |
| `goodwin.py` | Goodwin/Gonze, 3 states / 11 params. **Smoke test, not a science target** — smallest thing that oscillates, structurally unlike the others, gauge known-good from input_screen. |

### `engine/`
| File | Role |
|---|---|
| `orbit.py` | **CORE.** The periodic orbit as a solved BVP (Newton + a gauge-covariant phase condition, time-rescaled so `tau` IS the phase). Differentiable, vmappable. Also Floquet multipliers and `make_orbit_finder`, which owns every rule for getting a trustworthy orbit at a displaced parameter set. |
| `perturb.py` | The two generic perturbations (`pulse`, `instant`) and the forced RK4 flow, which also tracks the trajectory minimum so unstable points can be discarded. |
| `ptc.py` | **CORE.** The PTC on the solved orbit. Calibrated to the identity at dose 0; `recommended_skip` derives the transient allowance from the Floquet multiplier; `valid_mask` / `phase_or_nan` separate "no phase" from "a phase". |
| `reference.py` | **The independent ground truth**: scipy LSODA + peak-matching. Shares no numerical machinery with `ptc.py`, which is the entire point. Deliberately slow — do not optimise it. |
| `validate.py` | **ENTRY. The gate.** Eight checks per model. |

### `analysis/`
| File | Role |
|---|---|
| `winding.py` | **CORE.** Pure-numpy surface analysis: winding, dipole-filtered singularities, base continuation, circular hole imputation, twist curve. Touches no solver, which is why input_screen's equivalents survived every solver correction there. |
| `scrit.py` | ENTRY. S_crit, re-entrancy, the validity ceiling, and the adaptive dose grid — refining `dt` until S_crit is trustworthy. |
| `characterize.py` | ENTRY. Base PTC surfaces + features on those grids. Objective (a). |
| `lc_sens.py` | ENTRY. Per-parameter limit-cycle sensitivity. Target-independent: runs once per model. |
| `ptc_sens.py` | ENTRY. Per-parameter PTC sensitivity on a FIXED base dose grid. Saves every raw grid. |
| `coupling.py` | ENTRY. Objective (b): the scatter, the gauge-quotiented principal angles, and the shortlist to confirm. Pure read. |

### `fit/`
| File | Role |
|---|---|
| `config.py` | `RunConfig`: every setting a run depends on, in one dataclass. Two runs with equal configs ARE the same experiment, and the npz records which one. |
| `target.py` | The radial (Poincare) target, its `(k, psi)` registration, and the smooth `soft_singularity`. |
| `cost.py` | **CORE.** The pointwise surface cost, the gauge quotient (`quotient_basis`), the amplitude floor and the Hopf barrier. `--selftest` asserts the anti-degeneracy invariant. |
| `doses.py` | The FIT dose window -- capped at `max_factor * S_crit`, which is what keeps the gradient finite (hazard 11) -- plus `--promote` for the S_crit fixture (hazard 16). |
| `search.py` | L-BFGS, multistart, CMA-ES (anneal / ipop), BOBYQA, LM. BOBYQA is the measured winner (PROJECT_SUMMARY 5.6). |
| `parallel.py` | Population-parallel evaluation for CMA. The cluster buys THROUGHPUT, not latency (5.8). |
| `viability.py` | Rejection-sampling for `start='viable'`: draw random parameter sets, keep the healthy circadian clocks. Its rejects are a free viability map. |
| `radial.py` | ENTRY. One radialization (`run`), a seed set (`run_seeds`), the degeneracy checks and the verdict. |
| `recover.py` | ENTRY. T1, the self-recovery control: fit an IN-CLASS target whose answer is known. |
| `campaign.py` | ENTRY. One config with list-valued fields -> a matrix of runs, indexed for a SLURM array. `--dry-run` validates every entry at submission time. |
| `aggregate.py` | **ENTRY. Joins a campaign's array tasks into one result** -- the 16-seed comparison that no single task can see -- and re-derives the unsaturated twist metrics. Refuses to join runs that are not the same experiment. |
| `stability.py` | **ENTRY. The stability guard.** Perturb the fitted cycle and integrate: does it come back, run away, or settle on a point? Measures each orbit's own numerical FLOOR first, because that spans five orders across a campaign and a perturbation below it measures noise (hazard 19). |
| `rescan.py` | ENTRY. Re-renders a campaign's fitted PTCs finer and down to dose 0, with the dose-0 identity row as the readout-calibration control. Pins the run's gauge basis (hazard 18). |
| `contract.py` | **ENTRY. The six-check surface validity contract** (orbit / attractor / dose-0 calibration / phase resolution / aliveness / bracketing), pinned to a regression fixture. Five of six are free from what `_surface` already computes. |
| `compare_arms.py` | **ENTRY. Compares campaigns that used DIFFERENT objectives**, by re-scoring every endpoint under the CONTROL arm's objective -- the only column comparable across arms. Refuses to report a number for an endpoint where the cost is not well defined at double precision. |
| `promote_start.py` | ENTRY. Promotes a finished fit's parameters into `fixtures/starts/` so a later run can START there, without `out/` becoming an input (hazard 16). Keyed on THETA, not `v` (hazard 18). |
| `probe_window.py` | ENTRY. Decision probe: is a dose window anchored to the candidate's own transition continuous in parameters? (Yes, where an orbit exists -- PROJECT_SUMMARY 5.13b.) |
| `figures.py` | Pure read. `--which radial\|recover` for one run; `seeds\|genes\|cycles\|rescan` for a campaign. |
| `benchmark.py`, `multires.py` | Optimizer head-to-head; multi-resolution helpers. |

### Root
| File | Role |
|---|---|
| `paths.py` | Output protocol `out/<model>/<analysis>/<tag>/`, array-job tag/shard conventions, and the **provenance guard**. |
| `parallel.py` | One dispatcher (`serial`/`joblib`/`ray`) plus the shard helper. `ray` is optional on purpose. |
| `plotting.py` | House conventions: cyclic `cmocean.cm.phase` for phase, achromatic overlays, grey for missing. |

## Hazards

1. **Never autodiff the FIXED-STEP RK4 PTC at nontrivial dose.** In input_screen that
   produced `|J|` up to 3.95e83 against an actual response of 0.006 and invalidated a whole
   results table. The batch-1 analyses are all finite-difference, so this does not touch them.

   *Amended.* As first written this read as a blanket ban on gradients, and it is not one --
   the indictment is of fixed-step RK4, and input_screen's own resolution was the adaptive
   backend, with which its gradient optimizer outperformed CMA-ES. That backend is now ported
   (`engine/flow.py`, `backend='diffrax'`). The enforceable rule is therefore: **a gradient
   requires the adaptive backend AND a passing `$PY -m fit.cost --gradcheck`.** Do not
   assume the check passes -- it FAILS on an unrestricted dose grid, which is why
   `fit/doses.py` exists; see hazard 11.
2. **A small BVP residual does not mean you have a limit cycle.** Any equilibrium satisfies
   `phi_T(y0) = y0` for any `T`, and so does a runaway in the negative orthant where the
   clamped RHS is degenerate. `make_orbit_finder` tests amplitude, positivity and period band
   as well; do not bypass it.
3. **A phase is only meaningful if there is an oscillation.** `arg` of a near-zero Fourier
   coefficient is noise. Use the `raw` readout and `phase_or_nan`, so "strong resetting" and
   "you killed the clock" stay distinct.
4. **The fixed-step stability ceiling is a property of `dt`, not of the model.** At `dt=0.02`
   Almeida appeared to have 3 resetting targets; refined, it has 7. Any dose scan must refine.
5. **Read isochron change from TWIST, not the singularity.** `(S*, phi*)` is a topological
   defect: discontinuous and grid-quantised. Twist is smooth and gauge-invariant.
6. **Quotient the gauge before any parameter-space comparison.** A gauge direction is exactly
   LC-null while moving a fixed-absolute-dose PTC, so leaving it in manufactures precisely the
   "low LC / high PTC" signal `coupling.py` is looking for.
7. **Never swallow an exception**, and never let a downstream failure destroy an upstream
   result — PTC generation and feature detection sit in separate `try` blocks.
8. **Never `| tail` a background job**, and never redirect one without `python -u`. Output
   buffers, and a healthy job then looks hung — this cost a 7-minute run that had to be killed
   and restarted purely because its log was empty.
9. **SAVE THE RAW ARRAYS. Analysis is expensive; replotting is cheap.**
   Every driver must persist the actual integrated output — cycle profiles, PTC surfaces,
   amplitudes, validity masks — not just the scalars derived from them. A summary statistic
   cannot be re-derived into a figure, cannot be re-analysed when the feature extractor
   changes, and cannot be inspected when a number looks wrong; the only recovery is to re-run
   the sweep. Storage is a few hundred kB against minutes-to-hours of adaptive integration.

   Audit it, do not assume it:

   ```bash
   python -c "
   import numpy as np, glob
   for f in glob.glob('out/**/*.npz', recursive=True):
       z = np.load(f, allow_pickle=True)
       raw = [k for k in z.files if getattr(z[k], 'ndim', 0) >= 2]
       print(f, '<-- NO RAW ARRAYS' if not raw else raw)"
   ```

   A file of a few kB where its siblings are 60–200 kB is the tell. `analysis/confirm.py`
   shipped exactly that way: it ran ~10 minutes of adaptive integration per invocation and
   saved six scalars.

## Provenance

The model-agnostic core was copied and generalised from `../input_screen/` at commit
`e955873`; each file names its origin in its header. `input_screen/` is not modified by this
repo. What changed in the copy: the PERTURBATION tier collapsed to `perturbable_targets()`
(the perturbation is now generic), `gauge` gained `scale_constraints()` (Korencic needs it),
and the PTC gained a calibrated phase origin, a Floquet-derived transient skip, and validity
tracking.

10. **A radialization cost has a trivial global optimum: a dead oscillator.**
    input_screen/radialize.py fit Mirsky to a radial target with `twist_cost + feature_cost`
    and no limit-cycle term. It CONVERGED -- 0.89 -> 0.159 in 11 L-BFGS iterations -- by
    walking the clock to a Hopf bifurcation: singularity annihilated, LC amplitude collapsed,
    twist "improved" because the oscillation was dying, parameters moved less than 4%. Both
    terms are minimized by a dying oscillator.

    `fit/cost.py` is built against exactly this: a POINTWISE surface match (not a feature
    match), with unusable cells scored at the MAXIMUM rather than dropped, plus an amplitude
    floor and the Hopf barrier. `$PY -m fit.cost --selftest` asserts the invariant --
    measured base 0.292 against collapsed 2.748 -- and if it ever fails, the degeneracy has
    been rebuilt and no result from that pipeline should be believed.
11. **The PTC gradient explodes at far-supercritical dose, adaptive backend or not.** Measured
    on Almeida/BMAL1/instant at nominal: |grad| is 0.7 near S_crit and 2.4 at 3-5x S_crit, but
    2.0e75 on the full characterization grid (to 18x S_crit). The cost VALUE stays bounded in
    [0,1] throughout, so nothing in the objective looks wrong -- only the derivative reveals
    it. Fits use `fit/doses.fit_dose_grid`, capped at `max_factor * S_crit`; characterizations
    still use the wide grid, which is correct for them.
12. **A topological feature extractor always returns a number.** `detect_grid` reported
    `S_crit = 138.1` and a twist of 0.285 for a CRY surface that was phase-scrambled, and the
    dipole filter -- designed to clean up fast phase changes -- silently reduced 13 spurious
    plaquettes to one arbitrary survivor. Nothing downstream could tell. Gate every surface
    with `analysis/quality.py` BEFORE quoting any feature from it.

13. **A FIGURE IS A RESULT. IT BELONGS IN `out/`, NOT IN `docs/figures/`.**

    Every figure is written to `out/<model>/<analysis>/<tag>/` by `paths.save_figure`, beside
    the `.npz` that produced it, with a `.png.meta.json` sidecar recording the script, git
    commit, host, time and configuration, and a provenance line stamped inside the image. That
    is what makes a figure re-findable and re-derivable a month later.

    `docs/figures/` is NOT a second home for figures. It holds only the curated copies that
    PROJECT_SUMMARY actually cites, so the document travels with its illustrations. A figure
    that no text references does not belong there.

    **Publish only via `paths.save_figure(..., publish=...)`, never with `cp`.** The publish
    path writes a `.source.txt` next to the copy naming the run it came from; `cp` writes an
    orphan with no way back to its data. The test is one line:

        for f in docs/figures/*.png; do [ -f "$f.source.txt" ] || echo "ORPHAN: $f"; done

    Two orphans were found by exactly that check, both `cp`-ed there by hand while writing up
    results nobody had asked to publish. Removed; the originals were already in `out/` with
    their sidecars, so nothing was lost -- which is the point.

    The same check in reverse catches the other failure: a figure PROJECT_SUMMARY references
    that was never published at all (`almeida_coupling_BMAL1.png` is currently a broken link).

14. **THE PHASE OBSERVABLE IS A NUMERICAL CHOICE, AND THE WRONG ONE FAKES A DYNAMICAL FAILURE.**

    Almeida used `reference_variable = 'BMAL1'` for BOTH roles -- the orbit solver's Poincare
    section (`rhs(y0)[ref] = 0`) and the PTC's Fourier phase readout. BMAL1 was picked on its
    BASE-POINT numbers: largest relative amplitude (4.26), cleanly unimodal. Neither property
    survives the parameter sets an optimizer visits.

    At the RAD01 radialization optimum, over a 100 h transient:

        BMAL1  min 1.2e-27      ROR    min 4.1e-28      E4BP4  min 3.8e-17
        REV    min 6.8e+01      PER    min 3.4e+00      REV    max 1.2e+03

    Thirty decades inside one state vector. Three species collapse to numerical zero while REV
    sits at 1e3. Consequences, all of which were initially misread as the model being broken:

      * the section stops being unimodal (1.7 `dy/dt` sign changes per period instead of 2), so
        Newton lands wherever the guess points it -- the SAME parameter set returned a period of
        0.159 h from one guess and 43.1 h from another, a 99.6% disagreement;
      * the stiffness exhausts diffrax's `MAX_STEPS=50_000` on long spans, and the sampler then
        NaNs the ENTIRE trajectory, which is indistinguishable from a blow-up in a plot;
      * `solver.cycle` called on the resulting non-converged `y0` produces negative
        concentrations and visible numerical noise -- an artifact of the failed solve, NOT of
        the dynamics.

    THE DYNAMICS WERE FINE THE WHOLE TIME. End-to-end perturbation simulations at that same
    parameter set oscillate in every species, stay strictly positive, stay bounded out to 400 h
    (`out/almeida/fit_radial/RAD01/endtoend_perturbations.png`). The fit was largely correct and
    the observable was the problem.

    So the two roles are now SEPARATE attributes with different criteria:

      * `reference_variable` (section) wants UNIMODALITY -> Almeida: `PER`, the only species
        measured unimodal at both the base point and the optimum, baseline 3.9 -> 8.2;
      * `readout_variable` (phase) wants a healthy amplitude and baseline -> Almeida: `REV`,
        which never approaches zero and is what the experiments measure.

    Switching the readout costs nothing in principle and this was CHECKED, not assumed:
    asymptotic phase is a property of the state, so BMAL1 / REV / PER readouts agree to
    **max 5.6e-4 cyc** (~50 s of a 24.8 h period) at the base point. Earlier BMAL1-readout
    results stand.

    The rule: before blaming a model for a numerical failure, check the SPREAD OF SCALES in the
    state vector and check whether the phase species is one of the collapsing ones.

15. **A DIAGNOSTIC THAT CAN FAIL SILENTLY WILL BE MISTAKEN FOR THE RESULT.** Two separate
    misdiagnoses in one session came from tools, not from the system under study:

      * `fit/figures._backfill` re-solved the orbit with `solver.guess` (the numpy peak-hunt)
        while `fit/cost._surface` uses `make_guess_fn` (the jittable relaxation), and threw the
        residual away as `_r`. The figure therefore showed an orbit THE FIT HAD NEVER
        EVALUATED, and gave no way to notice. Diagnostics must reproduce the production path
        exactly and must display the residual that says whether the answer is valid.
      * a 517 h single solve returned all-NaN from `MAX_STEPS`, plotted as an empty panel, and
        read as "the model diverges". The same set is finite and bounded at 400 h. Never let a
        SOLVER LIMIT and a DIVERGENCE render identically -- walk the span out and report which
        one it is.

16. **`out/` IS OUTPUT. IT IS NEVER AN INPUT. Anything a later run DEPENDS on lives in
    `fixtures/`.**

    S_crit is an input to every fit -- it sets the dose window -- but `analysis.scrit` writes it
    into `out/`, which is gitignored. So a fresh clone has none of it and every cluster array
    task exited with "run `analysis.scrit` first". The first fix was to search `out/` and fall
    back to `fixtures/`, and that was WRONG for two reasons:

      * it contradicted its own justification. If a local `out/` run overrides the tracked
        value, two machines silently disagree about the dose window and their fits stop being
        comparable -- the exact failure the fixture existed to prevent.
      * it was fragile in a duller way: an `out/` directory that merely EXISTS but is empty, or
        that holds the other perturbation mode, changes which branch runs.

    THE RULE IS ABSOLUTE, and it is not only about S_crit. If a result is needed as the basis
    of another run, PROMOTE it:

        $PY -m fit.doses --promote --model almeida --mode instant [--dry-run]
        git add fixtures/scrit/almeida && git commit

    Promotion is deliberate, reviewable, and shows up as a diff. "The dose window changed"
    should be a commit, not a property of whichever machine ran `analysis.scrit` most recently.

    The test that this holds: hide the fixture and confirm the code FAILS even though a
    perfectly good result is sitting in `out/`.

        mv fixtures/scrit/almeida /tmp/ && python -c "from fit.doses import fit_dose_grid;
        fit_dose_grid('almeida','BMAL1','instant',8.0,14)"    # must raise, not succeed

17. **A LOW RESIDUAL AGAINST A RADIAL TARGET IS MOSTLY A STATEMENT ABOUT WHERE THE DOSE
    WINDOW SITS.**

    A Poincare target carries old-phase structure only BELOW its own singularity. Above it the
    target resets to nearly one phase whatever the old phase was: measured on Almeida/BMAL1,
    its old-phase span falls from the ceiling 0.50 below S\* (a full sweep of the phase circle)
    to **0.05 at 6x S\***. `fit_dose_grid`
    spans `0.5x` to `max_factor x S_crit` LOG-spaced, so at `max_factor = 8` most of the rows
    -- 9 of 14 -- lie in that flat asymptote, and a pointwise cost weights them equally with
    the informative ones.

    The 16-seed BMAL1 campaign is what this looks like when it happens (PROJECT_SUMMARY 5.10c).
    Cost 0.428 -> ~0.10, verdict RADIALIZED on 12 of 16 -- and split by dose:

        rms residual BELOW S*  (5 doses)   base 0.199 -> fits 0.181 - 0.221    <- no better
        rms residual ABOVE S*  (9 doses)   base 0.276 -> fits 0.035 - 0.057    <- all of it

    Eleven of the twelve pushed their singularity out of the window entirely (`n_sing = 0`), so
    they do not even share the target's TOPOLOGY. And five reached a PTC that does not resolve
    old phase at all (range < 0.05 cyc; one is constant to four decimals) -- a phaseless
    surface, which has zero twist by construction and therefore satisfies `less_twist`
    trivially while carrying no phase information whatever. Measure that span with
    `analysis.winding.circ_span`, not a peak-to-peak about an arithmetic mean: phase wraps,
    and the naive version read 4 informative dose rows for REV against a true 1. It passes every guard the cost has:
    alive, `Re(lambda) > 0`, `mu` healthy, amplitude in range.

    This is a THIRD degeneracy route, after the dead oscillator (hazard 10) and the bad
    observable (hazard 14), and it is the one a radial target invites -- a flat PTC *is* the
    target's own high-dose asymptote.

    Three rules follow:

      * **quote the residual split at S\*, never the pooled `c_ptc` alone.** `fig_seeds` panel
        (f) and `fit.aggregate.report` do this; a number without it is not interpretable.
      * **check that the fitted surface still resolves old phase** before reading any twist
        number off it. Zero twist on a phaseless surface is not radial isochrons.
      * **costs from different targets are not comparable.** Each gene's window comes from its
        FIXTURE `S_crit` while its target is pinned to the base surface's MEASURED singularity,
        and the two disagree by up to 4.7x (PER: 37.75 vs 178.2). REV therefore had 1 of 14
        doses below its defect and won the campaign on cost with the least demanding fit.

18. **`v` IS A MACHINE-LOCAL COORDINATE. THE GAUGE-QUOTIENT BASIS IS NOT REPRODUCIBLE.**

    `fit/cost.quotient_basis` builds `B` from `np.linalg.svd(I - QQ^T)`. That matrix is a
    PROJECTOR, so its nonzero singular values are **all exactly 1** -- for Almeida, `[1]*16`
    then `[0, 0]`. Every orthonormal basis of that 16-dimensional subspace is a valid SVD, so
    which one LAPACK returns is not defined by the mathematics, and `B` can differ between
    builds, versions, or machines.

    Everything the optimizer stores is in `v`. So a saved `v` only means something alongside
    the basis that produced it, and nothing in the pipeline used to record that.

    MEASURED, on the Aug-30 campaign's own output: re-deriving the basis in this session and
    reconstructing `theta = exp(z_base + B v)` from the stored `v_fit` gave parameter sets up
    to **3.0 decades** away from the `theta_fit` the same npz records. Every surface recomputed
    that way came back uniformly DEAD -- `alive_frac = 0.000`, `amp_lc = 0` -- which is how it
    was noticed: a rescan of finished fits produced nothing but dead clocks.

    Two things make this survivable rather than fatal, and both were checked, not assumed:

      * every run of that campaign shares one basis (`fit/aggregate` now verifies it and
        RAISES otherwise), so the multimodality distances in PROJECT_SUMMARY 5.10a are sound;
      * every run stores `B`, `z_base` and `theta_fit`, so nothing is lost.

    THE RULES:

      * **`theta_fit` is the parameter set. `v` is a search coordinate.** Report, compare and
        re-evaluate parameters through `theta`.
      * Anything re-evaluating a saved `v` must pin the basis: `make_cost(..., basis=B)` with
        the run's own `B`, and then CHECK the round trip -- `fit/rescan.py` refuses to run if
        `exp(z_base + B v)` and `theta_fit` disagree by more than 1e-8 decades.
      * A `v`-space distance is only meaningful WITHIN one basis. Across campaigns, compare in
        log-theta projected onto the gauge complement instead.

    The deeper fix -- making `quotient_basis` deterministic (a fixed orthonormalisation instead
    of an SVD of a degenerate spectrum) -- has NOT been made, because it would silently change
    the meaning of `v` for every stored result. It is in Open.

19. **A PERTURBATION BELOW AN ORBIT'S OWN NUMERICAL FLOOR MEASURES THE INTEGRATOR, NOT THE
    DYNAMICS -- AND EVERY ORBIT HAS A DIFFERENT FLOOR.**

    Walk an orbit from `y0` with NO perturbation at all. It starts exactly on the cycle, so any
    distance it accumulates is pure integration error. Measured on Almeida at fixed tolerances:
    **8e-9 of the cycle diameter at the base point, 7.6e-2 at the seed-5 radialization
    optimum** -- seven orders of magnitude apart, on the same solver. There is no epsilon that
    serves both, so a FIXED one is a bug waiting for a pathological parameter set.

    `fit/cost.make_growth_fn` has one: `eps = 1e-4`. That is fine at the base point and far
    below the floor at half the fitted optima, where its "growth" is the ratio of two noise
    measurements. The first stability audit of the Aug-30 campaign therefore reported **10 of 16
    orbits REPELLING**; measured above each orbit's own floor, **10 of 16 ATTRACT**. The
    conclusion was exactly inverted.

    THE TELL IS EPSILON DEPENDENCE. A dynamical multiplier cannot vary with the size of the
    probe; a noise floor must:

        eps            1e-4    1e-3    1e-2    3e-2    1e-1
        base           0.857   0.857   0.871   0.824   0.695     <- a plateau: real
        seed 4         1.714   1.431   1.127   0.968   0.839     <- no plateau: noise

    `fit/stability.measure` measures the floor, climbs an epsilon ladder until the perturbation
    clears it by 30x, and returns **UNRESOLVED** rather than a number when no rung does. A guard
    that cannot measure must say so; guessing is how this went wrong the first time.

    TWO COROLLARIES, both measured:

      * **Distance to a sampled cycle must be to the SEGMENTS, not the vertices.** Vertex
        distance floors at half a sample spacing -- 5e-4 diameters even at 2048 samples -- so a
        decaying deviation stops decaying there and a slope fitted through the flat tail
        returned 0.727 for an orbit whose Floquet multiplier is 0.546.
      * **Cross-check the walk where Floquet IS trustworthy.** At the base point the monodromy
        is well conditioned; `$PY -m fit.stability --selftest` asserts the walk recovers
        0.546 within 10% (it measures 0.549). Assert the VALUE, not the sign -- the
        tail-contaminated version passed a sign test.

    DO NOT enable `make_cost(w_stab=...)` until `make_growth_fn` climbs the same ladder. It
    currently lands on the wrong side of the threshold for 6 of 16 fitted orbits, and the fit
    would be penalising them for the integrator's error.

20. **DOSE 0 IS A FREE CONTROL ON THE PHASE READOUT. USE IT.**

    `engine/ptc` calibrates the phase origin so that a zero perturbation returns new phase ==
    old phase. One extra dose row therefore tests, at every parameter set, whether the readout
    is calibrated AT ALL -- and it is not free-standing pedantry: on the Aug-30 campaign the
    deviation separates into two clean populations,

        calibrated  (11/16)   1.0e-04 - 9.4e-04 cyc    the Fourier readout's own resolution
        NOT         ( 5/16)   4.1e-02 - 5.0e-01 cyc    seeds 3, 4, 5, 11, 14

    two orders clear -- and those five are **the same five** the stability walk flags as dead,
    diverging or unmeasurable. A one-row check predicts a 24-period integration. `fit/rescan.py`
    puts that row on every surface it renders; `ID_TOL = 1e-2` is where the two populations
    separate, not a guess.

## Open

- `analysis/characterize.py` and `ptc_sens.py` have not yet been run for Korencic or
  Goldbeter (Almeida only).
- `instant` mode: `scrit` + `characterize` + `quality` done for Almeida
  (BMAL1 is the cleanest surface in the project); not yet swept through `ptc_sens`/`coupling`.
- Fitting (`fit/`) is in progress: target, cost and search are built and self-tested; the T1
  self-recovery control has not yet been run.
- `fit/radial._diagnose` still solves the orbit with `solver.guess` where the cost uses `make_guess_fn` -- hazard 15's first bullet, fixed in `figures._backfill` but not here. It cost three runs of the Aug-30 campaign their verdict (PROJECT_SUMMARY 5.10f). One line; not changed yet because it re-opens finished fits.
- The radialization cost has no term requiring the fitted surface to RESOLVE OLD PHASE, and no weighting toward the doses where the target is informative. Both are what hazard 17 is about.
- **The time gauge is broken by pulse-mode data, and no driver knows it.** A pulse of duration
  fixed in HOURS is a clock: under a pure time-rescale gauge motion the instant-mode PTC is
  invariant to 1.1e-07 cyc but the 8 h-pulse PTC moves 3.8e-01. Every pulse-mode analysis
  quotients with `include_time=True` and so projects out a direction its own data can see; the
  identifiability counts in PROJECT_SUMMARY 3.x should be re-derived with `include_time=False`
  (almeida 16 -> 17 directions). Instant-mode results are unaffected. Same switch is what makes
  a MEASURED PERIOD usable -- it breaks the gauge rather than adding a constraint inside it, so
  a period cost term is not the answer (5.12b).
- `fit/cost.quotient_basis` is not deterministic (hazard 18). The fix is a fixed orthonormalisation in place of the SVD of a degenerate spectrum; not made, because it changes the meaning of `v` for every stored result and that should be a deliberate commit.
- `fit/cost.make_growth_fn`'s default `eps=1e-4` sits BELOW the numerical floor of many fitted orbits and reports spurious repulsion (hazard 19). `w_stab` must not be enabled until it climbs an epsilon ladder the way `fit/stability.measure` does.
- **Nine outputs in `out/` were written by scripts that lived only in a session scratchpad and
  are now gone.** This is the provenance failure `paths.provenance` exists to catch, and the
  guard did catch it -- it printed the warning at write time and recorded
  `script_tracked: false` in every sidecar. Nobody acted on it. Audit:

      python -c "
      import json, glob
      for f in glob.glob('out/**/*.meta.json', recursive=True):
          m = json.load(open(f))
          if m.get('script_tracked') is False: print(m['script'], '->', f)"

  Most are uncited exploratory figures and can simply be deleted. **One is cited**:
  `out/almeida/fit_radial/RAD01/endtoend_perturbations.png`, which PROJECT_SUMMARY 5.7 and
  hazard 14 both lean on for "the dynamics were fine the whole time". Its script
  (`scratchpad/endtoend.py`) no longer exists, so that figure currently cannot be regenerated
  and the claim rests on a picture with no derivation behind it. Either re-derive it from a
  tracked entry point or stop citing it. Set `CLOCKZOO_STRICT_PROVENANCE=1` locally, not only
  on the cluster, and the next one fails loudly instead of leaving a note in a sidecar.
