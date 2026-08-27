# Repository map

One place to answer "which module is canonical for X?". Companion to
[`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) (the scientific narrative); this is the *code*
index.

Everything runs as a module from the repo root: `python -m engine.orbit`,
`python -m analysis.scrit --model almeida`.

## Layout

| dir | what lives there |
|---|---|
| `models/` | the zoo + the capability contract. A model declares what it is; downstream code asks the model instead of importing its private tables. |
| `engine/` | limit-cycle solver, perturbations, the PTC, and an INDEPENDENT adaptive-solver reference. Model-agnostic throughout. |
| `analysis/` | the drivers that produce results. Headless, shardable, provenance-stamped. |
| `gauge/` | the exact unit-rescaling symmetry, derived per model. **Read before comparing any two parameter sets.** |
| `slurm/` | cluster templates. Thin: the drivers already shard. |
| `out/` | results, `out/<model>/<analysis>/<tag>/`. Git-ignored; each npz carries a `.meta.json` provenance sidecar. |
| `docs/figures/` | figures cited by PROJECT_SUMMARY, copied out of `out/` so they travel with the document. |

## "What do I run for…?"

| Task | Run |
|---|---|
| Check a new model is wired up correctly | `python -m models.api <name>` then `python -m engine.validate <name>` |
| **The full gate** (nothing in `analysis/` is trusted until this passes) | `python -m engine.validate` |
| Where does each target change PTC type, and on what dose grid? | `python -m analysis.scrit --model M` |
| The base PTC surfaces and their features | `python -m analysis.characterize --model M` |
| How much does each parameter move the limit cycle? | `python -m analysis.lc_sens --model M` |
| How much does each parameter move the PTC? | `python -m analysis.ptc_sens --model M --target T` |
| **Are the LC and the PTC coupled?** (the batch-1 question) | `python -m analysis.coupling --model M --target T` |
| Is the gauge declaration right? | `python -m gauge.validate [identity\|algebra\|invariance]` |

Order matters: `scrit` derives the dose grid and the integrator step that `characterize` and
`ptc_sens` consume, and `coupling` is a pure read of `lc_sens` + `ptc_sens`.

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

### Root
| File | Role |
|---|---|
| `paths.py` | Output protocol `out/<model>/<analysis>/<tag>/`, array-job tag/shard conventions, and the **provenance guard**. |
| `parallel.py` | One dispatcher (`serial`/`joblib`/`ray`) plus the shard helper. `ray` is optional on purpose. |
| `plotting.py` | House conventions: cyclic `cmocean.cm.phase` for phase, achromatic overlays, grey for missing. |

## Hazards

1. **Never autodiff the fixed-step PTC at nontrivial dose.** In input_screen that produced
   `|J|` up to 3.95e83 against an actual response of 0.006 and invalidated a whole results
   table. Everything here is finite-difference; a gradient-based cost would need an adaptive
   (Tsit5/diffrax) backend.
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

## Open

- `analysis/characterize.py` and `ptc_sens.py` have not yet been run for Korencic or
  Goldbeter (Almeida only).
- `instant` mode is implemented and gated but not yet swept.
- No optimisation yet — that is batch 2 (see PROJECT_SUMMARY §5).
