# `clock_zoo`

**Is a mechanistic circadian model more identifiable from phase-transition-curve (PTC) data
than from limit-cycle (LC) data alone?** Same question as `../input_screen/` (Mirsky 2009, 21
states, 132 parameters), asked of models small enough that the optimisation is tractable.

| doc | what it is |
|---|---|
| [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) | the scientific narrative, results, and canonical tables |
| [`REPO_MAP.md`](REPO_MAP.md) | the code index: which module is canonical for what |
| [`gauge/README.md`](gauge/README.md) | the exact unit-rescaling symmetry -- read before comparing any two parameter sets |

## Layout

```
models/    the zoo + the capability contract (models/api.py)
engine/    orbit solver (BVP), perturbations, PTC, and an independent adaptive reference
analysis/  the drivers that produce results
gauge/     the parameter-space unit symmetry, derived per model
```

## Running

Everything is a module, run from the repo root:

```bash
python -m models.api                 # capability conformance, whole registry
python -m gauge.gauge                # derived gauge generators per model
python -m engine.orbit               # BVP self-test: period, self-convergence, gauge covariance
python -m engine.perturb             # perturbation definitions
python -m engine.ptc                 # PTC self-test: identity at dose 0, readout consistency
python -m engine.reference           # THE cross-check: JAX/RK4/Fourier vs scipy/LSODA/peaks
```

Drivers shard for SLURM (`--shard i --nshards n`, or `SLURM_ARRAY_TASK_ID`) and write to
`out/<model>/<analysis>/<tag>/` with a provenance sidecar. Set
`CLOCKZOO_STRICT_PROVENANCE=1` on the cluster so an untracked entry script is a hard error.

## Provenance

The model-agnostic core (`engine/orbit.py`, `gauge/`, `paths.py`, and the PTC analysis
kernels) was copied and generalised from `input_screen/` at commit `e955873`. Each file names
its origin in its header. `input_screen/` is not modified by this repo.
