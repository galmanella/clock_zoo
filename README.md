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

### CALL THE INTERPRETER BY ITS FULL PATH. `python` IS NOT ENOUGH.

Every example below writes `$PY`. Set it once per shell:

```bash
# cluster
export PY=/home/galmanel/miniconda3/envs/mirsky/bin/python
# local (Windows / Git Bash)
export PY=/c/Users/galma/anaconda3/python.exe
```

A bare `python` resolves to whatever is first on `PATH`, which on a login node is a system
interpreter without jax, diffrax or cma, and inside a SLURM job may be nothing at all. The
failure is not always loud: a partially-satisfied environment imports and then produces
different numbers. **The SLURM scripts already do this** -- each pins
`PYTHON_EXE="${PYTHON_EXE:-/home/galmanel/miniconda3/envs/mirsky/bin/python}"`, overridable at
submit time -- and interactive use should match them, or a dry-run and its own array job are
not running the same code.

### AND RUN FROM THE REPO ROOT. `python -m` PUTS THE CURRENT DIRECTORY ON `sys.path`.

```
$ $PY -m fit.campaign --config campaigns/arm0_ctl.json --dry-run
Error while finding module specification for 'fit.campaign'
    (ModuleNotFoundError: No module named 'fit')
```

That is not a broken install. `-m` prepends the CURRENT WORKING DIRECTORY to `sys.path`, so
`fit` is importable only from the directory that contains it. `cd` to the repo root, or pass
the path explicitly:

```bash
PYTHONPATH=/path/to/clock_zoo $PY -m fit.campaign --config <cfg> --dry-run
```

**`sbatch` is immune** -- `slurm/*.sh` cd to `CODE_DIR` and export `PYTHONPATH` themselves, so
only the interactive dry-run trips on this. Which is the trap: the dry-run that is supposed to
validate an array can fail for a reason the array never would.

Everything is a module, run from the repo root:

```bash
$PY -m models.api                 # capability conformance, whole registry
$PY -m gauge.gauge                # derived gauge generators per model
$PY -m engine.orbit               # BVP self-test: period, self-convergence, gauge covariance
$PY -m engine.perturb             # perturbation definitions
$PY -m engine.ptc                 # PTC self-test: identity at dose 0, readout consistency
$PY -m engine.reference           # THE cross-check: JAX/RK4/Fourier vs scipy/LSODA/peaks
```

Drivers shard for SLURM (`--shard i --nshards n`, or `SLURM_ARRAY_TASK_ID`) and write to
`out/<model>/<analysis>/<tag>/` with a provenance sidecar. Set
`CLOCKZOO_STRICT_PROVENANCE=1` on the cluster so an untracked entry script is a hard error.

## Provenance

The model-agnostic core (`engine/orbit.py`, `gauge/`, `paths.py`, and the PTC analysis
kernels) was copied and generalised from `input_screen/` at commit `e955873`. Each file names
its origin in its header. `input_screen/` is not modified by this repo.
