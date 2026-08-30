#!/bin/bash
#SBATCH --job-name=cz-campaign
#SBATCH --output=/central/home/galmanel/slurmout/clock_zoo/campaign_%A_%a.out
#SBATCH --error=/central/home/galmanel/slurmout/clock_zoo/campaign_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# Usage:
#   mkdir -p /central/home/galmanel/slurmout/clock_zoo      # ONCE -- sbatch will not create it
#
#   cd <repo root>                                          # NOT slurm/, NOT the parent.
#   $PYTHON_EXE -m fit.campaign --config campaigns/genes.json --dry-run   # ALWAYS first
#   sbatch --array=0-3 slurm/campaign_cpu.sh campaigns/genes.json
#
#   `python -m` puts the CURRENT DIRECTORY on sys.path, so the dry-run finds `fit` only from
#   the repo root -- from anywhere else it is `ModuleNotFoundError: No module named 'fit'`, and
#   no PYTHONPATH-free invocation fixes it. `sbatch` is immune: this script cds to its own
#   parent and exports PYTHONPATH, so it can be submitted from anywhere.
#
#   ONE ARRAY TASK PER CONFIGURATION. --dry-run prints the exact --array range and validates
#   every entry before a queue slot is spent; a 24-task array that dies at task 0 on a typo has
#   cost an hour of queue for nothing.
#
# KEEP --cpus-per-task AND `popsize` MATCHED
#   Each task parallelises its CMA population across $SLURM_CPUS_PER_TASK processes
#   (fit/parallel.py). The campaign files set `popsize` EXPLICITLY to the core count, so one
#   member runs per core. If you change --cpus-per-task, change popsize with it.
#
#   Do not be tempted by a much larger population: `recommend_popsize` oversubscribes ~4.1x to
#   hide the straggler, which minimises wall-clock PER GENERATION but starves the search at a
#   fixed budget. CMA converges in GENERATIONS -- 8000 evaluations is 250 generations at
#   popsize 32 and only 61 at popsize 131, against the ~334 RAD03 needed at popsize 12.
#
# MEASURE THE NODE BEFORE TRUSTING A CORE COUNT
#   Every parallel number in this repo came from a hybrid-core laptop (Core Ultra 7 268V,
#   4 P-cores + 4 E-cores, 12 MB shared L3) where each evaluation ran 5.7x SLOWER under 8-way
#   concurrency, so 8 workers bought only ~2.4x. That is a property of that part, not of the
#   code. A homogeneous node should do better -- it should not be ASSUMED to.
#
# WHAT EACH TASK WRITES
#   out/<model>/fit_radial/<campaign>__<what-varies>__seed<N>/ ... one directory per seed, plus
#   seeds_*.npz comparing the seeds that share a task and viability_*.npz recording the
#   rejection sampling. The resolved RunConfig is stored inside every .npz as cfg_json, so a
#   result carries its own definition and never has to be reconstructed from a filename.
#
#   `out/` is gitignored: results do NOT travel back through git. rsync them.
set -uo pipefail
CONFIG="${1:?usage: sbatch --array=0-N slurm/campaign_cpu.sh <campaign.json>}"

# LOCATE THE CODE. A SCRIPT-RELATIVE cd DOES NOT WORK UNDER SBATCH.
#
# SLURM copies the batch script into a spool directory and runs it from there, so inside the
# job $0 is something like /var/spool/slurmd/job12345/slurm_script, and a script-relative cd
# lands in the spool dir where `fit` does not exist. It fails identically on EVERY array task
# while working perfectly by hand -- which is exactly how it presented. Mirsky's scripts pin
# CODE_DIR absolutely for this reason.
#
# Order: explicit CODE_DIR, else the directory sbatch was invoked FROM, else script-relative
# (correct when the script is executed directly rather than through SLURM).
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$_here}}"
cd "$CODE_DIR" || { echo "ERROR: cannot cd to CODE_DIR=$CODE_DIR"; exit 1; }
if [ ! -d "$CODE_DIR/fit" ]; then
    echo "ERROR: CODE_DIR=$CODE_DIR is not the clock_zoo repo root (no fit/ there)."
    echo "       Submit from the repo root, or pass CODE_DIR=/path/to/clock_zoo"
    exit 1
fi
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

# ENVIRONMENT: the same interpreter Mirsky uses, by ABSOLUTE PATH.
#
# `python` on a compute node is whatever the login shell happened to put first on PATH, which
# need not be the environment this code was tested against. A job that silently runs against
# the wrong interpreter fails hours later -- or worse, succeeds with different numerics.
# Override with PYTHON_EXE=... sbatch ... if the env moves.
PYTHON_EXE="${PYTHON_EXE:-/home/galmanel/miniconda3/envs/mirsky/bin/python}"
[ -x "$PYTHON_EXE" ] || { echo "ERROR: no python at $PYTHON_EXE"; exit 1; }

# ONE THREAD PER PROCESS. The population is parallelised across PROCESSES, so letting each also
# spawn a full thread pool only makes them contend. MEASURED: a single evaluation is no faster
# at 4 threads than at 1 (1.552 s vs 1.298 s -- it is slower).
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CLOCKZOO_STRICT_PROVENANCE=1

echo "host=$(hostname)  start=$(date)  config=$CONFIG  task=${SLURM_ARRAY_TASK_ID:-0}"
echo "python=$PYTHON_EXE  cores=${SLURM_CPUS_PER_TASK:-1}"

# Fail before the science if a dependency is missing, not three hours in. Required vs optional
# is deliberate: pybobyqa/matplotlib/cmocean matter only for non-CMA optimizers and figures,
# and a campaign of CMA fits should not be blocked by their absence.
"$PYTHON_EXE" - <<'PY'
import importlib.util as u, sys
req = ['jax', 'numpy', 'scipy', 'diffrax', 'cma']
opt = ['pybobyqa', 'matplotlib', 'cmocean']
miss = [m for m in req if u.find_spec(m) is None]
if miss:
    sys.exit(f"ERROR: missing REQUIRED package(s): {miss}  "
             f"-> pip install {' '.join(miss)}")
lack = [m for m in opt if u.find_spec(m) is None]
print("[campaign] required backends present"
      + (f"; optional missing (figures/non-CMA only): {lack}" if lack else ""))
PY
rc=$?
[ $rc -eq 0 ] || exit $rc

"$PYTHON_EXE" -u -m fit.campaign --config "$CONFIG" --index "${SLURM_ARRAY_TASK_ID:-0}" --workers "${SLURM_CPUS_PER_TASK:-1}"
echo "done=$(date)"
