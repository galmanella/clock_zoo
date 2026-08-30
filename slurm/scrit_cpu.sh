#!/bin/bash
#SBATCH --job-name=cz-scrit
#SBATCH --output=slurm-%A_%a.out
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
# Usage: sbatch slurm/scrit_cpu.sh <model> [mode]
#   Array over TARGETS. Size --array to the target count (models/<model>.py
#   perturbable_targets()); a task with no work in its shard exits cleanly.
set -euo pipefail
MODEL="${1:?usage: sbatch slurm/scrit_cpu.sh <model> [mode]}"
MODE="${2:-pulse}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CLOCKZOO_STRICT_PROVENANCE=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
PYTHON_EXE="${PYTHON_EXE:-/home/galmanel/miniconda3/envs/mirsky/bin/python}"
[ -x "$PYTHON_EXE" ] || { echo "ERROR: no python at $PYTHON_EXE"; exit 1; }

"$PYTHON_EXE" -m analysis.scrit --model "$MODEL" --mode "$MODE" \
       --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "$NSHARDS"

# the last task merges; all tasks share SLURM_ARRAY_JOB_ID, hence one output dir
if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq "$((NSHARDS - 1))" ]; then
    sleep 30
    "$PYTHON_EXE" -m analysis.scrit --model "$MODEL" --mode "$MODE" --merge
fi
