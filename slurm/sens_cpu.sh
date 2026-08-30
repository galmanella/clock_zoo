#!/bin/bash
#SBATCH --job-name=cz-sens
#SBATCH --output=slurm-%A_%a.out
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=24:00:00
# Usage: sbatch slurm/sens_cpu.sh <model> <target> [mode]
#   Array over PARAMETERS for analysis.ptc_sens. Run analysis.scrit first: this reads the
#   adaptive dose grid and the refined dt from its output.
set -euo pipefail
MODEL="${1:?usage: sbatch slurm/sens_cpu.sh <model> <target> [mode]}"
TARGET="${2:?target required}"
MODE="${3:-pulse}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CLOCKZOO_STRICT_PROVENANCE=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
PYTHON_EXE="${PYTHON_EXE:-/home/galmanel/miniconda3/envs/mirsky/bin/python}"
[ -x "$PYTHON_EXE" ] || { echo "ERROR: no python at $PYTHON_EXE"; exit 1; }

"$PYTHON_EXE" -m analysis.ptc_sens --model "$MODEL" --target "$TARGET" --mode "$MODE" \
       --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "$NSHARDS"

if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq "$((NSHARDS - 1))" ]; then
    sleep 30
    "$PYTHON_EXE" -m analysis.ptc_sens --model "$MODEL" --target "$TARGET" --mode "$MODE" --merge
    "$PYTHON_EXE" -m analysis.lc_sens  --model "$MODEL" --merge || true
    "$PYTHON_EXE" -m analysis.coupling --model "$MODEL" --target "$TARGET" --mode "$MODE"
fi
