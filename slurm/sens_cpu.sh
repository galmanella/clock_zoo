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
python -m analysis.ptc_sens --model "$MODEL" --target "$TARGET" --mode "$MODE" \
       --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "$NSHARDS"

if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq "$((NSHARDS - 1))" ]; then
    sleep 30
    python -m analysis.ptc_sens --model "$MODEL" --target "$TARGET" --mode "$MODE" --merge
    python -m analysis.lc_sens  --model "$MODEL" --merge || true
    python -m analysis.coupling --model "$MODEL" --target "$TARGET" --mode "$MODE"
fi
