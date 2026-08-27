#!/bin/bash
#SBATCH --job-name=cz-ptc-gpu
#SBATCH --output=slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# Usage: sbatch slurm/ptc_gpu.sh <model> <target> [mode]
#   The PTC over (phase x dose) is one vmapped kernel, so a whole grid is a single GPU launch
#   and sharding is unnecessary -- one task does all parameters.
set -euo pipefail
MODEL="${1:?usage: sbatch slurm/ptc_gpu.sh <model> <target> [mode]}"
TARGET="${2:?target required}"
MODE="${3:-pulse}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CLOCKZOO_STRICT_PROVENANCE=1
export JAX_ENABLE_X64=1                      # long-horizon circadian integration needs f64
unset JAX_PLATFORMS                          # let JAX pick the GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # play nicely with other jobs on the card

python -c "import jax; print('[gpu] devices:', jax.devices())"
python -m analysis.ptc_sens --model "$MODEL" --target "$TARGET" --mode "$MODE"
python -m analysis.lc_sens  --model "$MODEL"
python -m analysis.coupling --model "$MODEL" --target "$TARGET" --mode "$MODE"
