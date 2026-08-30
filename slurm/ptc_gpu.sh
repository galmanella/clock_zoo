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
export CLOCKZOO_STRICT_PROVENANCE=1
export JAX_ENABLE_X64=1                      # long-horizon circadian integration needs f64
unset JAX_PLATFORMS                          # let JAX pick the GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # play nicely with other jobs on the card
PYTHON_EXE="${PYTHON_EXE:-/home/galmanel/miniconda3/envs/mirsky/bin/python}"
[ -x "$PYTHON_EXE" ] || { echo "ERROR: no python at $PYTHON_EXE"; exit 1; }


"$PYTHON_EXE" -c "import jax; print('[gpu] devices:', jax.devices())"
"$PYTHON_EXE" -m analysis.ptc_sens --model "$MODEL" --target "$TARGET" --mode "$MODE"
"$PYTHON_EXE" -m analysis.lc_sens  --model "$MODEL"
"$PYTHON_EXE" -m analysis.coupling --model "$MODEL" --target "$TARGET" --mode "$MODE"
