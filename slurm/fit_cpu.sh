#!/bin/bash
#SBATCH --job-name=cz-fit
#SBATCH --output=slurm-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
# Usage: sbatch slurm/fit_cpu.sh <model> <target> [recover|radial] [eps]
#
#   One ARRAY TASK PER START. Multistart is the globality instrument here (see fit/search.py):
#   the question is whether distinct basins exist, which wants many INDEPENDENT converged
#   solutions to cluster, and those are embarrassingly parallel. SLURM_ARRAY_TASK_ID becomes
#   the RNG seed, and every task writes into the same SLURM_ARRAY_JOB_ID run directory, so one
#   array job produces one output dir -- the convention paths.py already implements.
#
#   CPU, not GPU. A fit is a long chain of sequential L-BFGS steps over a modest grid
#   (~100 cells), not one big vmapped launch, so it is latency-bound rather than
#   throughput-bound. The GPU script is for the sensitivity sweeps, which are the opposite.
#
#   DEPENDENCY: this needs `diffrax` in the cluster env (pip install diffrax). The fit runs on
#   the adaptive backend -- see REPO_MAP hazard 1 and fit/doses.py for why the fixed-step path
#   is not an option for a gradient.
set -euo pipefail
MODEL="${1:?usage: sbatch slurm/fit_cpu.sh <model> <target> [recover|radial] [eps]}"
TARGET="${2:?target required}"
WHAT="${3:-recover}"
EPS="${4:-0.3}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CLOCKZOO_STRICT_PROVENANCE=1
export JAX_ENABLE_X64=1                      # long-horizon circadian integration needs f64
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
TAG="${SLURM_ARRAY_JOB_ID:-local}"

# Fail loudly and early if the backend is missing, rather than after the queue wait.
python -c "import diffrax, cma; print('[fit] diffrax', diffrax.__version__, 'cma', cma.__version__)"

# Gate every gradient run on the gradient actually being real. This is cheap next to the fit
# and it is the check whose absence let hazard 1 invalidate a table in input_screen.
python -m fit.cost --gradcheck --model "$MODEL" --target "$TARGET" --backend diffrax

case "$WHAT" in
  recover) python -m fit.recover --model "$MODEL" --target "$TARGET" --eps "$EPS" \
                                 --seed "$SEED" --tag "$TAG" ;;
  radial)  python -m fit.radial  --model "$MODEL" --target "$TARGET" \
                                 --seed "$SEED" --tag "$TAG" ;;
  *) echo "unknown mode '$WHAT' (want recover|radial)" >&2; exit 2 ;;
esac
