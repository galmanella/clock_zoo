#!/bin/bash
#SBATCH --job-name=cz-fit
#SBATCH --output=slurm-%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
# Usage: sbatch slurm/fit_cpu.sh <model> <target> [recover|radial] [eps] [optimizer]
#
#   e.g. sbatch slurm/fit_cpu.sh almeida BMAL1 radial 0.3 bobyqa
#
#   PHASE IS READ FROM REV AND THE POINCARE SECTION IS PER (see models/almeida.py). <target>
#   is the PERTURBATION target, a different role -- passing BMAL1 there is still correct.
#
#   One ARRAY TASK PER START. Multistart is the globality instrument here (see fit/search.py):
#   the question is whether distinct basins exist, which wants many INDEPENDENT converged
#   solutions to cluster, and those are embarrassingly parallel. SLURM_ARRAY_TASK_ID becomes
#   the RNG seed, and every task writes into the same SLURM_ARRAY_JOB_ID run directory, so one
#   array job produces one output dir -- the convention paths.py already implements.
#
#   ONE CPU PER TASK, AND THAT IS NOT AN OVERSIGHT. MEASURED, one cost evaluation on the
#   20 x 14 grid (280 cells):
#
#       threads=1   1.298 s          threads=4   1.552 s
#       threads=2   1.482 s          threads=8   1.539 s
#
#   More threads make it SLOWER. The 280 cells are logically independent, but each is an
#   adaptive Tsit5 integration -- sequential by nature -- and XLA's CPU backend does not
#   parallelise the vmap across them, so extra threads only add contention. The previous
#   --cpus-per-task=4 reserved three idle cores per task; those cores belong in the ARRAY
#   WIDTH, where scaling is linear, which is why the array doubled to 0-15 as the cpus fell.
#
#   The serial fraction inside one evaluation is small -- the orbit BVP solve is 114 ms of
#   1.505 s, 7.5%, so Amdahl caps a perfect implementation near 13x -- but that headroom is
#   reachable only on a GPU, and only in f64, which runs at 1/32 rate on consumer cards. Until
#   `search.cma` evaluates its population in ONE batched call instead of a Python loop, no
#   launch here is big enough to be worth a card.
#
#   So the cluster buys THROUGHPUT, not latency: one fit takes as long as it does locally, and
#   the array is what makes sixteen of them cost the wall time of one.
#
#   DEPENDENCY: this needs `diffrax` in the cluster env (pip install diffrax). The fit runs on
#   the adaptive backend -- see REPO_MAP hazard 1 and fit/doses.py for why the fixed-step path
#   is not an option for a gradient.
set -euo pipefail
MODEL="${1:?usage: sbatch slurm/fit_cpu.sh <model> <target> [recover|radial] [eps]}"
TARGET="${2:?target required}"
WHAT="${3:-recover}"
EPS="${4:-0.3}"
OPT="${5:-bobyqa}"     # measured best on the T1 benchmark: 0.0104 against cma-anneal
                       # 0.0397, cma-ipop 0.0674 and lm 0.436 at a matched 1500-eval budget

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
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

# The interpreter by ABSOLUTE PATH, not whatever PATH resolves to on a compute node.
PYTHON_EXE="${PYTHON_EXE:-/home/galmanel/miniconda3/envs/mirsky/bin/python}"
[ -x "$PYTHON_EXE" ] || { echo "ERROR: no python at $PYTHON_EXE"; exit 1; }

SEED="${SLURM_ARRAY_TASK_ID:-0}"
TAG="${SLURM_ARRAY_JOB_ID:-local}"

# Fail loudly and early if the backend is missing, rather than after the queue wait.
"$PYTHON_EXE" -c "import diffrax, cma; print('[fit] diffrax', diffrax.__version__, 'cma', cma.__version__)"

# Gate every gradient run on the gradient actually being real. This is cheap next to the fit
# and it is the check whose absence let hazard 1 invalidate a table in input_screen.
"$PYTHON_EXE" -m fit.cost --gradcheck --model "$MODEL" --target "$TARGET" --backend diffrax

case "$WHAT" in
  recover) "$PYTHON_EXE" -m fit.recover --model "$MODEL" --target "$TARGET" --eps "$EPS" \
                                 --seed "$SEED" --tag "$TAG" ;;
  radial)  "$PYTHON_EXE" -m fit.radial  --model "$MODEL" --target "$TARGET" \
                                 --optimizer "$OPT" --n-phase 20 --n-dose 14 \
                                 --max-factor 8.0 --seed "$SEED" --tag "$TAG" ;;
  *) echo "unknown mode '$WHAT' (want recover|radial)" >&2; exit 2 ;;
esac
