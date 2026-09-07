#!/bin/bash
#SBATCH --job-name=cz-sens
#SBATCH --output=slurm-%A_%a.out
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#
# Usage: sbatch slurm/sens_cpu.sh <model> <target> [mode] [tag] [n_phase]
#
#   sbatch --array=0-15 slurm/sens_cpu.sh korencic Bmalx instant kor01 32
#
# ARRAY OVER PARAMETERS, for both analysis.lc_sens and analysis.ptc_sens. Run analysis.scrit
# FIRST: ptc_sens reads its adaptive dose grid and its refined dt from that output.
#
# WHY lc_sens IS IN HERE NOW. It used to be invoked only as `lc_sens --merge || true` on the
# last task -- merging shards that this script never created. That is a no-op wearing the
# costume of a dependency: it always "succeeded" (the `|| true`), and analysis.coupling, which
# is a pure read of lc_sens + ptc_sens, then either found a lc_sens from some earlier unrelated
# run or failed at the very end of a 16-task array. lc_sens is target-independent and cheap
# (Korencic: 34 parameters x 9 factors, ~15 min serial, ~1 min at 16 shards), so it is sharded
# on the SAME array and merged beside ptc_sens.
#
#   It is skipped when the merged output already exists for this tag, so running this script
#   for a second target does not recompute it. That check is on the FINAL file, not the
#   checkpoint, because a `.partial.npz` means an interrupted run that should resume.
#
# THE TAG IS EXPLICIT. Without it every submission lands in a directory named after its
# SLURM_ARRAY_JOB_ID, and `coupling` then has to guess which one -- so two arrays for two genes
# produce results that cannot be joined by name. Pass the same tag you used for
# analysis.scrit; it defaults to the array job id, which is the old behaviour.
set -euo pipefail
MODEL="${1:?usage: sbatch slurm/sens_cpu.sh <model> <target> [mode] [tag] [n_phase]}"
TARGET="${2:?target required}"
MODE="${3:-pulse}"
TAG="${4:-${SLURM_ARRAY_JOB_ID:-manual}}"
NPHASE="${5:-32}"

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
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
PYTHON_EXE="${PYTHON_EXE:-/home/galmanel/miniconda3/envs/mirsky/bin/python}"
[ -x "$PYTHON_EXE" ] || { echo "ERROR: no python at $PYTHON_EXE"; exit 1; }

echo "host=$(hostname) start=$(date) model=$MODEL target=$TARGET mode=$MODE tag=$TAG"
echo "shard=$SHARD/$NSHARDS n_phase=$NPHASE python=$PYTHON_EXE"

# S_crit must be a PROMOTED FIXTURE, not something sitting in out/ (hazard 16). ptc_sens still
# reads out/ for its dose grid, but the fit does not, and a missing fixture here means the tag
# was never promoted -- better to say so now than after 16 tasks.
if [ ! -d "$CODE_DIR/fixtures/scrit/$MODEL" ]; then
    echo "WARNING: fixtures/scrit/$MODEL is absent. analysis.ptc_sens will still run (it reads"
    echo "         out/), but nothing in fit/ will. Promote it:"
    echo "           \$PY -m fit.doses --promote --model $MODEL --mode $MODE --tag $TAG"
fi

LC_DONE="out/$MODEL/lc_sens/$TAG/lc_sens_merged.npz"
if [ -f "$LC_DONE" ]; then
    echo "[sens] lc_sens already merged for tag $TAG -- skipping (it is target-independent)"
else
    "$PYTHON_EXE" -u -m analysis.lc_sens --model "$MODEL" --tag "$TAG" \
           --shard "$SHARD" --nshards "$NSHARDS"
fi

"$PYTHON_EXE" -u -m analysis.ptc_sens --model "$MODEL" --target "$TARGET" --mode "$MODE" \
       --n-phase "$NPHASE" --tag "$TAG" --scrit-tag "$TAG" \
       --shard "$SHARD" --nshards "$NSHARDS"

# THE LAST TASK MERGES AND READS. `sleep 30` is not superstition: the other tasks' final
# np.savez_compressed calls can still be flushing on a shared filesystem, and a merge that
# races them silently joins fewer shards than it should.
if [ "$SHARD" -eq "$((NSHARDS - 1))" ]; then
    sleep 30
    [ -f "$LC_DONE" ] || "$PYTHON_EXE" -u -m analysis.lc_sens --model "$MODEL" --tag "$TAG" --merge
    "$PYTHON_EXE" -u -m analysis.ptc_sens --model "$MODEL" --target "$TARGET" --mode "$MODE" \
           --tag "$TAG" --merge
    # coupling is a PURE READ of the two above, and both tags are passed explicitly so it
    # cannot silently pick up an older run.
    #
    # --no-include-time BELONGS ON PULSE and not on instant: a pulse of duration fixed in HOURS
    # is itself a clock, so pulse data can see the time rescale and quotienting it out projects
    # away a direction the experiment determines (FIT_VALIDITY 3c).
    EXTRA=""
    [ "$MODE" = "pulse" ] && EXTRA="--no-include-time"
    "$PYTHON_EXE" -u -m analysis.coupling --model "$MODEL" --target "$TARGET" --mode "$MODE" \
           --lc-tag "$TAG" --pt-tag "$TAG" --tag "$TAG" $EXTRA

    cat <<EOF

  NEXT, and it is NOT optional -- a jacobian ratio overstates decoupling (PROJECT_SUMMARY 3.6
  predicted ~140x against an actual 11x). Confirm by finite displacement on the INDEPENDENT
  adaptive engine before quoting any rho:

      \$PY -m analysis.confirm --model $MODEL --target $TARGET --mode $MODE
      \$PY -m analysis.confirm --model $MODEL --target $TARGET --mode $MODE --all-dirs

  The second asks the stronger question -- does the measured dPTC/dLC ORDER with rho, or is
  the top direction a lucky draw -- and costs ~2*(n_free+1) adaptive runs.
EOF
fi
echo "done=$(date)"
