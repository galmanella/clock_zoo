#!/bin/bash
# ============================================================================
# KORENCIC objective (b), the CLUSTER half. Copy-paste, or `bash` it, from the
# repo root on the cluster AFTER `git pull`.
#
# Bmalx and lc_sens were run locally (tag kor01) -- but `out/` is gitignored and does
# NOT travel through git, so the cluster recomputes both. That is deliberate: it is
# also the reproduction check, and lc_sens is ~3 min at 16 shards.
#
# SCOPE, decided from analysis.identifiability (PROJECT_SUMMARY 5c.5): all four
# resetting probes, INSTANT ONLY. Dbpx is excluded because it does not reset.
# ============================================================================
set -euo pipefail
export PY=/home/galmanel/miniconda3/envs/mirsky/bin/python
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # repo root
TAG=kor01

# ---------------------------------------------------------------- 0. PREFLIGHT
# Cheap, and each of these has cost this project a multi-hour run at least once.
$PY -m models.api korencic                       # capability contract
$PY -m engine.validate korencic                  # THE GATE -- expect PASS, 5.35e-04 cyc
test -f fixtures/scrit/korencic/scrit_instant.npz \
  || { echo "FATAL: S_crit fixture missing -- git pull, or promote it"; exit 1; }

# scrit is NOT re-run: S_crit is tracked in fixtures/ and phi* is grid-quantised, so the
# per-target dose grid ptc_sens needs must be regenerated ONCE into out/ on this machine.
# ~4 min for five targets.
$PY -u -m analysis.scrit --model korencic --mode instant \
      --targets Bmalx,Perx,Cryx,Reverbx,Dbpx --tag $TAG

# ------------------------------------------------------- 1. THE SENSITIVITY ARRAYS
# One array per gene. Each shards lc_sens AND ptc_sens over the 34 parameters, merges
# both on the last task, and runs analysis.coupling as a pure read.
#
# lc_sens is computed by the FIRST array and skipped by the rest (it is
# target-independent), so submit Bmalx first and let it finish before the others --
# otherwise three arrays race to write one lc_sens.
#
# MEASURED LOCALLY, to size this: lc_sens 7.3 min serial (306 orbit solves at 0.6 s);
# ptc_sens ~45 min serial per gene. At 16 shards expect a few minutes each.
sbatch --array=0-15 --wait slurm/sens_cpu.sh korencic Bmalx   instant $TAG 32
for g in Perx Reverbx Cryx; do
  sbatch --array=0-15 slurm/sens_cpu.sh korencic "$g" instant $TAG 32
done

# ------------------------------------------------------------- 2. AFTER THE ARRAYS
# CONFIRM IS NOT OPTIONAL. A jacobian ratio OVERSTATES decoupling: PROJECT_SUMMARY 3.6
# predicted ~140x from the linear analysis against a measured 11x. The number to quote is
# the finite-displacement one, taken on engine/reference.py -- scipy LSODA + peak
# matching, which shares no numerical machinery with the engine the jacobian came from.
#
#   for g in Bmalx Perx Reverbx Cryx; do
#     $PY -u -m analysis.confirm --model korencic --target $g --mode instant
#   done
#
# ... and, on the best probe only, the STRONGER question: does the measured dPTC/dLC
# actually ORDER with rho, or is the top direction a lucky draw? ~2*(33+1) adaptive runs.
#
#   $PY -u -m analysis.confirm --model korencic --target Bmalx --mode instant --all-dirs
#   $PY -u -m analysis.sweep   --model korencic --target Bmalx --mode instant --tag $TAG
#
# ------------------------------------------------------------------- 3. SYNC BACK
# out/ is gitignored -- results do NOT travel through git.
#
#   rsync -av <cluster>:<repo>/out/korencic/ ./out/korencic/
#
# READ BEFORE QUOTING ANYTHING:
#   * gate every surface first -- a topological feature extractor ALWAYS returns a number
#     (hazard 12). `$PY -m analysis.quality --model korencic --mode instant --tag $TAG`
#   * quote acc_twist, never a total_twist sitting near 0.5 -- it SATURATES there (5.9a)
#   * the null directions of J_LC must be exactly null in J_PTC; that is coupling's own
#     self-check that the gauge projection is not leaking a false signal
echo "submitted. squeue -u $USER"
