#!/bin/bash
#SBATCH --job-name=cz-campaign
#SBATCH --output=slurm-%A_%a.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# Usage:
#   python -m fit.campaign --config campaigns/genes.json --dry-run     # ALWAYS do this first
#   sbatch --array=0-3 slurm/campaign_cpu.sh campaigns/genes.json
#
#   ONE ARRAY TASK PER CONFIGURATION. --dry-run prints the exact --array range to use, and
#   validates every entry before a queue slot is spent; a 24-task array that dies at task 0 on
#   a typo has cost an hour of queue for nothing.
#
# CPUS-PER-TASK IS THE CMA POPULATION WIDTH, AND IT IS NOT FREE
#   Each task parallelises its CMA population across $SLURM_CPUS_PER_TASK processes
#   (fit/parallel.py). MEASURE IT ON THIS NODE BEFORE TRUSTING A NUMBER: on the development
#   laptop -- an Intel Core Ultra 7 268V, 4 P-cores + 4 E-cores sharing 12 MB L3 -- each
#   evaluation ran 5.7x SLOWER under 8-way concurrency, so 8 workers bought only ~2.4x. That is
#   a property of a hybrid laptop part, not of the code, and a homogeneous node should do
#   better. It should NOT be assumed to.
#
#   The population is oversubscribed to the worker count by default (~4.1x, see
#   recommend_popsize) so that fast members fill the gaps behind a slow one; evaluation times
#   span 0.76 s to 15 s. Set `popsize` explicitly in the config to override.
#
# WHAT EACH TASK WRITES
#   out/<model>/fit_radial/<campaign>/<what-varies>/seed<N>/ ... one directory per seed, plus a
#   seeds_*.npz comparing them. The resolved RunConfig is stored inside every .npz as cfg_json,
#   so a result carries its own definition and never has to be reconstructed from a filename.
set -euo pipefail
CONFIG="${1:?usage: sbatch --array=0-N slurm/campaign_cpu.sh <campaign.json>}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CLOCKZOO_STRICT_PROVENANCE=1
export JAX_ENABLE_X64=1                      # long-horizon circadian integration needs f64
export JAX_PLATFORMS=cpu

# Fail before the science if a dependency is missing, not three hours in.
python -c "import diffrax, cma, pybobyqa; print('[campaign] backends present')"

python -m fit.campaign --config "$CONFIG" \
       --index "${SLURM_ARRAY_TASK_ID:-0}" \
       --workers "${SLURM_CPUS_PER_TASK:-1}"
