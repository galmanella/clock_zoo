#!/bin/sh
# Refined PTC surfaces on the SHARED dose grid, all three models, both modes.
#   sh logs/run_characterize_zoo01.sh <n_phase> <tag>
# The dose grid must already exist:  $PY -m analysis.dosegrid --model M --mode MD --tag TAG
PY=/c/Users/galma/anaconda3/python.exe
NPH=${1:-64}
TAG=${2:-zoo01}
for m in almeida korencic goldbeter; do
  for md in pulse instant; do
    echo "############ characterize $m $md n_phase=$NPH ############"
    date
    "$PY" -u -m analysis.characterize --model "$m" --mode "$md" --shared \
          --n-phase "$NPH" --tag "$TAG" --scrit-tag "$TAG" || echo "FAILED $m $md"
  done
done
echo "############ ALL CHARACTERIZE DONE ############"
date
