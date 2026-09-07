#!/bin/sh
# Stages 2-3: shared dose grid from the screens, then the refined PTC surfaces.
#
#   sh logs/run_zoo04_surfaces.sh > logs/zoo04_surfaces.log 2>&1 &
#
# Run only after run_zoo04_scrit.sh has finished: the grid is built FROM the screens.
PY=/c/Users/galma/anaconda3/python.exe
TAG=${1:-zoo04}
NPH=${2:-32}
# goldbeter_rev FIRST: it is the new model and the one whose surfaces have never been seen,
# so it is the one worth looking at before the rest of the queue drains.
MODELS="goldbeter_rev almeida korencic goldbeter"
for m in $MODELS; do
  for md in instant pulse; do
    echo "############ dosegrid $m $md ############"; date
    "$PY" -u -m analysis.dosegrid --model "$m" --mode "$md" --tag "$TAG" \
      || echo "FAILED dosegrid $m $md"
  done
done
for m in $MODELS; do
  for md in instant pulse; do
    echo "############ characterize $m $md n_phase=$NPH ############"; date
    "$PY" -u -m analysis.characterize --model "$m" --mode "$md" --shared \
          --n-phase "$NPH" --tag "$TAG" --scrit-tag "$TAG" \
      || echo "FAILED characterize $m $md"
  done
done
echo "############ ALL SURFACES DONE ############"; date
