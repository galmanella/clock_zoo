#!/bin/sh
# Korencic, re-screened on the NEW PHASE ORIGIN.
#
#   sh logs/run_kor01.sh > logs/kor01_surfaces.log 2>&1 &
#
# WHY THIS RE-RUN EXISTS. models/korencic.py now declares readout_variable = 'Dbpx' separately
# from reference_variable = 'Reverbx' (hazard 14). The readout carries the PHASE, so the origin
# of every surface moved. S_crit is origin-invariant and would survive, but phi* is not, and a
# tag holding some quantities on one origin and some on another is exactly the silent
# inconsistency this repo keeps paying for. Cheaper to re-derive the whole tag: Korencic is
# ~127 s of scrit and ~97 s of characterize per mode.
#
# Same three stages, same flags, as logs/run_zoo04_scrit.sh + run_zoo04_surfaces.sh -- this is
# not a new procedure, it is the established one on one model with one attribute changed.
#
# NEVER pipe this through `tail`: tail cannot emit until EOF, so a working job looks hung.
PY=${PY:-/c/Users/galma/anaconda3/python.exe}
TAG=${1:-kor01}
NPH=${2:-32}
for md in instant pulse; do
  echo "############ scrit korencic $md ############"; date
  "$PY" -u -m analysis.scrit --model korencic --mode "$md" \
        --targets Bmalx,Perx,Cryx,Reverbx,Dbpx --tag "$TAG" || echo "FAILED scrit $md"
done
for md in instant pulse; do
  echo "############ dosegrid korencic $md ############"; date
  "$PY" -u -m analysis.dosegrid --model korencic --mode "$md" --tag "$TAG" \
    || echo "FAILED dosegrid $md"
done
for md in instant pulse; do
  echo "############ characterize korencic $md n_phase=$NPH ############"; date
  "$PY" -u -m analysis.characterize --model korencic --mode "$md" --shared \
        --n-phase "$NPH" --tag "$TAG" --scrit-tag "$TAG" || echo "FAILED characterize $md"
done
for md in instant pulse; do
  echo "############ quality korencic $md ############"; date
  "$PY" -u -m analysis.quality --model korencic --mode "$md" --tag "$TAG" \
    || echo "FAILED quality $md"
done
echo "############ KOR01 DONE ############"; date
