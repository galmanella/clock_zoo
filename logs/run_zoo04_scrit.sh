#!/bin/sh
# Stage 1 of the Tsit5 re-run: S_crit screens, all four models, both modes.
#
#   sh logs/run_zoo04_scrit.sh > logs/zoo04_scrit.log 2>&1 &
#
# IN-SCOPE TARGETS ONLY. scrit defaults to every perturbable target; the out-of-scope ones
# (proteins, complexes) feed no figure and are not in the declared mRNA-level comparison, and
# dropping them roughly halves the screen.
#
# NEVER pipe this through `tail` -- tail cannot emit until EOF, so a working job looks hung.
# Redirect to a file and read the file.
PY=/c/Users/galma/anaconda3/python.exe
TAG=${1:-zoo04}
run() {   # model  targets
  for md in instant pulse; do
    echo "############ scrit $1 $md ############"; date
    "$PY" -u -m analysis.scrit --model "$1" --mode "$md" --targets "$2" --tag "$TAG" \
      || echo "FAILED scrit $1 $md"
  done
}
run almeida       BMAL1,PER,CRY,REV,DBP,ROR,E4BP4
run korencic      Bmalx,Perx,Cryx,Reverbx,Dbpx
run goldbeter     MB,MP,MC
run goldbeter_rev MB,MP,MC,MR
echo "############ ALL SCRIT DONE ############"; date
