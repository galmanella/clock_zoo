#!/bin/sh
# Figure 3 sensitivity chain, INSTANT mode, Almeida/BMAL1.
# Both stages checkpoint and resume, so an interrupt costs one unit of work, not the run.
# Re-running this script after a kill picks up where it stopped.
PY=/c/Users/galma/anaconda3/python.exe
set -e
echo "############ lc_sens (Tsit5, early-exit Newton) ############"; date
"$PY" -u -m analysis.lc_sens --model almeida --tag r01i
echo "############ ptc_sens BMAL1 instant, linear 0-100 x 49, 32 phase ############"; date
"$PY" -u -m analysis.ptc_sens --model almeida --target BMAL1 --mode instant \
      --n-phase 32 --doses 0,100,49 --scale linear --dt 0.02 --tag r01i
echo "############ coupling ############"; date
"$PY" -u -m analysis.coupling --model almeida --target BMAL1 --mode instant --lc-tag r01i --pt-tag r01i
echo "############ FIG3 SENS DONE ############"; date
