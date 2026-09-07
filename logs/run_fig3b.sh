#!/bin/sh
# ptc_sens + coupling only: lc_sens already completed under tag r01i.
PY=/c/Users/galma/anaconda3/python.exe
set -e
echo "############ ptc_sens BMAL1 instant, linear 0-100 x 49, 32 phase ############"; date
"$PY" -u -m analysis.ptc_sens --model almeida --target BMAL1 --mode instant \
      --n-phase 32 --doses 0,100,49 --scale linear --dt 0.02 --tag r01i
echo "############ coupling ############"; date
"$PY" -u -m analysis.coupling --model almeida --target BMAL1 --mode instant \
      --lc-tag r01i --pt-tag r01i
echo "############ FIG3 SENS DONE ############"; date
