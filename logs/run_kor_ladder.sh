#!/bin/sh
PY=/c/Users/galma/anaconda3/python.exe
for np in 64 128 256; do
  echo "############ korencic Bmalx instant  n_phase=$np ############"
  date
  "$PY" -u -m analysis.characterize --model korencic --mode instant --targets Bmalx --shared \
        --n-phase "$np" --skip-cap 48 --relax-tag zoo03 \
        --tag "korph$np" --scrit-tag zoo03rd || echo "FAILED n_phase=$np"
done
echo "############ LADDER DONE ############"
date
