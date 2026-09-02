#!/bin/sh
PY=/c/Users/galma/anaconda3/python.exe
for spec in "almeida BMAL1" "korencic Bmalx" "goldbeter MB"; do
  set -- $spec
  echo "############ characterize $1 $2 instant LINEAR 0..100 ############"
  date
  "$PY" -u -m analysis.characterize --model "$1" --mode instant --targets "$2" --shared \
        --n-phase 32 --skip-cap 48 --relax-tag zoo03 \
        --tag zoo04lin --scrit-tag zoo03rd || echo "FAILED $1 $2"
done
echo "############ ALL LINEAR INSTANT DONE ############"
date
