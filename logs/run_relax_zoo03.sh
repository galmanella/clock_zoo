#!/bin/sh
PY=/c/Users/galma/anaconda3/python.exe
for m in almeida korencic goldbeter; do
  for md in pulse instant; do
    echo "############ relax $m $md ############"
    date
    "$PY" -u -m analysis.relax --model "$m" --mode "$md" --scrit-tag zoo03rd --tag zoo03 \
      || echo "FAILED $m $md"
  done
done
echo "############ ALL RELAX DONE ############"
date
