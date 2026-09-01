#!/bin/sh
PY=/c/Users/galma/anaconda3/python.exe
for m in almeida korencic goldbeter; do
  for md in pulse instant; do
    echo "############ scrit $m $md ############"
    date
    "$PY" -u -m analysis.scrit --model "$m" --mode "$md" --tag zoo01 || echo "FAILED $m $md"
  done
done
echo "############ ALL SCRIT DONE ############"
date
