#!/bin/sh
PY=/c/Users/galma/anaconda3/python.exe
run() {  # model target
  for md in pulse instant; do
    echo "############ characterize $1 $2 $md ############"
    date
    "$PY" -u -m analysis.characterize --model "$1" --mode "$md" --targets "$2" --shared \
          --n-phase 32 --skip-cap 48 --relax-tag zoo03 \
          --tag zoo03 --scrit-tag zoo03rd || echo "FAILED $1 $2 $md"
  done
}
run almeida   BMAL1
run korencic  Bmalx
run goldbeter MB
echo "############ ALL BMAL SURFACES DONE ############"
date
