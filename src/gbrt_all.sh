#!/usr/bin/env bash
set -u
PY="C:/Users/P520/timesfm-sales/.venv/Scripts/python.exe"
cd "$(dirname "$0")"
for H in 24 48 96 336; do
  for D in OPSD-DE OPSD-FR OPSD-GB; do
    if [ ! -f "../results/${D}_HistGBR_L336_H${H}_s2021.json" ]; then
      echo "### HistGBR $D H=$H"
      "$PY" gbrt.py --data "$D" --H "$H" --iters 300 --stride 5 2>&1 | grep -E "^\[|step 1/"
    fi
  done
done
echo "### GBRT COMPLETE"
