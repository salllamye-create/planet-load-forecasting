#!/usr/bin/env bash
# Does the phase-locked tier help regardless of what models the residual?
# Pairs it with a second, architecturally unrelated backbone (N-BEATS-style basis stack)
# and compares each backbone with and without the tier.
set -u
PY="C:/Users/P520/timesfm-sales/.venv/Scripts/python.exe"
cd "$(dirname "$0")"
for D in OPSD-DE OPSD-FR OPSD-GB; do
  for H in 24 96 336; do
    T="../results/${D}_PLANET_L336_H${H}_s2021_resid-basis.json"
    [ -f "$T" ] || "$PY" exp.py --data "$D" --model PLANET --H "$H" --lr 5e-4 --epochs 10 \
        --ms 1 --train_stride 2 --collect -1 \
        --extra '{"resid":"basis"}' --tag "_resid-basis" 2>&1 | grep -E "^\[.*MSE=|!! "
  done
done
echo "### BACKBONE TRANSFER COMPLETE"
