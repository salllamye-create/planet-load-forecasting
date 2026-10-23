#!/usr/bin/env bash
# Stage-2 experiments: they depend on the main grid having produced H=48 forecasts
# on the national grids. Resumable -- finished runs are skipped.
set -u
PY="C:/Users/P520/timesfm-sales/.venv/Scripts/python.exe"
cd "$(dirname "$0")"
GRIDS="OPSD-DE OPSD-FR OPSD-GB"

skip() { [ -f "../results/$1.json" ]; }
skipp() { [ -f "../results/prob/$1.json" ]; }

# ---- 1. weather pathway, three information regimes ---------------------------------
echo ""; echo "##### PLANET-X weather pathway #####"
for D in $GRIDS; do
  for H in 24 96; do
    for MODE in perfect noisy persist; do
      T="${D}_PLANETX_L336_H${H}_s2021_exo-${MODE}"
      skip "$T" || "$PY" exp.py --data "$D" --model PLANETX --H "$H" --lr 5e-4 \
            --epochs 10 --ms 1 --exo_mode "$MODE" --tag "_exo-${MODE}" \
            2>&1 | grep -E "^\[.*MSE=|!! "
    done
  done
  # day-ahead protocol entry for the TSO comparison
  T="${D}_PLANETX_L336_H48_s2021"
  skip "$T" || "$PY" exp.py --data "$D" --model PLANETX --H 48 --lr 5e-4 --epochs 10 \
        --ms 1 --exo_mode noisy --collect -1 2>&1 | grep -E "^\[.*MSE=|!! "
done

# ---- 2. probabilistic day-ahead forecasting ----------------------------------------
echo ""; echo "##### probabilistic (H=24, 90% intervals) #####"
for D in $GRIDS; do
  for M in DLinear PatchTST iTransformer PLANET; do
    T="${D}_${M}_L336_H24_s2021"
    skipp "$T" || "$PY" prob.py --data "$D" --model "$M" --H 24 --lr 5e-4 --epochs 10 \
          2>&1 | grep -E "^\[prob:.*CRPS|!! "
  done
done

# ---- 3. TSO day-ahead comparison ----------------------------------------------------
echo ""; echo "##### TSO day-ahead baseline #####"
"$PY" tso_baseline.py 2>&1 | tail -8

echo ""; echo "##### PIPELINE-2 COMPLETE #####"
