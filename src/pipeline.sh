#!/usr/bin/env bash
# Full experiment pipeline, ordered so the results the paper leans on hardest land first.
# Every stage is resumable: finished runs are skipped on restart.
set -u
PY="C:/Users/P520/timesfm-sales/.venv/Scripts/python.exe"
cd "$(dirname "$0")"
R() { echo ""; echo "##### $* #####"; stdbuf -oL "$PY" run_all.py "$@" 2>&1 \
      | grep -vE "UserWarning|scaled_dot_product|^ *$"; }

# --- 1. learning-rate selection, one reference dataset per family ------------------
R lr   --datasets OPSD-DE --horizons 24,96,336 --epochs 10 --patience 3
R lr   --datasets ETTh1   --horizons 96,720     --epochs 10 --patience 3

# --- 2. the core result: national system load -------------------------------------
R main --datasets GRID --horizons 24,96,336 --epochs 10

# --- 3. day-ahead protocol for the TSO comparison ---------------------------------
R main --datasets GRID --horizons 48 --epochs 10 \
       --models SeasonalNaive,DLinear,TimeMixer,iTransformer,PatchTST,PLANET

# --- 4. ablations and lookback sensitivity ----------------------------------------
R ablation --datasets OPSD-DE --horizons 24,96,336 --epochs 10
R lookback --datasets OPSD-DE --horizons 96 --epochs 10

# --- 5. standard hourly benchmark --------------------------------------------------
R main --datasets ETTh1,ETTh2 --horizons 96,192,336,720 --epochs 10
R ablation --datasets ETTh1 --horizons 96,336 --epochs 10

# --- 6. encoder-decoder transformers at their native short lookback ----------------
R shortlook --datasets GRID --horizons 24,96,336 --epochs 10
R shortlook --datasets ETTh1,ETTh2 --horizons 96,192,336,720 --epochs 10

# --- 7. 15-minute benchmark ---------------------------------------------------------
R main --datasets ETTm1,ETTm2 --horizons 96,192,336,720 --epochs 10
R shortlook --datasets ETTm1,ETTm2 --horizons 96,192,336,720 --epochs 10

# --- 8. seed replication on the headline dataset -----------------------------------
R main --datasets OPSD-DE --horizons 24,96 --seeds 2022,2023 --epochs 10 \
       --models DLinear,NBEATS,iTransformer,PatchTST,CalTST,PLANET

# --- 9. Electricity, 321 channels (the most expensive stage, so it goes last) ------
R lr   --datasets ECL --horizons 96 --epochs 10 --patience 3 --lrs 1e-4,1e-3
R main --datasets ECL --horizons 96,336 --epochs 10 --patience 3
R ablation --datasets ECL --horizons 96 --epochs 10 --patience 3

echo ""; echo "##### PIPELINE COMPLETE #####"
