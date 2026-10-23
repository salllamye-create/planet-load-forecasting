#!/usr/bin/env bash
# v2 experiment pipeline (PLAN_v2.md sections 6-11). Serial on one GPU; every stage is
# resumable (finished runs are skipped), so it can be interrupted and restarted freely.
# Final protocol (2026-10-01): every model gets the same tuning budget (width x learning rate on
# DE/FR, learning rate per system, lookback per system and horizon), is trained to early stopping
# (cap 30 epochs, patience 5), and every reported cell is a three-seed mean.
set -u
PY="C:/Users/P520/timesfm-sales/.venv/Scripts/python.exe"
cd "$(dirname "$0")"
# A stage that exits with code 3 (CUDA state suspect after repeated failures) is re-invoked in
# a fresh process, up to eighty times (each restart costs seconds); finished and permanently
# failed jobs are skipped each time. Any other abnormal exit (a hard crash prints no Python
# traceback) is retried up to three times after a pause, so a driver hiccup cannot skip a stage.
R() { local n=0 c=0; while :; do
        echo ""; echo "##### $(date '+%F %T')  $* #####"
        stdbuf -oL "$PY" run_all_v2.py "$@" 2>&1 \
          | grep --line-buffered -vE "UserWarning|scaled_dot_product|Seed set to|^ *$"
        rc=${PIPESTATUS[0]}; n=$((n+1))
        [ "$rc" -eq 3 ] && [ "$n" -lt 80 ] && { echo "##### stage exited for restart ($n) #####"; continue; }
        [ "$rc" -ne 0 ] && [ "$c" -lt 3 ] && { c=$((c+1)); echo "##### stage exited abnormally (rc=$rc), retry $c after 30 s #####"; sleep 30; continue; }
        [ "$rc" -ne 0 ] && echo "##### stage abandoned after repeated abnormal exits (rc=$rc) #####"
        break; done; }
GRID6="OPSD-GB,OPSD-ES,OPSD-IT,OPSD-NL,OPSD-PL,OPSD-BE"

# --- 1. tuning: width x learning rate on the two reference systems, learning rate elsewhere
# TimesNet keeps the card at full load for one to three hours per run and was running when the
# driver faulted on 2026-10-02; it is scheduled in a deferred pass at the end (same protocol).
R tune --datasets OPSD-DE,OPSD-FR --models all --exclude TimesNet
R tune --datasets "$GRID6" --models all --exclude TimesNet

# --- 2. lookback selection for every model on every system and horizon
R lookback --datasets GRID --models all --horizons 24,96,336 --exclude TimesNet

# --- 3. headline: regime R2, three seeds, every model, eight systems
R main --datasets GRID --models all --horizons 24,96,336 --regimes R2 --exclude TimesNet

# --- 4. day-ahead protocol (H=48, saved forecasts) and the information regimes, headline models
R main --datasets GRID --models headline --horizons 48 --regimes R2
R main --datasets GRID --models headline --horizons 24,96,336 --regimes R0,R1
"$PY" dayahead_v2.py

# --- 5. PLANET ablation on both backbones, two systems, three seeds
R ablation --datasets OPSD-DE,OPSD-FR --horizons 24,96,336

# --- 6. mechanism experiments: learning curves, temporal drift, cross-grid transfer, three seeds
R mech --datasets OPSD-DE,OPSD-FR,OPSD-ES --models NBEATS,PatchTST,CalTST,PLANET,PLANETX,TFT,TiDE --horizons 96

# --- 7. probabilistic day-ahead, headline models with a quantile head, eight systems, three seeds
for D in OPSD-DE OPSD-FR OPSD-GB OPSD-ES OPSD-IT OPSD-NL OPSD-PL OPSD-BE; do
  for M in PLANET PLANETX NBEATS PatchTST CalTST DLinear TiDE NHITS; do
    for S in 2021 2022 2023; do
      "$PY" prob_v2.py --data "$D" --model "$M" --H 24 --seed "$S" --auto 1 --epochs 30 --patience 5 \
          --regime R2 --tag _v2-R2 2>&1 | grep -E "^\[prob:|!! "
    done
  done
done

# --- 8. Electricity (M setting, calendar regime) and ETT, headline models, three seeds
R main --datasets ECL --models NBEATS,PatchTST,CalTST,PLANET,PLANETX,DLinear,TiDE,NHITS,iTransformer --horizons 96,336 --regimes R1
R main --datasets ETT --models NBEATS,PatchTST,CalTST,PLANET,PLANETX,DLinear,FITS,RLinear,TiDE,NHITS --horizons 96,192,336,720 --regimes R1

# --- 9. zero-shot foundation models, covariate regime, both contexts
echo ""; echo "##### $(date '+%F %T')  foundation models R2 #####"
"$PY" fm_zero_shot.py --regimes R2 --L 336,1024 --horizons 24,96,336 \
    --datasets OPSD-DE,OPSD-FR,OPSD-GB,OPSD-ES,OPSD-IT,OPSD-NL,OPSD-PL,OPSD-BE 2>&1 | grep -E "WAPE=|!! |Error"

echo ""; echo "##### PIPELINE v2 COMPLETE $(date '+%F %T') #####"

# --- 10. budget sensitivity: the headline models on DE at three times the cap (90 epochs,
# patience 10, three seeds). A robustness check reported in the discussion, not a selection input.
R main --datasets OPSD-DE --models headline --horizons 24,96,336 --regimes R2 --epochs 90 --patience 10 --suffix=-ep90
echo ""; echo "##### PIPELINE v2 (incl. budget check) COMPLETE $(date '+%F %T') #####"

# --- 11. deferred pass: TimesNet under the identical protocol (tuning, lookback, headline)
R tune --datasets OPSD-DE,OPSD-FR --models TimesNet
R tune --datasets "$GRID6" --models TimesNet
R lookback --datasets GRID --models TimesNet --horizons 24,96,336
R main --datasets GRID --models TimesNet --horizons 24,96,336 --regimes R2
echo ""; echo "##### PIPELINE v2 (incl. deferred TimesNet pass) COMPLETE $(date '+%F %T') #####"
