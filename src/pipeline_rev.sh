#!/usr/bin/env bash
# Revision pipeline (PLAN_v2.md, 2026-10-19 21:30: consolidated referee plan; GPU order confirmed by
# the user on 2026-10-19 ~22:50, TimesNet last). Serial on one GPU; every stage is resumable, so the
# script can be interrupted and restarted freely. Waits for the budget-check stage of the main pass
# (run_all_v2 main -ep90, started by pipeline_v2_resume.sh) to release the GPU before it starts.
set -u
PY="C:/Users/P520/timesfm-sales/.venv/Scripts/python.exe"
cd "$(dirname "$0")"
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
GRID8="OPSD-DE OPSD-FR OPSD-GB OPSD-ES OPSD-IT OPSD-NL OPSD-PL OPSD-BE"
DISC="../results_v2_discarded"

echo "##### $(date '+%F %T')  waiting for the GPU (budget-check stage of the main pass) #####"
# (nvidia-smi lists every WDDM process holding a GPU context, so only Python processes count)
while [ "$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -ic python)" -gt 0 ]; do sleep 120; done
echo "##### $(date '+%F %T')  GPU free, revision pipeline starts #####"

# --- 0. the budget check itself, in case the main pass left gaps (resumable, normally a no-op)
R main --datasets OPSD-DE --models headline --horizons 24,96,336 --regimes R2 --epochs 90 --patience 10 --suffix=-ep90

# --- 1. covariate-pathway control: TiDE fed through the shared head at TiDE's tuned settings
R main --datasets GRID --models TiDEh --horizons 24,96,336 --regimes R2 --suffix=-viahead

# --- 2. probabilistic stage re-run with the operational (delayed-feedback) calibration, TFT added;
#        the look-ahead records are kept under results_v2_discarded/prob_lookahead (once)
if [ ! -f "$DISC/prob_lookahead/.moved" ]; then
  "$PY" discard_records.py prob_lookahead && mkdir -p "$DISC/prob_lookahead" && touch "$DISC/prob_lookahead/.moved"
fi
for D in $GRID8; do
  for M in PLANET PLANETX NBEATS PatchTST CalTST DLinear TiDE NHITS TFT; do
    for S in 2021 2022 2023; do
      "$PY" prob_v2.py --data "$D" --model "$M" --H 24 --seed "$S" --auto 1 --epochs 30 --patience 5 \
          --regime R2 --tag _v2-R2 2>&1 | grep -E "^\[prob:|!! |Traceback|Error"
    done
  done
done
echo "##### $(date '+%F %T')  probabilistic stage complete #####"

# --- 3. correlated (AR(1)) weather-noise sensitivity, TiDE and PLANET, three systems
R main --datasets OPSD-DE,OPSD-FR,OPSD-ES --models TiDE,PLANET --horizons 24,96,336 --regimes R2 --exo_mode noisy_ar --suffix=-arnoise

# --- 3b. population-weighted multi-point weather (six metropolitan points) on the three most
#         weather-driven systems, TiDE and PLANET at the base system's selections
R main --datasets OPSD-FR-mp,OPSD-ES-mp,OPSD-IT-mp --models TiDE,PLANET --horizons 24,96,336 --regimes R2

# --- 4. learning-rate selection under the other regimes (DE, headline models, H=96)
R tune --datasets OPSD-DE --models headline --regimes R0
R tune --datasets OPSD-DE --models headline --regimes R1

# --- 5. two further seeds for the key ablation variants (patch backbone)
R ablation --datasets OPSD-DE,OPSD-FR --horizons 24,96,336 --seeds 2024,2025 --backbones patch --variants full,utc,no_pl,no_resid,relative

# --- 6. Informer and Autoformer with instance normalisation, full protocol; the records made
#        without it are kept under results_v2_discarded/norevin (once)
if [ ! -f "$DISC/norevin/.moved" ]; then
  "$PY" discard_records.py norevin && mkdir -p "$DISC/norevin" && touch "$DISC/norevin/.moved"
fi
R tune --datasets OPSD-DE,OPSD-FR --models Informer,Autoformer
R tune --datasets "$GRID6" --models Informer,Autoformer
R lookback --datasets GRID --models Informer,Autoformer --horizons 24,96,336
R main --datasets GRID --models Informer,Autoformer --horizons 24,96,336 --regimes R2

# --- 7. ETT and Electricity: learning rate selected per dataset, cells re-run (records made under
#        the inherited selection are moved to results_v2/stale by the driver)
BENCH_ETT="NBEATS,PatchTST,CalTST,PLANET,PLANETX,DLinear,FITS,RLinear,TiDE,NHITS"
BENCH_ECL="NBEATS,PatchTST,CalTST,PLANET,PLANETX,DLinear,TiDE,NHITS,iTransformer"
R tune --datasets ETT --models "$BENCH_ETT"
R main --datasets ETT --models "$BENCH_ETT" --horizons 96,192,336,720 --regimes R1
R tune --datasets ECL --models "$BENCH_ECL"
R main --datasets ECL --models "$BENCH_ECL" --horizons 96,336 --regimes R1

# --- 8. budget sensitivity on the second reference system
R main --datasets OPSD-FR --models headline --horizons 24,96,336 --regimes R2 --epochs 90 --patience 10 --suffix=-ep90

# --- 9. day-ahead protocol with monthly refits (script added by the revision; skipped if absent)
[ -f ./refit_dayahead.sh ] && bash ./refit_dayahead.sh

echo ""; echo "##### REVISION PIPELINE COMPLETE $(date '+%F %T') #####"

# --- 10. deferred pass: TimesNet under the identical protocol (tuning, lookback, headline)
R tune --datasets OPSD-DE,OPSD-FR --models TimesNet
R tune --datasets "$GRID6" --models TimesNet
R lookback --datasets GRID --models TimesNet --horizons 24,96,336
R main --datasets GRID --models TimesNet --horizons 24,96,336 --regimes R2
echo ""; echo "##### REVISION PIPELINE (incl. TimesNet) COMPLETE $(date '+%F %T') #####"
