#!/usr/bin/env bash
# Revision pipeline, second script (PLAN_v2.md, 2026-10-20 09:30): the remaining stages of
# pipeline_rev.sh re-ordered so that the stages the manuscript's pending macros depend on run
# first, the two protocol corrections of existing results (ETT/ECL learning rates, Informer and
# Autoformer with instance normalisation) run last, and the deferred TimesNet pass is dropped
# (submission deadline; see the plan entry). Serial on one GPU; every stage is resumable, so the
# script can be interrupted and restarted freely. Waits for the GPU to be free before it starts
# (the last probabilistic run of the first script finishes as an orphan).
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

echo "##### $(date '+%F %T')  waiting for the GPU (last run of the first revision script) #####"
# (nvidia-smi lists every WDDM process holding a GPU context, so only Python processes count)
while [ "$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -ic python)" -gt 0 ]; do sleep 60; done
echo "##### $(date '+%F %T')  GPU free, revision pipeline (second script) starts #####"

# --- 2. probabilistic stage with the operational (delayed-feedback) calibration, TFT added
#        (resumable: prob_v2.py skips a cell whose record exists)
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

# --- 8. budget sensitivity on the second reference system
R main --datasets OPSD-FR --models headline --horizons 24,96,336 --regimes R2 --epochs 90 --patience 10 --suffix=-ep90

# --- 9. day-ahead protocol with monthly refits
[ -f ./refit_dayahead.sh ] && bash ./refit_dayahead.sh

echo ""; echo "##### $(date '+%F %T')  all stages the pending macros depend on are complete #####"

# --- 7. ETT and Electricity: learning rate selected per dataset, cells re-run (records made under
#        the inherited selection are moved to results_v2/stale by the driver)
BENCH_ETT="NBEATS,PatchTST,CalTST,PLANET,PLANETX,DLinear,FITS,RLinear,TiDE,NHITS"
BENCH_ECL="NBEATS,PatchTST,CalTST,PLANET,PLANETX,DLinear,TiDE,NHITS,iTransformer"
R tune --datasets ETT --models "$BENCH_ETT"
R main --datasets ETT --models "$BENCH_ETT" --horizons 96,192,336,720 --regimes R1
R tune --datasets ECL --models "$BENCH_ECL"
R main --datasets ECL --models "$BENCH_ECL" --horizons 96,336 --regimes R1
echo ""; echo "##### $(date '+%F %T')  ETT/ECL re-run complete #####"

# --- 6. Informer and Autoformer with instance normalisation, full protocol; the records made
#        without it are kept under results_v2_discarded/norevin (once)
if [ ! -f "$DISC/norevin/.moved" ]; then
  "$PY" discard_records.py norevin && mkdir -p "$DISC/norevin" && touch "$DISC/norevin/.moved"
fi
R tune --datasets OPSD-DE,OPSD-FR --models Informer,Autoformer
R tune --datasets "$GRID6" --models Informer,Autoformer
R lookback --datasets GRID --models Informer,Autoformer --horizons 24,96,336
R main --datasets GRID --models Informer,Autoformer --horizons 24,96,336 --regimes R2

echo ""; echo "##### REVISION PIPELINE (second script) COMPLETE $(date '+%F %T') #####"
