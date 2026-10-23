#!/usr/bin/env bash
# Revision pipeline, third script (PLAN_v2.md, 2026-10-21 05:40): the R0 and R1 cells of Germany
# re-run with the learning rate selected under their own regime, for the headline models whose
# selection differs from the R2 rate (the per-regime selection check; records tagged -ownlr; a
# cell whose own selection equals the R2 rate keeps its headline record). Waits for the second
# script's COMPLETE marker and a free GPU, then runs serially; resumable like the other scripts.
set -u
PY="C:/Users/P520/timesfm-sales/.venv/Scripts/python.exe"
cd "$(dirname "$0")"
R() { local n=0 c=0; while :; do
        echo ""; echo "##### $(date '+%F %T')  $* #####"
        stdbuf -oL "$PY" run_all_v2.py "$@" 2>&1 | grep --line-buffered -vE "UserWarning|scaled_dot_product|Seed set to|^ *$"
        rc=${PIPESTATUS[0]}; n=$((n+1))
        [ "$rc" -eq 3 ] && [ "$n" -lt 80 ] && { echo "##### stage exited for restart ($n) #####"; continue; }
        [ "$rc" -ne 0 ] && [ "$c" -lt 3 ] && { c=$((c+1)); echo "##### stage exited abnormally (rc=$rc), retry $c after 30 s #####"; sleep 30; continue; }
        [ "$rc" -ne 0 ] && echo "##### stage abandoned after repeated abnormal exits (rc=$rc) #####"
        break; done; }
echo "##### $(date '+%F %T')  waiting for the second revision script (its COMPLETE marker) #####"
until grep -q "second script) COMPLETE" ../logs/pipeline_rev2.log; do sleep 300; done
while [ "$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -ic python)" -gt 0 ]; do sleep 60; done
echo "##### $(date '+%F %T')  GPU free, revision pipeline (third script) starts #####"

# --- 10. per-regime learning rate: Germany R0/R1 cells of the models whose selection changed
R main --datasets OPSD-DE --models headline --horizons 24,96,336 --regimes R0,R1 --lr_regime own --suffix=-ownlr

echo ""; echo "##### REVISION PIPELINE (third script) COMPLETE $(date '+%F %T') #####"
