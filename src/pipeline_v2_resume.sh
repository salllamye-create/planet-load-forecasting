#!/usr/bin/env bash
# Resumes pipeline_v2.sh from its stage 10 after the 2026-10-19 defect: the budget-check stage
# was invoked with "--suffix -ep90", which the argument parser read as an option flag, so the
# stage was abandoned with no record and the script went on to the deferred TimesNet pass. This
# script runs the budget check with the corrected syntax (now also fixed in pipeline_v2.sh) and
# then the deferred TimesNet pass, identical to pipeline_v2.sh stage 11; finished runs are
# skipped, so the TimesNet tuning runs completed before the stop are kept.
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
