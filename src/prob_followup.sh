#!/usr/bin/env bash
# Follow-up to pipeline_v2.sh stage 7 (probabilistic day-ahead). The stage ran on 2026-10-18 while
# the quantile wrapper built PatchTST+Cal with H*Q outputs, which broke its calendar token count
# (fixed in prob_v2.py / baselines.py the same day: the head is widened, not H); the 24 CalTST runs
# and one DLinear seed (GB, 2023, transient) therefore have no record. This script re-runs exactly
# those 25 jobs under the identical command line, with the full output kept (the stage's own
# filter hid the tracebacks), once the pipeline has reached the ETT stage, whose small models
# leave room on the 8 GB card.
set -u
PY="C:/Users/P520/timesfm-sales/.venv/Scripts/python.exe"
cd "$(dirname "$0")"
LOG=../logs/pipeline_v2.log
echo "##### $(date '+%F %T')  prob follow-up: waiting for the ETT stage #####"
until grep -qE "^##### .* main --datasets ETT " "$LOG"; do sleep 120; done
echo "##### $(date '+%F %T')  prob follow-up: ETT stage detected, starting #####"
run() { # data model seed
  stdbuf -oL "$PY" prob_v2.py --data "$1" --model "$2" --H 24 --seed "$3" --auto 1 --epochs 30 --patience 5 \
      --regime R2 --tag _v2-R2 2>&1 | grep --line-buffered -vE "UserWarning|scaled_dot_product|Seed set to|^ *$"
  echo "   rc=${PIPESTATUS[0]}"
}
for D in OPSD-DE OPSD-FR OPSD-GB OPSD-ES OPSD-IT OPSD-NL OPSD-PL OPSD-BE; do
  for S in 2021 2022 2023; do run "$D" CalTST "$S"; done
done
run OPSD-GB DLinear 2023
echo "##### $(date '+%F %T')  prob follow-up COMPLETE #####"
ls ../results_v2/prob/*_v2-R2.json | wc -l
