#!/usr/bin/env bash
# Day-ahead protocol with monthly refits (refit_dayahead.py), then the day-ahead scoring.
set -u
PY="C:/Users/P520/timesfm-sales/.venv/Scripts/python.exe"
cd "$(dirname "$0")"
echo ""; echo "##### $(date '+%F %T')  monthly-refit day-ahead protocol #####"
for i in 1 2 3; do
  "$PY" refit_dayahead.py 2>&1 | grep --line-buffered -vE "UserWarning|scaled_dot_product|Seed set to|^ *$"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && break
  echo "##### refit stage exited abnormally (rc=$rc), retry $i after 30 s #####"; sleep 30
done
"$PY" dayahead_v2.py 2>&1 | grep -vE "UserWarning|^ *$"
echo "##### $(date '+%F %T')  monthly-refit day-ahead protocol complete #####"
