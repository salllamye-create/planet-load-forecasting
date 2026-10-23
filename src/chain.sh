#!/usr/bin/env bash
# Runs stage-2 automatically once the main pipeline reports completion, so the
# weather/probabilistic/TSO work cannot be forgotten at the end.
set -u
cd "$(dirname "$0")"
until grep -aq "PIPELINE COMPLETE" ../logs/pipeline.log 2>/dev/null; do sleep 120; done
echo "main pipeline finished -> starting stage 2"
bash pipeline2.sh
