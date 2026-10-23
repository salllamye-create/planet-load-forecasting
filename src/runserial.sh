#!/usr/bin/env bash
# Run the two pipelines strictly one after the other. Sharing an 8 GB thermally
# throttled GPU between them made both an order of magnitude slower than either alone.
set -u
cd "$(dirname "$0")"
echo "=== stage 2 first: weather + probabilistic (feeds two paper sections) ==="
bash pipeline2.sh
echo "=== then Electricity ==="
bash pipeline.sh
echo "=== ALL COMPLETE ==="
