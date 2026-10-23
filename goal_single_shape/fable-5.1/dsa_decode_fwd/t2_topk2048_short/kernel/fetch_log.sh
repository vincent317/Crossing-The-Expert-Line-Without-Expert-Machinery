#!/usr/bin/env bash
# Fetch the archived log of a finished job (retries while the log is still being archived).
# Usage: ./fetch_log.sh [JOB_ID]   (defaults to logs/last_job_id)
set -euo pipefail
ID="${1:-$(cat logs/last_job_id)}"
mkdir -p logs
for _ in $(seq 1 30); do
  OUT=$(cctl job logs "$ID" --only-log-lines --log-output raw --no-input 2>&1 || true)
  if [ -z "$OUT" ] || echo "$OUT" | grep -q 'UNAVAILABLE'; then sleep 10; continue; fi
  printf '%s\n' "$OUT" > "logs/$ID.log"
  echo "saved logs/$ID.log ($(wc -l < "logs/$ID.log") lines)"
  exit 0
done
echo "log still unavailable for $ID"; exit 1
