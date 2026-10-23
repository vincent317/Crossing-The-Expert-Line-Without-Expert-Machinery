#!/usr/bin/env bash
# Submit a job and re-submit (up to N attempts) while the archived log lacks the final benchmark line.
# Usage: ./run_until_done.sh "<description>" "<job_env lines>" [max_attempts]
set -uo pipefail
DESC=$1; ENVL=${2:-}; MAX=${3:-3}
for attempt in $(seq 1 "$MAX"); do
  ./submit_job.sh "$DESC (attempt $attempt)" "$ENVL" 2>&1 | grep -E 'JOB_ID|status=(Succeeded|Failed|Killed)' | tail -2
  ./fetch_log.sh 2>&1 | tail -1
  ID=$(cat logs/last_job_id)
  if grep -q '^\[RESULT\] warm-L2:' "logs/$ID.log"; then echo "complete: logs/$ID.log"; exit 0; fi
  echo "attempt $attempt incomplete (job $ID, $(wc -l < logs/$ID.log) lines); retrying"
done
exit 1
