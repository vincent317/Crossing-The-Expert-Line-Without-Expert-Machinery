#!/usr/bin/env bash
# wait.sh <job-id> [max-seconds]
set -uo pipefail
ID="$1"; MAX="${2:-2400}"; T=0
while :; do
  ST=$(cctl job get "$ID" -F status --no-input 2>/dev/null | tr -d '"[:space:]')
  echo "[$T s] $ST"
  case "$ST" in
    Succeeded|Failed|Cancelled|Killed) break ;;
  esac
  [ "$T" -ge "$MAX" ] && { echo "TIMEOUT waiting"; break; }
  sleep 20; T=$((T+20))
done
echo "--- logs ---"
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  OUT=$(cctl job logs "$ID" --only-log-lines --log-output raw --no-input 2>&1)
  if echo "$OUT" | grep -q 'UNAVAILABLE'; then sleep 15; continue; fi
  echo "$OUT"; break
done
