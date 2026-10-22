#!/usr/bin/env bash
# Poll a job until it terminates, then dump its logs.
set -uo pipefail
ID="$1"
for i in $(seq 1 240); do
  ST=$(cctl job get "$ID" -F status --no-input 2>/dev/null | tr -d '"' | tr -d ' \n')
  echo "[$i] status=$ST"
  case "$ST" in
    Succeeded|Failed|Stopped|Cancelled) break ;;
  esac
  sleep 15
done
echo "=== logs ==="
for i in $(seq 1 30); do
  OUT=$(cctl job logs "$ID" --only-log-lines --log-output raw --no-input 2>&1)
  if echo "$OUT" | grep -q 'UNAVAILABLE'; then sleep 10; continue; fi
  echo "$OUT"; break
done
