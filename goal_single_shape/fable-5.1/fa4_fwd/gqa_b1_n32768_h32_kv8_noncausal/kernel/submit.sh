#!/usr/bin/env bash
# Submit a batch job to the B200 pool with the listed files tarred into the entry.
# usage: ./submit.sh "<description>" "<remote command>" file1 [file2 ...]
set -euo pipefail
DESC="$1"; shift
CMD="$1"; shift
IMAGE="${IMAGE:-model-align/sglang:v0.5.19}"
B64=$(tar czf - "$@" | base64 | tr -d '\n')
cctl job create -q --no-input \
  --project moe-training --cluster jiuzhang_train --resource-pool aiforai-b200 \
  --image "$IMAGE" --billing-account-id N00007 \
  --gpu 1 --gpu-model B200 --cpu 8 --memory 32 --priority NORMAL \
  --description "$DESC" \
  --entry "mkdir -p /tmp/w && cd /tmp/w && echo $B64 | base64 -d | tar xz && $CMD"
