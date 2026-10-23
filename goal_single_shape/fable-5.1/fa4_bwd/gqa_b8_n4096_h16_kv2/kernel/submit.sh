#!/usr/bin/env bash
# submit.sh <image> <desc> <entry-cmd> <file>...
set -euo pipefail
IMAGE="$1"; shift
DESC="$1"; shift
ENTRY="$1"; shift
B64=$(tar czf - "$@" | base64 | tr -d '\n')
cctl job create -q --no-input \
  --project moe-training --cluster jiuzhang_train --resource-pool aiforai-b200 \
  --image "$IMAGE" --billing-account-id N00007 \
  --gpu 1 --gpu-model B200 --cpu 8 --memory 32 --priority NORMAL \
  --description "$DESC" \
  --entry "mkdir -p /tmp/w && cd /tmp/w && echo $B64 | base64 -d | tar xz && $ENTRY"
