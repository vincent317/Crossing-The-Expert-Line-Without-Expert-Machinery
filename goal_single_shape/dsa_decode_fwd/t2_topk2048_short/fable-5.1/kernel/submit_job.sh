#!/usr/bin/env bash
# Submit the bundle as a one-shot B200 batch job, poll to completion, save the log.
# Usage: ./submit_job.sh "<description>" ["shell lines written to job_env.sh, sourced by job.sh"]
set -euo pipefail
DESC="${1:-dsa fwd kernel experiment}"
printf '%s\n' "${2:-}" > job_env.sh
FILES=(job.sh job_env.sh sass_summary.py dsa_kernel.cu dsa_mma.cu dsa_ops.py naive_fp32.py workload.py run_ref.py check.py bench.py
       sparse_indices_9d4a5f21268e484ea05a2f2af91d9fa7.safetensors
       ref_repo/COMMIT ref_repo/kernels/dsa_attention/kernel.py)
mkdir -p logs
B64=$(tar czf - "${FILES[@]}" | base64 | tr -d '\n')
echo "bundle: ${#B64} base64 chars"
ID=$(cctl job create -q --no-input \
  --project moe-training --cluster jiuzhang_train --resource-pool aiforai-b200 \
  --image infra/nvidia-pytorch:latest --billing-account-id N00007 \
  --gpu 1 --gpu-model B200 --cpu 8 --memory 32 --priority NORMAL \
  --description "$DESC" \
  --entry "mkdir -p /tmp/w && cd /tmp/w && echo $B64 | base64 -d | tar xz && bash job.sh 2>&1")
echo "JOB_ID=$ID"
echo "$ID" > logs/last_job_id
for _ in $(seq 1 36); do
  S=$(cctl job get "$ID" -F status --no-input 2>/dev/null | python3 -c 'import sys,json
d=json.load(sys.stdin); print(d.get("status") if isinstance(d,dict) else d)')
  echo "[$(date +%H:%M:%S)] status=$S"
  case "$S" in Succeeded|Failed|Stopped|Killed|OOMKilled|Invalid|ImagePullBackOff) break;; esac
  sleep 15
done
