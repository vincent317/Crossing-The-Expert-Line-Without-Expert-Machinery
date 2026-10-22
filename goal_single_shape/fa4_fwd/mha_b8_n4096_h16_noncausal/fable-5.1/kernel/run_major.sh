#!/bin/bash
# MN-major A-operand encoding probe (each variant in its own process) +
# the missing MMA-only ablation point.
export PATH=$PATH:/usr/local/cuda/bin
nvcc -gencode=arch=compute_100a,code=sm_100a -O3 -o /tmp/t5major t5_major.cu 2>&1 | tail -5
echo "build_rc=$?"
for V in 0 1 2 3 4 5; do
  timeout 60 /tmp/t5major $V || echo "variant=$V exited rc=$?"
done
echo "=== ABLATE=6 (MMA only) ==="
export VLAYOUT=1 BLOCK_N=64 RESCALE_THRESH=0.0f USE_CPASYNC=1 ABLATE=6
timeout 420 python -u bench_only.py 2>&1 | grep -E "ABLATE_RESULT|rror" | head -3
echo "=== ABLATE=3+4 both loads removed: use ABLATE=7 ==="
