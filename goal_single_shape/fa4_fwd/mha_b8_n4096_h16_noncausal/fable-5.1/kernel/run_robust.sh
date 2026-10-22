#!/bin/bash
# Turn the thin 1.2% margin into a reproducible result: 200-iteration timing,
# multi-seed accuracy, and the two comparison configs for the record.
export PATH=$PATH:/usr/local/cuda/bin
FILT="^ccache\|^\[[0-9]*/[0-9]*\]\|^FAILED\|^ninja\|^#\\$\|bytes gmem\|Compiling entry\|Function properties"
export VLAYOUT=2 BLOCK_N=64 NSTAGE_KV=5 NSTAGE_S=3 RESCALE_THRESH=8.0f USE_CPASYNC=1 ABLATE=0

echo "=== compile gate ==="
nvcc -gencode=arch=compute_100a,code=sm_100a -O3 -DVLAYOUT=2 -DBLOCK_N=64 \
  -DNSTAGE_KV=5 -DNSTAGE_S=3 -DRESCALE_THRESH=8.0f -DUSE_CPASYNC=1 -DABLATE=0 \
  -c -o /tmp/g.o fmha.cu 2>&1 | tail -5
[ $? -ne 0 ] && { echo "GATE FAILED"; exit 1; }
echo "GATE OK"

echo "=== multi-seed accuracy (shipped config) ==="
timeout 900 python -u check_seeds.py 2>&1 | grep -v "$FILT" | grep -E "seed=|worst|SEED_SWEEP|rror" | head -20

echo "=== 200-iteration timing, shipped config ==="
FMHA_ITERS=200 timeout 900 python -u bench.py 2>&1 | grep -v "$FILT" | \
  grep -E "cuda_events|kernel-span|spans|TARGET|accuracy_pass|E_ref|E_mine" | head -10

for CFG in "2 4 3" "1 5 3"; do
  set -- $CFG
  export VLAYOUT=$1 NSTAGE_KV=$2 NSTAGE_S=$3
  echo "=== comparison: VLAYOUT=$1 NSTAGE_KV=$2 NSTAGE_S=$3 ==="
  FMHA_ITERS=100 timeout 900 python -u bench.py 2>&1 | grep -v "$FILT" | \
    grep -E "kernel-span|TARGET|accuracy_pass" | head -5
done
