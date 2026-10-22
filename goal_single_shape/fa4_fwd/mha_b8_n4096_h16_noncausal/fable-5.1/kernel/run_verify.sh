#!/bin/bash
# Compile gate FIRST, checking nvcc's own exit status directly (no pipe: the
# earlier gate tested tail's status and printed "GATE OK" under a compile
# error, which let a stale binary produce a bogus benchmark).
export PATH=$PATH:/usr/local/cuda/bin
FILT="^ccache\|^\[[0-9]*/[0-9]*\]\|^FAILED\|^ninja\|^#\\$\|bytes gmem\|Compiling entry\|Function properties"

gate() {  # vlayout block_n nkv ns
  nvcc -gencode=arch=compute_100a,code=sm_100a -O3 \
    -DVLAYOUT=$1 -DBLOCK_N=$2 -DNSTAGE_KV=$3 -DNSTAGE_S=$4 \
    -DRESCALE_THRESH=8.0f -DUSE_CPASYNC=1 -DABLATE=0 \
    -c -o /tmp/gate.o fmha.cu > /tmp/gate.log 2>&1
  local rc=$?
  echo "GATE v=$1 n=$2 kv=$3 s=$4 rc=$rc"
  if [ $rc -ne 0 ]; then tail -12 /tmp/gate.log; return 1; fi
  return 0
}

echo "=== compile gate, all configs ==="
gate 2 64 5 3 || exit 1
gate 2 64 4 3 || exit 1
gate 1 64 5 3 || exit 1
echo "ALL GATES PASSED"

export VLAYOUT=2 BLOCK_N=64 NSTAGE_KV=5 NSTAGE_S=3 RESCALE_THRESH=8.0f USE_CPASYNC=1 ABLATE=0
echo "=== accuracy, shipped config ==="
timeout 600 python -u check.py > /tmp/c.log 2>&1; echo "check_rc=$?"
grep -v "$FILT" /tmp/c.log | grep -E "registers|spill|bisect|E_ref|tol =|E_mine|PASS|nan_count|over tol|RESULT|smem|rror" | head -20

echo "=== bench, shipped config ==="
timeout 600 python -u bench.py 2>&1 | grep -v "$FILT" | \
  grep -E "cuda_events|kernel-span|spans|TARGET|accuracy_pass" | head -8

for CFG in "2 4 3" "1 5 3"; do
  set -- $CFG
  export VLAYOUT=$1 NSTAGE_KV=$2 NSTAGE_S=$3
  echo "=== bench VLAYOUT=$1 NSTAGE_KV=$2 NSTAGE_S=$3 ==="
  timeout 600 python -u bench.py 2>&1 | grep -v "$FILT" | \
    grep -E "kernel-span|TARGET|accuracy_pass" | head -4
done
