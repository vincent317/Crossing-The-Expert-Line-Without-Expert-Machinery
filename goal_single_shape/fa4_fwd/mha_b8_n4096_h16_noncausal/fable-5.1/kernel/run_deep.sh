#!/bin/bash
# Deep pipeline + MN-major V (no transpose). VLAYOUT=1 is the control that
# isolates how much the transpose-free V streaming is worth.
export PATH=$PATH:/usr/local/cuda/bin
FILT="^ccache\|^\[[0-9]*/[0-9]*\]\|^FAILED\|^ninja\|^#\\$\|bytes gmem\|Compiling entry\|Function properties"
for CFG in "2 64 5 3" "2 64 4 3" "1 64 5 3"; do
  set -- $CFG
  export VLAYOUT=$1 BLOCK_N=$2 NSTAGE_KV=$3 NSTAGE_S=$4
  export RESCALE_THRESH=8.0f USE_CPASYNC=1 ABLATE=0
  echo "######## VLAYOUT=$1 BLOCK_N=$2 NSTAGE_KV=$3 NSTAGE_S=$4 ########"
  timeout 420 python -u check.py > /tmp/c.txt 2>&1; RC=$?
  grep -v "$FILT" /tmp/c.txt | grep -E "registers|spill|bisect|E_ref|tol|E_mine|PASS|nan|over tol|RESULT|smem|rror" | head -20
  echo "check_rc=$RC"
  [ $RC -eq 124 ] && { echo "HANG -- aborting"; break; }
  [ $RC -ne 0 ] && continue
  timeout 420 python -u bench.py 2>&1 | grep -v "$FILT" | grep -E "cuda_events|kernel-span|spans|TARGET|accuracy_pass|rror" | head -10
done
