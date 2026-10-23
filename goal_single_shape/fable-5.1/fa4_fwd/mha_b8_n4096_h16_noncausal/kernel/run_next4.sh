#!/bin/bash
# tcgen05 peak probe, then FMHA configs:
#   N128 exact | N64 exact | N64 lazy | N64 lazy+cp.async | N128 lazy+cp.async
# A hung check.py (exit 124) aborts the remaining configs.
export PATH=$PATH:/usr/local/cuda/bin
echo "=== tcgen05 UMMA peak probe ==="
nvcc -gencode=arch=compute_100a,code=sm_100a -O3 -o /tmp/probe_umma probe_umma.cu 2>&1 | grep -v "^$" | tail -8
echo "probe_build_rc=${PIPESTATUS[0]}"
timeout 180 /tmp/probe_umma; echo "probe_run_rc=$?"
FILT="^ccache\|^\[.*/.*\]\|^FAILED\|^ninja\|^ptxas info    : Compiling\|^ptxas info    : Function properties\|^#\\$\|bytes gmem"
for CFG in "128 0.0f 0" "64 0.0f 0" "64 8.0f 0" "64 8.0f 1" "128 8.0f 1"; do
  set -- $CFG
  export VLAYOUT=1 BLOCK_N=$1 RESCALE_THRESH=$2 USE_CPASYNC=$3
  echo "################ BLOCK_N=$1 RESCALE_THRESH=$2 USE_CPASYNC=$3 ################"
  echo "=== accuracy check ==="
  timeout 420 python -u check.py > /tmp/check_out.txt 2>&1; RC=$?
  grep -v "$FILT" /tmp/check_out.txt | tail -40
  echo "check_rc=$RC"
  if [ $RC -eq 124 ]; then echo "CHECK TIMED OUT (hang?) -- aborting remaining configs"; break; fi
  echo "=== benchmark ==="
  timeout 420 python -u bench.py 2>&1 | grep -v "$FILT" | tail -16
done
