#!/bin/bash
# Attribute measured time to each pipeline phase.
export PATH=$PATH:/usr/local/cuda/bin
export VLAYOUT=1 BLOCK_N=64 RESCALE_THRESH=0.0f USE_CPASYNC=1
for A in 0 1 2 3 4 5 6; do
  export ABLATE=$A
  echo -n "ABLATE=$A : "
  timeout 420 python -u bench_only.py 2>&1 | grep -E "ABLATE_RESULT|error|Error" | head -3
done
