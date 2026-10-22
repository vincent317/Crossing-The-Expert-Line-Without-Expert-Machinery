#!/bin/bash
# Is 551 TFLOPS the mma.sync ceiling or an artifact of one configuration?
# -Xptxas -v prints register usage and spill counts so "no spilling" is
# observed rather than inferred.
export PATH=$PATH:/usr/local/cuda/bin
nvcc -gencode=arch=compute_100a,code=sm_100a -O3 --ptxas-options=-v \
     -o /tmp/mma_sweep probe_mma_sweep.cu 2>&1 | grep -E "registers|spill|error"
echo "build_rc=$?"
echo
/tmp/mma_sweep
