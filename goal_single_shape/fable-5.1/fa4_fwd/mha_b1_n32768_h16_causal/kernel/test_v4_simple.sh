#!/bin/bash
set -e
echo "=== [1] build v4 ==="
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 \
  -Xptxas -v --shared -Xcompiler -fPIC fa_fwd4.cu -o libfa_fwd4.so 2>&1 | tail -30
echo "build_done: $(ls -lh libfa_fwd4.so)"

echo "=== [2] smoke test ==="
export PIP_CONSTRAINT=
pip install -q torch 2>&1 | tail -3
python -u fa_fwd4.py 2>&1

echo "=== [3] full check + bench ==="
python -u check_v4.py 2>&1 | tail -20
python -u bench_v4.py 2>&1 | tail -20

echo "=== ALL_DONE ==="
