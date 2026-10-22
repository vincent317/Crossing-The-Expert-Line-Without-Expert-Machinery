#!/bin/bash
set -e
export PIP_CONSTRAINT=
pip install -q torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | grep -v "already satisfied" || true

echo "=== [1] build v2 ==="
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 --shared -Xcompiler -fPIC fa_fwd2.cu -o libfa_fwd2.so 2>&1 | grep -E "error|warning|registers|smem" | head -20
ls -lh libfa_fwd2.so

echo "=== [2] check v2 precision ==="
python -u check_v4.py 2>&1 | grep -E "CHECK|tol|PASS|FAIL"

echo "=== [3] benchmark v2 ==="
python -u bench_v4.py 2>&1 | grep -E "NEW|ref|CHECK|profiler|cuda_event|speedup"

echo "=== RETEST_V2_DONE ==="
