set -o pipefail
NVCC_FLAGS="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17"
echo "=== [1] build dbg v1 (fixed) and v2 ==="
nvcc $NVCC_FLAGS dbg_main.cu -o /tmp/dbg1 2>&1 | grep -E "error" 
nvcc $NVCC_FLAGS dbg_main2.cu -o /tmp/dbg2 2>&1 | grep -E "error"
nvcc $NVCC_FLAGS -DNO_DEBUG_PRINT dbg_main2.cu -o /tmp/dbg2q 2>&1 | grep -E "error"
ls -la /tmp/dbg1 /tmp/dbg2 /tmp/dbg2q
echo "=== [2] v1 dbg L=384 ==="
timeout 60 /tmp/dbg1 384 2>&1 | head -60; echo "exit=$?"
echo "=== [3] v2 dbg case 1 (L384) with trace ==="
timeout 60 /tmp/dbg2 1 2>&1 | head -80; echo "exit=$?"
echo "=== [4] v2 all cases (quiet) ==="
timeout 120 /tmp/dbg2q 2>&1 | grep -v "^\[" | head -40; echo "exit=$?"
echo "=== [5] build libfa_fwd.so from v2 ==="
nvcc $NVCC_FLAGS -Xptxas -v --shared -Xcompiler -fPIC fa_fwd2.cu -o libfa_fwd.so 2>&1 | grep -E "error|registers|spill" | head
echo "=== [6] v2 full check + bench ==="
python -u bench_ref.py 2>&1 | grep -E "^\[REF\]|^\[TOL\]"
timeout 900 python -u bench_new.py 2>&1 | grep -v "USDT"
echo "=== ALL_DONE ==="
