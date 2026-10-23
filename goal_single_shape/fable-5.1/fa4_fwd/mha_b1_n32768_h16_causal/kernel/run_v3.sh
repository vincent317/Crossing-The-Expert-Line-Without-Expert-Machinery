set -o pipefail
NVCC_FLAGS="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17"
echo "=== [1] build dbg v2(lazy) + v3(persistent) ==="
nvcc $NVCC_FLAGS -DNO_DEBUG_PRINT dbg_main2.cu -o /tmp/dbg2q 2>&1 | grep -E "error"
nvcc $NVCC_FLAGS -DNO_DEBUG_PRINT dbg_main3.cu -o /tmp/dbg3q 2>&1 | grep -E "error"
nvcc $NVCC_FLAGS dbg_main3.cu -o /tmp/dbg3 2>&1 | grep -E "error"
ls -la /tmp/dbg2q /tmp/dbg3q /tmp/dbg3
echo "=== [2] v2(lazy) all cases ==="
timeout 120 /tmp/dbg2q 2>&1 | head -40; echo "exit=$?"
echo "=== [3] v3 case 3 (2 docs, H2) with trace ==="
timeout 60 /tmp/dbg3 3 2>&1 | head -60; echo "exit=$?"
echo "=== [4] v3 all cases ==="
timeout 120 /tmp/dbg3q 2>&1 | head -40; echo "exit=$?"
echo "=== [5] build .so v2 and v3 ==="
nvcc $NVCC_FLAGS -Xptxas -v --shared -Xcompiler -fPIC fa_fwd2.cu -o libfa_fwd2.so 2>&1 | grep -E "error|registers|spill" | head -3
nvcc $NVCC_FLAGS -Xptxas -v --shared -Xcompiler -fPIC fa_fwd3.cu -o libfa_fwd3.so 2>&1 | grep -E "error|registers|spill" | head -3
echo "=== [6] ref ==="
python -u bench_ref.py 2>&1 | grep -E "^\[REF\]|^\[TOL\]"
echo "=== [7] v2(lazy) full check + bench ==="
FA_FWD_SO=$PWD/libfa_fwd2.so timeout 600 python -u bench_new.py 2>&1 | grep -v "USDT"
echo "=== [8] v3(persistent) full check + bench ==="
FA_FWD_SO=$PWD/libfa_fwd3.so timeout 600 python -u bench_new.py 2>&1 | grep -v "USDT"
echo "=== ALL_DONE ==="
