set -o pipefail
NVCC_FLAGS="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17"
echo "=== [0] mma_bench ==="
nvcc $NVCC_FLAGS mma_bench.cu -o /tmp/mb 2>&1 | grep error; /tmp/mb 2>&1
echo "=== [1] dbg v3 all cases ==="
nvcc $NVCC_FLAGS -DNO_DEBUG_PRINT dbg_main3.cu -o /tmp/dbg3q 2>&1 | grep -E "error"
timeout 120 /tmp/dbg3q 2>&1 | grep -E "DBG_RESULT|ALL_CASES|error|rc=" ; echo "exit=$?"
echo "=== [2] build v3 modes ==="
for M in 0 1 2 3 9; do nvcc $NVCC_FLAGS -DFA_MODE=$M -Xptxas -v --shared -Xcompiler -fPIC fa_fwd3.cu -o libfa_fwd3_m$M.so 2>&1 | grep -E "error|registers|spill" | head -2; done
echo "=== [3] ref ==="
python -u bench_ref.py 2>&1 | grep -E "^\[REF\]|^\[TOL\]"
echo "=== [4] v3 mode 0 (full) check + bench ==="
FA_FWD_SO=$PWD/libfa_fwd3_m0.so timeout 600 python -u bench_new.py 2>&1 | grep -E "CHECK|NEW|doc  [0-9]|DONE|Error|error"
for M in 1 2 3; do
echo "=== [5] v3 mode $M bench only ==="
FA_FWD_SO=$PWD/libfa_fwd3_m$M.so timeout 600 python -u bench_new.py 2>&1 | grep -E "^\[NEW\]|Error|error"
done
echo "=== [6] v3 mode 9 profile ==="
FA_FWD_SO=$PWD/libfa_fwd3_m9.so timeout 300 python -u prof_dump.py 2>&1 | grep -v USDT
echo "=== ALL_DONE ==="
