set -o pipefail
NVCC_FLAGS="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17"
echo "=== [1] smoke ==="
nvcc $NVCC_FLAGS -lineinfo smoke_umma.cu -o /tmp/smoke 2>&1 | grep -v "^$" | tail -20 && /tmp/smoke 2>&1
echo "=== [2] build fa_fwd ==="
nvcc $NVCC_FLAGS -Xptxas -v --shared -Xcompiler -fPIC fa_fwd.cu -o libfa_fwd.so 2>&1 | grep -E "error|warning|registers|spill|smem" | head -30
ls -la libfa_fwd.so || exit 1
echo "=== [3] ref ==="
python -u bench_ref.py 2>&1 | grep -v "UserWarning\|exe_args\|USDT"
echo "=== [4] new ==="
timeout 600 python -u bench_new.py 2>&1 | grep -v "USDT"
echo "=== ALL_DONE ==="
