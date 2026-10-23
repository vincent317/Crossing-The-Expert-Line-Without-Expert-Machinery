set -e
echo "=== [1] build v4 (QTILE=128 KTILE=128) ==="
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -Xptxas -v --shared -Xcompiler -fPIC fa_fwd4.cu -o libfa_fwd4.so 2>&1 | grep -E "error|registers|smem" || echo "build ok"
ls -lh libfa_fwd4.so

echo "=== [2] check v4 correctness ==="
python -u check_v4.py 2>&1

echo "=== [3] bench v4 device time ==="
python -u bench_v4.py 2>&1

echo "=== [4] ref device time (for comparison) ==="
python -u bench_ref.py 2>&1 | grep -E "device_time|profiler_kernel"

echo "=== ALL_DONE ==="
