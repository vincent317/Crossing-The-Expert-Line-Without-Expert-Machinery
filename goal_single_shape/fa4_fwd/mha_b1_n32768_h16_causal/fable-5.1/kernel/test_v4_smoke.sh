set -e
echo "=== v4 smoke build test ==="
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 fa_fwd4.cu -o /tmp/smoke_v4 2>&1 | head -30
echo "compile_exit=$?"
