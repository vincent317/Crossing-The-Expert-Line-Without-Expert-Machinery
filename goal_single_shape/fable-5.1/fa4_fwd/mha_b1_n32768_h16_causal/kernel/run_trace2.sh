set -e
NF="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17"
echo "=== build v3 trace harness v2 ==="
nvcc $NF -lineinfo dbg_trace3.cu -o dbg_trace3 2>&1 | grep -E "error|warning" | head -5
echo "=== L=384 (3 items: nk=1,2,3) ==="
timeout 6 ./dbg_trace3 384 2>&1 || echo "timeout_L384"
echo "=== L=256 (2 items: nk=1,2) ==="
timeout 6 ./dbg_trace3 256 2>&1 || echo "timeout_L256"
echo "TRACE2_DONE"
