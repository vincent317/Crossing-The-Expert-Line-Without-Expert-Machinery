set -e
NF="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17"
echo "=== build v3 trace harness ==="
nvcc $NF -lineinfo dbg_trace3.cu -o dbg_trace3 2>&1 | grep -E "error|warning" | head -10
ls -lh dbg_trace3
echo "=== run L=128 (expected 1 work-item, nk=1) ==="
./dbg_trace3 128 2>&1
echo "=== run L=384 (expected 3 work-items) ==="
timeout 6 ./dbg_trace3 384 2>&1 || echo "timed_out"
echo "TRACE_RUNS_DONE"
