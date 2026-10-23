echo "=== probe: ex2 throughput f32 vs f16x2 ==="
nvcc -gencode arch=compute_100a,code=sm_100a -O3 probe.cu -o /tmp/probe 2>&1 | grep -E "error|warning" || echo "build ok"
/tmp/probe 2>&1
