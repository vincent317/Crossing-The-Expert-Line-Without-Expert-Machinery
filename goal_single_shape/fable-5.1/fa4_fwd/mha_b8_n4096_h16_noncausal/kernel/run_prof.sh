#!/bin/bash
export PATH=$PATH:/usr/local/cuda/bin
echo "=== probe build (verbose) ==="
nvcc -gencode=arch=compute_100a,code=sm_100a -O3 -o /tmp/probe_umma probe_umma.cu
echo "probe_build_rc=$?"
ls -la /tmp/probe_umma
echo "=== probe run ==="
/tmp/probe_umma; echo "probe_run_rc=$?"
echo "=== which ncu ==="
which ncu || ls /usr/local/cuda/bin/ | grep -i ncu
export VLAYOUT=1 BLOCK_N=64 RESCALE_THRESH=8.0f USE_CPASYNC=1
echo "=== ncu on custom kernel (BLOCK_N=64 lazy cp.async) ==="
ncu --kernel-name fmha_fwd --launch-count 1 \
  --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  python -u prof_mine.py 2>&1 | tail -45
