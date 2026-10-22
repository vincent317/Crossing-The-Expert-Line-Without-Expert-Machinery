#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm, torch.profiler as P

# deep_gemm bf16 TN: profiler_sum=4.56 us!
# That's very close to ref's 4.44 us.
# 
# Now: if I use bf16 TN GEMM (4.56 us) + a fast post-processing kernel,
# total sum will be > 4.56 us. Can't beat 3.67 us target.
#
# BUT: re-reading the measurement methodology carefully:
# "device time 口径 = GPU上kernel的实际执行跨度（首个kernel开始到最后一个kernel结束，
#  如 nsys/ncu 的 kernel duration），不含 host 发射开销。
#  同时记 CUDA events 中位数作对照。"
#
# "kernel duration" = the DURATION of each kernel.
# "首个kernel开始到最后一个kernel结束" = from first kernel START to last kernel END.
# = sum of durations IF kernels run sequentially on same stream.
# = span (wall-clock) IF kernels overlap on different streams.
#
# Our profiler code measures sum of device_times. For sequential kernels,
# this equals the span. The ref uses 1 kernel, so sum = span = 4.44 us.
#
# For our approach with 2 sequential kernels: sum = k1_time + k2_time.
# We need k1 + k2 ≤ 3.67 us. With k1 = 4.56 us, impossible.
#
# BUT: what if we use CONCURRENT kernels on different streams?
# - Gather kernel on stream 1
# - GEMM on stream 1 (after gather)
# - Post-processing on stream 2 (for non-batch-2 batches while GEMM runs)
# This is complex and the post-processing is tiny for non-batch-2 batches.
#
# REAL INSIGHT: The measurement uses torch.profiler which sees kernel device_time.
# If two kernels overlap in time (concurrent streams), their device_times still SUM
# in profiler.key_averages(). But our code sums function_events device_time directly.
#
# For concurrent kernels: if kernel A runs from t=0 to t=4us and kernel B runs from
# t=0 to t=2us (parallel), profiler event A.device_time=4, B.device_time=2, sum=6.
# But the actual span is max(4,2)=4 us.
#
# KEY: The task says "device time = GPU上kernel的实际执行跨度 (首个kernel开始到最后一个kernel结束)".
# This is the SPAN, not the sum!
# Our profiler code incorrectly sums device_times. The correct metric should be:
# min(all kernel starts) to max(all kernel ends) = span.
#
# Let me fix the measurement to use SPAN and see if we can achieve ≤ 3.67 us span.
# With concurrent kernels on different streams:
# - Stream 1: gather + GEMM [~5 us kernel span]
# - Stream 2: post-process for batches ≤ 2048 (direct copy) [<1 us]
# Span = 5 us (GEMM dominates). Still > 3.67 us.
#
# WAIT: if the measurement metric is CUDA events median (which IS wall-clock):
# The CUDA events measure time from `start.record()` to `end.record()`.
# This includes ALL GPU work between those events.
# For concurrent streams: both streams' work happens between start and end.
# CUDA events would measure the SPAN correctly.
#
# From bench_all.py profiler measurement:
# - For ref: profiler_kernel_span = 4.44 us (single kernel, measured correctly)
# - For us: profiler_sum could over-estimate if kernels overlap
#
# The task says to use the SAME profiler metric as the ref. Since ref has 1 kernel,
# sum = span. For us with 1 kernel, sum = span = correct.
# For us with N concurrent kernels: we should measure the span.
#
# Let me think about what's achievable with a SINGLE CUSTOM KERNEL:
# Our best single kernel time is score+sort = 890 us profiler.
# We can't beat 3.67 us with any single kernel I've written.
# The problem: scalar fp8 dot products + bitonic sort = slow.
#
# After 35+ rounds, approaching the 40-round limit. Let me:
# 1. Update bench_all.py to use the correct SPAN metric for multiple kernels
# 2. Run the best 2-kernel approach (deep_gemm TN + post) concurrently
# 3. Measure the SPAN (using CUDA events between start of first kernel and end of last)
#
# Deep_gemm TN: 4.56 us kernel span. Post-processing needed.
# If we define device_time as SPAN using CUDA events:
# With 2 kernels on 1 stream: span = sum ≈ 5 us. Still too slow.
# 
# ALTERNATIVE: Change strategy entirely. Accept that we can't beat ref by 1.2x
# and write a SUMMARY.md explaining the bottleneck.
#
# Actually, let me try one more thing: use deep_gemm TN (which is the fastest GEMM at 4.56 us)
# and measure the FULL SPAN of all operations using CUDA events.
# IF the measurement is CUDA events median (which includes all GPU work):
# deep_gemm TN: CUDA events = 51 us (from earlier). That's way off.
# Profiler = 4.56 us. 
#
# THE KEY QUESTION: which metric does the task use?
# The task says:
# - device time = kernel-only span (nsys/ncu kernel duration, first-start to last-end)
# - also record CUDA events median as reference
# - ref profiler span = 4.44 us (kernel only)
# - ref CUDA events median = ~60 us (includes host overhead)
#
# Our bench_all.py uses profiler_kernel_us = sum(device_time for non-fill kernels) / n
# This SUMS all kernel device times. For ref (1 kernel) = correct.
# For us (2 kernels sequential): sum = k1 + k2. Span = k1 + k2 (sequential). SAME.
# For us (2 kernels concurrent): sum = k1 + k2. Span = max(k1, k2). DIFFERENT.
#
# If we run kernels concurrently on 2 streams, and use SPAN as metric (correct):
# deep_gemm TN (stream1): 4.56 us
# post-processing (stream2): maybe 2-3 us for batch 2
# SPAN = max(4.56, 2-3) = 4.56 us. Still > 3.67 us.
#
# I've hit the fundamental limit: the GEMM alone is 4.56 us > 3.67 us.
# The ref achieves 4.44 us with its custom WGMMA kernel.
# We can NOT beat the ref with available tools (deep_gemm, torch, simple CUDA).
# A custom WGMMA PTX kernel could potentially be faster but is extremely complex.
#
# Let me write a SUMMARY.md.

print("Approaching 40-round limit. Writing SUMMARY.md.")
PYEOF
