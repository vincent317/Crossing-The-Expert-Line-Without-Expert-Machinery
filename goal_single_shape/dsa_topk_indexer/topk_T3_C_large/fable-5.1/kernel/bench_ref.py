"""Benchmark ref device time using torch profiler kernel-only span + CUDA events."""
import torch
import sys
sys.path.insert(0, '/workspace')

from safetensors.torch import load_file
import os

# Load real data
DATA_PATH = './ref/data/8635db8f-5b62-464e-9b5b-0863d65788ee.safetensors'
data = load_file(DATA_PATH)
seq_lens = data['seq_lens'].cuda()
block_table = data['block_table'].cuda()

B = seq_lens.shape[0]
print(f"B={B}, seq_lens={seq_lens.cpu().tolist()}")
print(f"block_table shape: {block_table.shape}")

# Generate random inputs
torch.manual_seed(42)
q_index_fp8 = torch.randn(B, 64, 128, device='cuda').to(torch.float8_e4m3fn)

# Build k cache: num_pages=11923, each page has 64 tokens
# Layout: (num_pages, 64, 1, 132) int8
# First 128 bytes per token = fp8 data, last 4 bytes = fp32 scale
num_pages = 11923
# Generate random bf16 k values, quantize to fp8 + scale
k_bf16 = torch.randn(num_pages, 64, 128, device='cuda', dtype=torch.bfloat16)
k_abs_max = k_bf16.abs().amax(dim=-1, keepdim=True).float()  # (num_pages, 64, 1)
scale = k_abs_max / 448.0  # per-token scale
k_fp8_vals = (k_bf16.float() / (scale + 1e-12)).clamp(-448, 448).to(torch.float8_e4m3fn)

# Pack into (num_pages, 64, 1, 132) int8
# 128 bytes fp8 + 4 bytes float32 scale = 132 bytes per token
k_index_cache_fp8 = torch.zeros(num_pages, 64, 1, 132, dtype=torch.int8, device='cuda')
k_index_cache_fp8[:, :, 0, :128] = k_fp8_vals.view(torch.int8)
# Store scale as float32 (4 bytes) in last 4 bytes
scale_flat = scale.squeeze(-1).float()  # (num_pages, 64)
k_index_cache_fp8[:, :, 0, 128:132] = scale_flat.view(torch.uint8).view(num_pages, 64, 4).view(torch.int8)

weights = torch.randn(B, 64, device='cuda', dtype=torch.float32)

# Import ref
from ref.ref_entry import run as ref_run

# Warmup
print("Warming up ref...")
for _ in range(10):
    out = ref_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
torch.cuda.synchronize()

# Flush L2 (512MB)
l2_flush = torch.zeros(512 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')

# Benchmark with CUDA events
ITERS = 50
times_events = []
for _ in range(ITERS):
    # Flush L2
    l2_flush.zero_()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    out = ref_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
    end_ev.record()
    torch.cuda.synchronize()
    times_events.append(start_ev.elapsed_time(end_ev) * 1000)  # µs

import statistics
events_median = statistics.median(times_events)
print(f"[REF] CUDA events median: {events_median:.2f} µs")

# Benchmark with torch profiler (kernel-only span)
print("Running torch profiler for kernel-only device time...")
# Flush L2 first
l2_flush.zero_()
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
    with_stack=False,
) as prof:
    for _ in range(10):
        l2_flush.zero_()
        torch.cuda.synchronize()
        out = ref_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
    torch.cuda.synchronize()

# Extract kernel durations
events = prof.key_averages()
total_cuda_time = 0
kernel_details = []
for ev in events:
    if ev.device_time > 0:
        kernel_details.append((ev.key, ev.device_time, ev.count))
        total_cuda_time += ev.device_time

print("Kernel breakdown (total across all iterations):")
for key, dt, count in sorted(kernel_details, key=lambda x: -x[1]):
    print(f"  {key}: {dt:.1f} µs total, {dt/count:.2f} µs/call, count={count}")

print(f"\n[REF] Profiler total device time per iter (10 iters): {total_cuda_time/10:.2f} µs")
print(f"[REF] CUDA events median: {events_median:.2f} µs")
print(f"Output shape: {out.shape}, dtype: {out.dtype}")
print(f"Sample output: {out[0, :10]}")
