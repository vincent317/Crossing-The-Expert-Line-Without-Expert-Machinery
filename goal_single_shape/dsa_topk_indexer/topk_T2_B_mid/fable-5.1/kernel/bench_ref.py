"""Benchmark the reference implementation device time on B200."""
import torch
import numpy as np
from safetensors.torch import load_file
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ref"))
from ref_entry import run

DATA_PATH = os.path.join(os.path.dirname(__file__), "ref", "data",
    "81a953ea-f29f-4707-8456-67a3624a549d.safetensors")

def make_inputs(device="cuda"):
    data = load_file(DATA_PATH)
    seq_lens = data["seq_lens"].to(device)        # (B,) int32
    block_table = data["block_table"].to(device)  # (B, max_pages) int32

    B = seq_lens.shape[0]
    H, D = 64, 128
    num_pages = 11923
    page_size = 64

    # q: (B, H, D) float8_e4m3fn
    q_fp8 = torch.randn(B, H, D, device=device, dtype=torch.float32).clamp(-448, 448).to(torch.float8_e4m3fn)

    # k cache: (num_pages, page_size, 1, 132) int8
    # Layout per page-token: 128 bytes fp8 + 4 bytes fp32 scale
    # We build as raw int8 buffer
    raw_k = torch.zeros(num_pages, page_size, 1, 132, dtype=torch.int8, device=device)
    k_bf16 = torch.randn(num_pages, page_size, D, device=device, dtype=torch.bfloat16)
    k_fp32 = k_bf16.float()
    # per-token scale = absmax / 448
    scale = k_fp32.abs().amax(dim=-1, keepdim=True) / 448.0  # (num_pages, page_size, 1)
    scale = scale.clamp(min=1e-8)
    k_quantized = (k_fp32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    # Pack into raw_k: first 128 bytes = fp8 data, last 4 bytes = fp32 scale
    raw_k[:, :, 0, :128] = k_quantized.view(num_pages, page_size, 128).view(torch.int8)
    scale_int8 = scale.squeeze(-1).float().view(torch.int8).reshape(num_pages, page_size, 4)
    raw_k[:, :, 0, 128:] = scale_int8

    weights = torch.randn(B, H, device=device, dtype=torch.float32)

    return q_fp8, raw_k, weights, seq_lens, block_table


def benchmark_ref(warmup=10, iters=50):
    print("=== Benchmarking ref implementation ===")
    device = "cuda"
    torch.cuda.set_device(0)

    q_fp8, raw_k, weights, seq_lens, block_table = make_inputs(device)

    # Warmup
    for _ in range(warmup):
        out = run(q_fp8, raw_k, weights, seq_lens, block_table)
    torch.cuda.synchronize()

    # Flush L2 cache (512 MB)
    l2_flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=device)

    # Use torch profiler for kernel-only span
    times_cuda = []
    for _ in range(iters):
        l2_flush.fill_(0.0)  # flush L2
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = run(q_fp8, raw_k, weights, seq_lens, block_table)
        end.record()
        torch.cuda.synchronize()
        times_cuda.append(start.elapsed_time(end) * 1000)  # us

    times_cuda = sorted(times_cuda)
    median_us = times_cuda[len(times_cuda)//2]
    print(f"[REF] CUDA events median: {median_us:.2f} us (over {iters} iters, with L2 flush)")
    print(f"[REF] min={min(times_cuda):.2f} us  max={max(times_cuda):.2f} us")

    # Also use torch.profiler for kernel-only device time
    import torch.profiler as profiler
    l2_flush.fill_(0.0)
    torch.cuda.synchronize()

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            l2_flush.fill_(0.0)
            out = run(q_fp8, raw_k, weights, seq_lens, block_table)
    torch.cuda.synchronize()

    # Sum kernel durations
    kernel_us_list = []
    for evt in prof.key_averages():
        if evt.device_time > 0 and "aten" not in evt.key and "fill" not in evt.key.lower():
            pass

    # Get all individual events
    events = prof.profiler.function_events
    # filter to cuda events only (no fill)
    cuda_evts = [e for e in events if e.device_time > 0 and "fill" not in e.name.lower()]
    # group by iteration: events come in batches of 5
    # just take avg of all
    total_device_us = sum(e.device_time for e in cuda_evts)
    n_iters_prof = 5
    per_iter_us = total_device_us / n_iters_prof
    print(f"[REF] torch.profiler kernel device time per iter: {per_iter_us:.2f} us")

    return median_us

if __name__ == "__main__":
    median = benchmark_ref()
    print(f"\n[REF] Final reported device time (CUDA events median): {median:.2f} us")
