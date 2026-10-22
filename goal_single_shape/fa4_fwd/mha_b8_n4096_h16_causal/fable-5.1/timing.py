"""Shared timing utilities: device-time measurement via CUDA events and profiler.

Both paths are kernel-only (no host launch overhead):
  - cuda_events_median: per-iteration GPU span, L2 flushed between iterations.
  - profiler_kernel_time: sum of device kernel durations / iters.
"""
import torch
from torch.profiler import profile, ProfilerActivity

_FLUSH_BYTES = 384 * 1024 * 1024  # > B200 L2 (~126MB)
_flush_buf = None


def _flush_l2():
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    _flush_buf.zero_()


def cuda_events_median(fn, warmup=10, iters=50, flush=True):
    """Median per-iteration device time in ms, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if flush:
            _flush_l2()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], times[0]


_SKIP = ("Memcpy", "Memset", "memcpy", "memset")


def profiler_kernel_time(fn, warmup=10, iters=50):
    """Sum of all device kernel durations / iters (ms), plus per-kernel breakdown."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = []
    total_us = 0.0
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", 0.0) or 0.0
        if t <= 0:
            continue
        if any(s in e.key for s in _SKIP):
            continue
        total_us += t
        rows.append((e.key, t / iters, e.count / iters))
    rows.sort(key=lambda r: -r[1])
    return total_us / 1000.0 / iters, rows
