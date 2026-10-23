"""Shared device-time measurement: CUDA events around one call, cold L2.

Each iteration first overwrites a 1 GiB buffer (evicts L2 and keeps the GPU
busy long enough to hide host launch overhead), then records start/end events
around a single call. Reports the median over `iters` iterations in µs.
"""

import statistics
import warnings

import torch

FLUSH_BYTES = 1 << 30


def make_flush_buffer(device="cuda"):
    return torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=device)


def time_call(fn, flush_buf, warmup=10, iters=100, pre=None):
    """`pre` (optional) runs after the L2 flush and before the start event, untimed."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(iters):
        flush_buf.zero_()
        if pre is not None:
            pre()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times_us), min(times_us), times_us


def time_call_graph(fn, flush_buf, warmup=10, iters=100):
    """Same measurement, but the call is captured once into a CUDA graph and replayed.

    Raises RuntimeError if the callable launched nothing on the capturing stream
    (torch warns "The CUDA Graph is empty"); replaying such a graph measures nothing.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.cuda.graph(graph, stream=side):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    if any("Graph is empty" in str(w.message) for w in caught):
        raise RuntimeError("callable was not captured (empty CUDA graph)")
    return time_call(graph.replay, flush_buf, warmup=warmup, iters=iters)


def report(label, med, mn, times_us):
    times_sorted = sorted(times_us)
    p10 = times_sorted[len(times_sorted) // 10]
    p90 = times_sorted[(len(times_sorted) * 9) // 10]
    print(f"[bench] {label}: median={med:.3f} us  min={mn:.3f} us  p10={p10:.3f} us  p90={p90:.3f} us  n={len(times_us)}")
