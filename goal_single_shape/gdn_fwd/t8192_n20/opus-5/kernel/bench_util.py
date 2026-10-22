"""Shared device-time measurement: CUDA-event span around one full invocation.

The span covers the first kernel's start to the last kernel's end on the GPU
(host launch overhead is excluded because the events are recorded on the same
stream and only GPU timestamps are compared).  Median over many iterations.
"""

import statistics
import torch


def device_time_ms(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    ev0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ev1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        ev0[i].record()
        fn()
        ev1[i].record()
    torch.cuda.synchronize()
    for i in range(iters):
        samples.append(ev0[i].elapsed_time(ev1[i]))
    samples.sort()
    return statistics.median(samples), samples[0], samples[len(samples) // 10]


def device_time_amortized_ms(fn, warmup=20, iters=200):
    """Back-to-back launches, one event pair around the whole run.

    Divides out host launch overhead, so this is the friendlier (lower) number
    for implementations that issue many kernels per call.  Reported alongside
    the per-call span so both sides are compared on identical terms.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def maxerr(x, y):
    return (x.float() - y.float()).abs().max().item()
