"""Device-time measurement helpers (CUDA events, optional CUDA graph capture)."""

import torch


def device_time_us(fn, iters=50, warmup=15, reps=5, graph=True):
    """Median per-call device time in microseconds.

    CUDA events bracket `iters` back-to-back calls, so the measured span is the
    GPU-side execution span (host launch cost is pipelined away / removed
    entirely when graph=True).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    run = fn
    if graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        torch.cuda.synchronize()
        for _ in range(5):
            gr.replay()
        torch.cuda.synchronize()
        run = gr.replay

    ts = []
    for _ in range(reps):
        beg = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        beg.record()
        for _ in range(iters):
            run()
        end.record()
        torch.cuda.synchronize()
        ts.append(beg.elapsed_time(end) / iters * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts
