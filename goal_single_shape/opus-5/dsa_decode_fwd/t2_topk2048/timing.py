"""Device-time measurement helpers.

`device time` here = the GPU-side span of the op's kernels, with host launch
overhead removed. Two independent estimators, both applied identically to the
reference and to the custom kernel:

  * `graph_time_us`  - capture the op into a CUDA graph, replay it `inner`
    times inside one CUDA-event window, divide by `inner`. Host launch cost is
    paid once at capture, so the window is pure GPU time.
  * `kernel_breakdown` - CUDA-kernel durations straight out of the profiler
    (same quantity nsys/ncu call "kernel duration").
"""

import torch


def _capture(fn, inner):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    return graph


def graph_time_us(fn, inner=20, warmup=10, iters=60):
    graph = _capture(fn, inner)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / inner)
    samples.sort()
    return samples[len(samples) // 2]


def kernel_breakdown(fn, iters=50, warmup=20):
    """Return (total_us_per_iter, [(kernel_name, launches_per_iter, us_per_iter)])."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = []
    total = 0.0
    for e in prof.key_averages():
        dt = float(e.self_device_time_total)
        if dt <= 0:
            continue
        rows.append((e.key[:70], e.count / iters, dt / iters))
        total += dt / iters
    rows.sort(key=lambda r: -r[2])
    return total, rows


def kernel_span_us(fn, iters=50, warmup=20):
    """Per-invocation span: first kernel start -> last kernel end.

    Identical to `kernel_breakdown`'s total for a single-kernel op; for a
    multi-kernel op it additionally contains the inter-kernel gaps, which is
    exactly the "first kernel start to last kernel end" definition.
    """
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    evts = []
    for e in prof.events():
        if float(getattr(e, "self_device_time_total", 0.0)) <= 0:
            continue
        for k in getattr(e, "kernels", []) or []:
            evts.append((k.interval.start, k.interval.end))
    if not evts:
        return float("nan"), 0
    evts.sort()
    nk = max(1, len(evts) // iters)
    spans = []
    for i in range(0, len(evts) - nk + 1, nk):
        group = evts[i : i + nk]
        spans.append(max(e[1] for e in group) - min(e[0] for e in group))
    spans.sort()
    return spans[len(spans) // 2], nk


def report(tag, fn, inner=20):
    t_amort = graph_time_us(fn, inner=inner)
    t_single = graph_time_us(fn, inner=1)
    total, rows = kernel_breakdown(fn)
    print(f"[{tag}] graph-amortized {t_amort:.3f} us | graph-single-shot {t_single:.3f} us "
          f"| profiler kernel-sum {total:.3f} us")
    for name, cnt, us in rows:
        print(f"    {us:8.3f} us  x{cnt:.1f}  {name}")
    return t_amort, total
