"""Shared timing utilities: CUDA-event median and profiler kernel-only span."""
import json
import os
import statistics
import tempfile

import torch

_FLUSH_BUF = None


def l2_flush():
    global _FLUSH_BUF
    if _FLUSH_BUF is None:
        _FLUSH_BUF = torch.empty(int(256e6 // 4), dtype=torch.float32, device="cuda")
    _FLUSH_BUF.zero_()


def cuda_event_median_ms(fn, warmup=10, iters=50, flush=True):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush:
            l2_flush()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(starts[i].elapsed_time(ends[i]) for i in range(iters))


def profiler_kernel_span_ms(fn, warmup=10, iters=20, tag=""):
    """Median per-iteration span from first kernel start to last kernel end (device time)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    trace_path = os.path.join(tempfile.gettempdir(), f"trace_{tag or 'x'}.json")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    with open(trace_path) as f:
        trace = json.load(f)
    kernels = [
        e
        for e in trace["traceEvents"]
        if e.get("cat") in ("kernel", "Kernel") and e.get("ph") == "X"
    ]
    kernels.sort(key=lambda e: e["ts"])
    names = {}
    for k in kernels:
        names[k["name"]] = names.get(k["name"], 0) + 1
    print(f"[profiler:{tag}] kernel launches over {iters} iters:")
    for n, c in sorted(names.items(), key=lambda kv: -kv[1]):
        print(f"    {c:5d}  {n[:110]}")
    n_per_iter = len(kernels) // iters
    assert n_per_iter >= 1, "no kernels captured"
    assert len(kernels) == n_per_iter * iters, (
        f"kernel count {len(kernels)} not divisible by iters {iters}"
    )
    spans = []
    for i in range(iters):
        grp = kernels[i * n_per_iter : (i + 1) * n_per_iter]
        first = min(k["ts"] for k in grp)
        last = max(k["ts"] + k["dur"] for k in grp)
        spans.append((last - first) / 1000.0)  # us -> ms
    spans_sorted = sorted(spans)
    print(
        f"[profiler:{tag}] kernels/iter={n_per_iter} "
        f"min={spans_sorted[0]:.4f} med={statistics.median(spans):.4f} "
        f"max={spans_sorted[-1]:.4f} ms"
    )
    return statistics.median(spans)
