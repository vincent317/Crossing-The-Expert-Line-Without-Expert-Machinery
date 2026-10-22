"""Shared data generation, timing, and accuracy helpers.

Single source of truth so that ref and custom kernels are measured with
identical inputs, identical iteration counts and identical timing method.
"""

import json
import os
import statistics
import tempfile

import torch

B, S, H, D = 8, 512, 16, 128
SOFTMAX_SCALE = D ** -0.5
WARMUP = int(os.environ.get("FMHA_WARMUP", "10"))
ITERS = int(os.environ.get("FMHA_ITERS", "50"))


def make_inputs(device="cuda", dtype=torch.bfloat16, seed=1234, seqlen=S):
    """q/k/v with layout (B, S, H, D)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    shape = (B, seqlen, H, D)
    q = torch.randn(shape, generator=g, device=device, dtype=torch.float32).to(dtype)
    k = torch.randn(shape, generator=g, device=device, dtype=torch.float32).to(dtype)
    v = torch.randn(shape, generator=g, device=device, dtype=torch.float32).to(dtype)
    return q, k, v


def attn_fp32_naive(q, k, v):
    """Naive fp32 attention reference, (B,S,H,D) in and out. Slow on purpose."""
    qf = q.float().permute(0, 2, 1, 3)  # B,H,S,D
    kf = k.float().permute(0, 2, 1, 3)
    vf = v.float().permute(0, 2, 1, 3)
    out = torch.empty_like(qf)
    for b in range(qf.shape[0]):
        for h in range(qf.shape[1]):
            scores = (qf[b, h] @ kf[b, h].t()) * SOFTMAX_SCALE
            p = torch.softmax(scores, dim=-1)
            out[b, h] = p @ vf[b, h]
    return out.permute(0, 2, 1, 3).contiguous()


def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


class L2Flusher:
    def __init__(self, mb=256):
        self.buf = torch.empty(mb * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

    def flush(self):
        self.buf.zero_()


def time_cuda_events(fn, warmup=WARMUP, iters=ITERS):
    """Median per-iteration device time in us, measured with CUDA events,
    L2 flushed before every timed iteration."""
    flusher = L2Flusher()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flusher.flush()
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()
    times = [starts[i].elapsed_time(stops[i]) * 1e3 for i in range(iters)]
    return statistics.median(times), times


def time_profiler_kernel_span(fn, warmup=WARMUP, iters=ITERS, tag=""):
    """Median per-iteration kernel-only span in us, from the profiler trace.

    Per iteration: span = (max kernel end) - (min kernel start) across the
    kernels launched by fn(). Host launch overhead is excluded by construction.
    L2 is flushed between iterations; the flush kernel is filtered out by name.
    """
    flusher = L2Flusher()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    marks = []
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            flusher.flush()
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()
    torch.cuda.synchronize()

    path = os.path.join(tempfile.mkdtemp(), "trace.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        trace = json.load(f)

    flush_names = ("vectorized_elementwise", "elementwise_kernel", "fill", "zero", "Memset")
    kernels = []
    for ev in trace.get("traceEvents", []):
        if ev.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        name = ev.get("name", "")
        if any(fn_name.lower() in name.lower() for fn_name in flush_names):
            continue
        kernels.append((float(ev["ts"]), float(ev["ts"]) + float(ev["dur"]), name))
    kernels.sort()
    if not kernels:
        raise RuntimeError("no kernels captured in profiler trace")

    # Group kernels into iterations: every kernel whose start is within a gap
    # of the previous kernel's end belongs to the same iteration. Gaps between
    # iterations are large because of the L2 flush + sync in between.
    groups = []
    cur = [kernels[0]]
    for kev in kernels[1:]:
        if kev[0] - cur[-1][1] > 5.0:  # us
            groups.append(cur)
            cur = [kev]
        else:
            cur.append(kev)
    groups.append(cur)

    spans = [max(e for _, e, _ in g) - min(s for s, _, _ in g) for g in groups]
    names = sorted({n for g in groups for _, _, n in g})
    counts = statistics.median([len(g) for g in groups])
    print(f"[profiler{(' ' + tag) if tag else ''}] groups={len(groups)} "
          f"kernels_per_group(median)={counts} kernel_names={names[:4]}")
    return statistics.median(spans), spans


def report(label, q, k, v, out, ref_fp32):
    err = max_abs_err(out, ref_fp32)
    print(f"[{label}] max_abs_err vs fp32_naive = {err:.6e}")
    return err
