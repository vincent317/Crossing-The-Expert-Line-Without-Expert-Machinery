"""Shared timing helpers: CUDA-events median + profiler kernel-only span."""

import json

import torch

B, S, HQ, HKV, D = 4, 8192, 96, 8, 128


def bwd_flops(b=B, s=S, hq=HQ, d=D):
    """Causal attention backward matmul FLOPs (5 matmuls over the causal area)."""
    area = s * (s + 1) / 2
    return 10.0 * area * d * b * hq


def tflops(ms, flops=None):
    flops = bwd_flops() if flops is None else flops
    return flops / (ms * 1e-3) / 1e12


_FLUSH = None


def _flush_l2():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    _FLUSH.zero_()


def cuda_events_median(fn, warmup=10, iters=50, flush=True):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(True) for _ in range(iters)]
    en = [torch.cuda.Event(True) for _ in range(iters)]
    for i in range(iters):
        if flush:
            _flush_l2()
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    t = sorted(st[i].elapsed_time(en[i]) for i in range(iters))
    return t[iters // 2], t[0], t[-1]


def kernel_breakdown(fn, iters=20, label="", exclude=("FillFunctor", "vectorized_elementwise")):
    """Profiler kernel-only accounting. Returns (sum_ms_per_iter, span_ms_per_iter)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    path = f"/tmp/trace_{label or 'x'}.json"
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    ev = [e for e in tr["traceEvents"]
          if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel", "gpu_kernel")]
    keep = [e for e in ev if not any(x in e["name"] for x in exclude)]
    if not keep:
        return 0.0, 0.0
    agg = {}
    for e in keep:
        a = agg.setdefault(e["name"], [0, 0.0])
        a[0] += 1
        a[1] += e["dur"]
    print(f"--- kernel breakdown [{label}] {len(keep)/iters:.1f} kernels/iter", flush=True)
    for nm, (c, d) in sorted(agg.items(), key=lambda x: -x[1][1]):
        print(f"    {c:5d} {d/iters/1000:9.4f} ms/iter  {nm[:88]}", flush=True)
    ksum = sum(e["dur"] for e in keep) / iters / 1000
    span = (max(e["ts"] + e["dur"] for e in keep) - min(e["ts"] for e in keep)) / iters / 1000
    print(f"    kernel-sum {ksum:.4f} ms/iter   kernel-span(incl gaps) {span:.4f} ms/iter", flush=True)
    return ksum, span


def make_inputs(b=B, s=S, hq=HQ, hkv=HKV, d=D, seed=1234, requires_grad=True):
    torch.manual_seed(seed)
    q = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16, requires_grad=requires_grad)
    k = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16, requires_grad=requires_grad)
    v = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16, requires_grad=requires_grad)
    do = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16)
    return q, k, v, do
