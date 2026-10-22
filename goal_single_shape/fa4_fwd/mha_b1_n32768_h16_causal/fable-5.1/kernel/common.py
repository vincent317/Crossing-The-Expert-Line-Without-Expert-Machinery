"""Shared problem definition + measurement utilities.

Problem: FlashAttention forward, varlen packed causal MHA.
13 documents packed into T=32768 tokens, H=16, head_dim=128, bf16.
"""
import json
import math
import os
import tempfile

import torch

SEQLENS = [4096, 217, 8192, 1500, 513, 2800, 6144, 700, 1777, 257, 3500, 1024, 2048]
H = 16
D = 128
MAX_SEQLEN = 8192
DTYPE = torch.bfloat16
SOFTMAX_SCALE = 1.0 / math.sqrt(D)
T = sum(SEQLENS)


def make_inputs(device="cuda", seed=0):
    """Deterministic inputs, identical across all scripts (CPU generator)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(T, H, D, generator=g, dtype=torch.float32)
    k = torch.randn(T, H, D, generator=g, dtype=torch.float32)
    v = torch.randn(T, H, D, generator=g, dtype=torch.float32)
    q = q.to(device=device, dtype=DTYPE)
    k = k.to(device=device, dtype=DTYPE)
    v = v.to(device=device, dtype=DTYPE)
    cu = torch.zeros(len(SEQLENS) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(SEQLENS, dtype=torch.int32).cumsum(0)
    cu = cu.to(device=device)
    return q, k, v, cu


def naive_fp32(q, k, v):
    """fp32 naive causal varlen attention reference (slow, chunked)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.empty(T, H, D, dtype=torch.float32, device=q.device)
    off = 0
    QCHUNK = 512
    for L in SEQLENS:
        qs = qf[off:off + L].transpose(0, 1)  # H, L, D
        ks = kf[off:off + L].transpose(0, 1)
        vs = vf[off:off + L].transpose(0, 1)
        for i0 in range(0, L, QCHUNK):
            i1 = min(i0 + QCHUNK, L)
            s = torch.matmul(qs[:, i0:i1], ks.transpose(1, 2)) * SOFTMAX_SCALE  # H, m, L
            ar = torch.arange(i0, i1, device=q.device).unsqueeze(1)
            ac = torch.arange(L, device=q.device).unsqueeze(0)
            s = s.masked_fill(ac > ar, float("-inf"))
            p = torch.softmax(s, dim=-1)
            o = torch.matmul(p, vs)  # H, m, D
            out[off + i0:off + i1] = o.transpose(0, 1)
        off += L
    return out


def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


class L2Flusher:
    def __init__(self, mb=256):
        self.buf = torch.empty(mb * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")

    def flush(self):
        self.buf.zero_()


# Kernel-name substrings belonging to the L2 flush, excluded from the timed span.
FLUSH_KERNEL_MARKERS = ("FillFunctor", "vectorized_elementwise_kernel", "Memset")


def _is_flush_kernel(name):
    return any(m in name for m in FLUSH_KERNEL_MARKERS)


def cuda_event_median(fn, warmup=10, iters=50, flush=True):
    """Median per-iteration time via CUDA events, L2 flushed before each iter."""
    fl = L2Flusher() if flush else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if fl:
            fl.flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def profiler_kernel_span(fn, warmup=10, iters=50, flush=True, tag=""):
    """Median per-iteration GPU kernel span (first kernel start -> last kernel end).

    Only cat=='kernel' trace events are considered; L2 flush uses memset so it is
    excluded automatically.
    """
    fl = L2Flusher() if flush else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            if fl:
                fl.flush()
            fn()
        torch.cuda.synchronize()
    path = os.path.join(tempfile.gettempdir(), f"trace_{tag}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    evs = [e for e in tr.get("traceEvents", [])
           if e.get("ph") == "X" and e.get("cat") == "kernel"
           and not _is_flush_kernel(e.get("name", ""))]
    names = {}
    for e in evs:
        names[e["name"]] = names.get(e["name"], 0) + 1
    evs.sort(key=lambda e: e["ts"])
    n = len(evs)
    assert n % iters == 0, f"kernel count {n} not divisible by iters {iters}: {names}"
    per = n // iters
    spans = []
    for i in range(iters):
        grp = evs[i * per:(i + 1) * per]
        st = min(e["ts"] for e in grp)
        en = max(e["ts"] + e["dur"] for e in grp)
        spans.append((en - st) / 1000.0)  # us -> ms
    spans.sort()
    return spans[len(spans) // 2], per, names
