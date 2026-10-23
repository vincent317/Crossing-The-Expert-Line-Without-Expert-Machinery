"""Shared problem definition, timing harness and fp32 naive reference."""
import statistics
import torch

B, S, H, D = 1, 32768, 32, 128
SCALE = D ** -0.5
WARMUP = 10
ITERS = 50


def make_inputs(device="cuda", seed=1234):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, D, generator=g, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, generator=g, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, generator=g, device=device, dtype=torch.bfloat16)
    return q, k, v


def cuda_event_median(fn, warmup=WARMUP, iters=ITERS, flush_l2=True):
    """Median device time in ms. L2 is flushed outside the timed region."""
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.int8) if flush_l2 else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush is not None:
            flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return statistics.median(ts), ts[0], ts[-1]


def profiler_kernel_span(fn, iters=20):
    """Per-call CUDA kernel durations from the profiler: {kernel_name: (ms_per_call, count)}."""
    from torch.profiler import profile, ProfilerActivity
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = {}
    for evt in prof.key_averages():
        if "CUDA" not in str(evt.device_type):
            continue
        dur = getattr(evt, "self_device_time_total", 0.0)
        if dur > 0 and evt.count > 0:
            out[evt.key] = (dur / iters / 1000.0, evt.count / iters)
    return out


def report(tag, fn, iters=ITERS, warmup=WARMUP, profile_it=True):
    med, lo, hi = cuda_event_median(fn, warmup=warmup, iters=iters)
    flops = 4.0 * B * H * S * S * D
    print(f"[{tag}] cuda-events median = {med:.4f} ms (min {lo:.4f} max {hi:.4f}) "
          f"-> {flops / (med * 1e-3) / 1e12:.1f} TFLOP/s", flush=True)
    span = None
    if profile_it:
        ks = profiler_kernel_span(fn)
        span = 0.0
        for name, (ms, cnt) in sorted(ks.items(), key=lambda x: -x[1][0]):
            print(f"    kernel {name[:78]!r}: {ms:.4f} ms/call count/call={cnt:.2f}", flush=True)
            span += ms
        print(f"[{tag}] profiler kernel-only span = {span:.4f} ms "
              f"-> {flops / (span * 1e-3) / 1e12:.1f} TFLOP/s", flush=True)
    return med, span


def naive_fp32(q, k, v, scale=SCALE, q_chunk=1024):
    """Direct fp32 attention: full-row softmax, no online rescaling, TF32 disabled."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = torch.empty(B, S, H, D, device=q.device, dtype=torch.float32)
        for h in range(H):
            kh = k[0, :, h, :].float()          # [S, D]
            vh = v[0, :, h, :].float()          # [S, D]
            for lo in range(0, S, q_chunk):
                hi = min(lo + q_chunk, S)
                qh = q[0, lo:hi, h, :].float()  # [c, D]
                scores = (qh @ kh.T) * scale    # [c, S] fp32
                p = torch.softmax(scores, dim=-1)
                out[0, lo:hi, h, :] = p @ vh
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item()
