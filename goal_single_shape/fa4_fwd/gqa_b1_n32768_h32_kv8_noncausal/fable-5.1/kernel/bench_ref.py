"""Measure device time of the reference FlashAttention-4 CuTe forward kernel on B200.

Shape: batch=1, seqlen=32768, heads=32, head_dim=128, bf16, causal=False.
Only the public `flash_attn_func` entry point is called; its source is never read.
"""
import os
import statistics
import torch

B, S, H, D = 1, 32768, 32, 128
SCALE = D ** -0.5
WARMUP = 10
ITERS = 50


def make_inputs(device="cuda"):
    g = torch.Generator(device=device).manual_seed(1234)
    q = torch.randn(B, S, H, D, generator=g, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, generator=g, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, generator=g, device=device, dtype=torch.bfloat16)
    return q, k, v


def cuda_event_median(fn, warmup=WARMUP, iters=ITERS):
    """Median device time (ms) with an L2 flush before every timed iteration."""
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.int8)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return statistics.median(ts), min(ts), max(ts)


def profiler_kernel_span(fn, iters=20):
    """Kernel-only span per call: sum of all CUDA kernel durations / iters (us -> ms)."""
    from torch.profiler import profile, ProfilerActivity
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per_kernel = {}
    for evt in prof.key_averages():
        if evt.device_type.name != "CUDA" and str(evt.device_type) != "DeviceType.CUDA":
            continue
        dur = getattr(evt, "device_time_total", 0.0) or getattr(evt, "self_device_time_total", 0.0)
        if dur > 0 and evt.count > 0:
            per_kernel[evt.key] = (dur / iters / 1000.0, evt.count / iters)
    return per_kernel


def main():
    torch.cuda.init()
    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    try:
        import flash_attn
        print("flash_attn", getattr(flash_attn, "__version__", "?"))
    except Exception as e:
        print("flash_attn import failed:", e)
    from flash_attn.cute.interface import flash_attn_func

    q, k, v = make_inputs()
    out = flash_attn_func(q, k, v, softmax_scale=SCALE, causal=False)
    if isinstance(out, tuple):
        print("ref returns tuple of len", len(out))
        out = out[0]
    print("ref out", tuple(out.shape), out.dtype)
    torch.save(out.float().cpu(), "ref_out.pt")

    fn = lambda: flash_attn_func(q, k, v, softmax_scale=SCALE, causal=False)
    med, lo, hi = cuda_event_median(fn)
    print(f"REF cuda-events median = {med:.4f} ms (min {lo:.4f}, max {hi:.4f})")

    ks = profiler_kernel_span(fn)
    total = 0.0
    for name, (ms, cnt) in sorted(ks.items(), key=lambda x: -x[1][0]):
        print(f"  kernel {name[:70]!r}: {ms:.4f} ms/call  count/call={cnt:.2f}")
        total += ms
    print(f"REF profiler kernel-only span = {total:.4f} ms")

    flops = 4.0 * B * H * S * S * D
    print(f"total FLOPs = {flops/1e12:.3f} TFLOP -> {flops/(med*1e-3)/1e12:.1f} TFLOP/s at events median")


if __name__ == "__main__":
    main()
