#!/usr/bin/env python3
"""
benchmark.py
Measures device time (torch.profiler kernel-only span, µs) for ref and new kernel.
Same data, same measurement, L2 flushed each iteration, warmup 10 / iters 50.
"""
import os, sys
import torch
from torch.profiler import profile, ProfilerActivity

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ref"))

PAGE_SIZE = 64
HEAD_DIM  = 128
K_STRIDE  = 132
NUM_PAGES = 11923
WARMUP    = 10
ITERS     = 50
L2_MB     = 512


def load_data_and_inputs(device="cuda"):
    from safetensors import safe_open
    path = os.path.join(ROOT, "ref/data/30cecff1-7ea4-474b-90fc-7f4a87206d8e.safetensors")
    data = {}
    with safe_open(path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            data[k] = f.get_tensor(k)

    seq_lens = block_table = None
    for k, v in data.items():
        kl = k.lower()
        if "seq" in kl and v.ndim == 1:
            seq_lens = v.to(torch.int32)
        elif ("block" in kl or "table" in kl) and v.ndim >= 2:
            block_table = v.to(torch.int32)

    B = seq_lens.shape[0]
    q_fp8   = torch.randn(B, 64, HEAD_DIM, device=device).to(torch.float8_e4m3fn)
    weights = torch.rand(B, 64, device=device, dtype=torch.float32) + 0.1

    k_bf16   = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM, device=device,
                            dtype=torch.bfloat16)
    k_absmax = k_bf16.abs().amax(dim=-1, keepdim=True).float()
    k_scale  = (k_absmax / 448.0).clamp(min=1e-8)
    k_fp8    = (k_bf16.float() / k_scale).to(torch.float8_e4m3fn)

    k_cache = torch.zeros(NUM_PAGES, PAGE_SIZE, 1, K_STRIDE,
                          dtype=torch.int8, device=device)
    k_cache[:, :, 0, :HEAD_DIM] = (
        k_fp8.view(torch.int8).reshape(NUM_PAGES, PAGE_SIZE, HEAD_DIM))
    k_scale_i8 = (k_scale.squeeze(-1).contiguous()
                         .view(torch.int8).reshape(NUM_PAGES, PAGE_SIZE, 4))
    k_cache[:, :, 0, HEAD_DIM:K_STRIDE] = k_scale_i8

    return q_fp8, k_cache, weights, seq_lens, block_table


def measure(fn, label, device="cuda", warmup=WARMUP, iters=ITERS):
    flush = torch.empty(L2_MB * 1024 * 1024 // 4, dtype=torch.float32, device=device)

    for _ in range(warmup):
        flush.zero_()
        torch.cuda.synchronize()
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(iters):
            flush.zero_()
            fn()
        torch.cuda.synchronize()

    print(f"\n[{label}] profiler table:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))

    stats = []
    for e in prof.key_averages():
        if e.count > 0 and e.self_cuda_time_total > 0:
            stats.append((e.self_cuda_time_total / e.count, e.key, e.count))
    stats.sort()
    prof_us = None
    for avg, name, cnt in stats:
        if cnt >= iters // 2:   # skip one-off / rare kernels
            prof_us = avg
            print(f"  [{label}] kernel '{name}': avg={avg:.3f} µs (x{cnt})")
            break

    s = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    evt = []
    for _ in range(iters):
        flush.zero_()
        torch.cuda.synchronize()
        s.record()
        fn()
        en.record()
        torch.cuda.synchronize()
        evt.append(s.elapsed_time(en) * 1000.0)
    evt.sort()
    cuda_med = evt[len(evt) // 2]
    print(f"  [{label}] CUDA events median: {cuda_med:.2f} µs")
    return prof_us, cuda_med


def main():
    torch.cuda.set_device(0)
    device = "cuda"
    print("Loading data and generating inputs ...")
    q_fp8, k_cache, weights, seq_lens, block_table = load_data_and_inputs(device)

    import topk_ext
    topk_ext._get_ext(verbose=True)
    ext = topk_ext._get_ext()
    max_sl = int(seq_lens.max().item())

    from ref_entry import run as ref_run
    def ref_fn(): ref_run(q_fp8, k_cache, weights, seq_lens, block_table)
    def our_fn(): ext.forward(q_fp8, k_cache, weights, seq_lens, block_table, 2048, max_sl)

    print("Warming up ref (JIT compile) ...")
    for _ in range(3):
        ref_fn()
    torch.cuda.synchronize()

    print("\n=== BENCHMARK ===")
    ref_prof, ref_evt = measure(ref_fn, "ref",     device=device)
    our_prof, our_evt = measure(our_fn, "new_kernel", device=device)

    print("\n" + "=" * 62)
    print("BENCHMARK RESULTS (device time = profiler kernel-only span)")
    print("=" * 62)
    if ref_prof is not None and our_prof is not None:
        target  = ref_prof / 1.2
        speedup = ref_prof / our_prof
        print(f"  ref       profiler avg : {ref_prof:.3f} µs")
        print(f"  new kernel profiler avg: {our_prof:.3f} µs")
        print(f"  ref       CUDA events  : {ref_evt:.2f} µs (median)")
        print(f"  new kernel CUDA events : {our_evt:.2f} µs (median)")
        print(f"  target (ref/1.2)      : {target:.3f} µs")
        print(f"  speedup                : {speedup:.2f}x")
        ok = our_prof <= target
        print(f"  BENCHMARK: {'PASSED' if ok else 'FAILED'} "
              f"({speedup:.2f}x {'>= 1.2x' if ok else '< 1.2x'})")
        print("=" * 62)
        return 0 if ok else 1
    else:
        print("  Could not extract kernel time from profiler")
        print("=" * 62)
        return 1


if __name__ == "__main__":
    sys.exit(main())
