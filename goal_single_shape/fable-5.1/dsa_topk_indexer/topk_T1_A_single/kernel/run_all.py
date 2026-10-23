#!/usr/bin/env python3
"""
run_all.py  -- single entry-point that runs every required step on the B200.

Steps:
  1. Install safetensors
  2. Load real seq_lens / block_table from safetensors
  3. Generate random q_fp8, k_cache, weights
  4. Measure ref device time (profiler kernel-only span + CUDA events median)
  5. Run naive fp32, compute E_ref, tol = 7 * E_ref
  6. Compile new CUDA kernel
  7. Check accuracy: E_new vs tol
  8. Benchmark new kernel device time vs ref
"""
import os, sys, subprocess

# ── install deps ────────────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "safetensors"],
               check=False)

import torch
from torch.profiler import profile, ProfilerActivity
from safetensors import safe_open

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ref"))

PAGE_SIZE = 64
HEAD_DIM  = 128
K_STRIDE  = 132
NUM_PAGES = 11923
TOPK_K    = 2048
WARMUP    = 10
ITERS     = 50
L2_MB     = 512   # flush buffer (> B200 L2 cache size ~96 MB)

# ── helpers ─────────────────────────────────────────────────────────────────

def load_data(device="cuda"):
    path = os.path.join(ROOT, "ref/data/30cecff1-7ea4-474b-90fc-7f4a87206d8e.safetensors")
    out  = {}
    with safe_open(path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def find_tensors(data):
    seq_lens = block_table = None
    for k, v in data.items():
        kl = k.lower()
        if "seq" in kl and v.ndim == 1:
            seq_lens = v.to(torch.int32)
        elif ("block" in kl or "table" in kl) and v.ndim >= 2:
            block_table = v.to(torch.int32)
    return seq_lens, block_table


def gen_inputs(seq_lens, block_table, device="cuda"):
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
    return q_fp8, k_cache, weights


def idx_error(a, b):
    va = a[a != -1].float()
    vb = b[b != -1].float()
    n  = min(len(va), len(vb))
    return float((va[:n] - vb[:n]).abs().max()) if n > 0 else 0.0


def measure(fn, label, device="cuda"):
    flush = torch.empty(L2_MB * 1024 * 1024 // 4,
                        dtype=torch.float32, device=device)
    for _ in range(WARMUP):
        flush.zero_()
        torch.cuda.synchronize()
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(ITERS):
            flush.zero_()
            fn()
        torch.cuda.synchronize()

    print(f"\n[{label}] profiler CUDA table:")
    print(prof.key_averages().table(sort_by="device_time_total", row_limit=12))

    # FunctionEventAvg attribute name varies by PyTorch version:
    # cuda_time_total (µs aggregate) is always available; count gives avg.
    # Print all kernels with device time for debugging
    print(f"  [{label}] all CUDA kernels with device time:")
    for e in prof.key_averages():
        if e.device_time_total > 0 and e.count > 0:
            avg = e.device_time_total / e.count
            print(f"    {avg:8.3f} µs  x{e.count}  {e.key[:80]}")

    # Select the kernel of interest: exclude torch internal fill/memset ops
    # The target kernel is NOT "at::native::" prefixed
    stats = sorted(
        [(e.device_time_total / e.count, e.key, e.count)
         for e in prof.key_averages()
         if e.count > 0 and e.device_time_total > 0
         and "at::native::" not in e.key
         and "cudaMemset" not in e.key])
    prof_us = None
    for avg, name, cnt in stats:
        if cnt >= ITERS // 2:
            prof_us = avg
            print(f"  [{label}] selected kernel: '{name}'  avg={avg:.3f} µs (x{cnt})")
            break
    if prof_us is None:
        # fallback: pick shortest among all
        stats_all = sorted(
            [(e.device_time_total / e.count, e.key, e.count)
             for e in prof.key_averages()
             if e.count > 0 and e.device_time_total > 0])
        for avg, name, cnt in stats_all:
            if cnt >= ITERS // 2:
                prof_us = avg
                print(f"  [{label}] fallback kernel: '{name}'  avg={avg:.3f} µs (x{cnt})")
                break

    s, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    evt = []
    for _ in range(ITERS):
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


# ── compile CUDA extension inline ────────────────────────────────────────────

CUDA_SRC = open(os.path.join(ROOT, "topk_kernel.cu")).read()

def compile_ext():
    from torch.utils.cpp_extension import load_inline
    return load_inline(
        name="dsa_topk_ext",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100"],
        verbose=True,
    )


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.cuda.set_device(0)
    device = "cuda"

    print("=" * 70)
    print("DSA Top-K Indexer — full benchmark run")
    print("=" * 70)

    # ── 1. data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading data from safetensors ...")
    data = load_data(device)
    print("  keys:", list(data.keys()))
    for k, v in data.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    seq_lens, block_table = find_tensors(data)
    assert seq_lens is not None, f"seq_lens not found in {list(data.keys())}"
    assert block_table is not None, f"block_table not found in {list(data.keys())}"
    print(f"  seq_lens     = {seq_lens.tolist()}")
    print(f"  block_table  = {block_table.tolist()}")

    # ── 2. inputs ─────────────────────────────────────────────────────────────
    print("\n[2] Generating random inputs ...")
    q_fp8, k_cache, weights = gen_inputs(seq_lens, block_table, device)
    print(f"  q_fp8   : {tuple(q_fp8.shape)} {q_fp8.dtype}")
    print(f"  k_cache : {tuple(k_cache.shape)} {k_cache.dtype}")
    print(f"  weights : {tuple(weights.shape)} {weights.dtype}")

    # ── 3. ref warmup + first output ─────────────────────────────────────────
    print("\n[3] Warming up ref (JIT compile on first call) ...")
    from ref_entry import run as ref_run
    ref_out = ref_run(q_fp8, k_cache, weights, seq_lens, block_table)
    torch.cuda.synchronize()
    print(f"  ref output[0]: {ref_out[0].tolist()}")

    # ── 4. measure ref device time ────────────────────────────────────────────
    print("\n[4] Measuring ref device time ...")
    def ref_fn(): ref_run(q_fp8, k_cache, weights, seq_lens, block_table)
    ref_prof, ref_evt = measure(ref_fn, "ref", device=device)
    print(f"\n  REF device time (profiler): {ref_prof:.3f} µs")
    print(f"  REF CUDA events median   : {ref_evt:.2f} µs")

    # ── 5. naive fp32 ─────────────────────────────────────────────────────────
    print("\n[5] Running naive fp32 ...")
    import naive_fp32
    naive_out = naive_fp32.run(q_fp8, k_cache, weights, seq_lens, block_table)
    print(f"  naive output[0]: {naive_out[0].tolist()}")

    E_ref = idx_error(ref_out, naive_out)
    tol   = 7.0 * E_ref
    print(f"\n  E_ref = {E_ref:.2f}  (max abs diff of top-k indices: ref vs naive)")
    print(f"  tol   = 7 × E_ref = {tol:.2f}")

    # ── 6. compile new kernel ─────────────────────────────────────────────────
    print("\n[6] Compiling new CUDA kernel ...")
    ext    = compile_ext()
    max_sl = int(seq_lens.max().item())
    print("  Compilation OK")

    # ── 7. accuracy check ─────────────────────────────────────────────────────
    print("\n[7] Running new CUDA kernel (accuracy check) ...")
    our_out = ext.forward(q_fp8, k_cache, weights, seq_lens, block_table,
                          TOPK_K, max_sl)
    torch.cuda.synchronize()
    print(f"  new output[0]: {our_out[0].tolist()}")

    E_new = idx_error(our_out, naive_out)
    ok_acc = (E_new == 0.0) or (E_new <= tol)
    print(f"\n  E_new = {E_new:.2f}  (max abs diff: new kernel vs naive)")
    print(f"  tol   = {tol:.2f}")
    print(f"  ACCURACY: {'PASSED' if ok_acc else 'FAILED'}")

    # ── 8. benchmark new kernel ───────────────────────────────────────────────
    print("\n[8] Benchmarking new kernel device time ...")
    def our_fn():
        ext.forward(q_fp8, k_cache, weights, seq_lens, block_table, TOPK_K, max_sl)
    our_prof, our_evt = measure(our_fn, "new_kernel", device=device)
    print(f"\n  NEW KERNEL device time (profiler): {our_prof:.3f} µs")
    print(f"  NEW KERNEL CUDA events median   : {our_evt:.2f} µs")

    # ── final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    target  = ref_prof / 1.2 if ref_prof else None
    speedup = ref_prof / our_prof if (ref_prof and our_prof) else None
    print(f"  [1] Ref device time (profiler kernel-only span) : {ref_prof:.3f} µs")
    print(f"  [2] E_ref = {E_ref:.2f},  tol = {tol:.2f}")
    print(f"  [3] E_new = {E_new:.2f}  ({'<= tol PASS' if ok_acc else '> tol FAIL'})")
    print(f"  [4] New kernel device time                      : {our_prof:.3f} µs")
    print(f"      Target (ref / 1.2)                          : {target:.3f} µs")
    print(f"      Speedup                                     : {speedup:.2f}x")
    ok_spd = (our_prof <= target) if target else False
    print(f"      BENCHMARK: {'PASSED' if ok_spd else 'FAILED'}")
    print("=" * 70)
