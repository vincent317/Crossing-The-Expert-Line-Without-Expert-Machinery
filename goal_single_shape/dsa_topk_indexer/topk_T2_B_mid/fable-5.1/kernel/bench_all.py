"""
bench_all.py  —  accuracy check + benchmark for DSA top-k indexer.

Prints:
  1. REF device time (torch.profiler kernel-only span, median CUDA events)
  2. FP32 naive vs REF: max error E_ref, tolerance tol = 7 * E_ref
  3. New kernel vs FP32 naive: max error, pass/fail vs tol
  4. New kernel device time (same methodology as REF measurement)
"""

import torch
import sys, os, time
import numpy as np
from safetensors.torch import load_file

DATA_PATH = os.path.join(os.path.dirname(__file__), "ref", "data",
    "81a953ea-f29f-4707-8456-67a3624a549d.safetensors")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ref"))


# ── helpers ──────────────────────────────────────────────────────────────────

def make_inputs(device="cuda", seed=42):
    torch.manual_seed(seed)
    data = load_file(DATA_PATH)
    seq_lens    = data["seq_lens"].to(device=device, dtype=torch.int32)
    block_table = data["block_table"].to(device=device, dtype=torch.int32)

    B, H, D     = seq_lens.shape[0], 64, 128
    num_pages   = 11923
    page_size   = 64

    # q: random fp8_e4m3fn
    q_f32  = torch.randn(B, H, D, device=device).clamp(-448, 448)
    q_fp8  = q_f32.to(torch.float8_e4m3fn)

    # k cache: (num_pages, page_size, 1, 132) int8
    # Layout: first 128 bytes = fp8_e4m3fn data, last 4 bytes = float32 scale
    raw_k  = torch.zeros(num_pages, page_size, 1, 132, dtype=torch.int8, device=device)
    k_bf16 = torch.randn(num_pages, page_size, D, device=device, dtype=torch.bfloat16)
    k_f32  = k_bf16.float()
    scale  = k_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0  # (np,ps,1)
    k_q    = (k_f32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    raw_k[:, :, 0, :128] = k_q.view(num_pages, page_size, 128).view(torch.int8)
    scale_bytes = scale.squeeze(-1).float().view(torch.int8).reshape(num_pages, page_size, 4)
    raw_k[:, :, 0, 128:] = scale_bytes

    weights = torch.randn(B, H, device=device, dtype=torch.float32)

    return q_fp8, raw_k, weights, seq_lens, block_table


def l2_flush(device="cuda"):
    """Write 512 MB to flush L2 cache."""
    buf = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=device)
    buf.fill_(0.0)
    return buf


def cuda_median_us(fn, warmup=10, iters=50, device="cuda"):
    """Measure median CUDA event time (µs) with L2 flush between iterations."""
    buf = l2_flush(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        buf.fill_(0.0)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)   # ms -> µs
    times.sort()
    return times[len(times)//2]


def profiler_kernel_us(fn, n=5, device="cuda"):
    """Kernel-only device time via torch.profiler (µs), average over n iters."""
    import torch.profiler as profiler
    buf = l2_flush(device)
    buf.fill_(0.0)
    torch.cuda.synchronize()
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False,
    ) as prof:
        for _ in range(n):
            buf.fill_(0.0)
            fn()
    torch.cuda.synchronize()

    evts = prof.profiler.function_events
    # Sum all CUDA kernel durations except the fill operations
    total_us = sum(
        e.device_time for e in evts
        if e.device_time > 0 and "fill" not in e.name.lower()
    )
    return total_us / n   # µs per call


# ── set-of-ids comparison (order-invariant per batch) ────────────────────────

def max_id_error(pred, gold, seq_lens):
    """
    Returns fraction of non-(-1) positions that differ between pred and gold,
    treating each row as a SET (order does not matter).
    Also returns number of mismatched tokens.
    """
    B = pred.shape[0]
    total_miss = 0
    total_tok  = 0
    for b in range(B):
        k = min(2048, seq_lens[b].item())
        if k == 0:
            continue
        s_pred = set(pred[b, :k].tolist())
        s_gold = set(gold[b, :k].tolist())
        # Remove sentinel -1 from comparison
        s_pred.discard(-1)
        s_gold.discard(-1)
        miss = len(s_gold.symmetric_difference(s_pred))
        total_miss += miss
        total_tok  += len(s_gold)
    return total_miss, total_tok


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = "cuda"
    torch.cuda.set_device(0)

    print("=" * 60)
    print("Building inputs ...")
    q_fp8, k_cache, weights, seq_lens, block_table = make_inputs(device)
    print(f"  B={seq_lens.shape[0]}, seq_lens={seq_lens.cpu().tolist()}")
    print(f"  k_cache shape: {tuple(k_cache.shape)}")

    # ── 1. REF device time ────────────────────────────────────────────────
    print("\n[1/4] Measuring REF device time ...")
    from ref_entry import run as ref_run

    def _ref(): return ref_run(q_fp8, k_cache, weights, seq_lens, block_table)

    ref_median_us = cuda_median_us(_ref, warmup=10, iters=50)
    ref_profiler_us = profiler_kernel_us(_ref, n=5)

    print(f"  REF CUDA-events median : {ref_median_us:.2f} µs")
    print(f"  REF profiler kernel    : {ref_profiler_us:.2f} µs  ← device time")

    ref_raw = _ref()   # golden output (may be tensor or tuple)
    ref_out = ref_raw[0] if isinstance(ref_raw, (tuple, list)) else ref_raw

    # ── 2. FP32 naive vs REF error ────────────────────────────────────────
    print("\n[2/4] FP32 naive vs REF ...")
    from naive_fp32 import run_naive

    naive_out = run_naive(q_fp8, k_cache, weights, seq_lens, block_table)
    miss_naive, total = max_id_error(naive_out.cpu(), ref_out.cpu(), seq_lens.cpu())
    E_ref = miss_naive / max(total, 1)
    tol   = 7.0 * E_ref
    print(f"  Naive mismatched tokens: {miss_naive} / {total}")
    print(f"  E_ref (mismatch rate)  : {E_ref:.6f}")
    print(f"  tol = 7 * E_ref        : {tol:.6f}")

    # ── 3. Compile & check new kernel ─────────────────────────────────────
    print("\n[3/4] Compiling new CUDA kernel ...")
    import topk_cuda

    t0 = time.time()
    # Trigger compilation
    new_out = topk_cuda.run(q_fp8, k_cache, weights, seq_lens, block_table)
    print(f"  Compilation + first run: {time.time()-t0:.1f}s")

    # Accuracy vs naive
    miss_new, _ = max_id_error(new_out.cpu(), naive_out.cpu(), seq_lens.cpu())
    err_new = miss_new / max(total, 1)
    status = "PASS" if err_new <= tol else "FAIL"
    print(f"  New-kernel vs naive mismatch rate: {err_new:.6f}")
    print(f"  Tolerance: {tol:.6f}  -> {status}")

    # Also compare against ref directly
    miss_vs_ref, _ = max_id_error(new_out.cpu(), ref_out.cpu(), seq_lens.cpu())
    print(f"  New-kernel vs REF mismatch rate  : {miss_vs_ref/max(total,1):.6f}")

    if status == "FAIL":
        print("  [WARNING] Accuracy check FAILED; still benchmarking.")

    # ── 4. New kernel device time ─────────────────────────────────────────
    print("\n[4/4] Benchmarking new kernel ...")
    def _new(): return topk_cuda.run(q_fp8, k_cache, weights, seq_lens, block_table)

    new_median_us   = cuda_median_us(_new, warmup=10, iters=50)
    new_profiler_us = profiler_kernel_us(_new, n=5)

    print(f"  New CUDA-events median : {new_median_us:.2f} µs")
    print(f"  New profiler kernel    : {new_profiler_us:.2f} µs  ← device time")

    target_us = ref_profiler_us / 1.2
    speedup   = ref_profiler_us / new_profiler_us
    goal_ok   = new_profiler_us <= target_us

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  REF device time    : {ref_profiler_us:.2f} µs")
    print(f"  New device time    : {new_profiler_us:.2f} µs")
    print(f"  Target (ref/1.2)   : {target_us:.2f} µs")
    print(f"  Speedup            : {speedup:.2f}x")
    print(f"  Accuracy (new vs naive): {err_new:.6f} <= {tol:.6f} -> {status}")
    goal_speed = "PASS" if goal_ok else "FAIL"
    print(f"  Speed goal (>=1.2x): {goal_speed}")
    print("=" * 60)

    if goal_ok and status == "PASS":
        print("ALL GOALS MET.")
    else:
        print("NOT YET COMPLETE — see above for failing conditions.")


if __name__ == "__main__":
    main()
