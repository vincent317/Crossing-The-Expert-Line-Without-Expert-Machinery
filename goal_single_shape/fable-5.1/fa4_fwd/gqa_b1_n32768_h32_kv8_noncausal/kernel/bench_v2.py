"""Bench the TMA + warp-specialized v2 kernel against ref, with fp32-naive accuracy gate."""
import torch
import triton

import common as C
from fa_kernel_v2 import flash_attn_fwd_v2

CONFIGS = [
    # (BLOCK_M, BLOCK_N, num_warps, num_stages, warp_spec, loop_stages)
    (128, 128, 8, 2, True, 3),
    (128, 128, 8, 2, True, 2),
    (128, 128, 4, 2, True, 3),
    (128, 128, 8, 2, False, 3),
    (128, 64, 8, 2, True, 4),
    (128, 64, 4, 2, True, 4),
    (256, 128, 8, 2, True, 2),
    (256, 64, 8, 2, True, 3),
    (64, 128, 4, 2, True, 4),
    (128, 256, 8, 2, True, 2),
]


def main():
    print("triton", triton.__version__, torch.cuda.get_device_name(0), flush=True)
    q, k, v = C.make_inputs()

    print("fp32 naive reference...", flush=True)
    ref32 = C.naive_fp32(q, k, v)
    from flash_attn.cute.interface import flash_attn_func
    ro = flash_attn_func(q, k, v, softmax_scale=C.SCALE, causal=False)
    if isinstance(ro, tuple):
        ro = ro[0]
    e_ref = C.max_err(ref32, ro)
    tol = 7.0 * e_ref
    print(f"E_ref = {e_ref:.6e}   tol = 7*E_ref = {tol:.6e}", flush=True)

    flops = 4.0 * C.B * C.H * C.S * C.S * C.D
    results = []
    for bm, bn, w, s, ws, ls in CONFIGS:
        tag = f"BM{bm}_BN{bn}_w{w}_s{s}_ws{int(ws)}_ls{ls}"
        try:
            fn = lambda: flash_attn_fwd_v2(q, k, v, softmax_scale=C.SCALE, BLOCK_M=bm,
                                           BLOCK_N=bn, num_warps=w, num_stages=s,
                                           WARP_SPEC=ws, LOOP_STAGES=ls)
            o = fn()
            err = C.max_err(ref32, o)
            med, _, _ = C.cuda_event_median(fn, warmup=3, iters=10)
            print(f"  {tag}: {med:.4f} ms  {flops/(med*1e-3)/1e12:.1f} TFLOP/s  err={err:.3e} "
                  f"{'PASS' if err <= tol else 'FAIL'}", flush=True)
            if err <= tol:
                results.append((med, tag, bm, bn, w, s, ws, ls, err))
        except Exception as exc:
            print(f"  {tag}: FAILED {type(exc).__name__}: {str(exc)[:220]}", flush=True)

    if not results:
        print("no config passed")
        return
    results.sort()
    med, tag, bm, bn, w, s, ws, ls, err = results[0]
    print(f"\nbest = {tag} {med:.4f} ms", flush=True)
    fn = lambda: flash_attn_fwd_v2(q, k, v, softmax_scale=C.SCALE, BLOCK_M=bm, BLOCK_N=bn,
                                   num_warps=w, num_stages=s, WARP_SPEC=ws, LOOP_STAGES=ls)
    mine_med, mine_span = C.report(f"MINE {tag}", fn)
    ref_med, ref_span = C.report("REF", lambda: flash_attn_func(q, k, v, softmax_scale=C.SCALE,
                                                                causal=False))
    print(f"\nspeedup events = {ref_med/mine_med:.3f}x  span = "
          f"{(ref_span/mine_span) if (ref_span and mine_span) else float('nan'):.3f}x  need >=1.2x",
          flush=True)
    print(f"accuracy max|mine-naive_fp32| = {err:.6e} tol {tol:.6e} "
          f"{'PASS' if err <= tol else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
