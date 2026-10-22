"""Sweep the v3 kernel on B200: accuracy vs fp32 naive, device time vs the ref kernel."""
import torch
import triton

import common as C
from fa_kernel_v3 import flash_attn_fwd_v3

# (BLOCK_M, BLOCK_N, num_warps, loop_stages, packed_exp2, subtile_acc, head_first,
#  num_ctas, maxnreg)
CONFIGS = [
    (128, 128, 8, 3, False, True,  False, 1, None),
    (128, 128, 8, 3, False, True,  False, 1, 168),
    (128, 128, 8, 3, False, False, False, 1, 168),
    (128, 128, 8, 3, True,  True,  False, 1, 168),
    (128, 128, 8, 3, True,  True,  False, 1, None),
    (128, 128, 8, 3, True,  False, False, 1, 168),
    (128, 128, 4, 3, True,  True,  False, 1, 168),
    (128, 64,  8, 4, True,  True,  False, 1, 168),
    (128, 64,  8, 4, False, True,  False, 1, 168),
    (256, 64,  8, 3, True,  True,  False, 1, 168),
    (128, 128, 8, 3, True,  True,  True,  1, 168),
    (128, 128, 8, 3, False, True,  True,  1, 168),
    (128, 64,  8, 4, True,  True,  True,  1, 168),
    (128, 128, 8, 3, True,  True,  False, 2, 168),
    (128, 128, 8, 3, False, True,  False, 2, 168),
    (128, 128, 8, 2, True,  True,  False, 1, 128),
    (128, 128, 8, 4, True,  True,  False, 1, 168),
]

FLOPS = 4.0 * C.B * C.H * C.S * C.S * C.D


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
    print(f"E_ref = max|naive_fp32 - ref| = {e_ref:.6e}", flush=True)
    print(f"tol = 7 * E_ref = {tol:.6e}", flush=True)

    results = []
    for cfg in CONFIGS:
        bm, bn, w, ls, pe, sa, hf, nc, mr = cfg
        tag = (f"BM{bm}_BN{bn}_w{w}_ls{ls}_pe{int(pe)}_sa{int(sa)}_hf{int(hf)}"
               f"_ctas{nc}_reg{mr}")
        try:
            fn = lambda: flash_attn_fwd_v3(q, k, v, softmax_scale=C.SCALE, BLOCK_M=bm,
                                           BLOCK_N=bn, num_warps=w, LOOP_STAGES=ls,
                                           PACKED_EXP2=pe, SUBTILE_ACC=sa,
                                           HEAD_FIRST=hf, num_ctas=nc, maxnreg=mr)
            o = fn()
            err = C.max_err(ref32, o)
            med, _, _ = C.cuda_event_median(fn, warmup=3, iters=10)
            ok = err <= tol
            print(f"  {tag}: {med:.4f} ms  {FLOPS/(med*1e-3)/1e12:.1f} TFLOP/s  "
                  f"err={err:.3e} {'PASS' if ok else 'FAIL'}", flush=True)
            if ok:
                results.append((med, tag, cfg, err))
        except Exception as exc:
            print(f"  {tag}: FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    if not results:
        print("no config passed")
        return
    results.sort(key=lambda r: r[0])
    med, tag, cfg, err = results[0]
    bm, bn, w, ls, pe, sa, hf, nc, mr = cfg
    print(f"\nbest = {tag}  quick={med:.4f} ms  err={err:.3e}", flush=True)

    fn = lambda: flash_attn_fwd_v3(q, k, v, softmax_scale=C.SCALE, BLOCK_M=bm, BLOCK_N=bn,
                                   num_warps=w, LOOP_STAGES=ls, PACKED_EXP2=pe,
                                   SUBTILE_ACC=sa, HEAD_FIRST=hf, num_ctas=nc, maxnreg=mr)
    mine_med, mine_span = C.report(f"MINE {tag}", fn)
    ref_med, ref_span = C.report("REF", lambda: flash_attn_func(q, k, v,
                                                                softmax_scale=C.SCALE,
                                                                causal=False))
    print(f"\nspeedup events = {ref_med/mine_med:.3f}x   need >= 1.2x", flush=True)
    if ref_span and mine_span:
        print(f"speedup kernel-span = {ref_span/mine_span:.3f}x   need >= 1.2x", flush=True)
    print(f"accuracy max|mine - naive_fp32| = {err:.6e}  tol = {tol:.6e}  "
          f"{'PASS' if err <= tol else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
