"""Sweep Triton FlashAttention configs on B200 and compare against the fp32 naive path."""
import itertools
import torch
import triton

import common as C
from fa_kernel import flash_attn_fwd

CONFIGS = [
    # (BLOCK_M, BLOCK_N, num_warps, num_stages, warp_specialize)
    (128, 128, 8, 2, False),
    (128, 128, 8, 3, False),
    (128, 128, 8, 4, False),
    (128, 128, 4, 3, False),
    (128, 64, 8, 3, False),
    (128, 64, 4, 4, False),
    (128, 256, 8, 2, False),
    (64, 128, 4, 3, False),
    (64, 64, 4, 4, False),
    (256, 128, 8, 2, False),
    (128, 128, 8, 3, True),
    (128, 128, 4, 3, True),
    (128, 64, 4, 4, True),
    (256, 128, 8, 2, True),
    (128, 256, 8, 3, True),
]


def main():
    print("triton", triton.__version__, "torch", torch.__version__,
          torch.cuda.get_device_name(0), flush=True)
    q, k, v = C.make_inputs()

    print("computing fp32 naive reference (chunked, TF32 off)...", flush=True)
    ref32 = C.naive_fp32(q, k, v)
    torch.cuda.synchronize()

    # E_ref: how far the bf16 reference kernel sits from the fp32 naive result.
    from flash_attn.cute.interface import flash_attn_func
    ref_out = flash_attn_func(q, k, v, softmax_scale=C.SCALE, causal=False)
    if isinstance(ref_out, tuple):
        ref_out = ref_out[0]
    e_ref = C.max_err(ref32, ref_out)
    tol = 7.0 * e_ref
    print(f"E_ref = max|naive_fp32 - ref| = {e_ref:.6e}", flush=True)
    print(f"tol = 7 * E_ref = {tol:.6e}", flush=True)

    results = []
    for bm, bn, w, s, ws in CONFIGS:
        tag = f"BM{bm}_BN{bn}_w{w}_s{s}_ws{int(ws)}"
        try:
            fn = lambda: flash_attn_fwd(q, k, v, softmax_scale=C.SCALE, BLOCK_M=bm,
                                        BLOCK_N=bn, num_warps=w, num_stages=s,
                                        WARP_SPEC=ws)
            o = fn()
            err = C.max_err(ref32, o)
            med, _, _ = C.cuda_event_median(fn, warmup=3, iters=10)
            ok = "PASS" if err <= tol else "FAIL"
            flops = 4.0 * C.B * C.H * C.S * C.S * C.D
            print(f"  {tag}: {med:.4f} ms  {flops/(med*1e-3)/1e12:.1f} TFLOP/s  "
                  f"err={err:.6e} {ok}", flush=True)
            results.append((med, tag, bm, bn, w, s, ws, err, ok))
        except Exception as exc:
            print(f"  {tag}: FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    if not results:
        print("no config ran")
        return
    results.sort()
    med, tag, bm, bn, w, s, ws, err, ok = results[0]
    print(f"\nbest config = {tag} (quick median {med:.4f} ms, err {err:.6e} {ok})", flush=True)

    fn = lambda: flash_attn_fwd(q, k, v, softmax_scale=C.SCALE, BLOCK_M=bm, BLOCK_N=bn,
                                num_warps=w, num_stages=s, WARP_SPEC=ws)
    mine_med, mine_span = C.report(f"MINE {tag}", fn)
    ref_fn = lambda: flash_attn_func(q, k, v, softmax_scale=C.SCALE, causal=False)
    ref_med, ref_span = C.report("REF", ref_fn)

    print(f"\nspeedup (events) = {ref_med / mine_med:.3f}x   need >= 1.2x", flush=True)
    if ref_span and mine_span:
        print(f"speedup (kernel span) = {ref_span / mine_span:.3f}x   need >= 1.2x", flush=True)
    print(f"accuracy: max|mine - naive_fp32| = {err:.6e}  tol = {tol:.6e}  "
          f"{'PASS' if err <= tol else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
