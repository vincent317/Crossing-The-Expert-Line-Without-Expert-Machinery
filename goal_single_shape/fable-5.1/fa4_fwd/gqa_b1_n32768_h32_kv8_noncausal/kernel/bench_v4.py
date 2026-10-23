"""Bench the hand-written Gluon v4 kernel: accuracy vs fp32 naive, device time vs ref."""
import os
import subprocess
import sys
import traceback
import torch
import triton

import common as C
from fa_kernel_v4 import flash_attn_fwd_v4

FLOPS = 4.0 * C.B * C.H * C.S * C.S * C.D
CONFIGS = [  # (BLOCK_N, NUM_KV)
    (128, 3),
    (128, 2),
    (64, 4),
    (64, 6),
]


def smoke():
    """Tiny problem in a child process: a deadlock shows up as a timeout, not a dead job."""
    q, k, v = C.make_inputs()
    qs, ks, vs = (t[:, :1024].contiguous() for t in (q, k, v))
    o = flash_attn_fwd_v4(qs, ks, vs, softmax_scale=C.SCALE)
    torch.cuda.synchronize()
    ref_s = torch.nn.functional.scaled_dot_product_attention(
        qs.transpose(1, 2).float(), ks.transpose(1, 2).float(), vs.transpose(1, 2).float(),
        scale=C.SCALE).transpose(1, 2)
    print(f"smoke S=1024 max err vs fp32 sdpa = {C.max_err(o, ref_s):.3e}", flush=True)
    print(f"smoke out stats: mean={o.float().mean().item():.4e} absmax={o.float().abs().max().item():.4e} "
          f"ref absmax={ref_s.abs().max().item():.4e}", flush=True)


def main():
    print("triton", triton.__version__, torch.cuda.get_device_name(0), flush=True)
    if os.environ.get("FA_SMOKE") == "1":
        smoke()
        return
    try:
        r = subprocess.run([sys.executable, "-u", __file__], env={**os.environ, "FA_SMOKE": "1"},
                           timeout=420, capture_output=True, text=True)
        print(r.stdout[-4000:], flush=True)
        if r.returncode != 0:
            print(r.stderr[-6000:], flush=True)
            print("smoke FAILED (rc)", flush=True)
            return
    except subprocess.TimeoutExpired as e:
        print((e.stdout or b"")[-2000:] if isinstance(e.stdout, bytes) else (e.stdout or "")[-2000:], flush=True)
        print("smoke FAILED: TIMEOUT (kernel hang)", flush=True)
        return
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
    for bn, nkv in CONFIGS:
        tag = f"gluon_BN{bn}_KV{nkv}"
        try:
            fn = lambda: flash_attn_fwd_v4(q, k, v, softmax_scale=C.SCALE, BLOCK_N=bn, NUM_KV=nkv)
            o = fn()
            err = C.max_err(ref32, o)
            med, _, _ = C.cuda_event_median(fn, warmup=3, iters=10)
            ok = err <= tol
            print(f"  {tag}: {med:.4f} ms  {FLOPS/(med*1e-3)/1e12:.1f} TFLOP/s  err={err:.3e} "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
            if ok:
                results.append((med, tag, bn, nkv, err))
        except Exception as exc:
            traceback.print_exc()
            print(f"  {tag}: FAILED {type(exc).__name__}: {str(exc)[:300]}", flush=True)

    if not results:
        print("no config passed")
        return
    results.sort(key=lambda r: r[0])
    med, tag, bn, nkv, err = results[0]
    print(f"\nbest = {tag}  quick={med:.4f} ms  err={err:.3e}", flush=True)
    fn = lambda: flash_attn_fwd_v4(q, k, v, softmax_scale=C.SCALE, BLOCK_N=bn, NUM_KV=nkv)
    mine_med, mine_span = C.report(f"MINE {tag}", fn)
    ref_med, ref_span = C.report("REF", lambda: flash_attn_func(q, k, v, softmax_scale=C.SCALE,
                                                                causal=False))
    print(f"\nspeedup events = {ref_med/mine_med:.3f}x   need >= 1.2x", flush=True)
    if ref_span and mine_span:
        print(f"speedup kernel-span = {ref_span/mine_span:.3f}x   need >= 1.2x", flush=True)
    print(f"accuracy max|mine - naive_fp32| = {err:.6e}  tol = {tol:.6e}  "
          f"{'PASS' if err <= tol else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
