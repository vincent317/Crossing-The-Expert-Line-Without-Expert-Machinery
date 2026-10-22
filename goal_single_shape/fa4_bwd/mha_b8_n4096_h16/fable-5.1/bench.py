"""Full comparison in ONE job: ref FA4 bwd vs my Triton bwd vs fp32 naive.

Identical data, identical timing methodology, identical iteration counts.
Prints: ref device time, E_ref, tol = 7*E_ref, mine error, mine device time, speedup.

--tune : sweep configs first (each guarded), then benchmark the winner.
"""
import argparse, time, json
import torch
from common import (make_data, l2_flush_buffer, cuda_events_median, profiler_kernel_time,
                    SCALE, B, S, H, D)
from naive import naive_attn_bwd_fp32, report_err
import kernel as K

TOL_FACTOR = 7.0
FLOPS = 5.0 * B * H * S * S * D  # 5 gemms * 2*d flops, causal halves -> 5*b*h*s*s*d


def make_mine(variant, cfg, bufs, q, k, v, o, do, lse):
    if variant == "split":
        return lambda: K.fa_backward_split(q, k, v, o, do, lse, SCALE, cfg=cfg, bufs=bufs)
    return lambda: K.fa_backward(q, k, v, o, do, lse, SCALE, cfg=cfg, bufs=bufs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ref", action="store_true")
    ap.add_argument("--skip-naive", action="store_true")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--piters", type=int, default=20)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--variant", default="fused", choices=("fused", "split"))
    ap.add_argument("--cfg", default=None, help="json dict overriding the variant's default cfg")
    args = ap.parse_args()

    q, k, v, do = make_data()
    print(f"problem: b={B} s={S} h={H} d={D} bf16 causal scale={SCALE:.8f}")
    print(f"total bwd FLOPs (causal, 5 gemms) = {FLOPS/1e12:.3f} TFLOP")
    flush = l2_flush_buffer()

    # ---------------- my forward (untimed) to get O and LSE
    o_mine, lse = K.fa_forward(q, k, v, SCALE)
    torch.cuda.synchronize()
    bufs = K.make_bufs(q)
    dq, dk, dv = bufs["dq"], bufs["dk"], bufs["dv"]

    # ---------------- fail-fast smoke test of my kernel on the default config
    print("dq_tma supported:", K.descriptor_atomic_add_supported(), flush=True)
    for dq_tma in (True,):
        for ws in (False,):
            cfg0 = dict(K.DEFAULT, dq_tma=dq_tma, ws=ws)
            try:
                K.fa_backward(q, k, v, o_mine, do, lse, SCALE, cfg=cfg0, bufs=bufs)
                torch.cuda.synchronize()
                print(f"smoke tma{int(dq_tma)} ws{int(ws)}: OK  |dq|max={dq.float().abs().max().item():.3e}",
                      flush=True)
            except Exception as e:
                import traceback
                print(f"smoke tma{int(dq_tma)} ws{int(ws)}: FAIL {type(e).__name__}: {str(e)[:400]}",
                      flush=True)
                traceback.print_exc()

    # ---------------- reference
    ref_kt = ref_span = ref_med = None
    ref_grads = None
    if not args.skip_ref:
        from flash_attn.cute.interface import flash_attn_func
        qg = q.clone().requires_grad_(True)
        kg = k.clone().requires_grad_(True)
        vg = v.clone().requires_grad_(True)
        out = flash_attn_func(qg, kg, vg, softmax_scale=SCALE, causal=True)
        if isinstance(out, (tuple, list)):
            out = out[0]

        def ref_bwd():
            return torch.autograd.grad(out, (qg, kg, vg), grad_outputs=do, retain_graph=True)

        ref_grads = ref_bwd()
        torch.cuda.synchronize()
        print(f"(info) max|my_fwd_O - ref_fwd_O| = "
              f"{(o_mine.float()-out.float()).abs().max().item():.4e}")
        ref_med, ts = cuda_events_median(ref_bwd, flush, warmup=10, iters=args.iters)
        print(f"[REF ] CUDA-events median: {ref_med:.4f} ms  (min {ts[0]:.4f} max {ts[-1]:.4f})")
        ref_kt, nk, ref_span = profiler_kernel_time(ref_bwd, flush, warmup=10,
                                                    iters=args.piters, tag="REF bwd")
        print(f"[REF ] device time: kernel-sum {ref_kt:.4f} ms/iter | span {ref_span:.4f} ms/iter "
              f"| kernels/iter {nk:.1f} | {FLOPS/1e12/(ref_kt/1e3):.1f} TFLOP/s", flush=True)

    # ---------------- choose my config
    variant = args.variant
    cfg = dict(K.SPLIT_DEFAULT if variant == "split" else K.DEFAULT)
    if args.cfg:
        cfg.update(json.loads(args.cfg))
    if args.tune:
        best = tune(q, k, v, do, o_mine, lse, flush, ref_kt, bufs)
        if best is not None:
            variant, cfg = best
    print(f"\nvariant: {variant}  cfg: {cfg}", flush=True)
    mine_bwd = make_mine(variant, cfg, bufs, q, k, v, o_mine, do, lse)
    mine_bwd()
    torch.cuda.synchronize()

    # ---------------- mine timing (same methodology as ref)
    mine_med, ts = cuda_events_median(mine_bwd, flush, warmup=10, iters=args.iters)
    print(f"[MINE] CUDA-events median: {mine_med:.4f} ms  (min {ts[0]:.4f} max {ts[-1]:.4f})")
    mine_kt, nk, mine_span = profiler_kernel_time(mine_bwd, flush, warmup=10,
                                                  iters=args.piters, tag="MINE bwd")
    print(f"[MINE] device time: kernel-sum {mine_kt:.4f} ms/iter | span {mine_span:.4f} ms/iter "
          f"| kernels/iter {nk:.1f} | {FLOPS/1e12/(mine_kt/1e3):.1f} TFLOP/s")

    if ref_kt is not None:
        print(f"\n=== SPEED ===")
        print(f"REF_DEVICE_TIME_MS   = {ref_kt:.4f}   (span {ref_span:.4f}, events {ref_med:.4f})")
        print(f"MINE_DEVICE_TIME_MS  = {mine_kt:.4f}   (span {mine_span:.4f}, events {mine_med:.4f})")
        print(f"TARGET (ref/1.2)     = {ref_kt/1.2:.4f} ms")
        print(f"SPEEDUP              = {ref_kt/mine_kt:.4f}x   "
              f"{'PASS >=1.2x' if ref_kt/mine_kt >= 1.2 else 'FAIL <1.2x'}")
        print(f"SPEEDUP(events)      = {ref_med/mine_med:.4f}x")
        print(f"SPEEDUP(span)        = {ref_span/mine_span:.4f}x")

    # ---------------- accuracy
    if not args.skip_naive:
        print("\n=== ACCURACY ===")
        t0 = time.time()
        nref = naive_attn_bwd_fp32(q, k, v, do, SCALE)
        torch.cuda.synchronize()
        print(f"fp32 naive bwd computed in {time.time()-t0:.1f}s")
        tol = None
        if ref_grads is not None:
            e_ref = report_err("ref ", ref_grads, nref)
            tol = TOL_FACTOR * e_ref
            print(f"E_REF = {e_ref:.6e}")
            print(f"TOL   = {TOL_FACTOR} * E_REF = {tol:.6e}")
        e_mine = report_err("mine", (dq, dk, dv), nref)
        print(f"E_MINE = {e_mine:.6e}")
        if tol is not None:
            print(f"ACCURACY: E_MINE {e_mine:.6e} {'<=' if e_mine <= tol else '>'} TOL {tol:.6e}  "
                  f"-> {'PASS' if e_mine <= tol else 'FAIL'}")


def _time_cfg(fn, flush):
    fn(); torch.cuda.synchronize()
    m, _ = cuda_events_median(fn, flush, warmup=3, iters=8)
    return m


def tune(q, k, v, do, o, lse, flush, ref_kt, bufs):
    results = []

    def run(cfg, tag, variant="fused"):
        fn = make_mine(variant, cfg, bufs, q, k, v, o, do, lse)
        try:
            m = _time_cfg(fn, flush)
        except Exception as e:
            print(f"  {tag:34s} FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
            return
        print(f"  {tag:34s} {m:8.4f} ms" + (f"  ({ref_kt/m:.3f}x)" if ref_kt else ""), flush=True)
        if cfg.get("abl", 0) == 0:
            results.append((m, tag, (variant, cfg)))

    print("\n=== SPLIT sweep (7 GEMMs, zero atomics) ===", flush=True)
    for bm, bn in ((128, 128), (128, 64), (64, 128), (64, 64), (256, 128), (128, 256)):
        for w in (4, 8):
            for st in (1, 2, 3, 4):
                for ws in (False, True):
                    cfg = dict(K.SPLIT_DEFAULT, kv_M=bm, kv_N=bn, kv_warps=w, kv_stages=st,
                               kv_ws=ws, q_M=bm, q_N=bn, q_warps=w, q_stages=st, q_ws=ws)
                    run(cfg, f"split M{bm} N{bn} w{w} s{st} ws{int(ws)}", "split")

    print("\n=== FUSED reference point ===", flush=True)
    run(dict(K.DEFAULT, BLOCK_M=64, BLOCK_N=128, num_warps=8, num_stages=2,
             ws=False, dq_tma=True), "fused M64 N128 w8 s2")

    results.sort(key=lambda r: r[0])
    print("\nTOP 10:")
    for m, tag, _ in results[:10]:
        print(f"  {tag:34s} {m:8.4f} ms" + (f"  ({ref_kt/m:.3f}x)" if ref_kt else ""))
    if not results:
        return None
    print("BEST_CFG:", results[0][1], json.dumps(results[0][2][1]))
    return results[0][2]


if __name__ == "__main__":
    main()
