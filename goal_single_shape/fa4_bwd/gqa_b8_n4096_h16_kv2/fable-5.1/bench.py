"""Full run: ref device time, fp32 naive tolerance, my kernel accuracy + device time.

Both sides are timed identically: same inputs, same warmup/iters, same L2 flush, same
CUDA-event and profiler-kernel-span methodology. The only excluded kernel is the int8
L2-flush fill (uniquely named), so every kernel each implementation launches is counted.

SWEEP format: semicolon-separated specs, each "variant,BM,BN,W,S[,BM2,BN2,W2,S2]".
"""
import os
import statistics
import sys
import time

import torch

from common import (B, CAUSAL, D, FLUSH_KERNEL, HKV, HQ, S, SCALE, make_inputs,
                   measure_events, measure_profiler_span)
from kernel import Workspace, fa_bwd
from naive import max_err, naive_bwd_fp32, naive_fwd_fp32

WARMUP, ITERS = 10, 50
SKIP_NAIVE = os.environ.get("SKIP_NAIVE", "0") == "1"
SKIP_REF = os.environ.get("SKIP_REF", "0") == "1"

CFG = dict(variant=os.environ.get("VARIANT", "fused"),
           BM=int(os.environ.get("BM", 64)), BN=int(os.environ.get("BN", 64)),
           num_warps=int(os.environ.get("NW", 4)), num_stages=int(os.environ.get("NS", 2)),
           BM2=int(os.environ.get("BM2", 128)), BN2=int(os.environ.get("BN2", 64)),
           num_warps2=int(os.environ.get("NW2", 4)),
           num_stages2=int(os.environ.get("NS2", 2)))


def parse(spec):
    p = spec.split(",")
    c = dict(variant=p[0], BM=int(p[1]), BN=int(p[2]), num_warps=int(p[3]),
             num_stages=int(p[4]))
    if len(p) > 5:
        c.update(BM2=int(p[5]), BN2=int(p[6]), num_warps2=int(p[7]), num_stages2=int(p[8]))
    return c


def fmt(c):
    s = f"{c['variant']:5s} BM={c['BM']:3d} BN={c['BN']:3d} W={c['num_warps']} S={c['num_stages']}"
    if c["variant"] != "fused":
        s += f" | dq BM={c.get('BM2',128):3d} BN={c.get('BN2',64):3d} W={c.get('num_warps2',4)} S={c.get('num_stages2',2)}"
    return s


print("=== ENV ===", flush=True)
print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0))
import triton
print("triton", triton.__version__)
print(f"shape B{B} S{S} HQ{HQ} HKV{HKV} D{D} scale {SCALE:.6f} causal {CAUSAL}", flush=True)

q, k, v, do = make_inputs()

# ---------------- reference ----------------
ref_ev = ref_span = float("nan")
dq_r = dk_r = dv_r = None
if not SKIP_REF:
    from flash_attn.cute.interface import flash_attn_func
    qg = q.detach().clone().requires_grad_(True)
    kg = k.detach().clone().requires_grad_(True)
    vg = v.detach().clone().requires_grad_(True)
    out = flash_attn_func(qg, kg, vg, softmax_scale=SCALE, causal=CAUSAL)
    if isinstance(out, tuple):
        out = out[0]
    g = torch.autograd.grad(out, (qg, kg, vg), do, retain_graph=True)
    dq_r, dk_r, dv_r = g[0].clone(), g[1].clone(), g[2].clone()

    def ref_bwd():
        torch.autograd.grad(out, (qg, kg, vg), do, retain_graph=True)

    ref_ev, lo, hi = measure_events(ref_bwd, WARMUP, ITERS)
    ref_span, ref_per, ref_names = measure_profiler_span(ref_bwd, WARMUP, ITERS,
                                                         drop_names=(FLUSH_KERNEL,))
    print("\n=== REF device time ===")
    print(f"ref CUDA-events median   = {ref_ev:.4f} ms  (min {lo:.4f} max {hi:.4f})")
    print(f"ref kernel-span median   = {ref_span:.4f} ms   kernels/iter = {ref_per}")
    sys.stdout.flush()

# ---------------- my kernel ----------------
print("\n=== MINE ===", flush=True)
t0 = time.time()
o, lse2 = naive_fwd_fp32(q, k, v)
print(f"fp32 forward (o, lse2) done in {time.time() - t0:.1f}s", flush=True)
ws = Workspace(B, S, HQ, HKV, D)

SWEEP = os.environ.get("SWEEP", "")
if SWEEP:
    print("\n--- sweep (CUDA events, warmup5/iters20) ---", flush=True)
    best = None
    for spec in SWEEP.split(";"):
        c = parse(spec)
        try:
            def f(c=c):
                fa_bwd(q, k, v, o, do, lse2, SCALE, ws, **c)
            f()
            torch.cuda.synchronize()
            e, _, _ = measure_events(f, 5, 20)
            print(f"  {fmt(c)}   {e:.4f} ms", flush=True)
            if best is None or e < best[0]:
                best = (e, c)
        except Exception as ex:
            print(f"  {fmt(c)}   FAILED {type(ex).__name__}: {str(ex)[:130]}", flush=True)
    if best:
        print(f"  BEST {best[0]:.4f} ms  {fmt(best[1])}")
        CFG = best[1]

print(f"\nfinal cfg: {fmt(CFG)}", flush=True)


def mine_bwd():
    fa_bwd(q, k, v, o, do, lse2, SCALE, ws, **CFG)


dq_m, dk_m, dv_m = fa_bwd(q, k, v, o, do, lse2, SCALE, ws, **CFG)
torch.cuda.synchronize()
dq_m, dk_m, dv_m = dq_m.clone(), dk_m.clone(), dv_m.clone()
print("finite:", torch.isfinite(dq_m).all().item(), torch.isfinite(dk_m).all().item(),
      torch.isfinite(dv_m).all().item(), flush=True)

mine_ev, lo, hi = measure_events(mine_bwd, WARMUP, ITERS)
mine_span, mine_per, mine_names = measure_profiler_span(mine_bwd, WARMUP, ITERS,
                                                        drop_names=(FLUSH_KERNEL,))
print(f"\nmine CUDA-events median  = {mine_ev:.4f} ms  (min {lo:.4f} max {hi:.4f})")
print(f"mine kernel-span median  = {mine_span:.4f} ms   kernels/iter = {mine_per}")
print("per-kernel mean duration (us), from the same profiled run:")
for n, c in sorted(mine_names.items(), key=lambda x: -x[1]):
    print(f"    {c:5d}x  {n[:90]}")
sys.stdout.flush()

# ---------------- accuracy ----------------
print("\n=== ACCURACY ===", flush=True)
if not SKIP_NAIVE:
    t0 = time.time()
    dq_n, dk_n, dv_n = naive_bwd_fp32(q, k, v, do)
    print(f"fp32 naive backward done in {time.time() - t0:.1f}s", flush=True)
    if dq_r is not None:
        e = [max_err(dq_r, dq_n), max_err(dk_r, dk_n), max_err(dv_r, dv_n)]
        E_ref = max(e)
        tol = 7 * E_ref
        print(f"E_ref  dq={e[0]:.6e} dk={e[1]:.6e} dv={e[2]:.6e}  ->  E_ref={E_ref:.6e}")
        print(f"tol = 7 * E_ref = {tol:.6e}")
    else:
        tol = 2.325754e-01
        print(f"using tol recorded from job 1049182 = {tol:.6e}")
    em = [max_err(dq_m, dq_n), max_err(dk_m, dk_n), max_err(dv_m, dv_n)]
    E_mine = max(em)
    print(f"E_mine dq={em[0]:.6e} dk={em[1]:.6e} dv={em[2]:.6e}  ->  E_mine={E_mine:.6e}")
    print(f"ACCURACY {'PASS' if E_mine <= tol else 'FAIL'}: E_mine {E_mine:.6e} "
          f"{'<=' if E_mine <= tol else '>'} tol {tol:.6e}")

# ---------------- verdict ----------------
print("\n=== VERDICT ===")
FLOPS = 5 * B * HQ * S * S * D  # causal bwd: 5 GEMMs over half the score matrix
print(f"algorithmic FLOPs (5-GEMM causal) = {FLOPS / 1e12:.3f} TFLOP")
if not SKIP_REF:
    print(f"ref  events {ref_ev:.4f} ms | span {ref_span:.4f} ms | "
          f"{FLOPS / ref_ev / 1e9:.0f} TFLOP/s")
    print(f"mine events {mine_ev:.4f} ms | span {mine_span:.4f} ms | "
          f"{FLOPS / mine_ev / 1e9:.0f} TFLOP/s")
    print(f"speedup events {ref_ev / mine_ev:.3f}x | span {ref_span / mine_span:.3f}x")
    print(f"need <= {ref_ev / 1.2:.4f} ms (events) / {ref_span / 1.2:.4f} ms (span)")
    ok = mine_ev <= ref_ev / 1.2 and mine_span <= ref_span / 1.2
    print(f"SPEED {'PASS' if ok else 'FAIL'}")
else:
    print(f"mine events {mine_ev:.4f} ms | span {mine_span:.4f} ms | "
          f"{FLOPS / mine_ev / 1e9:.0f} TFLOP/s")
    print(f"recorded ref: events 3.0194 / span 2.9786 ms -> need <= 2.5162 / 2.4822 ms")
