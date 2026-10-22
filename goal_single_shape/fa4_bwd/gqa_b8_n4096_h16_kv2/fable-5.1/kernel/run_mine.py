"""Accuracy (vs fp32 naive) + device-time benchmark of my Triton FA backward."""

import os
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_bwd_triton import CONFIGS, CONFIGS_V3, fa_bwd as _fa_bwd_plain, fa_bwd_v3, make_buffers
from fwd_lse import attn_fwd_lse
from naive_ref import naive_bwd_fp32
from timing import B, D, HKV, HQ, S, cuda_events_median, kernel_breakdown, make_inputs, tflops

# IMPL=plain|v3|gluon selects the kernel variant
IMPL = os.environ.get("IMPL", "gluon")
if IMPL == "ws":
    from fa_bwd_gluon_ws import CONFIGS_WS, fa_bwd_ws
    fa_bwd, CONFIGS = fa_bwd_ws, CONFIGS_WS
elif IMPL == "gluon":
    from fa_bwd_gluon import CONFIGS_G, fa_bwd_gluon
    fa_bwd, CONFIGS = fa_bwd_gluon, CONFIGS_G
elif IMPL == "v3":
    fa_bwd, CONFIGS = fa_bwd_v3, CONFIGS_V3
else:
    fa_bwd = _fa_bwd_plain
print("IMPL =", IMPL, flush=True)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
SCALE = D ** -0.5
# tolerances from job 1049333: 7 * E_ref(full shape) per tensor
TOL = {"dq": 1.122956e-1, "dk": 1.990833e-1, "dv": 2.963152e-1}

cfgs = CONFIGS
if len(sys.argv) > 1:
    cfgs = [tuple(int(x) if x not in ("True", "False") else x == "True" for x in a.split(","))
            for a in sys.argv[1:]]


def max_err(mine, ref):
    return {nm: (a.float() - b_).abs().max().item() for nm, a, b_ in zip(("dq", "dk", "dv"), mine, ref)}


# ------------------------------------------------------------ small-shape gate
print("=" * 70, flush=True)
print("SMALL SHAPE CHECK  B=1 S=2048 Hq=24 Hkv=2 D=128", flush=True)
print("=" * 70, flush=True)
qs, ks, vs, dos = make_inputs(1, 2048, 24, 2, 128, seed=7, requires_grad=False)
os_, lses = attn_fwd_lse(qs, ks, vs, SCALE)
nref = naive_bwd_fp32(qs, ks, vs, dos, SCALE, True, q_chunk=1024)
ok_cfgs = []
for cfg in cfgs:
    try:
        t0 = time.time()
        mine = fa_bwd(qs, ks, vs, os_, dos, lses, SCALE, cfg=cfg)
        torch.cuda.synchronize()
        e = max_err(mine, nref)
        ablation = cfg[-1] == 2 if IMPL in ("gluon", "ws") else (len(cfg) >= 6 and cfg[5] == 2)
        good = all(e[nm] <= TOL[nm] for nm in e if not (ablation and nm == "dq"))
        print(f"cfg {cfg}: dq {e['dq']:.3e} dk {e['dk']:.3e} dv {e['dv']:.3e}  "
              f"{'OK' if good else 'BAD'}{' (ablation, dq ignored)' if ablation else ''}"
              f"  (compile+run {time.time()-t0:.1f}s)", flush=True)
        if good:
            ok_cfgs.append(cfg)
    except Exception as ex:
        import traceback
        print(f"cfg {cfg}: FAILED {type(ex).__name__}", flush=True)
        traceback.print_exc()
        print(str(ex)[:4000], flush=True)
del qs, ks, vs, dos, os_, lses, nref
torch.cuda.empty_cache()

# ------------------------------------------------------------ full shape
print("=" * 70, flush=True)
print(f"FULL SHAPE  B={B} S={S} Hq={HQ} Hkv={HKV} D={D}", flush=True)
print("=" * 70, flush=True)
q, k, v, do = make_inputs(requires_grad=False)
t0 = time.time()
o, lse = attn_fwd_lse(q, k, v, SCALE)
torch.cuda.synchronize()
print(f"fwd (O, LSE) done {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
nref = naive_bwd_fp32(q, k, v, do, SCALE, True, q_chunk=2048)
torch.cuda.synchronize()
print(f"fp32 naive bwd done {time.time()-t0:.1f}s", flush=True)

bufs = make_buffers(q, k, v)
best = None
for cfg in ok_cfgs:
    try:
        mine = fa_bwd(q, k, v, o, do, lse, SCALE, cfg=cfg, out=bufs)
        torch.cuda.synchronize()
        e = max_err(mine, nref)
        ablation = cfg[-1] == 2 if IMPL in ("gluon", "ws") else (len(cfg) >= 6 and cfg[5] == 2)
        good = all(e[nm] <= TOL[nm] for nm in e if not (ablation and nm == "dq"))
        print(f"[acc] cfg {cfg}: max|mine-naive| dq {e['dq']:.4e} (tol {TOL['dq']:.3e}) "
              f"dk {e['dk']:.4e} (tol {TOL['dk']:.3e}) dv {e['dv']:.4e} (tol {TOL['dv']:.3e}) "
              f"-> {'PASS' if good else 'FAIL'}", flush=True)
        f = lambda: fa_bwd(q, k, v, o, do, lse, SCALE, cfg=cfg, out=bufs)
        med, lo, hi = cuda_events_median(f, 10, 50)
        print(f"[time] cfg {cfg}: cuda-events median {med:.4f} ms (min {lo:.3f} max {hi:.3f}) "
              f"-> {tflops(med):.1f} TFLOPS   [ref 15.632 ms, target <= 13.027 ms]", flush=True)
        if good and not ablation and (best is None or med < best[0]):
            best = (med, cfg)
    except Exception as ex:
        print(f"cfg {cfg}: FAILED {type(ex).__name__}: {str(ex)[:300]}", flush=True)
        torch.cuda.empty_cache()

if best:
    med, cfg = best
    print(f"BEST cfg {cfg}: {med:.4f} ms", flush=True)
    kernel_breakdown(lambda: fa_bwd(q, k, v, o, do, lse, SCALE, cfg=cfg, out=bufs), 15, label="mine")

print("DONE-MINE", flush=True)
