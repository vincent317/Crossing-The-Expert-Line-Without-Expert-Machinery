"""Round 2b: E_ref = max|ref_bwd - fp32_naive_bwd| at the full target shape.

Prints E_ref per tensor and tol = 7 * E_ref, the acceptance band for my kernel.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from naive_ref import naive_bwd_fp32
from timing import make_inputs, B, S, HQ, HKV, D

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SCALE = D ** -0.5
from flash_attn.cute.interface import flash_attn_func

q, k, v, do = make_inputs()
out = flash_attn_func(q, k, v, softmax_scale=SCALE, causal=True)
out = out[0] if isinstance(out, tuple) else out
gq, gk, gv = torch.autograd.grad(out, (q, k, v), do)
gq, gk, gv = gq.detach(), gk.detach(), gv.detach()
del out
torch.cuda.empty_cache()
print("ref grads done", flush=True)

t0 = time.time()
nq, nk, nv = naive_bwd_fp32(q.detach(), k.detach(), v.detach(), do, SCALE, True, q_chunk=2048)
torch.cuda.synchronize()
print(f"fp32 naive done in {time.time()-t0:.1f} s", flush=True)

print("=" * 70, flush=True)
print("E_ref = max|ref - fp32_naive|   (full shape B=4 S=8192 Hq=96 Hkv=8 D=128)", flush=True)
print("=" * 70, flush=True)
eref = {}
for nm, a, b_ in (("dq", gq, nq), ("dk", gk, nk), ("dv", gv, nv)):
    d_ = (a.float() - b_).abs()
    e = d_.max().item()
    eref[nm] = e
    print(f"  {nm}: E_ref = {e:.6e}   naive absmax = {b_.abs().max().item():.5f}   "
          f"mean|diff| = {d_.mean().item():.3e}", flush=True)
E = max(eref.values())
print(f"E_ref (max over dq,dk,dv) = {E:.6e}", flush=True)
print(f"tol = 7 * E_ref = {7*E:.6e}", flush=True)
for nm in eref:
    print(f"tol_{nm} = 7 * {eref[nm]:.6e} = {7*eref[nm]:.6e}", flush=True)
print("DONE-EREF", flush=True)
