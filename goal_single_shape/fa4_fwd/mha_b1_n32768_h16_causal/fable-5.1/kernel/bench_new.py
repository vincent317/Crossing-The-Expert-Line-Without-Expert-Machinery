"""Correctness check + benchmark of fa_fwd (new kernel) vs fp32 naive and vs ref timing.

Prints: max|new - naive_fp32|, tol, device time of new kernel (same method as bench_ref.py).
"""
import os
import sys

import torch

import common as C
from fa_fwd import FaFwd

q, k, v, cu = C.make_inputs()
fa = FaFwd(C.SEQLENS, C.H)
print("work items:", fa.n)

out = fa(q, k, v)
torch.cuda.synchronize()
print("out finite:", torch.isfinite(out.float()).all().item())

onv = C.naive_fp32(q, k, v)
err = C.max_abs_err(out, onv)
tolinfo = torch.load("/tmp/tol.pt") if os.path.exists("/tmp/tol.pt") else None
print(f"[CHECK] max|new - naive_fp32| = {err:.6e}")
if tolinfo:
    print(f"[CHECK] tol = 7*E_ref = {tolinfo['tol']:.6e}  ->  {'PASS' if err <= tolinfo['tol'] else 'FAIL'}")
# per-doc error breakdown
off = 0
for i, L in enumerate(C.SEQLENS):
    e = C.max_abs_err(out[off:off + L], onv[off:off + L])
    print(f"  doc {i:2d} L={L:5d} maxerr={e:.3e}")
    off += L

if "--nobench" in sys.argv:
    sys.exit(0)

def run():
    return fa(q, k, v, out)

ev = C.cuda_event_median(run, warmup=10, iters=50)
sp, per, names = C.profiler_kernel_span(run, warmup=10, iters=50, tag="new")
print(f"[NEW] cuda_event_median      = {ev:.4f} ms")
print(f"[NEW] profiler_kernel_span   = {sp:.4f} ms   (kernels/iter = {per})")
if tolinfo:
    print(f"[NEW] ref span {tolinfo['ref_span']:.4f} ms -> speedup {tolinfo['ref_span'] / sp:.3f}x (need >= 1.2)")
print("DONE")
