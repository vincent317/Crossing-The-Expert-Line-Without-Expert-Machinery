"""Accuracy check for the DSA FWD decode kernel.

  1. fp32 naive (gather -> fp32 matmul -> fp64 softmax -> fp32 matmul) = ground truth
  2. E_ref = max |reference - naive|, reference = the KDA 0.5 CuTe-DSL contest kernel
  3. tol   = 7 * E_ref
  4. assert max |ours - naive| <= tol

Run:  python -u check.py
"""

import sys

import torch

from impls import build_all, build_reference
from workload import make_workload, max_abs_err, naive_fp32

torch.backends.cuda.matmul.allow_tf32 = False

p = torch.cuda.get_device_properties(0)
print(f"device = {p.name}  SMs = {p.multi_processor_count}  cc = {p.major}.{p.minor}")

w = make_workload()
print("workload: T=%d H=%d d_ckv=%d d_kpe=%d topk=%d pages=%d page_size=64 bf16"
      % (w["q_nope"].shape[0], w["q_nope"].shape[1], w["q_nope"].shape[2],
         w["q_pe"].shape[2], w["sparse_indices"].shape[1], w["ckv_cache"].shape[0]))
print("valid KV rows per request:", (w["sparse_indices"] >= 0).sum(dim=1).tolist(),
      " total:", int((w["sparse_indices"] >= 0).sum()))

o_naive = naive_fp32(w)
print("fp32 naive: mean|o| = %.6f   max|o| = %.6f" % (o_naive.abs().mean(), o_naive.abs().max()))

run_ref = build_reference(w)
ref_out = run_ref()
if isinstance(ref_out, (tuple, list)):
    ref_out = ref_out[0]
torch.cuda.synchronize()
E_ref = max_abs_err(ref_out.reshape(o_naive.shape), o_naive)
TOL = 7.0 * E_ref
print("-" * 72)
print(f"E_ref = max|ref - fp32_naive| = {E_ref:.6e}")
print(f"tol   = 7 * E_ref             = {TOL:.6e}")
print("-" * 72)

ok = True
for name, impl in build_all(w).items():
    out = impl().clone()
    torch.cuda.synchronize()
    err = max_abs_err(out, o_naive)
    ok &= err <= TOL
    print(f"{name:8s} max|ours - fp32_naive| = {err:.6e}  <= tol ? "
          f"{'PASS' if err <= TOL else 'FAIL'}   ({impl})")

print("-" * 72)
print("ACCURACY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
