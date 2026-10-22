import torch, ctypes, sys
from common import DOCS, H, HDIM, gen_data, naive_fp32_fa, E_ref, tol

lib = ctypes.CDLL('./libfa_fwd4.so')
q, k, v, cu_q, cu_k = gen_data(DOCS, H, HDIM)
out = torch.empty_like(q)
scale = 1.0 / (HDIM ** 0.5)

lib.fa_fwd_launch(
    ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
    ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(cu_q.data_ptr()),
    ctypes.c_void_p(cu_k.data_ptr()), ctypes.c_void_p(out.data_ptr()),
    len(DOCS), H, scale, 0)
torch.cuda.synchronize()

naive = naive_fp32_fa(q, k, v, cu_q, cu_k, scale)
err = torch.abs(out.float() - naive).max().item()
print(f"[CHECK v4] max|v4 - naive_fp32| = {err:.6e}")
print(f"[CHECK v4] tol = 7*E_ref = {tol:.6e}  ->  {'PASS' if err <= tol else 'FAIL'}")
sys.exit(0 if err <= tol else 1)
