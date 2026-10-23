"""Measure reference (flash-attn-4 cute varlen) device time on B200 + fp32 tolerance."""
import torch

import common as C

print("=" * 70)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import flash_attn
print("flash_attn", getattr(flash_attn, "__version__", "?"))
from flash_attn.cute.interface import flash_attn_varlen_func
print("=" * 70)
print("T", C.T, "H", C.H, "D", C.D, "seqlens", C.SEQLENS)

q, k, v, cu = C.make_inputs()
print("q", q.shape, q.dtype, "cu", cu.tolist())


def ref():
    return flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=C.MAX_SEQLEN, max_seqlen_k=C.MAX_SEQLEN,
        softmax_scale=C.SOFTMAX_SCALE, causal=True,
    )


o = ref()
if isinstance(o, tuple):
    o = o[0]
print("ref out", o.shape, o.dtype)

ev = C.cuda_event_median(ref, warmup=10, iters=50)
sp, per, names = C.profiler_kernel_span(ref, warmup=10, iters=50, tag="ref")
print(f"[REF] cuda_event_median      = {ev:.4f} ms")
print(f"[REF] profiler_kernel_span   = {sp:.4f} ms   (kernels/iter = {per})")
print(f"[REF] kernel names = {names}")

print("-" * 70)
print("computing fp32 naive reference ...")
onv = C.naive_fp32(q, k, v)
e_ref = C.max_abs_err(o, onv)
tol = 7.0 * e_ref
print(f"[TOL] E_ref = max|ref - naive_fp32| = {e_ref:.6e}")
print(f"[TOL] tol = 7 * E_ref               = {tol:.6e}")
print(f"[TOL] target device time (ref/1.2)  = {sp / 1.2:.4f} ms  (events: {ev / 1.2:.4f} ms)")
torch.save({"e_ref": e_ref, "tol": tol, "ref_span": sp, "ref_ev": ev}, "/tmp/tol.pt")
print("DONE")
