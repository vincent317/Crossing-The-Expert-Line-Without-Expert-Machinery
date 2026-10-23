import torch, ctypes, time, sys
from common import DOCS, H, HDIM, gen_data, cu_seqlens_from_docs, device_time_ms

lib = ctypes.CDLL('./libfa_fwd4.so')
q, k, v, cu_q, cu_k = gen_data(DOCS, H, HDIM)
out = torch.empty_like(q)
scale = 1.0 / (HDIM ** 0.5)

# Smoke
lib.fa_fwd_launch(
    ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
    ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(cu_q.data_ptr()),
    ctypes.c_void_p(cu_k.data_ptr()), ctypes.c_void_p(out.data_ptr()),
    len(DOCS), H, scale, 0)
torch.cuda.synchronize()
err = torch.cuda.get_last_error()
print(f"[V4] launch err={err}")
if err != 0:
    sys.exit(1)

# Benchmark
dt = device_time_ms(lambda: lib.fa_fwd_launch(
    ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
    ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(cu_q.data_ptr()),
    ctypes.c_void_p(cu_k.data_ptr()), ctypes.c_void_p(out.data_ptr()),
    len(DOCS), H, scale, 0), warmup=10, iters=50)
print(f"[V4] device_time = {dt:.4f} ms")
