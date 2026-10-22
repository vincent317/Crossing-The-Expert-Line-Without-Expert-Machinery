import torch, ctypes, time
from common import DOCS, H, HDIM, gen_data, device_time_ms

lib = ctypes.CDLL('./libfa_fwd2.so')
q, k, v, cu_q, cu_k = gen_data(DOCS, H, HDIM)
out = torch.empty_like(q)
scale = 1.0 / (HDIM ** 0.5)

print("[NEW v2] Benchmarking...")
t_dev = device_time_ms(lambda: lib.fa_fwd_launch(
    ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
    ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(cu_q.data_ptr()),
    ctypes.c_void_p(cu_k.data_ptr()), ctypes.c_void_p(out.data_ptr()),
    len(DOCS), H, scale, 0), warmup=10, iters=50)
print(f"[NEW v2] profiler_kernel_span = {t_dev:.4f} ms")
print(f"[NEW v2] ref span 0.5252 ms -> speedup {0.5252/t_dev:.3f}x (need >= 1.2)")
