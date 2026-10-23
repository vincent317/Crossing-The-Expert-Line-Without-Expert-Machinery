"""Device-time benchmark for the DSA FWD decode kernel.

Reference and candidate run in the same process, on the same tensors, and are
timed with the same routine. Primary metric = the profiler's CUDA kernel
duration, i.e. the GPU-side execution span of the op's kernels with host launch
overhead excluded - both the reference and our kernel are single-kernel ops, so
the duration is also the "first kernel start to last kernel end" span.

Run:  python -u bench.py
"""

import torch

from impls import build_all, build_reference
from timing import kernel_breakdown
from workload import make_workload, max_abs_err, naive_fp32

torch.backends.cuda.matmul.allow_tf32 = False
DECLARED_REF_US = 6.8  # KDA 0.5 CuTe-DSL contest kernel, as stated in the task
REPEATS = 3

p = torch.cuda.get_device_properties(0)
print(f"device = {p.name}  SMs = {p.multi_processor_count}  cc = {p.major}.{p.minor}")
w = make_workload()
print("valid KV rows per request:", (w["sparse_indices"] >= 0).sum(dim=1).tolist())
# Also establishes the ground truth, so the benchmark can assert that what it is
# timing is the correct kernel rather than a fast wrong one.
o_naive = naive_fp32(w)
print("fp32 naive ready: max|o| = %.6f" % o_naive.abs().max())


def device_time_us(tag, fn):
    """Median over REPEATS independent profiler measurements (50 iters each)."""
    samples, rows = [], []
    for _ in range(REPEATS):
        total, rows = kernel_breakdown(fn)
        samples.append(total)
    samples.sort()
    med = samples[len(samples) // 2]
    print(f"[{tag}] device time = {med:.3f} us   (samples: "
          f"{', '.join('%.3f' % x for x in samples)})")
    for name, cnt, us in rows:
        print(f"      {us:8.3f} us  x{cnt:.1f}  {name}")
    return med


run_ref = build_reference(w)
run_ref()
torch.cuda.synchronize()
ref_us = device_time_us("REF KDA0.5 CuTe-DSL", run_ref)
target = ref_us / 1.2
print("=" * 78)
print(f"reference device time = {ref_us:.3f} us  (task states {DECLARED_REF_US} us)")
print(f"TARGET = ref / 1.2 = {target:.3f} us")
print("=" * 78)

for name, impl in build_all(w).items():
    err = max_abs_err(impl().clone(), o_naive)
    torch.cuda.synchronize()
    print(f"[OURS {name}] max|ours - fp32_naive| = {err:.6e}   ({impl})")
    if hasattr(impl, "pointers"):
        print(f"[OURS {name}] buffers: {impl.pointers()}")
    us = device_time_us(f"OURS {name}", impl)
    print(f"      speedup vs reference = {ref_us / us:.3f}x  (need >= 1.200x)  -> "
          f"{'PASS' if us <= target else 'FAIL'}")
    print(f"      speedup vs the 6.8 us stated in the task = {DECLARED_REF_US / us:.3f}x")
    print("-" * 78)
print("BENCH DONE")
