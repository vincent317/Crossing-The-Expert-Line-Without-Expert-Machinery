"""Device-time benchmark: reference vs our Triton kernel — same data, same
process, same GPU, same CUDA-event methodology.

Two numbers are reported for both sides:
  per-call span : events recorded immediately around one invocation, so the
                  span runs from the first kernel's start to the last kernel's
                  end on the GPU (host launch overhead excluded, inter-kernel
                  GPU gaps included).
  amortized     : one event pair around 200 back-to-back invocations, which
                  also removes the inter-kernel gaps caused by host launch
                  latency.  Friendlier to implementations issuing many kernels.

Run:  python -u bench.py [C] [split]
"""
import sys, torch
sys.path.insert(0, ".")
from gdn_data import make_inputs, T, N, HQK, HV, D
from gdn_triton import GDN
from bench_util import device_time_ms, device_time_amortized_ms

C = int(sys.argv[1]) if len(sys.argv) > 1 else 64
SPLIT = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PUBLISHED_REF_MS = 0.1255      # KDA 0.5 CuTe-DSL GDN prefill, gdn_core path

d = make_inputs()
args = (d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"],
        d["state"], d["cu_seqlens"])
print("GPU:", torch.cuda.get_device_name(0))

ref_us = None
try:
    from gdn_ref import Ref
    from gdn_naive import gdn_naive_fp32
    o_nv, s_nv = gdn_naive_fp32(*args)
    torch.cuda.synchronize()
    ref = Ref(HQK, HV, D)
    ref.calibrate(*args, o_nv, s_nv, verbose=False)
    for nm, inc in (("ref (kernel only)", False), ("ref (incl. gate calc)", True)):
        fn = ref.make_callable(*args, include_gate=inc)
        med = device_time_ms(fn)[0] * 1e3
        am = device_time_amortized_ms(fn) * 1e3
        print(f"{nm:24s}: per-call={med:8.2f}us  amortized={am:8.2f}us")
        if not inc:
            ref_us = min(med, am)
except Exception as e:
    print(f"reference unavailable: {type(e).__name__}: {str(e)[:200]}")

g = GDN(T, N, HQK, HV, D, C=C, split=SPLIT, nsq=2, st_scan=3)
g(*args)
torch.cuda.synchronize()
med = device_time_ms(lambda: g(*args))[0] * 1e3
am = device_time_amortized_ms(lambda: g(*args)) * 1e3
ours = min(med, am)
print(f"{'ours (C=%d,split=%d)' % (C, SPLIT):24s}: per-call={med:8.2f}us  amortized={am:8.2f}us")
for nm, fn in ((" .. wy(+attn)", g.wy), (" .. attn", g.attn), (" .. scan", g.scan)):
    if nm == " .. attn" and g.fused:
        continue
    print(f"{nm:24s}: {device_time_ms(lambda f=fn: f(*args))[0]*1e3:8.2f}us")

print()
if ref_us is not None:
    print(f"measured ref device time     = {ref_us:.2f} us  (/1.2 = {ref_us/1.2:.2f} us)")
    print(f"ours                         = {ours:.2f} us   speedup = {ref_us/ours:.2f}x")
pub = PUBLISHED_REF_MS * 1e3
print(f"published ref device time    = {pub:.2f} us  (/1.2 = {pub/1.2:.2f} us)")
print(f"ours                         = {ours:.2f} us   speedup = {pub/ours:.2f}x")
print("TARGET MET" if ours <= pub / 1.2 else "TARGET NOT MET")
