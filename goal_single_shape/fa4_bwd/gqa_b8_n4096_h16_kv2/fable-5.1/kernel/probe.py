"""Round 1: environment probe + tiny-shape semantics check + ref device time."""

import json
import os
import subprocess
import sys

import torch

print("=" * 70, flush=True)
print("ENV", flush=True)
print("=" * 70, flush=True)
print("python", sys.version.replace("\n", " "), flush=True)
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
print("gpu", torch.cuda.get_device_name(0), flush=True)
cc = torch.cuda.get_device_capability(0)
print("capability", cc, flush=True)
props = torch.cuda.get_device_properties(0)
print("sms", props.multi_processor_count, "mem_GB", round(props.total_memory / 2**30, 1), flush=True)
print("nvcc:", subprocess.run(["bash", "-lc", "which nvcc && nvcc --version | tail -2"],
                              capture_output=True, text=True).stdout.strip(), flush=True)

try:
    import flash_attn
    print("flash_attn", getattr(flash_attn, "__version__", "?"), flash_attn.__file__, flush=True)
except Exception as e:
    print("flash_attn import FAILED:", e, flush=True)

try:
    from flash_attn.cute.interface import flash_attn_func
    import inspect
    print("cute flash_attn_func sig:", inspect.signature(flash_attn_func), flush=True)
    HAVE_CUTE = True
except Exception as e:
    print("cute interface import FAILED:", repr(e), flush=True)
    HAVE_CUTE = False

for mod in ["cutlass", "cutlass.cute", "nvidia.cutlass_dsl", "triton"]:
    try:
        m = __import__(mod, fromlist=["__version__"])
        print(f"{mod} OK version={getattr(m, '__version__', '?')} path={getattr(m, '__file__', '?')}", flush=True)
    except Exception as e:
        print(f"{mod} MISSING ({type(e).__name__})", flush=True)

print(subprocess.run(["bash", "-lc", "pip list 2>/dev/null | grep -iE 'cutlass|flash|triton|nvidia-cuda' | head -30"],
                     capture_output=True, text=True).stdout, flush=True)

if not HAVE_CUTE:
    sys.exit("no cute interface, abort")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from naive_ref import naive_bwd_fp32

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ------------------------------------------------------------------ tiny check
print("=" * 70, flush=True)
print("TINY SEMANTICS CHECK (B=1,S=256,Hq=4,Hkv=2,D=64)", flush=True)
print("=" * 70, flush=True)
torch.manual_seed(0)
B, S, Hq, Hkv, D = 1, 256, 4, 2, 64
scale = D ** -0.5
q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
do = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)

out = flash_attn_func(q, k, v, softmax_scale=scale, causal=True)
if isinstance(out, tuple):
    print("fwd returns tuple len", len(out), flush=True)
    out = out[0]
print("out", out.shape, out.dtype, flush=True)
gq, gk, gv = torch.autograd.grad(out, (q, k, v), do)
print("grad shapes", gq.shape, gk.shape, gv.shape, flush=True)

nq, nk, nv = naive_bwd_fp32(q.detach(), k.detach(), v.detach(), do, scale, True, q_chunk=128)
for nm, a, b_ in (("dq", gq, nq), ("dk", gk, nk), ("dv", gv, nv)):
    e = (a.float() - b_).abs().max().item()
    print(f"tiny max|{nm}_ref - {nm}_naive| = {e:.6e}  (ref absmax {a.float().abs().max().item():.4f})", flush=True)

# ------------------------------------------------------------------ full ref
print("=" * 70, flush=True)
print("REF DEVICE TIME  B=4 S=8192 Hq=96 Hkv=8 D=128 bf16 causal", flush=True)
print("=" * 70, flush=True)
B, S, Hq, Hkv, D = 4, 8192, 96, 8, 128
scale = D ** -0.5
torch.manual_seed(1234)
q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
do = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)

out = flash_attn_func(q, k, v, softmax_scale=scale, causal=True)
if isinstance(out, tuple):
    out = out[0]

flush_buf = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

def bwd():
    return torch.autograd.grad(out, (q, k, v), do, retain_graph=True)

WARM, ITERS = 10, 50
for _ in range(WARM):
    bwd()
torch.cuda.synchronize()

starts = [torch.cuda.Event(True) for _ in range(ITERS)]
ends = [torch.cuda.Event(True) for _ in range(ITERS)]
for i in range(ITERS):
    flush_buf.zero_()
    starts[i].record()
    bwd()
    ends[i].record()
torch.cuda.synchronize()
times = sorted(starts[i].elapsed_time(ends[i]) for i in range(ITERS))
med = times[ITERS // 2]
print(f"REF cuda-events median = {med:.3f} ms   min {times[0]:.3f}  max {times[-1]:.3f}", flush=True)

# profiler kernel-only span
from torch.profiler import profile, ProfilerActivity
PROF_IT = 20
with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    for _ in range(PROF_IT):
        bwd()
    torch.cuda.synchronize()
prof.export_chrome_trace("/tmp/ref_trace.json")
with open("/tmp/ref_trace.json") as f:
    tr = json.load(f)
ev = [e for e in tr["traceEvents"]
      if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel", "gpu_kernel")]
ev.sort(key=lambda e: e["ts"])
print(f"kernel events: {len(ev)} over {PROF_IT} iters -> {len(ev)/PROF_IT:.1f} per iter", flush=True)
names = {}
for e in ev:
    names.setdefault(e["name"], [0, 0.0])
    names[e["name"]][0] += 1
    names[e["name"]][1] += e["dur"]
for nm, (c, d) in sorted(names.items(), key=lambda x: -x[1][1]):
    print(f"  {c:5d}  {d/PROF_IT/1000:8.4f} ms/iter  {nm[:90]}", flush=True)
tot = sum(e["dur"] for e in ev) / PROF_IT / 1000
span = (max(e["ts"] + e["dur"] for e in ev) - min(e["ts"] for e in ev)) / PROF_IT / 1000
print(f"REF kernel-sum   = {tot:.3f} ms/iter", flush=True)
print(f"REF kernel-span  = {span:.3f} ms/iter  (includes inter-iter gaps)", flush=True)
print(f"TARGET (ref/1.2) = {med/1.2:.3f} ms  [from cuda-events median]", flush=True)
print("DONE", flush=True)
