set -e
echo "=== [1] build v4 ==="
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -DFA_MODE=1 \
  -Xptxas -v --shared -Xcompiler -fPIC fa_fwd4.cu -o libfa_fwd4.so 2>&1 | \
  grep -E "error|warning|registers|spill|smem" | head -20
ls -la libfa_fwd4.so || exit 1

echo "=== [2] ref benchmark ==="
export PIP_CONSTRAINT=
pip install -q torch 2>&1 | grep -v "already satisfied" || true
python -u bench_ref.py 2>&1 | grep -v "UserWarning\|exe_args\|USDT"

echo "=== [3] v4 check + bench ==="
timeout 300 python -u - <<'PY'
import torch, sys
sys.path.insert(0, '.')
from fa_fwd4 import fa_fwd

torch.manual_seed(42)
DOC_LENS = [4096, 217, 8192, 1500, 513, 2800, 6144, 700, 1777, 257, 3500, 1024, 2048]
H, D = 16, 128
T = sum(DOC_LENS)
scale = 1.0 / (D ** 0.5)

q = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
cu = torch.tensor([0] + [sum(DOC_LENS[:i+1]) for i in range(len(DOC_LENS))], dtype=torch.int32, device="cuda")

def naive_causal_mha(q, k, v, cu, scale):
    out = torch.zeros_like(q, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    for doc in range(len(cu) - 1):
        s, e = cu[doc].item(), cu[doc + 1].item()
        L = e - s
        for h in range(H):
            Q, K, V = qf[s:e, h, :], kf[s:e, h, :], vf[s:e, h, :]
            S = (Q @ K.T) * scale
            mask = torch.triu(torch.ones(L, L, device=S.device, dtype=torch.bool), diagonal=1)
            S.masked_fill_(mask, float('-inf'))
            P = torch.softmax(S, dim=-1)
            P = torch.where(torch.isnan(P), torch.zeros_like(P), P)
            out[s:e, h, :] = P @ V
    return out.to(q.dtype)

naive_out = naive_causal_mha(q, k, v, cu, scale)
out = fa_fwd("./libfa_fwd4.so", q, k, v, cu, cu, scale)

err = (out.float() - naive_out.float()).abs().max().item()
E_ref = float(open("E_ref.txt").read().strip())
tol = 7 * E_ref
status = "PASS" if err <= tol else "FAIL"
print(f"[CHECK] max|new - naive_fp32| = {err:.6e}")
print(f"[CHECK] tol = 7*E_ref = {tol:.6e}  ->  {status}")
if err > tol:
    sys.exit(1)

import torch.profiler as profiler
cache = torch.empty(512*1024*1024, dtype=torch.int8, device="cuda")
def flush():
    cache.zero_()

for _ in range(3):
    fa_fwd("./libfa_fwd4.so", q, k, v, cu, cu, scale)

times = []
for _ in range(50):
    flush()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    fa_fwd("./libfa_fwd4.so", q, k, v, cu, cu, scale)
    e1.record()
    torch.cuda.synchronize()
    times.append(e0.elapsed_time(e1))
times.sort()
print(f"[NEW] cuda_event_median = {times[len(times)//2]:.4f} ms")

with profiler.profile(activities=[profiler.ProfilerActivity.CUDA], with_stack=False) as p:
    for _ in range(10):
        flush()
        torch.cuda.synchronize()
        fa_fwd("./libfa_fwd4.so", q, k, v, cu, cu, scale)
torch.cuda.synchronize()

evs = [e for e in p.events() if e.device_type == profiler.DeviceType.CUDA and e.is_legacy]
kern = [e for e in evs if 'fa_fwd_kernel' in e.name]
if kern:
    ms = sum(e.cuda_time_us for e in kern) / 10 / 1000
    ref_span = float(open("ref_device_time.txt").read().strip())
    speedup = ref_span / ms
    target = ref_span / 1.2
    print(f"[NEW] profiler_kernel_span = {ms:.4f} ms   (kernels/iter = {len(kern)//10})")
    print(f"[NEW] ref span {ref_span:.4f} ms -> speedup {speedup:.3f}x (need >= 1.2)")
    if ms <= target:
        print("[NEW] TARGET MET!")
    else:
        print(f"[NEW] need {target:.4f} ms, got {ms:.4f} ms")
PY

echo "=== all done ==="
