"""Establish the tolerance: E_ref = max|ref - fp32_naive|, tol = 7 * E_ref.

Also probes whether low-precision (fp8) QK would stay inside that tolerance.
"""
import sys
import time

import torch

from naive_fp32 import naive_attention_fp32, max_abs_err

B, S, H, D = 8, 8192, 16, 128
scale = D ** -0.5
torch.manual_seed(0)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

from flash_attn.cute.interface import flash_attn_func

r = flash_attn_func(q, k, v, softmax_scale=scale, causal=True)
ref = (r[0] if isinstance(r, (tuple, list)) else r).float()
torch.cuda.synchronize()
print("ref computed", ref.shape, "abs max", ref.abs().max().item())
sys.stdout.flush()

t0 = time.time()
naive = naive_attention_fp32(q, k, v, scale, causal=True, q_chunk=512)
torch.cuda.synchronize()
print(f"fp32 naive computed in {time.time()-t0:.1f} s ; abs max {naive.abs().max().item():.6f}")

E_ref, mean_ref = max_abs_err(ref, naive)
tol = 7.0 * E_ref
print("=" * 70)
print(f"E_ref  = max|ref - fp32_naive| = {E_ref:.8f}   (mean {mean_ref:.8f})")
print(f"tol    = 7 * E_ref            = {tol:.8f}")
print("=" * 70)
sys.stdout.flush()

# ---- how much error would an fp8 QK path introduce? (feasibility probe) ----
def sim_attn(qd, kd, vd, out_dtype=torch.float32, q_chunk=512):
    out = torch.empty(B, S, H, D, device="cuda", dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            qh = qd[b, :, h, :].float()
            kh = kd[b, :, h, :].float()
            vh = vd[b, :, h, :].float()
            for s0 in range(0, S, q_chunk):
                s1 = min(s0 + q_chunk, S)
                sc = (qh[s0:s1] @ kh[:s1].T) * scale
                rows = torch.arange(s0, s1, device=qd.device).unsqueeze(1)
                cols = torch.arange(s1, device=qd.device).unsqueeze(0)
                sc = sc.masked_fill(cols > rows, float("-inf"))
                p = torch.softmax(sc, dim=-1)
                out[b, s0:s1, h, :] = p @ vh[:s1]
    return out


for name, dt in [("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)]:
    qq = q.to(dt).to(torch.bfloat16)
    kk = k.to(dt).to(torch.bfloat16)
    o = sim_attn(qq, kk, v)
    e, m = max_abs_err(o, naive)
    print(f"fp8 {name} QK only  : max err {e:.6f} (mean {m:.6f})  "
          f"{'PASS' if e <= tol else 'FAIL'} vs tol {tol:.6f}")
    sys.stdout.flush()

# bf16 QK (what my kernel does) as sanity
o = sim_attn(q, k, v)
e, m = max_abs_err(o, naive)
print(f"bf16 QK (fp32 softmax): max err {e:.6f} (mean {m:.6f})")
# bf16 output rounding alone
e2, _ = max_abs_err(naive.to(torch.bfloat16), naive)
print(f"bf16 output rounding alone: max err {e2:.6f}")
