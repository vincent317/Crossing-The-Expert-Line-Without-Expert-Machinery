"""Workload definition and fp32 naive ground truth for GDN forward.

Workload (contest 07aa7922 shape spec):
  T=5709 tokens packed into N=2 sequences of length 5637 and 72
  q/k: (T, 4, 128) bf16      v: (T, 8, 128) bf16
  a/b: (T, 8) bf16           A_log/dt_bias: (8,) fp32
  state: (N, 8, 128, 128) fp32, k-last  ->  state[n,h] is S_vk (v-major, k-last)
  cu_seqlens: (3,) int64
  out: (T, 8, 128)           new_state: (N, 8, 128, 128) fp32

Operator semantics (gated delta rule, GQA 4 kv-heads -> 8 v-heads):
  dt   = softplus(a + dt_bias)
  g    = -exp(A_log) * dt                  (log decay, per token per v-head)
  beta = sigmoid(b)
  qn   = l2normalize(q) * scale ; kn = l2normalize(k)      scale = 1/sqrt(128)
  per token t (v-head h uses kv-head h//2):
      S  = S * exp(g)
      u  = beta * (v - S @ kn)
      S  = S + outer(u, kn)
      o  = S @ qn
"""

import torch
import torch.nn.functional as F

T = 5709
N = 2
LENS = (5637, 72)
HQ = 4
HV = 8
D = 128
G = HV // HQ
SCALE = D ** -0.5


def make_inputs(device="cuda", seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    q = rn(T, HQ, D).to(device).bfloat16()
    k = rn(T, HQ, D).to(device).bfloat16()
    v = rn(T, HV, D).to(device).bfloat16()
    a = (rn(T, HV) * 0.5).to(device).bfloat16()
    b = rn(T, HV).to(device).bfloat16()
    A = torch.rand(HV, generator=gen) * 15.0 + 1.0
    A_log = torch.log(A).to(device).float()
    dt = torch.rand(HV, generator=gen) * 0.099 + 0.001
    dt_bias = (dt + torch.log(-torch.expm1(-dt))).to(device).float()  # inverse softplus
    state = (rn(N, HV, D, D) * 0.1).to(device).float()
    cu = torch.tensor([0, LENS[0], T], dtype=torch.int64, device=device)
    return dict(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
                state=state, cu_seqlens=cu)


def gates(a, b, A_log, dt_bias):
    """Return (g, beta) in fp32, shape (T, HV)."""
    dt = F.softplus(a.float() + dt_bias.float())
    g = -torch.exp(A_log.float()) * dt
    beta = torch.sigmoid(b.float())
    return g, beta


def naive_fp32(inp):
    """Token-sequential fp32 recurrence. Ground truth (slow)."""
    q, k, v = inp["q"], inp["k"], inp["v"]
    g, beta = gates(inp["a"], inp["b"], inp["A_log"], inp["dt_bias"])
    qn = (F.normalize(q.float(), dim=-1) * SCALE).repeat_interleave(G, dim=1)
    kn = F.normalize(k.float(), dim=-1).repeat_interleave(G, dim=1)
    vf = v.float()
    decay = torch.exp(g).view(T, HV, 1, 1)
    o = torch.empty(T, HV, D, device=q.device, dtype=torch.float32)
    new_state = torch.empty_like(inp["state"])
    cu = inp["cu_seqlens"].tolist()
    for n in range(N):
        S = inp["state"][n].clone()  # (HV, V, K)
        for t in range(cu[n], cu[n + 1]):
            kt = kn[t].unsqueeze(-1)  # (HV, K, 1)
            S = S * decay[t]
            u = beta[t].view(HV, 1) * (vf[t] - torch.bmm(S, kt).squeeze(-1))
            S = S + u.unsqueeze(-1) * kn[t].unsqueeze(-2)
            o[t] = torch.bmm(S, qn[t].unsqueeze(-1)).squeeze(-1)
        new_state[n] = S
    return o, new_state


def err(x, y):
    """Max abs difference; NaN-aware (returns inf if either side has NaNs)."""
    d = (x.float() - y.float()).abs()
    if not torch.isfinite(d).all():
        return float("inf")
    return d.max().item()


def err_finite(x, y):
    """(max abs diff over finite entries, number of non-finite entries in x)."""
    xf = x.float()
    d = (xf - y.float()).abs()
    good = torch.isfinite(d)
    m = d[good].max().item() if good.any() else float("nan")
    return m, int((~torch.isfinite(xf)).sum().item())
