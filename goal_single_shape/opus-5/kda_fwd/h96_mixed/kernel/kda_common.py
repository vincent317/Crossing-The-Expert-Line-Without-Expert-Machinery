"""Shared problem definition for the KDA forward benchmark (case h96_mixed).

Layout / shapes follow the task statement:
  q/k/v/g : (1, 8192, 96, 128) bf16
  beta    : (1, 8192, 96)      bf16
  A_log   : (96,)              fp32
  dt_bias : (96, 128)          fp32
  state   : (6, 96, 128, 128)  fp32, updated in place
  cu_seqlens : (7,) int64
  out     : (1, 8192, 96, 128) bf16
"""

import math
from itertools import accumulate

import torch
import torch.nn.functional as F

SEQ_LENS = [1300, 547, 2048, 963, 271, 3063]
B, T, H, DK, DV = 1, 8192, 96, 128, 128
N_SEQ = len(SEQ_LENS)
assert sum(SEQ_LENS) == T


def make_inputs(device="cuda", dtype=torch.bfloat16, seed=1234):
    """Deterministic h96_mixed-style inputs.

    A_log / dt_bias follow the documented KDA initialisation:
      A_log   = log(U(1, 16))                       -> A in [1, 16]
      dt_bias = inv_softplus(exp(U(log 1e-3, log 0.1)))
    so that the per-token log decay  -A * softplus(g + dt_bias)  lands in a
    realistic range instead of a degenerate one.
    """
    cpu = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=cpu, dtype=torch.float32)

    q = rn(B, T, H, DK).to(device=device, dtype=dtype)
    k = rn(B, T, H, DK).to(device=device, dtype=dtype)
    v = rn(B, T, H, DV).to(device=device, dtype=dtype)
    g = rn(B, T, H, DK).to(device=device, dtype=dtype)
    beta_logit = rn(B, T, H)
    beta = beta_logit.sigmoid().to(device=device, dtype=dtype)
    beta_logit = beta_logit.to(device=device, dtype=dtype)

    A_log = torch.log(torch.rand(H, generator=cpu) * 15.0 + 1.0).to(device=device, dtype=torch.float32)
    lo, hi = math.log(1e-3), math.log(0.1)
    dt = (torch.rand(H, DK, generator=cpu) * (hi - lo) + lo).exp().clamp(min=1e-4)
    dt_bias = (dt + torch.log(-torch.expm1(-dt))).to(device=device, dtype=torch.float32)

    state = (rn(N_SEQ, H, DK, DV) * 0.1).to(device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0] + list(accumulate(SEQ_LENS)), dtype=torch.int64, device=device)
    return dict(q=q, k=k, v=v, g=g, beta=beta, beta_logit=beta_logit, A_log=A_log,
                dt_bias=dt_bias, state=state, cu_seqlens=cu_seqlens)


def kda_log_decay(g, A_log, dt_bias, lower_bound=None):
    """log(alpha), shape (B, T, H, DK), fp32.

    lower_bound is None -> -exp(A_log) * softplus(g + dt_bias)              (softplus gate)
    lower_bound set     -> lower_bound * sigmoid(exp(A_log) * (g + dt_bias)) (safe gate)
    """
    A = torch.exp(A_log.float())[None, None, :, None]
    x = g.float() + dt_bias.float()[None, None]
    if lower_bound is None:
        return -A * F.softplus(x)
    return lower_bound * torch.sigmoid(A * x)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


@torch.no_grad()
def kda_naive_fp32(q, k, v, g, beta, A_log, dt_bias, state, cu_seqlens,
                   scale=None, use_l2norm=True, l2_eps=1e-6, lower_bound=None):
    """Token-by-token fp32 reference.

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T   (DK x DV)
    o_t = S_t^T q_t
    """
    q = q.float()
    k = k.float()
    v = v.float()
    beta = beta.float()
    if use_l2norm:
        q = l2norm(q, l2_eps)
        k = l2norm(k, l2_eps)
    if scale is None:
        scale = DK ** -0.5
    q = q * scale
    alpha = kda_log_decay(g, A_log, dt_bias, lower_bound).exp()

    out = torch.empty(B, T, H, DV, device=q.device, dtype=torch.float32)
    final = state.float().clone()
    cu = cu_seqlens.tolist()
    for n in range(len(cu) - 1):
        S = final[n]                                     # (H, DK, DV)
        for t in range(cu[n], cu[n + 1]):
            kt = k[0, t]                                 # (H, DK)
            Sa = alpha[0, t][:, :, None] * S             # (H, DK, DV)
            kv = (kt[:, :, None] * Sa).sum(1)            # (H, DV)
            S = Sa + (beta[0, t][:, None] * kt)[:, :, None] * (v[0, t] - kv)[:, None, :]
            out[0, t] = (q[0, t][:, :, None] * S).sum(1)
        final[n] = S
    return out, final


def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def device_time_ms(fn, warmup=10, iters=50, rounds=5, reset=None):
    """Median per-iteration GPU span measured with CUDA events (no host launch cost)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        if reset is not None:
            reset()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / iters)
    ts.sort()
    return ts[len(ts) // 2], ts


def profiler_kernel_ms(fn, iters=20):
    """Sum of CUDA kernel durations per iteration, from the torch profiler (ncu-style)."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot_us, names = 0.0, {}
    for e in prof.key_averages():
        if e.device_type.name == 'CUDA' and e.self_device_time_total > 0:
            tot_us += e.self_device_time_total
            names[e.key] = (e.self_device_time_total / iters, e.count // iters)
    return tot_us / 1000.0 / iters, names
