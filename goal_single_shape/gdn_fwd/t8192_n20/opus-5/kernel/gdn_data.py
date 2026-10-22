"""Deterministic workload generator for the GDN FWD long-sequence problem.

Shapes (fixed by the task statement):
  q, k        : (T=8192, Hqk=4, D=128)  bf16
  v           : (T=8192, Hv=8,  D=128)  bf16
  a, b        : (T=8192, Hv=8)          bf16
  A_log, dt_bias : (Hv=8,)              fp32
  state       : (N=20, Hv=8, D=128, D=128) fp32, k-last  -> [n, h, dv, dk]
  cu_seqlens  : (N+1=21,)               int64
  output      : (T=8192, Hv=8, D=128)
  new_state   : (N=20, Hv=8, D=128, D=128) fp32
"""

import torch

SEQ_LENS = [753, 371, 161, 41, 105, 66, 23, 31, 19, 15,
            2433, 39, 22, 3371, 20, 42, 219, 15, 45, 401]
T = sum(SEQ_LENS)          # 8192
N = len(SEQ_LENS)          # 20
HQK = 4
HV = 8
D = 128


def make_inputs(device="cuda", seed=0, l2norm_qk=True):
    """Build the workload. q/k are unit-L2-normalised before the bf16 cast so
    that kernels which do or do not re-normalise internally agree closely."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    q = rn(T, HQK, D)
    k = rn(T, HQK, D)
    if l2norm_qk:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
    v = rn(T, HV, D)
    # `a` is a projection output in a real model, so it is small; keeping it
    # small also keeps the per-token log-decay in the range where a chunked
    # kernel can evaluate exp(G_i - G_j) without overflowing fp32
    # (|g| * chunk must stay well under 88).
    a = rn(T, HV) * 0.25
    b = rn(T, HV)

    # Qwen3-Next style initialisation of the gate parameters.
    A = torch.empty(HV).uniform_(1.0, 8.0, generator=g)
    A_log = A.log()
    dt = torch.empty(HV).uniform_(0.001, 0.02, generator=g)
    dt_bias = dt + torch.log(-torch.expm1(-dt))  # inverse softplus

    state = rn(N, HV, D, D) * 0.25

    cu = torch.zeros(N + 1, dtype=torch.int64)
    cu[1:] = torch.tensor(SEQ_LENS, dtype=torch.int64).cumsum(0)

    out = dict(
        q=q.to(torch.bfloat16), k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16), b=b.to(torch.bfloat16),
        A_log=A_log, dt_bias=dt_bias, state=state, cu_seqlens=cu,
    )
    return {kk: vv.to(device) for kk, vv in out.items()}
