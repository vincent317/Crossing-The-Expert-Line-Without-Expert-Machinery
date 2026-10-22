"""Straightforward fp32 recurrent reference for GDN forward (slow, exact).

Recurrence per (sequence, v-head), state S with shape [dv, dk] (k-last):

    g_t   = -exp(A_log) * softplus(a_t + dt_bias)      # <= 0
    beta_t= sigmoid(b_t)
    S <- S * exp(g_t)
    u_t   = beta_t * (v_t - S @ k_t)
    S <- S + outer(u_t, k_t)
    o_t   = S @ (q_t * scale)

q/k heads are shared by groups of Hv/Hqk consecutive v-heads.
Everything below runs in fp32 with no chunking, so it is the numerical anchor.
"""

import torch
import torch.nn.functional as F


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + eps)


def gdn_naive_fp32(q, k, v, a, b, A_log, dt_bias, state, cu_seqlens,
                   scale=None, l2norm=False, head_map="group"):
    T, HQK, D = q.shape
    HV = v.shape[1]
    G = HV // HQK
    N = state.shape[0]
    dev = q.device
    if scale is None:
        scale = D ** -0.5

    qf = q.float()
    kf = k.float()
    if l2norm:
        qf = _l2norm(qf)
        kf = _l2norm(kf)
    vf = v.float()

    gl = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())  # (T,HV)
    beta = torch.sigmoid(b.float())                                      # (T,HV)

    if head_map == "group":
        idx = torch.arange(HV, device=dev) // G
    else:
        idx = torch.arange(HV, device=dev) % HQK
    qe = qf[:, idx, :] * scale     # (T,HV,D)
    ke = kf[:, idx, :]

    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    L = max(lens)
    # pad into (N, L, HV, *) so that the python loop is over L, not T
    pq = torch.zeros(N, L, HV, D, device=dev)
    pk = torch.zeros(N, L, HV, D, device=dev)
    pv = torch.zeros(N, L, HV, D, device=dev)
    pg = torch.zeros(N, L, HV, device=dev)
    pb = torch.zeros(N, L, HV, device=dev)
    for n in range(N):
        s, e = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        ln = e - s
        pq[n, :ln] = qe[s:e]
        pk[n, :ln] = ke[s:e]
        pv[n, :ln] = vf[s:e]
        pg[n, :ln] = gl[s:e]
        pb[n, :ln] = beta[s:e]

    S = state.float().clone()                       # (N,HV,D,D) [dv,dk]
    po = torch.zeros(N, L, HV, D, device=dev)
    alive = torch.tensor(lens, device=dev).view(N, 1)
    for t in range(L):
        m = (alive > t).float().view(N, 1, 1, 1)
        dec = pg[:, t].exp().view(N, HV, 1, 1)
        S = S * torch.where(m > 0, dec, torch.ones_like(dec))
        kt = pk[:, t]                                # (N,HV,D)
        vt = pv[:, t]
        sk = torch.einsum('nhvd,nhd->nhv', S, kt)    # (N,HV,dv)
        u = (vt - sk) * pb[:, t].unsqueeze(-1)
        u = u * m.view(N, 1, 1)
        S = S + u.unsqueeze(-1) * kt.unsqueeze(-2)
        po[:, t] = torch.einsum('nhvd,nhd->nhv', S, pq[:, t])

    o = torch.zeros(T, HV, D, device=dev)
    for n in range(N):
        s, e = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        o[s:e] = po[n, : e - s]
    return o, S
