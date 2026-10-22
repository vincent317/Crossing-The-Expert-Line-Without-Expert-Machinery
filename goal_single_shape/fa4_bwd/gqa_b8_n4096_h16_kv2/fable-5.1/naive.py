"""fp32 naive reference forward/backward for GQA causal attention. Slow but simple."""
import math

import torch

from common import B, CAUSAL, D, GQA, HKV, HQ, S, SCALE

LOG2E = 1.4426950408889634


def _mask(device):
    idx = torch.arange(S, device=device)
    return idx[:, None] >= idx[None, :]


@torch.no_grad()
def naive_fwd_fp32(q, k, v):
    """Returns o (bf16, B,S,HQ,D) and lse2 (fp32, B,HQ,S) in log2 units."""
    o = torch.empty(B, S, HQ, D, device=q.device, dtype=torch.bfloat16)
    lse2 = torch.empty(B, HQ, S, device=q.device, dtype=torch.float32)
    m = _mask(q.device)
    for b in range(B):
        for h in range(HQ):
            hk = h // GQA
            qi = q[b, :, h, :].float()
            ki = k[b, :, hk, :].float()
            vi = v[b, :, hk, :].float()
            s = (qi @ ki.T) * SCALE
            if CAUSAL:
                s = s.masked_fill(~m, float("-inf"))
            mx = s.max(-1, keepdim=True).values
            e = torch.exp(s - mx)
            ssum = e.sum(-1, keepdim=True)
            o[b, :, h, :] = ((e / ssum) @ vi).to(torch.bfloat16)
            lse2[b, h, :] = ((mx + ssum.log()).squeeze(-1)) * LOG2E
    return o, lse2


@torch.no_grad()
def naive_bwd_fp32(q, k, v, do):
    """q,do: (B,S,HQ,D); k,v: (B,S,HKV,D). Returns dq,dk,dv in fp32."""
    dq = torch.zeros(B, S, HQ, D, device=q.device, dtype=torch.float32)
    dk = torch.zeros(B, S, HKV, D, device=q.device, dtype=torch.float32)
    dv = torch.zeros(B, S, HKV, D, device=q.device, dtype=torch.float32)
    m = _mask(q.device)
    for b in range(B):
        for h in range(HQ):
            hk = h // GQA
            qi = q[b, :, h, :].float()
            ki = k[b, :, hk, :].float()
            vi = v[b, :, hk, :].float()
            doi = do[b, :, h, :].float()
            s = (qi @ ki.T) * SCALE
            if CAUSAL:
                s = s.masked_fill(~m, float("-inf"))
            p = torch.softmax(s, dim=-1)
            dv[b, :, hk, :] += p.T @ doi
            dp = doi @ vi.T
            delta = (p * dp).sum(-1, keepdim=True)
            ds = p * (dp - delta)
            dq[b, :, h, :] = (ds @ ki) * SCALE
            dk[b, :, hk, :] += (ds.T @ qi) * SCALE
    return dq, dk, dv


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item()
