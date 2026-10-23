"""Forward pass producing O (bf16) and LSE (fp32, natural log) for the backward.

Not timed. Chunked over query rows; QK^T and PV use bf16 tensor cores with fp32
accumulate, softmax statistics in fp32 -- i.e. the same numerics a real fused
forward kernel produces.
"""

import torch


def attn_fwd_lse(q, k, v, softmax_scale, causal=True, q_chunk=2048):
    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    group = Hq // Hkv
    o = torch.empty_like(q)
    lse = torch.empty(B, Hq, S, device=q.device, dtype=torch.float32)
    for b in range(B):
        for hq in range(Hq):
            hkv = hq // group
            kk = k[b, :, hkv]           # (S, D) bf16
            vv = v[b, :, hkv]
            for i0 in range(0, S, q_chunk):
                i1 = min(i0 + q_chunk, S)
                n = i1 if causal else S
                s = (q[b, i0:i1, hq] @ kk[:n].transpose(0, 1)).float() * softmax_scale
                if causal:
                    rows = torch.arange(i0, i1, device=q.device).unsqueeze(1)
                    cols = torch.arange(0, n, device=q.device).unsqueeze(0)
                    s.masked_fill_(cols > rows, float("-inf"))
                m = s.amax(dim=-1, keepdim=True)
                p = torch.exp(s - m)
                l = p.sum(dim=-1, keepdim=True)
                lse[b, hq, i0:i1] = (m + torch.log(l)).squeeze(1)
                o[b, i0:i1, hq] = ((p / l).to(torch.bfloat16) @ vv[:n]).to(torch.bfloat16)
                del s, p
    return o, lse
