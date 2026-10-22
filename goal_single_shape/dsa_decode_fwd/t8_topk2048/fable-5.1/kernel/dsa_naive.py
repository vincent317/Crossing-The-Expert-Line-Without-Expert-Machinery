"""Slow, obviously-correct reference for DSA sparse attention (fp32 or fp64 math).

Mirrors the definition: for each token, gather the valid (!= -1) KV rows,
logits = (q_nope @ ckv^T + q_pe @ kpe^T) * sm_scale, softmax over the valid
rows, output = P @ ckv, lse = log2(sum(exp(logits))).
"""

import math

import torch


@torch.no_grad()
def naive_attention(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, dtype=torch.float32):
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # keep fp32 matmul exact-ish, no TF32
    try:
        num_tokens, num_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        kc_all = ckv_cache.reshape(-1, head_dim_ckv)
        kp_all = kpe_cache.reshape(-1, head_dim_kpe)
        output = torch.zeros((num_tokens, num_heads, head_dim_ckv), dtype=dtype, device=q_nope.device)
        lse = torch.full((num_tokens, num_heads), -float("inf"), dtype=dtype, device=q_nope.device)
        for t in range(num_tokens):
            idx = sparse_indices[t]
            valid = idx[idx != -1].to(torch.long)
            if valid.numel() == 0:
                continue
            kc = kc_all[valid].to(dtype)
            kp = kp_all[valid].to(dtype)
            qn = q_nope[t].to(dtype)
            qp = q_pe[t].to(dtype)
            logits = (qn @ kc.T + qp @ kp.T) * sm_scale
            m = logits.max(dim=-1, keepdim=True).values
            p = torch.exp(logits - m)
            l = p.sum(dim=-1, keepdim=True)
            output[t] = (p @ kc) / l
            lse[t] = (m.squeeze(-1) + torch.log(l.squeeze(-1))) / math.log(2.0)
        return output, lse
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
