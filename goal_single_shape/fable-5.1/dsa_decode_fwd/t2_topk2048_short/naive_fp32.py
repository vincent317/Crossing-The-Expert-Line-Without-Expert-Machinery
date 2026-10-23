"""Slow, obviously-correct fp32 implementation of the DSA sparse attention definition.

Written from the definition text only: gather the valid (!= -1) top-k rows, fp32
logits, fp32 softmax with max subtraction, fp32 weighted sum, 2-based LSE.
Output is kept in fp32 (not rounded to bf16) so it can serve as the accuracy truth.
"""
import math

import torch


@torch.no_grad()
def dsa_naive_fp32(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    T, H, D = q_nope.shape
    Dp = q_pe.shape[-1]
    kc_all = ckv_cache.reshape(-1, D).float()
    kp_all = kpe_cache.reshape(-1, Dp).float()
    out = torch.zeros((T, H, D), dtype=torch.float32, device=q_nope.device)
    lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device=q_nope.device)
    for t in range(T):
        idx = sparse_indices[t]
        valid = idx[idx != -1].long()
        if valid.numel() == 0:
            continue
        kc = kc_all[valid]  # [n, D]
        kp = kp_all[valid]  # [n, Dp]
        for h in range(H):
            logits = (kc @ q_nope[t, h].float() + kp @ q_pe[t, h].float()) * sm_scale  # [n]
            mx = logits.max()
            p = torch.exp(logits - mx)
            s = p.sum()
            out[t, h] = (p / s) @ kc
            lse[t, h] = (mx + torch.log(s)) / math.log(2.0)
    return out, lse
