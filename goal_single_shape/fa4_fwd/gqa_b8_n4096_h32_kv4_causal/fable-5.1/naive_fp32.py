"""Slow fp32 reference for causal MHA forward.

Straightforward per-(batch, head) materialized softmax in fp32, no tensor
cores. This is the correctness anchor: every candidate kernel is compared
against this, not against the external reference.
"""
import torch

from common import CAUSAL, SCALE


def naive_fp32_attention(q, k, v, scale=SCALE, causal=CAUSAL):
    """q/k/v: (B, S, H, D) any dtype -> out fp32 (B, S, H, D)."""
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        b, s, h, d = q.shape
        out = torch.empty((b, s, h, d), dtype=torch.float32, device=q.device)
        mask = None
        if causal:
            mask = torch.triu(
                torch.ones(s, s, dtype=torch.bool, device=q.device), diagonal=1)
        for bi in range(b):
            for hi in range(h):
                qq = q[bi, :, hi, :].float()
                kk = k[bi, :, hi, :].float()
                vv = v[bi, :, hi, :].float()
                sc = (qq @ kk.transpose(0, 1)) * scale
                if mask is not None:
                    sc = sc.masked_fill(mask, float("-inf"))
                p = torch.softmax(sc, dim=-1)
                out[bi, :, hi, :] = p @ vv
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()
