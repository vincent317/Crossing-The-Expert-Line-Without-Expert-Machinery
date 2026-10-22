"""Slow but trustworthy fp32 causal attention reference.

Pure fp32 math (TF32 disabled), full softmax in fp32, chunked over queries to
bound memory. Used as the ground truth for accuracy comparisons.
"""
import torch


def naive_attention_fp32(q, k, v, softmax_scale, causal=True, q_chunk=512):
    """q,k,v: (B, S, H, D) any dtype -> out (B, S, H, D) float32.

    Computes attention independently per (batch, head) in float32.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    B, S, H, D = q.shape
    out = torch.empty(B, S, H, D, device=q.device, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            qh = q[b, :, h, :].float()          # (S, D)
            kh = k[b, :, h, :].float()
            vh = v[b, :, h, :].float()
            for s0 in range(0, S, q_chunk):
                s1 = min(s0 + q_chunk, S)
                kv_end = s1 if causal else S
                scores = (qh[s0:s1] @ kh[:kv_end].T) * softmax_scale  # (c, kv_end)
                if causal:
                    rows = torch.arange(s0, s1, device=q.device).unsqueeze(1)
                    cols = torch.arange(kv_end, device=q.device).unsqueeze(0)
                    scores = scores.masked_fill(cols > rows, float("-inf"))
                p = torch.softmax(scores, dim=-1)
                out[b, s0:s1, h, :] = p @ vh[:kv_end]
    return out


def max_abs_err(a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    return d.max().item(), d.mean().item()
