"""fp32 naive reference attention. Deliberately slow: materializes the full
S=8192 x 8192 score matrix per (batch, head) in fp32, no tensor cores."""
import torch

B, S, HQ, HKV, D = 4, 8192, 32, 2, 128
SCALE = D ** -0.5


def naive_fp32_attention(q, k, v, scale=SCALE):
    """q: (B,S,HQ,D) k,v: (B,S,HKV,D) -> (B,S,HQ,D) fp32.

    Pure fp32 math, TF32 disabled so matmuls are true fp32.
    """
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_red = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        b, s, hq, d = q.shape
        hkv = k.shape[2]
        group = hq // hkv
        out = torch.empty(b, s, hq, d, dtype=torch.float32, device=q.device)
        qf = q.float()
        kf = k.float()
        vf = v.float()
        for bi in range(b):
            for h in range(hq):
                hk = h // group
                qi = qf[bi, :, h, :]            # (S, D)
                ki = kf[bi, :, hk, :]           # (S, D)
                vi = vf[bi, :, hk, :]           # (S, D)
                scores = (qi @ ki.transpose(0, 1)) * scale   # (S, S) fp32
                m = scores.amax(dim=-1, keepdim=True)
                p = torch.exp(scores - m)
                p = p / p.sum(dim=-1, keepdim=True)
                out[bi, :, h, :] = p @ vi
                del scores, p, m
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = prev_red


def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()
