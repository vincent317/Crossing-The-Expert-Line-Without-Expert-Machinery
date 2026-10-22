"""fp32 naive (slow, unfused) causal attention backward -- the accuracy baseline.

Math, per (batch, head):
  s  = scale * q @ k^T,  masked (causal) -> -inf
  p  = softmax(s, dim=-1)
  o  = p @ v
  dv = p^T @ do
  dp = do @ v^T
  ds = p * (dp - rowsum(p*dp))
  dq = scale * ds @ k
  dk = scale * ds^T @ q
Everything in fp32 with tf32 disabled, full S x S score matrix materialised.
"""
import torch

from common import B, D, H, S, SCALE


@torch.no_grad()
def naive_bwd_fp32(q, k, v, do, s_chunk: int = 2048):
    """q/k/v/do: (B, S, H, D) any dtype. Returns dq, dk, dv in fp32, same layout."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        b, s, h, d = q.shape
        dq = torch.zeros((b, s, h, d), dtype=torch.float32, device=q.device)
        dk = torch.zeros((b, s, h, d), dtype=torch.float32, device=q.device)
        dv = torch.zeros((b, s, h, d), dtype=torch.float32, device=q.device)
        idx = torch.arange(s, device=q.device)
        for bi in range(b):
            for hi in range(h):
                qh = q[bi, :, hi, :].float()
                kh = k[bi, :, hi, :].float()
                vh = v[bi, :, hi, :].float()
                doh = do[bi, :, hi, :].float()
                dkh = torch.zeros_like(kh)
                dvh = torch.zeros_like(vh)
                for i0 in range(0, s, s_chunk):
                    i1 = min(i0 + s_chunk, s)
                    qi = qh[i0:i1]
                    doi = doh[i0:i1]
                    sc = (qi @ kh.T) * SCALE
                    mask = idx[i0:i1, None] < idx[None, :]
                    sc = sc.masked_fill(mask, float("-inf"))
                    p = torch.softmax(sc, dim=-1)
                    dvh += p.T @ doi
                    dp = doi @ vh.T
                    ds = p * (dp - (p * dp).sum(-1, keepdim=True))
                    dq[bi, i0:i1, hi, :] = (ds @ kh) * SCALE
                    dkh += (ds.T @ qi) * SCALE
                    del sc, p, dp, ds
                dk[bi, :, hi, :] = dkh
                dv[bi, :, hi, :] = dvh
        return dq, dk, dv
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def compare(name_a, grads_a, name_b, grads_b):
    errs = []
    for nm, ga, gb in zip(("dq", "dk", "dv"), grads_a, grads_b):
        e = max_err(ga, gb)
        errs.append(e)
        print(f"  max|{nm}_{name_a} - {nm}_{name_b}| = {e:.6e}")
    m = max(errs)
    print(f"  overall max err ({name_a} vs {name_b}) = {m:.6e}")
    return m
