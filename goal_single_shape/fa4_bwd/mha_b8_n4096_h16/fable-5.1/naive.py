"""fp32 naive (slow) reference of causal attention backward, chunked over query rows.

Everything is computed in fp32 with plain (non-flash) softmax; only inputs are bf16.
Returns dq, dk, dv in fp32 (caller may cast).
"""
import math
import torch


def naive_attn_bwd_fp32(q, k, v, do, scale, chunk=1024):
    b, s, h, d = q.shape
    # [b, h, s, d] fp32
    qf = q.float().transpose(1, 2).contiguous()
    kf = k.float().transpose(1, 2).contiguous()
    vf = v.float().transpose(1, 2).contiguous()
    dof = do.float().transpose(1, 2).contiguous()
    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)
    ar = torch.arange(s, device=q.device)
    for i0 in range(0, s, chunk):
        i1 = min(i0 + chunk, s)
        # causal: rows [i0,i1) only see cols [0,i1)
        kk = kf[:, :, :i1]
        vv = vf[:, :, :i1]
        sc = torch.matmul(qf[:, :, i0:i1], kk.transpose(-1, -2)) * scale  # [b,h,c,i1]
        mask = ar[i0:i1, None] < ar[None, :i1]
        sc.masked_fill_(mask, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        dv[:, :, :i1] += torch.matmul(p.transpose(-1, -2), dof[:, :, i0:i1])
        dp = torch.matmul(dof[:, :, i0:i1], vv.transpose(-1, -2))       # [b,h,c,i1]
        delta = (dp * p).sum(-1, keepdim=True)
        ds = p * (dp - delta) * scale
        dq[:, :, i0:i1] = torch.matmul(ds, kk)
        dk[:, :, :i1] += torch.matmul(ds.transpose(-1, -2), qf[:, :, i0:i1])
        del sc, p, dp, ds
    return (dq.transpose(1, 2).contiguous(), dk.transpose(1, 2).contiguous(),
            dv.transpose(1, 2).contiguous())


def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def report_err(name, got, ref):
    """Prints per-tensor max abs error vs fp32 reference. Returns overall max."""
    errs = []
    for tname, g, r in zip(("dq", "dk", "dv"), got, ref):
        e = max_abs_err(g, r)
        errs.append(e)
        print(f"  {name}: max|{tname} - naive_fp32| = {e:.6e}   (ref absmax {r.abs().max().item():.4e})")
    m = max(errs)
    print(f"  {name}: E = max over dq/dk/dv = {m:.6e}")
    return m
