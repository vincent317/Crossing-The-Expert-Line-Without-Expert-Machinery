"""fp32 naive reference for FlashAttention backward (GQA, causal).

Pure PyTorch, fp32 math, no tensor-core TF32. Chunked over query rows so the
causal upper triangle is never materialized and memory stays bounded.
"""

import torch


def naive_bwd_fp32(q, k, v, dout, softmax_scale, causal=True, q_chunk=2048):
    """q,dout: (B,S,Hq,D); k,v: (B,S,Hkv,D). Returns dq,dk,dv in fp32.

    All reductions in fp32. Query heads map to kv heads by contiguous grouping
    (hq // (Hq // Hkv)), which is the standard GQA layout.
    """
    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    group = Hq // Hkv

    dq = torch.zeros(B, S, Hq, D, dtype=torch.float32, device=q.device)
    dk = torch.zeros(B, S, Hkv, D, dtype=torch.float32, device=q.device)
    dv = torch.zeros(B, S, Hkv, D, dtype=torch.float32, device=q.device)

    for b in range(B):
        for hq in range(Hq):
            hkv = hq // group
            kf = k[b, :, hkv].float()  # (S, D)
            vf = v[b, :, hkv].float()
            dk_acc = torch.zeros(S, D, dtype=torch.float32, device=q.device)
            dv_acc = torch.zeros(S, D, dtype=torch.float32, device=q.device)

            for i0 in range(0, S, q_chunk):
                i1 = min(i0 + q_chunk, S)
                qc = q[b, i0:i1, hq].float()  # (M, D)
                doc = dout[b, i0:i1, hq].float()
                n = i1 if causal else S
                kc = kf[:n]
                vc = vf[:n]

                s = (qc @ kc.transpose(0, 1)) * softmax_scale  # (M, n)
                if causal:
                    rows = torch.arange(i0, i1, device=q.device).unsqueeze(1)
                    cols = torch.arange(0, n, device=q.device).unsqueeze(0)
                    s = s.masked_fill(cols > rows, float("-inf"))
                p = torch.softmax(s, dim=-1)

                o = p @ vc                              # (M, D)
                dp = doc @ vc.transpose(0, 1)           # (M, n)
                dcorr = (doc * o).sum(-1, keepdim=True)  # (M, 1)
                ds = p * (dp - dcorr)                   # (M, n)

                dq[b, i0:i1, hq] = (ds @ kc) * softmax_scale
                dk_acc[:n] += (ds.transpose(0, 1) @ qc) * softmax_scale
                dv_acc[:n] += p.transpose(0, 1) @ doc

                del s, p, o, dp, ds

            dk[b, :, hkv] += dk_acc
            dv[b, :, hkv] += dv_acc

    return dq, dk, dv
