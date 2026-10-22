"""FlashAttention forward v4: static softmax offset instead of a running max.

Why: for this shape the cost is not the MMA, it is the per-score-element non-MMA work
plus the serial dependency that the running max creates. Keeping a running max forces
`acc = acc * alpha` every tile, which drags the fp32 accumulator out of tensor memory
into registers and back on every iteration and serialises the tiles.

Instead each query row gets a *static* offset M_row that provably dominates every score
in that row, so exp2(qk*c - M_row) never overflows and no rescaling is ever needed:

    |q . k_j| <= ||q|| * ||k_j|| <= ||q|| * max_j ||k_j||        (Cauchy-Schwarz)

`_kmax_kernel` precomputes max_j ||k_j|| per (batch, head); ||q_row|| comes for free from
the already-loaded Q tile. The offset is additionally capped at (first-tile max + SLACK)
so a loose Cauchy-Schwarz bound can never push the whole row into denormals.

With no rescale the accumulator stays in tmem for the whole reduction and the loop body
is independent across tiles apart from the two accumulations.

Tuning knobs (all constexpr):
  LOOP_ST   pipeliner depth of the K/V loop (separate from the launch num_stages)
  LSUM_MMA  compute the row sums with a tiny extra MMA (P @ ones) instead of a
            cross-lane reduction, moving work off the ALU/shuffle path
  ACC_1BUF  forbid tmem multi-buffering of the accumulator (frees tmem columns)
"""
import torch
import triton
import triton.language as tl

import tma_alloc  # noqa: F401  installs the TMA scratch allocator

LOG2E = 1.4426950408889634
# Cap on how far the static offset may sit above the first tile's max, in log2 units.
# Keeps exp2 arguments inside fp32's normal range from both ends.
SLACK = 48.0


@triton.jit
def _kmax_kernel(K, Out, s_kb, s_ks, s_kh,
                 S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs_d = tl.arange(0, D)
    base = K + b * s_kb + h * s_kh
    best = 0.0
    for s0 in tl.range(0, S, BLOCK):
        rows = s0 + tl.arange(0, BLOCK)
        kk = tl.load(base + rows[:, None] * s_ks + offs_d[None, :]).to(tl.float32)
        best = tl.maximum(best, tl.max(tl.sum(kk * kk, 1)))
    tl.store(Out + pid, tl.sqrt(best))


@triton.jit
def _attn_fwd_v4(Q, K, V, Out, Kmax, qk_scale,
                 s_qb, s_qs, s_qh, s_kb, s_ks, s_kh, s_vb, s_vs, s_vh,
                 s_ob, s_os, s_oh,
                 B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 WS: tl.constexpr, SLACK: tl.constexpr, LOOP_ST: tl.constexpr,
                 LSUM_MMA: tl.constexpr, ACC_1BUF: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_desc = tl.make_tensor_descriptor(Q, [B, S, H, D], [s_qb, s_qs, s_qh, 1],
                                       [1, BLOCK_M, 1, D])
    k_desc = tl.make_tensor_descriptor(K, [B, S, H, D], [s_kb, s_ks, s_kh, 1],
                                       [1, BLOCK_N, 1, D])
    v_desc = tl.make_tensor_descriptor(V, [B, S, H, D], [s_vb, s_vs, s_vh, 1],
                                       [1, BLOCK_N, 1, D])
    o_desc = tl.make_tensor_descriptor(Out, [B, S, H, D], [s_ob, s_os, s_oh, 1],
                                       [1, BLOCK_M, 1, D])

    m0 = pid_m * BLOCK_M
    q = tl.reshape(q_desc.load([b, m0, h, 0]), [BLOCK_M, D])
    qf = q.to(tl.float32)
    qnorm = tl.sqrt(tl.sum(qf * qf, 1))
    kmax = tl.load(Kmax + pid_bh)
    ones = tl.full([BLOCK_N, 16], 1.0, dtype=Q.dtype.element_ty)

    # Prologue on tile 0: fixes the static offset, then folds straight into the sums.
    k0 = tl.reshape(k_desc.load([b, 0, h, 0]), [BLOCK_N, D])
    qk0 = tl.dot(q, tl.trans(k0))
    ft = tl.max(qk0, 1) * qk_scale
    m_off = tl.minimum(qnorm * (kmax * qk_scale), ft + SLACK)
    p0 = tl.exp2(qk0 * qk_scale - m_off[:, None])
    pb0 = p0.to(Q.dtype.element_ty)
    v0 = tl.reshape(v_desc.load([b, 0, h, 0]), [BLOCK_N, D])
    acc = tl.dot(pb0, v0)
    if LSUM_MMA:
        lacc = tl.dot(pb0, ones)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    else:
        lacc = tl.zeros([BLOCK_M, 16], dtype=tl.float32)
        l_i = tl.sum(p0, 1)

    for start_n in tl.range(BLOCK_N, S, BLOCK_N, num_stages=LOOP_ST, warp_specialize=WS,
                            disallow_acc_multi_buffer=ACC_1BUF):
        k = tl.reshape(k_desc.load([b, start_n, h, 0]), [BLOCK_N, D])
        qk = tl.dot(q, tl.trans(k))
        p = tl.exp2(qk * qk_scale - m_off[:, None])
        pb = p.to(Q.dtype.element_ty)
        if LSUM_MMA:
            lacc = tl.dot(pb, ones, lacc)
        else:
            l_i += tl.sum(p, 1)
        v = tl.reshape(v_desc.load([b, start_n, h, 0]), [BLOCK_N, D])
        acc = tl.dot(pb, v, acc)

    if LSUM_MMA:
        l_i = tl.sum(lacc, 1) * (1.0 / 16)
    acc = acc / l_i[:, None]
    o_desc.store([b, m0, h, 0],
                 tl.reshape(acc.to(Out.dtype.element_ty), [1, BLOCK_M, 1, D]))


def attention_v4(q, k, v, scale, BLOCK_M=128, BLOCK_N=128, num_warps=4, num_stages=2,
                 warp_specialize=True, loop_stages=None, lsum_mma=False, acc_1buf=False,
                 out=None, kmax_block=256, kmax_warps=4):
    B, S, H, D = q.shape
    o = torch.empty_like(q) if out is None else out
    kmax = torch.empty(B * H, device=q.device, dtype=torch.float32)
    _kmax_kernel[(B * H,)](
        k, kmax, k.stride(0), k.stride(1), k.stride(2),
        S=S, H=H, D=D, BLOCK=kmax_block, num_warps=kmax_warps,
    )
    _attn_fwd_v4[(S // BLOCK_M, B * H)](
        q, k, v, o, kmax, scale * LOG2E,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        B=B, S=S, H=H, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        WS=warp_specialize, SLACK=SLACK,
        LOOP_ST=num_stages if loop_stages is None else loop_stages,
        LSUM_MMA=lsum_mma, ACC_1BUF=acc_1buf,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o
