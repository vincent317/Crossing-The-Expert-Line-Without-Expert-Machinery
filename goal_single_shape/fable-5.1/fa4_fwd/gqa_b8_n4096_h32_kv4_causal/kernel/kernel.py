"""From-scratch FlashAttention forward for head_dim=256 causal MHA, bf16.

Written against the standard streaming-softmax formulation:
  for each Q tile, walk the allowed K/V tiles, keep running (m, l) statistics
  and rescale the O accumulator on the fly.
Layout is (B, S, H, D) so every tensor is addressed through explicit strides.

Implemented in Triton so the Blackwell (sm100) tcgen05 MMA + TMEM accumulator
path is reachable; tile shape / pipelining / warp specialization are exposed as
knobs and swept by bench_mine.py.
"""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _walk_kv(acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_vn,
             start_m, qk_scale, offs_m, offs_n, offs_d,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             HEAD_DIM: tl.constexpr, STAGE: tl.constexpr,
             WARP_SPEC: tl.constexpr):
    """One phase of the K/V walk. STAGE 1 = fully unmasked, STAGE 2 = diagonal."""
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    else:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M

    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=WARP_SPEC):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        # K is loaded already transposed: (HEAD_DIM, BLOCK_N)
        k = tl.load(k_base + n_idx[None, :] * stride_kn + offs_d[:, None])
        qk = tl.dot(q, k)
        if STAGE == 2:
            keep = offs_m[:, None] >= n_idx[None, :]
            qk = qk * qk_scale + tl.where(keep, 0.0, -1.0e6)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_new[:, None]
        else:
            m_new = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_new[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(v_base + n_idx[:, None] * stride_vn + offs_d[None, :])
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, Out, qk_scale,
              sqb, sqs, sqh, skb, sks, skh, svb, svs, svh, sob, sos, soh,
              H: tl.constexpr, N_M: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              HEAD_DIM: tl.constexpr, WARP_SPEC: tl.constexpr,
              REVERSE_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    # Causal work per Q tile grows with the tile index; emitting the heavy
    # tiles first shortens the tail of the last wave.
    start_m = (N_M - 1 - pid_m) if REVERSE_M else pid_m
    b = pid_bh // H
    h = pid_bh % H

    q_base = Q + b * sqb + h * sqh
    k_base = K + b * skb + h * skh
    v_base = V + b * svb + h * svh
    o_base = Out + b * sob + h * soh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_base + offs_m[:, None] * sqs + offs_d[None, :])

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    acc, l_i, m_i = _walk_kv(acc, l_i, m_i, q, k_base, v_base, sks, svs,
                             start_m, qk_scale, offs_m, offs_n, offs_d,
                             BLOCK_M, BLOCK_N, HEAD_DIM, 1, WARP_SPEC)
    acc, l_i, m_i = _walk_kv(acc, l_i, m_i, q, k_base, v_base, sks, svs,
                             start_m, qk_scale, offs_m, offs_n, offs_d,
                             BLOCK_M, BLOCK_N, HEAD_DIM, 2, WARP_SPEC)

    acc = acc / l_i[:, None]
    tl.store(o_base + offs_m[:, None] * sos + offs_d[None, :],
             acc.to(Out.dtype.element_ty))


def flash_attn_mine(q, k, v, scale, causal=True, BLOCK_M=128, BLOCK_N=64,
                    num_warps=8, num_stages=2, warp_specialize=False,
                    num_ctas=1, reverse_m=True, out=None):
    assert causal, "this kernel is specialized for the causal case"
    B, S, H, D = q.shape
    assert S % BLOCK_M == 0 and BLOCK_M % BLOCK_N == 0, (S, BLOCK_M, BLOCK_N)
    if out is None:
        out = torch.empty_like(q)
    grid = (S // BLOCK_M, B * H)
    _attn_fwd[grid](
        q, k, v, out, scale * LOG2E,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H=H, N_M=S // BLOCK_M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        WARP_SPEC=warp_specialize, REVERSE_M=reverse_m,
        num_warps=num_warps, num_stages=num_stages, num_ctas=num_ctas,
    )
    return out
