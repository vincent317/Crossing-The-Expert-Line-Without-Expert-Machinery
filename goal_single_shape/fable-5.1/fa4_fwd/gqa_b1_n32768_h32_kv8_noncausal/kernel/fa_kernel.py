"""FlashAttention forward in Triton, written from the published algorithm.

Online-softmax streaming attention: each CTA owns a BLOCK_M slice of queries for one
head, walks the whole key/value sequence in BLOCK_N tiles, and keeps a running max /
denominator so the softmax never materialises a full S x S score matrix.

Layout is (B, S, H, D) with D contiguous, so a row of one head is addressed by
  base + s * (H*D) + h * D + d
and no pre-transpose of q/k/v is needed.
"""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _attn_fwd(Q, K, V, Out, qk_scale,
              stride_s, stride_h,
              S,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
              WARP_SPEC: tl.constexpr):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n0 = tl.arange(0, BLOCK_N)

    head_base = off_h * stride_h
    q = tl.load(Q + head_base + offs_m[:, None] * stride_s + offs_d[None, :])

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    k_base = K + head_base + offs_d[:, None]
    v_base = V + head_base + offs_d[None, :]

    for start_n in tl.range(0, S, BLOCK_N, warp_specialize=WARP_SPEC):
        offs_n = start_n + offs_n0
        # K is read transposed ([HEAD_DIM, BLOCK_N]) so tl.dot sees a K-major B operand.
        kt = tl.load(k_base + offs_n[None, :] * stride_s)
        qk = tl.dot(q, kt) * qk_scale
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + offs_n[:, None] * stride_s)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(Out + head_base + offs_m[:, None] * stride_s + offs_d[None, :],
             acc.to(Out.dtype.element_ty))


def flash_attn_fwd(q, k, v, softmax_scale=None, out=None,
                   BLOCK_M=128, BLOCK_N=128, WARP_SPEC=False,
                   num_warps=8, num_stages=2):
    """q/k/v: (B, S, H, D) bf16 contiguous, non-causal. Returns (B, S, H, D) bf16."""
    B, S, H, D = q.shape
    assert B == 1, "single batch fast path"
    assert q.stride(-1) == 1 and S % BLOCK_M == 0 and S % BLOCK_N == 0
    if softmax_scale is None:
        softmax_scale = D ** -0.5
    if out is None:
        out = torch.empty_like(q)
    grid = (S // BLOCK_M, H)
    _attn_fwd[grid](
        q, k, v, out, softmax_scale * LOG2E,
        q.stride(1), q.stride(2),
        S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, WARP_SPEC=WARP_SPEC,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
