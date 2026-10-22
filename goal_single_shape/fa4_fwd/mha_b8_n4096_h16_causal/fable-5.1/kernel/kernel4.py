"""FlashAttention fwd causal (bf16) v4.

Adds over v3: maxnreg tuning, the split/permute accumulator rescale (needed for
Blackwell warp specialization to keep acc in tmem-friendly layout), scale folded
into q on the host, and an optional packed f16x2 exp2 path for the softmax.
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

LOG2E = 1.4426950408889634


@triton.jit
def _exp2_fast(x, USE_F16X2: tl.constexpr):
    if USE_F16X2:
        # ex2.approx.f16x2: two exponentials per instruction. Inputs here are
        # <= 0 so results land in (0, 1] -- well inside fp16 range.
        e = tl.inline_asm_elementwise(
            "ex2.approx.f16x2 $0, $1;", "=r,r", [x.to(tl.float16)],
            dtype=tl.float16, is_pure=True, pack=2)
    else:
        e = tl.math.exp2(x)
    return e


@triton.jit
def _rescale(acc, alpha, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
             SPLIT_RESCALE: tl.constexpr):
    if SPLIT_RESCALE:
        a0, a1 = acc.reshape([BLOCK_M, 2, HEAD_DIM // 2]).permute(0, 2, 1).split()
        a0 = a0 * alpha[:, None]
        a1 = a1 * alpha[:, None]
        r = tl.join(a0, a1).permute(0, 2, 1).reshape([BLOCK_M, HEAD_DIM])
    else:
        r = acc * alpha[:, None]
    return r


@triton.jit
def _fa_fwd_v4(
    desc_q, desc_k, desc_v, desc_o,
    qk_scale,
    H: tl.constexpr, S: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    WARP_SPEC: tl.constexpr, SPLIT_RESCALE: tl.constexpr, USE_F16X2: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    q_row0 = start_m * BLOCK_M

    q = desc_q.load([b, q_row0, h, 0]).reshape([BLOCK_M, HEAD_DIM])

    offs_m = q_row0 + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in tl.range(0, q_row0, BLOCK_N, warp_specialize=WARP_SPEC):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([b, start_n, h, 0]).reshape([BLOCK_N, HEAD_DIM])
        qk = tl.dot(q, k.T) * qk_scale
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = _exp2_fast(qk - m_new[:, None], USE_F16X2)
        l_i = l_i * alpha + tl.sum(p.to(tl.float32), 1)
        acc = _rescale(acc, alpha, BLOCK_M, HEAD_DIM, SPLIT_RESCALE)
        v = desc_v.load([b, start_n, h, 0]).reshape([BLOCK_N, HEAD_DIM])
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    for start_n in tl.range(q_row0, q_row0 + BLOCK_M, BLOCK_N, warp_specialize=WARP_SPEC):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([b, start_n, h, 0]).reshape([BLOCK_N, HEAD_DIM])
        qk = tl.dot(q, k.T) * qk_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], qk, -1.0e6)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = _exp2_fast(qk - m_new[:, None], USE_F16X2)
        l_i = l_i * alpha + tl.sum(p.to(tl.float32), 1)
        acc = _rescale(acc, alpha, BLOCK_M, HEAD_DIM, SPLIT_RESCALE)
        v = desc_v.load([b, start_n, h, 0]).reshape([BLOCK_N, HEAD_DIM])
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    desc_o.store([b, q_row0, h, 0], acc.to(desc_o.dtype).reshape([1, BLOCK_M, 1, HEAD_DIM]))


def flash_attn_fwd_v4(q, k, v, softmax_scale, BLOCK_M=128, BLOCK_N=128, num_warps=4,
                      num_stages=2, warp_spec=True, maxnreg=None,
                      use_f16x2=False, out=None):
    B, S, H, D = q.shape
    if out is None:
        out = torch.empty(B, S, H, D, device=q.device, dtype=torch.bfloat16)
    desc_q = TensorDescriptor.from_tensor(q, [1, BLOCK_M, 1, D])
    desc_k = TensorDescriptor.from_tensor(k, [1, BLOCK_N, 1, D])
    desc_v = TensorDescriptor.from_tensor(v, [1, BLOCK_N, 1, D])
    desc_o = TensorDescriptor.from_tensor(out, [1, BLOCK_M, 1, D])
    extra = {}
    if maxnreg is not None:
        extra["maxnreg"] = maxnreg
    split_rescale = warp_spec and BLOCK_M == 128 and D == 128
    _fa_fwd_v4[(S // BLOCK_M, B * H)](
        desc_q, desc_k, desc_v, desc_o,
        softmax_scale * LOG2E,
        H=H, S=S, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        WARP_SPEC=warp_spec, SPLIT_RESCALE=split_rescale, USE_F16X2=use_f16x2,
        num_warps=num_warps, num_stages=num_stages, **extra,
    )
    return out
