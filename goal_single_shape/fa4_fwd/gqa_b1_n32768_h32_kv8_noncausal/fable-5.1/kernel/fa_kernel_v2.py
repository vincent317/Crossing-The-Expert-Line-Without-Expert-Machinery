"""FlashAttention forward v2: TMA descriptors + warp specialization.

Same online-softmax algorithm as v1, but tiles are fetched through 3D TMA tensor
descriptors over the (S, H, D) view so the copies are async and the loop body can be
warp-specialized (load warps overlap with MMA/softmax warps).
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

LOG2E = 1.4426950408889634


@triton.jit
def _attn_fwd_tma(desc_q, desc_k, desc_v, desc_o, qk_scale, S,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
                  WARP_SPEC: tl.constexpr, LOOP_STAGES: tl.constexpr):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    qo_off = start_m * BLOCK_M

    q = desc_q.load([qo_off, off_h, 0]).reshape([BLOCK_M, HEAD_DIM])

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in tl.range(0, S, BLOCK_N, warp_specialize=WARP_SPEC,
                            num_stages=LOOP_STAGES):
        k = desc_k.load([start_n, off_h, 0]).reshape([BLOCK_N, HEAD_DIM])
        qk = tl.dot(q, k.T) * qk_scale
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = desc_v.load([start_n, off_h, 0]).reshape([BLOCK_N, HEAD_DIM])
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    desc_o.store([qo_off, off_h, 0], acc.to(desc_o.dtype)[:, None, :])


def _alloc_fn(size, align, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


def flash_attn_fwd_v2(q, k, v, softmax_scale=None, out=None,
                      BLOCK_M=128, BLOCK_N=128, WARP_SPEC=True, LOOP_STAGES=3,
                      num_warps=8, num_stages=2):
    B, S, H, D = q.shape
    assert B == 1
    if softmax_scale is None:
        softmax_scale = D ** -0.5
    if out is None:
        out = torch.empty_like(q)

    # (S, H, D) views; TMA box is [BLOCK, 1, D] so one head's rows are gathered by TMA.
    q3, k3, v3, o3 = (t.view(S, H, D) for t in (q, k, v, out))
    dq = TensorDescriptor.from_tensor(q3, [BLOCK_M, 1, D])
    dk = TensorDescriptor.from_tensor(k3, [BLOCK_N, 1, D])
    dv = TensorDescriptor.from_tensor(v3, [BLOCK_N, 1, D])
    do = TensorDescriptor.from_tensor(o3, [BLOCK_M, 1, D])

    grid = (S // BLOCK_M, H)
    _attn_fwd_tma[grid](dq, dk, dv, do, softmax_scale * LOG2E, S,
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
                        WARP_SPEC=WARP_SPEC, LOOP_STAGES=LOOP_STAGES,
                        num_warps=num_warps, num_stages=num_stages)
    return out
