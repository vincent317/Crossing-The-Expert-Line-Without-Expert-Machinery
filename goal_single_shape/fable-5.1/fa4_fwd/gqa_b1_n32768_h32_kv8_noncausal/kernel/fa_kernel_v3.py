"""FlashAttention forward v3: TMA + warp specialization + Blackwell-specific tuning.

Additions over v2, all derived from the published FA2/FA3 algorithm plus Triton's own
Blackwell guidance (register budget, accumulator subtiling, packed special functions):

* maxnreg control, needed on sm100 once the loop is warp-specialized.
* Accumulator rescale is subtiled along D so the alpha multiply interleaves better
  instead of hitting the whole [BLOCK_M, HEAD_DIM] tile at once.
* m_i / l_i updates deferred to the end of the body to cut live registers.
* Optional packed `ex2.approx.f16x2`: two exp2 per SFU instruction, which halves the
  special-function cost that otherwise rivals the MMA time at this shape.
* Optional 2-CTA cluster (num_ctas=2).
* Two layouts: 3D TMA over (S, H, D) directly, or a head-major copy first so the TMA
  box is fully contiguous (copy cost is paid inside the timed region).
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

LOG2E = 1.4426950408889634


def _alloc_fn(size, align, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


@triton.jit
def _exp2_packed_f16(x):
    """exp2 on fp16 pairs: one PTX instruction per two elements."""
    return tl.inline_asm_elementwise(
        "ex2.approx.f16x2 $0, $1;", "=r,r", [x],
        dtype=tl.float16, is_pure=True, pack=2)


@triton.jit
def _load_tile(desc, h, s, BLOCK: tl.constexpr, D: tl.constexpr,
               HEAD_FIRST: tl.constexpr):
    if HEAD_FIRST:
        return desc.load([h, s, 0]).reshape([BLOCK, D])
    else:
        return desc.load([s, h, 0]).reshape([BLOCK, D])


@triton.jit
def _attn_fwd_v3(desc_q, desc_k, desc_v, desc_o, qk_scale, S,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
                 HEAD_FIRST: tl.constexpr, WARP_SPEC: tl.constexpr,
                 LOOP_STAGES: tl.constexpr, PACKED_EXP2: tl.constexpr,
                 SUBTILE_ACC: tl.constexpr):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    qo_off = start_m * BLOCK_M

    q = _load_tile(desc_q, off_h, qo_off, BLOCK_M, HEAD_DIM, HEAD_FIRST)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in tl.range(0, S, BLOCK_N, warp_specialize=WARP_SPEC,
                            num_stages=LOOP_STAGES):
        k = _load_tile(desc_k, off_h, start_n, BLOCK_N, HEAD_DIM, HEAD_FIRST)
        qk = tl.dot(q, k.T)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        arg = qk * qk_scale - m_ij[:, None]
        if PACKED_EXP2:
            p16 = _exp2_packed_f16(arg.to(tl.float16))
            pf = p16.to(tl.float32)
        else:
            pf = tl.math.exp2(arg)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(pf, 1)

        if SUBTILE_ACC:
            acc0, acc1 = acc.reshape([BLOCK_M, 2, HEAD_DIM // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BLOCK_M, HEAD_DIM])
        else:
            acc = acc * alpha[:, None]

        v = _load_tile(desc_v, off_h, start_n, BLOCK_N, HEAD_DIM, HEAD_FIRST)
        if PACKED_EXP2:
            acc = tl.dot(p16.to(v.dtype), v, acc)
        else:
            acc = tl.dot(pf.to(v.dtype), v, acc)

        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc = acc / l_i[:, None]
    o = acc.to(desc_o.dtype)
    if HEAD_FIRST:
        desc_o.store([off_h, qo_off, 0], o.reshape([1, BLOCK_M, HEAD_DIM]))
    else:
        desc_o.store([qo_off, off_h, 0], o.reshape([BLOCK_M, 1, HEAD_DIM]))


def flash_attn_fwd_v3(q, k, v, softmax_scale=None, BLOCK_M=128, BLOCK_N=128,
                      WARP_SPEC=True, LOOP_STAGES=3, PACKED_EXP2=False,
                      SUBTILE_ACC=True, HEAD_FIRST=False, num_warps=8, num_stages=2,
                      num_ctas=1, maxnreg=None):
    """q/k/v: (1, S, H, D) bf16 contiguous, non-causal. Returns (1, S, H, D) bf16."""
    B, S, H, D = q.shape
    assert B == 1
    if softmax_scale is None:
        softmax_scale = D ** -0.5

    if HEAD_FIRST:
        # (S, H, D) -> (H, S, D) so the TMA box covers contiguous rows of one head.
        q3 = q[0].permute(1, 0, 2).contiguous()
        k3 = k[0].permute(1, 0, 2).contiguous()
        v3 = v[0].permute(1, 0, 2).contiguous()
        o3 = torch.empty_like(q3)
        dq = TensorDescriptor.from_tensor(q3, [1, BLOCK_M, D])
        dk = TensorDescriptor.from_tensor(k3, [1, BLOCK_N, D])
        dv = TensorDescriptor.from_tensor(v3, [1, BLOCK_N, D])
        do = TensorDescriptor.from_tensor(o3, [1, BLOCK_M, D])
    else:
        q3, k3, v3 = (t.view(S, H, D) for t in (q, k, v))
        o3 = torch.empty(S, H, D, device=q.device, dtype=q.dtype)
        dq = TensorDescriptor.from_tensor(q3, [BLOCK_M, 1, D])
        dk = TensorDescriptor.from_tensor(k3, [BLOCK_N, 1, D])
        dv = TensorDescriptor.from_tensor(v3, [BLOCK_N, 1, D])
        do = TensorDescriptor.from_tensor(o3, [BLOCK_M, 1, D])

    grid = (S // BLOCK_M, H)
    kwargs = dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, HEAD_FIRST=HEAD_FIRST,
                  WARP_SPEC=WARP_SPEC, LOOP_STAGES=LOOP_STAGES, PACKED_EXP2=PACKED_EXP2,
                  SUBTILE_ACC=SUBTILE_ACC, num_warps=num_warps, num_stages=num_stages,
                  num_ctas=num_ctas)
    if maxnreg is not None:
        kwargs["maxnreg"] = maxnreg
    _attn_fwd_v3[grid](dq, dk, dv, do, softmax_scale * LOG2E, S, **kwargs)

    if HEAD_FIRST:
        return o3.permute(1, 0, 2).contiguous().view(1, S, H, D)
    return o3.view(1, S, H, D)
