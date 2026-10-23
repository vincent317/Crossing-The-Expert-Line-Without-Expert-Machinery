"""FlashAttention forward, written from scratch in Triton for B200 (sm100).

Target shape: q (B,S,HQ,D) bf16, k/v (B,S,HKV,D) bf16, non-causal, GQA HQ/HKV = 12.

Design notes:
  * One CTA owns a BLOCK_M block of query rows for one (batch, q-head).
  * Grid is laid out so that consecutive CTAs (which are co-resident) share the same
    (batch, kv-head) and the same position in the K/V stream -> they hit the same K/V
    tiles in L2 at the same time. With GQA 12:1 this is the main memory-side win.
  * TMA (tl.make_tensor_descriptor) for all global accesses; descriptors are built
    in-kernel as 2D views (S, D) with row stride H*D, so tiles land as (BLOCK, D)
    without any reshape/transpose in registers.
  * Online softmax in fp32 with exp2; P is cast to bf16 for the PV MMA so both dots
    run on the tcgen05 bf16 path.
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _attn_fwd_tma(q_ptr, k_ptr, v_ptr, o_ptr,
                  qk_scale,
                  S, num_m,
                  HQ: tl.constexpr, HKV: tl.constexpr, GQA: tl.constexpr,
                  D: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  WS: tl.constexpr):
    pid_x = tl.program_id(0)          # (m_block, gqa_slot) — consecutive => same K/V tile
    pid_y = tl.program_id(1)          # (batch, kv_head)

    m_blk = pid_x // GQA
    j = pid_x % GQA
    b = pid_y // HKV
    hkv = pid_y % HKV
    hq = hkv * GQA + j

    # 2D views for this (b, head): shape (S, D), row stride H*D
    q_desc = tl.make_tensor_descriptor(
        q_ptr + (b * S * HQ + hq) * D,
        shape=[S, D], strides=[HQ * D, 1], block_shape=[BLOCK_M, D])
    o_desc = tl.make_tensor_descriptor(
        o_ptr + (b * S * HQ + hq) * D,
        shape=[S, D], strides=[HQ * D, 1], block_shape=[BLOCK_M, D])
    k_desc = tl.make_tensor_descriptor(
        k_ptr + (b * S * HKV + hkv) * D,
        shape=[S, D], strides=[HKV * D, 1], block_shape=[BLOCK_N, D])
    v_desc = tl.make_tensor_descriptor(
        v_ptr + (b * S * HKV + hkv) * D,
        shape=[S, D], strides=[HKV * D, 1], block_shape=[BLOCK_N, D])

    m0 = m_blk * BLOCK_M
    q = q_desc.load([m0, 0])

    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for n in tl.range(0, S, BLOCK_N, warp_specialize=WS):
        k = k_desc.load([n, 0])
        qk = tl.dot(q, k.T)                                  # (M, N) fp32
        m_new = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = v_desc.load([n, 0])
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_desc.store([m0, 0], acc.to(tl.bfloat16))


@triton.jit
def _attn_fwd_ptr(q_ptr, k_ptr, v_ptr, o_ptr,
                  qk_scale,
                  S, num_m,
                  HQ: tl.constexpr, HKV: tl.constexpr, GQA: tl.constexpr,
                  D: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  WS: tl.constexpr):
    """Same math without TMA, as a fallback / comparison path."""
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    m_blk = pid_x // GQA
    j = pid_x % GQA
    b = pid_y // HKV
    hkv = pid_y % HKV
    hq = hkv * GQA + j

    qo_base = q_ptr + (b * S * HQ + hq) * D
    o_base = o_ptr + (b * S * HQ + hq) * D
    k_base = k_ptr + (b * S * HKV + hkv) * D
    v_base = v_ptr + (b * S * HKV + hkv) * D

    offs_m = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(qo_base + offs_m[:, None] * (HQ * D) + offs_d[None, :])

    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for n in tl.range(0, S, BLOCK_N, warp_specialize=WS):
        kn = (n + offs_n)
        k = tl.load(k_base + kn[:, None] * (HKV * D) + offs_d[None, :])
        qk = tl.dot(q, k.T)
        m_new = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + kn[:, None] * (HKV * D) + offs_d[None, :])
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(o_base + offs_m[:, None] * (HQ * D) + offs_d[None, :], acc.to(tl.bfloat16))


_SCRATCH = {}


def _alloc(size: int, align: int, stream):
    key = (size, align)
    buf = _SCRATCH.get(key)
    if buf is None or buf.numel() < size:
        buf = torch.empty(size, dtype=torch.int8, device="cuda")
        _SCRATCH[key] = buf
    return buf


triton.set_allocator(_alloc)


def flash_attn_fwd(q, k, v, softmax_scale, out=None,
                   BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=2,
                   warp_specialize=True, use_tma=True):
    B, S, HQ, D = q.shape
    HKV = k.shape[2]
    GQA = HQ // HKV
    assert S % BLOCK_M == 0 and S % BLOCK_N == 0
    if out is None:
        out = torch.empty_like(q)

    num_m = S // BLOCK_M
    grid = (num_m * GQA, B * HKV)
    kernel = _attn_fwd_tma if use_tma else _attn_fwd_ptr
    kernel[grid](
        q, k, v, out,
        softmax_scale * LOG2E,
        S, num_m,
        HQ=HQ, HKV=HKV, GQA=GQA, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, WS=warp_specialize,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
