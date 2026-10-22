"""FlashAttention forward v4: hand-written Gluon (Blackwell tcgen05) kernel.

Warp-specialised FA3-style pipeline, one CTA per (256-row Q block, head):

  * load partition   (1 warp)   : TMA q/k/v tiles into a shared-memory ring.
  * mma partition    (1 warp)   : issues tcgen05 MMAs. S_h = Q_h K^T lands in TMEM,
                                  O_h += P_h V reads P_h straight from the S_h tile.
  * softmax x2       (4 warps each): each owns one 128-row half, keeps the online
                                  softmax state in registers, writes P (bf16) back into
                                  its S tile and hands alpha to the correction warps.
  * correction (default, 4 warps): rescales O_h in TMEM by alpha and, at the end,
                                  normalises and stores the output through TMA.

The two halves are issued ping-pong (QK0(i+1), PV1(i), QK1(i+1), PV0(i+1)) so the
tensor core always has one half's MMA in flight while the other half's softmax runs.
tcgen05 MMAs from one CTA execute in order, so QK(h, i+1) may overwrite the S/P tile
right after PV(h, i) was issued without waiting for its completion.

Layout is the (S, H, D) view of the (1, S, H, D) tensors: the TMA box is [rows, 1, D]
and the shared tile is reshaped to [rows, D] (inner dimension unchanged, so the
128B-swizzled NVMMA layout carries over).
"""
import torch
from triton.language.core import _aggregate as aggregate
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tma,
    mbarrier,
    tcgen05_mma,
)

LOG2E = 1.4426950408889634
SPLIT_M = 128     # rows per softmax half (one tcgen05 M=128 instruction)
SUB_COLS = 64     # column sub-tile for register-resident softmax / rescale work


@aggregate
class Ring:
    """(index, phase) cursor over a ring of NUM mbarrier-guarded buffers."""
    index: gl.tensor
    phase: gl.tensor
    num: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, index, phase, num):
        self.index = index
        self.phase = phase
        self.num = gl.constexpr(num)

    @gluon.must_use_result
    @gluon.jit
    def next(self):
        nxt = self.index + 1
        wrap = nxt == self.num
        return Ring(gl.where(wrap, 0, nxt), gl.where(wrap, self.phase ^ 1, self.phase), self.num)


@gluon.jit
def _ring0(num: gl.constexpr):
    return Ring(gl.to_tensor(0), gl.to_tensor(0), num)


@gluon.jit
def _tma_load(desc, smem, bar, s, h, FAKE: gl.constexpr = False):
    if FAKE:
        mbarrier.arrive(bar, count=1)
    else:
        mbarrier.expect(bar, desc.block_type.nbytes)
        tma.async_copy_global_to_shared(desc, [s, h, 0], bar, smem)


@gluon.jit
def _mma(a, b, acc, use_acc, bars, FAKE: gl.constexpr = False):
    """tcgen05 MMA, or (ablation) just signal its completion barriers."""
    if FAKE:
        for i in gl.static_range(len(bars)):
            mbarrier.arrive(bars[i], count=1)
    else:
        tcgen05_mma(a, b, acc, use_acc=use_acc, mbarriers=bars)


# ----------------------------------------------------------------------------
# packed-math helpers: 2-wide fp32 FMA/MUL (Blackwell f32x2 pipe) and fp16x2 exp2
# ----------------------------------------------------------------------------
@gluon.jit
def _fma2(a, b, c):
    """fp32 a*b+c on element pairs: one fma.rn.f32x2 per two elements."""
    return gl.inline_asm_elementwise(
        "{ .reg .b64 ra, rb, rc; mov.b64 ra, {$2, $3}; mov.b64 rb, {$4, $5}; mov.b64 rc, {$6, $7};"
        " fma.rn.f32x2 ra, ra, rb, rc; mov.b64 {$0, $1}, ra; }",
        "=r,=r,r,r,r,r,r,r", [a, b, c], dtype=gl.float32, is_pure=True, pack=2)


@gluon.jit
def _mul2(a, b):
    return gl.inline_asm_elementwise(
        "{ .reg .b64 ra, rb; mov.b64 ra, {$2, $3}; mov.b64 rb, {$4, $5};"
        " mul.rn.f32x2 ra, ra, rb; mov.b64 {$0, $1}, ra; }",
        "=r,=r,r,r,r,r", [a, b], dtype=gl.float32, is_pure=True, pack=2)


@gluon.jit
def _exp2_f16x2(x):
    """exp2 on fp16 pairs (one MUFU issue per two elements), result stays fp16."""
    return gl.inline_asm_elementwise(
        "ex2.approx.f16x2 $0, $1;", "=r,r", [x], dtype=gl.float16, is_pure=True, pack=2)


# ----------------------------------------------------------------------------
# load partition
# ----------------------------------------------------------------------------
@gluon.jit
def _load_part(desc_q, desc_k, desc_v, q_smem, kv_smem, q_ready, kv_ready, kv_empty,
               off_h, qo_off, S, BLOCK_N: gl.constexpr, NUM_KV: gl.constexpr,
               SPLIT_M: gl.constexpr, MODE: gl.constexpr):
    FAKE: gl.constexpr = MODE == 4
    _tma_load(desc_q, q_smem.index(0), q_ready.index(0), qo_off, off_h, FAKE)
    _tma_load(desc_q, q_smem.index(1), q_ready.index(1), qo_off + SPLIT_M, off_h, FAKE)

    ring = _ring0(NUM_KV)
    for start_n in range(0, S, BLOCK_N):
        mbarrier.wait(kv_empty.index(ring.index), ring.phase)
        _tma_load(desc_k, kv_smem.index(ring.index), kv_ready.index(ring.index), start_n, off_h, FAKE)
        ring = ring.next()
        mbarrier.wait(kv_empty.index(ring.index), ring.phase)
        _tma_load(desc_v, kv_smem.index(ring.index), kv_ready.index(ring.index), start_n, off_h, FAKE)
        ring = ring.next()


# ----------------------------------------------------------------------------
# mma partition
# ----------------------------------------------------------------------------
@gluon.jit
def _mma_part(q_smem, kv_smem, q_ready, kv_ready, kv_empty,
              s_tmem, o_tmem, s_ready, p_ready, o_free, o_done,
              S, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, NUM_KV: gl.constexpr,
              dtype: gl.constexpr, SPLIT_M: gl.constexpr, MODE: gl.constexpr):
    FAKE: gl.constexpr = MODE == 3
    p_layout: gl.constexpr = TensorMemoryLayout((SPLIT_M, BLOCK_N), col_stride=1)
    p0 = s_tmem.index(0).slice(0, BLOCK_N // 2)._reinterpret(dtype, [SPLIT_M, BLOCK_N], p_layout)
    p1 = s_tmem.index(1).slice(0, BLOCK_N // 2)._reinterpret(dtype, [SPLIT_M, BLOCK_N], p_layout)

    mbarrier.wait(q_ready.index(0), 0)
    mbarrier.wait(q_ready.index(1), 0)
    q0 = q_smem.index(0).reshape([SPLIT_M, HEAD_DIM])
    q1 = q_smem.index(1).reshape([SPLIT_M, HEAD_DIM])

    ring = _ring0(NUM_KV)
    n_iter = S // BLOCK_N

    # ---- prologue: S0(0), S1(0), PV0(0)
    mbarrier.wait(kv_ready.index(ring.index), ring.phase)
    k_smem = kv_smem.index(ring.index).reshape([BLOCK_N, HEAD_DIM])
    k_bar = kv_empty.index(ring.index)
    ring = ring.next()
    _mma(q0, k_smem.permute((1, 0)), s_tmem.index(0), False, [s_ready.index(0)], FAKE)
    _mma(q1, k_smem.permute((1, 0)), s_tmem.index(1), False, [s_ready.index(1), k_bar], FAKE)

    mbarrier.wait(kv_ready.index(ring.index), ring.phase)
    v_smem = kv_smem.index(ring.index).reshape([BLOCK_N, HEAD_DIM])
    v_bar = kv_empty.index(ring.index)
    ring = ring.next()
    mbarrier.wait(p_ready.index(0), 0)
    mbarrier.wait(o_free.index(0), 0)
    _mma(p0, v_smem, o_tmem.index(0), False, [o_done.index(0)], FAKE)

    phase = 0
    for it in range(n_iter - 1):
        # S0(it+1): in-order after PV0(it), so the S0/P0 tile can be reused now.
        mbarrier.wait(kv_ready.index(ring.index), ring.phase)
        k_smem = kv_smem.index(ring.index).reshape([BLOCK_N, HEAD_DIM])
        k_bar = kv_empty.index(ring.index)
        ring = ring.next()
        _mma(q0, k_smem.permute((1, 0)), s_tmem.index(0), False, [s_ready.index(0)], FAKE)

        # PV1(it)
        mbarrier.wait(p_ready.index(1), phase)
        mbarrier.wait(o_free.index(1), phase)
        _mma(p1, v_smem, o_tmem.index(1), it > 0, [o_done.index(1), v_bar], FAKE)

        # S1(it+1)
        _mma(q1, k_smem.permute((1, 0)), s_tmem.index(1), False, [s_ready.index(1), k_bar], FAKE)

        # PV0(it+1)
        mbarrier.wait(kv_ready.index(ring.index), ring.phase)
        v_smem = kv_smem.index(ring.index).reshape([BLOCK_N, HEAD_DIM])
        v_bar = kv_empty.index(ring.index)
        ring = ring.next()
        phase ^= 1
        mbarrier.wait(p_ready.index(0), phase)
        mbarrier.wait(o_free.index(0), phase)
        _mma(p0, v_smem, o_tmem.index(0), True, [o_done.index(0)], FAKE)

    # ---- epilogue: PV1(n-1)
    mbarrier.wait(p_ready.index(1), phase)
    mbarrier.wait(o_free.index(1), phase)
    _mma(p1, v_smem, o_tmem.index(1), n_iter > 1, [o_done.index(1), v_bar], FAKE)


# ----------------------------------------------------------------------------
# softmax partitions (one per 128-row half)
# ----------------------------------------------------------------------------
@gluon.jit
def _softmax_part(h: gl.constexpr, s_tmem, s_ready, p_ready, alpha_smem, corr_ready, corr_free,
                  lse_smem, qk_scale, S, BLOCK_N: gl.constexpr, num_warps: gl.constexpr,
                  dtype: gl.constexpr, SPLIT_M: gl.constexpr, SUB_COLS: gl.constexpr,
                  MODE: gl.constexpr=0, MATH: gl.constexpr=0):
    sub_tmem_layout: gl.constexpr = TensorMemoryLayout((SPLIT_M, SUB_COLS), col_stride=1)
    sub_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, [SPLIT_M, SUB_COLS],
                                                   sub_tmem_layout, num_warps)
    row_layout: gl.constexpr = gl.SliceLayout(1, sub_layout)
    p_layout: gl.constexpr = TensorMemoryLayout((SPLIT_M, BLOCK_N), col_stride=1)

    s_buf = s_tmem.index(h)
    p_buf = s_buf.slice(0, BLOCK_N // 2)._reinterpret(dtype, [SPLIT_M, BLOCK_N], p_layout)

    m_i = gl.full([SPLIT_M], -float("inf"), gl.float32, row_layout)
    l_i = gl.zeros([SPLIT_M], gl.float32, row_layout)

    phase = 0
    n_iter = S // BLOCK_N
    for it in range(n_iter):
        mbarrier.wait(s_ready.index(h), phase)
        if False:
            pass
        else:
            qk0 = s_buf.slice(0 * SUB_COLS, SUB_COLS).load(sub_layout)
            qk1 = s_buf.slice(1 * SUB_COLS, SUB_COLS).load(sub_layout)
            row_max = gl.maximum(gl.max(qk0, 1), gl.max(qk1, 1))
            m_ij = gl.maximum(m_i, row_max * qk_scale)
            alpha = gl.exp2(m_i - m_ij)

            # alpha -> correction warps (they rescale O before the next PV).
            mbarrier.wait(corr_free.index(h), phase ^ 1)
            alpha_smem.slice(h * SPLIT_M, SPLIT_M).store(alpha)
            gl.barrier()   # all warps' smem writes precede thread 0's arrive
            mbarrier.arrive(corr_ready.index(h), count=1)

            if False:
                pass
            else:
                p0 = gl.exp2(qk0 * qk_scale - m_ij[:, None])
                p1 = gl.exp2(qk1 * qk_scale - m_ij[:, None])
                p_buf.slice(0 * SUB_COLS, SUB_COLS).store(p0.to(dtype))
                p_buf.slice(1 * SUB_COLS, SUB_COLS).store(p1.to(dtype))
                # tmem store lowering ends with tcgen05.wait::st; every thread then arrives.
                mbarrier.arrive(p_ready.index(h), count=1)
                l_i = l_i * alpha + (gl.sum(p0, 1) + gl.sum(p1, 1))
            m_i = m_ij
        phase ^= 1

    mbarrier.wait(corr_free.index(h), phase ^ 1)
    lse_smem.slice(h * SPLIT_M, SPLIT_M).store(l_i)
    gl.barrier()
    mbarrier.arrive(corr_ready.index(h), count=1)


# ----------------------------------------------------------------------------
# correction / epilogue partition (default partition)
# ----------------------------------------------------------------------------
@gluon.jit
def _corr_part(desc_o, o_tmem, o_free, o_done, alpha_smem, corr_ready, corr_free, lse_smem,
               o_smem, off_h, qo_off, S, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
               num_warps: gl.constexpr, dtype: gl.constexpr, SPLIT_M: gl.constexpr,
               SUB_COLS: gl.constexpr, MODE: gl.constexpr=0, MATH: gl.constexpr=0):
    NSUB: gl.constexpr = HEAD_DIM // SUB_COLS
    sub_tmem_layout: gl.constexpr = TensorMemoryLayout((SPLIT_M, SUB_COLS), col_stride=1)
    sub_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, [SPLIT_M, SUB_COLS],
                                                   sub_tmem_layout, num_warps)
    row_layout: gl.constexpr = gl.SliceLayout(1, sub_layout)
    vec_layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])

    n_iter = S // BLOCK_N
    phase = 0
    for it in range(n_iter):
        for h in gl.static_range(2):
            mbarrier.wait(corr_ready.index(h), phase)
            alpha = alpha_smem.slice(h * SPLIT_M, SPLIT_M).load(vec_layout)
            gl.barrier()   # all warps have read alpha before thread 0 frees the slot
            mbarrier.arrive(corr_free.index(h), count=1)
            alpha_r = gl.convert_layout(alpha, row_layout)
            if it > 0:
                # PV(h, it-1) must have landed before O_h is rescaled.
                mbarrier.wait(o_done.index(h), phase ^ 1)
                for i in gl.static_range(NSUB):
                    o_ref = o_tmem.index(h).slice(i * SUB_COLS, SUB_COLS)
                    o = o_ref.load(sub_layout)
                    o_ref.store(o * alpha_r[:, None])
            mbarrier.arrive(o_free.index(h), count=1)
        phase ^= 1

    for h in gl.static_range(2):
        mbarrier.wait(corr_ready.index(h), phase)
        l = lse_smem.slice(h * SPLIT_M, SPLIT_M).load(vec_layout)
        gl.barrier()
        mbarrier.arrive(corr_free.index(h), count=1)
        inv_l = 1.0 / gl.convert_layout(l, row_layout)
        mbarrier.wait(o_done.index(h), phase ^ 1)      # last PV(h) done
        o_2d = o_smem.index(h).reshape([SPLIT_M, HEAD_DIM])
        for i in gl.static_range(NSUB):
            o = o_tmem.index(h).slice(i * SUB_COLS, SUB_COLS).load(sub_layout)
            o_2d.slice(i * SUB_COLS, SUB_COLS, dim=1).store((o * inv_l[:, None]).to(dtype))
        fence_async_shared()
        gl.barrier()   # every warp's smem tile rows are written before the TMA store is issued
        tma.async_copy_shared_to_global(desc_o, [qo_off + h * SPLIT_M, off_h, 0], o_smem.index(h))
    tma.store_wait(0)


# ----------------------------------------------------------------------------
# kernel
# ----------------------------------------------------------------------------
@gluon.jit
def _attn_fwd_gluon(desc_q, desc_k, desc_v, desc_o, qk_scale, S,
                    BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, NUM_KV: gl.constexpr,
                    dtype: gl.constexpr, num_warps: gl.constexpr,
                    SPLIT_M: gl.constexpr, SUB_COLS: gl.constexpr, MODE: gl.constexpr,
                    MATH: gl.constexpr):
    start_m = gl.program_id(0)
    off_h = gl.program_id(1)
    qo_off = start_m * (2 * SPLIT_M)

    q_smem = gl.allocate_shared_memory(dtype, [2, SPLIT_M, 1, HEAD_DIM], desc_q.layout)
    kv_smem = gl.allocate_shared_memory(dtype, [NUM_KV, BLOCK_N, 1, HEAD_DIM], desc_k.layout)
    o_smem = gl.allocate_shared_memory(dtype, [2, SPLIT_M, 1, HEAD_DIM], desc_o.layout)
    vec_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    alpha_smem = gl.allocate_shared_memory(gl.float32, [2 * SPLIT_M], vec_smem_layout)
    lse_smem = gl.allocate_shared_memory(gl.float32, [2 * SPLIT_M], vec_smem_layout)

    q_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    kv_ready = gl.allocate_shared_memory(gl.int64, [NUM_KV, 1], mbarrier.MBarrierLayout())
    kv_empty = gl.allocate_shared_memory(gl.int64, [NUM_KV, 1], mbarrier.MBarrierLayout())
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())   # mma -> softmax
    p_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())   # softmax -> mma
    o_free = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())    # corr -> mma
    o_done = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())    # mma -> corr
    corr_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    corr_free = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(q_ready.index(i), count=1)
        mbarrier.init(s_ready.index(i), count=1)
        mbarrier.init(p_ready.index(i), count=1)
        mbarrier.init(o_free.index(i), count=1)
        mbarrier.init(o_done.index(i), count=1)
        mbarrier.init(corr_ready.index(i), count=1)
        mbarrier.init(corr_free.index(i), count=1)
    for i in gl.static_range(NUM_KV):
        mbarrier.init(kv_ready.index(i), count=1)
        mbarrier.init(kv_empty.index(i), count=1)
        mbarrier.arrive(kv_empty.index(i), count=1)
    fence_async_shared()

    s_tmem = allocate_tensor_memory(gl.float32, [2, SPLIT_M, BLOCK_N],
                                    TensorMemoryLayout((SPLIT_M, BLOCK_N), col_stride=1))
    o_tmem = allocate_tensor_memory(gl.float32, [2, SPLIT_M, HEAD_DIM],
                                    TensorMemoryLayout((SPLIT_M, HEAD_DIM), col_stride=1))

    gl.warp_specialize([
        (_corr_part, (desc_o, o_tmem, o_free, o_done, alpha_smem, corr_ready, corr_free, lse_smem,
                      o_smem, off_h, qo_off, S, BLOCK_N, HEAD_DIM, num_warps, dtype, SPLIT_M, SUB_COLS, MODE, MATH)),
        (_softmax_part, (0, s_tmem, s_ready, p_ready, alpha_smem, corr_ready, corr_free, lse_smem,
                         qk_scale, S, BLOCK_N, num_warps, dtype, SPLIT_M, SUB_COLS, MODE, MATH)),
        (_softmax_part, (1, s_tmem, s_ready, p_ready, alpha_smem, corr_ready, corr_free, lse_smem,
                         qk_scale, S, BLOCK_N, num_warps, dtype, SPLIT_M, SUB_COLS, MODE, MATH)),
        (_mma_part, (q_smem, kv_smem, q_ready, kv_ready, kv_empty, s_tmem, o_tmem, s_ready, p_ready,
                     o_free, o_done, S, BLOCK_N, HEAD_DIM, NUM_KV, dtype, SPLIT_M, MODE)),
        (_load_part, (desc_q, desc_k, desc_v, q_smem, kv_smem, q_ready, kv_ready, kv_empty,
                      off_h, qo_off, S, BLOCK_N, NUM_KV, SPLIT_M, MODE)),
    ], [4, 4, 1, 1], [192, 192, 24, 24])

    for i in gl.static_range(2):
        mbarrier.invalidate(q_ready.index(i))
        mbarrier.invalidate(s_ready.index(i))
        mbarrier.invalidate(p_ready.index(i))
        mbarrier.invalidate(o_free.index(i))
        mbarrier.invalidate(o_done.index(i))
        mbarrier.invalidate(corr_ready.index(i))
        mbarrier.invalidate(corr_free.index(i))
    for i in gl.static_range(NUM_KV):
        mbarrier.invalidate(kv_ready.index(i))
        mbarrier.invalidate(kv_empty.index(i))


def flash_attn_fwd_v4(q, k, v, softmax_scale=None, BLOCK_N=128, NUM_KV=3, num_warps=4,
                      maxnreg=128, MODE=0, MATH=0):
    B, S, H, D = q.shape
    BLOCK_M = 2 * SPLIT_M
    assert B == 1 and D == 128 and S % BLOCK_M == 0 and S % BLOCK_N == 0
    if softmax_scale is None:
        softmax_scale = D ** -0.5
    out = torch.empty_like(q)
    q3, k3, v3, o3 = (t.view(S, H, D) for t in (q, k, v, out))
    gdt = gl.bfloat16 if q.dtype == torch.bfloat16 else gl.float16

    lq = gl.NVMMASharedLayout.get_default_for([SPLIT_M, 1, D], gdt)
    lkv = gl.NVMMASharedLayout.get_default_for([BLOCK_N, 1, D], gdt)
    dq = TensorDescriptor(q3, shape=[S, H, D], strides=[H * D, D, 1], block_shape=[SPLIT_M, 1, D], layout=lq)
    dk = TensorDescriptor(k3, shape=[S, H, D], strides=[H * D, D, 1], block_shape=[BLOCK_N, 1, D], layout=lkv)
    dv = TensorDescriptor(v3, shape=[S, H, D], strides=[H * D, D, 1], block_shape=[BLOCK_N, 1, D], layout=lkv)
    do = TensorDescriptor(o3, shape=[S, H, D], strides=[H * D, D, 1], block_shape=[SPLIT_M, 1, D], layout=lq)

    grid = (S // BLOCK_M, H)
    _attn_fwd_gluon[grid](dq, dk, dv, do, softmax_scale * LOG2E, S,
                          BLOCK_N=BLOCK_N, HEAD_DIM=D, NUM_KV=NUM_KV, dtype=gdt,
                          SPLIT_M=SPLIT_M, SUB_COLS=SUB_COLS, MODE=MODE, MATH=MATH,
                          num_warps=num_warps, maxnreg=maxnreg)
    return out
