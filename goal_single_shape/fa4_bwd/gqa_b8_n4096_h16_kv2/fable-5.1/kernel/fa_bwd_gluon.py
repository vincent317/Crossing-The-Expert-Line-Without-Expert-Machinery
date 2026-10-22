"""FlashAttention backward for GQA causal attention, hand-scheduled in Triton Gluon
for Blackwell (sm_100): explicit TMEM accumulators, tcgen05 MMA, TMA loads and
TMA bulk reduce-add.

Decomposition (kv-outer, GQA-packed):
  One CTA owns one kv block (BN rows) of one kv head and sweeps every q tile at or
  below the diagonal, for all GROUP query heads that share this kv head. dK and dV
  therefore accumulate in TMEM across (head, q_tile) and are stored exactly once in
  bf16 -- no atomics, no fp32 staging buffer, no cross-head reduction kernel.
  dQ is the only gradient that reduces across CTAs; it goes out as an fp32 TMA
  bulk reduce-add (cp.reduce.async.bulk.tensor), which runs in L2 at ~store
  bandwidth instead of the ~3 TB/s of per-thread fp32 atomics.

Five MMAs per (q_tile, kv_tile, head), no recompute pass:
  S^T = K Q^T        (BN,BM)
  dP^T = V dO^T      (BN,BM)
  dV += P^T dO       (BN,D)
  dK += dS^T Q       (BN,D)
  dQ  = dS K         (BM,D)   -> TMA reduce-add into an fp32 dQ buffer

Work is issued from a persistent grid; each CTA processes (kv block, mirror block)
pairs so every CTA gets the same amount of causal work.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.nvidia.hopper import tma
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    get_tmem_reg_layout,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from fa_bwd_triton import LOG2E, _cast_kernel, _delta_kernel

_REDUCE_ADD = None


@builtin
def tma_reduce_add(desc, coord, src, _semantic=None):
    """cp.reduce.async.bulk.tensor .add : smem tile -> global, accumulating in L2."""
    global _REDUCE_ADD
    if _REDUCE_ADD is None:
        from triton._C.libtriton import ir
        _REDUCE_ADD = ir.DESCRIPTOR_REDUCE_KIND.ADD
    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    _semantic.builder.create_async_tma_reduce(_REDUCE_ADD, desc.handle, coord, src.handle)


@gluon.jit
def _fa_bwd_kernel(q_desc, k_desc, v_desc, do_desc, dq_desc, dk_desc, dv_desc,
                   DQ_ACC, LSE, DELTA,
                   S, NUM_PAIRS, qk_log2, scale, sl_b, sl_h, sq_s, sq_h,
                   HKV: gl.constexpr, GROUP: gl.constexpr,
                   BM: gl.constexpr, BN: gl.constexpr, D: gl.constexpr,
                   DQ_MODE: gl.constexpr, num_warps: gl.constexpr):
    bf: gl.constexpr = gl.bfloat16

    # ---------------------------------------------------------------- smem, 224 KB
    k_smem = gl.allocate_shared_memory(bf, k_desc.block_type.shape, k_desc.layout)            # 32K
    v_smem = gl.allocate_shared_memory(bf, v_desc.block_type.shape, v_desc.layout)            # 32K
    q_smem = gl.allocate_shared_memory(bf, [2] + q_desc.block_type.shape, q_desc.layout)         # 32K
    do_smem = gl.allocate_shared_memory(bf, [2] + do_desc.block_type.shape, do_desc.layout)       # 32K
    ps_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, BM], bf)
    pT_smem = gl.allocate_shared_memory(bf, [BN, BM], ps_layout)              # 16K
    dsT_smem = gl.allocate_shared_memory(bf, [BN, BM], ps_layout)             # 16K
    dq_smem = gl.allocate_shared_memory(gl.float32, dq_desc.block_type.shape, dq_desc.layout)  # 32K
    dkv_smem = gl.allocate_shared_memory(bf, dk_desc.block_type.shape, dk_desc.layout)         # 32K

    ld_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    kv_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    s_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    o_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(ld_bars.index(0), count=1)
    mbarrier.init(ld_bars.index(1), count=1)
    mbarrier.init(kv_bar, count=1)
    mbarrier.init(s_bar, count=1)
    mbarrier.init(o_bar, count=1)

    # ---------------------------------------------------------------- tmem, 448 cols
    tm_s: gl.constexpr = TensorMemoryLayout([BN, BM], col_stride=1)
    tm_kv: gl.constexpr = TensorMemoryLayout([BN, D], col_stride=1)
    tm_qT: gl.constexpr = TensorMemoryLayout([D, BM], col_stride=1)
    sT_t = allocate_tensor_memory(gl.float32, [BN, BM], tm_s)                 # 64 cols
    dpT_t = allocate_tensor_memory(gl.float32, [BN, BM], tm_s)                # 64
    dk_t = allocate_tensor_memory(gl.float32, [BN, D], tm_kv)                 # 128
    dv_t = allocate_tensor_memory(gl.float32, [BN, D], tm_kv)                 # 128
    dqT_t = allocate_tensor_memory(gl.float32, [D, BM], tm_qT)                # 64
    s_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, (BN, BM), tm_s, num_warps)
    kv_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, (BN, D), tm_kv, num_warps)
    qT_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, (D, BM), tm_qT, num_warps)

    QBYTES: gl.constexpr = 2 * BM * D * 2
    KVBYTES: gl.constexpr = 2 * BN * D * 2
    nblk_m = S // BM
    num_j = S // BN
    half = num_j // 2

    it = 0          # global iteration counter: drives ld/s/o barrier phases
    kvc = 0         # global kv-block counter: drives the kv_bar phase

    for p in range(gl.program_id(0), NUM_PAIRS, gl.num_programs(0)):
        bh = p // half
        c = p % half
        b = bh // HKV
        hkv = bh % HKV

        for rep in gl.static_range(2):
            j = c + rep * (num_j - 1 - 2 * c)
            row_kv = b * S + j * BN
            mbarrier.expect(kv_bar, KVBYTES)
            tma.async_copy_global_to_shared(k_desc, [row_kv, hkv * D], kv_bar, k_smem)
            tma.async_copy_global_to_shared(v_desc, [row_kv, hkv * D], kv_bar, v_smem)

            i_start = (j * BN) // BM
            n_iter = (nblk_m - i_start) * GROUP

            # prologue: fill both q/dO buffers (iterations 0 and 1)
            for pre in gl.static_range(2):
                qi = nblk_m - 1 - pre // GROUP
                qh = hkv * GROUP + pre % GROUP
                mbarrier.expect(ld_bars.index(pre), QBYTES)
                tma.async_copy_global_to_shared(q_desc, [b * S + qi * BM, qh * D],
                                                ld_bars.index(pre), q_smem.index(pre))
                tma.async_copy_global_to_shared(do_desc, [b * S + qi * BM, qh * D],
                                                ld_bars.index(pre), do_smem.index(pre))

            mbarrier.wait(kv_bar, kvc & 1)
            kvc += 1

            for t in range(n_iter):
                buf = t % 2
                i = nblk_m - 1 - t // GROUP
                h = hkv * GROUP + t % GROUP
                row_q = b * S + i * BM
                mm_ph = it & 1

                # S^T = K Q^T, dP^T = V dO^T
                mbarrier.wait(ld_bars.index(buf), (it // 2) & 1)
                tcgen05_mma(k_smem, q_smem.index(buf).permute((1, 0)), sT_t, use_acc=False)
                tcgen05_mma(v_smem, do_smem.index(buf).permute((1, 0)), dpT_t, use_acc=False)
                tcgen05_commit(s_bar)

                # drain the previous dQ reduce underneath the two MMAs above
                if DQ_MODE == 0:
                    tma.store_wait(pendings=0)

                mbarrier.wait(s_bar, mm_ph)
                sT = sT_t.load(s_reg)
                dpT = dpT_t.load(s_reg)

                offs_m = i * BM + gl.arange(0, BM, gl.SliceLayout(0, s_reg))
                offs_n = j * BN + gl.arange(0, BN, gl.SliceLayout(1, s_reg))
                lse_v = gl.load(LSE + b * sl_b + h * sl_h + offs_m)
                del_v = gl.load(DELTA + b * sl_b + h * sl_h + offs_m)

                pT = gl.exp2(sT * qk_log2 - lse_v[None, :])
                pT = gl.where(offs_m[None, :] >= offs_n[:, None], pT, 0.0)
                dsT = pT * (dpT - del_v[None, :])
                pT_smem.store(pT.to(bf))
                dsT_smem.store(dsT.to(bf))
                gl.barrier()
                fence_async_shared()

                # dV += P^T dO ; dK += dS^T Q ; dQ^T = K^T dS^T
                tcgen05_mma(pT_smem, do_smem.index(buf), dv_t, use_acc=(t > 0))
                tcgen05_mma(dsT_smem, q_smem.index(buf), dk_t, use_acc=(t > 0))
                tcgen05_mma(k_smem.permute((1, 0)), dsT_smem, dqT_t, use_acc=False)
                tcgen05_commit(o_bar)
                mbarrier.wait(o_bar, mm_ph)

                dqT = dqT_t.load(qT_reg) * scale
                if DQ_MODE == 0:
                    dq_smem.permute((1, 0)).store(dqT)
                    gl.barrier()
                    fence_async_shared()
                    tma_reduce_add(dq_desc, [row_q, h * D], dq_smem)
                elif DQ_MODE == 1:
                    om = gl.arange(0, BM, gl.SliceLayout(0, qT_reg))
                    od = gl.arange(0, D, gl.SliceLayout(1, qT_reg))
                    gl.atomic_add(DQ_ACC + (row_q + om)[None, :] * sq_s + h * sq_h + od[:, None],
                                  dqT, sem="relaxed")

                # prefetch iteration t+2 into the buffer just consumed
                if t + 2 < n_iter:
                    q2 = nblk_m - 1 - (t + 2) // GROUP
                    h2 = hkv * GROUP + (t + 2) % GROUP
                    mbarrier.expect(ld_bars.index(buf), QBYTES)
                    tma.async_copy_global_to_shared(q_desc, [b * S + q2 * BM, h2 * D],
                                                    ld_bars.index(buf), q_smem.index(buf))
                    tma.async_copy_global_to_shared(do_desc, [b * S + q2 * BM, h2 * D],
                                                    ld_bars.index(buf), do_smem.index(buf))
                it += 1

            # ------------------------------------------------- dK / dV epilogue
            if DQ_MODE == 0:
                tma.store_wait(pendings=0)
            dk = dk_t.load(kv_reg) * scale
            dkv_smem.store(dk.to(bf))
            gl.barrier()
            fence_async_shared()
            tma.async_copy_shared_to_global(dk_desc, [row_kv, hkv * D], dkv_smem)
            tma.store_wait(pendings=0)
            dv = dv_t.load(kv_reg)
            dkv_smem.store(dv.to(bf))
            gl.barrier()
            fence_async_shared()
            tma.async_copy_shared_to_global(dv_desc, [row_kv, hkv * D], dkv_smem)
            tma.store_wait(pendings=0)

    tma.store_wait(pendings=0)


CONFIGS_G = [
    # (BM, BN, num_warps, dq_mode)  dq_mode 0=TMA reduce-add, 1=atomics, 2=skip(ablation)
    (64, 128, 8, 0),
    (64, 128, 8, 2),
]


def _descs(t, blk, hcount, dtype):
    B, S, H, D = t.shape
    lay = gl.NVMMASharedLayout.get_default_for([blk, D], dtype)
    return TensorDescriptor.from_tensor(t.view(B * S, H * D), [blk, D], lay)


def fa_bwd_gluon(q, k, v, o, do, lse, scale, cfg=(64, 128, 8, 0), out=None):
    """q,do,o (B,S,Hq,D) bf16; k,v (B,S,Hkv,D) bf16; lse (B,Hq,S) fp32 natural log."""
    from fa_bwd_triton import make_buffers

    B, S, HQ, D = q.shape
    HKV = k.shape[2]
    GROUP = HQ // HKV
    BM, BN, warps, dq_mode = cfg
    dq, dk, dv, dq_acc, delta, lse_log2 = make_buffers(q, k, v) if out is None else out

    torch.mul(lse, LOG2E, out=lse_log2)
    dq_acc.zero_()
    _delta_kernel[(B * HQ * (S // 128),)](
        o, do, delta, o.stride(0), o.stride(1), o.stride(2),
        delta.stride(0), delta.stride(1), S, HQ=HQ, D=D, BS=128, num_warps=4)

    q_desc = _descs(q, BM, HQ, gl.bfloat16)
    do_desc = _descs(do, BM, HQ, gl.bfloat16)
    k_desc = _descs(k, BN, HKV, gl.bfloat16)
    v_desc = _descs(v, BN, HKV, gl.bfloat16)
    dk_desc = _descs(dk, BN, HKV, gl.bfloat16)
    dv_desc = _descs(dv, BN, HKV, gl.bfloat16)
    dq_desc = TensorDescriptor.from_tensor(
        dq_acc.view(B * S, HQ * D), [BM, D],
        gl.NVMMASharedLayout.get_default_for([BM, D], gl.float32))

    num_pairs = (S // BN) // 2 * (B * HKV)
    grid = (min(torch.cuda.get_device_properties(0).multi_processor_count, num_pairs),)
    _fa_bwd_kernel[grid](
        q_desc, k_desc, v_desc, do_desc, dq_desc, dk_desc, dv_desc,
        dq_acc, lse_log2, delta,
        S, num_pairs, scale * LOG2E, scale,
        delta.stride(0), delta.stride(1), dq_acc.stride(1), dq_acc.stride(2),
        HKV=HKV, GROUP=GROUP, BM=BM, BN=BN, D=D, DQ_MODE=dq_mode, num_warps=warps)

    n = dq.numel()
    _cast_kernel[(triton.cdiv(n, 4096),)](dq_acc, dq, n, BS=4096, num_warps=4)
    return dq, dk, dv
