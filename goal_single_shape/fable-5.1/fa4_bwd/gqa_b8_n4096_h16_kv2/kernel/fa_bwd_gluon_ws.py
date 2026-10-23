"""Warp-specialized FlashAttention backward (Gluon / tcgen05 / sm_100).

Same decomposition as fa_bwd_gluon.py (kv-outer, GQA-packed, 5 MMAs, dQ through a
TMA bulk reduce-add) but split into two warp partitions so the MMA stream runs
ahead of the softmax stream instead of alternating with it:

  partition MMA (1 warp)   : issues all TMA loads and all tcgen05 MMAs
  partition SOFTMAX (8 wp) : TMEM -> registers, exp2/mask/dS, registers -> SMEM,
                             dQ^T -> SMEM -> TMA reduce-add, dK/dV epilogue

S^T/dP^T/dQ^T live in NB-deep TMEM buffers and P^T/dS^T in NB-deep SMEM buffers,
so at NB=2 the MMA partition can be a full tile ahead of the softmax partition.

TMEM cost: NB*(BN*BM + BN*BM + D*BM)/128 cols + 2*D cols for dK/dV.
  BM=32, NB=2 -> 2*(32+32+32) + 256 = 448  (run-ahead)
  BM=64, NB=1 -> 1*(64+64+64) + 256 = 448  (no run-ahead)
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
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

from fa_bwd_gluon import tma_reduce_add
from fa_bwd_triton import LOG2E, _cast_kernel, _delta_kernel


@gluon.jit
def _mma_partition(q_desc, k_desc, v_desc, do_desc,
                   k_smem, v_smem, q_smem, do_smem, pT_smem, dsT_smem,
                   sT_t, dpT_t, dqT_t, dk_t, dv_t,
                   ld_bars, kv_ready, s_free, s_ready, pd_ready, pd_free,
                   dq_ready, dq_free, kvdone, kvfree,
                   S, NUM_PAIRS,
                   HKV: gl.constexpr, GROUP: gl.constexpr, BM: gl.constexpr,
                   BN: gl.constexpr, D: gl.constexpr, NB: gl.constexpr,
                   NB_PD: gl.constexpr, NB_DQ: gl.constexpr):
    """Issues every TMA load and every tcgen05 MMA; never touches TMEM data."""
    QBYTES: gl.constexpr = 2 * BM * D * 2
    KVBYTES: gl.constexpr = 2 * BN * D * 2
    nblk_m = S // BM
    num_j = S // BN
    half = num_j // 2
    it = 0
    kvc = 0

    for p in range(gl.program_id(0), NUM_PAIRS, gl.num_programs(0)):
        bh = p // half
        c = p % half
        b = bh // HKV
        hkv = bh % HKV

        for rep in gl.static_range(2):
            j = c + rep * (num_j - 1 - 2 * c)
            row_kv = b * S + j * BN
            i_start = (j * BN) // BM
            n_iter = (nblk_m - i_start) * GROUP

            # K/V smem and the dK/dV accumulators are reusable only once the
            # softmax partition has read out the previous block's dK/dV
            mbarrier.wait(kvfree, (kvc & 1) ^ 1)
            mbarrier.expect(kv_ready, KVBYTES)
            tma.async_copy_global_to_shared(k_desc, [row_kv, hkv * D], kv_ready, k_smem)
            tma.async_copy_global_to_shared(v_desc, [row_kv, hkv * D], kv_ready, v_smem)

            for pre in gl.static_range(2):
                qi = nblk_m - 1 - pre // GROUP
                qh = hkv * GROUP + pre % GROUP
                mbarrier.expect(ld_bars.index(pre), QBYTES)
                tma.async_copy_global_to_shared(q_desc, [b * S + qi * BM, qh * D],
                                                ld_bars.index(pre), q_smem.index(pre))
                tma.async_copy_global_to_shared(do_desc, [b * S + qi * BM, qh * D],
                                                ld_bars.index(pre), do_smem.index(pre))
            mbarrier.wait(kv_ready, kvc & 1)

            for t in range(n_iter):
                buf = it % NB
                pbuf = it % NB_PD
                dbuf = it % NB_DQ
                qb = t % 2
                ph = (it // NB) & 1
                pph = (it // NB_PD) & 1
                dph = (it // NB_DQ) & 1

                # ---- S^T = K Q^T, dP^T = V dO^T
                mbarrier.wait(ld_bars.index(qb), (it // 2) & 1)
                mbarrier.wait(s_free.index(buf), ph ^ 1)
                tcgen05_mma(k_smem, q_smem.index(qb).permute((1, 0)), sT_t.index(buf), use_acc=False)
                mbarrier.wait(pd_free.index(pbuf), pph ^ 1)
                tcgen05_mma(v_smem, do_smem.index(qb).permute((1, 0)), dpT_t.index(pbuf), use_acc=False)
                tcgen05_commit(s_ready.index(buf))

                # ---- dV += P^T dO, dK += dS^T Q, dQ^T = K^T dS^T
                mbarrier.wait(pd_ready.index(pbuf), pph)
                mbarrier.wait(dq_free.index(dbuf), dph ^ 1)
                tcgen05_mma(pT_smem.index(pbuf), do_smem.index(qb), dv_t, use_acc=(t > 0))
                tcgen05_mma(dsT_smem.index(pbuf), q_smem.index(qb), dk_t, use_acc=(t > 0))
                tcgen05_mma(k_smem.permute((1, 0)), dsT_smem.index(pbuf), dqT_t.index(dbuf), use_acc=False)
                tcgen05_commit(dq_ready.index(dbuf))
                tcgen05_commit(pd_free.index(pbuf))

                # refill the q/dO buffer just consumed
                if t + 2 < n_iter:
                    q2 = nblk_m - 1 - (t + 2) // GROUP
                    h2 = hkv * GROUP + (t + 2) % GROUP
                    mbarrier.expect(ld_bars.index(qb), QBYTES)
                    tma.async_copy_global_to_shared(q_desc, [b * S + q2 * BM, h2 * D],
                                                    ld_bars.index(qb), q_smem.index(qb))
                    tma.async_copy_global_to_shared(do_desc, [b * S + q2 * BM, h2 * D],
                                                    ld_bars.index(qb), do_smem.index(qb))
                it += 1

            tcgen05_commit(kvdone)
            kvc += 1


@gluon.jit
def _softmax_partition(dq_desc, DQ_ACC, DK, DV, LSE, DELTA,
                       pT_smem, dsT_smem, dq_smem,
                       sT_t, dpT_t, dqT_t, dk_t, dv_t,
                       s_free, s_ready, pd_ready, pd_free, dq_ready, dq_free,
                       kvdone, kvfree,
                       S, NUM_PAIRS, qk_log2, scale, sl_b, sl_h, sk_s, sk_h,
                       HKV: gl.constexpr, GROUP: gl.constexpr, BM: gl.constexpr,
                       BN: gl.constexpr, D: gl.constexpr, NB: gl.constexpr,
                       NB_PD: gl.constexpr, NB_DQ: gl.constexpr,
                       DQ_MODE: gl.constexpr, num_warps: gl.constexpr):
    bf: gl.constexpr = gl.bfloat16
    tm_s: gl.constexpr = TensorMemoryLayout([BN, BM], col_stride=1)
    tm_kv: gl.constexpr = TensorMemoryLayout([BN, D], col_stride=1)
    tm_qT: gl.constexpr = TensorMemoryLayout([D, BM], col_stride=1)
    s_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, (BN, BM), tm_s, num_warps)
    kv_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, (BN, D), tm_kv, num_warps)
    qT_reg: gl.constexpr = get_tmem_reg_layout(gl.float32, (D, BM), tm_qT, num_warps)

    nblk_m = S // BM
    num_j = S // BN
    half = num_j // 2
    it = 0
    kvc = 0

    for p in range(gl.program_id(0), NUM_PAIRS, gl.num_programs(0)):
        bh = p // half
        c = p % half
        b = bh // HKV
        hkv = bh % HKV

        for rep in gl.static_range(2):
            j = c + rep * (num_j - 1 - 2 * c)
            i_start = (j * BN) // BM
            n_iter = (nblk_m - i_start) * GROUP
            offs_n = j * BN + gl.arange(0, BN, gl.SliceLayout(1, s_reg))

            for t in range(n_iter):
                buf = it % NB
                pbuf = it % NB_PD
                dbuf = it % NB_DQ
                ph = (it // NB) & 1
                pph = (it // NB_PD) & 1
                dph = (it // NB_DQ) & 1
                i = nblk_m - 1 - t // GROUP
                h = hkv * GROUP + t % GROUP

                offs_m = i * BM + gl.arange(0, BM, gl.SliceLayout(0, s_reg))
                lse_v = gl.load(LSE + b * sl_b + h * sl_h + offs_m)
                del_v = gl.load(DELTA + b * sl_b + h * sl_h + offs_m)

                mbarrier.wait(s_ready.index(buf), ph)
                sT = sT_t.index(buf).load(s_reg)
                pT = gl.exp2(sT * qk_log2 - lse_v[None, :])
                pT = gl.where(offs_m[None, :] >= offs_n[:, None], pT, 0.0)
                pT_bf = pT.to(bf)
                mbarrier.arrive(s_free.index(buf), count=1)

                dpT = dpT_t.index(pbuf).load(s_reg)
                dsT = pT * (dpT - del_v[None, :])
                pT_smem.index(pbuf).store(pT_bf)
                dsT_smem.index(pbuf).store(dsT.to(bf))
                fence_async_shared()
                mbarrier.arrive(pd_ready.index(pbuf), count=1)

                # ---- dQ epilogue
                mbarrier.wait(dq_ready.index(dbuf), dph)
                dqT = dqT_t.index(dbuf).load(qT_reg) * scale
                if DQ_MODE == 0:
                    tma.store_wait(pendings=0)
                    dq_smem.index(dbuf).permute((1, 0)).store(dqT)
                    fence_async_shared()
                    mbarrier.arrive(dq_free.index(dbuf), count=1)
                    tma_reduce_add(dq_desc, [b * S + i * BM, h * D], dq_smem.index(dbuf))
                else:
                    mbarrier.arrive(dq_free.index(dbuf), count=1)
                it += 1

            # ---- dK / dV epilogue: plain stores straight out of the TMEM layout
            mbarrier.wait(kvdone, kvc & 1)
            dk = dk_t.load(kv_reg) * scale
            dv = dv_t.load(kv_reg)
            rows = j * BN + gl.arange(0, BN, gl.SliceLayout(1, kv_reg))
            cols = gl.arange(0, D, gl.SliceLayout(0, kv_reg))
            off = b * (S * HKV * D) + rows[:, None] * sk_s + hkv * sk_h + cols[None, :]
            gl.store(DK + off, dk.to(bf))
            gl.store(DV + off, dv.to(bf))
            mbarrier.arrive(kvfree, count=1)
            kvc += 1

    if DQ_MODE == 0:
        tma.store_wait(pendings=0)


@gluon.jit
def _fa_bwd_ws_kernel(q_desc, k_desc, v_desc, do_desc, dq_desc,
                      DQ_ACC, DK, DV, LSE, DELTA,
                      S, NUM_PAIRS, qk_log2, scale, sl_b, sl_h, sk_s, sk_h,
                      HKV: gl.constexpr, GROUP: gl.constexpr, BM: gl.constexpr,
                      BN: gl.constexpr, D: gl.constexpr, NB: gl.constexpr,
                      NB_PD: gl.constexpr, NB_DQ: gl.constexpr,
                      DQ_MODE: gl.constexpr, num_warps: gl.constexpr):
    bf: gl.constexpr = gl.bfloat16
    k_smem = gl.allocate_shared_memory(bf, k_desc.block_type.shape, k_desc.layout)
    v_smem = gl.allocate_shared_memory(bf, v_desc.block_type.shape, v_desc.layout)
    q_smem = gl.allocate_shared_memory(bf, [2] + q_desc.block_type.shape, q_desc.layout)
    do_smem = gl.allocate_shared_memory(bf, [2] + do_desc.block_type.shape, do_desc.layout)
    ps_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, BM], bf)
    pT_smem = gl.allocate_shared_memory(bf, [NB_PD, BN, BM], ps_layout)
    dsT_smem = gl.allocate_shared_memory(bf, [NB_PD, BN, BM], ps_layout)
    dq_smem = gl.allocate_shared_memory(gl.float32, [NB_DQ] + dq_desc.block_type.shape, dq_desc.layout)

    ld_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    kv_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    s_free = gl.allocate_shared_memory(gl.int64, [NB, 1], mbarrier.MBarrierLayout())
    s_ready = gl.allocate_shared_memory(gl.int64, [NB, 1], mbarrier.MBarrierLayout())
    pd_ready = gl.allocate_shared_memory(gl.int64, [NB_PD, 1], mbarrier.MBarrierLayout())
    pd_free = gl.allocate_shared_memory(gl.int64, [NB_PD, 1], mbarrier.MBarrierLayout())
    dq_ready = gl.allocate_shared_memory(gl.int64, [NB_DQ, 1], mbarrier.MBarrierLayout())
    dq_free = gl.allocate_shared_memory(gl.int64, [NB_DQ, 1], mbarrier.MBarrierLayout())
    kvdone = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    kvfree = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(ld_bars.index(0), count=1)
    mbarrier.init(ld_bars.index(1), count=1)
    mbarrier.init(kv_ready, count=1)
    mbarrier.init(kvdone, count=1)
    mbarrier.init(kvfree, count=1)
    for u in gl.static_range(NB):
        mbarrier.init(s_free.index(u), count=1)
        mbarrier.init(s_ready.index(u), count=1)
    for u in gl.static_range(NB_PD):
        mbarrier.init(pd_ready.index(u), count=1)
        mbarrier.init(pd_free.index(u), count=1)
    for u in gl.static_range(NB_DQ):
        mbarrier.init(dq_ready.index(u), count=1)
        mbarrier.init(dq_free.index(u), count=1)

    tm_s: gl.constexpr = TensorMemoryLayout([BN, BM], col_stride=1)
    tm_kv: gl.constexpr = TensorMemoryLayout([BN, D], col_stride=1)
    tm_qT: gl.constexpr = TensorMemoryLayout([D, BM], col_stride=1)
    sT_t = allocate_tensor_memory(gl.float32, [NB, BN, BM], tm_s)
    dpT_t = allocate_tensor_memory(gl.float32, [NB_PD, BN, BM], tm_s)
    dqT_t = allocate_tensor_memory(gl.float32, [NB_DQ, D, BM], tm_qT)
    dk_t = allocate_tensor_memory(gl.float32, [BN, D], tm_kv)
    dv_t = allocate_tensor_memory(gl.float32, [BN, D], tm_kv)

    gl.warp_specialize([
        (_softmax_partition, (dq_desc, DQ_ACC, DK, DV, LSE, DELTA,
                              pT_smem, dsT_smem, dq_smem,
                              sT_t, dpT_t, dqT_t, dk_t, dv_t,
                              s_free, s_ready, pd_ready, pd_free, dq_ready, dq_free,
                              kvdone, kvfree,
                              S, NUM_PAIRS, qk_log2, scale, sl_b, sl_h, sk_s, sk_h,
                              HKV, GROUP, BM, BN, D, NB, NB_PD, NB_DQ, DQ_MODE, num_warps)),
        (_mma_partition, (q_desc, k_desc, v_desc, do_desc,
                          k_smem, v_smem, q_smem, do_smem, pT_smem, dsT_smem,
                          sT_t, dpT_t, dqT_t, dk_t, dv_t,
                          ld_bars, kv_ready, s_free, s_ready, pd_ready, pd_free,
                          dq_ready, dq_free, kvdone, kvfree,
                          S, NUM_PAIRS, HKV, GROUP, BM, BN, D, NB, NB_PD, NB_DQ)),
    ], [1], [40])


CONFIGS_WS = [
    # (BM, BN, warps, NB_S, NB_PD, NB_DQ, dq_mode); TMEM: 64*(NB_S+NB_PD+NB_DQ)+256 <= 512
    (64, 128, 8, 2, 1, 1, 0),
    (64, 128, 8, 2, 1, 1, 2),
    (64, 128, 12, 2, 1, 1, 0),
    (64, 128, 16, 2, 1, 1, 0),
    (64, 128, 8, 1, 2, 1, 0),
    (64, 128, 8, 1, 1, 2, 0),
]


def fa_bwd_ws(q, k, v, o, do, lse, scale, cfg=(64, 128, 8, 2, 1, 1, 0), out=None):
    from fa_bwd_gluon import _descs
    from fa_bwd_triton import make_buffers

    B, S, HQ, D = q.shape
    HKV = k.shape[2]
    GROUP = HQ // HKV
    BM, BN, warps, NB, NB_PD, NB_DQ, dq_mode = cfg
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
    dq_desc = TensorDescriptor.from_tensor(
        dq_acc.view(B * S, HQ * D), [BM, D],
        gl.NVMMASharedLayout.get_default_for([BM, D], gl.float32))

    num_pairs = (S // BN) // 2 * (B * HKV)
    grid = (min(torch.cuda.get_device_properties(0).multi_processor_count, num_pairs),)
    _fa_bwd_ws_kernel[grid](
        q_desc, k_desc, v_desc, do_desc, dq_desc, dq_acc, dk, dv, lse_log2, delta,
        S, num_pairs, scale * LOG2E, scale,
        delta.stride(0), delta.stride(1), dk.stride(1), dk.stride(2),
        HKV=HKV, GROUP=GROUP, BM=BM, BN=BN, D=D, NB=NB, NB_PD=NB_PD, NB_DQ=NB_DQ,
        DQ_MODE=dq_mode, num_warps=warps)

    n = dq.numel()
    _cast_kernel[(triton.cdiv(n, 4096),)](dq_acc, dq, n, BS=4096, num_warps=4)
    return dq, dk, dv
