"""DSA sparse MLA decode attention for NVIDIA Blackwell (SM100/SM103) in CuTe-DSL.

Computes, per query token t:
    out[t] = softmax( (q_nope[t] @ Kc^T + q_pe[t] @ Kp^T) * sm_scale ) @ Kc
    lse[t] = log2( sum_j exp(s_tj * sm_scale) )   (base-2, per head, fp32)
over the valid (non -1) rows of sparse_indices[t] gathered from the paged
compressed-KV cache (ckv) and RoPE cache (kpe). Token-level indices address
flattened [num_pages*64, 512] / [num_pages*64, 64] caches directly.

The kernel's softmax already runs in the exp2 domain (scores scaled by
sm_scale*log2(e)), so every path that produces the final normalized output
also holds the definition's base-2 LSE as m + log2(l) of its final running
max/denominator pair; each such epilogue stores it with one extra fp32 write
per (token, head). An all -1 token stores -inf, matching the reference.

Design:
  * One kernel launch per call. Grid = (T * NSPLIT) CTAs; each CTA handles a
    slice of the token's 2048 top-k slots ("flash-decoding" split-K).
  * Register MMA (mma.sync m16n8k16 x8 warps). QK^T is computed transposed
    (S^T = K @ Q^T) on a (4,1,2) atom layout: A = gathered K/V rows (each
    smem byte ldsm'd once) and B = Q (duplicated x4 instead of x8), with a
    pairwise smem reduce of the two k-half accumulators. PV stays on the
    (1,8,1) layout (warps split output cols).
  * Depth-2 TMA bulk (cp.async.bulk per row+cache) gather with mbarrier
    completion: stage barriers are armed with constant full-tile bytes at
    kernel entry and every row is gathered unconditionally (invalid slots
    gather cache row 0 - L2-hot, finite, exactly masked), so the gathers
    launch straight off per-warp uniform index loads with no scan/arm chain.
    Iteration it waits tile it, consumes it (QK/softmax/PV), then re-arms
    and issues tile it+2 into the freed stage. Q moves on its own mbarrier
    (barQ), issued by warps 6/7 right after the entry barrier.
  * Numerically stable softmax: per-head running max m over the valid rows
    (masked tile max via butterfly + cross-warp smem exchange), p =
    exp2(s*scale - m); l and the PV accumulator are rescaled by
    exp2(m_old - m_new) whenever a later tile raises the max. Split merges
    ship (l, m) per head and rescale partials by exp2(m_s - M) before
    combining, so arbitrary score magnitudes stay exact.
  * T=1/two-KV microkernel: 16 clusterless CTAs, one head per CTA;
    warp 0 computes the two BF16 dot products, then all eight warps emit the
    512-value row after one shared-memory handoff. It preserves fp32 score and
    P.V accumulation plus the hi/lo BF16 P representation, while bypassing the
    general gather/MMA/merge choreography for this launch-overhead floor case.
  * The T=2 micro eligibility tail uses explicit 256-bit G2R loads with
    L1 no-allocation and 256-byte L2 promotion. It still scans every live
    index slot, but avoids scalarized streaming loads on the critical path.
  * Per-tile validity scan parallel to the gathers: empty splits skip the
    main loop and epilogue-store work entirely, empty tiles skip their MMA,
    and prefix-packed tiles let fully-dead warps skip QK.
  * Strided tile->split mapping (split s owns tiles {s + j*NSPLIT}): with
    prefix-packed indices, a token with <= NSPLIT*64 valid rows lands at most
    ONE nonempty tile per split, so medium tokens (129..1024 valid at ns16)
    consume in parallel instead of 2 serial tiles on the low splits. Exact
    for any -1 pattern (pure slot permutation; the solo probe covers the
    complement ranges). NTILES==1 configs (ns32) are mapping-invariant.
  * Split 0 publishes the DIRECT decision to its own smem right at the
    pre-loop probe (sAf/sAny final there); mainloop barriers order it for
    the epilogue, which drops the re-derive + extra CTA barrier.
  * Solo fast path: split 0 prefetches the other splits' index slots after
    issuing its gathers (exact, any -1 pattern); when no other split has work
    it writes the normalized output directly, skipping partials and the merge.
  * Otherwise per-split partials merged in-kernel, column-split across the
    last NM splits (NM = 8 within one wave, else 4; launch-id tagged flags).
  * Partials (gmem mPartO, DSMEM sSlot, pair-hop rows) are carried in fp32
    end to end; the accumulated output is rounded to bf16 exactly once, at
    the final normalized store. (bf16 partials cost 1-2 output ULPs on
    cancelling elements and fail the official every-element 1e-2 gate.)
  * Softmax P is split hi/lo into two bf16 planes (p = p_hi + p_lo, exact to
    ~2^-17) and PV runs two MMAs per k-step, so the P quantization noise
    (~2^-9 rel, the other half of the official-gate 2-ULP collisions vs the
    baseline's own noise) drops below the baseline's rounding floor.
  * Packed-prefix QK warp-skip: warps whose 16-row m-tile lies beyond the
    tile's valid count skip QK (their acc stays zero; dead rows are
    validity-masked and land zero in P via the masked store).

  * Padded smem strides (16B pad) to kill ldsm bank conflicts.
"""

import math
from typing import Any, cast

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.math  # noqa: F401  (registers the cute.math namespace)
from cutlass import BFloat16, Float32, Int16, Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute as _cute_ir, llvm
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.nvgpu import cpasync, warp as warp_ops
from cutlass.cute.runtime import from_dlpack

HEADS = 16
DK_CK = 512  # compressed-KV dim
DK_PE = 64  # rope dim
DK = DK_CK + DK_PE
DV = 512
TOPK = 2048
TILE_N = 64  # gathered KV rows per tile
NTHREADS = 256
LOG2E = 1.4426950408889634
# "minus infinity" stand-in for masked scores / running-max init: finite so
# (-BIG) - (-BIG) stays 0 (never NaN), large enough that exp2(x - real) == 0.
NEG_BIG = -1.0e30
V8 = 8  # elements per 16B vector piece
# per-(token,split) flag word (int32): bits0..27 launch id, bit29 DIRECT
# (split 0 wrote the final output itself), bit30 FULL (split produced a
# partial, so the merger must include it). A split's flag is complete the
# moment its launch id matches (empties carry no bits).
LID_MASK = (1 << 28) - 1
DONE_BIT = 1 << 28  # reserved
DIRECT_BIT = 1 << 29
FULL_BIT = 1 << 30

# Generalized small-token microkernel: any token whose valid indices form a
# prefix of <= MICRO_NCAP rows is handled by 16 independent head-per-CTA
# consumers over ONE staged 64-row window (no splits, no merge, no cluster
# choreography). Exactness for arbitrary -1 patterns is preserved by a full
# tail scan that falls through to the general path when the prefix property
# does not hold. Isolated scalar QK/PV crosses over near 32 rows, but a full
# 64-row eligibility window wins the official objective by avoiding the
# general split/merge choreography on the 33/52-row T=2 capture.
MICRO_ON = True
IKET_TRACE = False
MICRO_VOTE = True
# Eligibility threshold for the one staged 64-row window. Kept selectable in
# dev so the committed A/B harness can reproduce the 32-vs-64 crossover;
# submissions use the suite-level winner (64).
# Row-paired QK candidate verified 23/23 at 36.8852x and 36.8572x locally.
MICRO_NCAP = 64

# smem row strides (elements) with one 16B unit of padding
SKC_STRIDE = DK_CK + V8  # 520 elems (65 16B pieces)
SKP_STRIDE = DK_PE + V8  # 72 elems (9 pieces)
SQ_STRIDE = DK + V8  # 584 elems (73 pieces)
SP_STRIDE = TILE_N + V8  # 72 elems (9 pieces)


@dsl_user_op
def _issue_tma_rect_mcast(
    atom: cute.CopyAtom,
    dst: cute.Pointer,
    bar: cute.Pointer,
    col: Int32,
    row: Int32,
    mask: Int16,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Issue one OOB-filled 2-D query tile to a CTA-cluster mask."""
    exec_atom = atom._trait.unpack(
        tma_bar_ptr=bar, mcast_mask=mask, loc=loc, ip=ip
    )
    desc_ptr_ty = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(),
        cute.AddressSpace.generic,
        64,
    )
    desc_ptr = _cute_nvgpu_ir.get_tma_desc_addr(
        desc_ptr_ty, exec_atom, loc=loc, ip=ip
    )
    desc_i64 = _cute_nvgpu_ir.cast_tma_desc_to_integer(
        Int64.mlir_type, cast(Any, desc_ptr).value, loc=loc, ip=ip
    )
    llvm.inline_asm(
        None,
        [
            dst.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            desc_i64,
            col.ir_value(loc=loc, ip=ip),
            row.ir_value(loc=loc, ip=ip),
            bar.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            mask.ir_value(loc=loc, ip=ip),
        ],
        (
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
            "mbarrier::complete_tx::bytes.multicast::cluster "
            "[$0], [$1, {$2, $3}], [$4], $5;"
        ),
        "r,l,r,r,r,h",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _thread_exit(
    *, loc: ir.Location | None = None, ip: ir.InsertionPoint | None = None
) -> None:
    """Terminate a thread after a CTA-uniform microkernel decision."""
    llvm.inline_asm(
        None,
        [],
        "exit;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _DSA(object):
    """One static configuration: NSPLIT splits along the 2048 top-k slots."""

    def __init__(
        self,
        nsplit: int,
        nm: int = 0,
        cluster: int = 0,
        micro: bool = False,
        direct_afy: bool = False,
        q_mcast: int = 0,
        micro_idx_g2r: bool = False,
        micro_idx_g2r256: bool = False,
    ):
        assert TOPK % nsplit == 0
        assert cluster == 0 or (nsplit == 16 and cluster == 16) or (nsplit == 32 and cluster == 16)
        self.nsplit = nsplit
        self.cluster = cluster  # 0=off, 16 one cluster/token, 8 two (pair merge)
        self.nm = nm if nm else min(nsplit, 4)
        self.rows_per_split = TOPK // nsplit
        assert self.rows_per_split % TILE_N == 0
        self.ntiles = self.rows_per_split // TILE_N
        # sparse officials path (exact-byte per-warp arming) vs dense path
        # (constant arms + dummy rows): officials' real captures are highly
        # sparse, so unconditional full-tile gathers inflate the waited
        # volume 3-6x; dense synthetic tiles are full anyway.
        self.sparse = nsplit >= 16 or cluster != 0
        # ns32 pair16: split 0's existing exact sibling scan can prove that
        # the upper 1024 sparse slots are empty. Carry that proof in the sign
        # of its normal DSMEM l word so lower consumers need not wait for the
        # upper cluster's launch-tagged flag.
        self.pair_scan_marker = nsplit == 32 and cluster == 16
        # T=8 has a distinct compiled body.  Split zero reads its exact
        # sibling-index quads straight into the scan registers instead of
        # round-tripping them through a bulk-TMA barrier and shared memory.
        # Submission validations: 37.7463x / 37.5940x, 23/23 both runs.
        self.direct_afy = bool(direct_afy and self.sparse and not micro)
        self.q_mcast = int(
            q_mcast
            if nsplit in (16, 32) and cluster == 16
            else 0
        )
        self.q_rect16 = self.q_mcast in (3, 4)
        # Generalized small-row microkernel (head-per-CTA over one staged
        # 64-row window). Same-lease A/B supports compiling it only for the
        # official T=2 and T=6 launch shapes; every other T gets the exact
        # champion instruction stream with no speculative probe footprint.
        self.micro = MICRO_ON and micro and nsplit >= 16
        self.micro_idx_g2r = bool(micro_idx_g2r and self.micro)
        self.micro_idx_g2r256 = bool(
            micro_idx_g2r256 and self.micro_idx_g2r
        )
        self.smem_bytes = (
            (2 if self.ntiles > 1 else 1) * TILE_N * SKC_STRIDE * 2  # sKc stages
            + (2 if self.ntiles > 1 else 1) * TILE_N * SKP_STRIDE * 2  # sKp stages
            + HEADS * (SKC_STRIDE + SKP_STRIDE) * 2  # sQn + sQp
            + TILE_N * (HEADS + V8) * 2 * 2  # sP hi + lo (kv-major, head-pad)
            + self.rows_per_split * 4  # sIdx
            + self.ntiles * 8  # sTileCnt + sPack
            + nsplit * 4  # sFull
            + 16 * 3  # sAny + sDirect + sAf
            + 48  # sBar (5 TMA/merge mbarriers + 1 micro-window mbarrier)
            + max(TOPK - self.rows_per_split, 4) * 4  # sAfIdx
            + HEADS * 4  # sRed
            + 8 * HEADS * 4  # sL (warp-level l partials for direct path)
            + 4 * HEADS * 4  # sMx4 (per-k0-warp per-head tile maxes)
            + HEADS * 4  # sMxF (running per-head max)
            + HEADS * 4  # sMxS (per-head accumulator rescale factors)
            + (nsplit * HEADS * 4 if cluster == 0 else 0)  # sFact
            + (max(nsplit * 2, 33) * 4 if cluster else 0)  # sLslot ((l, m) pairs + marker)
            + ((min(nsplit, 16) if cluster == 16 else nsplit) * DV * 4 if cluster else 0)
            # sSlot: per-sender fp32 partial head rows
            + 4 * 512 * 4  # sSredT
            + 512  # SmemAllocator alignment slack (overrun = launch failure)
        )

    @cute.jit
    def __call__(
        self,
        mQnope: cute.Tensor,  # [T, 16, 512] bf16
        mQpe: cute.Tensor,  # [T, 16, 64] bf16
        mCKV: cute.Tensor,  # [P, 64, 512] bf16
        mKPE: cute.Tensor,  # [P, 64, 64] bf16
        mIdx: cute.Tensor,  # [T, 2048] int32
        mOut: cute.Tensor,  # [T, 16, 512] bf16
        mLse: cute.Tensor,  # [T, 16] f32 (base-2 log-sum-exp per head)
        mPartO: cute.Tensor,  # [T*NSPLIT, 16, 512] f32
        mPartL: cute.Tensor,  # [T*NSPLIT, 32] f32: l rows [0,16), m rows [16,32)
        mFlags: cute.Tensor,  # [>= T*NSPLIT] int32
        num_tokens: Int32,
        total_rows: Int32,  # num_pages * 64
        scale_log2: Float32,
        launch_id: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.q_rect16):
            # Global descriptors expose the true query widths; the wider
            # shared tile uses TMA OOB fill for each row's 16-byte bank pad.
            gQN8 = cute.make_tensor(
                cute.make_ptr(
                    Int64,
                    mQnope.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(
                    (DK_CK // 4, mQnope.shape[0] * mQnope.shape[1]),
                    stride=(1, DK_CK // 4),
                ),
            )
            gQP8 = cute.make_tensor(
                cute.make_ptr(
                    Int64,
                    mQpe.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(
                    (DK_PE // 4, mQpe.shape[0] * mQpe.shape[1]),
                    stride=(1, DK_PE // 4),
                ),
            )
            rect_k_atom, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SMulticastOp(),
                gQN8,
                cute.make_layout(
                    (SKC_STRIDE // 4, HEADS),
                    stride=(1, SKC_STRIDE // 4),
                ),
                (SKC_STRIDE // 4, HEADS),
            )
            rect_p_atom, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SMulticastOp(),
                gQP8,
                cute.make_layout(
                    (SKP_STRIDE // 4, HEADS),
                    stride=(1, SKP_STRIDE // 4),
                ),
                (SKP_STRIDE // 4, HEADS),
            )
        else:
            rect_k_atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(), Int64, num_bits_per_copy=128
            )
            rect_p_atom = rect_k_atom
        tiled_mma = cute.make_tiled_mma(
            warp_ops.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            atom_layout_mnk=(1, 8, 1),
        )
        tiled_mma_t = cute.make_tiled_mma(
            warp_ops.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            atom_layout_mnk=(4, 1, 2),
        )
        ldsm_A = cute.make_copy_atom(
            warp_ops.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16
        )
        ldsm_Bt = cute.make_copy_atom(
            warp_ops.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16
        )
        # one (token, split) pair per CTA
        self.kernel(
            mQnope,
            mQpe,
            mCKV,
            mKPE,
            mIdx,
            mOut,
            mLse,
            mPartO,
            mPartL,
            mFlags,
            rect_k_atom,
            rect_p_atom,
            tiled_mma,
            tiled_mma_t,
            ldsm_A,
            ldsm_Bt,
            num_tokens,
            total_rows,
            scale_log2,
            launch_id,
        ).launch(
            grid=(num_tokens * self.nsplit, 1, 1),
            block=(NTHREADS, 1, 1),
            smem=self.smem_bytes,
            cluster=(self.cluster, 1, 1) if self.cluster else None,
        )

    @cute.kernel
    def kernel(
        self,
        mQnope: cute.Tensor,
        mQpe: cute.Tensor,
        mCKV: cute.Tensor,
        mKPE: cute.Tensor,
        mIdx: cute.Tensor,
        mOut: cute.Tensor,
        mLse: cute.Tensor,
        mPartO: cute.Tensor,
        mPartL: cute.Tensor,
        mFlags: cute.Tensor,
        rect_k_atom: cute.CopyAtom,
        rect_p_atom: cute.CopyAtom,
        tiled_mma: cute.TiledMma,
        tiled_mma_t: cute.TiledMma,
        ldsm_A: cute.CopyAtom,
        ldsm_Bt: cute.CopyAtom,
        num_tokens: Int32,
        total_rows: Int32,
        scale_log2: Float32,
        launch_id: Int32,
    ):
        NSPLIT = self.nsplit
        NTILES = self.ntiles
        ROWS = self.rows_per_split
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        token = bx // NSPLIT
        split = bx % NSPLIT
        warp_id = tidx // 32
        lane_id = tidx % 32
        if cutlass.const_expr(IKET_TRACE):
            cute.experimental.iket.mark("kernel_entry")
        # ---- micro-path probe + window index preload. Issued here (kernel
        # entry) so the values fly with the t0r/t1r prologue loads; the
        # branch decision itself is deferred to the post-arming point below,
        # where the general path has already covered the probe latency with
        # its entry choreography. u0r are per-warp uniform loads of the
        # micro window slots [0, 64) (warp w owns slots w*8..w*8+7), ready
        # to feed speculative TMA row gathers the moment the candidate test
        # passes.
        mprobe0 = Int32(-1)
        mprobeC = Int32(0)
        u0r = [None] * 8
        if cutlass.const_expr(self.micro):
            if lane_id == 0:
                mprobe0 = mIdx[token, Int32(0)]
                mprobeC = mIdx[token, Int32(MICRO_NCAP)]
            for rr in cutlass.range_constexpr(8):
                u0r[rr] = mIdx[token, warp_id * 8 + rr]
        nwarp8 = NTHREADS // 32  # 8

        # ---- shared memory (16B-padded strides; K/V double-buffered) ----
        smem = cutlass.utils.SmemAllocator()
        NSTAGE = 2 if NTILES > 1 else 1  # single-tile configs skip stage 1
        sKc = [
            smem.allocate_tensor(
                BFloat16, cute.make_layout((TILE_N, DK_CK), stride=(SKC_STRIDE, 1)), 128
            )
            for _ in range(NSTAGE)
        ]
        sKp = [
            smem.allocate_tensor(
                BFloat16, cute.make_layout((TILE_N, DK_PE), stride=(SKP_STRIDE, 1)), 128
            )
            for _ in range(NSTAGE)
        ]
        sQn = smem.allocate_tensor(
            BFloat16, cute.make_layout((HEADS, DK_CK), stride=(SKC_STRIDE, 1)), 128
        )
        sQp = smem.allocate_tensor(
            BFloat16, cute.make_layout((HEADS, DK_PE), stride=(SKP_STRIDE, 1)), 128
        )
        sP = smem.allocate_tensor(
            BFloat16, cute.make_layout((TILE_N, HEADS + V8), stride=(HEADS + V8, 1)), 128
        )
        # low half of the hi/lo split softmax P (p = p_hi + p_lo, both bf16):
        # PV runs two MMAs so the P operand is fp32-exact to ~2^-17, keeping
        # the kernel's output within the baseline's own rounding noise.
        sP2 = smem.allocate_tensor(
            BFloat16, cute.make_layout((TILE_N, HEADS + V8), stride=(HEADS + V8, 1)), 128
        )
        sIdx = smem.allocate_tensor(Int32, cute.make_layout((ROWS,)), 16)
        sTileCnt = smem.allocate_tensor(Int32, cute.make_layout((NTILES,)), 16)
        sPack = smem.allocate_tensor(Int32, cute.make_layout((NTILES,)), 16)
        sFull = smem.allocate_tensor(Int32, cute.make_layout((NSPLIT,)), 16)
        sAny = smem.allocate_tensor(Int32, cute.make_layout((1,)), 16)
        sRed = smem.allocate_tensor(Float32, cute.make_layout((HEADS,)), 16)
        sDirect = smem.allocate_tensor(Int32, cute.make_layout((1,)), 16)
        sAf = smem.allocate_tensor(Int32, cute.make_layout((1,)), 16)
        sL = smem.allocate_tensor(Float32, cute.make_layout((8 * HEADS,)), 16)
        # stable-softmax exchange: per-k0-warp masked tile maxes (sMx4), the
        # CTA's running per-head max (sMxF, shipped with l at merge time) and
        # the per-tile accumulator rescale factors exp2(m_old - m_new) (sMxS).
        sMx4 = smem.allocate_tensor(Float32, cute.make_layout((4 * HEADS,)), 16)
        sMxF = smem.allocate_tensor(Float32, cute.make_layout((HEADS,)), 16)
        sMxS = smem.allocate_tensor(Float32, cute.make_layout((HEADS,)), 16)
        sFact = None
        if cutlass.const_expr(self.cluster == 0):
            # gmem-merge per-(split, head) rescale factors
            sFact = smem.allocate_tensor(
                Float32, cute.make_layout((NSPLIT * HEADS,)), 16
            )
        sSredT = smem.allocate_tensor(Float32, cute.make_layout((4 * 512,)), 16)
        sBar = smem.allocate_tensor(Int64, cute.make_layout((6,)), 16)
        sAfIdx = smem.allocate_tensor(
            Int32, cute.make_layout((max(TOPK - ROWS, 4),)), 128
        )
        sLslot = None
        if cutlass.const_expr(self.cluster):
            sLslot = smem.allocate_tensor(Float32, cute.make_layout((NSPLIT * 2,)), 16)
        flags_base = mFlags.iterator
        # epilogue staging: acc_O streamed through the (dead) stage-0 K buffer
        # as Int32 pairs, then coalesced 16B vector stores to gmem.
        NGRP = HEADS * DV // V8  # 1024 16B groups in a [16, 512] bf16 plane
        sOb32 = cute.make_tensor(
            cute.recast_ptr(sKc[0].iterator, dtype=Int32),
            cute.make_layout((HEADS, DV // 2), stride=(SKC_STRIDE // 2, 1)),
        )
        # fp32 partial staging plane, also aliasing the dead stage-0 K buffer:
        # one 2080B (520 f32) row per head (16 x 2080 = 33280B <= 66560B stage).
        # acc_O lands as Int64 pairs, then 16B f32 vector pieces move it to the
        # partial consumers (gmem mPartO or DSMEM sSlot) with no bf16 rounding.
        NGRPF = HEADS * DV // 4  # 2048 16B pieces in a [16, 512] f32 plane
        SKC_F32P = SKC_STRIDE // 4  # 130 16B pieces per f32 plane row
        sOf64 = cute.make_tensor(
            cute.recast_ptr(sKc[0].iterator, dtype=Int64),
            cute.make_layout((HEADS, DV // 2), stride=(SKC_STRIDE // 2, 1)),
        )
        sOf_v = cute.make_tensor(
            cute.recast_ptr(sKc[0].iterator, dtype=Float32),
            cute.make_layout((HEADS * SKC_F32P, 4), stride=(4, 1)),
        )

        # 16B-vector views of global tensors, re-based to gmem-space pointers
        qn_g = cute.make_ptr(
            BFloat16, mQnope.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16
        )
        qpe_g = cute.make_ptr(
            BFloat16, mQpe.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16
        )
        ckv_g = cute.make_ptr(
            BFloat16, mCKV.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16
        )
        kpe_g = cute.make_ptr(
            BFloat16, mKPE.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16
        )
        qn_v = cute.make_tensor(
            qn_g,
            cute.make_layout((num_tokens * HEADS * (DK_CK // V8), V8), stride=(V8, 1)),
        )
        qpe_v = cute.make_tensor(
            qpe_g,
            cute.make_layout((num_tokens * HEADS * (DK_PE // V8), V8), stride=(V8, 1)),
        )
        ckv_v = cute.make_tensor(
            ckv_g,
            cute.make_layout((total_rows * (DK_CK // V8), V8), stride=(V8, 1)),
        )
        kpe_v = cute.make_tensor(
            kpe_g,
            cute.make_layout((total_rows * (DK_PE // V8), V8), stride=(V8, 1)),
        )
        sKc_v = [
            cute.make_tensor(
                sKc[st].iterator,
                cute.make_layout((TILE_N * (SKC_STRIDE // V8), V8), stride=(V8, 1)),
            )
            for st in range(len(sKc))
        ]
        sKp_v = [
            cute.make_tensor(
                sKp[st].iterator,
                cute.make_layout((TILE_N * (SKP_STRIDE // V8), V8), stride=(V8, 1)),
            )
            for st in range(len(sKp))
        ]
        # The scalar micro consumer does not need the padded ldmatrix layout.
        # Exact-validated consecutive runs can therefore land as compact
        # 8-row rectangles and be consumed with ordinary 128-bit loads.
        sKcm = cute.make_tensor(
            sKc[0].iterator,
            cute.make_layout((TILE_N, DK_CK), stride=(DK_CK, 1)),
        )
        sKpm = cute.make_tensor(
            sKp[0].iterator,
            cute.make_layout((TILE_N, DK_PE), stride=(DK_PE, 1)),
        )
        sKcm_v = cute.make_tensor(
            sKc[0].iterator,
            cute.make_layout(
                (TILE_N * (DK_CK // V8), V8), stride=(V8, 1)
            ),
        )
        atom16 = cute.make_copy_atom(
            cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=128
        )
        atom16univ = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
        )
        atomF4 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )
        # cluster merge: st.async remotes (value rides registers; no async-proxy
        # fence needed). Per-sender landing slots hold one l word each
        # (sLslot) and one 1KB partial head row per sender (sSlot) - a
        # dedicated buffer: stage smem stays live until the consumer's own
        # loop drains, so remote pushes must not overlap it.
        atomDS16 = cute.make_copy_atom(
            cpasync.CopyDsmemStoreOp(), BFloat16, num_bits_per_copy=128
        )
        atomDS4 = cute.make_copy_atom(
            cpasync.CopyDsmemStoreOp(), Float32, num_bits_per_copy=32
        )
        atomDS8 = cute.make_copy_atom(
            cpasync.CopyDsmemStoreOp(), Float32, num_bits_per_copy=64
        )
        atomDSF = cute.make_copy_atom(
            cpasync.CopyDsmemStoreOp(), Float32, num_bits_per_copy=128
        )
        # one-shot 2048B cluster bulk copy (cp.async.bulk shared::cluster <-
        # shared::cta) for the fp32 partial head-row push
        atomBS2S = cute.make_copy_atom(
            cpasync.CopyBulkS2SOp(), Float32, num_bits_per_copy=DV * 32
        )
        sSlot = None
        sSlot32 = None
        if cutlass.const_expr(self.cluster):
            SROWS = min(NSPLIT, 16)
            sSlot = smem.allocate_tensor(
                Float32, cute.make_layout((SROWS, DV), stride=(DV, 1)), 128
            )
        # sSredT as 16B chunks (4 f32 each) for the vectorized k-partial reduce
        sRedV = cute.make_tensor(
            sSredT.iterator,
            cute.make_layout((4 * 512 // 4, 4), stride=(4, 1)),
        )
        rRedT = cute.make_rmem_tensor((4, 4), Float32)
        # one TMA bulk copy per (row, cache): 1024B ckv + 128B kpe
        atomBK = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(), BFloat16, num_bits_per_copy=DK_CK * 16
        )
        atomBP = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(), BFloat16, num_bits_per_copy=DK_PE * 16
        )
        atomBKM = cute.make_copy_atom(
            cpasync.CopyBulkG2SMulticastOp(),
            BFloat16,
            num_bits_per_copy=DK_CK * 16,
        )
        atomBPM = cute.make_copy_atom(
            cpasync.CopyBulkG2SMulticastOp(),
            BFloat16,
            num_bits_per_copy=DK_PE * 16,
        )
        atomBK8 = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(),
            BFloat16,
            num_bits_per_copy=8 * DK_CK * 16,
        )
        atomBP8 = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(),
            BFloat16,
            num_bits_per_copy=8 * DK_PE * 16,
        )
        rowK = cute.make_layout((DK_CK,))
        rowP = cute.make_layout((DK_PE,))
        rowK8 = cute.make_layout((8 * DK_CK,))
        rowP8 = cute.make_layout((8 * DK_PE,))

        # ============================== PROLOGUE =============================
        # 1) Constant-byte arms at kernel entry break the old
        #    idx->scan->arm->issue chain: every tile of a launched split
        #    always receives exactly TILE_N rows via TMA, invalid rows
        #    gathering cache row 0 (L2-hot, finite -> fully NaN-robust).
        #    Each warp pre-loads its own prologue-tile indices with uniform
        #    gmem loads issued before the first barrier, so the gathers
        #    launch the moment the arms are visible; the per-tile validity
        #    scan (only needed for MMA-time skip decisions) now overlaps
        #    the gathers instead of gating them.
        SPARSE = self.sparse
        row0 = split * ROWS  # legacy blocked base (afy/dense paths below use tile bases)
        # strided tile->split mapping: CTA (split s) owns tiles {s + j*NSPLIT},
        # i.e. global slots [(s + j*NSPLIT)*TILE_N, +TILE_N) for j in 0..NTILES-1.
        # Prefix-packed tokens with <= NSPLIT*TILE_N valid rows then put at most
        # ONE nonempty tile on each split (parallel consume) instead of NTILES
        # serial tiles on the low splits. NTILES==1 configs are unchanged
        # (tile base == split*TILE_N == row0).
        tb0 = split * TILE_N
        RPW = TILE_N // (NTHREADS // 32)  # gathered rows per warp = 8
        rL = cute.make_rmem_tensor((1,), Float32)
        rL2 = cute.make_rmem_tensor((2,), Float32)
        rPb = cute.make_rmem_tensor((V8,), BFloat16)
        rPf = cute.make_rmem_tensor((4,), Float32)
        t0r = [None] * RPW
        t1r = [None] * RPW
        cnt0 = Int32(0)
        cnt1 = Int32(0)
        TB1 = (NSPLIT if NTILES > 1 else 0) * TILE_N  # tile-1 base offset (dummy=own tile at NTILES==1)
        for rr in cutlass.range_constexpr(RPW):
            t0r[rr] = mIdx[token, tb0 + warp_id * RPW + rr]
            # unused at NTILES==1 (offset 0 keeps the dummy index in bounds)
            t1r[rr] = mIdx[token, tb0 + TB1 + warp_id * RPW + rr]
            # per-warp valid counts (SPARSE arms ride these registers;
            # DENSE folds them to dead code).
            cnt0 = cnt0 + ((t0r[rr] >> 31) + 1)
            cnt1 = cnt1 + ((t1r[rr] >> 31) + 1)
        qproducer_rank = Int32(0)
        if cutlass.const_expr(self.q_mcast == 4):
            # T=2 captured rows leave ranks 15/31 empty; issuing from those
            # ranks avoids queueing query transport behind active KV gathers.
            qproducer_rank = Int32(self.cluster - 1)
        # split 0 only: sibling-slot prefetch for the solo check, issued at
        # kernel entry (LSU path, parallel to the TMA dispatch below) so the
        # pre-loop pv check never stalls. Branchless OOB clamp; exact for any
        # -1 pattern. Sibling slots are read as 128-bit quad vectors; the tail
        # quad clamps to the last valid quad (duplicates are harmless for the
        # OR-of-validity check).
        # SPARSE sibling-slot prefetch: warps 0-3 only (they also own the
        # probe), so the probe below syncs 128 threads on a named barrier
        # instead of funneling all 256 warps before the main loop - the busy
        # split-0 path of merge tokens keeps its warp-entry skew.
        AFY_NQ = (TOPK - ROWS + 3) // 4  # quads over siblings (480 for ns16)
        AFY_W = NTHREADS // 2 if SPARSE else NTHREADS
        AFY_IT = (AFY_NQ + AFY_W - 1) // AFY_W
        afy = cute.make_rmem_tensor((max(AFY_IT, 1) * 4,), Int32)
        atomAfy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128
        )
        # T=2's duplicated exact tail proof is latency-bound. Dedicated G2R
        # traits preserve wide transactions and mark this one-use stream
        # no-allocate in L1 while promoting the coalesced line in L2.
        atomMicroIdx = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            Int32,
            num_bits_per_copy=128,
            l2_prefetch_size=cute.nvgpu.L2PrefetchSize.SIZE_256B,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        atomMicroIdx256 = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            Int32,
            num_bits_per_copy=256,
            l2_prefetch_size=cute.nvgpu.L2PrefetchSize.SIZE_256B,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        SIBB = (NSPLIT - 1) * TILE_N  # sibling slots per tile-block (split 0 view)
        atomAfB = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(),
            Int32,
            num_bits_per_copy=max(SIBB if NSPLIT > 1 else 4, 4) * 32,
        )
        if cutlass.const_expr(self.direct_afy):
            # Direct exact sibling prefetch for the T=8 body.  The logical
            # scan concatenates each strided tile block after excluding split
            # zero's own leading tile; map those quads back to aligned global
            # addresses and retain them in the existing afy registers.
            if split == Int32(0):
                if tidx < Int32(AFY_W):
                    for i in cutlass.range_constexpr(AFY_IT):
                        q4 = i * AFY_W + tidx
                        dx4 = q4 - Int32(AFY_NQ - 1)
                        qq4 = Int32(AFY_NQ - 1) + (dx4 & (dx4 >> 31))
                        jb = Int32(0)
                        jq = qq4
                        if NTILES > 1:
                            if qq4 >= Int32(SIBB // 4):
                                jb = Int32(1)
                                jq = qq4 - Int32(SIBB // 4)
                        src4 = cute.make_tensor(
                            cute.make_ptr(
                                Int32,
                                (
                                    mIdx.iterator
                                    + token.to(Int64) * TOPK
                                    + jb * NSPLIT * TILE_N
                                    + TILE_N
                                    + jq * 4
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout((4,)),
                        )
                        cute.copy(
                            atomAfy,
                            src4,
                            cute.make_tensor(
                                afy.iterator + i * 4, cute.make_layout((4,))
                            ),
                        )
        sAny[0] = Int32(0)
        sAf[0] = Int32(0)
        sDirect[0] = Int32(0)
        if cutlass.const_expr(self.pair_scan_marker):
            if split == Int32(0):
                if tidx == Int32(0):
                    # Outside pair16's sixteen sender landing slots; live
                    # only until split 0 ships its normal partial.
                    sLslot[NSPLIT] = Float32(0.0)
        if tidx == 0:
            # one arrival from tidx0 for both stage barriers
            cute.arch.mbarrier_init(sBar.iterator + 0, 1)
            cute.arch.mbarrier_init(sBar.iterator + 1, 1)
            cute.arch.mbarrier_init(sBar.iterator + 2, 1)
            if cutlass.const_expr(self.micro):
                # micro-window gather barrier (armed exact-byte at the
                # candidate combine step below, after issue)
                cute.arch.mbarrier_init(sBar.iterator + 5, 1)
            if cutlass.const_expr(self.cluster):
                # cluster merge bar: one arrival per CTA of the (half-)cluster
                # (each split's tidx < CSZ thread), tx-driven data tracking
                cute.arch.mbarrier_init(sBar.iterator + 3, self.cluster)
            if cutlass.const_expr(self.sparse and not self.direct_afy):
                cute.arch.mbarrier_init(sBar.iterator + 4, 1)
                # afy arm: ONLY split 0's CTA expects bytes (branchless)
                afd = (split | (Int32(0) - split)) >> 31  # -1 iff split != 0
                cute.arch.mbarrier_arrive_and_expect_tx(
                    sBar.iterator + 4,
                    Int32(max(TOPK - ROWS, 4) * 4) * (afd + 1),
                )
            cute.arch.mbarrier_init_fence()
            if not SPARSE:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    sBar.iterator + 0, Int32(TILE_N * (DK_CK + DK_PE) * 2)
                )
                if NTILES > 1:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        sBar.iterator + 1, Int32(TILE_N * (DK_CK + DK_PE) * 2)
                    )
            # barQ arms here (canonical expect-before-completion order);
            # warps 6/7 issue Q against this arm below.
            if cutlass.const_expr(self.q_rect16):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    sBar.iterator + 2,
                    Int32(HEADS * (SKC_STRIDE + SKP_STRIDE) * 2),
                )
            else:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    sBar.iterator + 2, Int32(HEADS * DK * 2)
                )
        cute.arch.barrier()  # init/arms visible to all issuers
        if cutlass.const_expr(IKET_TRACE):
            cute.experimental.iket.range_push("pro_issue")
        # =================== SMALL-TOKEN MICROKERNEL ====================
        # Candidate iff slot 0 is valid and slot MICRO_NCAP is -1 (probe
        # values in flight since kernel entry). On a candidate, every head
        # CTA speculatively TMA-gathers the valid window rows into stage 0
        # (bar 5) straight off the u0r registers, overlapping an exact
        # full-tail scan; if the token's valid indices are a prefix of the
        # window, one CTA per head consumes the staged rows with fp32
        # scalar QK / softmax / PV (hi/lo bf16 P, single bf16 round at the
        # store) and exits before any cluster/merge choreography. Any other
        # -1 pattern drains the speculative gathers and falls through to
        # the untouched general path. Nothing async is pending at either
        # exit: Q/gathers of the general path issue strictly below.
        if cutlass.const_expr(self.micro):
            mprobe0 = cute.arch.shuffle_sync(mprobe0, 0)
            mprobeC = cute.arch.shuffle_sync(mprobeC, 0)
            mcand = ((mprobe0 >> 31) + 1) & ((mprobeC >> 31) & 1)
            if mcand != 0:
                if cutlass.const_expr(IKET_TRACE):
                    cute.experimental.iket.mark("micro_enter")
                    cute.experimental.iket.range_push("micro_gather_issue")
                # -- speculative window gathers (valid slots only; exact
                # bytes armed after the count reduce below). ns32 keeps
                # only the first 16 CTAs of a token (one per head).
                iwm = Int32(0)  # physical rows issued by this warp
                if split < Int32(HEADS):
                    ridxm = u0r[0]
                    if ridxm >= 0:
                        if ridxm <= total_rows - Int32(RPW):
                            r64m = ridxm.to(Int64)
                            cute.copy(
                                atomBK8,
                                cute.make_tensor(
                                    ckv_g + r64m * DK_CK, rowK8
                                ),
                                cute.make_tensor(
                                    sKcm.iterator + warp_id * RPW * DK_CK,
                                    rowK8,
                                ),
                                mbar_ptr=sBar.iterator + 5,
                            )
                            cute.copy(
                                atomBP8,
                                cute.make_tensor(
                                    kpe_g + r64m * DK_PE, rowP8
                                ),
                                cute.make_tensor(
                                    sKpm.iterator + warp_id * RPW * DK_PE,
                                    rowP8,
                                ),
                                mbar_ptr=sBar.iterator + 5,
                            )
                            iwm = Int32(RPW)
                        else:
                            # Preserve exactness at the flattened cache tail:
                            # a rectangle may not read beyond total_rows.
                            for rr in cutlass.range_constexpr(RPW):
                                ridxmr = u0r[rr]
                                if ridxmr >= 0:
                                    r64m = ridxmr.to(Int64)
                                    cute.copy(
                                        atomBK,
                                        cute.make_tensor(
                                            ckv_g + r64m * DK_CK, rowK
                                        ),
                                        cute.make_tensor(
                                            sKcm.iterator
                                            + (warp_id * RPW + rr) * DK_CK,
                                            rowK,
                                        ),
                                        mbar_ptr=sBar.iterator + 5,
                                    )
                                    cute.copy(
                                        atomBP,
                                        cute.make_tensor(
                                            kpe_g + r64m * DK_PE, rowP
                                        ),
                                        cute.make_tensor(
                                            sKpm.iterator
                                            + (warp_id * RPW + rr) * DK_PE,
                                            rowP,
                                        ),
                                        mbar_ptr=sBar.iterator + 5,
                                    )
                                    iwm = iwm + Int32(1)
                # -- q head row -> fp32 registers (lane owns dims
                # [8*lane, +8) and [256+8*lane, +8) of ckv, [2*lane, +2)
                # of pe); flies with the gathers.
                atomB32m = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=32
                )
                hq = split & Int32(15)
                qb64 = (token * HEADS + hq).to(Int64)
                rQa = cute.make_rmem_tensor((V8,), BFloat16)
                rQb = cute.make_rmem_tensor((V8,), BFloat16)
                rQp = cute.make_rmem_tensor((2,), BFloat16)
                # Re-assert the alignment implied by the physical tensor
                # layout after adding the dynamic (token, head, lane)
                # offsets.  Each q_nope lane chunk starts on a 16-byte
                # boundary and each q_pe lane pair on a 4-byte boundary, but
                # CuTe otherwise conservatively degrades the derived pointer
                # to scalar BF16 alignment and rejects the vector copy atom.
                qam = cute.make_ptr(
                    BFloat16,
                    (qn_g + qb64 * DK_CK + lane_id * V8).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                qbm = cute.make_ptr(
                    BFloat16,
                    (
                        qn_g
                        + qb64 * DK_CK
                        + Int32(DK_CK // 2)
                        + lane_id * V8
                    ).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                qpm = cute.make_ptr(
                    BFloat16,
                    (qpe_g + qb64 * DK_PE + lane_id * 2).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                cute.copy(
                    atom16univ,
                    cute.make_tensor(qam, cute.make_layout((V8,))),
                    rQa,
                )
                cute.copy(
                    atom16univ,
                    cute.make_tensor(qbm, cute.make_layout((V8,))),
                    rQb,
                )
                cute.copy(
                    atomB32m,
                    cute.make_tensor(qpm, cute.make_layout((2,))),
                    rQp,
                )
                if cutlass.const_expr(IKET_TRACE):
                    cute.experimental.iket.range_pop()
                    cute.experimental.iket.range_push("micro_tail_scan")
                # -- exact tail scan (slots [64, 2048) all -1?) overlapped
                # with the gather flight; scratch lives in the (unused this
                # path) sQn region.
                sMiF = cute.recast_ptr(sQn.iterator, dtype=Float32)
                sMiI = cute.recast_ptr(sQn.iterator, dtype=Int32)
                sMlg = cute.make_tensor(sMiF, cute.make_layout((TILE_N,)))
                sMpp = cute.make_tensor(sMiF + TILE_N, cute.make_layout((TILE_N,)))
                sMls = cute.make_tensor(sMiF + 2 * TILE_N, cute.make_layout((32,)))
                sMint = cute.make_tensor(
                    sMiI + (2 * TILE_N + 32), cute.make_layout((48,))
                )
                idxgm = cute.make_ptr(
                    Int32, mIdx.iterator.toint(), cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                ctail = Int32(0)
                if cutlass.const_expr(self.micro_idx_g2r256):
                    # The tail has 1984 i32 values: 248 aligned 32-byte
                    # vectors, one per active thread, with eight idle lanes.
                    rIt8 = cute.make_rmem_tensor((8,), Int32)
                    rIt8.fill(Int32(-1))
                    if tidx < Int32((TOPK - MICRO_NCAP) // 8):
                        idxp8m = cute.make_ptr(
                            Int32,
                            (
                                idxgm
                                + token.to(Int64) * TOPK
                                + MICRO_NCAP
                                + tidx.to(Int64) * 8
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        )
                        cute.copy(
                            atomMicroIdx256,
                            cute.make_tensor(idxp8m, cute.make_layout((8,))),
                            rIt8,
                        )
                    for e in cutlass.range_constexpr(8):
                        if cutlass.const_expr(MICRO_VOTE):
                            ctail = ctail | ((rIt8[e] >> 31) + 1)
                        else:
                            ctail = ctail + ((rIt8[e] >> 31) + 1)
                else:
                    # Two independent register buffers make both 128-bit
                    # rounds overlap instead of WAR-serializing.
                    rIt = cute.make_rmem_tensor((4,), Int32)
                    rIt2 = cute.make_rmem_tensor((4,), Int32)
                    rIt.fill(Int32(-1))
                    rIt2.fill(Int32(-1))
                    p1m = Int32(TILE_N // 4) + tidx
                    idxp1m = cute.make_ptr(
                        Int32,
                        (
                            idxgm
                            + token.to(Int64) * TOPK
                            + p1m.to(Int64) * 4
                        ).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    if cutlass.const_expr(self.micro_idx_g2r):
                        cute.copy(
                            atomMicroIdx,
                            cute.make_tensor(idxp1m, cute.make_layout((4,))),
                            rIt,
                        )
                    else:
                        cute.copy(
                            atomAfy,
                            cute.make_tensor(idxp1m, cute.make_layout((4,))),
                            rIt,
                        )
                    p2m = Int32(TILE_N // 4 + NTHREADS) + tidx
                    if p2m < Int32(TOPK // 4):
                        idxp2m = cute.make_ptr(
                            Int32,
                            (
                                idxgm
                                + token.to(Int64) * TOPK
                                + p2m.to(Int64) * 4
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        if cutlass.const_expr(self.micro_idx_g2r):
                            cute.copy(
                                atomMicroIdx,
                                cute.make_tensor(
                                    idxp2m, cute.make_layout((4,))
                                ),
                                rIt2,
                            )
                        else:
                            cute.copy(
                                atomAfy,
                                cute.make_tensor(
                                    idxp2m, cute.make_layout((4,))
                                ),
                                rIt2,
                            )
                    for e in cutlass.range_constexpr(4):
                        if cutlass.const_expr(MICRO_VOTE):
                            ctail = ctail | ((rIt[e] >> 31) + 1)
                            ctail = ctail | ((rIt2[e] >> 31) + 1)
                        else:
                            ctail = ctail + ((rIt[e] >> 31) + 1)
                            ctail = ctail + ((rIt2[e] >> 31) + 1)
                # Exact any-valid reduction: a single VOTE replaces the
                # five dependent shuffle-add steps exposed by IKET's 0.8us
                # tail-scan phase. The resulting bitmask is reduced by OR at
                # CTA scope, since only zero versus nonzero matters.
                if cutlass.const_expr(MICRO_VOTE):
                    ctail = cute.arch.vote_ballot_sync(ctail != Int32(0))
                else:
                    for off in (16, 8, 4, 2, 1):
                        ctail = ctail + cute.arch.shuffle_sync_bfly(ctail, off)
                # window count + last-valid-pos from the u0r registers
                # (warp-redundant; no shuffles needed)
                cwm = Int32(0)
                mwm = Int32(-1)
                seqm = Int32(0)
                for rr in cutlass.range_constexpr(RPW):
                    vvm = u0r[rr]
                    cwm = cwm + ((vvm >> 31) + 1)
                    posm = warp_id * RPW + Int32(rr)
                    mskm = Int32(0) - ((vvm >> 31) + 1)  # -1 iff valid
                    mwm = mwm + ((posm - mwm) & mskm)
                    if vvm >= Int32(0):
                        if vvm != mprobe0 + posm:
                            seqm = Int32(1)
                if lane_id == 0:
                    sMint[warp_id] = cwm
                    sMint[8 + warp_id] = mwm
                    sMint[16 + warp_id] = ctail
                    sMint[26 + warp_id] = iwm
                    sMint[34 + warp_id] = seqm
                cute.arch.barrier()
                if tidx == 0:
                    tnm = Int32(0)
                    tmm = Int32(-1)
                    ttm = Int32(0)
                    tim = Int32(0)
                    tsm = Int32(0)
                    for w in cutlass.range_constexpr(8):
                        tnm = tnm + sMint[w]
                        omm = sMint[8 + w]
                        tmm = tmm + (omm - tmm) * (Int32(0) - ((tmm - omm) >> 31))
                        if cutlass.const_expr(MICRO_VOTE):
                            ttm = ttm | sMint[16 + w]
                        else:
                            ttm = ttm + sMint[16 + w]
                        tim = tim + sMint[26 + w]
                        tsm = tsm | sMint[34 + w]
                    okm = Int32(0)
                    if tsm == Int32(0):
                        if ttm == Int32(0):
                            if tmm == tnm - Int32(1):
                                if tnm > Int32(0):
                                    okm = Int32(1)
                    sMint[24] = tnm
                    sMint[25] = okm
                    if split < Int32(HEADS):
                        # exact-byte arm: matches the per-slot-sign issue
                        # above for ANY -1 pattern
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            sBar.iterator + 5,
                            tim * Int32((DK_CK + DK_PE) * 2),
                        )
                cute.arch.barrier()
                mokm = sMint[25]
                mNm = sMint[24]
                if cutlass.const_expr(IKET_TRACE):
                    cute.experimental.iket.range_pop()
                if mokm != Int32(0):
                    if split >= Int32(HEADS):
                        _thread_exit()  # redundant ns32 sibling cluster
                    # fp32 q fragments (shared mapping with the K reads)
                    rQf = cute.make_rmem_tensor((16,), Float32)
                    rQpf = cute.make_rmem_tensor((2,), Float32)
                    for e in cutlass.range_constexpr(V8):
                        rQf[e] = rQa[e].to(Float32)
                        rQf[V8 + e] = rQb[e].to(Float32)
                    rQpf[0] = rQp[0].to(Float32)
                    rQpf[1] = rQp[1].to(Float32)
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_push("micro_gather_wait")
                    cute.arch.mbarrier_wait(sBar.iterator + 5, Int32(0))
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_pop()
                        cute.experimental.iket.range_push("micro_qk")
                    # -- QK: one row per warp per pass, fp32 dot + butterfly
                    SKCPm = DK_CK // V8
                    rKa = cute.make_rmem_tensor((V8,), BFloat16)
                    rKb = cute.make_rmem_tensor((V8,), BFloat16)
                    rKc = cute.make_rmem_tensor((V8,), BFloat16)
                    rKd = cute.make_rmem_tensor((V8,), BFloat16)
                    # per-warp running max of the SCALED dots, folded into
                    # the dot loops so the softmax max costs no extra barrier
                    # (global max = max over the 8 per-warp maxes, exchanged
                    # via the existing logits barrier).
                    mxw = Float32(NEG_BIG)
                    jrow = Int32(0) + warp_id
                    # Interleave two independent row dots per warp. Each row
                    # retains the exact scalar fp32 accumulation order, while
                    # row B fills row A's FMA dependency slots. Full 16-row
                    # groups advance together; the original loop handles the
                    # remaining rows unchanged.
                    # Full-suite validations: 37.2242x / 37.2366x, 23/23.
                    if cutlass.const_expr(self.nsplit == 16):
                        # T=6 has enough concurrent CTAs to tolerate the
                        # larger live set. Four independent rows cover a full
                        # 32-row window before the proven pair/scalar tails.
                        rKe = cute.make_rmem_tensor((V8,), BFloat16)
                        rKf = cute.make_rmem_tensor((V8,), BFloat16)
                        rKg = cute.make_rmem_tensor((V8,), BFloat16)
                        rKh = cute.make_rmem_tensor((V8,), BFloat16)
                        while jrow + Int32(24) < mNm:
                            jrowb = jrow + Int32(8)
                            jrowc = jrow + Int32(16)
                            jrowd = jrow + Int32(24)
                            cute.copy(
                                atom16univ,
                                sKcm_v[jrow * SKCPm + lane_id, None],
                                rKa,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[
                                    jrow * SKCPm
                                    + (Int32(32) + lane_id),
                                    None,
                                ],
                                rKb,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[jrowb * SKCPm + lane_id, None],
                                rKc,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[
                                    jrowb * SKCPm
                                    + (Int32(32) + lane_id),
                                    None,
                                ],
                                rKd,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[jrowc * SKCPm + lane_id, None],
                                rKe,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[
                                    jrowc * SKCPm
                                    + (Int32(32) + lane_id),
                                    None,
                                ],
                                rKf,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[jrowd * SKCPm + lane_id, None],
                                rKg,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[
                                    jrowd * SKCPm
                                    + (Int32(32) + lane_id),
                                    None,
                                ],
                                rKh,
                            )
                            kp0m = sKpm[jrow, lane_id * 2].to(Float32)
                            kp1m = sKpm[
                                jrow, lane_id * 2 + 1
                            ].to(Float32)
                            kp0b = sKpm[jrowb, lane_id * 2].to(Float32)
                            kp1b = sKpm[
                                jrowb, lane_id * 2 + 1
                            ].to(Float32)
                            kp0c = sKpm[jrowc, lane_id * 2].to(Float32)
                            kp1c = sKpm[
                                jrowc, lane_id * 2 + 1
                            ].to(Float32)
                            kp0d = sKpm[jrowd, lane_id * 2].to(Float32)
                            kp1d = sKpm[
                                jrowd, lane_id * 2 + 1
                            ].to(Float32)
                            dotm = rQpf[0] * kp0m
                            dotb = rQpf[0] * kp0b
                            dotc = rQpf[0] * kp0c
                            dotd = rQpf[0] * kp0d
                            dotm = dotm + rQpf[1] * kp1m
                            dotb = dotb + rQpf[1] * kp1b
                            dotc = dotc + rQpf[1] * kp1c
                            dotd = dotd + rQpf[1] * kp1d
                            for e in cutlass.range_constexpr(V8):
                                dotm = (
                                    dotm + rQf[e] * rKa[e].to(Float32)
                                )
                                dotb = (
                                    dotb + rQf[e] * rKc[e].to(Float32)
                                )
                                dotc = (
                                    dotc + rQf[e] * rKe[e].to(Float32)
                                )
                                dotd = (
                                    dotd + rQf[e] * rKg[e].to(Float32)
                                )
                                dotm = (
                                    dotm
                                    + rQf[V8 + e] * rKb[e].to(Float32)
                                )
                                dotb = (
                                    dotb
                                    + rQf[V8 + e] * rKd[e].to(Float32)
                                )
                                dotc = (
                                    dotc
                                    + rQf[V8 + e] * rKf[e].to(Float32)
                                )
                                dotd = (
                                    dotd
                                    + rQf[V8 + e] * rKh[e].to(Float32)
                                )
                            for off in (16, 8, 4, 2, 1):
                                shm = cute.arch.shuffle_sync_bfly(dotm, off)
                                shb = cute.arch.shuffle_sync_bfly(dotb, off)
                                shc = cute.arch.shuffle_sync_bfly(dotc, off)
                                shd = cute.arch.shuffle_sync_bfly(dotd, off)
                                dotm = dotm + shm
                                dotb = dotb + shb
                                dotc = dotc + shc
                                dotd = dotd + shd
                            mxw = cute.arch.fmax(mxw, dotm * scale_log2)
                            mxw = cute.arch.fmax(mxw, dotb * scale_log2)
                            mxw = cute.arch.fmax(mxw, dotc * scale_log2)
                            mxw = cute.arch.fmax(mxw, dotd * scale_log2)
                            if lane_id == 0:
                                sMlg[jrow] = dotm
                                sMlg[jrowb] = dotb
                                sMlg[jrowc] = dotc
                                sMlg[jrowd] = dotd
                            jrow = jrow + Int32(32)
                    if cutlass.const_expr(self.nsplit == 32):
                        # At T=2, three independent recurrences better match
                        # fp32 FMA latency without the four-row live set.
                        rKe = cute.make_rmem_tensor((V8,), BFloat16)
                        rKf = cute.make_rmem_tensor((V8,), BFloat16)
                        while jrow + Int32(16) < mNm:
                            jrowb = jrow + Int32(8)
                            jrowc = jrow + Int32(16)
                            cute.copy(
                                atom16univ,
                                sKcm_v[jrow * SKCPm + lane_id, None],
                                rKa,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[
                                    jrow * SKCPm
                                    + (Int32(32) + lane_id),
                                    None,
                                ],
                                rKb,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[jrowb * SKCPm + lane_id, None],
                                rKc,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[
                                    jrowb * SKCPm
                                    + (Int32(32) + lane_id),
                                    None,
                                ],
                                rKd,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[jrowc * SKCPm + lane_id, None],
                                rKe,
                            )
                            cute.copy(
                                atom16univ,
                                sKcm_v[
                                    jrowc * SKCPm
                                    + (Int32(32) + lane_id),
                                    None,
                                ],
                                rKf,
                            )
                            kp0m = sKpm[jrow, lane_id * 2].to(Float32)
                            kp1m = sKpm[
                                jrow, lane_id * 2 + 1
                            ].to(Float32)
                            kp0b = sKpm[jrowb, lane_id * 2].to(Float32)
                            kp1b = sKpm[
                                jrowb, lane_id * 2 + 1
                            ].to(Float32)
                            kp0c = sKpm[jrowc, lane_id * 2].to(Float32)
                            kp1c = sKpm[
                                jrowc, lane_id * 2 + 1
                            ].to(Float32)
                            dotm = rQpf[0] * kp0m
                            dotb = rQpf[0] * kp0b
                            dotc = rQpf[0] * kp0c
                            dotm = dotm + rQpf[1] * kp1m
                            dotb = dotb + rQpf[1] * kp1b
                            dotc = dotc + rQpf[1] * kp1c
                            for e in cutlass.range_constexpr(V8):
                                dotm = (
                                    dotm + rQf[e] * rKa[e].to(Float32)
                                )
                                dotb = (
                                    dotb + rQf[e] * rKc[e].to(Float32)
                                )
                                dotc = (
                                    dotc + rQf[e] * rKe[e].to(Float32)
                                )
                                dotm = (
                                    dotm
                                    + rQf[V8 + e] * rKb[e].to(Float32)
                                )
                                dotb = (
                                    dotb
                                    + rQf[V8 + e] * rKd[e].to(Float32)
                                )
                                dotc = (
                                    dotc
                                    + rQf[V8 + e] * rKf[e].to(Float32)
                                )
                            for off in (16, 8, 4, 2, 1):
                                shm = cute.arch.shuffle_sync_bfly(dotm, off)
                                shb = cute.arch.shuffle_sync_bfly(dotb, off)
                                shc = cute.arch.shuffle_sync_bfly(dotc, off)
                                dotm = dotm + shm
                                dotb = dotb + shb
                                dotc = dotc + shc
                            mxw = cute.arch.fmax(mxw, dotm * scale_log2)
                            mxw = cute.arch.fmax(mxw, dotb * scale_log2)
                            mxw = cute.arch.fmax(mxw, dotc * scale_log2)
                            if lane_id == 0:
                                sMlg[jrow] = dotm
                                sMlg[jrowb] = dotb
                                sMlg[jrowc] = dotc
                            jrow = jrow + Int32(24)
                    while jrow + Int32(8) < mNm:
                        jrowb = jrow + Int32(8)
                        cute.copy(
                            atom16univ,
                            sKcm_v[jrow * SKCPm + lane_id, None],
                            rKa,
                        )
                        cute.copy(
                            atom16univ,
                            sKcm_v[
                                jrow * SKCPm + (Int32(32) + lane_id), None
                            ],
                            rKb,
                        )
                        cute.copy(
                            atom16univ,
                            sKcm_v[jrowb * SKCPm + lane_id, None],
                            rKc,
                        )
                        cute.copy(
                            atom16univ,
                            sKcm_v[
                                jrowb * SKCPm + (Int32(32) + lane_id), None
                            ],
                            rKd,
                        )
                        kp0m = sKpm[jrow, lane_id * 2].to(Float32)
                        kp1m = sKpm[jrow, lane_id * 2 + 1].to(Float32)
                        kp0b = sKpm[jrowb, lane_id * 2].to(Float32)
                        kp1b = sKpm[jrowb, lane_id * 2 + 1].to(Float32)
                        dotm = rQpf[0] * kp0m
                        dotb = rQpf[0] * kp0b
                        dotm = dotm + rQpf[1] * kp1m
                        dotb = dotb + rQpf[1] * kp1b
                        for e in cutlass.range_constexpr(V8):
                            dotm = dotm + rQf[e] * rKa[e].to(Float32)
                            dotb = dotb + rQf[e] * rKc[e].to(Float32)
                            dotm = (
                                dotm
                                + rQf[V8 + e] * rKb[e].to(Float32)
                            )
                            dotb = (
                                dotb
                                + rQf[V8 + e] * rKd[e].to(Float32)
                            )
                        for off in (16, 8, 4, 2, 1):
                            shm = cute.arch.shuffle_sync_bfly(dotm, off)
                            shb = cute.arch.shuffle_sync_bfly(dotb, off)
                            dotm = dotm + shm
                            dotb = dotb + shb
                        mxw = cute.arch.fmax(mxw, dotm * scale_log2)
                        mxw = cute.arch.fmax(mxw, dotb * scale_log2)
                        if lane_id == 0:
                            sMlg[jrow] = dotm
                            sMlg[jrowb] = dotb
                        jrow = jrow + Int32(16)
                    while jrow < mNm:
                        cute.copy(
                            atom16univ,
                            sKcm_v[jrow * SKCPm + lane_id, None],
                            rKa,
                        )
                        cute.copy(
                            atom16univ,
                            sKcm_v[
                                jrow * SKCPm + (Int32(32) + lane_id), None
                            ],
                            rKb,
                        )
                        kp0m = sKpm[jrow, lane_id * 2].to(Float32)
                        kp1m = sKpm[jrow, lane_id * 2 + 1].to(Float32)
                        dotm = rQpf[0] * kp0m
                        dotm = dotm + rQpf[1] * kp1m
                        for e in cutlass.range_constexpr(V8):
                            dotm = dotm + rQf[e] * rKa[e].to(Float32)
                            dotm = dotm + rQf[V8 + e] * rKb[e].to(Float32)
                        for off in (16, 8, 4, 2, 1):
                            dotm = dotm + cute.arch.shuffle_sync_bfly(dotm, off)
                        mxw = cute.arch.fmax(mxw, dotm * scale_log2)
                        if lane_id == 0:
                            sMlg[jrow] = dotm
                        jrow = jrow + Int32(8)
                    if lane_id == 0:
                        sMls[16 + warp_id] = mxw
                    cute.arch.barrier()  # logits visible (and per-warp maxes)
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_pop()
                        cute.experimental.iket.range_push("micro_softmax")
                    # -- softmax: keep scalar-micro P directly in fp32 for
                    # the fp32 PV blend; l uses the same raw fp32 exps.
                    # Stable: p = exp2(z - max); the global max is the fmax
                    # of the 8 per-warp QK maxes already published behind the
                    # logits barrier, so stability costs no extra barrier.
                    lsm = Float32(0.0)
                    if tidx < mNm:
                        mmx = cute.arch.fmax(
                            cute.arch.fmax(
                                cute.arch.fmax(sMls[16], sMls[17]),
                                cute.arch.fmax(sMls[18], sMls[19]),
                            ),
                            cute.arch.fmax(
                                cute.arch.fmax(sMls[20], sMls[21]),
                                cute.arch.fmax(sMls[22], sMls[23]),
                            ),
                        )
                        eem = cute.arch.exp2(sMlg[tidx] * scale_log2 - mmx)
                        sMpp[tidx] = eem
                        lsm = eem
                    for off in (16, 8, 4, 2, 1):
                        lsm = lsm + cute.arch.shuffle_sync_bfly(lsm, off)
                    if lane_id == 0:
                        sMls[warp_id] = lsm
                    cute.arch.barrier()
                    if tidx == 0:
                        ltm = Float32(0.0)
                        for w in cutlass.range_constexpr(8):
                            ltm = ltm + sMls[w]
                        sMls[8] = Float32(1.0) / ltm
                        # lse: global max over the 8 per-warp QK maxes + log2(l)
                        mmx0 = sMls[16]
                        for w in cutlass.range_constexpr(1, 8):
                            mmx0 = cute.arch.fmax(mmx0, sMls[16 + w])
                        mLse[token, split] = mmx0 + cute.math.log2(
                            ltm, fastmath=True
                        )
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_pop()
                        cute.experimental.iket.range_push("micro_pv")
                    # -- PV: threads (rowblk = tidx>>6, colgrp = tidx&63)
                    # accumulate 8 output cols over rows rowblk, rowblk+4,...
                    # in fp32; replica partials reduce through sSredT.
                    rowbm = tidx >> 6
                    colgm = tidx & Int32(63)
                    rVa = cute.make_rmem_tensor((V8,), BFloat16)
                    raccm = cute.make_rmem_tensor((4, 2), Float32)
                    raccm.fill(Float32(0.0))
                    r3m = Int32(0) + rowbm
                    while r3m < mNm:
                        cute.copy(
                            atom16univ,
                            sKcm_v[r3m * SKCPm + colgm, None],
                            rVa,
                        )
                        ppm = sMpp[r3m]
                        for e in cutlass.range_constexpr(4):
                            raccm[e, 0] = raccm[e, 0] + ppm * rVa[e].to(Float32)
                            raccm[e, 1] = raccm[e, 1] + ppm * rVa[4 + e].to(Float32)
                        r3m = r3m + Int32(4)
                    for pce in cutlass.range_constexpr(2):
                        cute.copy(
                            atomF4,
                            raccm[None, pce],
                            sRedV[rowbm * 128 + colgm * 2 + pce, None],
                        )
                    cute.arch.barrier()  # replica partials visible
                    # PV consumes unnormalized P. This existing reduction
                    # barrier also orders tidx0's reciprocal store, avoiding
                    # a separate CTA barrier before the PV loop.
                    minvm = sMls[8]
                    if tidx < Int32(64):
                        rSum = cute.make_rmem_tensor((4, 2), Float32)
                        rTm = cute.make_rmem_tensor((4,), Float32)
                        rSum.fill(Float32(0.0))
                        for rb in cutlass.range_constexpr(4):
                            for pce in cutlass.range_constexpr(2):
                                cute.copy(
                                    atomF4,
                                    sRedV[rb * 128 + tidx * 2 + pce, None],
                                    rTm,
                                )
                                for e in cutlass.range_constexpr(4):
                                    rSum[e, pce] = rSum[e, pce] + rTm[e]
                        rOutV = cute.make_rmem_tensor((V8,), BFloat16)
                        for pce in cutlass.range_constexpr(2):
                            for e in cutlass.range_constexpr(4):
                                rOutV[pce * 4 + e] = BFloat16(
                                    rSum[e, pce] * minvm
                                )
                        if cutlass.const_expr(IKET_TRACE):
                            cute.experimental.iket.range_pop()
                            cute.experimental.iket.range_push("micro_store")
                        outgm = cute.make_ptr(
                            BFloat16, mOut.iterator.toint(),
                            cute.AddressSpace.gmem, assumed_align=16,
                        )
                        outpm = cute.make_ptr(
                            BFloat16,
                            (
                                outgm
                                + (token * HEADS + split).to(Int64) * DV
                                + tidx * V8
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        cute.copy(
                            atom16univ,
                            rOutV,
                            cute.make_tensor(outpm, cute.make_layout((V8,))),
                        )
                    if cutlass.const_expr(IKET_TRACE):
                        if tidx >= Int32(64):
                            cute.experimental.iket.range_pop()
                            cute.experimental.iket.range_push("micro_store")
                        cute.experimental.iket.range_pop()
                        cute.experimental.iket.mark("micro_done")
                    _thread_exit()
                # non-prefix -1 pattern: drain the speculative gathers so
                # nothing is pending against bar 5, then run the untouched
                # general path (its stage-0 gathers/zero-fill fully rewrite
                # the window rows).
                if split < Int32(HEADS):
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_push("micro_fallback_drain")
                    cute.arch.mbarrier_wait(sBar.iterator + 5, Int32(0))
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_pop()
        if cutlass.const_expr(self.cluster):
            # start sync: arrive now (non-blocking), wait right after the
            # index scan below, before ANY remote op can be issued. Remote
            # arrivals on a peer's mbar are only safe once that peer passed
            # its init barrier -> its cluster arrive.
            # relaxed arrive: mbarrier_init_fence above already publishes the
            # inits cluster-wide; cluster_wait's acquire pairs with it. The
            # full-release arrive lowered to MEMBAR.ALL.GPU+ERRBAR (~0.35us of
            # entry stall, NCU run_002/003) for ordering nothing extra.
            cute.arch.cluster_arrive_relaxed()
            if cutlass.const_expr(self.q_mcast):
                # Only the producer waits here; peers continue their sparse
                # prologue and join at the original DSMEM rendezvous below.
                if split % Int32(self.cluster) == qproducer_rank:
                    cute.arch.cluster_wait()
        # 2) Q rides barQ (warps 6/7). Two placements by config: with a
        #    stage-1 to protect (NTILES>1) Q keeps its original early spot -
        #    deferring it delays stage-1's dispatch by 32 engine ops and
        #    costs more than the tile-0 relief buys. Single-tile configs
        #    (ns32, the T1/T2 officials) issue Q right after the stage-0
        #    gathers instead (Q-late block below): the CTA's 32 Q ops no
        #    longer precede its own tile-0 rows on the TMA engine, and Q
        #    still completes inside the tile-0 wait. Unguarded by sAny: an
        #    empty split's Q rides L2-hot lines, and an sAny gate would
        #    serialize the issue behind the scan.
        Q_EARLY = (not SPARSE) or NTILES > 1
        if Q_EARLY:
            qlocal = Int32(1)
            if cutlass.const_expr(self.q_mcast):
                qlocal = Int32(0)
                if split % Int32(self.cluster) == qproducer_rank:
                    mcastq = Int32(-1)
                    if cutlass.const_expr(self.q_rect16):
                        if tidx == Int32(0):
                            _issue_tma_rect_mcast(
                                rect_k_atom,
                                sQn.iterator,
                                sBar.iterator + 2,
                                Int32(0),
                                token * Int32(HEADS),
                                Int16(-1),
                            )
                            _issue_tma_rect_mcast(
                                rect_p_atom,
                                sQp.iterator,
                                sBar.iterator + 2,
                                Int32(0),
                                token * Int32(HEADS),
                                Int16(-1),
                            )
                    else:
                        # T=6: two producer warps issue one native row per
                        # head while all peer CTAs continue their index work.
                        if warp_id >= Int32(6):
                            for h2 in cutlass.range_constexpr(HEADS // 2):
                                h = (warp_id - Int32(6)) * (HEADS // 2) + h2
                                cute.copy(
                                    atomBKM,
                                    cute.make_tensor(
                                        qn_g
                                        + (token * HEADS + h).to(Int64) * DK_CK,
                                        rowK,
                                    ),
                                    cute.make_tensor(
                                        sQn.iterator + h * SKC_STRIDE, rowK
                                    ),
                                    mbar_ptr=sBar.iterator + 2,
                                    mcast_mask=mcastq,
                                )
                                cute.copy(
                                    atomBPM,
                                    cute.make_tensor(
                                        qpe_g
                                        + (token * HEADS + h).to(Int64) * DK_PE,
                                        rowP,
                                    ),
                                    cute.make_tensor(
                                        sQp.iterator + h * SKP_STRIDE, rowP
                                    ),
                                    mbar_ptr=sBar.iterator + 2,
                                    mcast_mask=mcastq,
                                )
            if qlocal != Int32(0):
                if warp_id >= 6:
                    for h2 in cutlass.range_constexpr(HEADS // 2):
                        h = (warp_id - 6) * (HEADS // 2) + h2
                        cute.copy(
                            atomBK,
                            cute.make_tensor(
                                qn_g
                                + (token * HEADS + h).to(Int64) * DK_CK,
                                rowK,
                            ),
                            cute.make_tensor(
                                sQn.iterator + h * SKC_STRIDE, rowK
                            ),
                            mbar_ptr=sBar.iterator + 2,
                        )
                        cute.copy(
                            atomBP,
                            cute.make_tensor(
                                qpe_g
                                + (token * HEADS + h).to(Int64) * DK_PE,
                                rowP,
                            ),
                            cute.make_tensor(
                                sQp.iterator + h * SKP_STRIDE, rowP
                            ),
                            mbar_ptr=sBar.iterator + 2,
                        )
        # 3) Prologue tile gathers. SPARSE: valid rows only (exact bytes
        #    armed per warp). DENSE: every row, invalid slots gathering
        #    cache row 0 (constant armed bytes, in bounds for every live
        #    page count including num_pages=1, L2-hot, finite and exactly
        #    masked downstream).
        if SPARSE:
            for rr in cutlass.range_constexpr(RPW):
                ridx = t0r[rr]
                if ridx >= 0:
                    r64 = ridx.to(Int64)
                    cute.copy(
                        atomBK,
                        cute.make_tensor(ckv_g + r64 * DK_CK, rowK),
                        cute.make_tensor(
                            sKc[0].iterator + (warp_id * RPW + rr) * SKC_STRIDE,
                            rowK,
                        ),
                        mbar_ptr=sBar.iterator + 0,
                    )
                    cute.copy(
                        atomBP,
                        cute.make_tensor(kpe_g + r64 * DK_PE, rowP),
                        cute.make_tensor(
                            sKp[0].iterator + (warp_id * RPW + rr) * SKP_STRIDE,
                            rowP,
                        ),
                        mbar_ptr=sBar.iterator + 0,
                    )
            if cutlass.const_expr(NTILES > 1):
                for rr in cutlass.range_constexpr(RPW):
                    ridx = t1r[rr]
                    if ridx >= 0:
                        r64 = ridx.to(Int64)
                        cute.copy(
                            atomBK,
                            cute.make_tensor(ckv_g + r64 * DK_CK, rowK),
                            cute.make_tensor(
                                sKc[1].iterator + (warp_id * RPW + rr) * SKC_STRIDE,
                                rowK,
                            ),
                            mbar_ptr=sBar.iterator + 1,
                        )
                        cute.copy(
                            atomBP,
                            cute.make_tensor(kpe_g + r64 * DK_PE, rowP),
                            cute.make_tensor(
                                sKp[1].iterator + (warp_id * RPW + rr) * SKP_STRIDE,
                                rowP,
                            ),
                            mbar_ptr=sBar.iterator + 1,
                        )
        else:
            for rr in cutlass.range_constexpr(RPW):
                ridx = t0r[rr]
                mneg = ridx >> 31  # arith: -1 (all-ones mask) iff invalid
                # ridx if valid, else cache row 0 (always in bounds; valid
                # indices keep their exact live flattened row).
                sel = ridx - (ridx & mneg)
                cute.copy(
                    atomBK,
                    cute.make_tensor(ckv_g + sel.to(Int64) * DK_CK, rowK),
                    cute.make_tensor(
                        sKc[0].iterator + (warp_id * RPW + rr) * SKC_STRIDE, rowK
                    ),
                    mbar_ptr=sBar.iterator + 0,
                )
                cute.copy(
                    atomBP,
                    cute.make_tensor(kpe_g + sel.to(Int64) * DK_PE, rowP),
                    cute.make_tensor(
                        sKp[0].iterator + (warp_id * RPW + rr) * SKP_STRIDE, rowP
                    ),
                    mbar_ptr=sBar.iterator + 0,
                )
            if cutlass.const_expr(NTILES > 1):
                for rr in cutlass.range_constexpr(RPW):
                    ridx = t1r[rr]
                    mneg = ridx >> 31  # arith: -1 (all-ones mask) iff invalid
                    # ridx if valid, else cache row 0 (always in bounds).
                    sel = ridx - (ridx & mneg)
                    cute.copy(
                        atomBK,
                        cute.make_tensor(ckv_g + sel.to(Int64) * DK_CK, rowK),
                        cute.make_tensor(
                            sKc[1].iterator + (warp_id * RPW + rr) * SKC_STRIDE,
                            rowK,
                        ),
                        mbar_ptr=sBar.iterator + 1,
                    )
                    cute.copy(
                        atomBP,
                        cute.make_tensor(kpe_g + sel.to(Int64) * DK_PE, rowP),
                        cute.make_tensor(
                            sKp[1].iterator + (warp_id * RPW + rr) * SKP_STRIDE,
                            rowP,
                        ),
                        mbar_ptr=sBar.iterator + 1,
                    )
        # 2d) afy bulk copy: arm rode the init block (branchless tx). The copy
        #     uses the gather pattern: warp-scoped region + per-thread uniform
        #     predicate (the DSL elects one lane: exactly one bulk op).
        if cutlass.const_expr(SPARSE and not self.direct_afy):
            if NSPLIT > 1:
                if warp_id == 4:
                    dok = Int32(0) + ((split | (Int32(0) - split)) >> 31) + 1
                    if dok != Int32(0):
                        # strided mapping: split 0's siblings are NTILES ranges,
                        # each tile-block minus split 0's own leading tile.
                        for jb in cutlass.range_constexpr(NTILES):
                            cute.copy(
                                atomAfB,
                                cute.make_tensor(
                                    cute.make_ptr(
                                        Int32,
                                        (mIdx.iterator + (token.to(Int64) * TOPK + jb * NSPLIT * TILE_N + TILE_N)).toint(),
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((max(SIBB, 4),)),
                                ),
                                cute.make_tensor(
                                    sAfIdx.iterator + jb * SIBB,
                                    cute.make_layout((max(SIBB, 4),)),
                                ),
                                mbar_ptr=sBar.iterator + 4,
                            )
        # 2b) see (2): single-tile configs issue Q after the stage-0 burst.
        if not Q_EARLY:
            qlate_local = Int32(1)
            if cutlass.const_expr(self.q_mcast):
                qlate_local = Int32(0)
                if split % Int32(self.cluster) == qproducer_rank:
                    if cutlass.const_expr(self.q_rect16):
                        if tidx == Int32(0):
                            _issue_tma_rect_mcast(
                                rect_k_atom,
                                sQn.iterator,
                                sBar.iterator + 2,
                                Int32(0),
                                token * Int32(HEADS),
                                Int16(-1),
                            )
                            _issue_tma_rect_mcast(
                                rect_p_atom,
                                sQp.iterator,
                                sBar.iterator + 2,
                                Int32(0),
                                token * Int32(HEADS),
                                Int16(-1),
                            )
            if qlate_local != Int32(0):
                if warp_id >= 6:
                    for h2 in cutlass.range_constexpr(HEADS // 2):
                        h = (warp_id - 6) * (HEADS // 2) + h2
                        cute.copy(
                            atomBK,
                            cute.make_tensor(
                                qn_g
                                + (token * HEADS + h).to(Int64) * DK_CK,
                                rowK,
                            ),
                            cute.make_tensor(
                                sQn.iterator + h * SKC_STRIDE, rowK
                            ),
                            mbar_ptr=sBar.iterator + 2,
                        )
                        cute.copy(
                            atomBP,
                            cute.make_tensor(
                                qpe_g
                                + (token * HEADS + h).to(Int64) * DK_PE,
                                rowP,
                            ),
                            cute.make_tensor(
                                sQp.iterator + h * SKP_STRIDE, rowP
                            ),
                            mbar_ptr=sBar.iterator + 2,
                        )

        # 2c) Zero dead K/V rows while the gathers + index staging fly. The
        #    per-warp index registers (loaded at kernel entry) self-predicate
        #    the stores, so the slab needs no smem wait or barrier and lands
        #    inside the (otherwise idle) staging-LDG window instead of the old
        #    post-scan position, where its smem-store volume sat serially on
        #    the busy split's pre-loop chain (~0.35-0.45us for sparse tiles).
        #    Stores are row-disjoint from the in-flight TMA writes (valid
        #    rows only); the staging barrier below covers loop visibility.
        if SPARSE:
            zf = cute.make_rmem_tensor((V8,), BFloat16)
            zf.fill(BFloat16(0.0))
            NZP = (DK_CK // V8 + 31) // 32  # 64 data pieces over 32 lanes
            for rr in cutlass.range_constexpr(RPW):
                if t0r[rr] < 0:
                    for pc in cutlass.range_constexpr(NZP):
                        cute.copy(
                            atom16univ,
                            zf,
                            sKc_v[0][
                                (warp_id * RPW + rr) * (SKC_STRIDE // V8) + pc * 32 + lane_id,
                                None,
                            ],
                        )
                if cutlass.const_expr(NTILES > 1):
                    if t1r[rr] < 0:
                        for pc in cutlass.range_constexpr(NZP):
                            cute.copy(
                                atom16univ,
                                zf,
                                sKc_v[1][
                                    (warp_id * RPW + rr) * (SKC_STRIDE // V8) + pc * 32 + lane_id,
                                    None,
                                ],
                            )


        if cutlass.const_expr(IKET_TRACE):
            cute.experimental.iket.range_pop()
            cute.experimental.iket.range_push("pro_stage")
        # 4) Full index staging gmem->smem (drives refill addressing, per-row
        #    validity masks and the tile scan) + sAny flag, as quads (ROWS
        #    % 64 == 0 and the token row is 8KB-aligned, so 128b vectors are
        #    exact; sAny re-reads the just-stored smem). Overlaps the
        #    in-flight gathers.
        NQ4 = ROWS // 4
        Q4PT = TILE_N // 4  # quads per tile
        for i in cutlass.range_constexpr((NQ4 + NTHREADS - 1) // NTHREADS):
            q4 = i * NTHREADS + tidx
            if q4 < NQ4:
                gslot = (split + (q4 // Q4PT) * NSPLIT) * TILE_N + (q4 % Q4PT) * 4
                srcq = cute.make_tensor(
                    cute.make_ptr(
                        Int32,
                        (mIdx.iterator + (token.to(Int64) * TOPK + gslot)).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4,)),
                )
                dstq = cute.make_tensor(
                    cute.make_ptr(
                        Int32,
                        (sIdx.iterator + q4 * 4).toint(),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4,)),
                )
                cute.copy(atom16univ, srcq, dstq)
                for e in cutlass.range_constexpr(4):
                    if sIdx[q4 * 4 + e] >= 0:
                        sAny[0] = Int32(1)
        if cutlass.const_expr(IKET_TRACE):
            cute.experimental.iket.range_pop()
            cute.experimental.iket.range_push("pro_scan")
        cute.arch.barrier()  # sIdx visible
        # per-warp valid-row counts per tile (2 entries per lane, bfly reduce);
        # branchless arithmetic only (DSL staging rule)
        for tw in cutlass.range_constexpr((NTILES + (NTHREADS // 32) - 1) // (NTHREADS // 32)):
            tile = tw * (NTHREADS // 32) + warp_id
            if tile < NTILES:
                r0i = tile * TILE_N + lane_id
                r1i = r0i + 32
                v0 = sIdx[r0i]
                v1 = sIdx[r1i]
                cc = ((v0 >> 31) + 1) + ((v1 >> 31) + 1)
                # last-valid position via max-reduce (-1 when none),
                # branchless arithmetic only (DSL staging rule)
                m0 = (r0i + 1) * ((v0 >> 31) + 1) - 1
                m1 = (r1i + 1) * ((v1 >> 31) + 1) - 1
                ll = m0 + (m1 - m0) * (0 - ((m0 - m1) >> 31))  # max(m0, m1)
                cc = cc + cute.arch.shuffle_sync_bfly(cc, 16)
                cc = cc + cute.arch.shuffle_sync_bfly(cc, 8)
                cc = cc + cute.arch.shuffle_sync_bfly(cc, 4)
                cc = cc + cute.arch.shuffle_sync_bfly(cc, 2)
                cc = cc + cute.arch.shuffle_sync_bfly(cc, 1)
                for off in (16, 8, 4, 2, 1):
                    lo = cute.arch.shuffle_sync_bfly(ll, off)
                    ll = ll + (lo - ll) * (0 - ((ll - lo) >> 31))  # ll = max(ll, lo)
                if lane_id == 0:
                    sTileCnt[tile] = cc
                    # prefix-packed: count == last valid position + 1
                    dp = ll + 1 - cc
                    sPack[tile] = ((dp | (0 - dp)) >> 31) + 1  # 1 iff dp == 0
        cute.arch.barrier()  # counts visible
        if SPARSE:
            # variant A: tidx0 exact-byte arms off the scan counts
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    sBar.iterator + 0, sTileCnt[0] * Int32((DK_CK + DK_PE) * 2)
                )
                if NTILES > 1:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        sBar.iterator + 1, sTileCnt[1] * Int32((DK_CK + DK_PE) * 2)
                    )
            cute.arch.barrier()  # arms visible to issuers
        if cutlass.const_expr(IKET_TRACE):
            cute.experimental.iket.range_pop()
            cute.experimental.iket.range_push("pro_cwait")
        if cutlass.const_expr(self.cluster):
            if cutlass.const_expr(self.q_mcast):
                if split % Int32(self.cluster) != qproducer_rank:
                    cute.arch.cluster_wait()
            else:
                cute.arch.cluster_wait()  # all peers' mbars now init'd
            # empty splits publish l=0 to every consumer's landing slot
            # immediately after their scan (~2us), long before busy splits
            # post their partials.
            if NSPLIT > 1:
                if sAny[0] == Int32(0):
                    if cutlass.const_expr(self.q_mcast):
                        # Keep this multicast destination and its barQ alive
                        # until the producer transaction has completed.
                        cute.arch.mbarrier_wait(
                            sBar.iterator + 2, Int32(0)
                        )
                    if tidx < self.cluster:
                        dbar = cute.arch.map_dsmem_ptr(sBar.iterator + 3, tidx)
                        # (l, m) pair slots: empty split ships only l=0 (4B);
                        # its m word is never read (l==0 gates the merge).
                        dl = cute.arch.map_dsmem_ptr(
                            sLslot.iterator + (split % self.cluster) * 2, tidx
                        )
                        cute.arch.mbarrier_arrive_and_expect_tx(dbar, Int32(4))
                        rL[0] = Float32(0.0)
                        cute.copy(
                            atomDS4,
                            rL,
                            cute.make_tensor(dl, cute.make_layout((1,))),
                            mbar_ptr=dbar,
                        )
        # (dead-row zero fill now lives at (2c), in the staging window)

        if cutlass.const_expr(IKET_TRACE):
            cute.experimental.iket.range_pop()
        # ---- MMA / copy partitioning ----
        # PV A=P kept kv-major in smem: A-frags come from transposed ldsm and
        # the softmax scatter packs (head, head+1) pairs into one 32b store.
        thr_mma = tiled_mma.get_slice(tidx)
        tcA = cute.make_tiled_copy_A(ldsm_Bt, tiled_mma)
        thrA = tcA.get_slice(tidx)
        tcBt = cute.make_tiled_copy_B(ldsm_Bt, tiled_mma)
        thrBt = tcBt.get_slice(tidx)

        # transposed QK path on tiled_mma_t (4,1,2): S^T = K @ Q^T.
        # A = gathered K/V rows (each byte ldsm'd once); B = Q (dup x4 vs x8).
        thr_mma_t = tiled_mma_t.get_slice(tidx)
        tcAt = cute.make_tiled_copy_A(ldsm_A, tiled_mma_t)
        thrAt = tcAt.get_slice(tidx)
        tcBtt = cute.make_tiled_copy_B(ldsm_A, tiled_mma_t)
        thrBtt = tcBtt.get_slice(tidx)

        sQc = sQn
        sQp = sQp
        sVt = [
            cute.make_tensor(
                sKc[st].iterator,
                cute.make_layout((DK_CK, TILE_N), stride=(1, SKC_STRIDE)),
            )
            for st in range(len(sKc))
        ]
        sPt = cute.make_tensor(
            sP.iterator, cute.make_layout((HEADS, TILE_N), stride=(1, HEADS + V8))
        )
        sPk32 = cute.make_tensor(
            cute.recast_ptr(sP.iterator, dtype=Int32),
            cute.make_layout((TILE_N, (HEADS + V8) // 2), stride=((HEADS + V8) // 2, 1)),
        )
        sPt2 = cute.make_tensor(
            sP2.iterator, cute.make_layout((HEADS, TILE_N), stride=(1, HEADS + V8))
        )
        sPk32_2 = cute.make_tensor(
            cute.recast_ptr(sP2.iterator, dtype=Int32),
            cute.make_layout((TILE_N, (HEADS + V8) // 2), stride=((HEADS + V8) // 2, 1)),
        )

        tCsP = thrA.partition_S(sPt)
        tCsP2 = thrA.partition_S(sPt2)
        tCsVt = [thrBt.partition_S(sVt[st]) for st in range(len(sVt))]

        tCsKcT = [thrAt.partition_S(sKc[st]) for st in range(len(sKc))]
        tCsKpT = [thrAt.partition_S(sKp[st]) for st in range(len(sKp))]
        tCsQcT = thrBtt.partition_S(sQc)
        tCsQpT = thrBtt.partition_S(sQp)

        # single-k-step register fragments (double-buffered)
        frAK0 = thr_mma_t.make_fragment_A(tCsKcT[0][None, None, 0])
        frAK1 = thr_mma_t.make_fragment_A(tCsKcT[0][None, None, 0])
        # Q B-fragments for ALL k-steps live in registers across the whole
        # CTA lifetime (Q is invariant) - loaded once after the tile-0 wait.
        frBQall = thr_mma_t.make_fragment_B(tCsQcT)
        frBQp = thr_mma_t.make_fragment_B(tCsQpT)
        frAKp0 = thr_mma_t.make_fragment_A(tCsKpT[0][None, None, 0])
        frP0 = thr_mma.make_fragment_A(tCsP[None, None, 0])
        frP1 = thr_mma.make_fragment_A(tCsP[None, None, 0])
        frP0b = thr_mma.make_fragment_A(tCsP[None, None, 0])
        frP1b = thr_mma.make_fragment_A(tCsP[None, None, 0])
        frVt0 = thr_mma.make_fragment_B(tCsVt[0][None, None, 0])
        frVt1 = thr_mma.make_fragment_B(tCsVt[0][None, None, 0])

        acc_shape_ST = thr_mma_t.partition_shape_C((TILE_N, HEADS))
        acc_ST = thr_mma_t.make_fragment_C(acc_shape_ST)
        acc_shape_O = thr_mma.partition_shape_C((HEADS, DV))
        acc_O = thr_mma.make_fragment_C(acc_shape_O)
        acc_O.fill(Float32(0.0))
        accv = cute.make_tensor(
            acc_ST.iterator, cute.make_layout((4, cute.size(acc_ST) // 4), stride=(1, 4))
        )
        rO_bf = cute.make_fragment_like(acc_O, BFloat16)
        rO32 = cute.recast_tensor(rO_bf, Int32)
        rO64 = cute.recast_tensor(acc_O, Int64)

        cST = cute.make_identity_tensor((TILE_N, HEADS))
        tCcST = thr_mma_t.partition_C(cST)
        cO = cute.make_identity_tensor((HEADS, DV))
        tCcO = thr_mma.partition_C(cO)

        p32 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=32
        )
        rPT = cute.make_fragment_like(acc_ST, BFloat16)
        rPT32 = cute.recast_tensor(rPT, Int32)
        rPT2 = cute.make_fragment_like(acc_ST, BFloat16)
        rPT32_2 = cute.recast_tensor(rPT2, Int32)

        l_e0 = Float32(0.0)
        l_o0 = Float32(0.0)
        l_e1 = Float32(0.0)
        l_o1 = Float32(0.0)
        lF_e0 = Float32(0.0)
        lF_o0 = Float32(0.0)
        lF_e1 = Float32(0.0)
        lF_o1 = Float32(0.0)
        # per-head running softmax max, tracked redundantly by every k0 lane
        # for its 4 head columns {2c, 2c+1, 2c+8, 2c+9} (c = lane % 4)
        m_e0 = Float32(NEG_BIG)
        m_o0 = Float32(NEG_BIG)
        m_e1 = Float32(NEG_BIG)
        m_o1 = Float32(NEG_BIG)

        # ============================ MAIN LOOP ==============================
        # split 0 solo-probe: finish the sibling-slot check NOW (its prefetch
        # has been in flight since the prologue, so the consume stalls at most
        # as long as the tile-0 wait right after) and, when no sibling has
        # work, release FULL+DIRECT in the shadow of the tile gathers (~1.5us
        # in, vs ~3.4us mid-epilogue). Mergers see DIRECT ~2us earlier and
        # exit, so the token span drops to split 0's own end. The epilogue
        # re-derives the same decision (and re-releases the identical word);
        # kernel completion still orders the output through split 0's exit.
        if SPARSE:
            if NSPLIT > 1:
                if split == 0:
                    if warp_id < 4:
                        if cutlass.const_expr(not self.direct_afy):
                            cute.arch.mbarrier_wait(sBar.iterator + 4, Int32(0))
                            for i in cutlass.range_constexpr(AFY_IT):
                                q4 = i * AFY_W + tidx
                                dx4 = q4 - Int32(AFY_NQ - 1)
                                qq4 = Int32(AFY_NQ - 1) + (dx4 & (dx4 >> 31))
                                srcq = cute.make_tensor(
                                    cute.make_ptr(
                                        Int32,
                                        (sAfIdx.iterator + qq4 * 4).toint(),
                                        cute.AddressSpace.smem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((4,)),
                                )
                                cute.copy(
                                    atomAfy,
                                    srcq,
                                    cute.make_tensor(
                                        afy.iterator + i * 4,
                                        cute.make_layout((4,)),
                                    ),
                                )
                        pv0 = Int32(0)
                        for i in cutlass.range_constexpr(max(max(AFY_IT, 1) * 4, 1)):
                            pv0 = pv0 | ((afy[i] >> 31) + 1)
                        if cutlass.const_expr(self.pair_scan_marker):
                            # afy iteration 1 covers sibling quads 128..255;
                            # lanes 112..127 begin at global slot 1024.
                            # Iterations 2/3 cover the rest of the upper half.
                            pvh = Int32(0)
                            if tidx >= Int32(112):
                                for j in cutlass.range_constexpr(4):
                                    pvh = pvh | ((afy[4 + j] >> 31) + 1)
                            for j in cutlass.range_constexpr(8, 16):
                                pvh = pvh | ((afy[j] >> 31) + 1)
                            if pvh != Int32(0):
                                sLslot[NSPLIT] = Float32(1.0)
                        if pv0 != Int32(0):
                            sAf[0] = Int32(1)
                        cute.arch.barrier(
                            barrier_id=1, number_of_threads=NTHREADS // 2
                        )  # named: warps 0-3 only
                    if tidx == 0:
                        # publish the epilogue's direct decision now: sAf/sAny
                        # are final. Mainloop barriers (nonempty split 0 always
                        # runs the body) order the store before epilogue reads.
                        if sAf[0] == Int32(0):
                            if sAny[0] != Int32(0):
                                sDirect[0] = Int32(1)
                        if cutlass.const_expr(not self.cluster):
                            if sAf[0] == Int32(0):
                                if sAny[0] != Int32(0):
                                    cute.arch.atomic_exch(
                                        flags_base + bx,
                                        (Int32(launch_id) & Int32(LID_MASK))
                                        + Int32(FULL_BIT + DIRECT_BIT),
                                        sem="release",
                                        scope="gpu",
                                    )
                    # cluster: split 0 publishes l=-1 (direct marker) to every
                    # consumer in the shadow of the tile gathers, releasing the
                    # merge consumers ~2us in instead of gating them on a gmem
                    # flag round trip.
                    if cutlass.const_expr(self.cluster):
                        if tidx < self.cluster:
                            if sAf[0] == Int32(0):
                                if sAny[0] != Int32(0):
                                    dbar = cute.arch.map_dsmem_ptr(sBar.iterator + 3, tidx)
                                    dl = cute.arch.map_dsmem_ptr(
                                        sLslot.iterator + (split % self.cluster) * 2, tidx
                                    )
                                    cute.arch.mbarrier_arrive_and_expect_tx(
                                        dbar, Int32(4)
                                    )
                                    # Normal split-0 partials use their
                                    # sign for the empty-half marker. Keep
                                    # DIRECT disjoint from every real
                                    # max-subtracted l sum (<= TILE_N*NTILES).
                                    if cutlass.const_expr(self.pair_scan_marker):
                                        rL[0] = Float32(-1.0e30)
                                    else:
                                        rL[0] = Float32(-1.0)
                                    cute.copy(
                                        atomDS4,
                                        rL,
                                        cute.make_tensor(dl, cute.make_layout((1,))),
                                        mbar_ptr=dbar,
                                    )
        if not SPARSE:
            # DENSE: the afy prefetch rides in the shadow of the tile gathers
            # instead (sparse-only win), and the release stays in the epilogue.
            if split == 0:
                if NSPLIT > 1:
                    QSIB = SIBB // 4  # sibling quads per tile-block
                    for i in cutlass.range_constexpr(AFY_IT):
                        q4 = i * NTHREADS + tidx
                        dx4 = q4 - Int32(AFY_NQ - 1)
                        qq4 = Int32(AFY_NQ - 1) + (dx4 & (dx4 >> 31))
                        gsib = (qq4 // QSIB) * NSPLIT * TILE_N + TILE_N + (qq4 % QSIB) * 4
                        srcq = cute.make_tensor(
                            cute.make_ptr(
                                Int32,
                                (mIdx.iterator + (token.to(Int64) * TOPK + gsib)).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout((4,)),
                        )
                        cute.copy(
                            atomAfy,
                            srcq,
                            cute.make_tensor(afy.iterator + i * 4, cute.make_layout((4,))),
                        )

        # Uniform sAny guard: empty slices skip waits/MMA entirely. Depth-2
        # pipeline: tiles 0/1 were issued in the prologue; iteration it waits
        # tile it on stage it%2, consumes it, then arms+issues tile it+2 into
        # the freed stage (keeps one full gather in flight behind the MMA).
        # Fully unrolled for NTILES<=2 (cross-tile scheduling on the small
        # officials); a dynamic loop for NTILES>=4: 8x-unrolling the body
        # yields a ~100KB cubin that thrashes L1i and spills (~196 STL/LDL at
        # ns4). Stage views are rebuilt each iteration via pointer arithmetic
        # (stages are same-sized sequential SmemAllocator bumps), so a single
        # body text serves both loop forms.
        UNROLL_FULL = self.ntiles <= 2
        for it in cutlass.range(NTILES, unroll_full=UNROLL_FULL):
            if sAny[0] != Int32(0):
                st = it % 2
                # stage base pointers: re-wrapped at 16B so the downstream
                # ldsm/TMA/16B-vector alignment checks trace statically.
                kcb = cute.make_ptr(
                    BFloat16,
                    (sKc[0].iterator + st * Int32(TILE_N * SKC_STRIDE)).toint(),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                kpb = cute.make_ptr(
                    BFloat16,
                    (sKp[0].iterator + st * Int32(TILE_N * SKP_STRIDE)).toint(),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                tCsKcTs = thrAt.partition_S(
                    cute.make_tensor(
                        kcb, cute.make_layout((TILE_N, DK_CK), stride=(SKC_STRIDE, 1))
                    )
                )
                tCsKpTs = thrAt.partition_S(
                    cute.make_tensor(
                        kpb, cute.make_layout((TILE_N, DK_PE), stride=(SKP_STRIDE, 1))
                    )
                )
                tCsVts = thrBt.partition_S(
                    cute.make_tensor(
                        kcb, cute.make_layout((DK_CK, TILE_N), stride=(1, SKC_STRIDE))
                    )
                )
                # Q rides the separate barQ (filled pre-loop by warps 6/7).
                # Wait barQ + ldsm the B-frags BEFORE the stage wait: Q is
                # issued ahead of the gathers (Q_EARLY configs), so its frags
                # load inside the tile-0 arrival window instead of after it.
                # (Q-late configs: barQ completes last either way - the waits
                # must both pass before QK, so their order is cost-neutral.)
                if it == 0:
                    NKBQ = DK_CK // (2 * 16)
                    cute.arch.mbarrier_wait(sBar.iterator + 2, Int32(0))
                    for kb in cutlass.range_constexpr(NKBQ):
                        cute.copy(tcBtt, tCsQcT[None, None, kb], frBQall[None, None, kb])
                    for kb in cutlass.range_constexpr(DK_PE // (2 * 16)):
                        cute.copy(tcBtt, tCsQpT[None, None, kb], frBQp[None, None, kb])
                # wait tile it's TMA data. Unconditional: every stage is armed
                # once per phase (empty tiles complete on the arm alone).
                # Parity waits are sticky-complete.
                if cutlass.const_expr(IKET_TRACE):
                    cute.experimental.iket.range_push("tile_wait")
                cute.arch.mbarrier_wait(sBar.iterator + st, ((it >> 1) & 1))
                if cutlass.const_expr(IKET_TRACE):
                    cute.experimental.iket.range_pop()

                if sTileCnt[it] != Int32(0):
                    # k0 V-frags load now: independent of P/QK, retire under
                    # the QK+softmax window instead of after the P barrier.
                    cute.copy(tcBt, tCsVts[None, None, 0], frVt0)
                    # prefix-packed tiles: warps whose 8-row block starts at
                    # or beyond the valid count have all-dead S rows -> skip
                    # their QK/softmax/P-store; PV k-blocks fully beyond the
                    # count contribute exactly 0 -> skipped. Exact for any
                    # pattern: skips only fire when pkdv proves packing.
                    cntv = sTileCnt[it]
                    pkdv = sPack[it]
                    # ---- QK^T transposed: acc_ST = K @ Q^T [64 kv, 16 heads]
                    # on tiled mma (4,1,2): A = gathered rows (each ldsm'd
                    # once), B = Q (dup x4, k-split x2 pairwise-reduced).
                    # Warp's m-tile = kv rows 16m..16m+15; packed tiles skip
                    # fully-dead m-tiles (exact: skipped acc stays zero, dead
                    # rows validity-masked + zero-stored in P).
                    m_ = warp_id % 4
                    mk_ = warp_id // 4
                    skipm = 0 - ((cntv - m_ * 16 - 1) >> 31)  # 1 iff cnt <= 16*m
                    qk_on = Int32(1) - pkdv * skipm
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_push("tile_qk")
                    acc_ST.fill(Float32(0.0))
                    if qk_on != Int32(0):
                        cute.copy(tcAt, tCsKcTs[None, None, 0], frAK0)
                        NKC = DK_CK // (2 * 16)  # 16 k32-steps
                        for kb in cutlass.range_constexpr(NKC):
                            if kb + 1 < NKC:
                                if kb % 2 == 0:
                                    cute.copy(tcAt, tCsKcTs[None, None, kb + 1], frAK1)
                                else:
                                    cute.copy(tcAt, tCsKcTs[None, None, kb + 1], frAK0)
                            if kb % 2 == 0:
                                cute.gemm(tiled_mma_t, acc_ST, frAK0, frBQall[None, None, kb], acc_ST)
                            else:
                                cute.gemm(tiled_mma_t, acc_ST, frAK1, frBQall[None, None, kb], acc_ST)
                        NKP = DK_PE // (2 * 16)  # 2 k32-steps
                        cute.copy(tcAt, tCsKpTs[None, None, 0], frAKp0)
                        cute.gemm(tiled_mma_t, acc_ST, frAKp0, frBQp[None, None, 0], acc_ST)
                        if NKP == 2:
                            cute.copy(tcAt, tCsKpTs[None, None, 1], frAKp0)
                            cute.gemm(tiled_mma_t, acc_ST, frAKp0, frBQp[None, None, 1], acc_ST)

                    # pairwise k-partial reduce to the k=0 warps (thr layout
                    # m-fastest: partner warp +4 bears the other k-half);
                    # 128b vector transport (4 f32 per op).
                    # dead strips (qk_on==0) transport/read nothing: their acc
                    # is zero on BOTH k-halves and dead rows are validity-masked
                    # in softmax, so skipping moves no numerics. (smem then holds
                    # stale bytes for that strip - never read.)
                    if mk_ == 1:
                        if qk_on != Int32(0):
                            for i in cutlass.range_constexpr(cute.size(acc_ST) // 4):
                                cute.copy(
                                    atomF4,
                                    accv[None, i],
                                    sRedV[(warp_id % 4) * 128 + lane_id * 4 + i, None],
                                )
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_pop()
                        cute.experimental.iket.range_push("kred_bar")
                    cute.arch.barrier()
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_pop()
                    if mk_ == 0:
                        if qk_on != Int32(0):
                            for i in cutlass.range_constexpr(cute.size(acc_ST) // 4):
                                cute.copy(
                                    atomF4,
                                    sRedV[(warp_id % 4) * 128 + lane_id * 4 + i, None],
                                    rRedT[None, i],
                                )
                            for i in cutlass.range_constexpr(cute.size(acc_ST)):
                                acc_ST[i] = acc_ST[i] + rRedT[i]

                        # ---- stable softmax over heads per kv row ----
                        # Element i of acc_ST is head parity i&1, upper head
                        # half iff (i>>2)&1 (static; verified by
                        # dev/coord_probe.py). Phase 1: masked scores (fmin
                        # with a +/-BIG cap keeps the select rounding-free;
                        # invalid rows become NEG_BIG so their exp2 is exactly
                        # 0) and per-class tile maxes, butterfly-reduced over
                        # the 8 lanes sharing a head column, exchanged via
                        # sMx4 under a 128-thread named barrier.
                        if cutlass.const_expr(IKET_TRACE):
                            cute.experimental.iket.range_push("sm_p1")
                        tm_e0 = Float32(NEG_BIG)
                        tm_o0 = Float32(NEG_BIG)
                        tm_e1 = Float32(NEG_BIG)
                        tm_o1 = Float32(NEG_BIG)
                        rMK = cute.make_rmem_tensor((cute.size(acc_ST),), Float32)
                        for i2 in cutlass.range_constexpr(cute.size(acc_ST) // 2):
                            # frag pairs (i, i+1) = heads (2c, 2c+1) of one kv
                            # row: share the validity load and head-class.
                            i = 2 * i2
                            coord = tCcST[i]
                            ridx = sIdx[it * TILE_N + coord[0]]
                            validf = Float32((ridx >> 31) + 1)
                            cap = (validf + validf - Float32(1.0)) * Float32(-NEG_BIG)
                            mk0 = cute.arch.fmin(acc_ST[i] * scale_log2, cap)
                            mk1 = cute.arch.fmin(acc_ST[i + 1] * scale_log2, cap)
                            rMK[i] = mk0
                            rMK[i + 1] = mk1
                            if cutlass.const_expr(((i >> 2) & 1) == 0):
                                tm_e0 = cute.arch.fmax(tm_e0, mk0)
                                tm_o0 = cute.arch.fmax(tm_o0, mk1)
                            else:
                                tm_e1 = cute.arch.fmax(tm_e1, mk0)
                                tm_o1 = cute.arch.fmax(tm_o1, mk1)
                        for off in (4, 8, 16):
                            tm_e0 = cute.arch.fmax(
                                tm_e0, cute.arch.shuffle_sync_bfly(tm_e0, off)
                            )
                            tm_o0 = cute.arch.fmax(
                                tm_o0, cute.arch.shuffle_sync_bfly(tm_o0, off)
                            )
                            tm_e1 = cute.arch.fmax(
                                tm_e1, cute.arch.shuffle_sync_bfly(tm_e1, off)
                            )
                            tm_o1 = cute.arch.fmax(
                                tm_o1, cute.arch.shuffle_sync_bfly(tm_o1, off)
                            )
                        # head-major sMx4 [head][k0-warp]: phase-2 reads are 4
                        # 16B vector loads instead of 16 scalar LDS.
                        if lane_id < 4:
                            sMx4[(2 * lane_id) * 4 + warp_id] = tm_e0
                            sMx4[(2 * lane_id + 1) * 4 + warp_id] = tm_o0
                            sMx4[(2 * lane_id + 8) * 4 + warp_id] = tm_e1
                            sMx4[(2 * lane_id + 9) * 4 + warp_id] = tm_o1
                        if cutlass.const_expr(IKET_TRACE):
                            cute.experimental.iket.range_pop()
                            cute.experimental.iket.range_push("sm_bar2")
                        cute.arch.barrier(barrier_id=2, number_of_threads=128)
                        if cutlass.const_expr(IKET_TRACE):
                            cute.experimental.iket.range_pop()
                            cute.experimental.iket.range_push("sm_p23")
                        # Phase 2: cross-warp tile max, running-max update,
                        # accumulator rescale factors (all k0 lanes track the
                        # same values for their 4 head columns; warp 0
                        # publishes for the PV warps and the merge).
                        he = 2 * (lane_id % 4)
                        rT4 = cute.make_rmem_tensor((4, 4), Float32)
                        sMx4v = cute.make_tensor(
                            sMx4.iterator,
                            cute.make_layout((HEADS, 4), stride=(4, 1)),
                        )
                        cute.copy(atomF4, sMx4v[he, None], rT4[None, 0])
                        cute.copy(atomF4, sMx4v[he + 1, None], rT4[None, 1])
                        cute.copy(atomF4, sMx4v[he + 8, None], rT4[None, 2])
                        cute.copy(atomF4, sMx4v[he + 9, None], rT4[None, 3])
                        t_e0 = cute.arch.fmax(
                            cute.arch.fmax(rT4[0, 0], rT4[1, 0]),
                            cute.arch.fmax(rT4[2, 0], rT4[3, 0]),
                        )
                        t_o0 = cute.arch.fmax(
                            cute.arch.fmax(rT4[0, 1], rT4[1, 1]),
                            cute.arch.fmax(rT4[2, 1], rT4[3, 1]),
                        )
                        t_e1 = cute.arch.fmax(
                            cute.arch.fmax(rT4[0, 2], rT4[1, 2]),
                            cute.arch.fmax(rT4[2, 2], rT4[3, 2]),
                        )
                        t_o1 = cute.arch.fmax(
                            cute.arch.fmax(rT4[0, 3], rT4[1, 3]),
                            cute.arch.fmax(rT4[2, 3], rT4[3, 3]),
                        )
                        if cutlass.const_expr(NTILES > 1):
                            # running-max update + rescale factors: only a
                            # later tile can raise an existing max (folds away
                            # at unroll for the first tile and single-tile
                            # configs).
                            if it > 0:
                                t_e0 = cute.arch.fmax(m_e0, t_e0)
                                t_o0 = cute.arch.fmax(m_o0, t_o0)
                                t_e1 = cute.arch.fmax(m_e1, t_e1)
                                t_o1 = cute.arch.fmax(m_o1, t_o1)
                                f_e0 = cute.arch.exp2(m_e0 - t_e0)
                                f_o0 = cute.arch.exp2(m_o0 - t_o0)
                                f_e1 = cute.arch.exp2(m_e1 - t_e1)
                                f_o1 = cute.arch.exp2(m_o1 - t_o1)
                                l_e0 = l_e0 * f_e0
                                l_o0 = l_o0 * f_o0
                                l_e1 = l_e1 * f_e1
                                l_o1 = l_o1 * f_o1
                                if warp_id == 0:
                                    if lane_id < 4:
                                        sMxS[he] = f_e0
                                        sMxS[he + 1] = f_o0
                                        sMxS[he + 8] = f_e1
                                        sMxS[he + 9] = f_o1
                        m_e0 = t_e0
                        m_o0 = t_o0
                        m_e1 = t_e1
                        m_o1 = t_o1
                        if warp_id == 0:
                            if lane_id < 4:
                                sMxF[he] = m_e0
                                sMxF[he + 1] = m_o0
                                sMxF[he + 8] = m_e1
                                sMxF[he + 9] = m_o1
                        # Phase 3: p = exp2(masked - m) (masked rows land
                        # exactly 0, valid rows are <= 1: no overflow, no
                        # validity multiply); thread l partials split 4 ways
                        # by static head class.
                        for i2 in cutlass.range_constexpr(cute.size(acc_ST) // 2):
                            i = 2 * i2
                            if cutlass.const_expr(((i >> 2) & 1) == 0):
                                pm0 = cute.arch.exp2(rMK[i] - m_e0)
                                pm1 = cute.arch.exp2(rMK[i + 1] - m_o0)
                                l_e0 = l_e0 + pm0
                                l_o0 = l_o0 + pm1
                            else:
                                pm0 = cute.arch.exp2(rMK[i] - m_e1)
                                pm1 = cute.arch.exp2(rMK[i + 1] - m_o1)
                                l_e1 = l_e1 + pm0
                                l_o1 = l_o1 + pm1
                            # hi/lo split: p_hi + p_lo == pm to ~2^-17 rel
                            # (pm - f32(p_hi) is exact in fp32)
                            ph0 = BFloat16(pm0)
                            ph1 = BFloat16(pm1)
                            rPT[i] = ph0
                            rPT[i + 1] = ph1
                            rPT2[i] = BFloat16(pm0 - ph0.to(Float32))
                            rPT2[i + 1] = BFloat16(pm1 - ph1.to(Float32))

                        # ---- stage P (bf16, kv-major [64, 16+8]) to smem;
                        #      (head, head+1) pairs pack into 32b stores ----
                        for i2 in cutlass.range_constexpr(cute.size(rPT) // 2):
                            coord = tCcST[2 * i2]
                            sPk32[coord[0], coord[1] // 2] = rPT32[i2]
                            sPk32_2[coord[0], coord[1] // 2] = rPT32_2[i2]
                        if cutlass.const_expr(IKET_TRACE):
                            cute.experimental.iket.range_pop()
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_push("p_bar")
                    cute.arch.barrier()  # (p) P visible (and sMxS factors)
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_pop()
                        cute.experimental.iket.range_push("tile_pv")
                    # Online-softmax accumulator rescale: when this tile
                    # raised a head's max, all warps scale their acc_O rows
                    # by exp2(m_old - m_new). acc_O row = lane/4 + 8*((i>>1)&1)
                    # (verified by dev/coord_probe.py). Tile 0 needs none
                    # (acc_O is still zero).
                    if cutlass.const_expr(NTILES > 1):
                        if it > 0:
                            fr0 = sMxS[lane_id >> 2]
                            fr1 = sMxS[(lane_id >> 2) + 8]
                            for i in cutlass.range_constexpr(cute.size(acc_O)):
                                if cutlass.const_expr(((i >> 1) & 1) == 0):
                                    acc_O[i] = acc_O[i] * fr0
                                else:
                                    acc_O[i] = acc_O[i] * fr1
                    # ---- PV: acc_O [16, 512] += P @ V. Dead rows contribute
                    # exactly 0 (P zero-stored, V zero-filled); when packing is
                    # proven (pkdv) the k-steps fully past the valid count are
                    # skipped outright — exact (acc += 0 elided), and worth
                    # ~0.6us/tile on nearly-empty critical splits (NCU run_003:
                    # PV ran 778-807ns flat even with 2 valid rows).
                    KK = TILE_N // 16
                    # live k-steps: ceil(cntv/16) if packed else all
                    kstop = Int32(KK) - pkdv * (Int32(KK) - ((cntv + Int32(15)) >> 4))
                    cute.copy(tcA, tCsP[None, None, 0], frP0)
                    cute.copy(tcA, tCsP2[None, None, 0], frP0b)
                    if kstop >= Int32(KK):
                        # full tile: the original straight-line loop, verbatim
                        # (per-step guards broke the ldsm/HMMA pipelining and
                        # cost ~0.3us/tile on full tiles - measured ab_r3).
                        for kk in cutlass.range_constexpr(KK):
                            if kk + 1 < KK:
                                if kk % 2 == 0:
                                    cute.copy(tcA, tCsP[None, None, kk + 1], frP1)
                                    cute.copy(tcA, tCsP2[None, None, kk + 1], frP1b)
                                    cute.copy(tcBt, tCsVts[None, None, kk + 1], frVt1)
                                else:
                                    cute.copy(tcA, tCsP[None, None, kk + 1], frP0)
                                    cute.copy(tcA, tCsP2[None, None, kk + 1], frP0b)
                                    cute.copy(tcBt, tCsVts[None, None, kk + 1], frVt0)
                            if kk % 2 == 0:
                                cute.gemm(tiled_mma, acc_O, frP0, frVt0, acc_O)
                                cute.gemm(tiled_mma, acc_O, frP0b, frVt0, acc_O)
                            else:
                                cute.gemm(tiled_mma, acc_O, frP1, frVt1, acc_O)
                                cute.gemm(tiled_mma, acc_O, frP1b, frVt1, acc_O)
                    else:
                        # partial packed tile: run only the live k-steps
                        # (skipped steps add exactly 0 - P rows are zero)
                        for kk in cutlass.range_constexpr(KK):
                            if kk + 1 < KK:
                                if kstop > Int32(kk + 1):
                                    if kk % 2 == 0:
                                        cute.copy(tcA, tCsP[None, None, kk + 1], frP1)
                                        cute.copy(tcA, tCsP2[None, None, kk + 1], frP1b)
                                        cute.copy(tcBt, tCsVts[None, None, kk + 1], frVt1)
                                    else:
                                        cute.copy(tcA, tCsP[None, None, kk + 1], frP0)
                                        cute.copy(tcA, tCsP2[None, None, kk + 1], frP0b)
                                        cute.copy(tcBt, tCsVts[None, None, kk + 1], frVt0)
                            if kstop > Int32(kk):
                                if kk % 2 == 0:
                                    cute.gemm(tiled_mma, acc_O, frP0, frVt0, acc_O)
                                    cute.gemm(tiled_mma, acc_O, frP0b, frVt0, acc_O)
                                else:
                                    cute.gemm(tiled_mma, acc_O, frP1, frVt1, acc_O)
                                    cute.gemm(tiled_mma, acc_O, frP1b, frVt1, acc_O)

                if cutlass.const_expr(IKET_TRACE):
                    if sTileCnt[it] != Int32(0):
                        cute.experimental.iket.range_pop()
                # ---- refill: arm + issue tile it+2 into the just-freed stage.
                # Arm is unconditional (empty tiles complete on the arm alone);
                # the barrier separates PV's stage-st reads from the writers.
                if it + 2 < NTILES:
                    jt = it + 2
                    if tidx == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            sBar.iterator + st,
                            Int32(TILE_N * (DK_CK + DK_PE) * 2),
                        )
                    cute.arch.barrier()  # (c) st readers done; arm visible
                    # unconditional refill (constant bytes): invalid slots
                    # gather cache row 0, masked exactly downstream.
                    for rr in cutlass.range_constexpr(TILE_N // (NTHREADS // 32)):
                        row = warp_id * (TILE_N // (NTHREADS // 32)) + rr
                        ridx = sIdx[jt * TILE_N + row]
                        mneg = ridx >> 31  # arith: -1 (all-ones mask) iff invalid
                        # ridx if valid, else cache row 0 (always in bounds).
                        sel = ridx - (ridx & mneg)
                        cute.copy(
                            atomBK,
                            cute.make_tensor(ckv_g + sel.to(Int64) * DK_CK, rowK),
                            cute.make_tensor(
                                kcb + row * SKC_STRIDE, rowK
                            ),
                            mbar_ptr=sBar.iterator + st,
                        )
                        cute.copy(
                            atomBP,
                            cute.make_tensor(kpe_g + sel.to(Int64) * DK_PE, rowP),
                            cute.make_tensor(
                                kpb + row * SKP_STRIDE, rowP
                            ),
                            mbar_ptr=sBar.iterator + st,
                        )
                if not SPARSE:
                    if it == NTILES - 1:
                        # DENSE only (officials' unrolled-loop scheduling is
                        # fragile here - measured regressions at ns16/32):
                        # fold the softmax l partials into sL inside the last
                        # tile's window; the epilogue's barriers carry visi-
                        # bility, so the tail skips the shuffle chain + one
                        # barrier. Runs even with an empty last tile.
                        lF_e0 = l_e0 + cute.arch.shuffle_sync_bfly(l_e0, 4)
                        lF_e0 = lF_e0 + cute.arch.shuffle_sync_bfly(lF_e0, 8)
                        lF_e0 = lF_e0 + cute.arch.shuffle_sync_bfly(lF_e0, 16)
                        lF_o0 = l_o0 + cute.arch.shuffle_sync_bfly(l_o0, 4)
                        lF_o0 = lF_o0 + cute.arch.shuffle_sync_bfly(lF_o0, 8)
                        lF_o0 = lF_o0 + cute.arch.shuffle_sync_bfly(lF_o0, 16)
                        lF_e1 = l_e1 + cute.arch.shuffle_sync_bfly(l_e1, 4)
                        lF_e1 = lF_e1 + cute.arch.shuffle_sync_bfly(lF_e1, 8)
                        lF_e1 = lF_e1 + cute.arch.shuffle_sync_bfly(lF_e1, 16)
                        lF_o1 = l_o1 + cute.arch.shuffle_sync_bfly(l_o1, 4)
                        lF_o1 = lF_o1 + cute.arch.shuffle_sync_bfly(lF_o1, 8)
                        lF_o1 = lF_o1 + cute.arch.shuffle_sync_bfly(lF_o1, 16)
                        if lane_id < 4:
                            sL[warp_id * HEADS + 2 * lane_id] = lF_e0
                            sL[warp_id * HEADS + 2 * lane_id + 1] = lF_o0
                            sL[warp_id * HEADS + 2 * lane_id + 8] = lF_e1
                            sL[warp_id * HEADS + 2 * lane_id + 9] = lF_o1

        # ---- epilogue: split-0 direct-out fast path, else partial+merge ----
        # SPARSE: reduce per-thread l partials across lanes sharing a head
        # pair (acc_ST thread owns heads {2c,2c+1,2c+8,2c+9}, c = lane % 4);
        # DENSE folded them in-loop at the last tile (measured +1-1.7% at
        # ns1, and officials regress on the in-loop placement).
        if SPARSE:
            l_e0 = l_e0 + cute.arch.shuffle_sync_bfly(l_e0, 4)
            l_e0 = l_e0 + cute.arch.shuffle_sync_bfly(l_e0, 8)
            l_e0 = l_e0 + cute.arch.shuffle_sync_bfly(l_e0, 16)
            l_o0 = l_o0 + cute.arch.shuffle_sync_bfly(l_o0, 4)
            l_o0 = l_o0 + cute.arch.shuffle_sync_bfly(l_o0, 8)
            l_o0 = l_o0 + cute.arch.shuffle_sync_bfly(l_o0, 16)
            l_e1 = l_e1 + cute.arch.shuffle_sync_bfly(l_e1, 4)
            l_e1 = l_e1 + cute.arch.shuffle_sync_bfly(l_e1, 8)
            l_e1 = l_e1 + cute.arch.shuffle_sync_bfly(l_e1, 16)
            l_o1 = l_o1 + cute.arch.shuffle_sync_bfly(l_o1, 4)
            l_o1 = l_o1 + cute.arch.shuffle_sync_bfly(l_o1, 8)
            l_o1 = l_o1 + cute.arch.shuffle_sync_bfly(l_o1, 16)
            lF_e0 = l_e0
            lF_o0 = l_o0
            lF_e1 = l_e1
            lF_o1 = l_o1
        nonempty = sAny[0] != Int32(0)
        msk = Int32(launch_id) & Int32(LID_MASK)
        hh = lane_id // 4

        # split 0 checks the prefetched sibling slots: if no other split has
        # any valid row it writes the normalized output directly.
        # zero + conditional one-write are kept on the SAME warp subset
        # (warps 0-3, the probe owners) so program order makes the one-write
        # win without an extra CTA barrier; all threads read after B0 below.
        if not SPARSE:
            if warp_id < 4:
                sDirect[0] = Int32(0)
        if split == 0:
            if nonempty:
                if SPARSE:
                    pass  # sDirect published by the pre-loop probe
                else:
                    pv = Int32(0)
                    for i in cutlass.range_constexpr(max(max(AFY_IT, 1) * 4, 1)):
                        pv = pv | ((afy[i] >> 31) + 1)
                    if pv != Int32(0):
                        sAf[0] = Int32(1)
                    cute.arch.barrier()
                    if sAf[0] == Int32(0):
                        sDirect[0] = Int32(1)
        if not SPARSE:
            cute.arch.barrier()  # sDirect visible everywhere (DENSE re-derive)

        if sDirect[0] != Int32(0):
            # SPARSE: the FULL+DIRECT word was already released pre-loop
            # (identical condition: split 0, nonempty, sAf==0); a duplicate
            # release here measured -0.55us on split 0's exit path via the
            # atomic's round trip. DENSE keeps the pre-store release here:
            # mergers skip the output on this path and only wait on the flag.
            # Kernel completion still orders the output stores through split
            # 0's own exit either way.
            if not SPARSE:
                if NSPLIT > 1:
                    if tidx == 0:
                        cute.arch.atomic_exch(
                            flags_base + bx,
                            msk + Int32(FULL_BIT + DIRECT_BIT),
                            sem="release",
                            scope="gpu",
                        )
            # solo split: normalize by the local l sums and write out
            if SPARSE:
                if lane_id < 4:
                    sL[warp_id * HEADS + 2 * lane_id] = lF_e0
                    sL[warp_id * HEADS + 2 * lane_id + 1] = lF_o0
                    sL[warp_id * HEADS + 2 * lane_id + 8] = lF_e1
                    sL[warp_id * HEADS + 2 * lane_id + 9] = lF_o1
                cute.arch.barrier()
            if tidx < HEADS:
                lsum = Float32(0.0)
                for w in cutlass.range_constexpr(8):
                    lsum = lsum + sL[w * HEADS + tidx]
                # split 0 holds the token's full (m, l) on this path
                mLse[token, tidx] = sMxF[tidx] + cute.math.log2(
                    lsum, fastmath=True
                )
                if lsum > Float32(1e-30):
                    sRed[tidx] = Float32(1.0) / lsum
                else:
                    sRed[tidx] = Float32(0.0)
            cute.arch.barrier()  # sKc stages dead post-loop; staging safe
            gOVe = cute.make_tensor(
                cute.make_ptr(
                    BFloat16,
                    mOut.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((num_tokens * NGRP, V8), stride=(V8, 1)),
            )
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                rO_bf[i] = BFloat16(acc_O[i] * sRed[tCcO[i][0]])
            for j in cutlass.range_constexpr(cute.size(rO32)):
                coord = tCcO[2 * j]
                sOb32[coord[0], coord[1] // 2] = rO32[j]
            cute.arch.barrier()  # staged plane visible to all
            for v in cutlass.range_constexpr(NGRP // NTHREADS):
                p = v * NTHREADS + tidx
                sp_ = (p >> 6) * (SKC_STRIDE // V8) + (p & 63)
                cute.copy(atom16univ, sKc_v[0][sp_, None], gOVe[token * NGRP + p, None])
            cute.arch.barrier()  # all output stores issued before exit
            if cutlass.const_expr(self.cluster):
                if NSPLIT > 1:
                    # sibling markers target this CTA's own merge bar: hold
                    # smem until it completes (long done in practice) so no
                    # st.async lands in a dead CTA's memory.
                    cute.arch.mbarrier_wait(sBar.iterator + 3, Int32(0))
        else:
            # per-split partial outputs (fp32 end to end - the only bf16
            # rounding of merged tokens is the final normalized store), then
            # in-kernel merge by last split
            if cutlass.const_expr(self.cluster):
                if nonempty:
                    if cutlass.const_expr(IKET_TRACE):
                        cute.experimental.iket.range_push("epi_stage")
                    cute.arch.barrier()  # sKc stages dead post-loop; staging safe
                    for j in cutlass.range_constexpr(cute.size(rO64)):
                        coord = tCcO[2 * j]
                        sOf64[coord[0], coord[1] // 2] = rO64[j]
                    if SPARSE:
                        if lane_id < 4:
                            sL[warp_id * HEADS + 2 * lane_id] = lF_e0
                            sL[warp_id * HEADS + 2 * lane_id + 1] = lF_o0
                            sL[warp_id * HEADS + 2 * lane_id + 8] = lF_e1
                            sL[warp_id * HEADS + 2 * lane_id + 9] = lF_o1
                    cute.arch.barrier()  # plane + sL visible
                    lsumt = Float32(0.0)
                    if tidx < HEADS:
                        for w in cutlass.range_constexpr(8):
                            lsumt = lsumt + sL[w * HEADS + tidx]
                    if NSPLIT > 1:
                        # push: thread group (tidx%CSZ == c) ships this split's
                        # head row for consumer c into c's landing slots; the
                        # (l, m) pair rides the same bar. CSZ=16: one head row
                        # (64 16B pieces, 4/thread).
                        CSZ = self.cluster
                        cmer = tidx % CSZ
                        grp = tidx // CSZ
                        dbar = cute.arch.map_dsmem_ptr(sBar.iterator + 3, cmer)
                        dl = cute.arch.map_dsmem_ptr(
                            cute.make_ptr(
                                Float32,
                                (sLslot.iterator + (split % self.cluster) * 2).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                            ),
                            cmer,
                        )
                        # bulk-S2S push: ONE 2048B cp.async.bulk per
                        # consumer (this split's fp32 row for head cmer)
                        # replaces 128 LDS+st.async 16B pieces. Staged rows
                        # were written through the generic proxy: fence the
                        # async proxy (post-barrier) before the bulk reads.
                        # The op is warp-collective (DSL elects one lane) so
                        # operands must be warp-uniform: warp w ships head
                        # rows w and w+8. The DSL maps the DST by cta_rank
                        # but NOT the mbar operand: pre-map the consumer's
                        # bar and re-wrap the raw cluster address as smem.
                        if cutlass.const_expr(IKET_TRACE):
                            cute.experimental.iket.range_pop()
                            cute.experimental.iket.range_push("epi_push")
                        cute.arch.fence_view_async_shared()
                        if tidx < CSZ:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                dbar, Int32(DV * 4 + 8)
                            )
                            rL2[0] = lsumt
                            rL2[1] = sMxF[tidx]
                            if cutlass.const_expr(self.pair_scan_marker):
                                if split == Int32(0):
                                    if sLslot[NSPLIT] == Float32(0.0):
                                        rL2[0] = -lsumt
                            cute.copy(
                                atomDS8,
                                rL2,
                                cute.make_tensor(dl, cute.make_layout((2,))),
                                mbar_ptr=dbar,
                            )
                        dstp = cute.make_ptr(
                            Float32,
                            (sSlot.iterator + (split % self.cluster) * DV).toint(),
                            cute.AddressSpace.smem,
                            assumed_align=16,
                        )
                        dstT = cute.make_tensor(dstp, cute.make_layout((DV,)))
                        for rep in cutlass.range_constexpr(2):
                            cmer_u = warp_id + Int32(rep * 8)
                            bar_u = cute.make_ptr(
                                Int64,
                                cute.arch.map_dsmem_ptr(
                                    sBar.iterator + 3, cmer_u
                                ).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                            )
                            srcp = cute.make_ptr(
                                Float32,
                                (sKc[0].iterator + cmer_u * Int32(2 * SKC_STRIDE)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            )
                            cute.copy(
                                atomBS2S,
                                cute.make_tensor(srcp, cute.make_layout((DV,))),
                                dstT,
                                mbar_ptr=bar_u,
                                cta_rank=cmer_u,
                            )
                        if cutlass.const_expr(IKET_TRACE):
                            cute.experimental.iket.range_pop()
            else:
                gPVe = cute.make_tensor(
                    mPartO.iterator,
                    cute.make_layout((mPartO.shape[0] * NGRPF, 4), stride=(4, 1)),
                )
                if nonempty:
                    cute.arch.barrier()  # sKc stages dead post-loop; staging safe
                    for j in cutlass.range_constexpr(cute.size(rO64)):
                        coord = tCcO[2 * j]
                        sOf64[coord[0], coord[1] // 2] = rO64[j]
                    cute.arch.barrier()  # staged plane visible to all
                    for v in cutlass.range_constexpr(NGRPF // NTHREADS):
                        p = v * NTHREADS + tidx
                        sp_ = (p >> 7) * SKC_F32P + (p & 127)
                        cute.copy(atomF4, sOf_v[sp_, None], gPVe[bx * NGRPF + p, None])
                if nonempty:
                    # cross-warp l pre-reduction: 16 f32 per split; the merger
                    # then reads NSPLIT*16 f32 instead of NSPLIT*128.
                    if SPARSE:
                        if lane_id < 4:
                            sL[warp_id * HEADS + 2 * lane_id] = lF_e0
                            sL[warp_id * HEADS + 2 * lane_id + 1] = lF_o0
                            sL[warp_id * HEADS + 2 * lane_id + 8] = lF_e1
                            sL[warp_id * HEADS + 2 * lane_id + 9] = lF_o1
                        cute.arch.barrier()  # (l) sL visible; all CTA stores done
                    if tidx < HEADS:
                        lsumt = Float32(0.0)
                        for w in cutlass.range_constexpr(8):
                            lsumt = lsumt + sL[w * HEADS + tidx]
                        mPartL[bx, tidx] = lsumt
                        mPartL[bx, HEADS + tidx] = sMxF[tidx]
                    cute.arch.barrier()  # partial l+m stores issued before release
                if NSPLIT > 1:
                    if tidx == 0:
                        cute.arch.atomic_exch(
                            flags_base + bx,
                            msk + Int32(FULL_BIT) * sAny[0],
                            sem="release",
                            scope="gpu",
                        )

        # Merge the per-split partials into the final output.
        # CLUSTER: every CTA merges its own head slice out of its landing
        # slots - no gmem flags, no partial round trips. A negative l slot
        # marks a DIRECT token (split 0 wrote out itself) and is skipped.
        # Otherwise column-split across the last M=min(NSPLIT,4) splits
        # (usually empty+idle): merger rank r covers output columns
        # [r*DV/M, (r+1)*DV/M).
        if cutlass.const_expr(self.cluster == 16):
            if NSPLIT > 1:
                if cutlass.const_expr(IKET_TRACE):
                    cute.experimental.iket.range_push("merge_wait")
                cute.arch.mbarrier_wait(sBar.iterator + 3, Int32(0))
                if cutlass.const_expr(IKET_TRACE):
                    cute.experimental.iket.range_pop()
                    cute.experimental.iket.range_push("merge")
                crank16 = split % self.cluster
                nflag = Float32(0.0)
                lsumc = Float32(0.0)
                pair_empty = Int32(0)
                # landing slots hold (l, m) pairs; first find the DIRECT
                # marker and the live-slot global max, then combine with
                # exp2(m_s - mmax) weights (flash-decoding merge).
                mmax = Float32(NEG_BIG)
                if cutlass.const_expr(self.pair_scan_marker):
                    lv0m = sLslot[0]
                    if lv0m < Float32(-1.0e20):
                        nflag = Float32(1.0)
                    else:
                        if lv0m < Float32(0.0):
                            pair_empty = Int32(1)
                        if lv0m != Float32(0.0):
                            mmax = cute.arch.fmax(mmax, sLslot[1])
                    for s in cutlass.range_constexpr(1, self.cluster):
                        lv = sLslot[2 * s]
                        nflag = cute.arch.fmax(nflag, -lv)
                        if lv > Float32(0.0):
                            mmax = cute.arch.fmax(mmax, sLslot[2 * s + 1])
                else:
                    for s in cutlass.range_constexpr(self.cluster):
                        lv = sLslot[2 * s]
                        nflag = cute.arch.fmax(nflag, -lv)
                        if lv > Float32(0.0):
                            mmax = cute.arch.fmax(mmax, sLslot[2 * s + 1])
                # pair16 hop plumbing (NSPLIT==32): predefined before the
                # dynamic split-side branch (DSL type-join rule)
                px0 = Float32(0.0)
                px1 = Float32(0.0)
                pl0 = Float32(0.0)
                pm1v = Float32(NEG_BIG)
                fx = token * NSPLIT + self.cluster + crank16
                if nflag == Float32(0.0):
                    # merge head = my cluster rank over nonempty local slots
                    accm0 = Float32(0.0)
                    accm1 = Float32(0.0)
                    for s in cutlass.range_constexpr(self.cluster):
                        lvs = sLslot[2 * s]
                        slot_live = lvs > Float32(0.0)
                        if cutlass.const_expr(self.pair_scan_marker and s == 0):
                            slot_live = lvs != Float32(0.0)
                        if slot_live:
                            fsc = cute.arch.exp2(sLslot[2 * s + 1] - mmax)
                            lsumc = lsumc + fsc * cute.arch.fmax(lvs, -lvs)
                            accm0 = accm0 + fsc * sSlot[s, 2 * tidx]
                            accm1 = accm1 + fsc * sSlot[s, 2 * tidx + 1]
                    gO32 = cute.make_tensor(
                        cute.make_ptr(
                            Int32,
                            mOut.iterator.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        ),
                        cute.make_layout((num_tokens * HEADS * (DV // 2),)),
                    )
                    rOut = cute.make_rmem_tensor((2,), BFloat16)
                    rOut32 = cute.recast_tensor(rOut, Int32)
                    if cutlass.const_expr(NSPLIT > 16):
                        # PAIR16 (one 16-CTA cluster per half-token, NSPLIT==32):
                        # half 1 ships its L1-merged head row (unnormalized
                        # bf16) + l with a launch-id flag; half 0 polls that
                        # one flag, combines, normalizes and writes all 16
                        # output head rows. Officials' busy splits all sit in
                        # half 0, so the hop is an already-posted empty flag.
                        if split >= self.cluster:
                            # L1 partial producer (half 1) - fp32 partial row
                            gPf16 = cute.make_tensor(
                                mPartO.iterator,
                                cute.make_layout((mPartO.shape[0] * HEADS * DV,)),
                            )
                            if lsumc > Float32(1e-30):
                                gPf16[bx * HEADS * DV + crank16 * DV + 2 * tidx] = accm0
                                gPf16[bx * HEADS * DV + crank16 * DV + 2 * tidx + 1] = accm1
                                if tidx == 0:
                                    mPartL[bx, crank16] = lsumc
                                    mPartL[bx, HEADS + crank16] = mmax
                                    cute.arch.atomic_exch(
                                        flags_base + bx,
                                        msk + Int32(FULL_BIT),
                                        sem="release",
                                        scope="gpu",
                                    )
                            else:
                                if tidx == 0:
                                    # empty post: no data behind the flag (the
                                    # final skips mPartO/mPartL when FULL is
                                    # unset) -> relaxed; the release variant's
                                    # MEMBAR.ALL.GPU cost ~0.57us/CTA avg on
                                    # empty halves (NCU run_003).
                                    cute.arch.atomic_exch(
                                        flags_base + bx,
                                        msk,
                                        sem="relaxed",
                                        scope="gpu",
                                    )
                        else:
                            # L2 final (half 0): one polling thread, smem bcast
                            gPf16r = cute.make_tensor(
                                mPartO.iterator,
                                cute.make_layout((mPartO.shape[0] * HEADS * DV,)),
                            )
                            poll_peer = Int32(1)
                            if cutlass.const_expr(self.pair_scan_marker):
                                poll_peer = Int32(1) - pair_empty
                            if poll_peer != Int32(0):
                                if tidx == 0:
                                    fv = Int32(0)
                                    seen = Int32(0)
                                    while seen == 0:
                                        fv = cute.arch.load(
                                            flags_base + fx,
                                            Int32,
                                            sem="acquire",
                                            scope="gpu",
                                        )
                                        d = (fv & Int32(LID_MASK)) - msk
                                        seen = ((d | (0 - d)) >> 31) + 1
                                    sFull[0] = (fv >> 30) & 1
                                    if sFull[0] != Int32(0):
                                        sRed[0] = mPartL[fx, crank16]
                                        sRed[1] = mPartL[fx, HEADS + crank16]
                                cute.arch.barrier()  # flag bit + peer l+m visible
                                if sFull[0] != Int32(0):
                                    px0 = gPf16r[fx * HEADS * DV + crank16 * DV + 2 * tidx]
                                    px1 = gPf16r[fx * HEADS * DV + crank16 * DV + 2 * tidx + 1]
                                    pl0 = sRed[0]
                                    pm1v = sRed[1]
                            # combine the two halves at their pair max
                            mtot = cute.arch.fmax(mmax, pm1v)
                            f0c = cute.arch.exp2(mmax - mtot)
                            f1c = cute.arch.exp2(pm1v - mtot)
                            accm0 = accm0 * f0c + px0 * f1c
                            accm1 = accm1 * f0c + px1 * f1c
                            lsumc = lsumc * f0c + pl0 * f1c
                            invc = Float32(0.0)
                            if lsumc > Float32(1e-30):
                                invc = Float32(1.0) / lsumc
                            if tidx == Int32(0):
                                mLse[token, crank16] = mtot + cute.math.log2(
                                    lsumc, fastmath=True
                                )
                            rOut[0] = BFloat16(accm0 * invc)
                            rOut[1] = BFloat16(accm1 * invc)
                            gO32[token * HEADS * (DV // 2) + crank16 * (DV // 2) + tidx] = rOut32[0]
                    else:
                        invc = Float32(0.0)
                        if lsumc > Float32(1e-30):
                            invc = Float32(1.0) / lsumc
                        if tidx == Int32(0):
                            mLse[token, crank16] = mmax + cute.math.log2(
                                lsumc, fastmath=True
                            )
                        rOut[0] = BFloat16(accm0 * invc)
                        rOut[1] = BFloat16(accm1 * invc)
                        gO32[token * HEADS * (DV // 2) + crank16 * (DV // 2) + tidx] = rOut32[0]
        if cutlass.const_expr(self.cluster == 0):
            NM = self.nm
            if split >= NSPLIT - NM:
                mrank = split - (NSPLIT - NM)
                # Hoisted merge setup: every address and buffer below is a pure
                # function of (tidx, mrank) - independent of the flags - so it
                # computes BEFORE the flag poll instead of after it; the first
                # chunk load then issues directly off the poll-complete barrier.
                NGRPm = HEADS * DV // V8  # 1024 vector groups (64 per head)
                GPH = DV // V8  # 64 groups per head
                myG = NGRPm // NM  # groups in this merger's slice
                MB = NSPLIT if NSPLIT <= 16 else 16  # 16B load buffers
                NJ = myG // NTHREADS if myG % NTHREADS == 0 else 1
                bfrj = []
                accj = []
                gj = []
                for j in cutlass.range_constexpr(NJ):
                    jj = j * NTHREADS + tidx
                    acc = cute.make_rmem_tensor((V8,), Float32)
                    acc.fill(Float32(0.0))
                    bfr = [
                        [
                            cute.make_rmem_tensor((4,), Float32),
                            cute.make_rmem_tensor((4,), Float32),
                        ]
                        for _ in range(MB)
                    ]
                    h0 = jj // (GPH // NM)
                    c0 = mrank * (GPH // NM) + jj % (GPH // NM)
                    gj.append(h0 * GPH + c0)
                    bfrj.append(bfr)
                    accj.append(acc)
                # one thread per other split waits for its flag (acquire), records
                # DIRECT/FULL bits; own split's state computed locally
                if tidx < NSPLIT - 1:
                    sff = tidx if tidx < split else tidx + 1
                    fv = Int32(0)
                    seen = Int32(0)
                    while seen == 0:
                        # acquire load, not an RMW: cheaper poll and no
                        # atomic-unit serialization while busy splits run
                        fv = cute.arch.load(
                            flags_base + (token * NSPLIT + sff),
                            Int32,
                            sem="acquire",
                            scope="gpu",
                        )
                        d = (fv & Int32(LID_MASK)) - msk
                        seen = ((d | (0 - d)) >> 31) + 1  # 1 when launch id matches
                    sFull[sff] = (fv >> 29) & 3  # bit0 DIRECT, bit1 FULL
                sFull[split] = sDirect[0] + sAny[0] * 2
                cute.arch.barrier()
                # vector masks instead of per-split smem guard chains: one smem
                # word per lane + ballot puts every split's FULL/DIRECT bits in a
                # register of every thread, so the merge loads' guards predicate
                # off registers (the old sFull[sp] smem guard chain measured
                # ~0.2us per issue phase on merge officials).
                fv2 = sFull[lane_id & (NSPLIT - 1)]
                full_mask = cute.arch.vote_ballot_sync((fv2 & 2) != 0)
                dir_mask = cute.arch.vote_ballot_sync((fv2 & 1) != 0)
                if dir_mask == Int32(0):
                    # Merge partial O (fp32, 2x16B vectors per V8 out group)
                    # over full splits, restricted to this merger's column
                    # slice. All loads of an MB-split chunk (O groups AND
                    # per-split l) issue before any of their consumers, so
                    # each chunk rides ONE latency round instead of the old
                    # separate lsum phase followed by 4-deep rotating O loads
                    # (~NSPLIT/4 serialized rounds; the merger tail measured
                    # ~1.7-2.2us on merge-path officials).
                    gPV = cute.make_tensor(
                        mPartO.iterator,
                        cute.make_layout((mPartO.shape[0] * NGRPF, 4), stride=(4, 1)),
                    )
                    gOV = cute.make_tensor(
                        cute.make_ptr(
                            BFloat16,
                            mOut.iterator.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((num_tokens * NGRPm, V8), stride=(V8, 1)),
                    )
                    # Stable-softmax pre-phase: per-head global max over the
                    # full splits' shipped m, exp2(m_sp - M) factors into
                    # sFact, and the factor-weighted merged l -> sRed. The O
                    # chunks below then scale by sFact as they stream in.
                    if tidx < HEADS:
                        mgh = Float32(NEG_BIG)
                        rMs = cute.make_rmem_tensor((NSPLIT,), Float32)
                        for sp in cutlass.range_constexpr(NSPLIT):
                            rMs[sp] = Float32(NEG_BIG)
                            if ((full_mask >> sp) & 1) != 0:
                                rMs[sp] = mPartL[token * NSPLIT + sp, HEADS + tidx]
                            mgh = cute.arch.fmax(mgh, rMs[sp])
                        lsum = Float32(0.0)
                        for sp in cutlass.range_constexpr(NSPLIT):
                            fsp = cute.arch.exp2(rMs[sp] - mgh)
                            sFact[sp * HEADS + tidx] = fsp
                            if ((full_mask >> sp) & 1) != 0:
                                lsum = lsum + fsp * mPartL[token * NSPLIT + sp, tidx]
                        if lsum > Float32(1e-30):
                            sRed[tidx] = Float32(1.0) / lsum
                        else:
                            sRed[tidx] = Float32(0.0)
                        # all NM mergers derive the same (mgh, lsum); rank 0
                        # alone stores the token's lse row
                        if mrank == Int32(0):
                            mLse[token, tidx] = mgh + cute.math.log2(
                                lsum, fastmath=True
                            )
                    cute.arch.barrier()  # sFact + sRed visible
                    for j in cutlass.range_constexpr(NJ):
                        jj = j * NTHREADS + tidx
                        acc = accj[j]
                        bfr = bfrj[j]
                        g0 = gj[j]
                        if jj < myG:
                            h0j = jj // (GPH // NM)
                            for cc in cutlass.range_constexpr(NSPLIT // MB):
                                # skip wholly-empty split windows off register bits
                                cmask = (full_mask >> (cc * MB)) & ((1 << MB) - 1)
                                if cmask != Int32(0):
                                    for i in cutlass.range_constexpr(MB):
                                        sp = cc * MB + i
                                        if ((full_mask >> sp) & 1) != 0:
                                            cute.copy(
                                                atomF4,
                                                gPV[(token * NSPLIT + sp) * NGRPF + 2 * g0, None],
                                                bfr[i][0],
                                            )
                                            cute.copy(
                                                atomF4,
                                                gPV[(token * NSPLIT + sp) * NGRPF + 2 * g0 + 1, None],
                                                bfr[i][1],
                                            )
                                    for i in cutlass.range_constexpr(MB):
                                        sp = cc * MB + i
                                        if ((full_mask >> sp) & 1) != 0:
                                            fw = sFact[sp * HEADS + h0j]
                                            for e in cutlass.range_constexpr(V8):
                                                acc[e] = acc[e] + fw * bfr[i][e // 4][e % 4]
                    for j in cutlass.range_constexpr(NJ):
                        jj = j * NTHREADS + tidx
                        if jj < myG:
                            h0 = jj // (GPH // NM)
                            inv = sRed[h0]
                            for e in cutlass.range_constexpr(V8):
                                rPb[e] = BFloat16(accj[j][e] * inv)
                            cute.copy(atom16univ, rPb, gOV[token * NGRP + gj[j], None])


class _DSAT1(object):
    """Low-resource T=1 kernel with an exact two-row gate.

    The official two-row case assigns four compute warps per head plus two
    scan-assist warps, with no split/cluster workspace.  A complete scan proves
    that slots [2, 2048) are empty before entering that path.  Every other topology
    falls through to a deliberately simple but fully stable implementation,
    keeping the size dispatch correct for arbitrary live inputs and padding.
    """

    HEADS_PER_CTA = 1
    NTHREADS_T1 = 192

    def __init__(self):
        self.smem_bytes = self.HEADS_PER_CTA * TOPK * 4 + 32 * 4

    @cute.jit
    def __call__(
        self,
        mQnope: cute.Tensor,
        mQpe: cute.Tensor,
        mCKV: cute.Tensor,
        mKPE: cute.Tensor,
        mIdx: cute.Tensor,
        mOut: cute.Tensor,
        mLse: cute.Tensor,
        total_rows: Int32,
        scale_log2: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mQnope,
            mQpe,
            mCKV,
            mKPE,
            mIdx,
            mOut,
            mLse,
            total_rows,
            scale_log2,
        ).launch(
            grid=(HEADS // self.HEADS_PER_CTA, 1, 1),
            block=(self.NTHREADS_T1, 1, 1),
            smem=self.smem_bytes,
        )

    @cute.kernel
    def kernel(
        self,
        mQnope: cute.Tensor,
        mQpe: cute.Tensor,
        mCKV: cute.Tensor,
        mKPE: cute.Tensor,
        mIdx: cute.Tensor,
        mOut: cute.Tensor,
        mLse: cute.Tensor,
        total_rows: Int32,
        scale_log2: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        head_block, _, _ = cute.arch.block_idx()
        warp_id = tidx // 32
        lane_id = tidx % 32
        head_local = Int32(0)
        head = head_block
        group_tid = tidx & Int32(127)
        NWARPS = self.NTHREADS_T1 // 32
        TAIL_BASE = 2 * NWARPS
        TINY_SLOT = 3 * NWARPS
        META_BASE = TINY_SLOT + 1

        smem = cutlass.utils.SmemAllocator()
        sScore = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.HEADS_PER_CTA * TOPK,)),
            16,
        )
        sTmp = smem.allocate_tensor(
            Float32, cute.make_layout((32,)), 16
        )
        sTmpI = cute.recast_tensor(sTmp, Int32)

        qn = cute.make_tensor(
            cute.make_ptr(
                BFloat16,
                mQnope.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((HEADS * DK_CK,)),
        )
        qp = cute.make_tensor(
            cute.make_ptr(
                BFloat16,
                mQpe.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((HEADS * DK_PE,)),
        )
        kc = cute.make_tensor(
            cute.make_ptr(
                BFloat16,
                mCKV.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((total_rows * DK_CK,)),
        )
        kp = cute.make_tensor(
            cute.make_ptr(
                BFloat16,
                mKPE.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((total_rows * DK_PE,)),
        )
        go = cute.make_tensor(
            cute.make_ptr(
                BFloat16,
                mOut.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((HEADS * DV,)),
        )
        idxg = cute.make_tensor(
            cute.make_ptr(
                Int32,
                mIdx.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((TOPK,)),
        )

        atom_bf64 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=64
        )
        atom_i128 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128
        )
        rQ = cute.make_rmem_tensor((4,), BFloat16)
        rQf = cute.make_rmem_tensor((4,), Float32)
        qwarp = warp_id & Int32(3)
        dbase = qwarp * Int32(DK_CK // 4) + lane_id * Int32(4)
        qptr = cute.make_ptr(
            BFloat16,
            (qn.iterator + head * DK_CK + dbase).toint(),
            cute.AddressSpace.gmem,
            assumed_align=8,
        )
        cute.copy(
            atom_bf64,
            cute.make_tensor(qptr, cute.make_layout((4,))),
            rQ,
        )
        for e in cutlass.range_constexpr(4):
            rQf[e] = rQ[e].to(Float32)
        qpv = Float32(0.0)
        if lane_id < Int32(DK_PE // 4):
            qpv = qp[
                head * DK_PE
                + qwarp * Int32(DK_PE // 4)
                + lane_id
            ].to(Float32)

        # Independent 128-bit buffers cover the complete 8 KiB index row.
        # Their loads issue before the speculative two-row dot product, hiding
        # most of the exact tail proof behind useful work.
        rI0 = cute.make_rmem_tensor((4,), Int32)
        rI1 = cute.make_rmem_tensor((4,), Int32)
        rI2 = cute.make_rmem_tensor((4,), Int32)
        rI3 = cute.make_rmem_tensor((4,), Int32)
        ivecs = [rI0, rI1, rI2, rI3]
        for v in cutlass.range_constexpr(
            (TOPK // 4 + self.NTHREADS_T1 - 1) // self.NTHREADS_T1
        ):
            q4 = v * self.NTHREADS_T1 + tidx
            dq = q4 - Int32(TOPK // 4 - 1)
            qq4 = Int32(TOPK // 4 - 1) + (dq & (dq >> 31))
            off = qq4 * Int32(4)
            iptr = cute.make_ptr(
                Int32,
                (idxg.iterator + off).toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            cute.copy(
                atom_i128,
                cute.make_tensor(iptr, cute.make_layout((4,))),
                ivecs[v],
            )

        row_a = idxg[0]
        row_b = idxg[1]
        prelim = Int32(0)
        if row_a >= Int32(0):
            if row_b >= Int32(0):
                prelim = Int32(1)

        rKa = cute.make_rmem_tensor((4,), BFloat16)
        rKb = cute.make_rmem_tensor((4,), BFloat16)
        rKaf = cute.make_rmem_tensor((4,), Float32)
        rKbf = cute.make_rmem_tensor((4,), Float32)
        sa = Float32(0.0)
        sb = Float32(0.0)
        if prelim != Int32(0):
            pa = cute.make_ptr(
                BFloat16,
                (kc.iterator + row_a.to(Int64) * DK_CK + dbase).toint(),
                cute.AddressSpace.gmem,
                assumed_align=8,
            )
            pb = cute.make_ptr(
                BFloat16,
                (kc.iterator + row_b.to(Int64) * DK_CK + dbase).toint(),
                cute.AddressSpace.gmem,
                assumed_align=8,
            )
            cute.copy(
                atom_bf64,
                cute.make_tensor(pa, cute.make_layout((4,))),
                rKa,
            )
            cute.copy(
                atom_bf64,
                cute.make_tensor(pb, cute.make_layout((4,))),
                rKb,
            )
            for e in cutlass.range_constexpr(4):
                av = rKa[e].to(Float32)
                bv = rKb[e].to(Float32)
                rKaf[e] = av
                rKbf[e] = bv
                sa = sa + rQf[e] * av
                sb = sb + rQf[e] * bv
            if lane_id < Int32(DK_PE // 4):
                pd = qwarp * Int32(DK_PE // 4) + lane_id
                sa = sa + qpv * kp[row_a.to(Int64) * DK_PE + pd].to(Float32)
                sb = sb + qpv * kp[row_b.to(Int64) * DK_PE + pd].to(Float32)
            for off in (16, 8, 4, 2, 1):
                sa = sa + cute.arch.shuffle_sync_bfly(sa, off)
                sb = sb + cute.arch.shuffle_sync_bfly(sb, off)
            if lane_id == 0:
                sTmp[2 * warp_id] = sa
                sTmp[2 * warp_id + 1] = sb
        cute.arch.barrier()

        tail_any = Int32(0)
        for v in cutlass.range_constexpr(
            (TOPK // 4 + self.NTHREADS_T1 - 1) // self.NTHREADS_T1
        ):
            for e in cutlass.range_constexpr(4):
                q4 = v * self.NTHREADS_T1 + tidx
                dq = q4 - Int32(TOPK // 4 - 1)
                qq4 = Int32(TOPK // 4 - 1) + (dq & (dq >> 31))
                pos = qq4 * Int32(4) + Int32(e)
                if pos >= Int32(2):
                    tail_any = tail_any | ((ivecs[v][e] >> 31) + 1)
        tail_mask = cute.arch.vote_ballot_sync(tail_any != Int32(0))
        if lane_id == 0:
            sTmpI[TAIL_BASE + warp_id] = tail_mask
        cute.arch.barrier()
        if tidx == 0:
            tmask = Int32(0)
            for w in cutlass.range_constexpr(NWARPS):
                tmask = tmask | sTmpI[TAIL_BASE + w]
            tiny = Int32(0)
            if prelim != Int32(0):
                if tmask == Int32(0):
                    tiny = Int32(1)
                    wb = head_local * Int32(4)
                    s0 = sTmp[2 * wb] + sTmp[2 * (wb + 1)]
                    s0 = s0 + sTmp[2 * (wb + 2)] + sTmp[2 * (wb + 3)]
                    s1 = sTmp[2 * wb + 1] + sTmp[2 * (wb + 1) + 1]
                    s1 = (
                        s1
                        + sTmp[2 * (wb + 2) + 1]
                        + sTmp[2 * (wb + 3) + 1]
                    )
                    zd = (s0 - s1) * scale_log2
                    za = cute.arch.fmax(zd, -zd)
                    ee = cute.arch.exp2(-za)
                    e0 = Float32(1.0)
                    e1 = ee
                    if zd < Float32(0.0):
                        e0 = ee
                        e1 = Float32(1.0)
                    inv = Float32(1.0) / (e0 + e1)
                    sTmp[META_BASE + 2 * head_local] = e0 * inv
                    sTmp[META_BASE + 2 * head_local + 1] = e1 * inv
                    # two-row lse: max(s0,s1)*scale + log2(1 + 2^-|d|)
                    mLse[Int32(0), head] = (
                        cute.arch.fmax(s0, s1) * scale_log2
                        + cute.math.log2(Float32(1.0) + ee, fastmath=True)
                    )
            sTmpI[TINY_SLOT] = tiny
        cute.arch.barrier()

        if sTmpI[TINY_SLOT] != Int32(0):
            p0 = sTmp[META_BASE + 2 * head_local]
            p1 = sTmp[META_BASE + 2 * head_local + 1]
            rO = cute.make_rmem_tensor((4,), BFloat16)
            for e in cutlass.range_constexpr(4):
                rO[e] = BFloat16(p0 * rKaf[e] + p1 * rKbf[e])
            optr = cute.make_ptr(
                BFloat16,
                (go.iterator + head * DV + dbase).toint(),
                cute.AddressSpace.gmem,
                assumed_align=8,
            )
            if tidx < Int32(128):
                cute.copy(
                    atom_bf64,
                    rO,
                    cute.make_tensor(optr, cute.make_layout((4,))),
                )
            _thread_exit()

        # General T=1 fallback.  QK is scalar FP32 with a complete stable
        # max-subtracted softmax; probabilities remain FP32 until the final
        # BF16 output store.
        if tidx == 0:
            sTmp[META_BASE + head_local] = Float32(NEG_BIG)
        cute.arch.barrier()
        ii = Int32(0)
        while ii < Int32(TOPK):
            ridx = idxg[ii]
            if ridx >= Int32(0):
                if tidx < Int32(128):
                    rK = cute.make_rmem_tensor((4,), BFloat16)
                    kptr = cute.make_ptr(
                        BFloat16,
                        (kc.iterator + ridx.to(Int64) * DK_CK + dbase).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    )
                    cute.copy(
                        atom_bf64,
                        cute.make_tensor(kptr, cute.make_layout((4,))),
                        rK,
                    )
                    dot = Float32(0.0)
                    for e in cutlass.range_constexpr(4):
                        dot = dot + rQf[e] * rK[e].to(Float32)
                    if lane_id < Int32(DK_PE // 4):
                        pd = qwarp * Int32(DK_PE // 4) + lane_id
                        dot = dot + qpv * kp[
                            ridx.to(Int64) * DK_PE + pd
                        ].to(Float32)
                    for off in (16, 8, 4, 2, 1):
                        dot = dot + cute.arch.shuffle_sync_bfly(dot, off)
                    if lane_id == 0:
                        sTmp[warp_id] = dot
            cute.arch.barrier()
            if tidx == 0:
                score = Float32(NEG_BIG)
                if ridx >= Int32(0):
                    wb = head_local * Int32(4)
                    score = (
                        sTmp[wb]
                        + sTmp[wb + 1]
                        + sTmp[wb + 2]
                        + sTmp[wb + 3]
                    ) * scale_log2
                    sTmp[META_BASE + head_local] = cute.arch.fmax(
                        sTmp[META_BASE + head_local], score
                    )
                sScore[head_local * Int32(TOPK) + ii] = score
            cute.arch.barrier()
            ii = ii + Int32(1)

        mmax = sTmp[META_BASE + head_local]
        lsum = Float32(0.0)
        if tidx < Int32(128):
            jj = group_tid
            while jj < Int32(TOPK):
                sp = head_local * Int32(TOPK) + jj
                sc = sScore[sp]
                pp = Float32(0.0)
                if sc > Float32(-1.0e20):
                    pp = cute.arch.exp2(sc - mmax)
                sScore[sp] = pp
                lsum = lsum + pp
                jj = jj + Int32(128)
        for off in (16, 8, 4, 2, 1):
            lsum = lsum + cute.arch.shuffle_sync_bfly(lsum, off)
        if lane_id == 0:
            if warp_id < Int32(4):
                sTmp[warp_id] = lsum
        cute.arch.barrier()
        if tidx == 0:
            wb = head_local * Int32(4)
            den = sTmp[wb] + sTmp[wb + 1] + sTmp[wb + 2] + sTmp[wb + 3]
            invden = Float32(0.0)
            if den > Float32(1.0e-30):
                invden = Float32(1.0) / den
            # mmax still holds this head's score max (read pre-barrier)
            mLse[Int32(0), head] = mmax + cute.math.log2(den, fastmath=True)
            sTmp[META_BASE + head_local] = invden
        cute.arch.barrier()

        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        if tidx < Int32(128):
            kk = Int32(0)
            while kk < Int32(TOPK):
                pw = sScore[head_local * Int32(TOPK) + kk] * sTmp[
                    META_BASE + head_local
                ]
                if pw != Float32(0.0):
                    ridx = idxg[kk]
                    rV = cute.make_rmem_tensor((4,), BFloat16)
                    vptr = cute.make_ptr(
                        BFloat16,
                        (kc.iterator + ridx.to(Int64) * DK_CK + dbase).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    )
                    cute.copy(
                        atom_bf64,
                        cute.make_tensor(vptr, cute.make_layout((4,))),
                        rV,
                    )
                    acc0 = acc0 + pw * rV[0].to(Float32)
                    acc1 = acc1 + pw * rV[1].to(Float32)
                    acc2 = acc2 + pw * rV[2].to(Float32)
                    acc3 = acc3 + pw * rV[3].to(Float32)
                kk = kk + Int32(1)
        rOut = cute.make_rmem_tensor((4,), BFloat16)
        rOut[0] = BFloat16(acc0)
        rOut[1] = BFloat16(acc1)
        rOut[2] = BFloat16(acc2)
        rOut[3] = BFloat16(acc3)
        optr = cute.make_ptr(
            BFloat16,
            (go.iterator + head * DV + dbase).toint(),
            cute.AddressSpace.gmem,
            assumed_align=8,
        )
        if tidx < Int32(128):
            cute.copy(
                atom_bf64,
                rOut,
                cute.make_tensor(optr, cute.make_layout((4,))),
            )


_COMPILE_CACHE = {}
_WORKSPACE = {}
_LAUNCH_IDS = {}


def _splits_for(num_tokens: int) -> int:
    # Depth-2-pipeline sweep (dev/sweep_ns*.log): per-CTA consumer is LDS-
    # bound ~2.0us/tile with a large fixed cost (scan+issue+epi+merge ~9us),
    # so minimizing CTA *count* wins once one wave of parallelism is met:
    # ns16 for T<=8 (officials, sparse, 16-CTA cluster merge), ns8 for
    # 9<=T<=18 (T*8 <= 148), ns4 for 19<=T<=63, ns2 at T=64, and ns1 beyond
    # (dev/sweep_ns1.py: ns1 beats ns2 by 10-11% at T=128/256 - one CTA per
    # token removes the partial-write/merge phase entirely and halves the
    # fixed prologue/epi; T=64 starves at 0.43 waves and keeps ns2).
    # The two-row T=1 microkernel maps one output head to each ns16 CTA. Its
    # device-side gate falls through to the exact generic ns16 path otherwise.
    if num_tokens == 1:
        return 16
    if num_tokens >= 128:
        return 1
    if num_tokens >= 64:
        return 2
    if num_tokens >= 19:
        return 4
    if num_tokens >= 9:
        return 8
    if num_tokens <= 2:
        return 32
    return max(1, min(16, 512 // max(num_tokens, 1)))


def _workspace(device, nparts: int):
    ws = _WORKSPACE.get(device)
    if ws is None or ws[0].shape[0] < nparts:
        part_o = torch.empty((nparts, HEADS, DV), dtype=torch.float32, device=device)
        # per-split (l, m) per head: l rows [0, HEADS), m rows [HEADS, 2*HEADS)
        part_l = torch.empty((nparts, 2 * HEADS), dtype=torch.float32, device=device)
        flags = torch.zeros(nparts, dtype=torch.int32, device=device)
        _WORKSPACE[device] = (part_o, part_l, flags)
    return _WORKSPACE[device]


def _to_cute(t: torch.Tensor, leading_dim: int) -> cute.Tensor:
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens = q_nope.shape[0]
    num_pages = ckv_cache.shape[0]
    device = q_nope.device
    out = torch.empty((num_tokens, HEADS, DV), dtype=torch.bfloat16, device=device)
    lse = torch.empty((num_tokens, HEADS), dtype=torch.float32, device=device)

    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    scale_log2 = float(sm_scale) * LOG2E
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)

    # Compiled with the tvm-ffi ABI, so the launch takes torch tensors
    # directly.  --opt-level is left at the default: levels 2 and 3 were
    # measured identical here, and pinning level 2 costs the MoE kernel ~15%.
    # Otherwise:  The marked-dynamic CuTe tensors are only needed to describe
    # the layouts to cute.compile, so they are built on the compiling call
    # alone; rebuilding them per call cost ~3.3us each (~29us of the ~53us
    # host path) and left the GPU idle for most of a small-token launch.
    if num_tokens == 1:
        args_t1 = () if ("t1_exact",) in _COMPILE_CACHE else (
            _to_cute(q_nope, 2),
            _to_cute(q_pe, 2),
            _to_cute(ckv_cache, 2),
            _to_cute(kpe_cache, 2),
            _to_cute(sparse_indices, 1),
            _to_cute(out, 2),
            _to_cute(lse, 1),
            Int32(num_pages * 64),
            Float32(scale_log2),
            stream,
        )
        key_t1 = ("t1_exact",)
        compiled_t1 = _COMPILE_CACHE.get(key_t1)
        if compiled_t1 is None:
            compiled_t1 = cute.compile(_DSAT1(), *args_t1,
                                       options="--enable-tvm-ffi")
            _COMPILE_CACHE[key_t1] = compiled_t1
        compiled_t1(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, out,
                    lse, Int32(num_pages * 64), Float32(scale_log2), stream)
        return out, lse

    nsplit = _splits_for(num_tokens)
    part_o, part_l, flags = _workspace(device, num_tokens * nsplit)

    dev = str(device)
    lid = _LAUNCH_IDS.get(dev, 0) + 1
    _LAUNCH_IDS[dev] = lid
    lid = lid & LID_MASK  # flag words carry 28 bits of launch id

    nm = min(nsplit, 8 if num_tokens * nsplit <= 148 else 4)
    # ns16 merges inside CTA clusters (DSMEM partial exchange instead of
    # gmem flags/partials). One 16-CTA cluster per token for T<=7 (all
    # resident: the GPU hosts only ~7 concurrent 16-CTA clusters); at T=8
    # the layout moves to ns8 with one 8-CTA cluster per token and the
    # same DSMEM merge (64 CTAs, fully resident, no partner hop).
    cluster = 0
    if nsplit == 32:
        # T<=2 officials: the 32 splits of a token form two 16-CTA clusters;
        # partials move over DSMEM inside each half and one L2 flag hop
        # combines the halves (replaces the ~4.4us gmem-flags merge).
        cluster = 16
    elif nsplit == 16 and num_tokens != 1:
        # T=1 is handled by the dedicated _DSAT1 kernel and never reaches
        # this launcher; keeping the guard documents that ns16 clustering
        # was chosen for the multi-token configs only.
        cluster = 16
    micro = num_tokens in (2, 6)
    direct_afy = num_tokens == 8
    q_mcast = 4 if num_tokens == 2 else (1 if num_tokens == 6 else (3 if num_tokens == 7 else 0))
    # Same generic micro algorithm and full live-tail proof; T=2 alone uses
    # the measured G2R cache policy (T=6's heavy-token captures were neutral).
    micro_idx_g2r = num_tokens == 2
    micro_idx_g2r256 = num_tokens == 2
    key = (
        nsplit,
        nm,
        cluster,
        micro,
        direct_afy,
        q_mcast,
        micro_idx_g2r,
        micro_idx_g2r256,
    )
    args = () if key in _COMPILE_CACHE else (
        _to_cute(q_nope, 2),
        _to_cute(q_pe, 2),
        _to_cute(ckv_cache, 2),
        _to_cute(kpe_cache, 2),
        _to_cute(sparse_indices, 1),
        _to_cute(out, 2),
        _to_cute(lse, 1),
        _to_cute(part_o, 2),
        _to_cute(part_l, 1),
        _to_cute(flags, 0),
        Int32(num_tokens),
        Int32(num_pages * 64),
        Float32(scale_log2),
        Int32(lid),
        stream,
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            _DSA(
                nsplit,
                nm,
                cluster,
                micro,
                direct_afy,
                q_mcast,
                micro_idx_g2r,
                micro_idx_g2r256,
            ),
            *args,
            options="--enable-tvm-ffi",
        )
        _COMPILE_CACHE[key] = compiled
    compiled(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, out, lse,
        part_o, part_l, flags,
        Int32(num_tokens), Int32(num_pages * 64), Float32(scale_log2),
        Int32(lid), stream,
    )
    return out, lse
