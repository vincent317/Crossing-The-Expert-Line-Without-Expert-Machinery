"""FlashAttention backward for GQA causal attention, Triton, SM100.

Scheme: kv-outer / q-inner, with the 12 query heads that share one kv head all
handled by the same CTA. Consequence: dK/dV accumulate in registers across the
whole q loop AND across the 12 heads, so they are written exactly once with no
atomics and no fp32 staging buffer. Only dQ needs cross-CTA accumulation
(fp32 atomics + a cast pass), because dQ reduces over kv blocks.

5 matmuls per (q_tile, kv_tile, head): S, dV, dP, dK, dQ. No recompute pass.
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _delta_kernel(O, DO, DELTA, so_b, so_s, so_h, sd_b, sd_h, S,
                  HQ: tl.constexpr, D: tl.constexpr, BS: tl.constexpr):
    """DELTA[b,h,s] = sum_d O[b,s,h,d] * DO[b,s,h,d], in fp32."""
    pid = tl.program_id(0)
    nblk = S // BS
    bh = pid // nblk
    i = pid % nblk
    b = bh // HQ
    h = bh % HQ
    offs = i * BS + tl.arange(0, BS)
    dd = tl.arange(0, D)
    base = b * so_b + h * so_h + offs[:, None] * so_s + dd[None, :]
    o = tl.load(O + base).to(tl.float32)
    do = tl.load(DO + base).to(tl.float32)
    tl.store(DELTA + b * sd_b + h * sd_h + offs, tl.sum(o * do, 1))


@triton.jit
def _cast_kernel(SRC, DST, N, BS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BS + tl.arange(0, BS)
    m = offs < N
    tl.store(DST + offs, tl.load(SRC + offs, mask=m).to(tl.bfloat16), mask=m)


@triton.jit
def _bwd_kv_kernel(Q, K, V, DO, LSE, DELTA, DQ_ACC, DK, DV,
                   sq_b, sq_s, sq_h,
                   sk_b, sk_s, sk_h,
                   sl_b, sl_h,
                   S, NUM_BH, GROUP,
                   qk_log2, scale,
                   HKV: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    """One CTA owns one kv block of one kv head, and sweeps every q row above
    the diagonal for all GROUP query heads sharing that kv head.

    dK/dV therefore accumulate over (head, q_tile) inside the CTA and are stored
    once, in bf16, with no atomics. dQ reduces over kv blocks, so it goes through
    fp32 atomics into DQ_ACC.
    """
    pid = tl.program_id(0)
    j = pid // NUM_BH          # kv block index; slow axis -> heaviest work first
    bh = pid % NUM_BH
    b = bh // HKV
    hkv = bh % HKV

    offs_n = j * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)

    kv_off = b * sk_b + hkv * sk_h + offs_n[:, None] * sk_s + offs_d[None, :]
    k_tile = tl.load(K + kv_off)          # (BN, D) bf16
    v_tile = tl.load(V + kv_off)
    kT = tl.trans(k_tile)                 # (D, BN)
    vT = tl.trans(v_tile)

    dk_acc = tl.zeros([BN, D], dtype=tl.float32)
    dv_acc = tl.zeros([BN, D], dtype=tl.float32)

    i_start = (j * BN) // BM
    nblk_m = S // BM

    for hg in range(0, GROUP):
        hq = hkv * GROUP + hg
        qb = b * sq_b + hq * sq_h
        lb = b * sl_b + hq * sl_h
        for i in range(i_start, nblk_m):
            offs_m = i * BM + tl.arange(0, BM)
            mo = qb + offs_m[:, None] * sq_s + offs_d[None, :]
            q_tile = tl.load(Q + mo)                       # (BM, D) bf16
            do_tile = tl.load(DO + mo)
            lse_v = tl.load(LSE + lb + offs_m)             # (BM,) fp32
            del_v = tl.load(DELTA + lb + offs_m)

            s = tl.dot(q_tile, kT)                         # (BM, BN) fp32
            p = tl.math.exp2(s * qk_log2 - lse_v[:, None])
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
            p_bf = p.to(tl.bfloat16)
            dv_acc = tl.dot(tl.trans(p_bf), do_tile, dv_acc)

            dp = tl.dot(do_tile, vT)                       # (BM, BN) fp32
            ds = p * (dp - del_v[:, None])
            ds_bf = ds.to(tl.bfloat16)
            dk_acc = tl.dot(tl.trans(ds_bf), q_tile, dk_acc)

            dq_c = tl.dot(ds_bf, k_tile) * scale           # (BM, D) fp32
            tl.atomic_add(DQ_ACC + mo, dq_c, sem="relaxed")

    tl.store(DK + kv_off, (dk_acc * scale).to(tl.bfloat16))
    tl.store(DV + kv_off, dv_acc.to(tl.bfloat16))


@triton.jit
def _bwd_kv_kernel_v3(q_desc, k_desc, v_desc, do_desc, dq_desc, DQ_ACC, LSE, DELTA, DK, DV,
                      sk_b, sk_s, sk_h, sl_b, sl_h, sq_s, sq_h,
                      S, NUM_J, qk_log2, scale,
                      HKV: tl.constexpr, GROUP: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                      WS: tl.constexpr, NSTAGE: tl.constexpr, DQ_MODE: tl.constexpr,
                      NO_ACC_MB: tl.constexpr):
    """kv-outer, GQA-packed, transposed orientation (S^T = K Q^T).

    * Each CTA owns TWO kv blocks, c and NUM_J-1-c, so every CTA does the same
      amount of causal work (perfect static load balance) and dK/dV of each block
      are still produced by exactly one CTA (no atomics).
    * CTAs of one (b, hkv) are adjacent in the grid and sweep the q tiles from the
      end of the sequence down to the diagonal, head-innermost, so concurrently
      running CTAs touch the same narrow band of Q / dO / dQ_acc -> L2 resident.
    * S^T/dP^T being (BN, BM) makes dV += P^T dO and dK += dS^T Q transpose-free.
    * DQ_MODE 0: TMA reduce-add via descriptor; 1: per-thread atomics; 2: skip (ablation).
    """
    pid = tl.program_id(0)
    half = NUM_J // 2
    bh = pid // half
    c = pid % half
    b = bh // HKV
    hkv = bh % HKV
    offs_d = tl.arange(0, D)
    nblk_m = S // BM

    for rep in range(0, 2):
        j = c + rep * (NUM_J - 1 - 2 * c)          # rep 0 -> c, rep 1 -> NUM_J-1-c
        row_kv = b * S + j * BN
        k_tile = k_desc.load([row_kv, hkv * D])      # (BN, D) bf16
        v_tile = v_desc.load([row_kv, hkv * D])
        offs_n = j * BN + tl.arange(0, BN)
        dk_acc = tl.zeros([BN, D], dtype=tl.float32)
        dv_acc = tl.zeros([BN, D], dtype=tl.float32)

        i_start = (j * BN) // BM
        n_iter = (nblk_m - i_start) * GROUP
        for t in tl.range(0, n_iter, num_stages=NSTAGE, warp_specialize=WS,
                          disallow_acc_multi_buffer=NO_ACC_MB):
            i = nblk_m - 1 - t // GROUP
            hq = hkv * GROUP + t % GROUP
            row_q = b * S + i * BM
            offs_m = i * BM + tl.arange(0, BM)
            q_tile = q_desc.load([row_q, hq * D])     # (BM, D) bf16
            do_tile = do_desc.load([row_q, hq * D])
            lb = b * sl_b + hq * sl_h
            lse_v = tl.load(LSE + lb + offs_m)
            del_v = tl.load(DELTA + lb + offs_m)

            sT = tl.dot(k_tile, tl.trans(q_tile))     # (BN, BM) fp32
            pT = tl.math.exp2(sT * qk_log2 - lse_v[None, :])
            pT = tl.where(offs_m[None, :] >= offs_n[:, None], pT, 0.0)
            pT_bf = pT.to(tl.bfloat16)
            dv_acc = tl.dot(pT_bf, do_tile, dv_acc)   # (BN, BM) x (BM, D)

            dpT = tl.dot(v_tile, tl.trans(do_tile))   # (BN, BM) fp32
            dsT = pT * (dpT - del_v[None, :])
            dsT_bf = dsT.to(tl.bfloat16)
            dk_acc = tl.dot(dsT_bf, q_tile, dk_acc)   # (BN, BM) x (BM, D)

            dq_c = tl.dot(tl.trans(dsT_bf), k_tile) * scale   # (BM, BN) x (BN, D)
            if DQ_MODE == 0:
                dq_desc.atomic_add([row_q, hq * D], dq_c)
            elif DQ_MODE == 1:
                tl.atomic_add(DQ_ACC + (row_q + tl.arange(0, BM))[:, None] * sq_s + hq * sq_h
                              + offs_d[None, :], dq_c, sem="relaxed")

        kv_off = b * sk_b + hkv * sk_h + offs_n[:, None] * sk_s + offs_d[None, :]
        tl.store(DK + kv_off, (dk_acc * scale).to(tl.bfloat16))
        tl.store(DV + kv_off, dv_acc.to(tl.bfloat16))


CONFIGS = [
    # (BM, BN, num_warps, num_stages)   -- v1 kernel
    (64, 64, 8, 2),
]

# (BM, BN, warps, stages, warp_specialize, dq_mode, no_acc_multibuffer) -- v3 kernel
CONFIGS_V3 = [
    (64, 128, 8, 2, False, 0, True),
    (64, 128, 8, 2, False, 2, True),   # ablation: no dq write
    (64, 128, 8, 2, False, 1, True),   # ablation: per-thread atomics
    (64, 128, 8, 3, False, 0, True),
    (64, 128, 8, 2, True, 0, True),
    (64, 128, 4, 2, False, 0, True),
    (64, 128, 8, 2, False, 0, False),
    (128, 128, 8, 2, False, 0, True),
    (128, 64, 8, 2, False, 0, True),
    (64, 64, 8, 2, False, 0, True),
    (64, 64, 8, 2, False, 2, True),    # ablation: no dq write
]


def fa_bwd(q, k, v, o, do, lse, scale, cfg=(128, 64, 8, 2), out=None):
    """q,do,o: (B,S,Hq,D) bf16; k,v: (B,S,Hkv,D) bf16; lse: (B,Hq,S) fp32 natural log.

    Returns dq,dk,dv in bf16. `out` lets the caller reuse buffers across bench iters.
    """
    B, S, HQ, D = q.shape
    HKV = k.shape[2]
    GROUP = HQ // HKV
    BM, BN, warps, stages = cfg

    if out is None:
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dq_acc = torch.empty(B, S, HQ, D, device=q.device, dtype=torch.float32)
        delta = torch.empty(B, HQ, S, device=q.device, dtype=torch.float32)
        lse_log2 = torch.empty(B, HQ, S, device=q.device, dtype=torch.float32)
    else:
        dq, dk, dv, dq_acc, delta, lse_log2 = out

    # exp2-based softmax: fold log2(e) into both the score scale and the lse
    torch.mul(lse, LOG2E, out=lse_log2)
    dq_acc.zero_()

    _delta_kernel[(B * HQ * (S // 128),)](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2),
        delta.stride(0), delta.stride(1), S,
        HQ=HQ, D=D, BS=128, num_warps=4)

    num_bh = B * HKV
    grid = ((S // BN) * num_bh,)
    _bwd_kv_kernel[grid](
        q, k, v, do, lse_log2, delta, dq_acc, dk, dv,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        delta.stride(0), delta.stride(1),
        S, num_bh, GROUP,
        scale * LOG2E, scale,
        HKV=HKV, BM=BM, BN=BN, D=D,
        num_warps=warps, num_stages=stages)

    n = dq.numel()
    _cast_kernel[(triton.cdiv(n, 4096),)](dq_acc, dq, n, BS=4096, num_warps=4)
    return dq, dk, dv


def fa_bwd_v3(q, k, v, o, do, lse, scale, cfg=(64, 128, 8, 2, False, 0, True), out=None):
    """v3 kernel driver (same signature / numerics as fa_bwd)."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    B, S, HQ, D = q.shape
    HKV = k.shape[2]
    GROUP = HQ // HKV
    BM, BN, warps, stages, ws, dq_mode, no_acc_mb = cfg
    dq, dk, dv, dq_acc, delta, lse_log2 = make_buffers(q, k, v) if out is None else out

    torch.mul(lse, LOG2E, out=lse_log2)
    dq_acc.zero_()
    _delta_kernel[(B * HQ * (S // 128),)](
        o, do, delta, o.stride(0), o.stride(1), o.stride(2),
        delta.stride(0), delta.stride(1), S, HQ=HQ, D=D, BS=128, num_warps=4)

    # 2-D views (rows = b*S + s, cols = h*D + d): a (BM, D) box at column h*D is one head
    q_desc = TensorDescriptor.from_tensor(q.view(B * S, HQ * D), [BM, D])
    do_desc = TensorDescriptor.from_tensor(do.view(B * S, HQ * D), [BM, D])
    k_desc = TensorDescriptor.from_tensor(k.view(B * S, HKV * D), [BN, D])
    v_desc = TensorDescriptor.from_tensor(v.view(B * S, HKV * D), [BN, D])
    dq_desc = TensorDescriptor.from_tensor(dq_acc.view(B * S, HQ * D), [BM, D])

    num_j = S // BN
    _bwd_kv_kernel_v3[((num_j // 2) * B * HKV,)](
        q_desc, k_desc, v_desc, do_desc, dq_desc, dq_acc, lse_log2, delta, dk, dv,
        k.stride(0), k.stride(1), k.stride(2), delta.stride(0), delta.stride(1),
        dq_acc.stride(1), dq_acc.stride(2),
        S, num_j, scale * LOG2E, scale,
        HKV=HKV, GROUP=GROUP, BM=BM, BN=BN, D=D,
        WS=ws, NSTAGE=stages, DQ_MODE=dq_mode, NO_ACC_MB=no_acc_mb,
        num_warps=warps, num_stages=stages)

    n = dq.numel()
    _cast_kernel[(triton.cdiv(n, 4096),)](dq_acc, dq, n, BS=4096, num_warps=4)
    return dq, dk, dv


def make_buffers(q, k, v):
    B, S, HQ, D = q.shape
    return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v),
            torch.empty(B, S, HQ, D, device=q.device, dtype=torch.float32),
            torch.empty(B, HQ, S, device=q.device, dtype=torch.float32),
            torch.empty(B, HQ, S, device=q.device, dtype=torch.float32))

