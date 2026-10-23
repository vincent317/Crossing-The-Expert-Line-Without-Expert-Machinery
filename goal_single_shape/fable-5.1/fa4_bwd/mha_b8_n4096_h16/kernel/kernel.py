"""FlashAttention backward for causal MHA on Blackwell (B200), written from scratch in Triton.

Math (attention backward, derived from the definitions; no reference code consulted):
  P  = exp(Q K^T * s - lse)            dV = P^T dO
  dP = dO V^T                          d  = rowsum(dO * O)
  dS = P * (dP - d)                    dQ = s * dS K ,  dK = s * dS^T Q

Design
------
* One fused pass, 5 GEMMs per (Q-block, K-block) pair. A 2-kernel dQ/dKdV split costs
  7 GEMMs (S recomputed twice) -- a 1.4x FLOP penalty no tuning recovers.
* Everything is computed in the TRANSPOSED orientation, S^T = K Q^T, with the K/V block as
  the M=128 row operand.  Then P^T and dS^T fall out of the MMA already in the layout the
  next MMAs need as their A operand (dV = P^T dO, dK = dS^T Q), and dQ is produced
  transposed, dQ^T = K^T dS^T, so dS^T is consumed as a B operand untransposed as well.
  Result: no accumulator is ever transposed; the only transposes are on TMA-loaded tiles,
  which are free (smem descriptor major-ness).
* grid = (kv_block, batch*head); dK/dV live in the accumulator for the whole causal
  Q-range; dQ^T contributions are reduced into an fp32 [B*H, D, S] scratch with TMA
  reduce-add (falls back to tl.atomic_add), then transposed+cast to bf16 [B,S,H,D].
  Atomic volume is S^2*D*4*BH/(2*BN) ~ 17 GB at BN=128, independent of BLOCK_M, and it
  lands in L2 because the programs in flight share one (b,h).
* The causal diagonal is peeled into a separate masked loop so interior iterations carry
  no `where`.
* TMEM budget (512 cols): S^T, dP^T, dQ^T each cost BLOCK_M cols, dK/dV 128 each ->
  BLOCK_M=64 fits, BLOCK_M=128 does not.
"""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634

_ALLOCATOR_SET = False


def _ensure_allocator():
    """Device-side TMA descriptors need a scratch allocator (one-time host setup)."""
    global _ALLOCATOR_SET
    if not _ALLOCATOR_SET:
        triton.set_allocator(lambda size, align, stream:
                             torch.empty(size, dtype=torch.int8, device="cuda"))
        _ALLOCATOR_SET = True


def descriptor_atomic_add_supported():
    try:
        return hasattr(tl.tensor_descriptor, "atomic_add")
    except AttributeError:
        return False


# ---------------------------------------------------------------- forward (untimed)
@triton.jit
def _fwd_kernel(Q, K, V, O, LSE2, qk_log2,
                stride_b, stride_s, stride_h, S, H,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    base = (bh // H) * stride_b + (bh % H) * stride_h
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + base + offs_m[:, None] * stride_s + offs_d[None, :])
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv = base + offs_n[:, None] * stride_s + offs_d[None, :]
        k = tl.load(K + kv)
        v = tl.load(V + kv)
        s = tl.dot(q, tl.trans(k)) * qk_log2
        s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(k.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    # lse stored in the log2 domain: lse2 = m + log2(l)
    tl.store(O + base + offs_m[:, None] * stride_s + offs_d[None, :], acc.to(O.dtype.element_ty))
    tl.store(LSE2 + bh * S + offs_m, m_i + tl.log2(l_i))


# ---------------------------------------------------------------- delta = rowsum(dO*O)
@triton.jit
def _bwd_preprocess(O, DO, Delta, stride_b, stride_s, stride_h, S, H,
                    BLOCK_M: tl.constexpr, D: tl.constexpr):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    base = (bh // H) * stride_b + (bh % H) * stride_h
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    p = base + offs_m[:, None] * stride_s + offs_d[None, :]
    o = tl.load(O + p).to(tl.float32)
    do = tl.load(DO + p).to(tl.float32)
    tl.store(Delta + bh * S + offs_m, tl.sum(o * do, 1))


# ---------------------------------------------------------------- scratch helpers
@triton.jit
def _zero_kernel(P, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(P + o, tl.zeros([BLOCK], tl.float32), mask=o < n)


@triton.jit
def _dq_finalize(DQT, DQ, sm_scale, stride_b, stride_s, stride_h, S, H,
                 BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr):
    """dq[b, s, h, :] = sm_scale * dqT[bh, :, s]   (transpose fp32 [BH, D, S] -> bf16 [B,S,H,D])."""
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = tl.program_id(2)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    t = tl.load(DQT + bh * D * S + offs_d[:, None] * S + offs_s[None, :])   # [BD, BS]
    out = tl.trans(t) * sm_scale                                            # [BS, BD]
    base = (bh // H) * stride_b + (bh % H) * stride_h
    tl.store(DQ + base + offs_s[:, None] * stride_s + offs_d[None, :], out.to(DQ.dtype.element_ty))


# ---------------------------------------------------------------- fused backward
# ABL (ablation, benchmarking only -- results are wrong unless ABL == 0):
#   1: no dQ at all      2: dQ dot but plain store instead of reduce-add
#   3: no exp2 (P := S)  4: no elementwise at all (P := S, dS := dP)
@triton.jit
def _bwd_inner(dk, dv, k, v, q_desc, do_desc, dqt_desc, DQT, LSE2, Delta,
               lse_base, dqt_base, offs_n, offs_dd, qk_log2, S,
               m_start, m_end,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
               MASK: tl.constexpr, DQ_TMA: tl.constexpr, WS: tl.constexpr,
               ABL: tl.constexpr):
    for mi in tl.range(m_start, m_end, warp_specialize=WS):
        m0 = mi * BLOCK_M
        offs_m = m0 + tl.arange(0, BLOCK_M)
        q = q_desc.load([m0, 0])                        # [BM, D] bf16
        do = do_desc.load([m0, 0])                      # [BM, D] bf16
        lse2 = tl.load(LSE2 + lse_base + offs_m)        # [BM]
        delta = tl.load(Delta + lse_base + offs_m)      # [BM]

        sT = tl.dot(k, tl.trans(q))                     # [BN, BM]  = (Q K^T)^T
        if ABL >= 3:
            pT = sT
        else:
            pT = tl.exp2(sT * qk_log2 - lse2[None, :])
        if MASK:
            pT = tl.where(offs_m[None, :] >= offs_n[:, None], pT, 0.0)
        dv = tl.dot(pT.to(tl.bfloat16), do, dv)         # dV += P^T dO
        dpT = tl.dot(v, tl.trans(do))                   # [BN, BM]  = (dO V^T)^T
        if ABL >= 4:
            dsT = dpT.to(tl.bfloat16)
        else:
            dsT = (pT * (dpT - delta[None, :])).to(tl.bfloat16)
        dk = tl.dot(dsT, q, dk)                         # dK += dS^T Q
        if ABL != 1:
            dqT = tl.dot(tl.trans(k), dsT)              # [D, BM]   = (dS K)^T
            if ABL == 2:
                dqt_desc.store([0, m0], dqT)
            elif DQ_TMA:
                dqt_desc.atomic_add([0, m0], dqT)
            else:
                tl.atomic_add(DQT + dqt_base + offs_dd[:, None] * S + offs_m[None, :], dqT)
    return dk, dv


@triton.jit
def _fused_bwd(Q, K, V, DO, DQT, DK, DV, LSE2, Delta, sm_scale, qk_log2,
               stride_s, S, H,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
               NUM_M: tl.constexpr, DQ_TMA: tl.constexpr, WS: tl.constexpr,
               ABL: tl.constexpr):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    base = (bh // H) * S * stride_s + (bh % H) * D      # element offset of [b, :, h, :]
    lse_base = bh * S
    dqt_base = bh * D * S

    q_desc = tl.make_tensor_descriptor(Q + base, shape=[S, D], strides=[stride_s, 1],
                                       block_shape=[BLOCK_M, D])
    do_desc = tl.make_tensor_descriptor(DO + base, shape=[S, D], strides=[stride_s, 1],
                                        block_shape=[BLOCK_M, D])
    k_desc = tl.make_tensor_descriptor(K + base, shape=[S, D], strides=[stride_s, 1],
                                       block_shape=[BLOCK_N, D])
    v_desc = tl.make_tensor_descriptor(V + base, shape=[S, D], strides=[stride_s, 1],
                                       block_shape=[BLOCK_N, D])
    dqt_desc = tl.make_tensor_descriptor(DQT + dqt_base, shape=[D, S], strides=[S, 1],
                                         block_shape=[D, BLOCK_M])

    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_dd = tl.arange(0, D)
    k = k_desc.load([n0, 0])                            # [BN, D]
    v = v_desc.load([n0, 0])
    dk = tl.zeros([BLOCK_N, D], tl.float32)
    dv = tl.zeros([BLOCK_N, D], tl.float32)

    m_lo = n0 // BLOCK_M
    m_mask_end = (n0 + BLOCK_N + BLOCK_M - 1) // BLOCK_M
    dk, dv = _bwd_inner(dk, dv, k, v, q_desc, do_desc, dqt_desc, DQT, LSE2, Delta,
                        lse_base, dqt_base, offs_n, offs_dd, qk_log2, S,
                        m_lo, m_mask_end, BLOCK_M, BLOCK_N, D, True, DQ_TMA, WS, ABL)
    dk, dv = _bwd_inner(dk, dv, k, v, q_desc, do_desc, dqt_desc, DQT, LSE2, Delta,
                        lse_base, dqt_base, offs_n, offs_dd, qk_log2, S,
                        m_mask_end, NUM_M, BLOCK_M, BLOCK_N, D, False, DQ_TMA, WS, ABL)

    kv = base + offs_n[:, None] * stride_s + offs_dd[None, :]
    tl.store(DK + kv, (dk * sm_scale).to(DK.dtype.element_ty))
    tl.store(DV + kv, dv.to(DV.dtype.element_ty))


# ---------------------------------------------------------------- python wrappers
def fa_forward(q, k, v, scale, BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=2):
    b, s, h, d = q.shape
    o = torch.empty_like(q)
    lse2 = torch.empty((b * h, s), dtype=torch.float32, device=q.device)
    _fwd_kernel[(triton.cdiv(s, BLOCK_M), b * h)](
        q, k, v, o, lse2, scale * LOG2E,
        q.stride(0), q.stride(1), q.stride(2), s, h,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=d,
        num_warps=num_warps, num_stages=num_stages)
    return o, lse2


DEFAULT = dict(pre_m=256, pre_warps=4,
               BLOCK_M=64, BLOCK_N=128, num_warps=8, num_stages=2,
               dq_tma=True, ws=True)


def make_bufs(q):
    b, s, h, d = q.shape
    return dict(
        dq=torch.empty_like(q), dk=torch.empty_like(q), dv=torch.empty_like(q),
        dqT=torch.empty((b * h, d, s), dtype=torch.float32, device=q.device),
        delta=torch.empty((b * h, s), dtype=torch.float32, device=q.device),
    )


def fa_backward(q, k, v, o, do, lse2, scale, cfg=DEFAULT, bufs=None):
    _ensure_allocator()
    b, s, h, d = q.shape
    if bufs is None:
        bufs = make_bufs(q)
    dq, dk, dv, dqT, delta = bufs["dq"], bufs["dk"], bufs["dv"], bufs["dqT"], bufs["delta"]
    sb, ss, sh = q.stride(0), q.stride(1), q.stride(2)
    assert sh == d and q.stride(3) == 1, "expects [b, s, h, d] with d contiguous"
    dq_tma = cfg["dq_tma"] and descriptor_atomic_add_supported()

    pm = cfg["pre_m"]
    _bwd_preprocess[(triton.cdiv(s, pm), b * h)](
        o, do, delta, sb, ss, sh, s, h, BLOCK_M=pm, D=d,
        num_warps=cfg["pre_warps"], num_stages=1)

    n = dqT.numel()
    _zero_kernel[(triton.cdiv(n, 8192),)](dqT, n, BLOCK=8192, num_warps=8)

    BM, BN = cfg["BLOCK_M"], cfg["BLOCK_N"]
    _fused_bwd[(triton.cdiv(s, BN), b * h)](
        q, k, v, do, dqT, dk, dv, lse2, delta, scale, scale * LOG2E,
        ss, s, h, BLOCK_M=BM, BLOCK_N=BN, D=d, NUM_M=triton.cdiv(s, BM),
        DQ_TMA=dq_tma, WS=cfg["ws"], ABL=cfg.get("abl", 0),
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])

    _dq_finalize[(s // 64, d // 64, b * h)](
        dqT, dq, scale, sb, ss, sh, s, h, BLOCK_S=64, BLOCK_D=64, D=d, num_warps=4)
    return dq, dk, dv


# ================================================================ split variant
# Two kernels, ZERO atomics: dK/dV loops over Q keeping dK/dV in TMEM; dQ loops over K
# keeping dQ in TMEM. Costs 7 GEMMs instead of 5 (S is recomputed in the dQ pass) but the
# ablation shows the cross-program dQ reduce-add costs ~4.6 ms -- far more than the
# ~1.2 ms an extra GEMM pair costs. Both kernels keep the transposed orientation and TMA.
@triton.jit
def _split_dkdv(Q, K, V, DO, DK, DV, LSE2, Delta, sm_scale, qk_log2,
                stride_s, S, H,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
                NUM_M: tl.constexpr, WS: tl.constexpr):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    base = (bh // H) * S * stride_s + (bh % H) * D
    lse_base = bh * S
    q_desc = tl.make_tensor_descriptor(Q + base, shape=[S, D], strides=[stride_s, 1],
                                       block_shape=[BLOCK_M, D])
    do_desc = tl.make_tensor_descriptor(DO + base, shape=[S, D], strides=[stride_s, 1],
                                        block_shape=[BLOCK_M, D])
    k_desc = tl.make_tensor_descriptor(K + base, shape=[S, D], strides=[stride_s, 1],
                                       block_shape=[BLOCK_N, D])
    v_desc = tl.make_tensor_descriptor(V + base, shape=[S, D], strides=[stride_s, 1],
                                       block_shape=[BLOCK_N, D])
    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    k = k_desc.load([n0, 0])
    v = v_desc.load([n0, 0])
    dk = tl.zeros([BLOCK_N, D], tl.float32)
    dv = tl.zeros([BLOCK_N, D], tl.float32)

    m_lo = n0 // BLOCK_M
    m_mask_end = (n0 + BLOCK_N + BLOCK_M - 1) // BLOCK_M
    for mi in tl.range(m_lo, m_mask_end, warp_specialize=False):
        m0 = mi * BLOCK_M
        offs_m = m0 + tl.arange(0, BLOCK_M)
        q = q_desc.load([m0, 0])
        do = do_desc.load([m0, 0])
        lse2 = tl.load(LSE2 + lse_base + offs_m)
        delta = tl.load(Delta + lse_base + offs_m)
        sT = tl.dot(k, tl.trans(q))
        pT = tl.exp2(sT * qk_log2 - lse2[None, :])
        pT = tl.where(offs_m[None, :] >= offs_n[:, None], pT, 0.0)
        dv = tl.dot(pT.to(tl.bfloat16), do, dv)
        dpT = tl.dot(v, tl.trans(do))
        dsT = (pT * (dpT - delta[None, :])).to(tl.bfloat16)
        dk = tl.dot(dsT, q, dk)
    for mi in tl.range(m_mask_end, NUM_M, warp_specialize=WS):
        m0 = mi * BLOCK_M
        offs_m = m0 + tl.arange(0, BLOCK_M)
        q = q_desc.load([m0, 0])
        do = do_desc.load([m0, 0])
        lse2 = tl.load(LSE2 + lse_base + offs_m)
        delta = tl.load(Delta + lse_base + offs_m)
        sT = tl.dot(k, tl.trans(q))
        pT = tl.exp2(sT * qk_log2 - lse2[None, :])
        dv = tl.dot(pT.to(tl.bfloat16), do, dv)
        dpT = tl.dot(v, tl.trans(do))
        dsT = (pT * (dpT - delta[None, :])).to(tl.bfloat16)
        dk = tl.dot(dsT, q, dk)

    kv = base + offs_n[:, None] * stride_s + offs_d[None, :]
    tl.store(DK + kv, (dk * sm_scale).to(DK.dtype.element_ty))
    tl.store(DV + kv, dv.to(DV.dtype.element_ty))


@triton.jit
def _split_dq(Q, K, V, DO, DQ, LSE2, Delta, sm_scale, qk_log2,
              stride_s, S, H,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
              WS: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    base = (bh // H) * S * stride_s + (bh % H) * D
    lse_base = bh * S
    k_desc = tl.make_tensor_descriptor(K + base, shape=[S, D], strides=[stride_s, 1],
                                       block_shape=[BLOCK_N, D])
    v_desc = tl.make_tensor_descriptor(V + base, shape=[S, D], strides=[stride_s, 1],
                                       block_shape=[BLOCK_N, D])
    q_desc = tl.make_tensor_descriptor(Q + base, shape=[S, D], strides=[stride_s, 1],
                                       block_shape=[BLOCK_M, D])
    do_desc = tl.make_tensor_descriptor(DO + base, shape=[S, D], strides=[stride_s, 1],
                                        block_shape=[BLOCK_M, D])
    m0 = pid_m * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = q_desc.load([m0, 0])
    do = do_desc.load([m0, 0])
    lse2 = tl.load(LSE2 + lse_base + offs_m)
    delta = tl.load(Delta + lse_base + offs_m)
    dq = tl.zeros([BLOCK_M, D], tl.float32)

    n_full = m0 // BLOCK_N
    for ni in tl.range(0, n_full, warp_specialize=WS):
        n0 = ni * BLOCK_N
        k = k_desc.load([n0, 0])
        v = v_desc.load([n0, 0])
        s = tl.dot(q, tl.trans(k))
        p = tl.exp2(s * qk_log2 - lse2[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(tl.bfloat16)
        dq = tl.dot(ds, k, dq)
    for ni in tl.range(n_full, (m0 + BLOCK_M + BLOCK_N - 1) // BLOCK_N, warp_specialize=False):
        n0 = ni * BLOCK_N
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k = k_desc.load([n0, 0])
        v = v_desc.load([n0, 0])
        s = tl.dot(q, tl.trans(k))
        p = tl.exp2(s * qk_log2 - lse2[:, None])
        p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(tl.bfloat16)
        dq = tl.dot(ds, k, dq)
    tl.store(DQ + base + offs_m[:, None] * stride_s + offs_d[None, :],
             (dq * sm_scale).to(DQ.dtype.element_ty))


SPLIT_DEFAULT = dict(pre_m=256, pre_warps=4,
                     kv_M=128, kv_N=128, kv_warps=8, kv_stages=2, kv_ws=False,
                     q_M=128, q_N=128, q_warps=8, q_stages=2, q_ws=False)


def fa_backward_split(q, k, v, o, do, lse2, scale, cfg=SPLIT_DEFAULT, bufs=None):
    _ensure_allocator()
    b, s, h, d = q.shape
    if bufs is None:
        bufs = make_bufs(q)
    dq, dk, dv, delta = bufs["dq"], bufs["dk"], bufs["dv"], bufs["delta"]
    sb, ss, sh = q.stride(0), q.stride(1), q.stride(2)

    pm = cfg["pre_m"]
    _bwd_preprocess[(triton.cdiv(s, pm), b * h)](
        o, do, delta, sb, ss, sh, s, h, BLOCK_M=pm, D=d,
        num_warps=cfg["pre_warps"], num_stages=1)
    _split_dkdv[(triton.cdiv(s, cfg["kv_N"]), b * h)](
        q, k, v, do, dk, dv, lse2, delta, scale, scale * LOG2E,
        ss, s, h, BLOCK_M=cfg["kv_M"], BLOCK_N=cfg["kv_N"], D=d,
        NUM_M=triton.cdiv(s, cfg["kv_M"]), WS=cfg["kv_ws"],
        num_warps=cfg["kv_warps"], num_stages=cfg["kv_stages"])
    _split_dq[(triton.cdiv(s, cfg["q_M"]), b * h)](
        q, k, v, do, dq, lse2, delta, scale, scale * LOG2E,
        ss, s, h, BLOCK_M=cfg["q_M"], BLOCK_N=cfg["q_N"], D=d, WS=cfg["q_ws"],
        num_warps=cfg["q_warps"], num_stages=cfg["q_stages"])
    return dq, dk, dv
