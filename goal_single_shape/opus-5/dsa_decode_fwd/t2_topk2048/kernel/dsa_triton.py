"""DSA FWD decode kernel (sparse MLA decode) written from scratch in Triton.

Structure: flash-decoding.
  * kernel 1 (`_dsa_partial`): grid (SPLITS, T). The top-k index list is split over
    SPLITS programs round-robin *by tile*, so a short valid prefix still spreads
    across SMs. Each program runs an online-softmax pass over its tiles for all
    H heads at once and writes a normalised partial output plus its log-sum-exp.
  * kernel 2 (`_dsa_combine`): grid (T, H). Rescale/merge the per-split partials.

ckv rows double as V, so each KV row is fetched from HBM exactly once per program.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_partial(
    Q_NOPE, Q_PE, CKV, KPE, IDX,
    O_PART, LSE_PART, OUT,
    sm_scale,
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    stride_ckv_r, stride_kpe_r,
    stride_idx_t,
    stride_op_t, stride_op_s, stride_op_h,
    stride_lp_t, stride_lp_s,
    stride_o_t, stride_o_h,
    H: tl.constexpr, D: tl.constexpr, DPE: tl.constexpr,
    TOPK: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr,
    TPP: tl.constexpr, DIRECT: tl.constexpr,
):
    s = tl.program_id(0)
    t = tl.program_id(1)

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_pe = tl.arange(0, DPE)
    offs_n = tl.arange(0, BLOCK_N)

    q_nope = tl.load(Q_NOPE + t * stride_qn_t + offs_h[:, None] * stride_qn_h + offs_d[None, :])
    q_pe = tl.load(Q_PE + t * stride_qp_t + offs_h[:, None] * stride_qp_h + offs_pe[None, :])

    m_i = tl.full([H], float("-inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D], tl.float32)

    # Fully unrolled: TPP is a compile-time constant, so the index loads of all
    # of this program's tiles can be hoisted and overlapped by the compiler
    # instead of forming one dependent round-trip per loop iteration.
    idx_base = IDX + t * stride_idx_t
    for i in tl.static_range(TPP):
        tile = s + i * SPLITS
        idx = tl.load(idx_base + tile * BLOCK_N + offs_n)
        valid = idx >= 0
        if tl.max(valid.to(tl.int32)) > 0:
            sidx = tl.where(valid, idx, 0)
            k = tl.load(CKV + sidx[:, None] * stride_ckv_r + offs_d[None, :])
            kpe = tl.load(KPE + sidx[:, None] * stride_kpe_r + offs_pe[None, :])

            qk = tl.dot(q_nope, tl.trans(k))
            qk += tl.dot(q_pe, tl.trans(kpe))
            qk = qk * sm_scale
            qk = tl.where(valid[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(k.dtype), k, acc)
            m_i = m_new

    has = l_i > 0
    l_safe = tl.where(has, l_i, 1.0)
    if DIRECT:
        # SPLITS == 1: no cross-CTA merge needed, emit the final output here and
        # save a whole kernel launch.
        acc = acc / l_safe[:, None]
        tl.store(
            OUT + t * stride_o_t + offs_h[:, None] * stride_o_h + offs_d[None, :],
            acc.to(OUT.dtype.element_ty),
        )
    else:
        lse = tl.where(has, m_i + tl.log(l_safe), float("-inf"))
        tl.store(LSE_PART + t * stride_lp_t + s * stride_lp_s + offs_h, lse)
        if tl.max(has.to(tl.int32)) > 0:
            acc = acc / l_safe[:, None]
            tl.store(
                O_PART + t * stride_op_t + s * stride_op_s + offs_h[:, None] * stride_op_h + offs_d[None, :],
                acc,
            )


@triton.jit
def _dsa_combine(
    O_PART, LSE_PART, OUT,
    stride_op_t, stride_op_s, stride_op_h,
    stride_lp_t, stride_lp_s,
    stride_o_t, stride_o_h,
    BLOCK_D: tl.constexpr, SPLITS: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    dc = tl.program_id(2)
    offs_s = tl.arange(0, SPLITS)
    offs_d = dc * BLOCK_D + tl.arange(0, BLOCK_D)

    lse = tl.load(LSE_PART + t * stride_lp_t + offs_s * stride_lp_s + h)
    m = tl.max(lse)
    wgt = tl.exp(lse - m)
    o = tl.load(
        O_PART + t * stride_op_t + offs_s[:, None] * stride_op_s + h * stride_op_h + offs_d[None, :],
        mask=(wgt > 0)[:, None],
        other=0.0,
    )
    out = tl.sum(o * wgt[:, None], 0) / tl.sum(wgt)
    tl.store(OUT + t * stride_o_t + h * stride_o_h + offs_d, out.to(OUT.dtype.element_ty))


class DsaTriton:
    """Callable wrapper holding pre-allocated scratch so the timed path is launch-only."""

    def __init__(self, w, block_n=64, splits=16, num_warps=8, num_stages=2,
                 combine_warps=4, out_dtype=torch.bfloat16):
        self.w = w
        self.block_n = block_n
        self.splits = splits
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.combine_warps = combine_warps
        q_nope = w["q_nope"]
        T, H, D = q_nope.shape
        DPE = w["q_pe"].shape[-1]
        dev = q_nope.device
        self.T, self.H, self.D, self.DPE = T, H, D, DPE
        self.ckv_flat = w["ckv_cache"].reshape(-1, D)
        self.kpe_flat = w["kpe_cache"].reshape(-1, DPE)
        self.topk = w["sparse_indices"].shape[1]
        ntiles = self.topk // block_n
        assert ntiles % splits == 0, "splits must divide the tile count"
        self.tpp = ntiles // splits
        # Keep the combine's (SPLITS, BLOCK_D) fp32 tile small enough to stay in
        # registers: >32 values per thread starts spilling and dominates the op.
        self.block_d = max(32, min(D, 4096 // splits))
        self.o_part = torch.zeros((T, splits, H, D), device=dev, dtype=torch.float32)
        self.lse_part = torch.zeros((T, splits, H), device=dev, dtype=torch.float32)
        self.out = torch.empty((T, H, D), device=dev, dtype=out_dtype)

    def __repr__(self):
        return (f"DsaTriton(bn={self.block_n}, splits={self.splits}, tpp={self.tpp}, "
                f"warps={self.num_warps}, stages={self.num_stages}, bd={self.block_d})")

    def __call__(self):
        w, T, H, D, DPE = self.w, self.T, self.H, self.D, self.DPE
        _dsa_partial[(self.splits, T)](
            w["q_nope"], w["q_pe"], self.ckv_flat, self.kpe_flat, w["sparse_indices"],
            self.o_part, self.lse_part, self.out,
            w["sm_scale"],
            w["q_nope"].stride(0), w["q_nope"].stride(1),
            w["q_pe"].stride(0), w["q_pe"].stride(1),
            self.ckv_flat.stride(0), self.kpe_flat.stride(0),
            w["sparse_indices"].stride(0),
            self.o_part.stride(0), self.o_part.stride(1), self.o_part.stride(2),
            self.lse_part.stride(0), self.lse_part.stride(1),
            self.out.stride(0), self.out.stride(1),
            H=H, D=D, DPE=DPE, TOPK=self.topk, BLOCK_N=self.block_n, SPLITS=self.splits,
            TPP=self.tpp, DIRECT=(self.splits == 1),
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        if self.splits == 1:
            return self.out
        _dsa_combine[(T, H, D // self.block_d)](
            self.o_part, self.lse_part, self.out,
            self.o_part.stride(0), self.o_part.stride(1), self.o_part.stride(2),
            self.lse_part.stride(0), self.lse_part.stride(1),
            self.out.stride(0), self.out.stride(1),
            BLOCK_D=self.block_d, SPLITS=self.splits,
            num_warps=self.combine_warps, num_stages=1,
        )
        return self.out
