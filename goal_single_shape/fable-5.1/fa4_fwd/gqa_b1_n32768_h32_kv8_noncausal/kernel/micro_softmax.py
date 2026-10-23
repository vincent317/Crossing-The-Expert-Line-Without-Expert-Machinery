"""Micro-benchmarks for the v4 softmax partition, isolated from the attention pipeline.

Each kernel runs the same per-tile softmax work as one 128-row half of v4 (S in TMEM ->
registers, row max, exp2, P back to TMEM) for N_ITER iterations with no MMA/TMA at all,
so the number is the pure issue cost of one softmax variant on one SM.  4 warps, 148 CTAs
(one per SM), so per-CTA time == wall time.
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout)

from fa_kernel_v4 import _fma2, _mul2, _exp2_f16x2
import common as C


@gluon.jit
def _ex2_f32(x):
    return gl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=gl.float32,
                                     is_pure=True, pack=1)


@gluon.jit
def _sm_micro(out, qk_scale, N_ITER, VARIANT: gl.constexpr, SUB_COLS: gl.constexpr,
              NSUB: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_N: gl.constexpr = SUB_COLS * NSUB
    s_tmem = allocate_tensor_memory(gl.float32, [128, BLOCK_N],
                                    TensorMemoryLayout((128, BLOCK_N), col_stride=1))
    sub_tmem_layout: gl.constexpr = TensorMemoryLayout((128, SUB_COLS), col_stride=1)
    sub_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, [128, SUB_COLS], sub_tmem_layout, num_warps)
    row_layout: gl.constexpr = gl.SliceLayout(1, sub_layout)
    p_buf = s_tmem.slice(0, BLOCK_N // 2)._reinterpret(gl.bfloat16, [128, BLOCK_N],
                                                        TensorMemoryLayout((128, BLOCK_N), col_stride=1))
    # seed S with something finite
    init = gl.full([128, SUB_COLS], 0.5, gl.float32, sub_layout)
    for i in gl.static_range(NSUB):
        s_tmem.slice(i * SUB_COLS, SUB_COLS).store(init)

    m_i = gl.full([128], -float("inf"), gl.float32, row_layout)
    l_i = gl.zeros([128], gl.float32, row_layout)
    for it in range(N_ITER):
        row_max = gl.full([128], -float("inf"), gl.float32, row_layout)
        if VARIANT == 0:
            # plain: exp2 fp32 on the full tile (v4 MATH=0)
            qk0 = s_tmem.slice(0, SUB_COLS).load(sub_layout)
            qk1 = s_tmem.slice(SUB_COLS, SUB_COLS).load(sub_layout)
            row_max = gl.maximum(gl.max(qk0, 1), gl.max(qk1, 1))
            m_ij = gl.maximum(m_i, row_max * qk_scale)
            alpha = gl.exp2(m_i - m_ij)
            p0 = gl.exp2(qk0 * qk_scale - m_ij[:, None])
            p1 = gl.exp2(qk1 * qk_scale - m_ij[:, None])
            p_buf.slice(0, SUB_COLS).store(p0.to(gl.bfloat16))
            p_buf.slice(SUB_COLS, SUB_COLS).store(p1.to(gl.bfloat16))
            l_i = l_i * alpha + (gl.sum(p0, 1) + gl.sum(p1, 1))
            m_i = m_ij
        elif VARIANT == 1:
            # load + max only (no exp, no store): cost of tmem->reg + reductions
            qk0 = s_tmem.slice(0, SUB_COLS).load(sub_layout)
            qk1 = s_tmem.slice(SUB_COLS, SUB_COLS).load(sub_layout)
            row_max = gl.maximum(gl.max(qk0, 1), gl.max(qk1, 1))
            m_i = gl.maximum(m_i, row_max * qk_scale)
            l_i = l_i + row_max
        elif VARIANT == 2:
            # load + exp2 + store, no reductions at all
            qk0 = s_tmem.slice(0, SUB_COLS).load(sub_layout)
            qk1 = s_tmem.slice(SUB_COLS, SUB_COLS).load(sub_layout)
            p0 = gl.exp2(qk0 * qk_scale - m_i[:, None])
            p1 = gl.exp2(qk1 * qk_scale - m_i[:, None])
            p_buf.slice(0, SUB_COLS).store(p0.to(gl.bfloat16))
            p_buf.slice(SUB_COLS, SUB_COLS).store(p1.to(gl.bfloat16))
        elif VARIANT == 3:
            # load + store only (tmem round trip + cvt)
            qk0 = s_tmem.slice(0, SUB_COLS).load(sub_layout)
            qk1 = s_tmem.slice(SUB_COLS, SUB_COLS).load(sub_layout)
            p_buf.slice(0, SUB_COLS).store(qk0.to(gl.bfloat16))
            p_buf.slice(SUB_COLS, SUB_COLS).store(qk1.to(gl.bfloat16))
        elif VARIANT == 4:
            # fp16x2 exp path (v4 MATH=1 exponent) with plain fp32 FMA argument
            qk0 = s_tmem.slice(0, SUB_COLS).load(sub_layout)
            qk1 = s_tmem.slice(SUB_COLS, SUB_COLS).load(sub_layout)
            row_max = gl.maximum(gl.max(qk0, 1), gl.max(qk1, 1))
            m_ij = gl.maximum(m_i, row_max * qk_scale)
            alpha = gl.exp2(m_i - m_ij)
            a0 = (qk0 * qk_scale - m_ij[:, None]).to(gl.float16)
            a1 = (qk1 * qk_scale - m_ij[:, None]).to(gl.float16)
            p0h = _exp2_f16x2(a0)
            p1h = _exp2_f16x2(a1)
            p_buf.slice(0, SUB_COLS).store(p0h.to(gl.bfloat16))
            p_buf.slice(SUB_COLS, SUB_COLS).store(p1h.to(gl.bfloat16))
            l_i = l_i * alpha + (gl.sum(p0h.to(gl.float32), 1) + gl.sum(p1h.to(gl.float32), 1))
            m_i = m_ij
        elif VARIANT == 6:
            # load + exp2 only, result reduced (no store): pure SFU + FMA cost
            qk0 = s_tmem.slice(0, SUB_COLS).load(sub_layout)
            qk1 = s_tmem.slice(SUB_COLS, SUB_COLS).load(sub_layout)
            p0 = gl.exp2(qk0 * qk_scale - m_i[:, None])
            p1 = gl.exp2(qk1 * qk_scale - m_i[:, None])
            l_i = l_i + gl.sum(p0, 1) + gl.sum(p1, 1)
        elif VARIANT == 7:
            # load + FMA only (no exp2): how much of V6 is the SFU?
            qk0 = s_tmem.slice(0, SUB_COLS).load(sub_layout)
            qk1 = s_tmem.slice(SUB_COLS, SUB_COLS).load(sub_layout)
            p0 = qk0 * qk_scale - m_i[:, None]
            p1 = qk1 * qk_scale - m_i[:, None]
            l_i = l_i + gl.sum(p0, 1) + gl.sum(p1, 1)
        elif VARIANT == 8:
            # exp2 via ex2.approx.f16x2 on fp16 args, no store, reduce in fp16->fp32
            qk0 = s_tmem.slice(0, SUB_COLS).load(sub_layout)
            qk1 = s_tmem.slice(SUB_COLS, SUB_COLS).load(sub_layout)
            a0 = (qk0 * qk_scale - m_i[:, None]).to(gl.float16)
            a1 = (qk1 * qk_scale - m_i[:, None]).to(gl.float16)
            p0h = _exp2_f16x2(a0)
            p1h = _exp2_f16x2(a1)
            l_i = l_i + gl.sum(p0h.to(gl.float32), 1) + gl.sum(p1h.to(gl.float32), 1)
        elif VARIANT == 9:
            # raw ex2.approx.f32 via inline asm (bypass libdevice exp2 lowering)
            qk0 = s_tmem.slice(0, SUB_COLS).load(sub_layout)
            qk1 = s_tmem.slice(SUB_COLS, SUB_COLS).load(sub_layout)
            a0 = qk0 * qk_scale - m_i[:, None]
            a1 = qk1 * qk_scale - m_i[:, None]
            p0 = _ex2_f32(a0)
            p1 = _ex2_f32(a1)
            l_i = l_i + gl.sum(p0, 1) + gl.sum(p1, 1)
        elif VARIANT == 5:
            # exp2 fp32 but l via fp16 sums (halve reduction work): exactness aside
            qk0 = s_tmem.slice(0, SUB_COLS).load(sub_layout)
            qk1 = s_tmem.slice(SUB_COLS, SUB_COLS).load(sub_layout)
            row_max = gl.maximum(gl.max(qk0, 1), gl.max(qk1, 1))
            m_ij = gl.maximum(m_i, row_max * qk_scale)
            alpha = gl.exp2(m_i - m_ij)
            p0 = gl.exp2(qk0 * qk_scale - m_ij[:, None])
            p1 = gl.exp2(qk1 * qk_scale - m_ij[:, None])
            p_buf.slice(0, SUB_COLS).store(p0.to(gl.bfloat16))
            p_buf.slice(SUB_COLS, SUB_COLS).store(p1.to(gl.bfloat16))
            l_i = l_i * alpha + gl.sum(p0 + p1, 1)
            m_i = m_ij
    gl.store(out + gl.program_id(0) * 128 + gl.arange(0, 128, row_layout), l_i + m_i)


def run(variant, n_iter=1024, sub=64, nsub=2, num_warps=4):
    out = torch.empty(148 * 128, device="cuda", dtype=torch.float32)
    fn = lambda: _sm_micro[(148,)](out, 0.1, n_iter, VARIANT=variant, SUB_COLS=sub, NSUB=nsub,
                                   num_warps=num_warps, maxnreg=192)
    fn(); torch.cuda.synchronize()
    med, lo, hi = C.cuda_event_median(fn, warmup=3, iters=10, flush_l2=False)
    return med


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    N = 1024
    names = {0: "plain full softmax (MATH=0)", 1: "load+max only", 2: "load+exp2+store (no reduce)",
             3: "load+store only", 4: "fp16x2 exp2 + fp32 args", 5: "exp2 fp32, sum(p0+p1)",
             6: "load+fma+exp2+sum (no store)", 7: "load+fma+sum (no exp2)",
             8: "load+fma+f16x2 exp2+sum", 9: "load+fma+ex2.approx.f32 asm+sum"}
    print(f"per 128x128 tile cost on one SM, 4 warps, {N} iters:", flush=True)
    for v in range(10):
        try:
            ms = run(v, N)
            print(f"  V{v} {names[v]:32s}: {ms:8.3f} ms  -> {ms / N * 1e3:7.2f} us/tile  "
                  f"(x256 tiles x2 halves = {ms / N * 256 * 2 / 148 * 148 / 1e0 * 1e0:.2f} ms-equiv/CTA-row)",
                  flush=True)
        except Exception as e:
            print(f"  V{v} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    # 8-warp variant of the plain softmax: does doubling warps per half help?
    try:
        ms = run(0, N, num_warps=8)
        print(f"  V0 plain, 8 warps                : {ms:8.3f} ms  -> {ms / N * 1e3:7.2f} us/tile", flush=True)
    except Exception as e:
        print(f"  V0 8-warp FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
