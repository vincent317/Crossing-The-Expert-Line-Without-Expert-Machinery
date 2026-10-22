// FlashAttention forward for B=8 S=512 H=16 D=128 bf16, non-causal, on
// Blackwell (sm_100a) using 5th-generation tensor cores (tcgen05 / UMMA).
//
// Written from scratch against the public PTX ISA / CUTLASS instruction
// encodings, then shaped by measurement. Profiling the first working version
// showed tensor pipes only 10% active with `long_scoreboard` the dominant
// stall, and phase ablation attributed the time to (a) exposed global-load
// latency and (b) a full MMA-completion round trip on the critical path of
// every iteration. Both are latency, not throughput, problems. Hence:
//
//   - Deep pipeline: NSTAGE_KV tiles of K and V in flight, and the S-GEMM for
//     iteration it+NSTAGE_S is issued while softmax(it) runs, so tensor-core
//     completion latency is never waited on directly.
//   - Every global load is cp.async. V is stored MN-major (see off_mn), the
//     encoding validated by t5_major.cu variant 2, so it needs no transpose
//     and can stream asynchronously like K.
//   - The O accumulator lives in TMEM. Consecutive O-GEMMs into the same
//     accumulator are ordered by the tensor core, so no wait is needed to
//     accumulate; the barrier is awaited only when a rescale must read O back,
//     which lazy rescale (RESCALE_THRESH) makes rare.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define HEAD_DIM 128
#define BLOCK_M 128
#ifndef BLOCK_N
#define BLOCK_N 64
#endif
#define MMA_K 16
#define NTHREADS 128

// Pipeline depth. K/V/S are staged; P needs only enough stages to cover the
// in-flight O-GEMMs.
#ifndef NSTAGE_KV
#define NSTAGE_KV 5
#endif
#ifndef NSTAGE_S
#define NSTAGE_S 3
#endif
#ifndef NSTAGE_P
#define NSTAGE_P 2
#endif

// Lazy rescale: keep a stale row max unless the block max exceeds it by more
// than this, so most iterations need no TMEM read-modify-write of O (and no
// wait on the O barrier). P may then reach 2^RESCALE_THRESH, which bf16 and
// fp32 represent with unchanged relative precision.
#ifndef RESCALE_THRESH
#define RESCALE_THRESH 8.0f
#endif

// VLAYOUT=2: V stored MN-major, no transpose (preferred).
// VLAYOUT=1: V transposed into K-major on the way to smem (fallback).
#ifndef VLAYOUT
#define VLAYOUT 2
#endif

// ABLATE isolates the cost of each pipeline phase; non-zero values produce
// WRONG results on purpose. 1 skip rescale, 2 skip exp, 6 MMA only.
#ifndef ABLATE
#define ABLATE 0
#endif

// ------------------------------------------------------------------ PTX utils

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

// UMMA shared-memory descriptor, SWIZZLE_NONE. Offsets in 16B units.
// addr[0,14) lbo[16,30) sbo[32,46) version[46,48) layout_type[61,64)
__device__ __forceinline__ uint64_t make_desc(uint32_t addr, uint32_t lbo,
                                             uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFFu) << 0;
  d |= (uint64_t)(lbo & 0x3FFFu) << 16;
  d |= (uint64_t)(sbo & 0x3FFFu) << 32;
  d |= (uint64_t)1u << 46;  // Blackwell descriptor version
  return d;
}

// tcgen05.mma.kind::f16 instruction descriptor.
__device__ __forceinline__ uint32_t make_idesc(int n, int m, int a_major,
                                               int b_major) {
  uint32_t d = 0;
  d |= 1u << 4;                   // c_format = F32
  d |= 1u << 7;                   // a_format = BF16
  d |= 1u << 10;                  // b_format = BF16
  d |= (uint32_t)a_major << 15;   // 0 = K-major, 1 = MN-major
  d |= (uint32_t)b_major << 16;
  d |= (uint32_t)(n >> 3) << 17;
  d |= (uint32_t)(m >> 4) << 24;
  return d;
}

__device__ __forceinline__ void umma(uint32_t tmem_c, uint64_t da, uint64_t db,
                                     uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"(accumulate));
}

__device__ __forceinline__ void umma_commit(uint64_t *mbar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
      "[%0];" ::"r"(smem_u32(mbar)));
}

// Generic-proxy smem writes must be visible to the async proxy (MMA operand
// reads) before the MMA is issued.
__device__ __forceinline__ void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_16(uint32_t dst_smem, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_smem),
               "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
// Wait until at most N cp.async groups remain outstanding.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// Order tcgen05 operations across thread synchronisation (PTX requirement).
__device__ __forceinline__ void t5_fence_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}
__device__ __forceinline__ void t5_fence_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}

__device__ __forceinline__ void mbar_init(uint64_t *mbar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(smem_u32(mbar)), "r"(count));
}

__device__ __forceinline__ void mbar_wait(uint64_t *mbar, uint32_t parity) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra LAB_WAIT;\n\t"
      "}\n"
      :
      : "r"(smem_u32(mbar)), "r"(parity));
}

#define TMEM_LD32(addr, r)                                                    \
  asm volatile(                                                               \
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "                               \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"               \
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "    \
      "[%32];\n"                                                              \
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]),           \
        "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),           \
        "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]),      \
        "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),      \
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),      \
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]),      \
        "=r"(r[30]), "=r"(r[31])                                              \
      : "r"(addr))

#define TMEM_ST32(addr, r)                                                    \
  asm volatile(                                                               \
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "                        \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"               \
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};\n"   \
      :                                                                       \
      : "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]),     \
        "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]),   \
        "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]),           \
        "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]),           \
        "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]),           \
        "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]),           \
        "r"(addr))

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t *>(&t);
}

// --------------------------------------------------------- smem tile layouts
//
// Both encodings are in 16B (u128) units and were validated against a CPU
// reference (t5_gemm.cu for K-major, t5_major.cu variant 2 for MN-major).
//
// K-major (descriptor lbo=8, sbo=k_total): one u128 holds 8 consecutive values
// of the contraction axis for one MN row; MN groups of 8 are interleaved.
__device__ __host__ __forceinline__ int off_k(int mn, int k, int k_total) {
  return (mn & 7) + (mn >> 3) * k_total + (k >> 3) * 8;
}
// MN-major (descriptor lbo=mn_total, sbo=8): one u128 holds 8 consecutive MN
// values; note lbo and sbo swap roles relative to K-major.
__device__ __host__ __forceinline__ int off_mn(int mn, int k, int mn_total) {
  return (k & 7) + (k >> 3) * mn_total + (mn >> 3) * 8;
}

struct SmemLayout {
  __nv_bfloat16 q[BLOCK_M * HEAD_DIM];
  __nv_bfloat16 k[NSTAGE_KV][BLOCK_N * HEAD_DIM];
  __nv_bfloat16 v[NSTAGE_KV][BLOCK_N * HEAD_DIM];
  __nv_bfloat16 p[NSTAGE_P][BLOCK_M * BLOCK_N];
  uint32_t tmem_base[4];
  uint64_t bar_s[NSTAGE_S];
  uint64_t bar_o[NSTAGE_P];
};

// TMEM columns: O accumulator + NSTAGE_S S accumulators, rounded up to the
// power of two tcgen05.alloc requires.
constexpr int tmem_cols_alloc() {
  return (HEAD_DIM + NSTAGE_S * BLOCK_N) <= 256 ? 256 : 512;
}
#define TMEM_COLS (tmem_cols_alloc())

extern "C" size_t fmha_smem_bytes() { return sizeof(SmemLayout); }

// Q tile: (BLOCK_M x HEAD_DIM) K-major. One u128 is 8 contiguous head-dim
// values of one row, i.e. one 16B chunk of the row-major global tile.
__device__ __forceinline__ void load_q(__nv_bfloat16 *dst,
                                       const __nv_bfloat16 *src, int gstride,
                                       int tid) {
  constexpr int NU128 = BLOCK_M * HEAD_DIM / 8;
#pragma unroll
  for (int i = tid; i < NU128; i += NTHREADS) {
    int row = i / (HEAD_DIM / 8), kg = i % (HEAD_DIM / 8);
    cp_async_16(smem_u32((uint4 *)dst + off_k(row, kg * 8, HEAD_DIM)),
                src + (size_t)row * gstride + kg * 8);
  }
}

// K tile: (BLOCK_N x HEAD_DIM) K-major, B operand of S = Q @ K^T with
// b_major=0. Same shape as Q, so the same 16B chunks.
__device__ __forceinline__ void load_k(__nv_bfloat16 *dst,
                                       const __nv_bfloat16 *src, int gstride,
                                       int tid) {
  constexpr int NU128 = BLOCK_N * HEAD_DIM / 8;
#pragma unroll
  for (int i = tid; i < NU128; i += NTHREADS) {
    int row = i / (HEAD_DIM / 8), kg = i % (HEAD_DIM / 8);
    cp_async_16(smem_u32((uint4 *)dst + off_k(row, kg * 8, HEAD_DIM)),
                src + (size_t)row * gstride + kg * 8);
  }
}

// V tile: B operand of O += P @ V, contraction over keys and N = head_dim.
// VLAYOUT=2 declares b_major=1 (MN-major, head_dim contiguous), which is
// exactly V's global layout, so each u128 is again one contiguous 16B chunk
// and the whole tile streams with cp.async.
__device__ __forceinline__ void load_v(__nv_bfloat16 *dst,
                                       const __nv_bfloat16 *src, int gstride,
                                       int tid) {
#if VLAYOUT == 2
  constexpr int NU128 = BLOCK_N * HEAD_DIM / 8;
#pragma unroll
  for (int i = tid; i < NU128; i += NTHREADS) {
    int key = i / (HEAD_DIM / 8), dg = i % (HEAD_DIM / 8);
    cp_async_16(smem_u32((uint4 *)dst + off_mn(dg * 8, key, HEAD_DIM)),
                src + (size_t)key * gstride + dg * 8);
  }
#else
  // Fallback: transpose 8x8 bf16 blocks through registers into K-major V^T.
  constexpr int NBLK = (BLOCK_N / 8) * (HEAD_DIM / 8);
#pragma unroll 2
  for (int bi = tid; bi < NBLK; bi += NTHREADS) {
    const int k0 = (bi / (HEAD_DIM / 8)) * 8;
    const int d0 = (bi % (HEAD_DIM / 8)) * 8;
    __nv_bfloat16 blk[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      *(uint4 *)blk[i] = *(const uint4 *)(src + (size_t)(k0 + i) * gstride + d0);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      __nv_bfloat16 col[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) col[i] = blk[i][j];
      ((uint4 *)dst)[off_k(d0 + j, k0, BLOCK_N)] = *(const uint4 *)col;
    }
  }
#endif
}

extern "C" __global__ __launch_bounds__(NTHREADS) void fmha_fwd(
    const __nv_bfloat16 *__restrict__ Q, const __nv_bfloat16 *__restrict__ K,
    const __nv_bfloat16 *__restrict__ V, __nv_bfloat16 *__restrict__ O,
    int seqlen, int nheads, float scale_log2e) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  SmemLayout &sm = *reinterpret_cast<SmemLayout *>(smem_raw);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int m_block = blockIdx.x;
  const int bh = blockIdx.y;
  const int b = bh / nheads, h = bh % nheads;
  const int n_iter = seqlen / BLOCK_N;

  // (B, S, H, D) row-major: (b, s, h, d) at ((b*S + s)*H + h)*D + d.
  const size_t sstride = (size_t)nheads * HEAD_DIM;
  const size_t base = (size_t)b * seqlen * sstride + (size_t)h * HEAD_DIM;
  const __nv_bfloat16 *Qp = Q + base + (size_t)m_block * BLOCK_M * sstride;
  const __nv_bfloat16 *Kp = K + base;
  const __nv_bfloat16 *Vp = V + base;
  __nv_bfloat16 *Op = O + base + (size_t)m_block * BLOCK_M * sstride;

  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(smem_u32(sm.tmem_base)), "r"(TMEM_COLS));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
  }
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < NSTAGE_S; ++s) mbar_init(&sm.bar_s[s], 1);
#pragma unroll
    for (int s = 0; s < NSTAGE_P; ++s) mbar_init(&sm.bar_o[s], 1);
  }
  __syncthreads();
  t5_fence_after_sync();
  const uint32_t tmem_o = sm.tmem_base[0];
  const uint32_t tmem_s0 = tmem_o + HEAD_DIM;

  // Thread `tid` owns row `tid`: warp w covers TMEM lanes 32w..32w+31.
  const int row = warp * 32 + lane;
  const uint32_t lane_off = (uint32_t)(warp * 32) << 16;

  const uint32_t idesc_s = make_idesc(BLOCK_N, BLOCK_M, 0, 0);
  const uint32_t idesc_o = make_idesc(HEAD_DIM, BLOCK_M, 0, VLAYOUT == 2 ? 1 : 0);
  const uint32_t q_base = smem_u32(sm.q);

  auto issue_s = [&](int it) {
    if (warp == 0 && lane == 0) {
      uint32_t kb = smem_u32(sm.k[it % NSTAGE_KV]);
      uint32_t tc = tmem_s0 + (it % NSTAGE_S) * BLOCK_N;
#pragma unroll
      for (int ks = 0; ks < HEAD_DIM / MMA_K; ++ks) {
        uint64_t da = make_desc(q_base + off_k(0, ks * MMA_K, HEAD_DIM) * 16, 8,
                                HEAD_DIM);
        uint64_t db = make_desc(kb + off_k(0, ks * MMA_K, HEAD_DIM) * 16, 8,
                                HEAD_DIM);
        umma(tc, da, db, idesc_s, ks == 0 ? 0 : 1);
      }
      umma_commit(&sm.bar_s[it % NSTAGE_S]);
    }
  };
  auto issue_o = [&](int it, uint32_t acc) {
    if (warp == 0 && lane == 0) {
      uint32_t pb = smem_u32(sm.p[it % NSTAGE_P]);
      uint32_t vb = smem_u32(sm.v[it % NSTAGE_KV]);
#pragma unroll
      for (int ks = 0; ks < BLOCK_N / MMA_K; ++ks) {
        uint64_t da = make_desc(pb + off_k(0, ks * MMA_K, BLOCK_N) * 16, 8,
                                BLOCK_N);
#if VLAYOUT == 2
        uint64_t db = make_desc(vb + off_mn(0, ks * MMA_K, HEAD_DIM) * 16,
                                HEAD_DIM, 8);
#else
        uint64_t db = make_desc(vb + off_k(0, ks * MMA_K, BLOCK_N) * 16, 8,
                                BLOCK_N);
#endif
        umma(tmem_o, da, db, idesc_o, acc || ks > 0);
      }
      umma_commit(&sm.bar_o[it % NSTAGE_P]);
    }
  };
  // Stage `it % NSTAGE_KV` of K and V, as one cp.async group.
  auto load_kv = [&](int it) {
    const int st = it % NSTAGE_KV;
    load_k(sm.k[st], Kp + (size_t)it * BLOCK_N * sstride, sstride, tid);
    load_v(sm.v[st], Vp + (size_t)it * BLOCK_N * sstride, sstride, tid);
    cp_async_commit();
  };

  // Prologue: Q, then the first NSTAGE_KV K/V tiles, each its own group.
  load_q(sm.q, Qp, sstride, tid);
  cp_async_commit();
#pragma unroll
  // A V stage may not be overwritten until the O-GEMM that reads it has
  // completed, and completion is only guaranteed up to wait_o(it-NSTAGE_P+1),
  // so a prefetch issued at iteration `it` must target a stage whose previous
  // tile is at least NSTAGE_P iterations old: PREFETCH_D = NSTAGE_KV-NSTAGE_P.
  // S_AHEAD is then bounded by PREFETCH_D (the K tile must already be in
  // flight) and by NSTAGE_S (the early S-GEMM must not target the stage being
  // read this iteration).
  constexpr int PREFETCH_D = NSTAGE_KV - NSTAGE_P;
  constexpr int S_AHEAD = (PREFETCH_D - 1) < (NSTAGE_S - 1) ? (PREFETCH_D - 1)
                                                            : (NSTAGE_S - 1);
  for (int s = 0; s < PREFETCH_D; ++s)
    if (s < n_iter) load_kv(s);

  cp_async_wait_group<PREFETCH_D - S_AHEAD>();
  fence_async_smem();
  __syncthreads();
  t5_fence_after_sync();
#pragma unroll
  for (int s = 0; s < S_AHEAD; ++s)
    if (s < n_iter) issue_s(s);

  float row_max = -INFINITY, row_sum = 0.f;
  uint32_t par_s[NSTAGE_S] = {};
  int o_issued = 0;  // O-GEMMs issued
  int o_waited = 0;  // O-GEMMs whose completion barrier has been consumed

  // Wait until O-GEMMs [0, target) have completed. Each barrier phase is
  // consumed exactly once, so its parity follows from the wait index; a single
  // monotone counter keeps the rescale path and the P-reuse path consistent.
  auto wait_o = [&](int target) {
    while (o_waited < target) {
      if (tid == 0)
        mbar_wait(&sm.bar_o[o_waited % NSTAGE_P],
                  (uint32_t)((o_waited / NSTAGE_P) & 1));
      ++o_waited;
    }
    __syncthreads();
  };

  for (int it = 0; it < n_iter; ++it) {
    // S for this iteration.
    if (tid == 0) mbar_wait(&sm.bar_s[it % NSTAGE_S], par_s[it % NSTAGE_S]);
    par_s[it % NSTAGE_S] ^= 1;
    __syncthreads();
    t5_fence_after_sync();

    uint32_t sreg[BLOCK_N];
    const uint32_t s_addr = tmem_s0 + (it % NSTAGE_S) * BLOCK_N + lane_off;
#if ABLATE != 6
#pragma unroll
    for (int c = 0; c < BLOCK_N / 32; ++c) {
      uint32_t *rp = sreg + c * 32;
      TMEM_LD32(s_addr + c * 32, rp);
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
    t5_fence_before_sync();
#else
#pragma unroll
    for (int j = 0; j < BLOCK_N; ++j) sreg[j] = 0x3f800000u;
#endif

    // The S stage just read is free, so the S-GEMM S_AHEAD iterations ahead may
    // be issued; its K tile was prefetched several iterations ago, so this
    // waits on no memory. (The next K/V prefetch is issued further down, once
    // the O-GEMM that frees its stage has been awaited.)
    const int s_next = it + S_AHEAD;
    if (s_next < n_iter) {
      cp_async_wait_group<PREFETCH_D - S_AHEAD>();
      fence_async_smem();
    }
    __syncthreads();
    t5_fence_after_sync();
    if (s_next < n_iter) issue_s(s_next);

    // Online softmax, base 2.
    float bm = -INFINITY;
#pragma unroll
    for (int j = 0; j < BLOCK_N; ++j) {
      float s = __int_as_float(sreg[j]) * scale_log2e;
      sreg[j] = __float_as_int(s);
      bm = fmaxf(bm, s);
    }
    float m, corr;
    if (row_max == -INFINITY) {
      m = bm; corr = 0.f;
    } else if (bm > row_max + RESCALE_THRESH) {
      m = bm; corr = exp2f(row_max - bm);
    } else {
      m = row_max; corr = 1.0f;
    }
    row_max = m;

    float sum = 0.f;
    __nv_bfloat162 pbf[BLOCK_N / 2];
#if ABLATE == 2 || ABLATE == 6
#pragma unroll
    for (int j = 0; j < BLOCK_N; j += 2) {
      float e0 = __int_as_float(sreg[j]), e1 = __int_as_float(sreg[j + 1]);
      sum += e0 + e1;
      pbf[j / 2] = __floats2bfloat162_rn(e0, e1);
    }
#else
#pragma unroll
    for (int j = 0; j < BLOCK_N; j += 2) {
      float e0 = exp2f(__int_as_float(sreg[j]) - m);
      float e1 = exp2f(__int_as_float(sreg[j + 1]) - m);
      sum += e0 + e1;
      pbf[j / 2] = __floats2bfloat162_rn(e0, e1);
    }
#endif
    row_sum = row_sum * corr + sum;

    // A rescale must read O back from TMEM, so it -- and only it -- has to
    // wait for the most recent O-GEMM. __syncthreads_count gives the CTA-wide
    // answer so the whole CTA agrees on whether to wait.
#if ABLATE == 1 || ABLATE == 6
    const bool need_rescale = false;
#else
    const bool need_rescale =
        __syncthreads_count(corr != 1.0f && it > 0) != 0;
#endif
    if (need_rescale) {
      wait_o(o_issued);  // the accumulator must be settled before reading it
      t5_fence_after_sync();
#pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; ++c) {
        uint32_t oreg[32];
        TMEM_LD32(tmem_o + lane_off + c * 32, oreg);
        asm volatile("tcgen05.wait::ld.sync.aligned;\n");
#pragma unroll
        for (int j = 0; j < 32; ++j)
          oreg[j] = __float_as_int(__int_as_float(oreg[j]) * corr);
        TMEM_ST32(tmem_o + lane_off + c * 32, oreg);
        asm volatile("tcgen05.wait::st.sync.aligned;\n");
      }
      t5_fence_before_sync();
    }

    // P[it % NSTAGE_P] was last read by O-GEMM(it - NSTAGE_P).
    if (it >= NSTAGE_P) wait_o(it - NSTAGE_P + 1);

    // That wait also frees K/V stage (it+PREFETCH_D) % NSTAGE_KV, whose tile
    // was it+PREFETCH_D-NSTAGE_KV = it-NSTAGE_P: its O-GEMM has completed.
    const int kv_next = it + PREFETCH_D;
    if (kv_next < n_iter) load_kv(kv_next);

#if ABLATE != 6
    {
      uint4 *pdst = (uint4 *)sm.p[it % NSTAGE_P];
#pragma unroll
      for (int j = 0; j < BLOCK_N; j += 8) {
        uint4 val;
        val.x = *(const uint32_t *)&pbf[j / 2];
        val.y = *(const uint32_t *)&pbf[j / 2 + 1];
        val.z = *(const uint32_t *)&pbf[j / 2 + 2];
        val.w = *(const uint32_t *)&pbf[j / 2 + 3];
        pdst[off_k(row, j, BLOCK_N)] = val;
      }
    }
#endif
    fence_async_smem();
    __syncthreads();
    t5_fence_after_sync();
    issue_o(it, it == 0 ? 0 : 1);
    ++o_issued;
  }

  // Final O.
  wait_o(n_iter);
  t5_fence_after_sync();
  const float inv = 1.0f / row_sum;
  __nv_bfloat16 *orow = Op + (size_t)row * sstride;
#pragma unroll
  for (int c = 0; c < HEAD_DIM / 32; ++c) {
    uint32_t oreg[32];
    TMEM_LD32(tmem_o + lane_off + c * 32, oreg);
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
#pragma unroll
    for (int j = 0; j < 32; j += 8) {
      uint4 v;
      v.x = pack_bf16x2(__int_as_float(oreg[j + 0]) * inv,
                        __int_as_float(oreg[j + 1]) * inv);
      v.y = pack_bf16x2(__int_as_float(oreg[j + 2]) * inv,
                        __int_as_float(oreg[j + 3]) * inv);
      v.z = pack_bf16x2(__int_as_float(oreg[j + 4]) * inv,
                        __int_as_float(oreg[j + 5]) * inv);
      v.w = pack_bf16x2(__int_as_float(oreg[j + 6]) * inv,
                        __int_as_float(oreg[j + 7]) * inv);
      *(uint4 *)(orow + c * 32 + j) = v;
    }
  }
  t5_fence_before_sync();
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :
                 : "r"(tmem_o), "r"(TMEM_COLS));
  }
}
