// FlashAttention forward, varlen packed causal MHA, bf16, head_dim=128, on SM100 (B200).
// Written from scratch against the PTX ISA / UMMA descriptor layout.
//
// v2 tiling: one CTA per (doc, head, q_tile). QTILE=256 queries (2 x M=128), KTILE=64 keys, D=128.
//   S = Q K^T   via tcgen05.mma kind::f16 (A=Q K-major smem, B=K K-major smem) -> tmem fp32 128x128
//   O += P V    via tcgen05.mma kind::f16 (A=P K-major smem, B=V MN-major smem) -> tmem fp32 128x128
// Warps: 0..3 softmax (each thread owns one S/O row), warp 4 = TMA loader + MMA issuer.
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

#define QTILE 256      // 2 halves of 128 rows (one UMMA M=128 each)
#define KTILE 64
#define HDIM 128
#define NSOFT_WARPS 8  // warps 0-3: half 0, warps 4-7: half 1
#define NTHREADS (32 * (NSOFT_WARPS + 1))
#define LOG2E 1.4426950408889634f
#ifdef FA_DEBUG
#define DBG(...) do { if (blockIdx.x == 0) printf(__VA_ARGS__); } while (0)
#else
#define DBG(...) do {} while (0)
#endif
#define KSTAGES 3
#ifndef FA_MODE
#define FA_MODE 0
#endif
#if FA_MODE == 9
#define PROF_N 16
#define PROF_BEGIN() unsigned long long _pt0 = clock64()
#define PROF_END(slot) do { unsigned long long _pt1 = clock64(); prof_acc[slot] += _pt1 - _pt0; _pt0 = _pt1; } while (0)
#else
#define PROF_BEGIN() do {} while (0)
#define PROF_END(slot) do {} while (0)
#endif
#define RESCALE_THRESH 8.0f   // log2 units; P <= 2^8 stays exact in bf16 range
#define VSTAGES 3

// tmem column layout: S[buf][half] = buf*128 + half*64 ; O[half] = 256 + half*128
#define TM_S(buf, half) ((buf) * 128 + (half) * 64)
#define TM_O(half) (256 + (half) * 128)
#define TM_COLS 512

// smem byte layout
#define Q_BYTES 65536                    // 256 rows x 128 dims bf16: [half][chunk] 4 x 16KB
#define K_BYTES 16384                    // 64 keys x 128 dims: [chunk] 2 x 8KB
#define V_BYTES 16384
#define P_BYTES 16384                    // 128 rows x 64 keys bf16 (one SW128 atom wide)
#define SM_Q 0
#define SM_K (SM_Q + Q_BYTES)
#define SM_V (SM_K + KSTAGES * K_BYTES)
#define SM_P (SM_V + VSTAGES * V_BYTES)  // 2 halves
#define SM_BAR (SM_P + 2 * P_BYTES)
#define SM_TOTAL (SM_BAR + 1024)

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t *bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t *bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ bool mbar_try_wait(uint32_t addr, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      " selp.b32 %0, 1, 0, p;\n"
      "}\n" : "=r"(ok) : "r"(addr), "r"(parity) : "memory");
  return ok != 0;
}
// single-thread wait (MMA/loader lane)
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t parity) {
  const uint32_t addr = smem_u32(bar);
  while (!mbar_try_wait(addr, parity)) {}
}
// whole-warp wait; reconverges before the warp-collective tcgen05 ops that follow
__device__ __forceinline__ void mbar_wait_warp(uint64_t *bar, uint32_t parity) {
  const uint32_t addr = smem_u32(bar);
  while (!mbar_try_wait(addr, parity)) {}
  __syncwarp();
}
__device__ __forceinline__ void tma_load_2d(void *dst, const void *tmap, uint64_t *bar, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      ::"r"(smem_u32(dst)), "l"((uint64_t)tmap), "r"(smem_u32(bar)), "r"(x), "r"(y) : "memory");
}
__device__ __forceinline__ void tcgen05_alloc(uint32_t *dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)), "r"(ncols));
}
__device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}
__device__ __forceinline__ void tcgen05_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::); }
__device__ __forceinline__ void tcgen05_commit(uint64_t *bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void umma_bf16(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
      ::"r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::); }
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::); }
__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

#define TM_R32_OUT(r) \
  "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), \
  "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), \
  "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), \
  "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
#define TM_R32_IN(r) \
  "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), \
  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), \
  "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), \
  "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31])
#define TM_SLOTS32 \
  "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
  "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"

__device__ __forceinline__ void tmem_ld32(uint32_t taddr, uint32_t *r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 " TM_SLOTS32 ", [%32];" : TM_R32_OUT(r) : "r"(taddr));
}
// NB: operand %0 is the address, data registers are %1..%32.
#define TM_SLOTS32_FROM1 \
  "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16," \
  "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32}"
__device__ __forceinline__ void tmem_st32(uint32_t taddr, const uint32_t *r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], " TM_SLOTS32_FROM1 ";" ::"r"(taddr), TM_R32_IN(r));
}

__device__ __forceinline__ uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFFu) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFFu) << 32;
  d |= (uint64_t)1 << 46;  // version = 1 (SM100)
  d |= (uint64_t)2 << 61;  // SWIZZLE_128B
  return d;
}
__host__ __device__ constexpr uint32_t make_idesc(int M, int N, int b_mn_major) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)b_mn_major << 16) |
         ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
__device__ __forceinline__ float ex2(float x) {
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t d;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}


struct WorkItem {
  int q_row;    // absolute packed row of this q tile start
  int k_row;    // absolute packed row of the doc start
  int L;        // doc length
  int head;     // head index
  int qi;       // q tile index within doc
  int nk;       // number of 64-key blocks to process
};

// ------------------------------------------------------------------ kernel
__global__ void __launch_bounds__(NTHREADS) fa_fwd_kernel(
    const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK,
    const __grid_constant__ CUtensorMap tmV, const WorkItem *__restrict__ work,
    __nv_bfloat16 *__restrict__ out, int H, float scale, unsigned long long *__restrict__ prof) {
  extern __shared__ uint8_t smem_raw[];
#if FA_MODE == 9
  unsigned long long prof_acc[PROF_N];
  for (int i = 0; i < PROF_N; ++i) prof_acc[i] = 0;
  const unsigned long long prof_t_start = clock64();
#endif
  uint8_t *sm = (uint8_t *)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t *sQ = sm + SM_Q;
  uint8_t *sK = sm + SM_K;
  uint8_t *sV = sm + SM_V;
  uint8_t *sP = sm + SM_P;
  uint64_t *bar = (uint64_t *)(sm + SM_BAR);
  uint64_t *b_q = bar + 0;
  uint64_t *b_k = bar + 1;         // [KSTAGES]
  uint64_t *b_v = bar + 4;         // [VSTAGES]
  uint64_t *b_sready = bar + 8;    // [2]
  uint64_t *b_sfree = bar + 10;    // [2]
  uint64_t *b_pready = bar + 12;
  uint64_t *b_pv = bar + 14;       // [2]
  uint32_t *tslot = (uint32_t *)(bar + 16);

  const WorkItem w = work[blockIdx.x];
  const int nk = w.nk;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int col_q = w.head * HDIM;

  if (tid == 0) {
    mbar_init(b_q, 1);
    for (int i = 0; i < KSTAGES; ++i) mbar_init(b_k + i, 1);
    for (int i = 0; i < VSTAGES; ++i) mbar_init(b_v + i, 1);
    for (int i = 0; i < 2; ++i) { mbar_init(b_sready + i, 1); mbar_init(b_sfree + i, NSOFT_WARPS); mbar_init(b_pv + i, 1); }
    mbar_init(b_pready, NSOFT_WARPS);
    asm volatile("fence.mbarrier_init.release.cluster;" ::);
  }
  __syncthreads();
  if (warp == 0) tcgen05_alloc(tslot, TM_COLS);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  if (tid == 0) DBG("[blk0] tmem=0x%08x nk=%d qi=%d L=%d head=%d q_row=%d k_row=%d\n", tmem, nk, w.qi, w.L, w.head, w.q_row, w.k_row);

  // ---------------------------------------------------------------- MMA + loader warp
  if (warp == NSOFT_WARPS) {
    if (lane == 0) {
      const uint32_t idesc_qk = make_idesc(128, KTILE, 0);
      const uint32_t idesc_pv = make_idesc(128, HDIM, 1);
      auto load_k = [&](int kb) {
        const int st = kb % KSTAGES;
        uint8_t *dst = sK + st * K_BYTES;
        mbar_arrive_expect_tx(b_k + st, K_BYTES);
        tma_load_2d(dst, &tmK, b_k + st, col_q, w.k_row + kb * KTILE);
        tma_load_2d(dst + 8192, &tmK, b_k + st, col_q + 64, w.k_row + kb * KTILE);
      };
      auto load_v = [&](int kb) {
        const int st = kb % VSTAGES;
        uint8_t *dst = sV + st * V_BYTES;
        mbar_arrive_expect_tx(b_v + st, V_BYTES);
        tma_load_2d(dst, &tmV, b_v + st, col_q, w.k_row + kb * KTILE);
        tma_load_2d(dst + 8192, &tmV, b_v + st, col_q + 64, w.k_row + kb * KTILE);
      };
      auto issue_qk = [&](int kb) {
        const int st = kb % KSTAGES, sb = kb & 1;
        const uint32_t kbase = smem_u32(sK + st * K_BYTES);
#if FA_MODE < 3
        for (int half = 0; half < 2; ++half) {
          const uint32_t qbase = smem_u32(sQ + half * 32768);
          const uint32_t tS = tmem + TM_S(sb, half);
          for (int kk = 0; kk < 8; ++kk) {
            uint64_t da = make_smem_desc(qbase + (kk / 4) * 16384 + (kk % 4) * 32, 16, 1024);
            uint64_t db = make_smem_desc(kbase + (kk / 4) * 8192 + (kk % 4) * 32, 16, 1024);
            umma_bf16(tS, da, db, idesc_qk, kk > 0);
          }
        }
#endif
        tcgen05_commit(b_sready + sb);
      };
      auto issue_pv = [&](int kb) {
        const int st = kb % VSTAGES;
        const uint32_t vbase = smem_u32(sV + st * V_BYTES);
#if FA_MODE < 2
        for (int half = 0; half < 2; ++half) {
          const uint32_t pbase = smem_u32(sP + half * P_BYTES);
          const uint32_t tO = tmem + TM_O(half);
          for (int kk = 0; kk < 4; ++kk) {
            uint64_t da = make_smem_desc(pbase + kk * 32, 16, 1024);
            uint64_t db = make_smem_desc(vbase + kk * 2048, 8192, 1024);
            umma_bf16(tO, da, db, idesc_pv, (kb > 0) || (kk > 0));
          }
        }
#endif
        tcgen05_commit(b_pv + (kb & 1));
      };

      // prologue
      mbar_arrive_expect_tx(b_q, Q_BYTES);
      for (int half = 0; half < 2; ++half)
        for (int c = 0; c < 2; ++c)
          tma_load_2d(sQ + half * 32768 + c * 16384, &tmQ, b_q, col_q + c * 64, w.q_row + half * 128);
      for (int i = 0; i < KSTAGES && i < nk; ++i) load_k(i);
      for (int i = 0; i < VSTAGES - 1 && i < nk; ++i) load_v(i);
      int ph_k[KSTAGES] = {0, 0, 0}, ph_v[VSTAGES] = {0, 0, 0};
      int ph_sready[2] = {0, 0}, ph_sfree[2] = {0, 0}, ph_pv[2] = {0, 0}, ph_p = 0;
      PROF_BEGIN();
      mbar_wait(b_q, 0);
      mbar_wait(b_k + 0, 0); ph_k[0]++;
      PROF_END(0);  // prologue (Q + K0 wait)
      DBG("[mma] Q,K0 landed\n");
      issue_qk(0);
      PROF_END(2);  // QK issue
      for (int j = 0; j < nk; ++j) {
        // A: QK_{j+1} (overlaps softmax_j)
        if (j + 1 < nk) {
          const int st = (j + 1) % KSTAGES, sb = (j + 1) & 1;
          mbar_wait(b_k + st, ph_k[st] & 1); ph_k[st]++;
          PROF_END(3);  // K wait
          if (j + 1 >= 2) { mbar_wait(b_sfree + sb, ph_sfree[sb] & 1); ph_sfree[sb]++; }
          PROF_END(1);  // sfree wait
          issue_qk(j + 1);
          PROF_END(2);  // QK issue
          DBG("[mma] j=%d QK_%d issued\n", j, j + 1);
        }
        // B: K_{j+3} into the stage QK_j read from
        if (j + 3 < nk) {
          const int sb = j & 1;
          mbar_wait(b_sready + sb, ph_sready[sb] & 1); ph_sready[sb]++;
          PROF_END(4);  // sready wait (QK_j done)
          load_k(j + 3);
          PROF_END(5);  // TMA issue
        }
        // C: V_{j+2} into the stage PV_{j-1} read from
        if (j + 2 < nk) {
          if (j >= 1) { const int pb = (j - 1) & 1; mbar_wait(b_pv + pb, ph_pv[pb] & 1); ph_pv[pb]++; }
          PROF_END(6);  // pv wait (PV_{j-1} done)
          load_v(j + 2);
          PROF_END(5);  // TMA issue
        }
        // D: PV_j
        {
          const int st = j % VSTAGES;
          mbar_wait(b_pready, ph_p & 1); ph_p++;
          PROF_END(7);  // pready wait (softmax_j done)
          mbar_wait(b_v + st, ph_v[st] & 1); ph_v[st]++;
          PROF_END(8);  // V wait
          issue_pv(j);
          PROF_END(9);  // PV issue
          DBG("[mma] j=%d PV issued\n", j);
        }
      }
      DBG("[mma] loop done\n");
#if FA_MODE == 9
      prof_acc[15] = clock64() - prof_t_start;
      for (int i = 0; i < PROF_N; ++i) prof[blockIdx.x * 2 * PROF_N + i] = prof_acc[i];
#endif
    }
    return;
  }

  // ---------------------------------------------------------------- softmax warps
  const int half = warp / 4;
  const int hrow = (warp % 4) * 32 + lane;      // row within the half (0..127)
  const int row = half * 128 + hrow;            // row within the 256-row tile
  const uint32_t lane_base = (uint32_t)((warp % 4) * 32) << 16;
  uint8_t *myP = sP + half * P_BYTES + hrow * 128;
  const uint32_t tO = tmem + TM_O(half) + lane_base;
  float m_run = -INFINITY, l_run = 0.f;
  const float c_scale = scale * LOG2E;
  int ph_sready[2] = {0, 0}, ph_pv[2] = {0, 0};
  const int diag_start = (w.qi * QTILE) / KTILE;   // first k-block that touches the causal diagonal

  for (int kb = 0; kb < nk; ++kb) {
    const int sb = kb & 1;
    const uint32_t tS = tmem + TM_S(sb, half) + lane_base;
    PROF_BEGIN();
    mbar_wait_warp(b_sready + sb, ph_sready[sb] & 1); ph_sready[sb]++;
    PROF_END(11);  // smx: sready wait
    if (tid == 0) DBG("[smx] kb=%d S%d ready\n", kb, sb);
    tc_fence_after();
    float s[KTILE];
    {
      uint32_t r0[32], r1[32];
      tmem_ld32(tS, r0);
      tmem_ld32(tS + 32, r1);
      tc_wait_ld();
#pragma unroll
      for (int i = 0; i < 32; ++i) { s[i] = __uint_as_float(r0[i]); s[32 + i] = __uint_as_float(r1[i]); }
    }
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(b_sfree + sb);
    PROF_END(12);  // smx: tmem ld S
    if (tid == 0) DBG("[smx] kb=%d S loaded s[0]=%f s[1]=%f s[63]=%f\n", kb, s[0], s[1], s[63]);
#if FA_MODE == 1
    // ablation: no softmax math, publish P = 0 (keeps every barrier hand-off identical)
    if (kb >= 1) { const int pb = (kb - 1) & 1; mbar_wait_warp(b_pv + pb, ph_pv[pb] & 1); ph_pv[pb]++; }
#pragma unroll
    for (int u = 0; u < 8; ++u) *(uint4 *)(myP + u * 16) = make_uint4(0, 0, 0, 0);
    fence_proxy_async_shared();
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(b_pready);
    l_run += s[0];
    continue;
#endif

    // Raw (unscaled) scores; softmax_scale*log2(e) folded into the exp2 argument.
    float m_blk = -INFINITY;
    if (kb >= diag_start) {
      const int lim = w.qi * QTILE + row - kb * KTILE;  // allowed: c <= lim
#pragma unroll
      for (int i = 0; i < KTILE; ++i) {
        s[i] = (i > lim) ? -INFINITY : s[i];
        m_blk = fmaxf(m_blk, s[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < KTILE; ++i) m_blk = fmaxf(m_blk, s[i]);
    }
    // Lazy rescale: only move the running max when the block max exceeds it by more than
    // RESCALE_THRESH (log2 units); otherwise P may exceed 1 (bounded by 2^RESCALE_THRESH).
    float alpha = 1.f;
    if (m_blk > m_run + RESCALE_THRESH / c_scale) {
      alpha = (m_run == -INFINITY) ? 0.f : ex2((m_run - m_blk) * c_scale);
      m_run = m_blk;
    }
    const float mb = m_run * c_scale;
    float l_blk = 0.f;
#pragma unroll
    for (int i = 0; i < KTILE; ++i) {
      s[i] = ex2(fmaf(s[i], c_scale, -mb));
      l_blk += s[i];
    }
    l_run = fmaf(l_run, alpha, l_blk);

    PROF_END(13);  // smx: math (mask/max/exp/sum)
    if (kb >= 1) { const int pb = (kb - 1) & 1; mbar_wait_warp(b_pv + pb, ph_pv[pb] & 1); ph_pv[pb]++; }
    PROF_END(14);  // smx: pv wait
#pragma unroll
    for (int u = 0; u < 8; ++u) {  // 8 x 16B = 64 keys, SW128: unit ^= row%8
      const int phys = u ^ (hrow & 7);
      uint32_t p0 = pack_bf16x2(s[u * 8 + 0], s[u * 8 + 1]);
      uint32_t p1 = pack_bf16x2(s[u * 8 + 2], s[u * 8 + 3]);
      uint32_t p2 = pack_bf16x2(s[u * 8 + 4], s[u * 8 + 5]);
      uint32_t p3 = pack_bf16x2(s[u * 8 + 6], s[u * 8 + 7]);
      *(uint4 *)(myP + phys * 16) = make_uint4(p0, p1, p2, p3);
    }
    // warp-uniform branch: tcgen05.ld/st are .aligned (whole-warp) instructions
    if (kb > 0 && __any_sync(0xffffffffu, alpha != 1.f)) {
      for (int c = 0; c < 4; ++c) {
        uint32_t r[32];
        tmem_ld32(tO + 32 * c, r);
        tc_wait_ld();
#pragma unroll
        for (int i = 0; i < 32; ++i) r[i] = __float_as_uint(__uint_as_float(r[i]) * alpha);
        tmem_st32(tO + 32 * c, r);
      }
      tc_wait_st();
    }
    fence_proxy_async_shared();
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(b_pready);
    PROF_END(10);  // smx: P write + rescale + publish
    if (tid == 0) DBG("[smx] kb=%d P published m=%f l=%f alpha=%f\n", kb, m_run, l_run, alpha);
  }
#if FA_MODE == 9
  if (tid == 0) for (int i = 0; i < PROF_N; ++i) prof[blockIdx.x * 2 * PROF_N + PROF_N + i] = prof_acc[i];
#endif

  // ---------------------------------------------------------------- epilogue
  { const int pb = (nk - 1) & 1; mbar_wait_warp(b_pv + pb, ph_pv[pb] & 1); }
  tc_fence_after();
  const float inv = l_run > 0.f ? 1.f / l_run : 0.f;
  const int grow = w.q_row + row;
  const bool valid = (w.qi * QTILE + row) < w.L;
  for (int c = 0; c < 4; ++c) {
    uint32_t r[32];
    tmem_ld32(tO + 32 * c, r);
    tc_wait_ld();
    if (valid) {
      __nv_bfloat16 *dst = out + (size_t)grow * H * HDIM + col_q + 32 * c;
      uint32_t pk[16];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        pk[i] = pack_bf16x2(__uint_as_float(r[2 * i]) * inv, __uint_as_float(r[2 * i + 1]) * inv);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        *(uint4 *)(dst + 8 * i) = make_uint4(pk[4 * i], pk[4 * i + 1], pk[4 * i + 2], pk[4 * i + 3]);
    }
  }
  tc_fence_before();
  asm volatile("bar.sync 1, 256;" ::: "memory");  // softmax warps only
  if (warp == 0) { tcgen05_dealloc(tmem, TM_COLS); tcgen05_relinquish(); }
  if (tid == 0) DBG("[smx] dealloc done\n");
}

// ------------------------------------------------------------------ host
typedef CUresult (*PFN_encodeTiled)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                                    const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
                                    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static PFN_encodeTiled get_encode() {
  static PFN_encodeTiled fn = nullptr;
  if (!fn) {
    void *p = nullptr;
    cudaDriverEntryPointQueryResult qr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr);
    fn = (PFN_encodeTiled)p;
  }
  return fn;
}

// (T, H*D) bf16 row-major view; box = 64 cols x box_rows rows, SW128.
static int make_tmap(CUtensorMap *m, const void *ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstride[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)ptr, gdim, gstride, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int)r;
}

static unsigned long long *g_prof = nullptr;
#define PROF_BLOCKS 4096

extern "C" int fa_fwd_launch(const void *q, const void *k, const void *v, void *out, const WorkItem *work_dev,
                             int n_work, int T, int H, float scale, cudaStream_t stream) {
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(fa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL + 1024);
    if (e != cudaSuccess) return 1000 + (int)e;
    if (cudaMalloc(&g_prof, (size_t)PROF_BLOCKS * 32 * sizeof(unsigned long long)) != cudaSuccess) return 1102;
    cudaMemset(g_prof, 0, (size_t)PROF_BLOCKS * 32 * sizeof(unsigned long long));
    attr_set = true;
  }
  CUtensorMap tq, tk, tv;
  int r;
  if ((r = make_tmap(&tq, q, T, (uint64_t)H * HDIM, 128))) return 2000 + r;
  if ((r = make_tmap(&tk, k, T, (uint64_t)H * HDIM, 64))) return 2000 + r;
  if ((r = make_tmap(&tv, v, T, (uint64_t)H * HDIM, 64))) return 2000 + r;
  fa_fwd_kernel<<<n_work, NTHREADS, SM_TOTAL + 1024, stream>>>(tq, tk, tv, work_dev, (__nv_bfloat16 *)out, H, scale, g_prof);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 3000 + (int)e;
}

extern "C" int fa_fwd_prof_dump(unsigned long long *host_out, int n) {
  if (!g_prof) return -1;
  return (int)cudaMemcpy(host_out, g_prof, (size_t)n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
}
