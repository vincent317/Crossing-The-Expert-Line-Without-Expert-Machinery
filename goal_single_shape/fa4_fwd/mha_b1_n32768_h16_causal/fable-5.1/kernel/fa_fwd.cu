// FlashAttention forward, varlen packed causal MHA, bf16, head_dim=128, on SM100 (B200).
// Written from scratch against the PTX ISA / UMMA descriptor layout.
//
// Tiling: one CTA per (doc, head, q_tile). QTILE=128 queries, KTILE=128 keys, D=128.
//   S = Q K^T   via tcgen05.mma kind::f16 (A=Q K-major smem, B=K K-major smem) -> tmem fp32 128x128
//   O += P V    via tcgen05.mma kind::f16 (A=P K-major smem, B=V MN-major smem) -> tmem fp32 128x128
// Warps: 0..3 softmax (each thread owns one S/O row), warp 4 = TMA loader + MMA issuer.
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

#define QTILE 128
#define KTILE 128
#define HDIM 128
#define NSOFT_WARPS 4
#define NTHREADS (32 * (NSOFT_WARPS + 1))
#define LOG2E 1.4426950408889634f
#ifdef FA_DEBUG
#define DBG(...) do { if (blockIdx.x == 0) printf(__VA_ARGS__); } while (0)
#else
#define DBG(...) do {} while (0)
#endif

// tmem column layout
#define TM_S0 0
#define TM_S1 128
#define TM_O 256
#define TM_COLS 512

// smem byte layout
#define SM_Q 0
#define SM_K 32768                 // 2 stages x 32768
#define SM_V (32768 + 65536)       // 2 stages x 32768
#define SM_P (32768 + 65536 + 65536)  // 1 x 32768
#define SM_BAR (32768 + 65536 + 65536 + 32768)
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
};

// ------------------------------------------------------------------ kernel
__global__ void __launch_bounds__(NTHREADS) fa_fwd_kernel(
    const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK,
    const __grid_constant__ CUtensorMap tmV, const WorkItem *__restrict__ work,
    __nv_bfloat16 *__restrict__ out, int H, float scale) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t *sm = (uint8_t *)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t *sQ = sm + SM_Q;
  uint8_t *sK = sm + SM_K;
  uint8_t *sV = sm + SM_V;
  uint8_t *sP = sm + SM_P;
  uint64_t *bar = (uint64_t *)(sm + SM_BAR);
  uint64_t *b_q = bar + 0;         // Q tile arrival
  uint64_t *b_kv = bar + 2;        // [2] K/V stage full
  uint64_t *b_sready = bar + 4;    // [2] S tmem ready
  uint64_t *b_sfree = bar + 6;     // [2] softmax done reading S
  uint64_t *b_pready = bar + 8;    // P in smem ready
  uint64_t *b_pv = bar + 10;       // [2] PV(kb) complete
  uint32_t *tslot = (uint32_t *)(bar + 16);

  const WorkItem w = work[blockIdx.x];
  const int nk = w.qi + 1;  // causal: k tiles 0..qi
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int col_q = w.head * HDIM;

  if (tid == 0) {
    mbar_init(b_q, 1);
    for (int i = 0; i < 2; ++i) { mbar_init(b_kv + i, 1); mbar_init(b_sready + i, 1); mbar_init(b_sfree + i, NSOFT_WARPS); mbar_init(b_pv + i, 1); }
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

  if (tid == 0) {
    mbar_arrive_expect_tx(b_q, 32768);
    tma_load_2d(sQ, &tmQ, b_q, col_q, w.q_row);
    tma_load_2d(sQ + 16384, &tmQ, b_q, col_q + 64, w.q_row);
  }

  // ---------------------------------------------------------------- MMA + loader warp
  if (warp == NSOFT_WARPS) {
    const uint32_t idesc_qk = make_idesc(QTILE, KTILE, 0);
    const uint32_t idesc_pv = make_idesc(QTILE, HDIM, 1);
    if (lane == 0) {
      // prologue: load K/V tile 0
      mbar_arrive_expect_tx(b_kv, 65536);
      tma_load_2d(sK, &tmK, b_kv, col_q, w.k_row);
      tma_load_2d(sK + 16384, &tmK, b_kv, col_q + 64, w.k_row);
      tma_load_2d(sV, &tmV, b_kv, col_q, w.k_row);
      tma_load_2d(sV + 16384, &tmV, b_kv, col_q + 64, w.k_row);
      mbar_wait(b_q, 0);
      DBG("[mma] Q landed\n");
      int ph_kv[2] = {0, 0}, ph_sfree[2] = {0, 0}, ph_pv[2] = {0, 0}, ph_p = 0;
      for (int kb = 0; kb < nk; ++kb) {
        const int st = kb & 1;
        // prefetch next stage
        if (kb + 1 < nk) {
          const int st2 = st ^ 1;
          if (kb >= 1) { mbar_wait(b_pv + st2, ph_pv[st2] & 1); ph_pv[st2]++; }
          const int krow = w.k_row + (kb + 1) * KTILE;
          mbar_arrive_expect_tx(b_kv + st2, 65536);
          tma_load_2d(sK + st2 * 32768, &tmK, b_kv + st2, col_q, krow);
          tma_load_2d(sK + st2 * 32768 + 16384, &tmK, b_kv + st2, col_q + 64, krow);
          tma_load_2d(sV + st2 * 32768, &tmV, b_kv + st2, col_q, krow);
          tma_load_2d(sV + st2 * 32768 + 16384, &tmV, b_kv + st2, col_q + 64, krow);
        }
        mbar_wait(b_kv + st, ph_kv[st] & 1); ph_kv[st]++;
        DBG("[mma] kb=%d KV stage %d landed\n", kb, st);
        const int sb = kb & 1;
        if (kb >= 2) { mbar_wait(b_sfree + sb, ph_sfree[sb] & 1); ph_sfree[sb]++; }
        // S = Q K^T : A=Q (K-major), B=K (K-major, rows are keys)
        const uint32_t tS = tmem + (sb ? TM_S1 : TM_S0);
        for (int kk = 0; kk < 8; ++kk) {
          uint64_t da = make_smem_desc(smem_u32(sQ) + (kk / 4) * 16384 + (kk % 4) * 32, 16, 1024);
          uint64_t db = make_smem_desc(smem_u32(sK + st * 32768) + (kk / 4) * 16384 + (kk % 4) * 32, 16, 1024);
          umma_bf16(tS, da, db, idesc_qk, kk > 0);
        }
        tcgen05_commit(b_sready + sb);
        DBG("[mma] kb=%d QK issued -> S%d\n", kb, sb);
        // O += P V : A=P (K-major over keys), B=V (MN-major: dims contiguous)
        mbar_wait(b_pready, ph_p & 1); ph_p++;
        DBG("[mma] kb=%d P ready\n", kb);
        for (int kk = 0; kk < 8; ++kk) {
          uint64_t da = make_smem_desc(smem_u32(sP) + (kk / 4) * 16384 + (kk % 4) * 32, 16, 1024);
          uint64_t db = make_smem_desc(smem_u32(sV + st * 32768) + kk * 2048, 16384, 1024);
          umma_bf16(tmem + TM_O, da, db, idesc_pv, (kb > 0) || (kk > 0));
        }
        tcgen05_commit(b_pv + st);
        DBG("[mma] kb=%d PV issued\n", kb);
      }
      DBG("[mma] loop done\n");
    }
    return;
  }

  // ---------------------------------------------------------------- softmax warps
  const int row = warp * 32 + lane;           // 0..127, this thread's q row in tile
  const uint32_t tS_lane = (uint32_t)(warp * 32) << 16;
  float m_run = -INFINITY, l_run = 0.f;
  int ph_sready[2] = {0, 0}, ph_pv[2] = {0, 0};

  for (int kb = 0; kb < nk; ++kb) {
    const int sb = kb & 1;
    const uint32_t tS = tmem + (sb ? TM_S1 : TM_S0) + tS_lane;
    mbar_wait_warp(b_sready + sb, ph_sready[sb] & 1); ph_sready[sb]++;
    if (tid == 0) DBG("[smx] kb=%d S%d ready\n", kb, sb);
    tc_fence_after();
    float s[KTILE];
    {
      uint32_t r[32];
      for (int c = 0; c < 4; ++c) {
        tmem_ld32(tS + 32 * c, r);
        tc_wait_ld();
#pragma unroll
        for (int i = 0; i < 32; ++i) s[32 * c + i] = __uint_as_float(r[i]);
      }
    }
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(b_sfree + sb);
    if (tid == 0) DBG("[smx] kb=%d S loaded s[0]=%f s[1]=%f s[127]=%f\n", kb, s[0], s[1], s[127]);

    const bool diag = (kb == w.qi);
    float m_blk = -INFINITY;
#pragma unroll
    for (int i = 0; i < KTILE; ++i) {
      s[i] = diag && (i > row) ? -INFINITY : s[i] * scale;
      m_blk = fmaxf(m_blk, s[i]);
    }
    const float m_new = fmaxf(m_run, m_blk);
    const float alpha = (kb == 0 || m_new == m_run) ? 1.f : ex2((m_run - m_new) * LOG2E);
    float l_blk = 0.f;
#pragma unroll
    for (int i = 0; i < KTILE; ++i) {
      s[i] = ex2((s[i] - m_new) * LOG2E);
      l_blk += s[i];
    }
    l_run = l_run * alpha + l_blk;
    m_run = m_new;

    // wait until PV(kb-1) consumed sP and O is settled, then publish P(kb)
    if (kb >= 1) { mbar_wait_warp(b_pv + ((kb - 1) & 1), ph_pv[(kb - 1) & 1] & 1); ph_pv[(kb - 1) & 1]++; }
    {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        uint8_t *chunk = sP + c * 16384 + row * 128;
#pragma unroll
        for (int u = 0; u < 8; ++u) {  // 8 x 16B = 64 keys per chunk (SW128: unit ^= row%8)
          const int phys = u ^ (row & 7);
          uint32_t p0 = pack_bf16x2(s[c * 64 + u * 8 + 0], s[c * 64 + u * 8 + 1]);
          uint32_t p1 = pack_bf16x2(s[c * 64 + u * 8 + 2], s[c * 64 + u * 8 + 3]);
          uint32_t p2 = pack_bf16x2(s[c * 64 + u * 8 + 4], s[c * 64 + u * 8 + 5]);
          uint32_t p3 = pack_bf16x2(s[c * 64 + u * 8 + 6], s[c * 64 + u * 8 + 7]);
          *(uint4 *)(chunk + phys * 16) = make_uint4(p0, p1, p2, p3);
        }
      }
    }
    // rescale O in tmem when the running max moved
    // warp-uniform branch: tcgen05.ld/st are .aligned (whole-warp) instructions
    if (kb > 0 && __any_sync(0xffffffffu, alpha != 1.f)) {
      for (int c = 0; c < 4; ++c) {
        uint32_t r[32];
        tmem_ld32(tmem + TM_O + tS_lane + 32 * c, r);
        tc_wait_ld();
#pragma unroll
        for (int i = 0; i < 32; ++i) r[i] = __float_as_uint(__uint_as_float(r[i]) * alpha);
        tmem_st32(tmem + TM_O + tS_lane + 32 * c, r);
      }
      tc_wait_st();
    }
    fence_proxy_async_shared();
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(b_pready);
    if (tid == 0) DBG("[smx] kb=%d P published m=%f l=%f alpha=%f\n", kb, m_run, l_run, alpha);
  }

  // ---------------------------------------------------------------- epilogue
  mbar_wait_warp(b_pv + ((nk - 1) & 1), ph_pv[(nk - 1) & 1] & 1);
  if (tid == 0) DBG("[smx] final PV done\n");
  tc_fence_after();
  const float inv = l_run > 0.f ? 1.f / l_run : 0.f;
  const int grow = w.q_row + row;
  const bool valid = (w.qi * QTILE + row) < w.L;
  for (int c = 0; c < 4; ++c) {
    uint32_t r[32];
    tmem_ld32(tmem + TM_O + tS_lane + 32 * c, r);
    tc_wait_ld();
    if (valid) {
      __nv_bfloat16 *dst = out + (size_t)grow * H * HDIM + col_q + 32 * c;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        uint32_t packed = pack_bf16x2(__uint_as_float(r[2 * i]) * inv, __uint_as_float(r[2 * i + 1]) * inv);
        *(uint32_t *)(dst + 2 * i) = packed;
      }
    }
  }
  tc_fence_before();
  asm volatile("bar.sync 1, 128;" ::: "memory");  // softmax warps only (MMA warp has exited)
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

// (T, H*D) bf16 row-major view; box = 64 cols x 128 rows, SW128.
static int make_tmap(CUtensorMap *m, const void *ptr, uint64_t rows, uint64_t cols) {
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstride[1] = {cols * 2};
  cuuint32_t box[2] = {64, 128};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)ptr, gdim, gstride, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int)r;
}

extern "C" int fa_fwd_launch(const void *q, const void *k, const void *v, void *out, const WorkItem *work_dev,
                             int n_work, int T, int H, float scale, cudaStream_t stream) {
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(fa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL + 1024);
    if (e != cudaSuccess) return 1000 + (int)e;
    attr_set = true;
  }
  CUtensorMap tq, tk, tv;
  int r;
  if ((r = make_tmap(&tq, q, T, (uint64_t)H * HDIM))) return 2000 + r;
  if ((r = make_tmap(&tk, k, T, (uint64_t)H * HDIM))) return 2000 + r;
  if ((r = make_tmap(&tv, v, T, (uint64_t)H * HDIM))) return 2000 + r;
  fa_fwd_kernel<<<n_work, NTHREADS, SM_TOTAL + 1024, stream>>>(tq, tk, tv, work_dev, (__nv_bfloat16 *)out, H, scale);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 3000 + (int)e;
}

extern "C" int fa_fwd_smem_bytes() { return SM_TOTAL + 1024; }
