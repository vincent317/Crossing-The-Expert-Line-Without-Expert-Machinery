// DSA sparse attention forward (decode), B200 / sm_100a — tcgen05 UMMA + TMEM.
//
// Grid = 2 x #SMs CTAs of 4 warps, exactly 2 resident per SM; the per-SM slot
// counter gives every CTA a (sm, slot) pair, which selects a 64-row chunk in
// chunk-major order (chunk i -> token i % T, rows [64(i/T), +64)), so the busy
// low chunks of all tokens land one per SM. Q (16 x 576) and the chunk's KV
// rows are fetched by TMA tensor copies with 128B swizzle straight into the
// canonical UMMA layout: [64-dim block][row][128 B]. The same K tile serves as
// K-major A (S^T = K Q^T, M=128 with rows 64..127 ignored, N=16 heads) and as
// MN-major A (O^T = V^T P^T, M=128 dims per block, K=64 rows); Q and P^T are
// K-major B operands. Accumulators live in TMEM (S^T at column 0, O^T blocks at
// 16 + 16*mb); softmax runs once on lanes 0..63 of S^T and writes P^T (bf16,
// swizzled) to smem. Each CTA publishes (m, l) and its normalised partial
// O^T (bf16, [dim][head]) and arrives on the token's monotonic counter; CTA
// (t, c) then combines slice c (dims [16c, +16) x 16 heads) once all 32 chunk
// CTAs of token t have arrived.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

constexpr int H = 16;         // query heads
constexpr int DC = 512;       // ckv head dim (also the value dim)
constexpr int DP = 64;        // kpe head dim
constexpr int DQ = DC + DP;   // 576
constexpr int TOPK = 2048;
constexpr int R = 64;                   // KV rows per CTA
constexpr int NCHUNK = TOPK / R;        // chunks per token (32)
constexpr int SLICES = 32;              // reduction slices per token (== NCHUNK)
constexpr int SLICE_DIMS = DC / SLICES; // 16 dims x 16 heads per slice
constexpr int THREADS = 128;
constexpr int NWARPS = THREADS / 32;
constexpr int CTAS_PER_SM = 2;
constexpr int KB = DQ / 64;                       // 9 blocks of 64 dims (8 ckv + 1 kpe)
constexpr int SEG_BYTES = 128;                    // one 64-element row segment
constexpr int KV_BLK_BYTES = R * SEG_BYTES;       // 8192
constexpr int Q_BLK_BYTES = H * SEG_BYTES;        // 2048
constexpr int SMEM_KV = KB * KV_BLK_BYTES;        // 73728
constexpr int SMEM_Q = KB * Q_BLK_BYTES;          // 18432
constexpr int SMEM_PT = H * SEG_BYTES;            // 2048: P^T, K-major SW128 (16 heads x 64 rows)
constexpr int OFF_KV = 0;
constexpr int OFF_Q = OFF_KV + SMEM_KV;           // 73728
constexpr int OFF_PT = OFF_Q + SMEM_Q;            // 92160
constexpr int OFF_IDX = OFF_PT + SMEM_PT;         // 94208: [2][R] candidate chunk indices
constexpr int OFF_RED = OFF_IDX + 2 * R * 4;      // 94720: [4][16] cross-warp softmax exchange
constexpr int OFF_MISC = OFF_RED + 4 * H * 4;     // 94976: slot, tmem base
constexpr int OFF_BAR = OFF_MISC + 16;            // 94992: bar_tma, bar_mma[4]
constexpr int SMEM_USED = OFF_BAR + 5 * 8;        // 95032
constexpr int SMEM_TOTAL = SMEM_USED + 1024;      // slack for manual 1 KB alignment
constexpr int TMEM_COLS = 128;                    // S^T partial j: cols [16j, +16) (4); O^T block mb: cols [64+16mb, +16)
constexpr int TMEM_COL_O = 64;
constexpr uint32_t TX_BYTES = R * KB * SEG_BYTES + H * KB * SEG_BYTES;  // 92160 (OOB rows still count)
constexpr int MAX_T = 32;
constexpr int CTR_PAD = 16;  // u64 per counter slot (128 B apart -> different L2 lines)
// UMMA instruction descriptor: f32 accumulate, bf16 A/B, N=16 (n>>3 = 2), M=128 (m>>4 = 8)
constexpr uint32_t IDESC_S = (1u << 4) | (1u << 7) | (1u << 10) | (2u << 17) | (8u << 24);  // K-major A and B
constexpr uint32_t IDESC_PV = IDESC_S | (1u << 15);                                          // MN-major A (V^T)
constexpr uint32_t IDESC_S64 = (IDESC_S & ~(31u << 24)) | (4u << 24);                        // M=64 variant

static_assert(SLICE_DIMS == 4 * NWARPS, "reduction: 4 consecutive dims per warp");
static_assert(SLICES == NCHUNK, "one reduction slice per chunk CTA");
static_assert(THREADS >= KB * (R / 4) / 2, "gather loop assumes <= 2 ops per thread per pass");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- mbarrier ----
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}

// ---- TMA tensor copies (128B swizzle is applied by the tensor map) ----
__device__ __forceinline__ void tma_tile_2d(uint32_t dst, const CUtensorMap* tm, int x, int y, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::
          "r"(dst),
      "l"(reinterpret_cast<uint64_t>(tm)), "r"(x), "r"(y), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_gather4_2d(uint32_t dst, const CUtensorMap* tm, int x, int y0, int y1, int y2,
                                               int y3, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(tm)), "r"(x), "r"(y0), "r"(y1), "r"(y2), "r"(y3), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void tma_gather4_2d_cta(uint32_t dst, const CUtensorMap* tm, int x, int y0, int y1,
                                                   int y2, int y3, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(tm)), "r"(x), "r"(y0), "r"(y1), "r"(y2), "r"(y3), "r"(bar)
      : "memory");
}

// ---- tcgen05 ----
__device__ __forceinline__ void tmem_alloc(uint32_t dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(dst_smem), "r"(ncols)
               : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tc_fence_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}
__device__ __forceinline__ void umma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar)
               : "memory");
}
// D[tmem] (+)= A[desc] * B[desc], bf16 inputs, f32 accumulate; acc == 0 overwrites D
__device__ __forceinline__ void umma_bf16(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
                                          uint32_t acc) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
      "}\n" ::"r"(d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc)
      : "memory");
}
// 32 lanes x 16 consecutive 32-bit columns; lane i of the warp reads TMEM lane (taddr.lane + i)
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t* v) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15}, [%16];\n"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
        "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// UMMA shared memory descriptor, 128B-swizzled canonical layout, 1 KB aligned atoms.
__device__ __forceinline__ uint64_t umma_desc_sw128(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4);              // start address   [0,14)
  d |= static_cast<uint64_t>((lbo_bytes >> 4) & 0x3FFFu) << 16;     // leading offset  [16,30)
  d |= static_cast<uint64_t>((sbo_bytes >> 4) & 0x3FFFu) << 32;     // stride offset   [32,46)
  d |= static_cast<uint64_t>(1) << 46;                              // version (sm100) [46,48)
  d |= static_cast<uint64_t>(2) << 61;                              // SWIZZLE_128B    [61,64)
  return d;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack_bf16x2(uint32_t u) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u);
  return __bfloat1622float2(v);
}

// ---- inter-CTA signalling (monotonic 64-bit counters) ----
__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long ld_relaxed_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_relaxed_add_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned int atom_add_u32(unsigned int* p, unsigned int v) {
  unsigned int old;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;\n" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ float2 ld_cg_f32x2(const float* p) {
  float2 v;
  asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_cg_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.cg.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

constexpr int NPROF = 16;  // per-CTA phase timestamps when profiling (see dsa_kernel.py)
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long v;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(v));
  return v;
}
__device__ __forceinline__ unsigned int smid() {
  unsigned int v;
  asm volatile("mov.u32 %0, %%smid;\n" : "=r"(v));
  return v;
}
#define PROF(k)                                                               \
  do {                                                                        \
    if (prof != nullptr && threadIdx.x == 0) prof[b * NPROF + (k)] = gtimer(); \
  } while (0)

__global__ void __launch_bounds__(THREADS, CTAS_PER_SM)
dsa_fwd_kernel(const __grid_constant__ CUtensorMap tm_k4,   // ckv  [rows][512], box {64,4}  (gather4)
               const __grid_constant__ CUtensorMap tm_p4,   // kpe  [rows][64],  box {64,4}  (gather4)
               const __grid_constant__ CUtensorMap tm_k1,   // ckv  box {64,1} (gather4 needs this box)
               const __grid_constant__ CUtensorMap tm_p1,   // kpe  box {64,1}
               const __grid_constant__ CUtensorMap tm_qn,   // q_nope [T*16][512], box {64,16}
               const __grid_constant__ CUtensorMap tm_qp,   // q_pe   [T*16][64],  box {64,16}
               const int* __restrict__ indices,             // [T][TOPK]
               float scale_log2,                            // sm_scale * log2(e)
               int T, int gather_mode, int num_sms,         // gather_mode: 1 = per-row tiles, else gather4 variants
               int slot_mode, int m64,                      // slot_mode: 0 = per-SM atomic, 1 = TMEM column; m64: S with M=64
               __nv_bfloat16* __restrict__ part_o,          // [T][NCHUNK][DC][H] normalised partials
               float* __restrict__ part_m,                  // [T][NCHUNK][H]
               float* __restrict__ part_l,                  // [T][NCHUNK][H]
               unsigned long long* __restrict__ arrive,     // [T*CTR_PAD] monotonic CTA arrivals
               unsigned int* __restrict__ smctr,            // [num_sms*32] monotonic per-SM CTA counter
               __nv_bfloat16* __restrict__ out,             // [T][H][DC]
               float* __restrict__ lse,                     // [T][H]
               unsigned long long* __restrict__ prof) {     // [T*NCHUNK][NPROF] or nullptr
  extern __shared__ unsigned char smem_raw[];
  const uint32_t raw_u32 = smem_u32(smem_raw);
  const uint32_t base_u32 = (raw_u32 + 1023u) & ~1023u;  // swizzle atoms must be 1 KB aligned
  unsigned char* smem = smem_raw + (base_u32 - raw_u32);
  int* sIdx2 = reinterpret_cast<int*>(smem + OFF_IDX);  // [2][R]: candidate chunks of slot 0 / 1
  float* sRed = reinterpret_cast<float*>(smem + OFF_RED);  // [4 warps][16] max, then [4][16] sum (reused)
  int* sMisc = reinterpret_cast<int*>(smem + OFF_MISC);
  const uint32_t bar_tma = base_u32 + OFF_BAR;
  const uint32_t bar_mma0 = base_u32 + OFF_BAR + 8;
  const uint32_t sKV = base_u32 + OFF_KV;
  const uint32_t sQ = base_u32 + OFF_Q;
  const uint32_t sPT = base_u32 + OFF_PT;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int G = gridDim.x;
  const int per_sm = G / num_sms;
  const int sm = static_cast<int>(smid());
  const int NIT = T * NCHUNK;
  const unsigned long long t_start = (prof != nullptr) ? gtimer() : 0ull;

  // TMEM first: the second CTA of this SM is only scheduled once we relinquish the alloc permit.
  if (warp == 0) {
    tmem_alloc(smem_u32(sMisc + 1), TMEM_COLS);
    tmem_relinquish();
    tc_fence_before_sync();
  }
  // per_sm CTAs land on every SM; chunk i = slot * #SMs + sm (+ k * G for the rare extra rounds),
  // chunk-major, so the low (busy) chunks of all tokens go to slot 0, one per SM. Both candidates'
  // indices are fetched before the slot is known. All owned chunks publish their arrivals before
  // any reduction waits, so the scheme cannot deadlock whatever the grid size.
  if (tid < 2 * R) {
    const int cand = tid / R;
    const int i0 = cand * num_sms + sm;
    sIdx2[tid] = (i0 < NIT) ? indices[static_cast<size_t>(i0 % T) * TOPK + (i0 / T) * R + (tid % R)] : -1;
  }
  if (tid == 0) {
    mbar_init(bar_tma, 1);
    mbar_init(bar_mma0, 1);
    mbar_fence_init();
    sMisc[0] = (slot_mode == 0) ? static_cast<int>(atom_add_u32(smctr + static_cast<size_t>(sm) * 32, 1u) % per_sm) : 0;
  }
  __syncthreads();
  tc_fence_after_sync();
  const uint32_t tmem = static_cast<uint32_t>(sMisc[1]);
  // TMEM columns are handed out bottom-up per SM: the CTA that allocated first owns column 0.
  const int slot = (slot_mode == 0) ? sMisc[0] : static_cast<int>(((tmem & 0xFFFFu) / TMEM_COLS) % per_sm);
  const int first = slot * num_sms + sm;
  int* sIdx = sIdx2 + (slot & 1) * R;  // the first owned chunk's indices are already here
  uint32_t tma_phase = 0u, mma_phase = 0u;
  // UMMA descriptors: only the start address changes between MMAs -> base + (byte offset >> 4)
  const uint64_t dA_S = umma_desc_sw128(sKV, 16, 1024);            // K-major A (K rows)
  const uint64_t dB_S = umma_desc_sw128(sQ, 16, 1024);             // K-major B (Q)
  const uint64_t dA_PV = umma_desc_sw128(sKV, KV_BLK_BYTES, 1024);  // MN-major A (V^T), LBO = dim-block stride
  const uint64_t dB_PV = umma_desc_sw128(sPT, 16, 1024);           // K-major B (P^T)
  const uint32_t idesc_s = m64 ? IDESC_S64 : IDESC_S;
  const uint32_t lane_base = static_cast<uint32_t>(warp * 32) << 16;

  constexpr int MAX_OWNED = 8;
  unsigned long long arrive_at_start[MAX_OWNED];  // per owned chunk, read before this CTA's own arrival
  int owned = 0;
  for (int i = first; i < NIT && owned < MAX_OWNED; i += G, ++owned) {
    const int t = i % T;
    const int c = i / T;
    const int b = i;  // profiling row
    unsigned long long* my_arrive = arrive + static_cast<size_t>(t) * CTR_PAD;
    __syncthreads();  // previous chunk finished with sIdx / smem
    if (owned > 0 && tid < R) sIdx[tid] = indices[static_cast<size_t>(t) * TOPK + c * R + tid];
    if (tid == 0) arrive_at_start[owned] = ld_relaxed_u64(my_arrive);
    const int nvalid = __syncthreads_count(tid < R && sIdx[tid] >= 0);
    if (prof != nullptr && tid == 0) prof[b * NPROF + 0] = t_start;
    PROF(1);

    if (nvalid == 0) {
      if (tid < H) {
        part_m[(static_cast<size_t>(t) * NCHUNK + c) * H + tid] = -INFINITY;
        part_l[(static_cast<size_t>(t) * NCHUNK + c) * H + tid] = 0.f;
      }
    } else {
      // ---- gather: Q blocks and the chunk's KV rows into the swizzled block layout ----
      if (tid == 0) mbar_arrive_expect_tx(bar_tma, TX_BYTES);
      __syncthreads();
      if (tid < KB) {
        if (tid < KB - 1) tma_tile_2d(sQ + tid * Q_BLK_BYTES, &tm_qn, tid * 64, t * H, bar_tma);
        else tma_tile_2d(sQ + tid * Q_BLK_BYTES, &tm_qp, 0, t * H, bar_tma);
      }
      if (gather_mode != 1) {
        // gather4 variants: 0 = box{64,4} map + shared::cluster (illegal on B200), 2 = box{64,1} + cluster,
        // 3 = box{64,4} + shared::cta (illegal), 4 = box{64,1} + shared::cta
        const bool box1 = (gather_mode == 2) || (gather_mode == 4);
        const bool cta = (gather_mode >= 3);
        const CUtensorMap* mk = box1 ? &tm_k1 : &tm_k4;
        const CUtensorMap* mp = box1 ? &tm_p1 : &tm_p4;
        for (int o = tid; o < KB * (R / 4); o += THREADS) {
          const int kb = o / (R / 4), g = o % (R / 4);
          const int* ip = sIdx + 4 * g;  // index -1 is out of bounds -> TMA zero-fills the row
          const uint32_t dst = sKV + kb * KV_BLK_BYTES + g * 4 * SEG_BYTES;
          const CUtensorMap* tm = (kb < KB - 1) ? mk : mp;
          const int x = (kb < KB - 1) ? kb * 64 : 0;
          if (cta) tma_gather4_2d_cta(dst, tm, x, ip[0], ip[1], ip[2], ip[3], bar_tma);
          else tma_gather4_2d(dst, tm, x, ip[0], ip[1], ip[2], ip[3], bar_tma);
        }
      } else {
        for (int o = tid; o < KB * R; o += THREADS) {
          const int kb = o / R, r = o % R;
          const uint32_t dst = sKV + kb * KV_BLK_BYTES + r * SEG_BYTES;
          if (kb < KB - 1) tma_tile_2d(dst, &tm_k1, kb * 64, sIdx[r], bar_tma);
          else tma_tile_2d(dst, &tm_p1, 0, sIdx[r], bar_tma);
        }
      }
      mbar_wait(bar_tma, tma_phase);
      tma_phase ^= 1u;
      PROF(2);

      // ---- S^T = K Q^T: 36 (block, k-step) MMAs round-robin over 4 accumulators. tcgen05.mma must be
      //      issued by a single thread (concurrent issuers corrupt results) and costs ~160 cycles per
      //      instruction to issue, which is the floor of this phase ----
      if (tid == 0) {
#pragma unroll
        for (int j = 0; j < KB * 4; ++j) {
          const int kb = j >> 2, ks = j & 3;
          const uint64_t a = dA_S + static_cast<uint64_t>((kb * KV_BLK_BYTES + ks * 32) >> 4);
          const uint64_t bq = dB_S + static_cast<uint64_t>((kb * Q_BLK_BYTES + ks * 32) >> 4);
          umma_bf16(tmem + 16 * (j & 3), a, bq, idesc_s, (j >= 4) ? 1u : 0u);
        }
        PROF(13);
        umma_commit(bar_mma0);
      }
      mbar_wait(bar_mma0, mma_phase);
      mma_phase ^= 1u;
      PROF(14);
      tc_fence_after_sync();
      PROF(3);

      // ---- softmax over the 64 rows, once per CTA. M=128: rows 0..63 = lanes 0..63 (warps 0,1);
      //      M=64: row r = lane 32*(r/16) + r%16 (16 rows per warp). Every warp sums the 4 partials. ----
      float s_h[H], m_h[H], l_h[H];
      int row;
      bool has_row;
      if (m64) {
        row = warp * 16 + (lane & 15);
        has_row = lane < 16;
      } else {
        row = warp * 32 + lane;
        has_row = warp < 2;
      }
      const bool valid = has_row && sIdx[row] >= 0;
      {
        uint32_t v[4][H];
#pragma unroll
        for (int w = 0; w < 4; ++w) tmem_ld16(tmem + lane_base + 16 * w, v[w]);
        tmem_wait_ld();
        PROF(15);
#pragma unroll
        for (int h = 0; h < H; ++h) {
          const float acc = (__uint_as_float(v[0][h]) + __uint_as_float(v[1][h])) +
                            (__uint_as_float(v[2][h]) + __uint_as_float(v[3][h]));
          s_h[h] = valid ? acc * scale_log2 : -INFINITY;
        }
      }
#pragma unroll
      for (int h = 0; h < H; ++h) {
        float m = s_h[h];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        m_h[h] = m;
      }
      if (lane == 0) {
#pragma unroll
        for (int h = 0; h < H; ++h) sRed[warp * H + h] = m_h[h];
      }
      __syncthreads();
#pragma unroll
      for (int h = 0; h < H; ++h) {
        m_h[h] = fmaxf(fmaxf(sRed[h], sRed[H + h]), fmaxf(sRed[2 * H + h], sRed[3 * H + h]));
        const float p = exp2f(s_h[h] - m_h[h]);
        float l = p;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        l_h[h] = l;
        if (has_row) {
          // P^T element (head h, row) in the K-major SW128 tile: row h at h*128, 16 B chunk (row/8) ^ (h%8)
          const uint32_t addr = sPT + h * SEG_BYTES + ((((row >> 3) ^ (h & 7)) << 4) | ((row & 7) << 1));
          const unsigned short pb = __bfloat16_as_ushort(__float2bfloat16_rn(p));
          asm volatile("st.shared.b16 [%0], %1;\n" ::"r"(addr), "h"(pb) : "memory");
        }
      }
      __syncthreads();  // everyone has read the maxima before the sums overwrite sRed
      if (lane == 0) {
#pragma unroll
        for (int h = 0; h < H; ++h) sRed[warp * H + h] = l_h[h];
      }
      fence_proxy_async_smem();  // P^T (generic proxy) -> UMMA (async proxy)
      __syncthreads();
#pragma unroll
      for (int h = 0; h < H; ++h) l_h[h] = (sRed[h] + sRed[H + h]) + (sRed[2 * H + h] + sRed[3 * H + h]);

      // ---- O^T = V^T P^T: 4 blocks of 128 dims (MN-major A from the same K tile) x 4 k-steps of 16 rows ----
      if (tid == 0) {
#pragma unroll
        for (int ks = 0; ks < R / 16; ++ks) {  // k-step major: consecutive MMAs hit different accumulators
#pragma unroll
          for (int mb = 0; mb < DC / 128; ++mb) {
            const uint64_t a = dA_PV + static_cast<uint64_t>(((2 * mb) * KV_BLK_BYTES + ks * 16 * SEG_BYTES) >> 4);
            const uint64_t bp = dB_PV + static_cast<uint64_t>((ks * 32) >> 4);
            umma_bf16(tmem + TMEM_COL_O + 16 * mb, a, bp, IDESC_PV, ks != 0 ? 1u : 0u);
          }
        }
        umma_commit(bar_mma0);
      }
      mbar_wait(bar_mma0, mma_phase);
      mma_phase ^= 1u;
      tc_fence_after_sync();
      PROF(4);

      // ---- epilogue: warp w owns dims 32w..32w+31 of every block; publish normalised bf16 [dim][head] ----
      float inv_l[H];
#pragma unroll
      for (int h = 0; h < H; ++h) inv_l[h] = 1.f / l_h[h];  // l >= 1: the row max contributes exp2(0)
      __nv_bfloat16* my_o = part_o + (static_cast<size_t>(t) * NCHUNK + c) * DC * H;
      uint32_t v[DC / 128][H];
#pragma unroll
      for (int mb = 0; mb < DC / 128; ++mb) tmem_ld16(tmem + lane_base + TMEM_COL_O + 16 * mb, v[mb]);
      tmem_wait_ld();
#pragma unroll
      for (int mb = 0; mb < DC / 128; ++mb) {
        const int dim = mb * 128 + warp * 32 + lane;
        uint32_t packed[H / 2];
#pragma unroll
        for (int h = 0; h < H; h += 2)
          packed[h / 2] =
              pack_bf16x2(__uint_as_float(v[mb][h]) * inv_l[h], __uint_as_float(v[mb][h + 1]) * inv_l[h + 1]);
        uint4* dst = reinterpret_cast<uint4*>(my_o + static_cast<size_t>(dim) * H);
        dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
      }
      if (tid < H) {  // from the exchange buffers: register arrays must not be indexed dynamically
        part_l[(static_cast<size_t>(t) * NCHUNK + c) * H + tid] =
            (sRed[tid] + sRed[H + tid]) + (sRed[2 * H + tid] + sRed[3 * H + tid]);
      }
      if (tid >= 32 && tid < 32 + H) {
        float mm = -INFINITY;
#pragma unroll
        for (int h = 0; h < H; ++h) mm = (h == tid - 32) ? m_h[h] : mm;  // select, no dynamic indexing
        part_m[(static_cast<size_t>(t) * NCHUNK + c) * H + (tid - 32)] = mm;
      }
      tc_fence_before_sync();  // TMEM reads done before the next chunk's MMAs (after the barrier below)
    }
    // ---- arrive: CTA barrier, then one thread fences (cumulative over the barrier) and adds ----
    __syncthreads();
    if (tid == 0) {
      fence_acq_rel_gpu();
      red_relaxed_add_u64(my_arrive, 1ull);
    }
    PROF(5);
    if (prof != nullptr && tid == 0) {
      prof[b * NPROF + 10] = sm;
      prof[b * NPROF + 11] = static_cast<unsigned long long>(nvalid > 0);
      prof[b * NPROF + 12] = static_cast<unsigned long long>(tmem) | (static_cast<unsigned long long>(slot) << 32);
    }
  }
  if (warp == 0) tmem_dealloc(tmem, TMEM_COLS);

  // ---- combine slice c of every owned chunk (t, c): dims [16c, +16) x 16 heads ----
  // Thread (warp w, lane l): dim = 16c + 4w + (l & 3), head pair hp = l >> 2. A warp's 32 loads of one
  // chunk cover 4 dims x 16 heads = 128 contiguous bytes. The 4 lanes sharing hp each hold the
  // statistics of 8 chunks; max / sum run over those 4 lanes and chunk weights are shuffled in.
  const int dd = lane & 3;
  const int hp = lane >> 2;
  for (int k = 0; k < owned; ++k) {
    const int i = first + k * G;
    const int t = i % T;
    const int c = i / T;
    const int b = i;
    const int dim = c * SLICE_DIMS + warp * 4 + dd;
    unsigned long long* my_arrive = arrive + static_cast<size_t>(t) * CTR_PAD;
    if (tid == 0) {  // bounded: a violated co-residency assumption yields wrong results, not a hang
      const unsigned long long target = (arrive_at_start[k] / NCHUNK + 1ull) * NCHUNK;
      for (int spins = 0; ld_acquire_u64(my_arrive) < target && spins < (1 << 22); ++spins) {
      }
    }
    __syncthreads();
    PROF(6);
    const float* pm = part_m + static_cast<size_t>(t) * NCHUNK * H + 2 * hp;
    const float* pl = part_l + static_cast<size_t>(t) * NCHUNK * H + 2 * hp;
    const __nv_bfloat16* po = part_o + (static_cast<size_t>(t) * NCHUNK * DC + dim) * H + 2 * hp;
    constexpr int CPL = NCHUNK / 4;  // chunks per lane of a 4-lane statistics group
    float2 mv[CPL], lv[CPL];
#pragma unroll
    for (int j = 0; j < CPL; ++j) {
      mv[j] = ld_cg_f32x2(pm + (dd * CPL + j) * H);
      lv[j] = ld_cg_f32x2(pl + (dd * CPL + j) * H);
    }
    uint32_t ov[NCHUNK];
#pragma unroll
    for (int q = 0; q < NCHUNK; ++q) ov[q] = ld_cg_u32(po + static_cast<size_t>(q) * DC * H);
    float M0 = -INFINITY, M1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < CPL; ++j) {
      M0 = fmaxf(M0, mv[j].x);
      M1 = fmaxf(M1, mv[j].y);
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {  // over the 4 lanes sharing hp
      M0 = fmaxf(M0, __shfl_xor_sync(0xffffffffu, M0, o));
      M1 = fmaxf(M1, __shfl_xor_sync(0xffffffffu, M1, o));
    }
    float w0[CPL], w1[CPL];
    float L0 = 0.f, L1 = 0.f;
#pragma unroll
    for (int j = 0; j < CPL; ++j) {  // empty chunks publish m = -inf for every head -> weight 0
      w0[j] = (mv[j].x > -INFINITY) ? exp2f(mv[j].x - M0) * lv[j].x : 0.f;
      w1[j] = (mv[j].y > -INFINITY) ? exp2f(mv[j].y - M1) * lv[j].y : 0.f;
      L0 += w0[j];
      L1 += w1[j];
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
      L0 += __shfl_xor_sync(0xffffffffu, L0, o);
      L1 += __shfl_xor_sync(0xffffffffu, L1, o);
    }
    PROF(8);
    float acc0 = 0.f, acc1 = 0.f;
    const int grp = lane & ~3;
#pragma unroll
    for (int q = 0; q < NCHUNK; ++q) {  // chunk q lives in lane (q / CPL) of the group, slot q % CPL
      const int src = grp | (q / CPL);
      const float wq0 = __shfl_sync(0xffffffffu, w0[q % CPL], src);
      const float wq1 = __shfl_sync(0xffffffffu, w1[q % CPL], src);
      const float2 f = unpack_bf16x2(ov[q]);
      acc0 += (wq0 != 0.f) ? wq0 * f.x : 0.f;  // stale partials of empty chunks may hold NaN
      acc1 += (wq1 != 0.f) ? wq1 * f.y : 0.f;
    }
    PROF(9);
    const bool any = M0 > -INFINITY;
    const float o0 = any ? acc0 / L0 : 0.f;
    const float o1 = any ? acc1 / L1 : 0.f;
    __nv_bfloat16* orow = out + (static_cast<size_t>(t) * H + 2 * hp) * DC + dim;
    orow[0] = __float2bfloat16_rn(o0);
    orow[DC] = __float2bfloat16_rn(o1);
    if (warp == 0 && dd == 0) {
      lse[t * H + 2 * hp] = any ? (M0 + log2f(L0)) : -INFINITY;
      lse[t * H + 2 * hp + 1] = any ? (M1 + log2f(L1)) : -INFINITY;
    }
    PROF(7);
  }
}

__global__ void noop_kernel() {}

// UMMA micro-benchmark: one CTA issues n_mma MMAs (M = 64 or 128, N = 16, K = 16) round-robin over n_acc
// accumulators; reports cycles for issue, commit->mbarrier completion and one tcgen05.ld round trip.
__global__ void umma_bench_kernel(int n_mma, int n_acc, int m64, int issuers, long long* out) {
  extern __shared__ unsigned char smem_raw[];
  const uint32_t raw_u32 = smem_u32(smem_raw);
  const uint32_t base = (raw_u32 + 1023u) & ~1023u;
  const uint32_t sA = base, sB = base + 16384, bar = base + 16384 + 2048, tslot = bar + 8;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (warp == 0) {
    tmem_alloc(tslot, 128);
    tmem_relinquish();
    tc_fence_before_sync();
  }
  if (tid == 0) {
    mbar_init(bar, issuers);
    mbar_fence_init();
  }
  __syncthreads();
  tc_fence_after_sync();
  const uint32_t tmem = *reinterpret_cast<const uint32_t*>(smem_raw + (tslot - raw_u32));
  const uint64_t dA = umma_desc_sw128(sA, 16, 1024), dB = umma_desc_sw128(sB, 16, 1024);
  const uint32_t idesc = m64 ? IDESC_S64 : IDESC_S;
  long long t0 = 0, t1 = 0, t2 = 0, t3 = 0;
  if (lane == 0 && warp < issuers) {
    t0 = clock64();
    for (int j = warp; j < n_mma; j += issuers) {
      const int acc = j % n_acc;
      umma_bf16(tmem + 16 * acc, dA + ((j & 7) * 32 >> 4), dB + ((j & 7) * 32 >> 4), idesc, (j >= n_acc) ? 1u : 0u);
    }
    t1 = clock64();
    umma_commit(bar);
  }
  mbar_wait(bar, 0u);
  tc_fence_after_sync();
  if (tid == 0) t2 = clock64();
  {
    uint32_t v[16];
    tmem_ld16(tmem + (static_cast<uint32_t>(warp * 32) << 16), v);
    tmem_wait_ld();
    if (tid == 0) t3 = clock64() + (v[0] == 0x7fc00001u ? 1 : 0);  // keep the load alive
  }
  tc_fence_before_sync();
  __syncthreads();
  if (warp == 0) tmem_dealloc(tmem, 128);
  if (tid == 0) {
    out[0] = t1 - t0;
    out[1] = t2 - t1;
    out[2] = t3 - t2;
  }
}

// Occupancy probes: which feature makes the runtime report 1 CTA per SM?
__global__ void __launch_bounds__(THREADS, CTAS_PER_SM) probe_plain_kernel(int* out) {
  if (threadIdx.x == 0 && out != nullptr) out[blockIdx.x] = 1;
}
__global__ void __launch_bounds__(THREADS, CTAS_PER_SM) probe_tmem_kernel(int* out) {
  __shared__ int slot[4];
  if (threadIdx.x < 32) {
    tmem_alloc(smem_u32(slot), 128);
    tmem_relinquish();
    tc_fence_before_sync();
  }
  __syncthreads();
  tc_fence_after_sync();
  if (threadIdx.x < 32) tmem_dealloc(static_cast<uint32_t>(slot[0]), 128);
  if (threadIdx.x == 0 && out != nullptr) out[blockIdx.x] = slot[0];
}
__global__ void __launch_bounds__(THREADS, CTAS_PER_SM) probe_tmap_kernel(const __grid_constant__ CUtensorMap tm,
                                                                          int* out) {
  if (threadIdx.x == 0 && out != nullptr) out[blockIdx.x] = static_cast<int>(reinterpret_cast<uint64_t>(&tm) & 0xff);
}
__global__ void __launch_bounds__(THREADS, CTAS_PER_SM) probe_tma_kernel(const __grid_constant__ CUtensorMap tm,
                                                                         int* out) {
  __shared__ __align__(1024) unsigned char buf[2048];
  __shared__ uint64_t bar;
  const uint32_t b = smem_u32(&bar);
  if (threadIdx.x == 0) {
    mbar_init(b, 1);
    mbar_fence_init();
    mbar_arrive_expect_tx(b, 2048);
    tma_tile_2d(smem_u32(buf), &tm, 0, 0, b);
    mbar_wait(b, 0u);
    if (out != nullptr) out[blockIdx.x] = buf[0];
  }
}
int g_per_sm_override = 0;

// ---- host: tensor maps (encoded through the driver entry point, cached per tensor) ----
using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiledFn encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
    cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(e == cudaSuccess && p != nullptr && q == cudaDriverEntryPointSuccess,
                "cuTensorMapEncodeTiled unavailable: ", cudaGetErrorString(e));
    fn = reinterpret_cast<EncodeTiledFn>(p);
  }
  return fn;
}

// 2D bf16 row-major [rows][cols] tensor, 128B swizzle, box = {box_cols, box_rows}
CUtensorMap make_tmap(const void* base, uint64_t rows, uint64_t cols, uint32_t box_cols, uint32_t box_rows,
                      int l2_promo) {
  using Key = std::tuple<const void*, uint64_t, uint64_t, uint32_t, uint32_t, int>;
  static std::map<Key, CUtensorMap> cache;
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  const Key key{base, rows, cols, box_cols, box_rows, l2_promo};
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap m;
  const cuuint64_t gdim[2] = {cols, rows};
  const cuuint64_t gstride[1] = {cols * 2};
  const cuuint32_t box[2] = {box_cols, box_rows};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), gdim, gstride, box,
                                 estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 static_cast<CUtensorMapL2promotion>(l2_promo), CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed with CUresult ", static_cast<int>(r));
  cache.emplace(key, m);
  return m;
}

int g_num_sms = 0;
int g_per_sm = 0;

void ensure_launchable() {
  static bool configured = false;
  if (configured) return;
  cudaError_t err = cudaFuncSetAttribute(dsa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
  TORCH_CHECK(err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(err));
  err = cudaFuncSetAttribute(dsa_fwd_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
  TORCH_CHECK(err == cudaSuccess, "carveout attribute failed: ", cudaGetErrorString(err));
  int dev = 0;
  cudaGetDevice(&dev);
  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
  int per_sm = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, dsa_fwd_kernel, THREADS, SMEM_TOTAL);
  TORCH_CHECK(err == cudaSuccess, "occupancy query failed: ", cudaGetErrorString(err));
  cudaFuncAttributes attr{};
  cudaFuncGetAttributes(&attr, dsa_fwd_kernel);
  int smem_per_sm = 0, smem_per_block_optin = 0, regs_per_sm = 0;
  cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
  cudaDeviceGetAttribute(&smem_per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaDeviceGetAttribute(&regs_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, dev);
  int occ_small = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_small, dsa_fwd_kernel, THREADS, 32 * 1024);
  printf("[kernel] numRegs=%d localSizeBytes=%zu staticSmem=%zu dynSmem=%d maxDynSmem=%d occupancy(dyn)=%d "
         "occupancy(32KB)=%d smemPerSM=%d smemPerBlockOptin=%d regsPerSM=%d sms=%d clusterDimMustBeSet=%d "
         "requiredCluster=%dx%dx%d maxThreadsPerBlock=%d\n",
         attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes, SMEM_TOTAL, attr.maxDynamicSharedSizeBytes, per_sm,
         occ_small, smem_per_sm, smem_per_block_optin, regs_per_sm, num_sms, attr.clusterDimMustBeSet,
         attr.requiredClusterWidth, attr.requiredClusterHeight, attr.requiredClusterDepth, attr.maxThreadsPerBlock);
  int occ_plain = 0, occ_tmem = 0, occ_tmap = 0, occ_tma = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_plain, probe_plain_kernel, THREADS, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_tmem, probe_tmem_kernel, THREADS, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_tmap, probe_tmap_kernel, THREADS, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_tma, probe_tma_kernel, THREADS, 0);
  cudaFuncAttributes at2{};
  cudaFuncGetAttributes(&at2, probe_tmem_kernel);
  printf("[kernel] probe occupancy (128 thr, no dyn smem): plain=%d tmem=%d tmap_param=%d tma_copy=%d; "
         "tmem probe regs=%d static=%zu\n", occ_plain, occ_tmem, occ_tmap, occ_tma, at2.numRegs, at2.sharedSizeBytes);
  if (g_per_sm_override > 0) printf("[kernel] per-SM CTA count overridden to %d\n", g_per_sm_override);
  fflush(stdout);
  // The grid is exactly per_sm CTAs on every SM: the spin barrier needs the whole grid resident,
  // and the per-SM slot counter needs every SM to receive exactly per_sm CTAs per launch.
  TORCH_CHECK(per_sm >= 1, "kernel does not fit on an SM");
  g_per_sm = (g_per_sm_override > 0) ? g_per_sm_override : std::min(per_sm, CTAS_PER_SM);
  g_num_sms = num_sms;
  configured = true;
}

}  // namespace

std::vector<torch::Tensor> run(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv_cache,
                               torch::Tensor kpe_cache, torch::Tensor sparse_indices, double sm_scale,
                               int gather_mode, int per_sm_override, int slot_mode, int m64, int l2_promo,
                               torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l, torch::Tensor arrive,
                               torch::Tensor smctr, torch::Tensor prof, bool do_prof) {
  TORCH_CHECK(q_nope.is_cuda() && q_nope.is_contiguous() && q_nope.dtype() == torch::kBFloat16);
  TORCH_CHECK(q_pe.is_contiguous() && ckv_cache.is_contiguous() && kpe_cache.is_contiguous());
  TORCH_CHECK(sparse_indices.is_contiguous() && sparse_indices.dtype() == torch::kInt32);
  const int T = q_nope.size(0);
  TORCH_CHECK(T >= 1 && T <= MAX_T, "num_tokens must be in [1, ", MAX_T, "]");
  TORCH_CHECK(q_nope.size(1) == H && q_nope.size(2) == DC && q_pe.size(2) == DP);
  TORCH_CHECK(ckv_cache.size(1) == 64 && ckv_cache.size(2) == DC && kpe_cache.size(2) == DP);
  TORCH_CHECK(sparse_indices.size(0) == T && sparse_indices.size(1) == TOPK);
  TORCH_CHECK(part_o.dtype() == torch::kBFloat16 && part_o.numel() >= static_cast<int64_t>(T) * NCHUNK * DC * H);
  TORCH_CHECK(part_m.numel() >= T * NCHUNK * H && part_l.numel() >= T * NCHUNK * H);
  TORCH_CHECK(arrive.dtype() == torch::kInt64 && arrive.numel() >= T * CTR_PAD);
  TORCH_CHECK(gather_mode >= 0 && gather_mode <= 4);
  if (per_sm_override > 0) g_per_sm_override = per_sm_override;

  ensure_launchable();
  const int G = g_per_sm * g_num_sms;
  TORCH_CHECK(smctr.dtype() == torch::kInt32 && smctr.numel() >= g_num_sms * 32);
  TORCH_CHECK(!do_prof || prof.numel() >= static_cast<int64_t>(std::max(G, T * NCHUNK)) * NPROF);

  const uint64_t kv_rows = static_cast<uint64_t>(ckv_cache.size(0)) * 64;
  TORCH_CHECK(l2_promo >= 0 && l2_promo <= 3);
  const CUtensorMap tm_k4 = make_tmap(ckv_cache.data_ptr(), kv_rows, DC, 64, 4, l2_promo);
  const CUtensorMap tm_p4 = make_tmap(kpe_cache.data_ptr(), kv_rows, DP, 64, 4, l2_promo);
  const CUtensorMap tm_k1 = make_tmap(ckv_cache.data_ptr(), kv_rows, DC, 64, 1, l2_promo);
  const CUtensorMap tm_p1 = make_tmap(kpe_cache.data_ptr(), kv_rows, DP, 64, 1, l2_promo);
  const CUtensorMap tm_qn = make_tmap(q_nope.data_ptr(), static_cast<uint64_t>(T) * H, DC, 64, H, l2_promo);
  const CUtensorMap tm_qp = make_tmap(q_pe.data_ptr(), static_cast<uint64_t>(T) * H, DP, 64, H, l2_promo);

  auto out = torch::empty({T, H, DC}, q_nope.options());
  auto lse = torch::empty({T, H}, q_nope.options().dtype(torch::kFloat32));
  const float scale_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;
  auto stream = at::cuda::getCurrentCUDAStream();
  dsa_fwd_kernel<<<G, THREADS, SMEM_TOTAL, stream>>>(
      tm_k4, tm_p4, tm_k1, tm_p1, tm_qn, tm_qp, sparse_indices.data_ptr<int>(), scale_log2, T, gather_mode,
      g_num_sms, slot_mode, m64, reinterpret_cast<__nv_bfloat16*>(part_o.data_ptr()), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
      reinterpret_cast<unsigned long long*>(arrive.data_ptr()), reinterpret_cast<unsigned int*>(smctr.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), lse.data_ptr<float>(),
      do_prof ? reinterpret_cast<unsigned long long*>(prof.data_ptr()) : nullptr);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "dsa_fwd_kernel launch failed: ", cudaGetErrorString(err));
  return {out, lse};
}

// Empty kernel launch: calibrates the fixed overhead of the timing method.
void run_noop(int grid, int smem_bytes) {
  static int configured_for = -1;
  if (smem_bytes > 48 * 1024 && configured_for < smem_bytes) {
    cudaFuncSetAttribute(noop_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    configured_for = smem_bytes;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  noop_kernel<<<grid, 128, smem_bytes, stream>>>();
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "noop launch failed: ", cudaGetErrorString(err));
}

void set_per_sm(int per_sm) { g_per_sm_override = per_sm; }

std::vector<long long> umma_bench(int n_mma, int n_acc, int m64, int issuers) {
  static bool configured = false;
  const int smem = 16384 + 2048 + 16 + 1024;
  if (!configured) {
    cudaFuncSetAttribute(umma_bench_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  auto out = torch::zeros({3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  umma_bench_kernel<<<1, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      n_mma, n_acc, m64, issuers, reinterpret_cast<long long*>(out.data_ptr<int64_t>()));
  cudaError_t err = cudaDeviceSynchronize();
  TORCH_CHECK(err == cudaSuccess, "umma_bench failed: ", cudaGetErrorString(err));
  auto h = out.cpu();
  return {static_cast<long long>(h[0].item<int64_t>()), static_cast<long long>(h[1].item<int64_t>()),
          static_cast<long long>(h[2].item<int64_t>())};
}
int num_chunks() { return NCHUNK; }
int num_prof() { return NPROF; }
int counter_pad() { return CTR_PAD; }
int grid_for(int num_tokens) {
  ensure_launchable();
  return std::max(g_per_sm * g_num_sms, num_tokens * NCHUNK);
}
int num_sms() {
  ensure_launchable();
  return g_num_sms;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "DSA sparse attention forward");
  m.def("run_noop", &run_noop, "empty kernel launch (grid, dynamic smem bytes)");
  m.def("set_per_sm", &set_per_sm, "override CTAs per SM (0 = occupancy query); call before any launch");
  m.def("umma_bench", &umma_bench, "cycles: [issue, commit->done, tmem ld] for n_mma MMAs over n_acc accumulators");
  m.def("num_chunks", &num_chunks, "chunks per token");
  m.def("num_prof", &num_prof, "profile stamps per CTA");
  m.def("counter_pad", &counter_pad, "u64 slots per arrival counter");
  m.def("grid_for", &grid_for, "profile rows for a token count");
  m.def("num_sms", &num_sms, "SM count (per-SM counter slots)");
}
