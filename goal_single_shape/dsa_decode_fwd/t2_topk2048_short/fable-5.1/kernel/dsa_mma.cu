// DSA sparse-attention forward (bf16 in / bf16 out, fp32 softmax) on tensor cores:
// one CTA per token, all 16 heads at once.
//
//   S[16 heads x rows] = Q K^T            mma.sync m16n8k16 (bf16 -> fp32), M = heads
//   P = softmax_rows(S * scale)           fp32, online across chunks of 32 rows
//   O[16 heads x 512]  = P K              mma.sync, K^T fragments via ldmatrix.trans
//
// The 2048 sparse indices are loaded coalesced by all 256 threads; every warp also
// loads positions 0..31 itself. Because decode workloads put a token's few valid
// indices at the front of the list, all threads speculatively start cp.async copies
// of the rows at positions 0..31 as soon as the indices land; the exact compaction
// runs while those copies are in flight and only decides whether the guess covered
// every valid index (otherwise the general chunked path is taken). Everything a CTA
// needs (Q, K rows, P) lives in shared memory, so the arithmetic is a few hundred
// tensor-core / ldmatrix instructions instead of thousands of scalar FMAs, and only
// 2 CTAs are launched (no straggler skew).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kHeads = 16;
constexpr int kDCkv = 512;
constexpr int kDKpe = 64;
constexpr int kDK = kDCkv + kDKpe;  // 576: contraction length of S
constexpr int kTopK = 2048;
constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kRows = 32;                // rows per chunk = 4 n-tiles of 8
constexpr int kRowBytes = kDK * 2;       // 1152: ckv row (1024B) followed by kpe row (128B)
constexpr int kRowStride = kRowBytes + 16;  // 1168: 8 consecutive rows hit 8 distinct 16B bank groups
constexpr int kPStride = kRows * 2 + 16;    // 80: same trick for the P tile
constexpr int kIdxPerThread = kTopK / kThreads;  // 8 (two uint4)
constexpr int kListCap = kTopK / kWarps;         // per-warp compaction capacity

#ifndef DSA_TIMING
#define DSA_TIMING 0
#endif
constexpr int kTimingCtas = 32;
constexpr int kTimingStamps = 7;
#if DSA_TIMING
__device__ unsigned long long g_timing[kTimingCtas][2 * kTimingStamps];
__device__ __forceinline__ unsigned long long global_timer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define DSA_STAMP(i)                                  \
  do {                                                \
    if (tid == 0 && tok < kTimingCtas) {              \
      g_timing[tok][i] = global_timer_ns();           \
      g_timing[tok][kTimingStamps + (i)] = clock64(); \
    }                                                 \
  } while (0)
#else
#define DSA_STAMP(i) \
  do {               \
  } while (0)
#endif

// shared memory layout (bytes)
constexpr int kOffQ = 0;
constexpr int kOffK = kOffQ + kHeads * kRowStride;      // 18688
constexpr int kOffIdxW = kOffK + kRows * kRowStride;   // + 37376 = 56064 : per-warp lists [8][256] int
constexpr int kOffIdx = kOffIdxW + kWarps * kListCap * 4;  // + 8192 = 64256 : compacted list [2048] int
constexpr int kOffP = kOffIdx + kTopK * 4;             // + 8192 = 72448 : P [16][kPStride] bf16
constexpr int kOffS = kOffP + kHeads * kPStride;        // + 1280 = 73728 : partial S [8 warps][16 heads][32 rows] f32
constexpr int kOffL = kOffS + kWarps * kHeads * kRows * 4;  // + 16384 = 90112 : L [16] f32
constexpr int kOffAlpha = kOffL + 64;                   // 90176 : alpha [16] f32
constexpr int kOffCnt = kOffAlpha + 64;                 // 90240 : [8] warp counts
constexpr int kOStride = kDCkv * 2 + 16;                // 1040: O staging row stride (conflict-free 4B stores)
constexpr int kOffO = kOffCnt + 64;                     // 90304 : O staging [16 heads][kOStride] bf16
constexpr int kSmemBytes = kOffO + kHeads * kOStride;   // 106944
constexpr int kChunks16 = kRows * (kRowBytes / 16);     // 2304 16B pieces per chunk of rows
constexpr int kQChunks16 = kHeads * (kRowBytes / 16);   // 1152 16B pieces of Q
static_assert(kOffK % 16 == 0 && kOffP % 16 == 0 && kOffS % 16 == 0 && kOffL % 16 == 0, "smem alignment");
static_assert(kChunks16 % kThreads == 0, "row gather is 9 cp.async per thread");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- cp.async (16B, L1-bypassing) --------------------------------------------------------
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit_wait_all() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
}

// ---- ldmatrix / mma ----------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}
// D = A(16x16 bf16, row) * B(16x8 bf16, col) + C, fp32 accumulate
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float fast_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float fast_log2(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// All 256 threads copy one chunk of rows (32 x 1152B) into sK with 9 cp.async each.
// Lane l of every warp holds the row index of slot l (identical across warps).
__device__ __forceinline__ void gather_rows(uint32_t s_k, int tid, int row_of_my_slot, const char* __restrict__ ckv,
                                            const char* __restrict__ kpe) {
#pragma unroll
  for (int k = 0; k < kChunks16 / kThreads; ++k) {
    const int c = tid + k * kThreads;  // 16B piece id over [32 slots][72 pieces]
    const int slot = c / 72, cc = c - slot * 72;
    const int row = __shfl_sync(0xffffffffu, row_of_my_slot, slot);
    const char* src = cc < 64 ? ckv + (size_t)row * (kDCkv * 2) + cc * 16
                              : kpe + (size_t)row * (kDKpe * 2) + (cc - 64) * 16;
    cp_async_16(s_k + slot * kRowStride + cc * 16, src);
  }
}

__global__ void __launch_bounds__(kThreads, 1)
dsa_mma_kernel(const char* __restrict__ q_nope, const char* __restrict__ q_pe, const char* __restrict__ ckv,
               const char* __restrict__ kpe, const int32_t* __restrict__ indices, float scale_log2e,
               char* __restrict__ out, float* __restrict__ lse) {
  extern __shared__ __align__(128) char smem[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int tok = blockIdx.x;
  const int g = lane >> 2;  // mma "groupID"
  const int t4 = lane & 3;  // mma "threadID in group"

  const uint32_t s_q = smem_u32(smem + kOffQ);
  const uint32_t s_k = smem_u32(smem + kOffK);
  const uint32_t s_p = smem_u32(smem + kOffP);
  int* s_idxw = reinterpret_cast<int*>(smem + kOffIdxW);
  int* s_idx = reinterpret_cast<int*>(smem + kOffIdx);
  float* s_s = reinterpret_cast<float*>(smem + kOffS);          // [8 warps][16 heads][32 rows] partial S
  float* s_l = reinterpret_cast<float*>(smem + kOffL);          // [16]
  float* s_alpha = reinterpret_cast<float*>(smem + kOffAlpha);  // [16]
  int* s_cnt = reinterpret_cast<int*>(smem + kOffCnt);          // [8] warp counts
  DSA_STAMP(0);

  // ---- 1. loads in flight together: positions 0..31 (every warp, for the speculative gather)
  //         and the whole index list (coalesced uint4, for the exact compaction)
  const int32_t* idx = indices + (size_t)tok * kTopK;
  const uint4* idx4 = reinterpret_cast<const uint4*>(idx);
  const int pos_l = __ldg(idx + lane);  // position `lane` of the list
  const uint4 iv0 = __ldg(idx4 + tid);
  const uint4 iv1 = __ldg(idx4 + kThreads + tid);

  // ---- 2. speculative gather of positions 0..31 as soon as they land, then Q (18KB) into smem;
  //         K first so its 36KB of cp.async traffic is not queued behind Q's
  const uint32_t valid_mask = __ballot_sync(0xffffffffu, pos_l >= 0);  // identical in every warp
  const bool spec_issued = valid_mask != 0;
  if (spec_issued) {
    const int dup = __shfl_sync(0xffffffffu, pos_l, __ffs(valid_mask) - 1);  // padding slots: a real (finite) row
    gather_rows(s_k, tid, pos_l >= 0 ? pos_l : dup, ckv, kpe);
  }
#pragma unroll
  for (int k = 0; k < (kQChunks16 + kThreads - 1) / kThreads; ++k) {
    const int c = tid + k * kThreads;  // 16B piece id over [16 heads][72 pieces]
    if (c < kQChunks16) {
      const int h = c / 72, cc = c - h * 72;
      const char* src = cc < 64 ? q_nope + ((size_t)(tok * kHeads + h) * 64 + cc) * 16
                                : q_pe + ((size_t)(tok * kHeads + h) * 8 + (cc - 64)) * 16;
      cp_async_16(s_q + h * kRowStride + cc * 16, src);
    }
  }
  DSA_STAMP(1);  // speculative row copies and Q copies issued

  // ---- 3. exact per-warp compaction (runs while the speculative copies are in flight)
  const int iv[8] = {(int)iv0.x, (int)iv0.y, (int)iv0.z, (int)iv0.w, (int)iv1.x, (int)iv1.y, (int)iv1.z, (int)iv1.w};
  const uint32_t lt_mask = (1u << lane) - 1u;
  int cnt_w = 0;
#pragma unroll
  for (int r = 0; r < kIdxPerThread; ++r) {
    const uint32_t b = __ballot_sync(0xffffffffu, iv[r] >= 0);
    if (iv[r] >= 0) s_idxw[warp * kListCap + cnt_w + __popc(b & lt_mask)] = iv[r];
    cnt_w += __popc(b);
  }
  if (lane == 0) s_cnt[warp] = cnt_w;
  cp_async_commit_wait_all();  // own Q / speculative row pieces have landed ...
  __syncthreads();             // ... and everybody's are visible; counts visible too
  DSA_STAMP(2);  // Q and speculative rows in smem

  int pre[kWarps + 1];
  pre[0] = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) pre[w + 1] = pre[w] + s_cnt[w];
  const int cnt = pre[kWarps];
  const bool fast = spec_issued && cnt == __popc(valid_mask);  // the guess covered every valid index

  if (cnt == 0) {  // reference semantics: zero output, lse = -inf (no copies were issued)
    uint4* o4 = reinterpret_cast<uint4*>(out + (size_t)tok * kHeads * kDCkv * 2);
#pragma unroll
    for (int k = 0; k < 4; ++k) o4[k * kThreads + tid] = make_uint4(0u, 0u, 0u, 0u);
    if (tid < kHeads) lse[tok * kHeads + tid] = -INFINITY;
    return;
  }
  if (!fast) {  // general path: build the global compacted list
    for (int j = lane; j < s_cnt[warp]; j += 32) s_idx[pre[warp] + j] = s_idxw[warp * kListCap + j];
    __syncthreads();
  }

  // ---- 4. chunk loop -------------------------------------------------------------------
  // S: warp w contracts k-steps [kb, kb+kn) for all 4 row tiles (Q and K rows are each read
  //    from smem exactly once); the 8 partial S tiles are summed by the softmax threads.
  // softmax: thread (h = tid/16, part = tid%16) owns rows 2*part, 2*part+1 of head h.
  // PV: warp w owns dims [64w, 64w+64) as 8 n-tiles.
  float o_acc[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) o_acc[i][0] = o_acc[i][1] = o_acc[i][2] = o_acc[i][3] = 0.f;
  const int sh = tid >> 4, spart = tid & 15;  // softmax role
  float m_run = -INFINITY, l_run = 0.f;       // running softmax state of head sh (16 threads each)
  const int kb = warp < 4 ? warp * 5 : 20 + (warp - 4) * 4;  // 36 k-steps = 4x5 + 4x4
  const int kn = warp < 4 ? 5 : 4;

  const int nchunks = fast ? 1 : (cnt + kRows - 1) / kRows;
  for (int chunk = 0; chunk < nchunks; ++chunk) {
    const int base = chunk * kRows;
    if (!fast) {
      if (chunk > 0) __syncthreads();  // everyone is done reading sK / sP / alpha of the previous chunk
      const int nv = min(kRows, cnt - base);
      gather_rows(s_k, tid, s_idx[base + (lane < nv ? lane : 0)], ckv, kpe);  // padding: first row
      cp_async_commit_wait_all();
      __syncthreads();
    }

    // -- partial S over this warp's k-slice, 4 independent accumulator chains (one per row tile)
    {
      float acc[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
      const uint32_t a_base = s_q + (lane & 15) * kRowStride + (lane >> 4) * 16;
      const uint32_t b_base = s_k + (lane & 7) * kRowStride + ((lane >> 3) & 1) * 16;
#pragma unroll
      for (int j = 0; j < 5; ++j) {
        if (j < kn) {  // warp-uniform
          const int ks = kb + j;
          uint32_t a0, a1, a2, a3, b0[4], b1[4];
          ldmatrix_x4(a0, a1, a2, a3, a_base + ks * 32);  // all loads of the step first ...
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) ldmatrix_x2(b0[nt], b1[nt], b_base + nt * 8 * kRowStride + ks * 32);
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) mma_bf16(acc[nt], a0, a1, a2, a3, b0[nt], b1[nt]);  // ... then the mmas
        }
      }
      float* mine = s_s + (warp * kHeads) * kRows;  // [head][row] of this warp
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {  // c0,c1: head g rows 8nt+2t4,+1 ; c2,c3: head g+8
        *reinterpret_cast<float2*>(mine + g * kRows + nt * 8 + 2 * t4) = make_float2(acc[nt][0], acc[nt][1]);
        *reinterpret_cast<float2*>(mine + (g + 8) * kRows + nt * 8 + 2 * t4) = make_float2(acc[nt][2], acc[nt][3]);
      }
    }
    DSA_STAMP(3);  // partial S written
    __syncthreads();  // (A) all partial S tiles visible

    // -- softmax: sum the 8 partials, mask, online max / exp / sum for head sh, rows 2*spart, +1
    {
      const int r0 = 2 * spart;
      float2 acc2 = make_float2(0.f, 0.f);
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        const float2 v = *reinterpret_cast<const float2*>(s_s + (w * kHeads + sh) * kRows + r0);
        acc2.x += v.x;
        acc2.y += v.y;
      }
      const bool ok0 = fast ? ((valid_mask >> r0) & 1u) != 0u : (base + r0 < cnt);
      const bool ok1 = fast ? ((valid_mask >> (r0 + 1)) & 1u) != 0u : (base + r0 + 1 < cnt);
      const float s0 = ok0 ? acc2.x * scale_log2e : -INFINITY;
      const float s1 = ok1 ? acc2.y * scale_log2e : -INFINITY;
      float mx = fmaxf(s0, s1);
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      const float m_new = fmaxf(m_run, mx);
      const float alpha = fast_exp2(m_run - m_new);  // -inf -> 0 on the first chunk
      const float p0 = fast_exp2(s0 - m_new), p1 = fast_exp2(s1 - m_new);
      float ls = p0 + p1;
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) ls += __shfl_xor_sync(0xffffffffu, ls, o);
      l_run = fmaf(l_run, alpha, ls);
      m_run = m_new;
      *reinterpret_cast<uint32_t*>(smem + kOffP + sh * kPStride + r0 * 2) = pack_bf16x2(p0, p1);
      if (spart == 0) s_alpha[sh] = alpha;
    }
    __syncthreads();  // (B) P and alpha visible
    DSA_STAMP(4);

    // -- O = alpha * O + P K for this warp's 64 dims
    const float al0 = s_alpha[g], al8 = s_alpha[g + 8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      o_acc[i][0] *= al0;
      o_acc[i][1] *= al0;
      o_acc[i][2] *= al8;
      o_acc[i][3] *= al8;
    }
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {  // rows 16ks .. 16ks+15
      uint32_t a0, a1, a2, a3, b0[8], b1[8];
      ldmatrix_x4(a0, a1, a2, a3, s_p + (lane & 15) * kPStride + (16 * ks + 8 * (lane >> 4)) * 2);
      const uint32_t kbase = s_k + (16 * ks + (lane & 15)) * kRowStride + (warp * 64) * 2;
#pragma unroll
      for (int i = 0; i < 8; ++i) ldmatrix_x2_trans(b0[i], b1[i], kbase + i * 16);
#pragma unroll
      for (int i = 0; i < 8; ++i) mma_bf16(o_acc[i], a0, a1, a2, a3, b0[i], b1[i]);
    }
    DSA_STAMP(5);  // PV done
  }

  // ---- 5. epilogue: O /= L, lse = m + log2 L; O goes through smem so the global stores are
  //         full 16B-per-lane lines instead of 4B pieces scattered over 16 rows
  if (spart == 0) {
    s_l[sh] = l_run;
    lse[tok * kHeads + sh] = m_run + fast_log2(l_run);  // 2-based log-sum-exp of the scaled logits
  }
  __syncthreads();  // (D)
  const float inv0 = fast_rcp(s_l[g]), inv8 = fast_rcp(s_l[g + 8]);
  char* s_o = smem + kOffO;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int d = warp * 64 + i * 8 + 2 * t4;
    *reinterpret_cast<uint32_t*>(s_o + g * kOStride + d * 2) = pack_bf16x2(o_acc[i][0] * inv0, o_acc[i][1] * inv0);
    *reinterpret_cast<uint32_t*>(s_o + (g + 8) * kOStride + d * 2) =
        pack_bf16x2(o_acc[i][2] * inv8, o_acc[i][3] * inv8);
  }
  __syncthreads();  // (E)
  uint4* out_tok = reinterpret_cast<uint4*>(out + (size_t)tok * kHeads * kDCkv * 2);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int p = tid + k * kThreads;  // 16B piece over [16 heads][64 pieces]
    const int h = p >> 6, cc = p & 63;
    out_tok[p] = *reinterpret_cast<const uint4*>(s_o + h * kOStride + cc * 16);
  }
  DSA_STAMP(6);  // output written
}

// Probes: 64 dependent mma.sync (latency) vs 64 mma.sync over 8 independent accumulators.
__global__ void dsa_mma_probe_kernel(int chain, float* __restrict__ sink) {
  const uint32_t a = 0x3f803f80u, b = 0x3f803f80u;  // bf16x2(1,1)
  float acc[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  if (chain == 1) {
#pragma unroll
    for (int i = 0; i < 64; ++i) mma_bf16(acc[0], a, a, a, a, b, b);
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) mma_bf16(acc[j], a, a, a, a, b, b);
    }
  }
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) t += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
  sink[threadIdx.x] = t;
}

}  // namespace

extern "C" int dsa_mma_probe_launch(int chain, void* sink, cudaStream_t stream) {
  dsa_mma_probe_kernel<<<1, 32, 0, stream>>>(chain, reinterpret_cast<float*>(sink));
  return (int)cudaGetLastError();
}

extern "C" int dsa_launch(const void* q_nope, const void* q_pe, const void* ckv_cache, const void* kpe_cache,
                          const int32_t* sparse_indices, float sm_scale, void* out, float* lse, int num_tokens,
                          cudaStream_t stream) {
  static bool attr_set = false;
  if (!attr_set) {
    if (cudaFuncSetAttribute(dsa_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes) !=
        cudaSuccess)
      return (int)cudaGetLastError();
    attr_set = true;
  }
  dsa_mma_kernel<<<num_tokens, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<const char*>(q_nope), reinterpret_cast<const char*>(q_pe),
      reinterpret_cast<const char*>(ckv_cache), reinterpret_cast<const char*>(kpe_cache), sparse_indices,
      sm_scale * 1.4426950408889634f, reinterpret_cast<char*>(out), lse);
  return (int)cudaGetLastError();
}

extern "C" const char* dsa_config() {
#if DSA_TIMING
  return "mma: 1 CTA/token, 8 warps, k-split tensor-core S, tensor-core PV, cp.async gather timing=1";
#else
  return "mma: 1 CTA/token, 8 warps, k-split tensor-core S, tensor-core PV, cp.async gather";
#endif
}

extern "C" int dsa_timing_read(void* dst, size_t bytes) {
#if DSA_TIMING
  if (bytes != sizeof(g_timing)) return -2;
  return (int)cudaMemcpyFromSymbol(dst, g_timing, sizeof(g_timing));
#else
  (void)dst;
  (void)bytes;
  return -1;
#endif
}
