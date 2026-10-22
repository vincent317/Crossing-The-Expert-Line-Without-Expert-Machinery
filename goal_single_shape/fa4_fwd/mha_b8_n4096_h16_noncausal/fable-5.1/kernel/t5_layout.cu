// Probe: which smem core-matrix layout does tcgen05.mma expect for each of
// a_major / b_major under SWIZZLE_NONE, and what is the TMEM accumulator ->
// (thread, register) mapping?
//
// Computes C[128,128] = A[128,128] @ B[128,128] where B is given row-major as
// (K=128, N=128), i.e. the natural "B is not transposed" case, plus the
// A @ B^T case. Each candidate layout is tried and scored against a CPU
// reference, so the correct encoding is identified by construction.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define MM 128
#define NN 128
#define KK 128
#define MMA_K 16

__device__ __forceinline__ uint32_t sm_u32(const void *p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint64_t mk_desc(uint32_t a, uint32_t lbo, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((a >> 4) & 0x3FFFu);
  d |= (uint64_t)(lbo & 0x3FFFu) << 16;
  d |= (uint64_t)(sbo & 0x3FFFu) << 32;
  d |= (uint64_t)1u << 46;
  return d;
}
__device__ __forceinline__ uint32_t mk_idesc(int n, int m, int am, int bm) {
  uint32_t d = 0;
  d |= 1u << 4;
  d |= 1u << 7;
  d |= 1u << 10;
  d |= (uint32_t)am << 15;
  d |= (uint32_t)bm << 16;
  d |= (uint32_t)(n >> 3) << 17;
  d |= (uint32_t)(m >> 4) << 24;
  return d;
}
__device__ __forceinline__ void do_mma(uint32_t tc, uint64_t da, uint64_t db,
                                       uint32_t id, uint32_t acc) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
               :
               : "r"(tc), "l"(da), "l"(db), "r"(id), "r"(acc));
}

// layout A: 8 values of the CONTIGUOUS axis per u128, MN groups of 8 interleaved
__device__ __host__ __forceinline__ int lay_a(int mn, int k, int ktot) {
  return (mn & 7) + (mn >> 3) * ktot + (k >> 3) * 8;
}
// layout B: 8 values of the MN axis per u128
__device__ __host__ __forceinline__ int lay_b(int mn, int k, int ktot) {
  return (mn >> 3) * ktot + (k & 7) + (k >> 3) * 8;
}

// CAND selects the B-operand handling:
//  0: B^T case, B stored (N,K) row-major, b_major=0, layout lay_a
//  1: B   case, B stored (K,N) row-major, b_major=1, layout lay_a over (n,k)
//  2: B   case, B stored (K,N) row-major, b_major=1, layout lay_b over (n,k)
//  3: B   case, B stored (K,N) row-major, b_major=0, layout lay_b over (n,k)
extern "C" __global__ __launch_bounds__(128) void probe(
    const __nv_bfloat16 *__restrict__ A, const __nv_bfloat16 *__restrict__ Bin,
    float *__restrict__ out, int cand) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __nv_bfloat16 *sA = (__nv_bfloat16 *)smem;
  __nv_bfloat16 *sB = (__nv_bfloat16 *)(smem + MM * KK * 2);
  uint32_t *tmp = (uint32_t *)(smem + MM * KK * 2 + NN * KK * 2);
  uint64_t *mbar = (uint64_t *)(tmp + 8);

  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :: "r"(sm_u32(tmp)), "r"(NN));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
  }
  if (tid == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(sm_u32(mbar)));
  __syncthreads();
  uint32_t tc = tmp[0];

  // A is always (M,K) row-major, K-major layout lay_a
  for (int i = tid; i < MM * KK / 8; i += 128) {
    int m = i / (KK / 8), kg = i % (KK / 8);
    ((uint4 *)sA)[lay_a(m, kg * 8, KK)] = *(const uint4 *)(A + (size_t)m * KK + kg * 8);
  }
  // B loading depends on candidate
  if (cand == 0) {
    // Bin is (N,K) row-major; treat mn=n, contiguous axis = k
    for (int i = tid; i < NN * KK / 8; i += 128) {
      int n = i / (KK / 8), kg = i % (KK / 8);
      ((uint4 *)sB)[lay_a(n, kg * 8, KK)] = *(const uint4 *)(Bin + (size_t)n * KK + kg * 8);
    }
  } else {
    // Bin is (K,N) row-major; contiguous axis = n
    for (int i = tid; i < NN * KK / 8; i += 128) {
      int k = i / (NN / 8), ng = i % (NN / 8);
      uint4 val = *(const uint4 *)(Bin + (size_t)k * NN + ng * 8);
      int off;
      if (cand == 1) off = lay_a(k, ng * 8, NN);        // mn=k?? contiguous=n
      else           off = lay_b(ng * 8, k, NN);        // 8 n-values per u128
      ((uint4 *)sB)[off] = val;
    }
  }
  __syncthreads();

  int bmaj = (cand == 0) ? 0 : ((cand == 3) ? 0 : 1);
  uint32_t id = mk_idesc(NN, MM, 0, bmaj);
  if (warp == 0 && lane == 0) {
    uint32_t ba = sm_u32(sA), bb = sm_u32(sB);
    for (int ks = 0; ks < KK / MMA_K; ++ks) {
      uint64_t da = mk_desc(ba + lay_a(0, ks * MMA_K, KK) * 16, 8, KK);
      uint32_t bo;
      if (cand == 0)      bo = lay_a(0, ks * MMA_K, KK) * 16;
      else if (cand == 1) bo = lay_a(ks * MMA_K, 0, NN) * 16;
      else                bo = lay_b(0, ks * MMA_K, NN) * 16;
      uint64_t db = mk_desc(bb + bo, 8, (cand == 0) ? KK : NN);
      do_mma(tc, da, db, id, ks == 0 ? 0 : 1);
    }
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(sm_u32(mbar)));
  }
  if (tid == 0)
    asm volatile("{\n\t.reg .pred p;\n\tLW:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
                 "@!p bra LW;\n\t}\n" :: "r"(sm_u32(mbar)));
  __syncthreads();

  uint32_t addr = tc + ((uint32_t)(warp * 32) << 16);
  uint32_t r[32];
  for (int cb = 0; cb < NN / 32; ++cb) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                 "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},[%32];\n"
                 : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),
                   "=r"(r[6]),"=r"(r[7]),"=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                   "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),"=r"(r[16]),"=r"(r[17]),
                   "=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                   "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),
                   "=r"(r[30]),"=r"(r[31])
                 : "r"(addr + cb * 32));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
    for (int j = 0; j < 32; ++j)
      out[(size_t)tid * NN + cb * 32 + j] = __int_as_float(r[j]);
  }
  __syncthreads();
  if (warp == 0)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tc), "r"(NN));
}

int main() {
  // A[m,k] = (m*7 + k*3) % 5 - 2 ; B[k,n] = (k*11 + n*5) % 7 - 3  (exact in bf16)
  __nv_bfloat16 *hA = new __nv_bfloat16[MM * KK];
  __nv_bfloat16 *hBkn = new __nv_bfloat16[KK * NN];  // (K,N) row-major
  __nv_bfloat16 *hBnk = new __nv_bfloat16[NN * KK];  // (N,K) row-major = B^T
  float *cpuAB = new float[MM * NN];                 // A @ B
  float *cpuABt = new float[MM * NN];                // A @ (B^T stored as (N,K))
  auto fa = [](int m, int k) { return (float)((m * 7 + k * 3) % 5 - 2); };
  auto fb = [](int k, int n) { return (float)((k * 11 + n * 5) % 7 - 3); };
  for (int m = 0; m < MM; ++m)
    for (int k = 0; k < KK; ++k) hA[m * KK + k] = __float2bfloat16(fa(m, k));
  for (int k = 0; k < KK; ++k)
    for (int n = 0; n < NN; ++n) hBkn[k * NN + n] = __float2bfloat16(fb(k, n));
  for (int n = 0; n < NN; ++n)
    for (int k = 0; k < KK; ++k) hBnk[n * KK + k] = __float2bfloat16(fb(k, n));
  for (int m = 0; m < MM; ++m)
    for (int n = 0; n < NN; ++n) {
      float s = 0;
      for (int k = 0; k < KK; ++k) s += fa(m, k) * fb(k, n);
      cpuAB[m * NN + n] = s;
      cpuABt[m * NN + n] = s;  // same product; only the storage differs
    }

  __nv_bfloat16 *dA, *dBkn, *dBnk;
  float *dOut;
  cudaMalloc(&dA, MM * KK * 2);
  cudaMalloc(&dBkn, KK * NN * 2);
  cudaMalloc(&dBnk, NN * KK * 2);
  cudaMalloc(&dOut, 128 * NN * 4);
  cudaMemcpy(dA, hA, MM * KK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dBkn, hBkn, KK * NN * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dBnk, hBnk, NN * KK * 2, cudaMemcpyHostToDevice);
  size_t smem = MM * KK * 2 + NN * KK * 2 + 1024;
  cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

  float *hOut = new float[128 * NN];
  for (int cand = 0; cand < 4; ++cand) {
    cudaMemset(dOut, 0, 128 * NN * 4);
    probe<<<1, 128, smem>>>(dA, cand == 0 ? dBnk : dBkn, dOut, cand);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("cand=%d ERR %s\n", cand, cudaGetErrorString(e)); continue; }
    cudaMemcpy(hOut, dOut, 128 * NN * 4, cudaMemcpyDeviceToHost);
    // score against the hypothesis "thread t holds row t, register j holds col j"
    double err = 0; int nz = 0;
    for (int t = 0; t < 128; ++t)
      for (int j = 0; j < NN; ++j) {
        float got = hOut[(size_t)t * NN + j];
        err = fmax(err, fabs((double)got - cpuAB[t * NN + j]));
        if (got != 0) nz++;
      }
    printf("cand=%d  max_err_vs(row=tid,col=reg)=%.1f  nonzero=%d/%d  "
           "sample got[0][0..3]=%.0f %.0f %.0f %.0f  want=%.0f %.0f %.0f %.0f\n",
           cand, err, nz, 128 * NN, hOut[0], hOut[1], hOut[2], hOut[3],
           cpuAB[0], cpuAB[1], cpuAB[2], cpuAB[3]);
    printf("        got[1][0..3]=%.0f %.0f %.0f %.0f  want=%.0f %.0f %.0f %.0f\n",
           hOut[NN], hOut[NN + 1], hOut[NN + 2], hOut[NN + 3],
           cpuAB[NN], cpuAB[NN + 1], cpuAB[NN + 2], cpuAB[NN + 3]);
  }
  return 0;
}
