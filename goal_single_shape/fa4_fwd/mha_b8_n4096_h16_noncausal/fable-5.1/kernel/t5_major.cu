// Probe: which smem encoding does tcgen05.mma accept for an MN-major operand?
//
// If an MN-major A operand works, attention can compute O^T = V^T @ P^T with V
// in its natural (key, head_dim) layout -- no transpose, so V can be fetched
// with cp.async. Each variant runs in its own process (argv), because a bad
// descriptor faults the context.
//
// Reference GEMM: C[128,128] = A[128,128] @ B[128,128] with
//   A[m,k] = fa(m,k), B[k,n] = fb(k,n)
// A is supplied MN-major (m contiguous in memory, i.e. A^T row-major) and
// declared a_major=1. B is K-major (validated encoding), b_major=0.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define MM 128
#define NN 128
#define KK 128
#define MMA_K 16

__device__ __forceinline__ uint32_t sm_u32(const void *p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint64_t mk_desc(uint32_t a, uint32_t lbo, uint32_t sbo,
                                            uint32_t ltype) {
  uint64_t d = 0;
  d |= (uint64_t)((a >> 4) & 0x3FFFu);
  d |= (uint64_t)(lbo & 0x3FFFu) << 16;
  d |= (uint64_t)(sbo & 0x3FFFu) << 32;
  d |= (uint64_t)1u << 46;
  d |= (uint64_t)(ltype & 7u) << 61;
  return d;
}
__device__ __forceinline__ uint32_t mk_idesc(int n, int m, int am, int bm) {
  uint32_t d = 0;
  d |= 1u << 4; d |= 1u << 7; d |= 1u << 10;
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

// K-major (validated): u128 holds 8 consecutive K; MN groups of 8 interleaved.
__device__ __host__ __forceinline__ int off_k(int mn, int k, int ktot) {
  return (mn & 7) + (mn >> 3) * ktot + (k >> 3) * 8;
}
// MN-major candidate A: u128 holds 8 consecutive MN values.
__device__ __host__ __forceinline__ int off_mn_a(int mn, int k, int mntot) {
  return (mn >> 3) * 8 + (k & 7) * (mntot / 8) + (k >> 3) * 8 * (mntot / 8) * 8;
}
// MN-major candidate B: mirror of off_k with the roles of mn and k swapped.
__device__ __host__ __forceinline__ int off_mn_b(int mn, int k, int mntot) {
  return (k & 7) + (k >> 3) * mntot + (mn >> 3) * 8;
}

extern "C" __global__ __launch_bounds__(128) void probe(
    const __nv_bfloat16 *__restrict__ Amn, const __nv_bfloat16 *__restrict__ Bkn,
    float *__restrict__ out, int variant) {
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

  // A comes in as A^T row-major, i.e. Amn[k * MM + m] = A[m,k]; m contiguous.
  // Place it with the candidate MN-major encoding.
  for (int i = tid; i < MM * KK / 8; i += 128) {
    int k = i / (MM / 8), mg = i % (MM / 8);
    uint4 val = *(const uint4 *)(Amn + (size_t)k * MM + mg * 8);
    int off;
    switch (variant) {
      case 0: case 4: off = off_mn_a(mg * 8, k, MM); break;
      default:        off = off_mn_b(mg * 8, k, MM); break;
    }
    ((uint4 *)sA)[off] = val;
  }
  // B is (N,K) row-major = B^T, K-major encoding (validated).
  for (int i = tid; i < NN * KK / 8; i += 128) {
    int n = i / (KK / 8), kg = i % (KK / 8);
    ((uint4 *)sB)[off_k(n, kg * 8, KK)] =
        *(const uint4 *)(Bkn + (size_t)n * KK + kg * 8);
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();

  uint32_t id = mk_idesc(NN, MM, 1, 0);
  if (warp == 0 && lane == 0) {
    uint32_t ba = sm_u32(sA), bb = sm_u32(sB);
    for (int ks = 0; ks < KK / MMA_K; ++ks) {
      // Candidate (lbo, sbo, layout_type, per-step byte advance) combinations.
      uint32_t alb, asb, alt, aoff;
      switch (variant) {
        case 0: alb = 8;  asb = MM;      alt = 0; aoff = off_mn_a(0, ks * MMA_K, MM) * 16; break;
        case 1: alb = 8;  asb = MM;      alt = 0; aoff = off_mn_b(0, ks * MMA_K, MM) * 16; break;
        case 2: alb = MM; asb = 8;       alt = 0; aoff = off_mn_b(0, ks * MMA_K, MM) * 16; break;
        case 3: alb = 8;  asb = MM / 8;  alt = 0; aoff = off_mn_b(0, ks * MMA_K, MM) * 16; break;
        case 4: alb = MM; asb = 8;       alt = 0; aoff = off_mn_a(0, ks * MMA_K, MM) * 16; break;
        case 5: alb = 16; asb = MM;      alt = 0; aoff = off_mn_b(0, ks * MMA_K, MM) * 16; break;
        default: alb = 8; asb = MM;      alt = 0; aoff = off_mn_b(0, ks * MMA_K, MM) * 16; break;
      }
      uint64_t da = mk_desc(ba + aoff, alb, asb, alt);
      uint64_t db = mk_desc(bb + off_k(0, ks * MMA_K, KK) * 16, 8, KK, 0);
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
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

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
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
  __syncthreads();
  if (warp == 0)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tc), "r"(NN));
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int variant = argc > 1 ? atoi(argv[1]) : 0;
  auto fa = [](int m, int k) { return (float)((m * 7 + k * 3) % 5 - 2); };
  auto fb = [](int k, int n) { return (float)((k * 11 + n * 5) % 7 - 3); };

  __nv_bfloat16 *hAmn = new __nv_bfloat16[KK * MM];  // A^T row-major (m contiguous)
  __nv_bfloat16 *hBnk = new __nv_bfloat16[NN * KK];  // B^T row-major (k contiguous)
  float *cpu = new float[MM * NN];
  for (int k = 0; k < KK; ++k)
    for (int m = 0; m < MM; ++m) hAmn[k * MM + m] = __float2bfloat16(fa(m, k));
  for (int n = 0; n < NN; ++n)
    for (int k = 0; k < KK; ++k) hBnk[n * KK + k] = __float2bfloat16(fb(k, n));
  for (int m = 0; m < MM; ++m)
    for (int n = 0; n < NN; ++n) {
      float s = 0;
      for (int k = 0; k < KK; ++k) s += fa(m, k) * fb(k, n);
      cpu[m * NN + n] = s;
    }

  __nv_bfloat16 *dA, *dB;
  float *dOut;
  cudaMalloc(&dA, KK * MM * 2);
  cudaMalloc(&dB, NN * KK * 2);
  cudaMalloc(&dOut, 128 * NN * 4);
  cudaMemcpy(dA, hAmn, KK * MM * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hBnk, NN * KK * 2, cudaMemcpyHostToDevice);
  cudaMemset(dOut, 0, 128 * NN * 4);
  size_t smem = MM * KK * 2 + NN * KK * 2 + 1024;
  cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  probe<<<1, 128, smem>>>(dA, dB, dOut, variant);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    printf("variant=%d FAULT %s\n", variant, cudaGetErrorString(e));
    return 1;
  }
  float *hOut = new float[128 * NN];
  cudaMemcpy(hOut, dOut, 128 * NN * 4, cudaMemcpyDeviceToHost);
  double err = 0;
  for (int t = 0; t < 128; ++t)
    for (int j = 0; j < NN; ++j)
      err = fmax(err, fabs((double)hOut[(size_t)t * NN + j] - cpu[t * NN + j]));
  printf("variant=%d max_err=%.1f  %s  got[0][0..3]=%.0f %.0f %.0f %.0f want=%.0f %.0f %.0f %.0f\n",
         variant, err, err == 0 ? "PASS" : "MISMATCH", hOut[0], hOut[1], hOut[2],
         hOut[3], cpu[0], cpu[1], cpu[2], cpu[3]);
  return 0;
}
