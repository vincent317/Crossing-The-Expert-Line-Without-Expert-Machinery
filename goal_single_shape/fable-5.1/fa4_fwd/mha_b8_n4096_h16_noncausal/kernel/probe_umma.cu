// Sustained tcgen05.mma throughput probe: how many bf16 FLOP/cycle/SM does a
// single-CTA (cta_group::1) M=128 N=128 K=16 UMMA stream reach when it is
// the only thing running? This bounds what the attention kernel can achieve
// and calibrates the per-iteration cycle estimates.
#include <cstdio>
#include <cstdint>
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
__device__ __forceinline__ uint64_t mk_desc(uint32_t a, uint32_t lbo, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((a >> 4) & 0x3FFFu);
  d |= (uint64_t)(lbo & 0x3FFFu) << 16;
  d |= (uint64_t)(sbo & 0x3FFFu) << 32;
  d |= (uint64_t)1u << 46;
  return d;
}
__device__ __forceinline__ uint32_t mk_idesc(int n, int m) {
  uint32_t d = 0;
  d |= 1u << 4; d |= 1u << 7; d |= 1u << 10;
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
__device__ __host__ __forceinline__ int lay_a(int mn, int k, int ktot) {
  return (mn & 7) + (mn >> 3) * ktot + (k >> 3) * 8;
}

// NACC independent accumulators of NN columns each, round-robin, so the
// tensor core sees independent chains (as in attention: S and O).
template <int NACC>
__global__ __launch_bounds__(128) void umma_peak(int iters, float *sink) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __nv_bfloat16 *sA = (__nv_bfloat16 *)smem;
  __nv_bfloat16 *sB = (__nv_bfloat16 *)(smem + MM * KK * 2);
  uint32_t *tmp = (uint32_t *)(smem + MM * KK * 2 + NN * KK * 2);
  uint64_t *mbar = (uint64_t *)(tmp + 8);
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < MM * KK; i += 128) {
    sA[i] = __float2bfloat16(0.001f * (i % 7));
    sB[i] = __float2bfloat16(0.001f * (i % 5));
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :: "r"(sm_u32(tmp)), "r"(NACC * NN));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
  }
  if (tid == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(sm_u32(mbar)));
  __syncthreads();
  uint32_t tc = tmp[0];
  uint32_t id = mk_idesc(NN, MM);
  uint32_t ba = sm_u32(sA), bb = sm_u32(sB);
  if (warp == 0 && lane == 0) {
    uint32_t par = 0;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int a = 0; a < NACC; ++a) {
#pragma unroll
        for (int ks = 0; ks < KK / MMA_K; ++ks) {
          uint64_t da = mk_desc(ba + lay_a(0, ks * MMA_K, KK) * 16, 8, KK);
          uint64_t db = mk_desc(bb + lay_a(0, ks * MMA_K, KK) * 16, 8, KK);
          do_mma(tc + a * NN, da, db, id, (it | ks) ? 1 : 0);
        }
      }
      // Bound the in-flight MMA queue: commit + wait every 8 batches.
      if ((it & 7) == 7 || it == iters - 1) {
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                     :: "r"(sm_u32(mbar)));
        asm volatile("{\n\t.reg .pred p;\n\tLW:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra LW;\n\t}\n" :: "r"(sm_u32(mbar)), "r"(par));
        par ^= 1;
      }
    }
  }
  __syncthreads();
  // tcgen05.ld is .sync.aligned: the whole warp must execute it.
  if (warp == 0) {
    uint32_t r;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n" : "=r"(r) : "r"(tc));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
    if (lane == 0) sink[blockIdx.x] = __int_as_float(r);
  }
  __syncthreads();
  if (warp == 0)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tc), "r"(NACC * NN));
}

template <int NACC>
void run(int nsm, int blocks_per_sm, float *sink) {
  size_t smem = MM * KK * 2 + NN * KK * 2 + 1024;
  cudaFuncSetAttribute(umma_peak<NACC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int blocks = nsm * blocks_per_sm;
  int iters = 2000;
  umma_peak<NACC><<<blocks, 128, smem>>>(50, sink);
  cudaDeviceSynchronize();
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  umma_peak<NACC><<<blocks, 128, smem>>>(iters, sink);
  cudaEventRecord(e1);
  cudaDeviceSynchronize();
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  double flop = (double)blocks * iters * NACC * (2.0 * MM * NN * KK);
  int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  double cyc = ms * 1e-3 * clk * 1e3;
  printf("umma_peak NACC=%d blocks/SM=%d: %.3f ms  %.1f TFLOPS  %.0f FLOP/cycle/SM  err=%s\n",
         NACC, blocks_per_sm, ms, flop / (ms * 1e-3) / 1e12, flop / cyc / nsm,
         cudaGetErrorString(cudaGetLastError()));
  fflush(stdout);
}

int main() {
  setvbuf(stdout, NULL, _IONBF, 0);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  float *sink; cudaMalloc(&sink, 4096 * 4);
  run<1>(prop.multiProcessorCount, 1, sink);
  run<2>(prop.multiProcessorCount, 1, sink);
  run<2>(prop.multiProcessorCount, 2, sink);
  run<4>(prop.multiProcessorCount, 1, sink);
  return 0;
}
