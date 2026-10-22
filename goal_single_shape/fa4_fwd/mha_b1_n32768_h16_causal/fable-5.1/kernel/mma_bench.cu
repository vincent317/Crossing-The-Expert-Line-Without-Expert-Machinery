// tcgen05.mma throughput microbenchmark: back-to-back MMAs of several shapes from one thread
// per CTA (1 CTA per SM), fp32 accumulate in tmem, bf16 operands in smem (contents irrelevant).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ bool mbar_try_wait(uint32_t addr, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.b32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(addr), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void tcgen05_alloc(uint32_t *dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)), "r"(ncols));
}
__device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}
__device__ __forceinline__ void tcgen05_commit(uint64_t *bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void umma_bf16(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               ::"r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFFu) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFFu) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
__host__ __device__ constexpr uint32_t make_idesc(int M, int N, int b_mn_major) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)b_mn_major << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

// smem: A region 64KB, B region 64KB (both zero; K-major SW128 layout assumed)
__global__ void __launch_bounds__(32) mma_bench(int N, int iters, int b_mn, unsigned long long *cycles) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t *sm = (uint8_t *)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint64_t *bar = (uint64_t *)(sm + 131072);
  uint32_t *tslot = (uint32_t *)(sm + 131072 + 64);
  if (threadIdx.x == 0) { mbar_init(bar, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::); }
  __syncwarp();
  tcgen05_alloc(tslot, 512);
  asm volatile("tcgen05.fence::before_thread_sync;" ::);
  __syncwarp();
  asm volatile("tcgen05.fence::after_thread_sync;" ::);
  uint32_t tmem = *tslot;
  const uint32_t idesc = make_idesc(128, N, b_mn);
  if (threadIdx.x == 0) {
    // warmup
    for (int i = 0; i < 64; ++i) {
      uint64_t da = make_smem_desc(smem_u32(sm) + (i % 4) * 32, 16, 1024);
      uint64_t db = make_smem_desc(smem_u32(sm + 65536) + (i % 4) * 32, b_mn ? 8192 : 16, 1024);
      umma_bf16(tmem, da, db, idesc, i > 0);
    }
    tcgen05_commit(bar);
    while (!mbar_try_wait(smem_u32(bar), 0)) {}
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; ++i) {
      uint64_t da = make_smem_desc(smem_u32(sm) + (i % 4) * 32, 16, 1024);
      uint64_t db = make_smem_desc(smem_u32(sm + 65536) + (i % 4) * 32, b_mn ? 8192 : 16, 1024);
      umma_bf16(tmem, da, db, idesc, 1);
    }
    tcgen05_commit(bar);
    while (!mbar_try_wait(smem_u32(bar), 1)) {}
    unsigned long long t1 = clock64();
    cycles[blockIdx.x] = t1 - t0;
  }
  __syncwarp();
  asm volatile("tcgen05.fence::before_thread_sync;" ::);
  tcgen05_dealloc(tmem, 512);
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
}

int main() {
  int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  unsigned long long *d_cyc; cudaMalloc(&d_cyc, sms * 8);
  unsigned long long *h_cyc = new unsigned long long[sms];
  size_t smem = 131072 + 2048;
  cudaFuncSetAttribute(mma_bench, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  const int iters = 4096;
  printf("SMs=%d clock=%.3f GHz  (M=128, K=16 per instruction, bf16->f32)\n", sms, clk * 1e-6);
  int shapes[][2] = {{64, 0}, {128, 0}, {256, 0}, {128, 1}, {256, 1}};
  for (auto &s : shapes) {
    int N = s[0], bmn = s[1];
    for (int rep = 0; rep < 2; ++rep) {
      cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
      cudaEventRecord(a);
      mma_bench<<<sms, 32, smem>>>(N, iters, bmn, d_cyc);
      cudaEventRecord(b);
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("N=%d err %s\n", N, cudaGetErrorString(e)); return 1; }
      float ms; cudaEventElapsedTime(&ms, a, b);
      cudaMemcpy(h_cyc, d_cyc, sms * 8, cudaMemcpyDeviceToHost);
      double avg = 0; for (int i = 0; i < sms; ++i) avg += h_cyc[i]; avg /= sms;
      double flop_per_mma = 2.0 * 128 * N * 16;
      double cyc_per_mma = avg / iters;
      double tflops = flop_per_mma * iters * sms / (ms * 1e-3) / 1e12;
      if (rep == 1)
        printf("N=%3d b_mn=%d : %.2f clk/MMA (SM clock64), %.0f FLOP/clk/SM, wall %.3f ms -> %.0f TFLOPS\n",
               N, bmn, cyc_per_mma, flop_per_mma / cyc_per_mma, ms, tflops);
    }
  }
  printf("MMA_BENCH_DONE\n");
  return 0;
}
