// Peak throughput microbenchmark for warp-level mma.sync bf16 on the target GPU.
// Decides whether mma.sync alone can reach the FLOPS needed, or whether the
// 5th-gen tensor core path (tcgen05) is required.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define NACC 8
#define INNER 8

__global__ __launch_bounds__(256) void mma_peak(float *out, int iters) {
  uint32_t a[4];
  uint32_t b[2];
  float d[NACC][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 0x3f803f80u;  // two bf16 1.0
  b[0] = 0x3f803f80u;
  b[1] = 0x3f803f80u;
#pragma unroll
  for (int i = 0; i < NACC; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) d[i][j] = 0.0f;

  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < INNER; ++u) {
#pragma unroll
      for (int i = 0; i < NACC; ++i) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      }
    }
  }
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < NACC; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) s += d[i][j];
  if (s == 123456789.0f) out[0] = s;  // never true; keeps work alive
}

int main() {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  int clk = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  printf("device=%s SMs=%d clockRate=%d kHz sharedPerBlockOptin=%zu\n", prop.name,
         prop.multiProcessorCount, clk, (size_t)prop.sharedMemPerBlockOptin);

  float *out = nullptr;
  cudaMalloc(&out, sizeof(float) * 4);

  const int blocks = prop.multiProcessorCount * 2;
  const int threads = 256;
  const int iters = 2000;

  mma_peak<<<blocks, threads>>>(out, 100);
  cudaDeviceSynchronize();

  cudaEvent_t s0, s1;
  cudaEventCreate(&s0);
  cudaEventCreate(&s1);
  cudaEventRecord(s0);
  mma_peak<<<blocks, threads>>>(out, iters);
  cudaEventRecord(s1);
  cudaDeviceSynchronize();
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, s0, s1);

  double warps = (double)blocks * threads / 32.0;
  double flop = warps * (double)iters * INNER * NACC * (16.0 * 8.0 * 16.0 * 2.0);
  printf("mma.sync m16n8k16 bf16: time=%.3f ms  TFLOPS=%.1f\n", ms,
         flop / (ms * 1e-3) / 1e12);
  cudaError_t e = cudaGetLastError();
  printf("cuda_err=%s\n", cudaGetErrorString(e));
  return 0;
}
