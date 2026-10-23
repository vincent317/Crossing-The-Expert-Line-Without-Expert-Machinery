// Stress-test the mma.sync bf16 peak measurement across the two axes that
// could have made the original single-datapoint probe (551 TFLOPS) an
// artifact rather than a ceiling: accumulator-level ILP and occupancy.
//
// The architectural prediction is 2048 bf16 FLOP/cycle/SM for warp-level MMA
// (the legacy tensor-core path, unchanged since Ampere), i.e.
// 148 SM * 1.965 GHz * 2048 = 596 TFLOPS on this part. If the original probe
// was bottlenecked by register pressure or issue latency, raising NACC or the
// block count should move the number materially. If 551 is the real ceiling,
// every sufficiently-parallel configuration should land just under 596 and
// none should exceed it.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define INNER 8

template <int NACC>
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
  if (s == 123456789.0f) out[0] = s;  // never true; keeps the work alive
}

template <int NACC>
void run(int nsm, int clk_khz, int blocks_per_sm, double theo_tflops) {
  const int threads = 256;
  const int iters = 2000;
  const int blocks = nsm * blocks_per_sm;
  float *out = nullptr;
  cudaMalloc(&out, sizeof(float) * 4);

  mma_peak<NACC><<<blocks, threads>>>(out, 100);
  cudaDeviceSynchronize();

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  mma_peak<NACC><<<blocks, threads>>>(out, iters);
  cudaEventRecord(e1);
  cudaDeviceSynchronize();
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, e0, e1);

  double warps = (double)blocks * threads / 32.0;
  double flop = warps * (double)iters * INNER * NACC * (16.0 * 8.0 * 16.0 * 2.0);
  double tflops = flop / (ms * 1e-3) / 1e12;
  double cyc = ms * 1e-3 * (double)clk_khz * 1e3;
  printf("NACC=%2d blocks/SM=%d warps/SM=%2d : %7.3f ms  %7.1f TFLOPS  "
         "%5.0f FLOP/cycle/SM  %5.1f%% of architectural %.0f  err=%s\n",
         NACC, blocks_per_sm, blocks_per_sm * threads / 32, ms, tflops,
         flop / cyc / nsm, 100.0 * tflops / theo_tflops, theo_tflops,
         cudaGetErrorString(cudaGetLastError()));
  fflush(stdout);
  cudaFree(out);
}

int main() {
  setvbuf(stdout, NULL, _IONBF, 0);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  int clk = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  const int nsm = prop.multiProcessorCount;
  // Legacy warp-level MMA rate: 2048 bf16 FLOP/cycle/SM.
  const double theo = (double)nsm * (clk * 1e3) * 2048.0 / 1e12;
  printf("device=%s SMs=%d clock=%.3f GHz\n", prop.name, nsm, clk * 1e-6);
  printf("architectural warp-MMA prediction: 2048 FLOP/cycle/SM -> %.1f TFLOPS\n\n",
         theo);

  // Vary accumulator-level ILP at fixed occupancy.
  run<2>(nsm, clk, 2, theo);
  run<4>(nsm, clk, 2, theo);
  run<8>(nsm, clk, 2, theo);   // the original configuration
  run<16>(nsm, clk, 2, theo);
  printf("\n");
  // Vary occupancy at the original ILP.
  run<8>(nsm, clk, 1, theo);
  run<8>(nsm, clk, 4, theo);
  printf("\n");
  // High ILP and high occupancy together.
  run<16>(nsm, clk, 4, theo);
  return 0;
}
