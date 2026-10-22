// Microbenchmark: MUFU ex2 throughput (f32 vs packed f16x2) and f32->f16x2 cvt,
// to size the softmax cost budget against tcgen05 MMA on B200.
#include <cstdio>
#include <cuda_fp16.h>
#include <cstdint>

#define LOOP 2000

__global__ void k_base(float *out, float s) {
  float b0 = s, b1 = s + 1, b2 = s + 2, b3 = s + 3, b4 = s + 4, b5 = s + 5, b6 = s + 6, b7 = s + 7;
  float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
  for (int i = 0; i < LOOP; ++i) {
    float r0 = b0, r1 = b1, r2 = b2, r3 = b3, r4 = b4, r5 = b5, r6 = b6, r7 = b7;
    asm volatile("" : "+f"(r0), "+f"(r1), "+f"(r2), "+f"(r3), "+f"(r4), "+f"(r5), "+f"(r6), "+f"(r7));
    a0 += r0; a1 += r1; a2 += r2; a3 += r3; a4 += r4; a5 += r5; a6 += r6; a7 += r7;
  }
  if (s > 1e30f) out[0] = a0+a1+a2+a3+a4+a5+a6+a7;
}

__global__ void k_ex2_f32(float *out, float s) {
  float b0 = s, b1 = s + 1, b2 = s + 2, b3 = s + 3, b4 = s + 4, b5 = s + 5, b6 = s + 6, b7 = s + 7;
  float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
  for (int i = 0; i < LOOP; ++i) {
    float r0, r1, r2, r3, r4, r5, r6, r7;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r0) : "f"(b0));
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r1) : "f"(b1));
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r2) : "f"(b2));
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r3) : "f"(b3));
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r4) : "f"(b4));
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r5) : "f"(b5));
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r6) : "f"(b6));
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r7) : "f"(b7));
    a0 += r0; a1 += r1; a2 += r2; a3 += r3; a4 += r4; a5 += r5; a6 += r6; a7 += r7;
  }
  if (s > 1e30f) out[0] = a0+a1+a2+a3+a4+a5+a6+a7;
}

// 4 x f16x2 = 8 elements per loop iteration, same element count as k_ex2_f32.
__global__ void k_ex2_f16x2(float *out, float s) {
  unsigned b0 = 0x3c003c00u, b1 = 0x40004000u, b2 = 0x42004200u, b3 = 0x44004400u;
  unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (int i = 0; i < LOOP; ++i) {
    unsigned r0, r1, r2, r3;
    asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(r0) : "r"(b0));
    asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(r1) : "r"(b1));
    asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(r2) : "r"(b2));
    asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(r3) : "r"(b3));
    a0 ^= r0; a1 ^= r1; a2 ^= r2; a3 ^= r3;
  }
  if (s > 1e30f) out[0] = (float)(a0 ^ a1 ^ a2 ^ a3);
}

__global__ void k_base_x2(float *out, float s) {
  unsigned b0 = 0x3c003c00u, b1 = 0x40004000u, b2 = 0x42004200u, b3 = 0x44004400u;
  unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (int i = 0; i < LOOP; ++i) {
    unsigned r0 = b0, r1 = b1, r2 = b2, r3 = b3;
    asm volatile("" : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3));
    a0 ^= r0; a1 ^= r1; a2 ^= r2; a3 ^= r3;
  }
  if (s > 1e30f) out[0] = (float)(a0 ^ a1 ^ a2 ^ a3);
}

// 4 x cvt = 8 elements converted per iteration.
__global__ void k_cvt_f16x2(float *out, float s) {
  float b0 = s, b1 = s + 1, b2 = s + 2, b3 = s + 3, b4 = s + 4, b5 = s + 5, b6 = s + 6, b7 = s + 7;
  unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (int i = 0; i < LOOP; ++i) {
    unsigned r0, r1, r2, r3;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r0) : "f"(b0), "f"(b1));
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r1) : "f"(b2), "f"(b3));
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r2) : "f"(b4), "f"(b5));
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r3) : "f"(b6), "f"(b7));
    a0 ^= r0; a1 ^= r1; a2 ^= r2; a3 ^= r3;
  }
  if (s > 1e30f) out[0] = (float)(a0 ^ a1 ^ a2 ^ a3);
}

// 8 FFMA per iteration: the fp32 pipe reference rate.
__global__ void k_ffma(float *out, float s) {
  float a0=s, a1=s+1, a2=s+2, a3=s+3, a4=s+4, a5=s+5, a6=s+6, a7=s+7;
  float m = 1.0000001f, c = 1e-6f;
  for (int i = 0; i < LOOP; ++i) {
    a0 = fmaf(a0, m, c); a1 = fmaf(a1, m, c); a2 = fmaf(a2, m, c); a3 = fmaf(a3, m, c);
    a4 = fmaf(a4, m, c); a5 = fmaf(a5, m, c); a6 = fmaf(a6, m, c); a7 = fmaf(a7, m, c);
  }
  if (s > 1e30f) out[0] = a0+a1+a2+a3+a4+a5+a6+a7;
}

int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int sm = p.multiProcessorCount;
  int clkKHz = 0;
  cudaDeviceGetAttribute(&clkKHz, cudaDevAttrClockRate, 0);
  double clk = clkKHz * 1e3;  // Hz
  printf("device=%s SM=%d nominalClock=%.3f GHz\n", p.name, sm, clk / 1e9);

  float *out;
  cudaMalloc(&out, 4);
  volatile float seed_host = 1.0f;
  float seed = seed_host;  // opaque to the compiler, keeps stores live
  int grid = sm * 4, block = 256;
  double threads = (double)grid * block;

  struct { const char *name; void (*fn)(float *, float); double elems; } tests[] = {
      {"base_f32(8 add)", k_base, 8},
      {"ex2.approx.ftz.f32", k_ex2_f32, 8},
      {"base_x2(4 xor)", k_base_x2, 8},
      {"ex2.approx.f16x2", k_ex2_f16x2, 8},
      {"cvt.rn.f16x2.f32", k_cvt_f16x2, 8},
      {"ffma.f32", k_ffma, 8},
  };
  for (auto &t : tests) {
    for (int w = 0; w < 3; ++w) t.fn<<<grid, block>>>(out, seed);
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    int it = 5;
    for (int i = 0; i < it; ++i) t.fn<<<grid, block>>>(out, seed);
    cudaEventRecord(b);
    cudaDeviceSynchronize();
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    double ms1 = ms / it;
    double ops = threads * LOOP * t.elems;
    double per_sm_clk = ops / (ms1 * 1e-3) / sm / clk;
    printf("%-24s  %8.3f ms   %10.2f elem/clk/SM\n", t.name, ms1, per_sm_clk);
  }
  printf("err=%s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
