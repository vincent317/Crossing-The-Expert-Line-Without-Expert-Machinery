// v3 hang locator: host-mapped trace buffer polled while the kernel runs.
#define FA_TRACE 1
#define NO_DEBUG_PRINT 1
#include "fa_fwd3.cu"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <unistd.h>

int main(int argc, char **argv) {
  const int L = argc > 1 ? atoi(argv[1]) : 128, H = 1, T = L, D = 128;
  std::vector<__nv_bfloat16> hq(T * H * D), hk(T * H * D), hv(T * H * D);
  uint32_t s = 777;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xFFFF) / 65536.0f * 2.f - 1.f; };
  for (int i = 0; i < T * H * D; ++i) { hq[i] = __float2bfloat16(rnd()); hk[i] = __float2bfloat16(rnd()); hv[i] = __float2bfloat16(rnd()); }
  std::vector<WorkItem> items;
  int nq = (L + QTILE - 1) / QTILE;
  for (int qi = 0; qi < nq; ++qi) { int nk = std::min(((qi + 1) * QTILE) / KTILE, (L + KTILE - 1) / KTILE); items.push_back({qi * QTILE, 0, L, 0, qi, nk}); }
  __nv_bfloat16 *dq, *dk, *dv, *dout; WorkItem *dw;
  size_t nb = (size_t)T * H * D * 2;
  cudaMalloc(&dq, nb); cudaMalloc(&dk, nb); cudaMalloc(&dv, nb); cudaMalloc(&dout, nb);
  cudaMalloc(&dw, items.size() * sizeof(WorkItem));
  cudaMemcpy(dq, hq.data(), nb, cudaMemcpyHostToDevice); cudaMemcpy(dk, hk.data(), nb, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), nb, cudaMemcpyHostToDevice); cudaMemcpy(dw, items.data(), items.size() * sizeof(WorkItem), cudaMemcpyHostToDevice);
  unsigned int *h_trace; cudaHostAlloc(&h_trace, 148 * 16 * 4, cudaHostAllocMapped);
  memset(h_trace, 0, 148 * 16 * 4);
  unsigned int *d_trace; cudaHostGetDevicePointer(&d_trace, h_trace, 0);
  printf("set_trace rc=%d\n", fa_fwd_set_trace(d_trace));
  cudaStream_t st; cudaStreamCreate(&st);
  int rc = fa_fwd_launch(dq, dk, dv, dout, dw, (int)items.size(), T, H, 1.f / sqrtf(128.f), st);
  printf("launch rc=%d items=%zu nk0=%d\n", rc, items.size(), items[0].nk);
  cudaEvent_t ev; cudaEventCreate(&ev); cudaEventRecord(ev, st);
  for (int t = 0; t < 50; ++t) {
    if (cudaEventQuery(ev) == cudaSuccess) { printf("kernel finished after %d polls: %s\n", t, cudaGetErrorString(cudaGetLastError())); return 0; }
    usleep(100000);
  }
  printf("HUNG after 5s. per-warp last trace code (all blocks with activity):\n");
  for (int b = 0; b < std::min((int)items.size(), 8); ++b) {
    printf("  block %d:", b);
    for (int w = 0; w < 9; ++w) printf(" w%d=0x%03x", w, h_trace[b * 16 + w]);
    printf("\n");
  }
  printf("TRACE_DONE\n");
  _exit(2);
}
