// Standalone debug harness: tiny problem (1 doc, L=384, H=1) with printf tracing for block 0.
#define FA_DEBUG 1
#include "fa_fwd.cu"
#include <vector>
#include <cmath>
#include <cstdlib>

int main(int argc, char **argv) {
  const int L = argc > 1 ? atoi(argv[1]) : 384, H = 1, T = L, D = 128;
  std::vector<__nv_bfloat16> hq(T * H * D), hk(T * H * D), hv(T * H * D);
  std::vector<float> fq(T * D), fk(T * D), fv(T * D);
  uint32_t s = 777;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xFFFF) / 65536.0f * 2.f - 1.f; };
  for (int i = 0; i < T * D; ++i) { float a = rnd(), b = rnd(), c = rnd(); hq[i] = __float2bfloat16(a); hk[i] = __float2bfloat16(b); hv[i] = __float2bfloat16(c);
    fq[i] = __bfloat162float(hq[i]); fk[i] = __bfloat162float(hk[i]); fv[i] = __bfloat162float(hv[i]); }
  const float scale = 1.f / sqrtf(128.f);
  // CPU reference
  std::vector<float> ref(T * D, 0.f), sc(T);
  for (int i = 0; i < T; ++i) {
    float m = -1e30f;
    for (int j = 0; j <= i; ++j) { float d = 0; for (int k = 0; k < D; ++k) d += fq[i * D + k] * fk[j * D + k]; sc[j] = d * scale; m = fmaxf(m, sc[j]); }
    float l = 0; for (int j = 0; j <= i; ++j) { sc[j] = expf(sc[j] - m); l += sc[j]; }
    for (int j = 0; j <= i; ++j) for (int k = 0; k < D; ++k) ref[i * D + k] += sc[j] * fv[j * D + k];
    for (int k = 0; k < D; ++k) ref[i * D + k] /= l;
  }
  std::vector<WorkItem> items;
  int nq = (L + QTILE - 1) / QTILE;
  for (int qi = nq - 1; qi >= 0; --qi) items.push_back({qi * QTILE, 0, L, 0, qi});
  __nv_bfloat16 *dq, *dk, *dv, *dout; WorkItem *dw;
  cudaMalloc(&dq, T * D * 2); cudaMalloc(&dk, T * D * 2); cudaMalloc(&dv, T * D * 2); cudaMalloc(&dout, T * D * 2);
  cudaMalloc(&dw, items.size() * sizeof(WorkItem));
  cudaMemcpy(dq, hq.data(), T * D * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dk, hk.data(), T * D * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), T * D * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dw, items.data(), items.size() * sizeof(WorkItem), cudaMemcpyHostToDevice);
  cudaMemset(dout, 0, T * D * 2);
  printf("launch: L=%d items=%zu smem=%d\n", L, items.size(), SM_TOTAL + 1024);
  int rc = fa_fwd_launch(dq, dk, dv, dout, dw, (int)items.size(), T, H, scale, 0);
  printf("launch rc=%d\n", rc);
  cudaError_t e = cudaDeviceSynchronize();
  printf("sync: %s\n", cudaGetErrorString(e));
  if (e != cudaSuccess) return 1;
  std::vector<__nv_bfloat16> hout(T * D);
  cudaMemcpy(hout.data(), dout, T * D * 2, cudaMemcpyDeviceToHost);
  double maxerr = 0; int bad_rows = 0;
  for (int i = 0; i < T; ++i) { double re = 0; for (int k = 0; k < D; ++k) re = fmax(re, fabs(__bfloat162float(hout[i * D + k]) - ref[i * D + k])); if (re > 0.05) { if (bad_rows < 8) printf("  row %d maxerr %.4f (out[0]=%.4f ref[0]=%.4f)\n", i, re, __bfloat162float(hout[i*D]), ref[i*D]); bad_rows++; } maxerr = fmax(maxerr, re); }
  printf("DBG_RESULT maxerr=%.5f bad_rows=%d/%d\n", maxerr, bad_rows, T);
  return 0;
}
