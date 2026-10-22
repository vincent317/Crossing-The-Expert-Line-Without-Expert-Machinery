// Debug harness for fa_fwd2.cu: several small configs vs CPU fp32 reference.
#ifndef NO_DEBUG_PRINT
#define FA_DEBUG 1
#endif
#include "fa_fwd2.cu"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

static int run_case(const char *name, std::vector<int> seqlens, int H) {
  int T = 0; for (int L : seqlens) T += L;
  const int D = 128;
  std::vector<__nv_bfloat16> hq(T * H * D), hk(T * H * D), hv(T * H * D);
  std::vector<float> fq(T * H * D), fk(T * H * D), fv(T * H * D);
  uint32_t s = 777;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xFFFF) / 65536.0f * 2.f - 1.f; };
  for (int i = 0; i < T * H * D; ++i) { hq[i] = __float2bfloat16(rnd()); hk[i] = __float2bfloat16(rnd()); hv[i] = __float2bfloat16(rnd());
    fq[i] = __bfloat162float(hq[i]); fk[i] = __bfloat162float(hk[i]); fv[i] = __bfloat162float(hv[i]); }
  const float scale = 1.f / sqrtf(128.f);
  std::vector<float> ref(T * H * D, 0.f);
  {
    std::vector<float> sc;
    int off = 0;
    for (int L : seqlens) {
      sc.resize(L);
      for (int h = 0; h < H; ++h)
        for (int i = 0; i < L; ++i) {
          const float *qi = &fq[((size_t)(off + i) * H + h) * D];
          float m = -1e30f;
          for (int j = 0; j <= i; ++j) { const float *kj = &fk[((size_t)(off + j) * H + h) * D]; float d = 0; for (int k = 0; k < D; ++k) d += qi[k] * kj[k]; sc[j] = d * scale; m = fmaxf(m, sc[j]); }
          float l = 0; for (int j = 0; j <= i; ++j) { sc[j] = expf(sc[j] - m); l += sc[j]; }
          float *o = &ref[((size_t)(off + i) * H + h) * D];
          for (int j = 0; j <= i; ++j) { const float *vj = &fv[((size_t)(off + j) * H + h) * D]; for (int k = 0; k < D; ++k) o[k] += sc[j] * vj[k]; }
          for (int k = 0; k < D; ++k) o[k] /= l;
        }
      off += L;
    }
  }
  std::vector<WorkItem> items;
  {
    int off = 0;
    for (int L : seqlens) {
      int nq = (L + QTILE - 1) / QTILE;
      for (int qi = 0; qi < nq; ++qi) {
        int nk = std::min(((qi + 1) * QTILE) / KTILE, (L + KTILE - 1) / KTILE);
        for (int h = 0; h < H; ++h) items.push_back({off + qi * QTILE, off, L, h, qi, nk});
      }
      off += L;
    }
    std::sort(items.begin(), items.end(), [](const WorkItem &a, const WorkItem &b) { return a.nk > b.nk; });
  }
  __nv_bfloat16 *dq, *dk, *dv, *dout; WorkItem *dw;
  size_t nb = (size_t)T * H * D * 2;
  cudaMalloc(&dq, nb); cudaMalloc(&dk, nb); cudaMalloc(&dv, nb); cudaMalloc(&dout, nb);
  cudaMalloc(&dw, items.size() * sizeof(WorkItem));
  cudaMemcpy(dq, hq.data(), nb, cudaMemcpyHostToDevice);
  cudaMemcpy(dk, hk.data(), nb, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), nb, cudaMemcpyHostToDevice);
  cudaMemcpy(dw, items.data(), items.size() * sizeof(WorkItem), cudaMemcpyHostToDevice);
  cudaMemset(dout, 0, nb);
  printf("##### %s: T=%d H=%d items=%zu\n", name, T, H, items.size());
  int rc = fa_fwd_launch(dq, dk, dv, dout, dw, (int)items.size(), T, H, scale, 0);
  cudaError_t e = cudaDeviceSynchronize();
  printf("launch rc=%d sync=%s\n", rc, cudaGetErrorString(e));
  if (e != cudaSuccess || rc) return 1;
  std::vector<__nv_bfloat16> hout(T * H * D);
  cudaMemcpy(hout.data(), dout, nb, cudaMemcpyDeviceToHost);
  double maxerr = 0; int bad = 0;
  for (int i = 0; i < T * H; ++i) {
    double re = 0;
    for (int k = 0; k < D; ++k) re = fmax(re, fabs(__bfloat162float(hout[(size_t)i * D + k]) - ref[(size_t)i * D + k]));
    if (re > 0.05) { if (bad < 6) printf("  tok %d head %d maxerr %.4f (out[0]=%.4f ref[0]=%.4f)\n", i / H, i % H, re, __bfloat162float(hout[(size_t)i * D]), ref[(size_t)i * D]); bad++; }
    maxerr = fmax(maxerr, re);
  }
  printf("DBG_RESULT %s maxerr=%.5f bad_rows=%d/%d\n", name, maxerr, bad, T * H);
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dw);
  return bad > 0;
}

int main(int argc, char **argv) {
  int which = argc > 1 ? atoi(argv[1]) : -1;
  int fails = 0;
  if (which < 0 || which == 0) fails += run_case("L128_H1", {128}, 1);
  if (which < 0 || which == 1) fails += run_case("L384_H1", {384}, 1);
  if (which < 0 || which == 2) fails += run_case("L300_H2", {300}, 2);
  if (which < 0 || which == 3) fails += run_case("docs200_700_H2", {200, 700}, 2);
  if (which < 0 || which == 4) fails += run_case("L1500_H1", {1500}, 1);
  printf("ALL_CASES_DONE fails=%d\n", fails);
  return fails;
}
