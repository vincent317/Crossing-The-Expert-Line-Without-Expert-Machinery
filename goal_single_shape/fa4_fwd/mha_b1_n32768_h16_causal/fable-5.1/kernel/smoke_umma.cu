// tcgen05 GEMM smoke test: C[128x128] = A[128xK] * B, bf16 in, fp32 out.
// Variant 0: B given as Bt[N,K] (K-major)  -> mimics S = Q K^T
// Variant 1: B given as B[K,N]  (MN-major) -> mimics O = P V
// Validates: TMA SW128 loads, UMMA smem/instr descriptors, tmem alloc/ld.
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t *bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t *bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@p bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_u32(bar)), "r"(parity));
}
__device__ __forceinline__ void tma_load_2d(void *dst, const void *tmap, uint64_t *bar, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      ::"r"(smem_u32(dst)), "l"((uint64_t)tmap), "r"(smem_u32(bar)), "r"(x), "r"(y) : "memory");
}
__device__ __forceinline__ void tcgen05_alloc(uint32_t *dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)), "r"(ncols));
}
__device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}
__device__ __forceinline__ void tcgen05_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
}
__device__ __forceinline__ void tcgen05_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::); }
__device__ __forceinline__ void tcgen05_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::); }
__device__ __forceinline__ void tcgen05_commit(uint64_t *bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void umma_bf16_ss(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
      "}\n" ::"r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void tcgen05_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::); }

#define R32(r) \
  "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), \
  "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), \
  "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), \
  "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])

// 32 lanes x 32 columns of 32-bit: thread t gets row (lane t), 32 consecutive columns.
__device__ __forceinline__ void tmem_ld_32x32b_x32(uint32_t taddr, uint32_t *r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : R32(r) : "r"(taddr));
}

// UMMA shared-memory matrix descriptor (SM100), SWIZZLE_128B.
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFFu) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFFu) << 32;
  d |= (uint64_t)1 << 46;   // version = 1 (SM100)
  d |= (uint64_t)2 << 61;   // SWIZZLE_128B
  return d;
}
// UMMA instruction descriptor: bf16 x bf16 -> f32, M=128, N=128.
__host__ __device__ constexpr uint32_t make_idesc(int M, int N, int a_mn_major, int b_mn_major) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn_major << 15) | ((uint32_t)b_mn_major << 16) |
         ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

// ------------------------------------------------------------------ kernel
// smem: [A chunk0 16K][A chunk1 16K][B chunk0 16K][B chunk1 16K][bar][tmem slot]
// A: [128 x 128] K-major, loaded as two 64-col boxes (TMA SW128).
// B variant 0: Bt[128 x 128] K-major (rows = n).  variant 1: B[128(k) x 128(n)] MN-major.
__global__ void __launch_bounds__(128) gemm_smoke(const __grid_constant__ CUtensorMap tmA,
                                                  const __grid_constant__ CUtensorMap tmB,
                                                  float *C, int variant, int lbo_mn, int sbo_mn) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem = (uint8_t *)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t *sA = smem;
  uint8_t *sB = smem + 32768;
  uint64_t *bar = (uint64_t *)(smem + 65536);
  uint32_t *tslot = (uint32_t *)(smem + 65536 + 64);
  uint64_t *mma_bar = (uint64_t *)(smem + 65536 + 128);

  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  if (threadIdx.x == 0) {
    mbar_init(bar, 1);
    mbar_init(mma_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::);
  }
  __syncthreads();
  if (warp == 0) tcgen05_alloc(tslot, 256);
  tcgen05_fence_before();
  __syncthreads();
  tcgen05_fence_after();
  uint32_t tmem = *tslot;

  if (threadIdx.x == 0) {
    mbar_arrive_expect_tx(bar, 65536);
    tma_load_2d(sA, &tmA, bar, 0, 0);
    tma_load_2d(sA + 16384, &tmA, bar, 64, 0);
    tma_load_2d(sB, &tmB, bar, 0, 0);
    tma_load_2d(sB + 16384, &tmB, bar, 64, 0);
  }
  mbar_wait(bar, 0);

  if (threadIdx.x == 0) {
    uint32_t idesc = make_idesc(128, 128, 0, variant == 1 ? 1 : 0);
    for (int kk = 0; kk < 8; ++kk) {
      uint32_t a_addr = smem_u32(sA) + (kk / 4) * 16384 + (kk % 4) * 32;
      uint64_t da = make_smem_desc(a_addr, 16, 1024);
      uint64_t db;
      if (variant == 0) {
        uint32_t b_addr = smem_u32(sB) + (kk / 4) * 16384 + (kk % 4) * 32;
        db = make_smem_desc(b_addr, 16, 1024);
      } else {
        uint32_t b_addr = smem_u32(sB) + kk * 2048;  // 16 k-rows * 128B
        db = make_smem_desc(b_addr, lbo_mn, sbo_mn);
      }
      umma_bf16_ss(tmem, da, db, idesc, kk > 0 ? 1u : 0u);
    }
    tcgen05_commit(mma_bar);
  }
  mbar_wait(mma_bar, 0);
  tcgen05_fence_after();

  // Each warp reads lanes [32*warp, 32*warp+32), 4 x 32 columns.
  for (int c = 0; c < 4; ++c) {
    uint32_t r[32];
    uint32_t taddr = tmem + ((uint32_t)(32 * warp) << 16) + (uint32_t)(32 * c);
    tmem_ld_32x32b_x32(taddr, r);
    tcgen05_wait_ld();
    int row = 32 * warp + lane;
    for (int i = 0; i < 32; ++i) C[row * 128 + 32 * c + i] = __uint_as_float(r[i]);
  }
  tcgen05_fence_before();
  __syncthreads();
  if (warp == 0) { tcgen05_dealloc(tmem, 256); tcgen05_relinquish(); }
}

// ------------------------------------------------------------------ host
typedef CUresult (*PFN_encodeTiled)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                                    const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
                                    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static PFN_encodeTiled get_encode() {
  void *fn = nullptr;
  cudaDriverEntryPointQueryResult qr;
  CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &qr));
  if (!fn) { printf("no cuTensorMapEncodeTiled\n"); exit(1); }
  return (PFN_encodeTiled)fn;
}

// 2D bf16 row-major [rows x cols]; box = 64 cols x 128 rows, SW128.
static CUtensorMap make_tmap(const void *ptr, uint64_t rows, uint64_t cols) {
  CUtensorMap m;
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstride[1] = {cols * 2};
  cuuint32_t box[2] = {64, 128};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)ptr, gdim, gstride, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("cuTensorMapEncodeTiled failed %d\n", (int)r); exit(1); }
  return m;
}

static float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }

int main() {
  const int M = 128, N = 128, K = 128;
  __nv_bfloat16 *hA = new __nv_bfloat16[M * K], *hBt = new __nv_bfloat16[N * K], *hB = new __nv_bfloat16[K * N];
  uint32_t s = 12345;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xFFFF) / 65536.0f - 0.5f; };
  for (int i = 0; i < M * K; ++i) hA[i] = __float2bfloat16(rnd());
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) { float v = rnd(); hBt[n * K + k] = __float2bfloat16(v); hB[k * N + n] = __float2bfloat16(v); }
  float *ref = new float[M * N];
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += (double)bf2f(hA[m * K + k]) * bf2f(hBt[n * K + k]);
      ref[m * N + n] = (float)acc;
    }
  __nv_bfloat16 *dA, *dBt, *dB; float *dC;
  CHECK(cudaMalloc(&dA, M * K * 2)); CHECK(cudaMalloc(&dBt, N * K * 2)); CHECK(cudaMalloc(&dB, K * N * 2));
  CHECK(cudaMalloc(&dC, M * N * 4));
  CHECK(cudaMemcpy(dA, hA, M * K * 2, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dBt, hBt, N * K * 2, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB, K * N * 2, cudaMemcpyHostToDevice));
  CUtensorMap tA = make_tmap(dA, M, K), tBt = make_tmap(dBt, N, K), tB = make_tmap(dB, K, N);

  size_t smem = 65536 + 1024 + 1024;
  CHECK(cudaFuncSetAttribute(gemm_smoke, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  float *hC = new float[M * N];

  auto run = [&](const char *name, const CUtensorMap &tb, int variant, int lbo, int sbo) {
    CHECK(cudaMemset(dC, 0, M * N * 4));
    gemm_smoke<<<1, 128, smem>>>(tA, tb, dC, variant, lbo, sbo);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("%-40s launch error: %s\n", name, cudaGetErrorString(e)); exit(1); }
    CHECK(cudaMemcpy(hC, dC, M * N * 4, cudaMemcpyDeviceToHost));
    double maxerr = 0, maxref = 0;
    for (int i = 0; i < M * N; ++i) { maxerr = fmax(maxerr, fabs(hC[i] - ref[i])); maxref = fmax(maxref, fabs(ref[i])); }
    printf("%-40s maxerr=%.3e (maxref=%.3f) C[0]=%.4f ref[0]=%.4f C[1]=%.4f ref[1]=%.4f C[129]=%.4f ref[129]=%.4f\n",
           name, maxerr, maxref, hC[0], ref[0], hC[1], ref[1], hC[129], ref[129]);
  };
  run("v0 K-major x K-major", tBt, 0, 16, 1024);
  run("v1 MN-major lbo=16384 sbo=1024", tB, 1, 16384, 1024);
  run("v1 MN-major lbo=1024 sbo=16384", tB, 1, 1024, 16384);
  printf("SMOKE_DONE\n");
  return 0;
}
