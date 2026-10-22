// Step 1: validate tcgen05 (5th-gen tensor core) bf16 MMA end to end, and
// empirically discover the TMEM accumulator -> (thread, register) mapping.
//
// Computes C[128,128] = A[128,128] @ B[128,128]^T using 8 x MMA-K=16 UMMA
// instructions, with A and B in SWIZZLE_NONE core-matrix smem layout.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define M 128
#define N 128
#define K 128
#define MMA_K 16

// ---------------------------------------------------------------- descriptors

// SWIZZLE_NONE UMMA smem descriptor.
// Bit layout (from PTX/CUTLASS): addr[0,14) lbo[16,30) sbo[32,46) ver[46,48)
// base_off[49,52) lbo_mode[52,53) layout_type[61,64). All offsets in 16B units.
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr, uint32_t lbo,
                                              uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((smem_addr >> 4) & 0x3FFF) << 0;
  d |= (uint64_t)(lbo & 0x3FFF) << 16;
  d |= (uint64_t)(sbo & 0x3FFF) << 32;
  d |= (uint64_t)1u << 46;  // version = 1 (Blackwell)
  d |= (uint64_t)0u << 49;  // base_offset = 0
  d |= (uint64_t)0u << 52;  // lbo_mode = legacy
  d |= (uint64_t)0u << 61;  // layout_type = SWIZZLE_NONE
  return d;
}

// Instruction descriptor for tcgen05.mma.kind::f16.
// c_format[4,6)=1(F32) a_format[7,10) b_format[10,13) a_major[15,16)
// b_major[16,17) n_dim[17,23)=N>>3 m_dim[24,29)=M>>4
__device__ __forceinline__ uint32_t make_idesc(int n, int m, int a_major,
                                              int b_major) {
  uint32_t d = 0;
  d |= 1u << 4;                       // c_format = F32
  d |= 1u << 7;                       // a_format = BF16
  d |= 1u << 10;                      // b_format = BF16
  d |= (uint32_t)a_major << 15;       // 0 = K-major
  d |= (uint32_t)b_major << 16;
  d |= (uint32_t)(n >> 3) << 17;
  d |= (uint32_t)(m >> 4) << 24;
  return d;
}

__device__ __forceinline__ void umma_f16(uint32_t tmem_c, uint64_t da,
                                         uint64_t db, uint32_t idesc,
                                         uint32_t scaleC) {
#if VARIANT == 0
  // plain form: no output-lane mask
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"(scaleC));
#else
  // masked form, as emitted by CUTLASS SM100_MMA_F16BF16_SS
  uint32_t mask[4] = {0, 0, 0, 0};
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"(scaleC), "r"(mask[0]),
        "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
#endif
}

__device__ __forceinline__ void umma_commit(uint64_t *mbar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
      "[%0];" ::"r"((uint32_t)__cvta_generic_to_shared(mbar)));
}

// --------------------------------------------------------------- smem layouts

// SWIZZLE_NONE core-matrix layout, in 16B (u128) units.
// K-major (A/Q/K): 8 consecutive MN rows share a core matrix, 8 K-values per u128.
__device__ __host__ __forceinline__ int off_kmajor(int mn, int k, int k_total) {
  return (mn & 7) + (mn >> 3) * (k_total / 8 * 8) + (k >> 3) * 8;
}
// MN-major (B/V for P@V): 8 consecutive MN values per u128.
__device__ __host__ __forceinline__ int off_mnmajor(int mn, int k, int k_total) {
  return (mn >> 3) * (k_total / 8 * 8) + (k & 7) + (k >> 3) * 8;
}

extern "C" __global__ __launch_bounds__(128) void t5_gemm(
    const __nv_bfloat16 *__restrict__ A, const __nv_bfloat16 *__restrict__ B,
    float *__restrict__ out_raw, float *__restrict__ out_ref) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __nv_bfloat16 *sA = (__nv_bfloat16 *)smem;                  // 128*128*2 = 32KB
  __nv_bfloat16 *sB = (__nv_bfloat16 *)(smem + M * K * 2);    // 32KB
  uint32_t *tmem_ptr = (uint32_t *)(smem + M * K * 2 + N * K * 2);
  uint64_t *mbar = (uint64_t *)(smem + M * K * 2 + N * K * 2 + 16);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // TMEM alloc: whole warp 0 must execute (.sync.aligned).
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(tmem_ptr)), "r"(N));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
  }
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 :
                 : "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  }
  __syncthreads();
  uint32_t tmem_c = tmem_ptr[0];

  // Cooperative global->smem copy into core-matrix layout, 16B at a time.
  // Each u128 of a K-major tile is 8 contiguous K values of one row => a
  // contiguous 16B chunk of the row-major global tile.
  for (int i = tid; i < M * K / 8; i += 128) {
    int mn = i / (K / 8), kg = i % (K / 8);
    const uint4 *src = (const uint4 *)(A + mn * K + kg * 8);
    ((uint4 *)sA)[off_kmajor(mn, kg * 8, K)] = *src;
  }
  for (int i = tid; i < N * K / 8; i += 128) {
    int mn = i / (K / 8), kg = i % (K / 8);
    const uint4 *src = (const uint4 *)(B + mn * K + kg * 8);
    ((uint4 *)sB)[off_kmajor(mn, kg * 8, K)] = *src;
  }
  __syncthreads();

  // 8 UMMA instructions, one per MMA-K=16 slice, accumulating into TMEM.
  const uint32_t idesc = make_idesc(N, M, 0, 0);
  if (warp == 0) {
    if (__builtin_expect(lane == 0, 0)) {
      uint32_t baseA = (uint32_t)__cvta_generic_to_shared(sA);
      uint32_t baseB = (uint32_t)__cvta_generic_to_shared(sB);
      for (int ks = 0; ks < K / MMA_K; ++ks) {
        // advance by one MMA-K=16 slice = 2 core matrices = 32B*... :
        // off_kmajor(0, ks*16, K) u128 units
        uint32_t oa = baseA + off_kmajor(0, ks * MMA_K, K) * 16;
        uint32_t ob = baseB + off_kmajor(0, ks * MMA_K, K) * 16;
        uint64_t da = make_desc(oa, 8, K / 8 * 8);
        uint64_t db = make_desc(ob, 8, K / 8 * 8);
        umma_f16(tmem_c, da, db, idesc, ks == 0 ? 0 : 1);
      }
      umma_commit(mbar);
    }
  }
  // wait for MMA completion
  if (tid == 0) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "waitLoop:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
        "@!p bra waitLoop;\n\t"
        "}\n" ::"r"((uint32_t)__cvta_generic_to_shared(mbar)));
  }
  __syncthreads();

  // Read back TMEM: each warp reads its own 32 lanes, 32 columns at a time.
  // tmem address = (lane_base << 16) | column
  uint32_t addr = tmem_c + (uint32_t)(warp * 32 << 16);
  uint32_t r[32];
  for (int cblk = 0; cblk < N / 32; ++cblk) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "[%32];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]),
          "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
          "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]),
          "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
          "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
          "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]),
          "=r"(r[30]), "=r"(r[31])
        : "r"(addr + cblk * 32));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
#pragma unroll
    for (int j = 0; j < 32; ++j)
      out_raw[(size_t)tid * N + cblk * 32 + j] = __int_as_float(r[j]);
  }
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :
                 : "r"(tmem_c), "r"(N));
  }
  // fp32 reference C = A @ B^T computed cooperatively
  for (int i = tid; i < M * N; i += 128) {
    int m = i / N, n = i % N;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
      acc += __bfloat162float(A[m * K + k]) * __bfloat162float(B[n * K + k]);
    out_ref[i] = acc;
  }
}

int main(int argc, char **argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 0;
  __nv_bfloat16 *hA = new __nv_bfloat16[M * K], *hB = new __nv_bfloat16[N * K];
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) hA[m * K + k] = __float2bfloat16(m == k ? 1.f : 0.f);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      hB[n * K + k] = __float2bfloat16(mode == 0 ? (float)n : (float)k);

  __nv_bfloat16 *dA, *dB;
  float *dRaw, *dRef;
  cudaMalloc(&dA, M * K * 2);
  cudaMalloc(&dB, N * K * 2);
  cudaMalloc(&dRaw, 128 * N * 4);
  cudaMalloc(&dRef, M * N * 4);
  cudaMemcpy(dA, hA, M * K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, N * K * 2, cudaMemcpyHostToDevice);

  size_t smem = M * K * 2 + N * K * 2 + 1024;
  cudaFuncSetAttribute(t5_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  t5_gemm<<<1, 128, smem>>>(dA, dB, dRaw, dRef);
  cudaError_t e = cudaDeviceSynchronize();
  printf("mode=%d launch_err=%s\n", mode, cudaGetErrorString(e));

  float *hRaw = new float[128 * N], *hRef = new float[M * N];
  cudaMemcpy(hRaw, dRaw, 128 * N * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hRef, dRef, M * N * 4, cudaMemcpyDeviceToHost);

  // Print raw TMEM readout for a few threads so the mapping is visible.
  for (int t : {0, 1, 2, 31, 32, 33, 64, 127}) {
    printf("RAW mode=%d tid=%3d :", mode, t);
    for (int j = 0; j < 8; ++j) printf(" %.0f", hRaw[(size_t)t * N + j]);
    printf(" | j=32..35:");
    for (int j = 32; j < 36; ++j) printf(" %.0f", hRaw[(size_t)t * N + j]);
    printf("\n");
  }
  printf("REF C[0][0..7] = ");
  for (int j = 0; j < 8; ++j) printf("%.0f ", hRef[j]);
  printf("\nREF C[1][0..7] = ");
  for (int j = 0; j < 8; ++j) printf("%.0f ", hRef[N + j]);
  printf("\n");
  return 0;
}
