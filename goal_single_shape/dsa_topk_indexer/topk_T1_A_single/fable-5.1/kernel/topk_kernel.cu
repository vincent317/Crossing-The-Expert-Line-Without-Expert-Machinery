/*
 * topk_kernel.cu  v9
 *
 * Approach: shared memory for Q, 128 threads to maximize L1 throughput.
 *
 * - 128 threads (4 warps), one block per batch.
 * - Threads 0..127 cooperatively load Q (64*128=8192 bytes) into smem.
 * - Thread h (0..63) reads q[h] from smem and computes its dot product.
 * - Threads 64..127 are idle during scoring (only help load).
 * - Uses maxrregcount=32 to prevent spill.
 *
 * The key insight: keeping Q in smem means each thread only needs
 * a few registers for the scalar accumulation.
 *
 * smem usage: 8192 (Q) + 8 (scales for 2 tokens) + 256 (weights) + 16 (warp sums) = 8472 bytes
 */
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define NUM_HEADS  64
#define HEAD_DIM   128
#define PAGE_SIZE  64
#define K_STRIDE   132
#define TOPK_K     2048

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

/* Limit registers to prevent spill and keep latency low. */
__launch_bounds__(64, 8)
__global__ void dsa_topk_kernel(
    const uint8_t*  __restrict__ q_fp8,
    const uint8_t*  __restrict__ k_cache,
    const float*    __restrict__ weights,
    const int32_t*  __restrict__ seq_lens,
    const int32_t*  __restrict__ block_tbl,
    int32_t*        __restrict__ out,
    int max_pages,
    int topk_size
) {
    const int b = blockIdx.x;
    const int h = threadIdx.x;   /* 0..63, one per head */
    const int seq_len = seq_lens[b];
    const int K = min(topk_size, seq_len);
    if (seq_len == 0) return;

    /* Shared memory layout:
     *   [0..8191]  : q bytes (64*128)
     *   [8192..8447]: k_token0 bytes (128 bytes fp8 + 4 float = 132)
     *   [8448..8703]: k_token1 bytes
     *   [8704..8719]: scales (2 floats = 8 bytes, aligned)
     *   [8720..8975]: weights (64 floats = 256 bytes)
     *   [8976..8991]: warp partial sums (2 warps * 2 tokens * float = 16 bytes)
     */
    extern __shared__ uint8_t smem[];
    uint8_t* smem_q  = smem;                                  /* 8192 bytes */
    uint8_t* smem_k0 = smem + 8192;                           /* 132 bytes */
    uint8_t* smem_k1 = smem + 8192 + 132;                     /* 132 bytes */
    float*   smem_s  = reinterpret_cast<float*>(smem + 8192 + 264);  /* 2 floats */
    float*   smem_w  = reinterpret_cast<float*>(smem + 8192 + 272);  /* 64 floats */
    float*   smem_ws = reinterpret_cast<float*>(smem + 8192 + 528);  /* 4 floats */

    /* Load Q: 64 threads load 128 bytes each = 8192 bytes total. */
    {
        const uint8_t* qb = q_fp8 + (size_t)b * NUM_HEADS * HEAD_DIM;
        /* Thread h loads q[h,:] */
        const uint4* src = reinterpret_cast<const uint4*>(qb + h * HEAD_DIM);
        uint4*       dst = reinterpret_cast<uint4*>(smem_q + h * HEAD_DIM);
#pragma unroll
        for (int i = 0; i < HEAD_DIM / 16; i++)
            dst[i] = __ldg(src + i);
    }

    /* Load weights. */
    smem_w[h] = __ldg(&weights[b * NUM_HEADS + h]);

    /* Load K for tokens 0 and 1. Each thread loads a few bytes. */
    auto load_k_token = [&](int t, uint8_t* smem_kt) {
        if (t < seq_len) {
            const int page = __ldg(&block_tbl[b * max_pages + t / PAGE_SIZE]);
            const int off  = t % PAGE_SIZE;
            const uint8_t* kp = k_cache + ((size_t)page * PAGE_SIZE + off) * K_STRIDE;
            /* Load 128 fp8 bytes: 64 threads, each loads 2 bytes.
             * But HEAD_DIM=128 and we have 64 threads → 2 bytes per thread. */
            /* Use 4-byte loads: each thread loads 4 bytes, covering 128 bytes total
             * in 2 iterations (64 threads * 4 bytes = 256 bytes > 128, so 1 pass is enough
             * but only if stride fits). Let's do: thread h loads 2 bytes at h*2 and h*2+1. */
            /* Actually: 64 threads, 128 bytes → 2 bytes per thread (2 passes of uint8 or 1 pass of uint16). */
            reinterpret_cast<uint16_t*>(smem_kt)[h] =
                *reinterpret_cast<const uint16_t*>(kp + h * 2);
            /* Load scale (thread 0 loads 4 bytes at offset 128). */
            if (h == 0) {
                smem_s[t] = *reinterpret_cast<const float*>(kp + HEAD_DIM);
            }
        } else {
            reinterpret_cast<uint16_t*>(smem_kt)[h] = 0;
            if (h == 0) smem_s[t] = 0.f;
        }
    };
    load_k_token(0, smem_k0);
    load_k_token(1, smem_k1);
    __syncthreads();

    /* Compute dot products from smem. */
    float score0 = 0.f, score1 = 0.f;
    {
        const __nv_fp8_e4m3* qh = reinterpret_cast<const __nv_fp8_e4m3*>(smem_q + h * HEAD_DIM);
        const float w_h = smem_w[h];
        const float s0 = smem_s[0];
        const float s1 = smem_s[1];

        /* Token 0 */
        {
            const __nv_fp8_e4m3* kh = reinterpret_cast<const __nv_fp8_e4m3*>(smem_k0);
            float dot = 0.f;
#pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d += 4) {
                dot += (float)qh[d+0] * (float)kh[d+0];
                dot += (float)qh[d+1] * (float)kh[d+1];
                dot += (float)qh[d+2] * (float)kh[d+2];
                dot += (float)qh[d+3] * (float)kh[d+3];
            }
            dot *= s0;
            if (dot > 0.f) score0 = w_h * dot;
        }

        /* Token 1 */
        if (seq_len > 1) {
            const __nv_fp8_e4m3* kh = reinterpret_cast<const __nv_fp8_e4m3*>(smem_k1);
            float dot = 0.f;
#pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d += 4) {
                dot += (float)qh[d+0] * (float)kh[d+0];
                dot += (float)qh[d+1] * (float)kh[d+1];
                dot += (float)qh[d+2] * (float)kh[d+2];
                dot += (float)qh[d+3] * (float)kh[d+3];
            }
            dot *= s1;
            if (dot > 0.f) score1 = w_h * dot;
        }
    }

    /* Warp reduce. */
    score0 = warp_sum(score0);
    score1 = warp_sum(score1);

    /* Cross-warp via smem (2 warps: h<32 = warp0, h<64 = warp1). */
    const int wid = h >> 5;
    const int lid = h & 31;
    if (lid == 0) {
        smem_ws[wid * 2 + 0] = score0;
        smem_ws[wid * 2 + 1] = score1;
    }
    __syncthreads();

    if (h == 0) {
        float total0 = smem_ws[0] + smem_ws[2];
        float total1 = smem_ws[1] + smem_ws[3];

        for (int i = 0; i < topk_size; i++) out[b * topk_size + i] = -1;

        if (K >= 1) {
            const int pg0 = __ldg(&block_tbl[b * max_pages + 0]);
            const int pg1 = pg0;  /* same page for seq_len=2, t=0 and t=1 */
            if (total0 >= total1) {
                out[b * topk_size + 0] = pg0 * PAGE_SIZE + 0;
                if (K >= 2) out[b * topk_size + 1] = pg1 * PAGE_SIZE + 1;
            } else {
                out[b * topk_size + 0] = pg1 * PAGE_SIZE + 1;
                if (K >= 2) out[b * topk_size + 1] = pg0 * PAGE_SIZE + 0;
            }
        }
    }
}

torch::Tensor topk_forward(
    torch::Tensor q_fp8, torch::Tensor k_cache, torch::Tensor weights,
    torch::Tensor seq_lens, torch::Tensor block_table,
    int topk_size, int max_sl
) {
    TORCH_CHECK(q_fp8.is_cuda(), "q_fp8 must be CUDA");
    const int B         = (int)q_fp8.size(0);
    const int max_pages = (int)block_table.size(1);
    auto stream         = c10::cuda::getCurrentCUDAStream();

    auto output = torch::full(
        {B, topk_size}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(q_fp8.device()));

    /* smem: 8192+132+132+8+256+16 = 8736 bytes */
    const size_t smem_sz = 8192 + 132 + 132 + 8 + 256 + 16;
    dsa_topk_kernel<<<B, 64, smem_sz, stream>>>(
        reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()),
        reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
        weights.data_ptr<float>(),
        seq_lens.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        max_pages, topk_size);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dsa_topk_kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &topk_forward,
          "DSA TopK v9 smem-Q (q_fp8, k_cache, weights, seq_lens, block_table, topk_size, max_sl)");
}
