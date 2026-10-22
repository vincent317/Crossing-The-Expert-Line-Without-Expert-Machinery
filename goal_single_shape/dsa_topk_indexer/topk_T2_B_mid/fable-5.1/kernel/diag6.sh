#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, torch.utils.cpp_extension as ext, os, tempfile
# Try: warp-level parallel heads (one warp handles one token, parallelising H_DIM)
# 64 threads/block, each warp=32 threads handles HALF of 64 heads per token
# 2 tokens per block (2 warps)
# OR: 32 threads per token, token parallelism = bsz/32
CUDA_SRC = r"""
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#define H_DIM 64
#define D_DIM 128
#define PAGE_SZ 64
#define K_STRIDE 132
#define DIM32 32   // D_DIM/4

__device__ __forceinline__ void dot4(uint32_t q4,uint32_t k4,float& acc){
    __half2 qh0=__nv_cvt_fp8x2_to_halfraw2(q4&0xFFFFu,__NV_E4M3);
    __half2 qh1=__nv_cvt_fp8x2_to_halfraw2(q4>>16,__NV_E4M3);
    __half2 kh0=__nv_cvt_fp8x2_to_halfraw2(k4&0xFFFFu,__NV_E4M3);
    __half2 kh1=__nv_cvt_fp8x2_to_halfraw2(k4>>16,__NV_E4M3);
    float2 qf0=__half22float2(qh0),qf1=__half22float2(qh1);
    float2 kf0=__half22float2(kh0),kf1=__half22float2(kh1);
    acc+=qf0.x*kf0.x+qf0.y*kf0.y+qf1.x*kf1.x+qf1.y*kf1.y;
}

// Grid(B, num_pages), block=128 threads
// Each block: 2 tokens (one per warp-pair of 64 threads)
// Each 64-thread group: token t, heads split H=2 groups of 32 heads
// Then warp-shuffle reduces within 32-thread warp
// Score = sum_h relu(dot_h)*wt_h
__global__ void score_warp_heads(
    const uint8_t* __restrict__ q_fp8,
    const int8_t*  __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int*     __restrict__ seq_lens,
    const int*     __restrict__ blk_tbl,
    float*         __restrict__ scores,
    int*           __restrict__ ids,
    int max_pages
){
    // blockDim=128: 4 warps, 2 tokens per block (each token=2 warps)
    int b=blockIdx.x, pg_idx=blockIdx.y;
    int tid=threadIdx.x; // 0..127
    int seq_len=seq_lens[b];
    int t_base=pg_idx*PAGE_SZ;
    if(t_base>=seq_len) return;

    extern __shared__ char smem[];
    uint8_t* q_sh=(uint8_t*)smem;
    float* wt_sh=(float*)(smem+H_DIM*D_DIM);

    // Load q[b] into shared memory
    for(int i=tid; i<H_DIM*D_DIM/4; i+=128)
        ((uint32_t*)q_sh)[i]=((const uint32_t*)(q_fp8+(long)b*H_DIM*D_DIM))[i];
    for(int i=tid; i<H_DIM; i+=128) wt_sh[i]=weights[b*H_DIM+i];
    __syncthreads();

    int page=blk_tbl[b*max_pages+pg_idx];

    // token within page: each pair of 2 warps handles one token
    // warp_in_block = tid/32, token_idx = warp_in_block/2, head_offset = (warp_in_block%2)*32
    int warp_id=tid/32;
    int lane=tid%32;
    int tok_local=warp_id/2;   // 0 or 1
    int head_off=(warp_id%2)*32; // 0 or 32

    int loc=tok_local; // process tokens loc=0,1 per block (just 2 per block)
    // To handle all 64 tokens, we'd need to loop -- but this is a design test
    // Let's instead have each 64-thread-group handle 1 token, loop over tokens
    // Actually: let's use blockDim=64, one warp pair = one token at a time
    // Redesign: blockDim=64, warp0 handles heads 0..31, warp1 handles heads 32..63
    // After each token, both warps collaborate and move to next token

    // For simplicity, each thread in a 64-block handles H_DIM/64=1 head,
    // with full 128-dim dot product per head per thread
    // Then reduce by warp shuffle -- NO, each thread handles 1 head fully.

    // Here: blockDim=64, tid=0..63
    // Each thread handles heads: h = tid (0..63 = all H_DIM heads)
    // But then we need to load k once per token and loop over tokens
    // k is 128 bytes -- fits in registers (32 uint32s)
    // For each token (loop), each thread computes score contribution from its head
    // Then warp-reduce (or block-reduce) to get total score

    // With 64 threads, 1 head per thread: TOTAL token parallelism = ?
    // Each token takes 1 block-step. With PAGE_SZ=64 tokens per block, need 64 steps.
    // But with 64 threads each doing 1 head: all heads computed simultaneously per token.
    // 1 token per step, 64 steps total per block.

    // Actually let's do: each thread handles 1 head, loop over 64 tokens in the page.
    // One __syncthreads per token. 64 tokens → 64 syncs per block.

    // I think the math bottleneck is different. Let's just count:
    // Per token: 64 heads × 32 dot4 calls = 2048 calls.
    // With 64 threads each doing 1 head = 32 dot4 calls per thread per token.
    // 64 tokens × 32 dot4 calls = 2048 dot4 calls per thread total.
    // But we need to BROADCAST k to all threads: k is 128 bytes.
    // With 64 threads, each loads 2 uint32s of k into shared memory.
    // Then __syncthreads, each thread reads all 32 k-uint32s from shared.
    // Shared memory latency × 32 reads per thread per head = 32 × 1 = 32 accesses.
    // But since all 64 threads read THE SAME k data, it's broadcast-friendly.
    // This should be L1-cache-friendly.

    return; // placeholder
}

// REAL implementation: 64 threads/block, 1 head per thread, loop over tokens
__global__ void score_1head_per_thread(
    const uint8_t* __restrict__ q_fp8,
    const int8_t*  __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int*     __restrict__ seq_lens,
    const int*     __restrict__ blk_tbl,
    float*         __restrict__ scores,
    int*           __restrict__ ids,
    int max_pages
){
    int b=blockIdx.x, pg_idx=blockIdx.y;
    int h=threadIdx.x; // 0..63 = head index
    int seq_len=seq_lens[b];
    int t_base=pg_idx*PAGE_SZ;
    if(t_base>=seq_len) return;

    extern __shared__ char smem[];
    // k cache for current token: 128 bytes = 32 uint32s
    uint32_t* k_sh=(uint32_t*)smem;          // 128 bytes
    float* sc_sh=(float*)(smem+128);          // 64 float scores, one per thread
    // q[b,h]: this thread's head data is in global
    // Access: q_fp8 + b*64*128 + h*128
    const uint32_t* qh=(const uint32_t*)(q_fp8+(long)b*H_DIM*D_DIM+h*D_DIM);
    float wt=weights[b*H_DIM+h];

    int page=blk_tbl[b*max_pages+pg_idx];

    for(int loc=0; loc<PAGE_SZ && (t_base+loc)<seq_len; loc++) {
        const uint8_t* kp=(const uint8_t*)k_cache+((long)page*PAGE_SZ+loc)*K_STRIDE;

        // Load k into shared (64 threads each load 2 uint32s = 8 bytes each → 512 bytes, only need 128)
        // Thread h loads 2 uint32s: k_sh[h%32*2] and k_sh[h%32*2+1]
        // Only first 32 threads load (h<32)
        if(h<DIM32) k_sh[h]=((const uint32_t*)kp)[h];
        // Remaining 32 elements
        if(h>=32) k_sh[h-32+32]=((const uint32_t*)kp)[h-32+32]; // = k_sh[h]
        // Actually: each thread loads one uint32 if h<32, and one if h-32>=0
        // Simplified: all 64 threads load, wrapping to 32 entries
        // Thread h: loads k_sh[h & 31] = k[h%32]
        // k has 32 uint32s, each thread loads k[tid % 32]
        k_sh[h&31]=((const uint32_t*)kp)[h&31]; // each of 32 entries written twice (ok, same value)
        __syncthreads();

        // Each thread computes dot for its head
        float d=0.f;
        for(int i=0;i<DIM32;i++) dot4(qh[i],k_sh[i],d);
        float ks=*(const float*)(kp+D_DIM);
        d*=ks;
        sc_sh[h]=(d>0.f?d:0.f)*wt;
        __syncthreads();

        // Thread 0 sums all head contributions
        if(h==0) {
            float total=0.f;
            for(int i=0;i<H_DIM;i++) total+=sc_sh[i];
            int t=t_base+loc;
            scores[(long)b*4096+t]=total;
            ids[(long)b*4096+t]=page*PAGE_SZ+loc;
        }
        __syncthreads();
    }
}

std::tuple<torch::Tensor,torch::Tensor> run_1hpt(
    torch::Tensor q, torch::Tensor k, torch::Tensor w,
    torch::Tensor sl, torch::Tensor bt
){
    int B=q.size(0),mp=bt.size(1);
    auto opts=torch::TensorOptions().device(q.device());
    auto sc=torch::full({B,4096},-1e38f,opts.dtype(torch::kFloat32));
    auto id=torch::full({B,4096},-1,opts.dtype(torch::kInt32));
    dim3 grid(B,mp);
    int smem=128+H_DIM*4; // k_sh+sc_sh
    score_1head_per_thread<<<grid,H_DIM,smem>>>(
        (const uint8_t*)q.data_ptr(),(const int8_t*)k.data_ptr(),
        w.data_ptr<float>(),sl.data_ptr<int>(),bt.data_ptr<int>(),
        sc.data_ptr<float>(),id.data_ptr<int>(),mp);
    return {sc,id};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_1hpt",&run_1hpt);}
"""
tmpdir=tempfile.mkdtemp()
src=os.path.join(tmpdir,"d6.cu")
open(src,"w").write(CUDA_SRC)
mod=ext.load(name="d6_ext",sources=[src],extra_cuda_cflags=["-O3","--use_fast_math"],verbose=False)
import sys; sys.path.insert(0,".")
from bench_all import make_inputs
q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
q_u8=q_fp8.view(torch.uint8).contiguous()
args=(q_u8,k_cache.contiguous(),weights.contiguous(),seq_lens.contiguous(),block_table.contiguous())
buf=torch.empty(512*1024*1024//4,dtype=torch.float32,device="cuda")
def bench(fn,w=5,n=20):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(n):
        buf.fill_(0.); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1000)
    ts.sort(); return ts[len(ts)//2]

print(f"1-head-per-thread: {bench(lambda: mod.run_1hpt(*args)):.2f} us")
PYEOF
