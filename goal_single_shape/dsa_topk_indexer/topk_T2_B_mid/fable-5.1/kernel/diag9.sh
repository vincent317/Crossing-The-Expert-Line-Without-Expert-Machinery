#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch
import sys; sys.path.insert(0,".")
from bench_all import make_inputs

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128; PAGE_SZ=64
k_raw=k_cache.view(-1,132)
all_sl=[seq_lens[b].item() for b in range(B)]
max_sl=max(all_sl)  # 2817

buf=torch.empty(512*1024*1024//4,dtype=torch.float32,device="cuda")
def bench(fn,w=5,n=30):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(n):
        buf.fill_(0.); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1000)
    ts.sort(); return ts[len(ts)//2]

# Key insight: gather is slow (845 us for all batches). Why?
# For B=8 batches, sequential Python loop calls tensor indexing operations.
# Let's do a SINGLE fused gather kernel using CUDA.

import torch.utils.cpp_extension as ext, os, tempfile

CUDA_SRC = r"""
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#define H_DIM 64
#define D_DIM 128
#define PAGE_SZ 64
#define K_STRIDE 132

__device__ __forceinline__ void dot4v(uint32_t q4, uint32_t k4, float& acc) {
    __half2 qh0=__nv_cvt_fp8x2_to_halfraw2(q4&0xFFFFu,__NV_E4M3);
    __half2 qh1=__nv_cvt_fp8x2_to_halfraw2(q4>>16,__NV_E4M3);
    __half2 kh0=__nv_cvt_fp8x2_to_halfraw2(k4&0xFFFFu,__NV_E4M3);
    __half2 kh1=__nv_cvt_fp8x2_to_halfraw2(k4>>16,__NV_E4M3);
    float2 qf0=__half22float2(qh0),qf1=__half22float2(qh1);
    float2 kf0=__half22float2(kh0),kf1=__half22float2(kh1);
    acc+=qf0.x*kf0.x+qf0.y*kf0.y+qf1.x*kf1.x+qf1.y*kf1.y;
}

// Fused: gather + score + topk for ALL batches
// Grid: (B, num_pages_per_batch), each block processes 64 tokens (1 page)
// All 8 batches run in parallel
// After scoring, we need to select top-k.
// Since seq_len <= 2048 for 7 batches: just output all.
// For batch 2: write scores to global, then second pass select top-2048.
// 
// To avoid 2 kernels: use cooperative groups? Or just accept 2 kernels.
// For now: kernel1=score (multiblock), kernel2=topk (per-batch block)

// Kernel 1: Compute scores and IDs for all tokens
// Grid: (B, max_pages), Block: 64 threads
__global__ void k1_score(
    const uint8_t* __restrict__ q_fp8,
    const int8_t*  __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int*     __restrict__ seq_lens,
    const int*     __restrict__ blk_tbl,
    float*         __restrict__ sc_global,  // (B, 4096)
    int*           __restrict__ id_global,  // (B, 4096)
    int max_pages
) {
    int b=blockIdx.x, pg_idx=blockIdx.y, tid=threadIdx.x;
    int sl=seq_lens[b], t_base=pg_idx*PAGE_SZ;
    if(t_base>=sl) return;

    extern __shared__ char smem[];
    uint8_t* q_sh=(uint8_t*)smem;
    float* wt_sh=(float*)(smem+H_DIM*D_DIM);

    for(int i=tid; i<H_DIM*D_DIM/4; i+=64)
        ((uint32_t*)q_sh)[i]=((const uint32_t*)(q_fp8+(long)b*H_DIM*D_DIM))[i];
    for(int i=tid; i<H_DIM; i+=64) wt_sh[i]=weights[b*H_DIM+i];
    __syncthreads();

    int pg=blk_tbl[b*max_pages+pg_idx];
    int loc=tid;  // tid in [0,63]
    if(t_base+loc < sl) {
        const uint8_t* kp=(const uint8_t*)k_cache+((long)pg*PAGE_SZ+loc)*K_STRIDE;
        float ks=*(const float*)(kp+D_DIM);
        uint32_t kreg[32];
        for(int i=0;i<32;i++) kreg[i]=((const uint32_t*)kp)[i];
        float sc=0.f;
        for(int h=0;h<H_DIM;h++){
            const uint32_t* qh=(const uint32_t*)(q_sh+h*D_DIM);
            float d=0.f;
            for(int i=0;i<32;i++) dot4v(qh[i],kreg[i],d);
            d*=ks; sc+=(d>0.f?d:0.f)*wt_sh[h];
        }
        int t=t_base+loc;
        sc_global[(long)b*4096+t]=sc;
        id_global[(long)b*4096+t]=pg*PAGE_SZ+loc;
    }
}

// Kernel 2: Top-k selection per batch (only batch 2 needs real top-k)
// Grid: (B,), Block: 256
// For seq_len<=2048: copy ids directly
// For seq_len>2048: find top-2048 via partial bitonic sort on 4096 elements
// But bitonic sort is expensive. Alternative: use warp-parallel threshold scan.
// Fastest: use atomic-based selection or just shared-memory insertion sort.
// For n=2817, finding top-2048: scan once to find threshold, then emit.
// 
// SIMPLE APPROACH: since we need top-2048 of 2817, exclude the bottom 769.
// Step1: warp-parallel min-reduction to find the 769th percentile score
// Step2: output all tokens above threshold
//
// Even simpler for this problem: CUB DeviceSelect::FlaggedIf would work,
// but requires temporary storage. Let's use a shared-memory approach.
// 
// For batch 2 with 2817 scores in global memory:
// - 256 threads each hold 11-12 scores
// - Each thread maintains a local sorted list of its worst scores
// - Block-level merge to find threshold
// This is complex. Instead, let's use the global atomicAdd approach:
// count how many scores are above various thresholds.

// Actually: for goal completion, let me just use thrust in the kernel.
// Or: precompute scores with k1_score, then use torch.topk for k2.

torch::Tensor run_k1(
    torch::Tensor q, torch::Tensor k, torch::Tensor w,
    torch::Tensor sl, torch::Tensor bt
) {
    int B=q.size(0), mp=bt.size(1);
    auto opts=torch::TensorOptions().device(q.device());
    auto sc=torch::full({B,4096},-1e38f,opts.dtype(torch::kFloat32));
    auto id=torch::full({B,4096},-1,opts.dtype(torch::kInt32));
    dim3 grid(B,mp);
    int smem=H_DIM*D_DIM+H_DIM*4;
    k1_score<<<grid,64,smem>>>(
        (const uint8_t*)q.data_ptr(),(const int8_t*)k.data_ptr(),
        w.data_ptr<float>(),sl.data_ptr<int>(),bt.data_ptr<int>(),
        sc.data_ptr<float>(),id.data_ptr<int>(),mp);
    return sc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_k1",&run_k1);}
"""
tmpdir=tempfile.mkdtemp()
src=os.path.join(tmpdir,"d9.cu")
open(src,"w").write(CUDA_SRC)
mod=ext.load(name="d9_ext",sources=[src],extra_cuda_cflags=["-O3","--use_fast_math"],verbose=False)

q_u8=q_fp8.view(torch.uint8).contiguous()
args=(q_u8,k_cache.contiguous(),weights.contiguous(),seq_lens.contiguous(),block_table.contiguous())

print(f"k1_score (multiblock, 64t, k-in-regs): {bench(lambda:mod.run_k1(*args)):.2f} us")

# Full pipeline: k1_score + torch.topk (batched)
def full_v2():
    sc=mod.run_k1(*args)
    # All 8 batches: topk over first max_sl columns
    _,idx=torch.topk(sc[:,:max_sl],2048,dim=1)
    return idx  # (B,2048) indices into [0,max_sl)
print(f"full_v2 (k1 + batched topk 2048): {bench(full_v2,w=3,n=20):.2f} us")
PYEOF
