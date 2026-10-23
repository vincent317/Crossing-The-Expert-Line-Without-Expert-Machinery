#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, torch.utils.cpp_extension as ext, os, tempfile

# Key question: can we do score + top-k in one shot fast enough?
# Ref: 4.4 µs for 4061 tokens across 8 batches
# Our score multiblock: 78 µs -- still 18x too slow
#
# Root cause: ref does fp8 dot-products, but processes ALL heads in parallel
# using WGMMA/tensor cores on B200 for a GEMV.
# B200 has powerful tensor cores: fp8 GEMM is 18 PFLOPS
#
# Key insight: for each token, we need q(64x128) @ k(128,) = (64,)
# This is a GEMV. On B200 with FP8 warp-group MMA, we can use
# mma.sync or wgmma.
#
# But let's first check: why is even our multiblock slow?
# We do 78 µs for (B*45) = 360 blocks, each 64 threads.
# Each block loads q (8KB from global each time!).
# Total q loads: 360 blocks × 8KB = 2.9 MB redundant!
# (q[b] is 8KB, loaded 45 times for batch 2 alone)
#
# Fix: preload q to L2 by sharing blocks, or cache q in L1 smem.
# Actually: the q_fp8 for batch b is 8KB. Each block independently loads it.
# 360 blocks × 8192 bytes = 2.94 MB from L2.
# At 8TB/s L2: 2.94MB / 8TB = 0.37 µs just for q loads. OK.
# But 360 blocks × 132 bytes/token × 64 tokens = 360 × 8448 = 3MB of k.
# Total L2: ~6MB reads. At 8TB/s: 0.75 µs.
# But actual is 78 µs -- 100x slower than L2 bandwidth.
#
# REAL bottleneck: 64 heads × 32 dot4 calls = 2048 inner ops per token.
# With 64 threads per block processing 64 tokens in parallel:
# each thread does 1 token × 2048 inner ops.
# At 2 TFLOPS per SM (FP32): 2048 ops / (2e12/32 threads) = 33 ns per thread.
# But 64 threads execute simultaneously: wall clock = 33 ns per block.
# 360 blocks on 132 SMs: ceil(360/132)=3 wave × 33 ns = 99 ns.
# Real is 78 µs → 789x slower than theoretical.
# Register spilling or memory latency from within the inner loop!
#
# The issue is the q_sh shared memory: for the h loop, each thread
# accesses q_sh[h*128 .. h*128+127] in shared memory for ALL 64 heads.
# That's 64 × 128 = 8KB of shared memory reads per token.
# L1/shared latency: ~20 cycles. 64 × 32 = 2048 shared mem accesses per token.
# 2048 × 20 cycles / 1.7GHz = 24 µs per block-serial execution!
# With 64 threads in parallel: 24 µs / 64 = 0.375 µs per block.
# With 360 blocks on 132 SMs: 360/132 = 2.7 waves × 0.375 µs = 1 µs.
# Closer to 78 µs but still not matching.
#
# Actually the k_cache global loads may be the bottleneck within the h loop:
# kh32[i] is read from GLOBAL memory every h iteration (64×) since kp doesn't change!
# The compiler may not cache kp in registers across the h loop.
# Fix: load k into a register array (128 bytes = 32 uint32s) before the h loop.

CUDA_SRC = r"""
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

#define H_DIM 64
#define D_DIM 128
#define PAGE_SZ 64
#define K_STRIDE 132
#define DIM32 (D_DIM/4)  // 32

__device__ __forceinline__ void dot4(uint32_t q4, uint32_t k4, float& acc) {
    __half2 qh0=__nv_cvt_fp8x2_to_halfraw2(q4&0xFFFFu,__NV_E4M3);
    __half2 qh1=__nv_cvt_fp8x2_to_halfraw2(q4>>16,__NV_E4M3);
    __half2 kh0=__nv_cvt_fp8x2_to_halfraw2(k4&0xFFFFu,__NV_E4M3);
    __half2 kh1=__nv_cvt_fp8x2_to_halfraw2(k4>>16,__NV_E4M3);
    float2 qf0=__half22float2(qh0),qf1=__half22float2(qh1);
    float2 kf0=__half22float2(kh0),kf1=__half22float2(kh1);
    acc+=qf0.x*kf0.x+qf0.y*kf0.y+qf1.x*kf1.x+qf1.y*kf1.y;
}

// Load k into registers FIRST, then loop over heads
__global__ void score_kregs(
    const uint8_t* __restrict__ q_fp8,
    const int8_t*  __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int*     __restrict__ seq_lens,
    const int*     __restrict__ blk_tbl,
    float*         __restrict__ scores,
    int*           __restrict__ ids,
    int max_pages
) {
    int b=blockIdx.x, pg_idx=blockIdx.y, tid=threadIdx.x, bsz=blockDim.x;
    int seq_len=seq_lens[b];
    int t_base=pg_idx*PAGE_SZ;
    if(t_base>=seq_len) return;

    extern __shared__ char smem[];
    uint8_t* q_sh=(uint8_t*)smem;
    float* wt_sh=(float*)(smem+H_DIM*D_DIM);

    for(int i=tid; i<H_DIM*D_DIM/4; i+=bsz)
        ((uint32_t*)q_sh)[i]=((const uint32_t*)(q_fp8+(long)b*H_DIM*D_DIM))[i];
    for(int i=tid; i<H_DIM; i+=bsz)
        wt_sh[i]=weights[b*H_DIM+i];
    __syncthreads();

    int page=blk_tbl[b*max_pages+pg_idx];

    // Each thread handles one token offset (tid < 64 since bsz=64)
    int loc=tid;
    if(loc < PAGE_SZ && (t_base+loc)<seq_len) {
        const uint8_t* kp=(const uint8_t*)k_cache+((long)page*PAGE_SZ+loc)*K_STRIDE;
        float ks=*(const float*)(kp+D_DIM);

        // Load k into register array to avoid global re-reads in h loop
        uint32_t kreg[DIM32];
        for(int i=0;i<DIM32;i++) kreg[i]=((const uint32_t*)kp)[i];

        float sc=0.f;
        for(int h=0;h<H_DIM;h++){
            const uint32_t* qh=(const uint32_t*)(q_sh+h*D_DIM);
            float d=0.f;
            for(int i=0;i<DIM32;i++) dot4(qh[i],kreg[i],d);
            d*=ks; sc+=(d>0.f?d:0.f)*wt_sh[h];
        }
        int t=t_base+loc;
        scores[(long)b*4096+t]=sc;
        ids[(long)b*4096+t]=page*PAGE_SZ+loc;
    }
}

std::tuple<torch::Tensor,torch::Tensor> run(
    torch::Tensor q, torch::Tensor k, torch::Tensor w,
    torch::Tensor sl, torch::Tensor bt
) {
    int B=q.size(0), mp=bt.size(1);
    auto opts=torch::TensorOptions().device(q.device());
    auto sc=torch::full({B,4096},-1e38f,opts.dtype(torch::kFloat32));
    auto id=torch::full({B,4096},-1,opts.dtype(torch::kInt32));
    dim3 grid(B,mp);
    int smem=H_DIM*D_DIM+H_DIM*4;
    score_kregs<<<grid,64,smem>>>(
        (const uint8_t*)q.data_ptr(),(const int8_t*)k.data_ptr(),
        w.data_ptr<float>(),sl.data_ptr<int>(),bt.data_ptr<int>(),
        sc.data_ptr<float>(),id.data_ptr<int>(),mp);
    return {sc,id};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
"""
tmpdir=tempfile.mkdtemp()
src=os.path.join(tmpdir,"d5.cu")
open(src,"w").write(CUDA_SRC)
mod=ext.load(name="d5_ext",sources=[src],extra_cuda_cflags=["-O3","--use_fast_math",
    "--maxrregcount=64"],verbose=False)

import sys; sys.path.insert(0,".")
from bench_all import make_inputs
q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
q_u8=q_fp8.view(torch.uint8).contiguous()
args=(q_u8,k_cache.contiguous(),weights.contiguous(),seq_lens.contiguous(),block_table.contiguous())

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

t=bench(lambda: mod.run(*args))
print(f"score_kregs (regs, 64t, multiblock): {t:.2f} us")

# Now full pipeline: score + topk via torch
sc, id_t = mod.run(*args)
torch.cuda.synchronize()

def full_pipeline():
    sc2, id2 = mod.run(*args)
    sl_cpu=seq_lens.cpu()
    out=torch.full((8,2048),-1,dtype=torch.int32,device="cuda")
    for b in range(8):
        sl=sl_cpu[b].item()
        k=min(2048,sl)
        if k>0:
            topk_idx=torch.topk(sc2[b,:sl],k).indices
            out[b,:k]=id2[b].index_select(0,topk_idx)
    return out

t2=bench(full_pipeline,w=3,n=20)
print(f"score+topk (full pipeline): {t2:.2f} us")

# Also try: fully on GPU topk via batch topk
def full_gpu():
    sc2, id2 = mod.run(*args)
    # topk over first 2817 columns (conservative)
    _, idx=torch.topk(sc2[:,:2817],2048,dim=1)
    out=torch.gather(id2[:,:2817],1,idx)
    return out

t3=bench(full_gpu,w=3,n=20)
print(f"score+topk (batched topk): {t3:.2f} us")
PYEOF
