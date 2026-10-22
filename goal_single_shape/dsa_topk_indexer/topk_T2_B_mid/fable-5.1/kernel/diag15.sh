#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, torch.utils.cpp_extension as ext, os, tempfile, subprocess
import sys; sys.path.insert(0,".")
from bench_all import make_inputs

# 159 registers with kr[] = NO SPILLS. But 159 regs * 256 threads = 40704 > 65536/2 = 32768.
# Actually B200 has 65536 registers per SM. 159 * 256 = 40704 regs.
# This limits occupancy: only 1 block per SM (can't fit 2 blocks).
# With 1 block per SM and 8 SMs: 8 blocks concurrently.
# But occupancy = 1/2 → reduced warp-level hiding of memory latency.
# 
# The real issue: 890 us for our kernel vs 4.4 us ref.
# We're 200x slower. With 159 regs causing low occupancy, it's even worse.
#
# NEW STRATEGY: Since kr[] has NO SPILLS (all fits in 159 regs), the issue is
# something else. Let me think about what takes 890 us.
# 
# Batch 2: 2817 tokens, 256 threads, ceil(2817/256) = 11 rounds.
# Each round processes 256 tokens × (64 heads × 32 dot4 calls).
# dot4 call: 4 fp8→half2 conversions + 4 half2→float2 + 4 muls + 4 adds = ~12 ops
# Per token per round: 64 × 32 × 12 = 24576 ops
# Per 256 threads: parallel, so wall-clock per round: 24576/256 = 96 ops per thread
# At 1 op/cycle and 1.7 GHz: 96/1.7G = 56 ns per round
# 11 rounds: 616 ns = 0.6 us
# BITONIC SORT: 4096 elements, 78 steps × (4096/256) = 78 × 16 = 1248 iterations per step
# Wait no: each step has BSORT_N/2 = 2048 pairs, 256 threads do 2048/256 × 1 each = 8 per step
# But BSORT_N loop is over all BSORT_N=4096, with ixj>i condition → only half active
# Effective: 78 steps × 4096/(256*2) = 78 × 8 = 624 iterations per thread
# Per iteration: 2 smem reads + compare + conditional 2 writes = ~8 ops
# 624 × 8 = 4992 ops per thread
# At 1 op/ns: 4992/1.7G = 2.9 us
# TOTAL: 0.6 + 2.9 = 3.5 us. But actual is 890 us!
#
# Something is MASSIVELY wrong with my estimate. Either:
# 1. The smem bandwidth is the bottleneck (bank conflicts?)
# 2. The register pressure (159 regs) causes severe occupancy loss

# Let me check: with 159 regs and 256 threads:
# Max blocks per SM = floor(65536 / (159 * 256)) = floor(65536 / 40704) = 1 block
# Warps per block: 256/32 = 8 warps
# Active warps = 8. SM needs ≥ 16 for good occupancy on typical GPUs.
# LOW OCCUPANCY = poor memory latency hiding = SLOW.
# 
# FIX: Reduce register count to allow more blocks.
# With 96 regs: 96 * 256 = 24576 regs → floor(65536/24576) = 2 blocks → 16 warps. Better.
# With 64 regs: 64 * 256 = 16384 → floor(65536/16384) = 4 blocks → 32 warps. 
#
# But how to reduce regs? The kr[32] array contributes 32 regs.
# Without kr[]: 159 - 32 = 127 regs → still only 1 block per SM.
# Need to reduce to < 65536/(2*256) = 128 regs for 2 blocks.
# 127 < 128 → barely 2 blocks! That's why nokr was the same speed (still 1 block).
# Wait: floor(65536/(127*256)) = floor(65536/32512) = 2 blocks. So 16 warps!
# But it was SLOWER (3566 us vs 890 us). Why?
# 
# The nokr version is slower because it re-loads k data 64 times from L2/global
# instead of once from registers. Each re-load = L2 latency.
# The kr[] version DOES fit in registers (no spills) and is faster despite lower occupancy.
#
# So the 890 us is the TRUE performance of this algorithm approach.
# Let's verify by running the score-only part.

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
q_u8=q_fp8.view(torch.uint8).contiguous()

CUDA_SCOREONLY = r"""
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <torch/extension.h>
#define H_DIM 64
#define D_DIM 128
#define PAGE_SZ 64
#define K_STRIDE 132
#define DIM32 32
#define BSORT_N 4096
__device__ __forceinline__ void dot4v(uint32_t q4,uint32_t k4,float& acc){
    __half2 qh0=__nv_cvt_fp8x2_to_halfraw2(q4&0xFFFFu,__NV_E4M3);
    __half2 qh1=__nv_cvt_fp8x2_to_halfraw2(q4>>16,__NV_E4M3);
    __half2 kh0=__nv_cvt_fp8x2_to_halfraw2(k4&0xFFFFu,__NV_E4M3);
    __half2 kh1=__nv_cvt_fp8x2_to_halfraw2(k4>>16,__NV_E4M3);
    float2 qf0=__half22float2(qh0),qf1=__half22float2(qh1);
    float2 kf0=__half22float2(kh0),kf1=__half22float2(kh1);
    acc+=qf0.x*kf0.x+qf0.y*kf0.y+qf1.x*kf1.x+qf1.y*kf1.y;
}
__global__ void score_only(
    const uint8_t* __restrict__ q_fp8, const int8_t* __restrict__ k_cache,
    const float* __restrict__ weights, const int* __restrict__ seq_lens,
    const int* __restrict__ blk_tbl, float* __restrict__ sc_out, int max_pages
){
    int b=blockIdx.x,tid=threadIdx.x,bsz=blockDim.x,sl=seq_lens[b];
    extern __shared__ char smem[];
    uint8_t* q_sh=(uint8_t*)smem; float* wt_sh=(float*)(smem+H_DIM*D_DIM);
    for(int i=tid;i<H_DIM*D_DIM/4;i+=bsz)
        ((uint32_t*)q_sh)[i]=((const uint32_t*)(q_fp8+(long)b*H_DIM*D_DIM))[i];
    for(int i=tid;i<H_DIM;i+=bsz) wt_sh[i]=weights[b*H_DIM+i];
    __syncthreads();
    for(int t=tid;t<sl;t+=bsz){
        int pg=blk_tbl[b*max_pages+t/PAGE_SZ],off=t%PAGE_SZ;
        const uint8_t* kp=(const uint8_t*)k_cache+((long)pg*PAGE_SZ+off)*K_STRIDE;
        float ks=*(const float*)(kp+D_DIM);
        uint32_t kr[DIM32];
        for(int i=0;i<DIM32;i++) kr[i]=((const uint32_t*)kp)[i];
        float score=0.f;
        for(int h=0;h<H_DIM;h++){
            const uint32_t* qh=(const uint32_t*)(q_sh+h*D_DIM); float d=0.f;
            for(int i=0;i<DIM32;i++) dot4v(qh[i],kr[i],d);
            d*=ks; score+=(d>0.f?d:0.f)*wt_sh[h];
        }
        sc_out[(long)b*BSORT_N+t]=score;
    }
}
// Sort-only (scores pre-computed in global memory)
#define TOPK_K 2048
__global__ void sort_only(float* sc_sh_g, int* id_sh_g, int sl, int* out_b){
    int tid=threadIdx.x,bsz=blockDim.x;
    extern __shared__ char smem[];
    float* sc_sh=(float*)smem; int* id_sh=(int*)(smem+BSORT_N*4);
    for(int i=tid;i<sl;i+=bsz){sc_sh[i]=sc_sh_g[i];id_sh[i]=id_sh_g[i];}
    for(int i=sl+tid;i<BSORT_N;i+=bsz){sc_sh[i]=-FLT_MAX;id_sh[i]=-1;}
    __syncthreads();
    for(int k=2;k<=BSORT_N;k<<=1) for(int j=k>>1;j>0;j>>=1){
        for(int i=tid;i<BSORT_N;i+=bsz){int ixj=i^j;
            if(ixj>i){bool sw=((i&k)==0)?(sc_sh[i]<sc_sh[ixj]):(sc_sh[i]>sc_sh[ixj]);
                if(sw){float ts=sc_sh[i];sc_sh[i]=sc_sh[ixj];sc_sh[ixj]=ts;
                       int ti=id_sh[i];id_sh[i]=id_sh[ixj];id_sh[ixj]=ti;}}
        }
        __syncthreads();
    }
    for(int t=tid;t<TOPK_K;t+=bsz) out_b[t]=id_sh[t];
}
std::tuple<torch::Tensor,torch::Tensor> run_score(torch::Tensor q,torch::Tensor k,torch::Tensor w,torch::Tensor sl,torch::Tensor bt){
    int B=q.size(0),mp=bt.size(1);
    auto opts=torch::TensorOptions().device(q.device());
    auto sc=torch::full({B,BSORT_N},-1e38f,opts.dtype(torch::kFloat32));
    auto id=torch::full({B,BSORT_N},-1,opts.dtype(torch::kInt32));
    score_only<<<B,256,H_DIM*D_DIM+H_DIM*4>>>((const uint8_t*)q.data_ptr(),(const int8_t*)k.data_ptr(),
        w.data_ptr<float>(),sl.data_ptr<int>(),bt.data_ptr<int>(),sc.data_ptr<float>(),mp);
    return {sc,id};
}
torch::Tensor run_sort(torch::Tensor sc, torch::Tensor id, int sl){
    auto out=torch::full({TOPK_K},-1,torch::TensorOptions().dtype(torch::kInt32).device(sc.device()));
    sort_only<<<1,256,BSORT_N*8>>>(sc.data_ptr<float>(),id.data_ptr<int>(),sl,out.data_ptr<int>());
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_score",&run_score);m.def("run_sort",&run_sort);}
"""
tmpdir=tempfile.mkdtemp()
src=os.path.join(tmpdir,"so.cu"); open(src,"w").write(CUDA_SCOREONLY)
mod=ext.load(name="so_ext",sources=[src],extra_cuda_cflags=["-O3","--use_fast_math"],verbose=False)
args=(q_u8,k_cache.contiguous(),weights.contiguous(),seq_lens.contiguous(),block_table.contiguous())
buf=torch.empty(512*1024*1024//4,dtype=torch.float32,device="cuda")

def bench(fn,w=10,n=50):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(n):
        buf.fill_(0.); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1000)
    ts.sort(); return ts[len(ts)//2]

import torch.profiler as P
def prof_span(fn,n=5):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    buf.fill_(0.); torch.cuda.synchronize()
    with P.profile(activities=[P.ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            buf.fill_(0.); fn()
    torch.cuda.synchronize()
    evts=prof.profiler.function_events
    total=sum(e.device_time for e in evts if e.device_time>0 and "fill" not in e.name.lower())
    return total/n

t_sc=bench(lambda: mod.run_score(*args))
p_sc=prof_span(lambda: mod.run_score(*args))
print(f"score_only (256t,1block/batch): events={t_sc:.2f} us, profiler={p_sc:.2f} us")

# Sort only (batch 2)
sl2=seq_lens[2].item()
sc2,id2=mod.run_score(*args)
t_sort=bench(lambda: mod.run_sort(sc2[2].contiguous(),id2[2].contiguous(),sl2))
p_sort=prof_span(lambda: mod.run_sort(sc2[2].contiguous(),id2[2].contiguous(),sl2))
print(f"sort_only (batch2, 2817->2048): events={t_sort:.2f} us, profiler={p_sort:.2f} us")

# Combined (score + sort for just batch 2)
sc_dummy,id_dummy=mod.run_score(*args)
def combined():
    sc,id=mod.run_score(*args)
    return mod.run_sort(sc[2].contiguous(),id[2].contiguous(),sl2)
t_comb=bench(combined)
p_comb=prof_span(combined)
print(f"combined (score+sort batch2): events={t_comb:.2f} us, profiler={p_comb:.2f} us")
PYEOF
