#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, torch.utils.cpp_extension as ext, os, tempfile, subprocess
import sys; sys.path.insert(0,".")
from bench_all import make_inputs

# Register count check using a standalone .cu without torch extension
CUDA_STANDALONE = r"""
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#define H_DIM 64
#define D_DIM 128
#define PAGE_SZ 64
#define K_STRIDE 132
#define DIM32 32
#define TOPK_K 2048
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
__global__ void fused_topk(
    const uint8_t* q_fp8, const int8_t* k_cache, const float* weights,
    const int* seq_lens, const int* blk_tbl, int* out, int max_pages
){
    int b=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x, sl=seq_lens[b];
    extern __shared__ char smem[];
    uint8_t* q_sh=(uint8_t*)smem; float* wt_sh=(float*)(smem+H_DIM*D_DIM);
    float* sc_sh=(float*)(smem+H_DIM*D_DIM+H_DIM*4);
    int* id_sh=(int*)(smem+H_DIM*D_DIM+H_DIM*4+BSORT_N*4);
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
        sc_sh[t]=score; id_sh[t]=pg*PAGE_SZ+off;
    }
    for(int t=sl+tid;t<BSORT_N;t+=bsz){sc_sh[t]=-FLT_MAX;id_sh[t]=-1;}
    __syncthreads();
    int* out_b=out+(long)b*TOPK_K;
    if(sl<=TOPK_K){
        for(int t=tid;t<sl;t+=bsz) out_b[t]=id_sh[t];
        for(int t=sl+tid;t<TOPK_K;t+=bsz) out_b[t]=-1;
    } else {
        for(int k=2;k<=BSORT_N;k<<=1) for(int j=k>>1;j>0;j>>=1){
            for(int i=tid;i<BSORT_N;i+=bsz){
                int ixj=i^j;
                if(ixj>i){bool sw=((i&k)==0)?(sc_sh[i]<sc_sh[ixj]):(sc_sh[i]>sc_sh[ixj]);
                    if(sw){float ts=sc_sh[i];sc_sh[i]=sc_sh[ixj];sc_sh[ixj]=ts;
                           int ti=id_sh[i];id_sh[i]=id_sh[ixj];id_sh[ixj]=ti;}}
            }
            __syncthreads();
        }
        for(int t=tid;t<TOPK_K;t+=bsz) out_b[t]=id_sh[t];
    }
}
int main(){return 0;}
"""
tmpdir=tempfile.mkdtemp()
src=os.path.join(tmpdir,"sa.cu")
open(src,"w").write(CUDA_STANDALONE)
r=subprocess.run(["nvcc","--ptxas-options=-v","-O3","--use_fast_math",
    "-gencode=arch=compute_100,code=sm_100",
    "-I/usr/local/cuda/include","-c",src,"-o","/tmp/sa.o"],
    capture_output=True,text=True)
for line in (r.stdout+r.stderr).split("\n"):
    if line.strip() and ("register" in line or "ptxas" in line.lower() or "spill" in line.lower()):
        print(line.strip())

# Also test: are we having shared mem bank conflicts?
# With kr[32] in registers: 32 uint32 = 32 regs just for kr
# With 256 threads: 32 * 256 = 8192 register slots for kr alone
# Per-thread total: kr(32) + score(1) + d(1) + h,t,b,tid,bsz,pg,off,ks,kp,qh + bitonic vars
# ptxas will tell us.

# THE REAL PROBLEM: kr[32] is a 32-element local array.
# On GPU, large local arrays go to local memory (device DRAM).
# 32 * 4 = 128 bytes of local memory per thread for kr alone.
# With 256 threads: 32 KB of local memory per block.
# Local memory accesses are like L2 cache - much slower than registers.
# This is causing the 900 us overhead!

# FIX: Don't use local array for kr. Instead, stream k data directly in inner loop.
# The compiler CANNOT keep 128 bytes in registers when there are 256 threads.
# Available registers per thread with 256 threads: 65536/256 = 256 max,
# but compiler needs many for computation. kr[32] = 32 extra regs => too many.

# Solution: Load k data directly in the inner loop (don't pre-cache in kr[]).
# The compiler will cache it in L1 cache automatically.
# This avoids local memory for kr but increases global memory accesses for k.
# Actually: since we access k in the h loop (64 times), the FIRST access
# loads from L2 (fast) and subsequent accesses hit L1 cache.
print("\n--- Testing without kr register array ---")

CUDA_NOKR = r"""
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
#define TOPK_K 2048
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
// NO kr[] array: load k directly from global (kp ptr) every head iteration
// Relies on L1 cache hitting for re-used k data across 64 heads
__global__ void fused_nokr(
    const uint8_t* __restrict__ q_fp8,
    const int8_t* __restrict__ k_cache,
    const float* __restrict__ weights,
    const int* __restrict__ seq_lens,
    const int* __restrict__ blk_tbl,
    int* __restrict__ out, int max_pages
){
    int b=blockIdx.x,tid=threadIdx.x,bsz=blockDim.x,sl=seq_lens[b];
    extern __shared__ char smem[];
    uint8_t* q_sh=(uint8_t*)smem; float* wt_sh=(float*)(smem+H_DIM*D_DIM);
    float* sc_sh=(float*)(smem+H_DIM*D_DIM+H_DIM*4);
    int* id_sh=(int*)(smem+H_DIM*D_DIM+H_DIM*4+BSORT_N*4);
    for(int i=tid;i<H_DIM*D_DIM/4;i+=bsz)
        ((uint32_t*)q_sh)[i]=((const uint32_t*)(q_fp8+(long)b*H_DIM*D_DIM))[i];
    for(int i=tid;i<H_DIM;i+=bsz) wt_sh[i]=weights[b*H_DIM+i];
    __syncthreads();
    for(int t=tid;t<sl;t+=bsz){
        int pg=blk_tbl[b*max_pages+t/PAGE_SZ],off=t%PAGE_SZ;
        const uint32_t* kp32=(const uint32_t*)((const uint8_t*)k_cache+((long)pg*PAGE_SZ+off)*K_STRIDE);
        float ks=*(const float*)(kp32+DIM32);
        float score=0.f;
        for(int h=0;h<H_DIM;h++){
            const uint32_t* qh=(const uint32_t*)(q_sh+h*D_DIM); float d=0.f;
            // Load k directly each time (rely on L1 for repeated access)
            for(int i=0;i<DIM32;i++) dot4v(qh[i],kp32[i],d);
            d*=ks; score+=(d>0.f?d:0.f)*wt_sh[h];
        }
        sc_sh[t]=score; id_sh[t]=pg*PAGE_SZ+off;
    }
    for(int t=sl+tid;t<BSORT_N;t+=bsz){sc_sh[t]=-FLT_MAX;id_sh[t]=-1;}
    __syncthreads();
    int* out_b=out+(long)b*TOPK_K;
    if(sl<=TOPK_K){
        for(int t=tid;t<sl;t+=bsz) out_b[t]=id_sh[t];
        for(int t=sl+tid;t<TOPK_K;t+=bsz) out_b[t]=-1;
    } else {
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
}
torch::Tensor run_nokr(torch::Tensor q,torch::Tensor k,torch::Tensor w,torch::Tensor sl,torch::Tensor bt){
    int B=q.size(0),mp=bt.size(1);
    auto out=torch::full({B,TOPK_K},-1,torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
    int smem=H_DIM*D_DIM+H_DIM*4+BSORT_N*4*2;
    fused_nokr<<<B,256,smem>>>((const uint8_t*)q.data_ptr(),(const int8_t*)k.data_ptr(),
        w.data_ptr<float>(),sl.data_ptr<int>(),bt.data_ptr<int>(),out.data_ptr<int>(),mp);
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_nokr",&run_nokr);}
"""
src2=os.path.join(tmpdir,"nokr.cu"); open(src2,"w").write(CUDA_NOKR)
mod=ext.load(name="nokr_ext",sources=[src2],extra_cuda_cflags=["-O3","--use_fast_math"],verbose=False)

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
q_u8=q_fp8.view(torch.uint8).contiguous()
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

t=bench(lambda: mod.run_nokr(*args))
p=prof_span(lambda: mod.run_nokr(*args))
print(f"no-kr-array: events={t:.2f} us, profiler={p:.2f} us")

# Also check register count for nokr
src_sa2=os.path.join(tmpdir,"sa2.cu")
open(src_sa2,"w").write(CUDA_NOKR.replace('#include <torch/extension.h>','').replace(
    'torch::Tensor run_nokr','int dummy_fn').replace(
    'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_nokr",&run_nokr);}','int main(){return 0;}'))
r2=subprocess.run(["nvcc","--ptxas-options=-v","-O3","--use_fast_math",
    "-gencode=arch=compute_100,code=sm_100","-I/usr/local/cuda/include",
    "-c",src_sa2,"-o","/tmp/sa2.o"],capture_output=True,text=True)
for line in (r2.stdout+r2.stderr).split("\n"):
    if line.strip() and ("register" in line.lower() or "spill" in line.lower() or "ptxas" in line.lower()):
        print("nokr ptxas:", line.strip())
PYEOF
