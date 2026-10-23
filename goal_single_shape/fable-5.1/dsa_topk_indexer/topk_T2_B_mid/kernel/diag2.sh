#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already satisfied" || true
python -u - << 'PYEOF'
import torch
import torch.utils.cpp_extension as ext
import os, tempfile

# Diagnose: is the score loop slow per-token or per-head?
CUDA_SRC = r"""
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

#define H_DIM 64
#define D_DIM 128
#define PAGE_SZ 64
#define K_STRIDE 132

// Variant A: scalar fp8->float conversion (no half2)
__device__ __forceinline__ float fp8f(uint8_t x) {
    return __half2float(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}

__global__ void score_scalar(
    const uint8_t* __restrict__ q_fp8,
    const int8_t*  __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int*     __restrict__ seq_lens,
    const int*     __restrict__ blk_tbl,
    float* __restrict__ out, int max_pages
) {
    int b=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    int seq_len=seq_lens[b];
    extern __shared__ char smem[];
    uint8_t* q_sh=(uint8_t*)smem;
    float*   wt_sh=(float*)(smem+H_DIM*D_DIM);
    for (int i=tid;i<H_DIM*D_DIM;i+=bsz) q_sh[i]=q_fp8[(long)b*H_DIM*D_DIM+i];
    for (int i=tid;i<H_DIM;i+=bsz) wt_sh[i]=weights[b*H_DIM+i];
    __syncthreads();
    for (int t=tid;t<seq_len;t+=bsz) {
        int page=blk_tbl[b*max_pages+t/PAGE_SZ];
        int off=t%PAGE_SZ;
        const uint8_t* kp=(const uint8_t*)k_cache+((long)page*PAGE_SZ+off)*K_STRIDE;
        float ks=*(const float*)(kp+D_DIM);
        float sc=0.f;
        for (int h=0;h<H_DIM;h++) {
            const uint8_t* qh=q_sh+h*D_DIM;
            float d=0.f;
            for (int i=0;i<D_DIM;i++) d+=fp8f(qh[i])*fp8f(kp[i]);
            d*=ks;
            sc+=(d>0.f?d:0.f)*wt_sh[h];
        }
        out[(long)b*4096+t]=sc;
    }
}

// Variant B: half2 vectorized but accumulate in half (risk overflow) -- skip
// Variant C: accumulate in float2 with half2 convert
__device__ __forceinline__ void dot4_fp32(uint32_t q4, uint32_t k4, float& acc) {
    __half2 qh0=__nv_cvt_fp8x2_to_halfraw2(q4&0xFFFFu,__NV_E4M3);
    __half2 qh1=__nv_cvt_fp8x2_to_halfraw2(q4>>16,__NV_E4M3);
    __half2 kh0=__nv_cvt_fp8x2_to_halfraw2(k4&0xFFFFu,__NV_E4M3);
    __half2 kh1=__nv_cvt_fp8x2_to_halfraw2(k4>>16,__NV_E4M3);
    float2 qf0=__half22float2(qh0),qf1=__half22float2(qh1);
    float2 kf0=__half22float2(kh0),kf1=__half22float2(kh1);
    acc+=qf0.x*kf0.x+qf0.y*kf0.y+qf1.x*kf1.x+qf1.y*kf1.y;
}

__global__ void score_half2(
    const uint8_t* __restrict__ q_fp8,
    const int8_t*  __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int*     __restrict__ seq_lens,
    const int*     __restrict__ blk_tbl,
    float* __restrict__ out, int max_pages
) {
    int b=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    int seq_len=seq_lens[b];
    extern __shared__ char smem[];
    uint8_t* q_sh=(uint8_t*)smem;
    float*   wt_sh=(float*)(smem+H_DIM*D_DIM);
    for (int i=tid;i<H_DIM*D_DIM/4;i+=bsz)
        ((uint32_t*)q_sh)[i]=((const uint32_t*)(q_fp8+(long)b*H_DIM*D_DIM))[i];
    for (int i=tid;i<H_DIM;i+=bsz) wt_sh[i]=weights[b*H_DIM+i];
    __syncthreads();
    for (int t=tid;t<seq_len;t+=bsz) {
        int page=blk_tbl[b*max_pages+t/PAGE_SZ];
        int off=t%PAGE_SZ;
        const uint8_t* kp=(const uint8_t*)k_cache+((long)page*PAGE_SZ+off)*K_STRIDE;
        float ks=*(const float*)(kp+D_DIM);
        float sc=0.f;
        for (int h=0;h<H_DIM;h++) {
            const uint32_t* qh32=(const uint32_t*)(q_sh+h*D_DIM);
            const uint32_t* kh32=(const uint32_t*)kp;
            float d=0.f;
            for (int i=0;i<D_DIM/4;i++) dot4_fp32(qh32[i],kh32[i],d);
            d*=ks;
            sc+=(d>0.f?d:0.f)*wt_sh[h];
        }
        out[(long)b*4096+t]=sc;
    }
}

torch::Tensor run_scalar(torch::Tensor q,torch::Tensor k,torch::Tensor w,torch::Tensor sl,torch::Tensor bt) {
    int B=q.size(0),mp=bt.size(1);
    auto o=torch::zeros({B,4096},torch::dtype(torch::kFloat).device(q.device()));
    score_scalar<<<B,256,H_DIM*D_DIM+H_DIM*4>>>(
        (const uint8_t*)q.data_ptr(),(const int8_t*)k.data_ptr(),
        w.data_ptr<float>(),sl.data_ptr<int>(),bt.data_ptr<int>(),o.data_ptr<float>(),mp);
    return o;
}
torch::Tensor run_half2v(torch::Tensor q,torch::Tensor k,torch::Tensor w,torch::Tensor sl,torch::Tensor bt) {
    int B=q.size(0),mp=bt.size(1);
    auto o=torch::zeros({B,4096},torch::dtype(torch::kFloat).device(q.device()));
    score_half2<<<B,256,H_DIM*D_DIM+H_DIM*4>>>(
        (const uint8_t*)q.data_ptr(),(const int8_t*)k.data_ptr(),
        w.data_ptr<float>(),sl.data_ptr<int>(),bt.data_ptr<int>(),o.data_ptr<float>(),mp);
    return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("run_scalar",&run_scalar);
    m.def("run_half2v",&run_half2v);
}
"""

tmpdir=tempfile.mkdtemp()
src=os.path.join(tmpdir,"diag2.cu")
open(src,"w").write(CUDA_SRC)
mod=ext.load(name="diag2_ext",sources=[src],extra_cuda_cflags=["-O3","--use_fast_math"],verbose=False)

from safetensors.torch import load_file
import sys; sys.path.insert(0,".")
from bench_all import make_inputs

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
q_u8=q_fp8.view(torch.uint8).contiguous()
args=(q_u8,k_cache.contiguous(),weights.contiguous(),seq_lens.contiguous(),block_table.contiguous())

buf=torch.empty(512*1024*1024//4,dtype=torch.float32,device="cuda")

def bench(fn,warmup=5,iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times=[]
    for _ in range(iters):
        buf.fill_(0.); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e)*1000)
    times.sort(); return times[len(times)//2]

print(f"scalar fp8->f32   : {bench(lambda: mod.run_scalar(*args)):.2f} us")
print(f"half2 vectorised  : {bench(lambda: mod.run_half2v(*args)):.2f} us")

# Check register usage
import subprocess
nvcc_out=subprocess.run(
    ["nvcc","--ptxas-options=-v","-O3","--use_fast_math",
     "-gencode=arch=compute_100,code=sm_100","-c",src,"-o","/tmp/diag2.o"],
    capture_output=True,text=True
)
for line in nvcc_out.stderr.split("\n"):
    if "registers" in line: print("NVCC reg info:",line.strip())
PYEOF
