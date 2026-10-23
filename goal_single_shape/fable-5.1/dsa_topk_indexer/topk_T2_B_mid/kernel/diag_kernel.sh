#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already satisfied" || true
python -u - << 'PYEOF'
import torch
import torch.utils.cpp_extension as ext
import os, tempfile

# Minimal diagnostic: time just the score loop (no sort) vs full kernel
CUDA_DIAG = r"""
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <torch/extension.h>

#define H_DIM    64
#define D_DIM    128
#define PAGE_SZ  64
#define K_STRIDE 132

__device__ __forceinline__ float dot4(uint32_t q4, uint32_t k4) {
    __half2 qh0 = __nv_cvt_fp8x2_to_halfraw2(q4 & 0xFFFFu, __NV_E4M3);
    __half2 qh1 = __nv_cvt_fp8x2_to_halfraw2(q4 >> 16, __NV_E4M3);
    __half2 kh0 = __nv_cvt_fp8x2_to_halfraw2(k4 & 0xFFFFu, __NV_E4M3);
    __half2 kh1 = __nv_cvt_fp8x2_to_halfraw2(k4 >> 16, __NV_E4M3);
    float2 qf0=__half22float2(qh0), qf1=__half22float2(qh1);
    float2 kf0=__half22float2(kh0), kf1=__half22float2(kh1);
    return qf0.x*kf0.x+qf0.y*kf0.y+qf1.x*kf1.x+qf1.y*kf1.y;
}

/* Score-only kernel (no sort) — test score compute time */
__global__ void score_only_kernel(
    const uint8_t* __restrict__ q_fp8,
    const int8_t*  __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int*     __restrict__ seq_lens,
    const int*     __restrict__ blk_tbl,
    float*         __restrict__ score_out,
    int max_pages
) {
    int b   = blockIdx.x;
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    int seq_len = seq_lens[b];

    extern __shared__ char smem[];
    uint8_t* q_sh  = (uint8_t*)smem;
    float*   wt_sh = (float*)(smem + H_DIM*D_DIM);

    for (int i=tid; i < H_DIM*D_DIM/4; i+=bsz)
        ((uint32_t*)q_sh)[i] = ((const uint32_t*)(q_fp8+(long)b*H_DIM*D_DIM))[i];
    for (int i=tid; i < H_DIM; i+=bsz)
        wt_sh[i] = weights[b*H_DIM+i];
    __syncthreads();

    for (int t=tid; t<seq_len; t+=bsz) {
        int page   = blk_tbl[b*max_pages + t/PAGE_SZ];
        int pg_off = t % PAGE_SZ;
        const uint8_t* kp = (const uint8_t*)k_cache + ((long)page*PAGE_SZ+pg_off)*K_STRIDE;
        float kscale = *(const float*)(kp+D_DIM);

        float score=0.0f;
        for (int h=0; h<H_DIM; h++) {
            const uint32_t* qh32=(const uint32_t*)(q_sh+h*D_DIM);
            const uint32_t* kh32=(const uint32_t*)kp;
            float dot=0.0f;
            for (int d=0; d<D_DIM/4; d++) dot+=dot4(qh32[d],kh32[d]);
            dot*=kscale;
            score+=(dot>0.0f?dot:0.0f)*wt_sh[h];
        }
        score_out[(long)b*seq_len+t]=score;
    }
}

/* Bitonic-sort-only kernel (scores pre-computed) */
#define BSORT_N 4096
#define TOPK_K  2048

__global__ void sort_only_kernel(float* sc, int* ids, int seq_len) {
    int tid=threadIdx.x, bsz=blockDim.x;
    // Descending bitonic sort on sc[0..BSORT_N-1] in-place
    for (int k=2; k<=BSORT_N; k<<=1) {
        for (int j=k>>1; j>0; j>>=1) {
            for (int i=tid; i<BSORT_N; i+=bsz) {
                int ixj=i^j;
                if (ixj>i) {
                    bool sw;
                    if ((i&k)==0) sw=sc[i]<sc[ixj];
                    else          sw=sc[i]>sc[ixj];
                    if (sw) {
                        float ts=sc[i]; sc[i]=sc[ixj]; sc[ixj]=ts;
                        int ti=ids[i]; ids[i]=ids[ixj]; ids[ixj]=ti;
                    }
                }
            }
            __syncthreads();
        }
    }
}

torch::Tensor run_score_only(
    torch::Tensor q_u8, torch::Tensor k_cache,
    torch::Tensor weights, torch::Tensor seq_lens, torch::Tensor blk_tbl
) {
    int B=q_u8.size(0), max_pages=blk_tbl.size(1);
    auto out=torch::zeros({B,4096},torch::TensorOptions().dtype(torch::kFloat32).device(q_u8.device()));
    int smem=H_DIM*D_DIM+H_DIM*4;
    score_only_kernel<<<B,256,smem>>>(
        (const uint8_t*)q_u8.data_ptr(),(const int8_t*)k_cache.data_ptr(),
        weights.data_ptr<float>(),seq_lens.data_ptr<int>(),blk_tbl.data_ptr<int>(),
        out.data_ptr<float>(),max_pages);
    return out;
}

std::tuple<torch::Tensor,torch::Tensor> run_sort_only(
    torch::Tensor sc, torch::Tensor ids
) {
    auto sc_=sc.clone(); auto ids_=ids.clone();
    sort_only_kernel<<<1,256>>>(sc_.data_ptr<float>(),ids_.data_ptr<int>(),BSORT_N);
    return {sc_,ids_};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("run_score_only",&run_score_only);
    m.def("run_sort_only",&run_sort_only);
}
"""

tmpdir=tempfile.mkdtemp()
src=os.path.join(tmpdir,"diag.cu")
open(src,"w").write(CUDA_DIAG)
mod=ext.load(name="diag_ext",sources=[src],extra_cuda_cflags=["-O3","--use_fast_math"],verbose=False)

from safetensors.torch import load_file
import sys
sys.path.insert(0,".")
from bench_all import make_inputs

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
q_u8=q_fp8.view(torch.uint8).contiguous()

# Warmup
for _ in range(3):
    mod.run_score_only(q_u8,k_cache.contiguous(),weights.contiguous(),seq_lens.contiguous(),block_table.contiguous())
torch.cuda.synchronize()

# Time score-only
buf=torch.empty(512*1024*1024//4,dtype=torch.float32,device="cuda")
times=[]
for _ in range(20):
    buf.fill_(0.); torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    mod.run_score_only(q_u8,k_cache.contiguous(),weights.contiguous(),seq_lens.contiguous(),block_table.contiguous())
    e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e)*1000)
times.sort()
print(f"Score-only CUDA events median: {times[len(times)//2]:.2f} us")

# Time sort-only on batch 2 (2817 tokens)
sc_dummy=torch.randn(4096,device="cuda")
ids_dummy=torch.arange(4096,dtype=torch.int32,device="cuda")
for _ in range(3):
    mod.run_sort_only(sc_dummy,ids_dummy)
torch.cuda.synchronize()

times2=[]
for _ in range(20):
    buf.fill_(0.); torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    mod.run_sort_only(sc_dummy,ids_dummy)
    e.record(); torch.cuda.synchronize()
    times2.append(s.elapsed_time(e)*1000)
times2.sort()
print(f"Sort-only CUDA events median: {times2[len(times2)//2]:.2f} us")

print("Done diagnosis")
PYEOF
