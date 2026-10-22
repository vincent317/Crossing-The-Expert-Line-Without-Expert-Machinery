#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, torch.utils.cpp_extension as ext, os, tempfile, subprocess

# Check actual register counts for score_half2 kernel
# Also test with more blocks per batch
CUDA_SRC = r"""
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#define H_DIM 64
#define D_DIM 128
#define PAGE_SZ 64
#define K_STRIDE 132

__device__ __forceinline__ void dot4_f32(uint32_t q4, uint32_t k4, float& acc) {
    __half2 qh0=__nv_cvt_fp8x2_to_halfraw2(q4&0xFFFFu,__NV_E4M3);
    __half2 qh1=__nv_cvt_fp8x2_to_halfraw2(q4>>16,__NV_E4M3);
    __half2 kh0=__nv_cvt_fp8x2_to_halfraw2(k4&0xFFFFu,__NV_E4M3);
    __half2 kh1=__nv_cvt_fp8x2_to_halfraw2(k4>>16,__NV_E4M3);
    float2 qf0=__half22float2(qh0),qf1=__half22float2(qh1);
    float2 kf0=__half22float2(kh0),kf1=__half22float2(kh1);
    acc+=qf0.x*kf0.x+qf0.y*kf0.y+qf1.x*kf1.x+qf1.y*kf1.y;
}

// Each block handles one page-chunk of tokens for one batch.
// Grid: (B, max_chunks) where chunk covers PAGE_SZ=64 tokens.
// This way many blocks work in parallel for batch 2 (44 pages).
__global__ void score_multiblock(
    const uint8_t* __restrict__ q_fp8,
    const int8_t*  __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int*     __restrict__ seq_lens,
    const int*     __restrict__ blk_tbl,
    float* __restrict__ score_out,  // (B, max_seq) padded to 2817
    int max_pages, int max_seq
) {
    int b       = blockIdx.x;
    int page_id = blockIdx.y;  // page index within batch b
    int tid     = threadIdx.x;
    int bsz     = blockDim.x;

    int seq_len = seq_lens[b];
    // page_id * PAGE_SZ = first token in this chunk
    int t_base  = page_id * PAGE_SZ;
    if (t_base >= seq_len) return;
    int t_end = min(t_base + PAGE_SZ, seq_len);

    extern __shared__ char smem[];
    uint8_t* q_sh  = (uint8_t*)smem;
    float*   wt_sh = (float*)(smem + H_DIM*D_DIM);

    // Load q[b] — only needs to happen once per block
    for (int i=tid; i<H_DIM*D_DIM/4; i+=bsz)
        ((uint32_t*)q_sh)[i]=((const uint32_t*)(q_fp8+(long)b*H_DIM*D_DIM))[i];
    for (int i=tid; i<H_DIM; i+=bsz)
        wt_sh[i]=weights[b*H_DIM+i];
    __syncthreads();

    int page = blk_tbl[b*max_pages + page_id];

    // Each thread handles one token within this page
    for (int local_t = tid; local_t < PAGE_SZ && (t_base+local_t) < seq_len; local_t += bsz) {
        int pg_off = local_t;
        const uint8_t* kp = (const uint8_t*)k_cache + ((long)page*PAGE_SZ+pg_off)*K_STRIDE;
        float ks = *(const float*)(kp+D_DIM);

        float sc=0.f;
        for (int h=0; h<H_DIM; h++) {
            const uint32_t* qh32=(const uint32_t*)(q_sh+h*D_DIM);
            const uint32_t* kh32=(const uint32_t*)kp;
            float d=0.f;
            for (int i=0; i<D_DIM/4; i++) dot4_f32(qh32[i],kh32[i],d);
            d*=ks;
            sc+=(d>0.f?d:0.f)*wt_sh[h];
        }
        int t = t_base + local_t;
        score_out[(long)b*max_seq + t] = sc;
    }
}

torch::Tensor run_multiblock(
    torch::Tensor q, torch::Tensor k, torch::Tensor w,
    torch::Tensor sl, torch::Tensor bt
) {
    int B=q.size(0), mp=bt.size(1), max_seq=2817;
    auto o=torch::zeros({B,max_seq},torch::dtype(torch::kFloat).device(q.device()));
    dim3 grid(B, mp);
    int smem=H_DIM*D_DIM+H_DIM*4;
    score_multiblock<<<grid,64,smem>>>(
        (const uint8_t*)q.data_ptr(),(const int8_t*)k.data_ptr(),
        w.data_ptr<float>(),sl.data_ptr<int>(),bt.data_ptr<int>(),
        o.data_ptr<float>(),mp,max_seq);
    return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_multiblock",&run_multiblock);}
"""
tmpdir=tempfile.mkdtemp()
src=os.path.join(tmpdir,"d3.cu")
open(src,"w").write(CUDA_SRC)

# Also get register info
nvcc=subprocess.run(
    ["nvcc","--ptxas-options=-v","-O3","--use_fast_math",
     "-gencode=arch=compute_100,code=sm_100",
     "-I/opt/sglang/lib/python3.12/site-packages/torch/include",
     "-I/usr/local/cuda/include",
     "-c",src,"-o","/tmp/d3.o"],
    capture_output=True, text=True
)
for line in (nvcc.stdout+nvcc.stderr).split("\n"):
    if "registers" in line or "smem" in line.lower():
        print("NVCC:", line.strip())

mod=ext.load(name="d3_ext",sources=[src],extra_cuda_cflags=["-O3","--use_fast_math"],verbose=False)

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

print(f"multiblock score  : {bench(lambda: mod.run_multiblock(*args)):.2f} us")
PYEOF
