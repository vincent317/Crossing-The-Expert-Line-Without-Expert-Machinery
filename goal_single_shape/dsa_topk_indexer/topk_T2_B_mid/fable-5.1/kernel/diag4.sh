#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, torch.utils.cpp_extension as ext, os, tempfile, subprocess
# Test: time NVCC ptxas register info for score_multiblock
CUDA_SRC = r"""
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#define H_DIM 64
#define D_DIM 128
#define PAGE_SZ 64
#define K_STRIDE 132

__device__ __forceinline__ void dot4_f32(uint32_t q4,uint32_t k4,float& acc){
    __half2 qh0=__nv_cvt_fp8x2_to_halfraw2(q4&0xFFFFu,__NV_E4M3);
    __half2 qh1=__nv_cvt_fp8x2_to_halfraw2(q4>>16,__NV_E4M3);
    __half2 kh0=__nv_cvt_fp8x2_to_halfraw2(k4&0xFFFFu,__NV_E4M3);
    __half2 kh1=__nv_cvt_fp8x2_to_halfraw2(k4>>16,__NV_E4M3);
    float2 qf0=__half22float2(qh0),qf1=__half22float2(qh1);
    float2 kf0=__half22float2(kh0),kf1=__half22float2(kh1);
    acc+=qf0.x*kf0.x+qf0.y*kf0.y+qf1.x*kf1.x+qf1.y*kf1.y;
}

// Single-block score kernel, 256 threads, scores written to global then CUB top-k
__global__ void score_write(
    const uint8_t* __restrict__ q_fp8,
    const int8_t*  __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int*     __restrict__ seq_lens,
    const int*     __restrict__ blk_tbl,
    float* __restrict__ sc_out,
    int*   __restrict__ id_out,
    int max_pages, int max_seq
){
    int b=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    int seq_len=seq_lens[b];
    extern __shared__ char smem[];
    uint8_t* q_sh=(uint8_t*)smem;
    float*   wt_sh=(float*)(smem+H_DIM*D_DIM);
    for(int i=tid;i<H_DIM*D_DIM/4;i+=bsz)
        ((uint32_t*)q_sh)[i]=((const uint32_t*)(q_fp8+(long)b*H_DIM*D_DIM))[i];
    for(int i=tid;i<H_DIM;i+=bsz) wt_sh[i]=weights[b*H_DIM+i];
    __syncthreads();
    for(int t=tid;t<seq_len;t+=bsz){
        int page=blk_tbl[b*max_pages+t/PAGE_SZ];
        int off=t%PAGE_SZ;
        const uint8_t* kp=(const uint8_t*)k_cache+((long)page*PAGE_SZ+off)*K_STRIDE;
        float ks=*(const float*)(kp+D_DIM);
        float sc=0.f;
        for(int h=0;h<H_DIM;h++){
            const uint32_t* qh32=(const uint32_t*)(q_sh+h*D_DIM);
            const uint32_t* kh32=(const uint32_t*)kp;
            float d=0.f;
            for(int i=0;i<D_DIM/4;i++) dot4_f32(qh32[i],kh32[i],d);
            d*=ks; sc+=(d>0.f?d:0.f)*wt_sh[h];
        }
        sc_out[(long)b*max_seq+t]=sc;
        id_out[(long)b*max_seq+t]=page*PAGE_SZ+off;
    }
    for(int t=seq_len+tid;t<max_seq;t+=bsz){
        sc_out[(long)b*max_seq+t]=-1e38f;
        id_out[(long)b*max_seq+t]=-1;
    }
}

// Single-block per-batch top-k using warp-level radix selection
// For batch with seq_len<=2048: just copy ids
// For batch 2: need 2048 of 2817 -- use parallel threshold find
// METHOD: count scores >=each threshold using warp reduction; binary-search
// SIMPLER: warp-level partial sort using __shfl
// SIMPLEST for timing: just use thrust::sort in a second kernel

torch::Tensor run_score_write(
    torch::Tensor q, torch::Tensor k, torch::Tensor w,
    torch::Tensor sl, torch::Tensor bt, int max_seq
){
    int B=q.size(0), mp=bt.size(1);
    auto sc=torch::zeros({B,max_seq},torch::dtype(torch::kFloat).device(q.device()));
    auto id=torch::zeros({B,max_seq},torch::dtype(torch::kInt32).device(q.device()));
    int smem=H_DIM*D_DIM+H_DIM*4;
    score_write<<<B,256,smem>>>(
        (const uint8_t*)q.data_ptr(),(const int8_t*)k.data_ptr(),
        w.data_ptr<float>(),sl.data_ptr<int>(),bt.data_ptr<int>(),
        sc.data_ptr<float>(),id.data_ptr<int>(),mp,max_seq);
    return sc; // return scores for inspection
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_score_write",&run_score_write);}
"""
tmpdir=tempfile.mkdtemp()
src=os.path.join(tmpdir,"d4.cu")
open(src,"w").write(CUDA_SRC)

# Register info
r=subprocess.run(["nvcc","--ptxas-options=-v","-O3","--use_fast_math",
    "-gencode=arch=compute_100,code=sm_100",
    "-I/opt/sglang/lib/python3.12/site-packages/torch/include",
    "-I/usr/local/cuda/include","-c",src,"-o","/tmp/d4.o"],
    capture_output=True, text=True)
for line in (r.stdout+r.stderr).split("\n"):
    if "register" in line or "reg" in line.lower() and "ptxas" in line.lower():
        print("NVCC ptxas:", line.strip())

mod=ext.load(name="d4_ext",sources=[src],extra_cuda_cflags=["-O3","--use_fast_math"],verbose=False)
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

t=bench(lambda: mod.run_score_write(*args, 2817))
print(f"score_write 256t: {t:.2f} us")

# Now test: how much does torch.topk cost on (8,2817) float tensor?
sc_dummy=torch.randn(8,2817,device="cuda")
t2=bench(lambda: torch.topk(sc_dummy,2048,dim=1))
print(f"torch.topk(8,2817,2048): {t2:.2f} us")

# And argsort
t3=bench(lambda: torch.argsort(sc_dummy,dim=1,descending=True))
print(f"torch.argsort(8,2817): {t3:.2f} us")
PYEOF
