#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, torch.utils.cpp_extension as ext, os, tempfile
import sys; sys.path.insert(0,".")
from bench_all import make_inputs
import torch.profiler as P

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
q_u8=q_fp8.view(torch.uint8).contiguous()

CUDA2 = r"""
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

__global__ void fused_topk(
    const uint8_t* __restrict__ q_fp8,
    const int8_t*  __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int*     __restrict__ seq_lens,
    const int*     __restrict__ blk_tbl,
    int*           __restrict__ out,
    int max_pages
){
    int b=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    int sl=seq_lens[b];
    extern __shared__ char smem[];
    uint8_t* q_sh =(uint8_t*)smem;
    float*   wt_sh=(float*)(smem+H_DIM*D_DIM);
    float*   sc_sh=(float*)(smem+H_DIM*D_DIM+H_DIM*4);
    int*     id_sh=(int*)(smem+H_DIM*D_DIM+H_DIM*4+BSORT_N*4);
    for(int i=tid;i<H_DIM*D_DIM/4;i+=bsz)
        ((uint32_t*)q_sh)[i]=((const uint32_t*)(q_fp8+(long)b*H_DIM*D_DIM))[i];
    for(int i=tid;i<H_DIM;i+=bsz) wt_sh[i]=weights[b*H_DIM+i];
    __syncthreads();
    for(int t=tid;t<sl;t+=bsz){
        int pg=blk_tbl[b*max_pages+t/PAGE_SZ], off=t%PAGE_SZ;
        const uint8_t* kp=(const uint8_t*)k_cache+((long)pg*PAGE_SZ+off)*K_STRIDE;
        float ks=*(const float*)(kp+D_DIM);
        uint32_t kr[DIM32];
        for(int i=0;i<DIM32;i++) kr[i]=((const uint32_t*)kp)[i];
        float score=0.f;
        for(int h=0;h<H_DIM;h++){
            const uint32_t* qh=(const uint32_t*)(q_sh+h*D_DIM);
            float d=0.f;
            for(int i=0;i<DIM32;i++) dot4v(qh[i],kr[i],d);
            d*=ks; score+=(d>0.f?d:0.f)*wt_sh[h];
        }
        sc_sh[t]=score; id_sh[t]=pg*PAGE_SZ+off;
    }
    for(int t=sl+tid;t<BSORT_N;t+=bsz){ sc_sh[t]=-FLT_MAX; id_sh[t]=-1; }
    __syncthreads();
    int* out_b=out+(long)b*TOPK_K;
    if(sl<=TOPK_K){
        for(int t=tid;t<sl;t+=bsz) out_b[t]=id_sh[t];
        for(int t=sl+tid;t<TOPK_K;t+=bsz) out_b[t]=-1;
    } else {
        for(int k=2;k<=BSORT_N;k<<=1){
            for(int j=k>>1;j>0;j>>=1){
                for(int i=tid;i<BSORT_N;i+=bsz){
                    int ixj=i^j;
                    if(ixj>i){
                        bool sw=((i&k)==0)?(sc_sh[i]<sc_sh[ixj]):(sc_sh[i]>sc_sh[ixj]);
                        if(sw){float ts=sc_sh[i];sc_sh[i]=sc_sh[ixj];sc_sh[ixj]=ts;
                               int ti=id_sh[i];id_sh[i]=id_sh[ixj];id_sh[ixj]=ti;}
                    }
                }
                __syncthreads();
            }
        }
        for(int t=tid;t<TOPK_K;t+=bsz) out_b[t]=id_sh[t];
    }
}
torch::Tensor run_fused(torch::Tensor q,torch::Tensor k,torch::Tensor w,torch::Tensor sl,torch::Tensor bt){
    int B=q.size(0),mp=bt.size(1);
    auto out=torch::full({B,TOPK_K},-1,torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
    int smem=H_DIM*D_DIM+H_DIM*4+BSORT_N*4*2;
    fused_topk<<<B,256,smem>>>((const uint8_t*)q.data_ptr(),(const int8_t*)k.data_ptr(),
        w.data_ptr<float>(),sl.data_ptr<int>(),bt.data_ptr<int>(),out.data_ptr<int>(),mp);
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_fused",&run_fused);}
"""
tmpdir=tempfile.mkdtemp(); src2=os.path.join(tmpdir,"sc2.cu")
open(src2,"w").write(CUDA2)
mod2=ext.load(name="sc2_ext",sources=[src2],extra_cuda_cflags=["-O3","--use_fast_math","--maxrregcount=96"],verbose=False)

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

print(f"fused_topk CUDA events: {bench(lambda: mod2.run_fused(*args)):.2f} us")

for _ in range(5): mod2.run_fused(*args)
torch.cuda.synchronize()
buf.fill_(0.); torch.cuda.synchronize()
with P.profile(activities=[P.ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        buf.fill_(0.); mod2.run_fused(*args)
torch.cuda.synchronize()
evts=prof.profiler.function_events
total=sum(e.device_time for e in evts if e.device_time>0 and "fill" not in e.name.lower())
print(f"fused_topk profiler span: {total/5:.2f} us")
PYEOF
