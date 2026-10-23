"""Fast DSA top-k: multi-block score + PyTorch topk."""
import torch
import torch.utils.cpp_extension as ext
import os, tempfile

CUDA_SRC = r"""
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

#define H_DIM 64
#define D_DIM 128
#define PAGE_SZ 64
#define K_STRIDE 132

__device__ __forceinline__ void dot4(uint32_t q4, uint32_t k4, float& acc) {
    __half2 qh0=__nv_cvt_fp8x2_to_halfraw2(q4&0xFFFFu,__NV_E4M3);
    __half2 qh1=__nv_cvt_fp8x2_to_halfraw2(q4>>16,__NV_E4M3);
    __half2 kh0=__nv_cvt_fp8x2_to_halfraw2(k4&0xFFFFu,__NV_E4M3);
    __half2 kh1=__nv_cvt_fp8x2_to_halfraw2(k4>>16,__NV_E4M3);
    float2 qf0=__half22float2(qh0),qf1=__half22float2(qh1);
    float2 kf0=__half22float2(kh0),kf1=__half22float2(kh1);
    acc+=qf0.x*kf0.x+qf0.y*kf0.y+qf1.x*kf1.x+qf1.y*kf1.y;
}

// Grid: (B, max_pages); each block processes 64 tokens from one page
__global__ void score_multiblock(
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
    for(int loc=tid; loc<PAGE_SZ && (t_base+loc)<seq_len; loc+=bsz) {
        const uint8_t* kp=(const uint8_t*)k_cache+((long)page*PAGE_SZ+loc)*K_STRIDE;
        float ks=*(const float*)(kp+D_DIM);
        float sc=0.f;
        for(int h=0;h<H_DIM;h++){
            const uint32_t* qh=(const uint32_t*)(q_sh+h*D_DIM);
            const uint32_t* kh=(const uint32_t*)kp;
            float d=0.f;
            for(int i=0;i<D_DIM/4;i++) dot4(qh[i],kh[i],d);
            d*=ks; sc+=(d>0.f?d:0.f)*wt_sh[h];
        }
        int t=t_base+loc;
        scores[(long)b*4096+t]=sc;
        ids[(long)b*4096+t]=page*PAGE_SZ+loc;
    }
}

std::tuple<torch::Tensor,torch::Tensor> dsa_topk_fast(
    torch::Tensor q_u8, torch::Tensor k_cache, torch::Tensor weights,
    torch::Tensor seq_lens, torch::Tensor blk_tbl
) {
    int B=q_u8.size(0), mp=blk_tbl.size(1);
    auto opts=torch::TensorOptions().device(q_u8.device());
    auto scores=torch::full({B,4096},-1e38f,opts.dtype(torch::kFloat32));
    auto ids=torch::full({B,4096},-1,opts.dtype(torch::kInt32));

    dim3 grid(B,mp);
    int smem=H_DIM*D_DIM+H_DIM*4;
    score_multiblock<<<grid,64,smem>>>(
        (const uint8_t*)q_u8.data_ptr(),(const int8_t*)k_cache.data_ptr(),
        weights.data_ptr<float>(),seq_lens.data_ptr<int>(),blk_tbl.data_ptr<int>(),
        scores.data_ptr<float>(),ids.data_ptr<int>(),mp);

    // Top-2048 via torch (host fallback for simplicity)
    auto out=torch::full({B,2048},-1,opts.dtype(torch::kInt32));
    auto sl_cpu=seq_lens.cpu();
    for(int b=0;b<B;b++){
        int sl=sl_cpu[b].item<int>();
        int k=std::min(2048,sl);
        if(k>0){
            auto sc_b=scores[b].slice(0,0,sl);
            auto id_b=ids[b].slice(0,0,sl);
            auto topk_out=torch::topk(sc_b,k);
            auto topk_idx=std::get<1>(topk_out);
            out[b].slice(0,0,k)=id_b.index_select(0,topk_idx);
        }
    }
    return {out,scores};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("dsa_topk_fast",&dsa_topk_fast);
}
"""

_mod=None
def _load():
    global _mod
    if _mod is None:
        tmpdir=tempfile.mkdtemp()
        src=os.path.join(tmpdir,"fast.cu")
        open(src,"w").write(CUDA_SRC)
        _mod=ext.load(name="fast_ext",sources=[src],
            extra_cuda_cflags=["-O3","--use_fast_math"],verbose=True)
    return _mod

def run(q_fp8,k_cache,weights,seq_lens,blk_tbl):
    mod=_load()
    q_u8=q_fp8.view(torch.uint8) if q_fp8.dtype!=torch.uint8 else q_fp8
    out,_=mod.dsa_topk_fast(q_u8.contiguous(),k_cache.contiguous(),
        weights.contiguous(),seq_lens.contiguous(),blk_tbl.contiguous())
    return out
