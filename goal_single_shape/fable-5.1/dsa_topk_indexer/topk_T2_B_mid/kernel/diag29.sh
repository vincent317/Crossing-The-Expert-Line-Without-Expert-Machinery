#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm
import torch.profiler as P

# deep_gemm bf16_gemm_nt: events=86us, profiler=4.53us !!
# profiler span = 4.53 us -- MATCHES REF! But CUDA events = 86 us (host overhead).
# The profiler is the right metric. 4.53 us kernel span.
# But ref's profiler span = 4.4 us. We're essentially matching ref just with bf16!
# 
# Target: profiler span <= ref_profiler/1.2 = 4.4/1.2 = 3.67 us
# Our deep_gemm bf16: profiler=4.53 us (close but not < 3.67 us)
# 
# The full torch.mm pipeline: profiler=149 us (many extra kernels).
# 
# To beat 3.67 us: the GEMM itself is 4.53 us. We need to reduce overhead.
# The extra steps (scale, relu, weighted sum, topk) add more kernels.
# 
# Let's try: fuse all post-GEMM ops into one CUDA kernel.
# Pipeline: 
# 1. deep_gemm bf16_gemm_nt: (512,128)@(4096,128) = (512,4096) [4.53 us]
# 2. Fused CUDA kernel: apply k_scale, relu, weighted sum per batch -> scores (8,4096) [fast]
# 3. Top-k selection in the same fused kernel
#
# Step 2+3 should be very fast since it's just reading (512,4096) = 2MB of data.
# Can we fuse step 1 and 2+3?
# NOT easily if they use different parallelism.
#
# But can we make step 1 faster?
# Our (512,128)@(4096,128) is a medium GEMM. deep_gemm achieves 4.53 us profiler.
# This is optimal given the size.
# 
# What if we use the ACTUAL fp8 from deep_gemm? fp8_gemm_nt (which is fp8_fp4_gemm_nt in the code)
# requires fp4 for B. But maybe using fp8 A and fp8 B (through a different function)?
#
# Let's check: what is the actual deep_gemm function for fp8 A x fp8 B?
# The library might route through cuBLAS fp8 matmul.
# Let's try deep_gemm.cublaslt_gemm_nt

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128; PAGE_SZ=64

all_sl=[seq_lens[b].item() for b in range(B)]
total=sum(all_sl)
k_raw=k_cache.view(-1,132)

all_flat=[]
for b in range(B):
    sl=all_sl[b]
    t=torch.arange(sl,device="cuda")
    pg=block_table[b,t//PAGE_SZ]; off=t%PAGE_SZ
    all_flat.append(pg*PAGE_SZ+off)
all_flat_t=torch.cat(all_flat)
K_fp8=k_raw[all_flat_t,:D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)
Ks=k_raw[all_flat_t,D:D+4].contiguous().view(torch.float32).view(total)
Gids=all_flat_t.int()
q_all=q_fp8.reshape(B*H,D)  # (512,128) fp8
offsets=[0]+list(torch.cumsum(torch.tensor(all_sl),0).tolist())

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

# Try: torch.mm on fp8 directly
q_bf16=q_all.float().bfloat16()
K_bf16=K_fp8.float().bfloat16()
N_a=((total+127)//128)*128
K_pad_bf16=torch.zeros(N_a,D,device="cuda",dtype=torch.bfloat16); K_pad_bf16[:total]=K_bf16
d_bf16=torch.zeros(B*H,N_a,device="cuda",dtype=torch.bfloat16)

# deep_gemm bf16 (aligned inputs)
M_a=((B*H+127)//128)*128
q_pad_bf16=torch.zeros(M_a,D,device="cuda",dtype=torch.bfloat16); q_pad_bf16[:B*H]=q_bf16
d_deep=torch.zeros(M_a,N_a,device="cuda",dtype=torch.bfloat16)

# Measure deep_gemm bf16 kernel span alone
def dg_mm():
    deep_gemm.bf16_gemm_nt(q_pad_bf16, K_pad_bf16, d_deep)
t=bench(dg_mm); p=prof_span(dg_mm)
print(f"deep_gemm bf16: events={t:.2f}us, profiler={p:.2f}us")

# Now: fuse post-gemm steps into one CUDA kernel
import torch.utils.cpp_extension as ext, os, tempfile
CUDA_POST = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <float.h>
#include <torch/extension.h>
// Grid: (B,), Block: 256 threads
// Applies: scale, relu, weighted sum over H heads, then top-k selection
// Input: raw_dots (B*H, total_tokens) bf16
//        k_scales (total_tokens,) float32
//        weights (B, H) float32
//        gids (total_tokens,) int32
//        batch_offsets (B,) int32 -- start of each batch in total_tokens
//        seq_lens (B,) int32
// Output: topk_out (B, 2048) int32
#define H_DIM 64
#define TOPK_K 2048
#define BSORT_N 4096
__global__ void post_gemm_topk(
    const __half* __restrict__ raw,  // (B*H, total_tokens) but we use offsets
    const float* __restrict__ ksc,   // (total_tokens,) k scales
    const float* __restrict__ wt,    // (B, H)
    const int*   __restrict__ gids,  // (total_tokens,) global ids
    const int*   __restrict__ offsets, // (B,) batch start offsets
    const int*   __restrict__ sl_arr,  // (B,) seq_lens
    int* __restrict__ out,           // (B, TOPK_K)
    int total_tokens
) {
    int b=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    int sl=sl_arr[b], off=offsets[b];

    extern __shared__ char smem[];
    float* sc_sh=(float*)smem;
    int* id_sh=(int*)(smem+BSORT_N*4);

    // Compute score for each token in batch b
    for(int t=tid; t<sl; t+=bsz){
        int tok=off+t;
        float ks=ksc[tok];
        float score=0.f;
        for(int h=0; h<H_DIM; h++){
            // raw[(b*H_DIM+h), tok] -- row b*H_DIM+h, col tok
            float dot=__half2float(raw[(long)(b*H_DIM+h)*total_tokens+tok]) * ks;
            float relu_dot=dot>0.f?dot:0.f;
            score+=relu_dot*wt[b*H_DIM+h];
        }
        sc_sh[t]=score;
        id_sh[t]=gids[tok];
    }
    for(int t=sl+tid; t<BSORT_N; t+=bsz){ sc_sh[t]=-FLT_MAX; id_sh[t]=-1; }
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

torch::Tensor run_post(
    torch::Tensor raw, torch::Tensor ksc, torch::Tensor wt,
    torch::Tensor gids, torch::Tensor offsets, torch::Tensor sls, int total
){
    int B=sls.size(0);
    auto out=torch::full({B,TOPK_K},-1,torch::TensorOptions().dtype(torch::kInt32).device(raw.device()));
    int smem=BSORT_N*4*2;
    post_gemm_topk<<<B,256,smem>>>(
        (const __half*)raw.data_ptr(),ksc.data_ptr<float>(),wt.data_ptr<float>(),
        gids.data_ptr<int>(),offsets.data_ptr<int>(),sls.data_ptr<int>(),
        out.data_ptr<int>(),total);
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_post",&run_post);}
"""
tmpdir=tempfile.mkdtemp(); src=os.path.join(tmpdir,"post.cu")
open(src,"w").write(CUDA_POST)
mod=ext.load(name="post_ext",sources=[src],extra_cuda_cflags=["-O3","--use_fast_math"],verbose=False)

offs_t=torch.tensor(offsets[:-1],device="cuda",dtype=torch.int32)
sls_t=seq_lens.int()
# raw_dots needs shape (B*H, total_tokens) -- from bf16 GEMM we get (M_a, N_a)
# Slice to (B*H, total_tokens)
raw_slice=d_deep[:B*H, :total].contiguous()

# Warmup
deep_gemm.bf16_gemm_nt(q_pad_bf16, K_pad_bf16, d_deep)
mod.run_post(raw_slice, Ks.contiguous(), weights.contiguous(), Gids.contiguous(), offs_t, sls_t, total)
torch.cuda.synchronize()

def full_pipeline():
    deep_gemm.bf16_gemm_nt(q_pad_bf16, K_pad_bf16, d_deep)
    raw=d_deep[:B*H, :total].contiguous()
    return mod.run_post(raw, Ks.contiguous(), weights.contiguous(), Gids.contiguous(), offs_t, sls_t, total)

t=bench(full_pipeline); p=prof_span(full_pipeline)
print(f"deep_gemm+post_kernel pipeline: events={t:.2f}us profiler={p:.2f}us")

# Check profiler span just for post kernel
def just_post():
    return mod.run_post(raw_slice, Ks.contiguous(), weights.contiguous(), Gids.contiguous(), offs_t, sls_t, total)
t2=bench(just_post); p2=prof_span(just_post)
print(f"post_kernel alone: events={t2:.2f}us profiler={p2:.2f}us")
PYEOF
