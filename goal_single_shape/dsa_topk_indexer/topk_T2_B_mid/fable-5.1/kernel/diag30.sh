#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm, torch.utils.cpp_extension as ext, os, tempfile
import torch.profiler as P

# Post kernel is slow because it reads (B*H, total_tokens) row by row for each batch.
# raw_dots shape: (512, 4061). For batch b: read rows b*64 to b*64+63, cols off:off+sl.
# For batch 2 (sl=2817): 64 rows × 2817 cols × 2 bytes = 361 KB scattered reads.
# All 8 batches: 512 rows × avg_sl cols × 2 bytes ≈ 512×507×2 = 519 KB.
# At 8TB/s: trivial. But post_kernel=292us.
#
# Issue: post_kernel reads raw[h*total_tokens + tok] for h=0..63, tok=off..off+sl.
# This is column-major scan: stride = total_tokens = 4061 floats = 8122 bytes.
# Very non-coalesced! 64 strided reads per token × 4061 tokens = bad.
# 
# FIX: Transpose the output: d_deep is (B*H, N_a) bf16.
# After GEMM, transpose to (N_a, B*H) = (4096, 512). Then per-token access is coalesced.
# For token tok from batch b: read [tok, b*64:b*64+64] -- coalesced 64-element row.
#
# But transpose itself takes time (~1us).
#
# Better fix: change layout in post_kernel to access transposed_raw[tok, b*64+h]
# instead of raw[b*64+h, tok]. If raw is (N_a, M_a) transposed in memory, this
# requires different GEMM ordering.
#
# Actually: bf16_gemm_nt computes A(M,K) @ B^T(N,K) = C(M,N).
# Our A = q_all (B*H, D), B = K_all (total, D) → C = q @ K^T = (B*H, total).
# C[h, tok] = q[h] · K[tok] for each head h and token tok.
# 
# What if we compute K(total,D) @ q^T(D, B*H) = (total, B*H)?
# This gives transposed output: [tok, h] = K[tok] · q[h].
# Then for each tok: 8 batches × 64 heads = 512 values, all coalesced per tok row.
# 
# In deep_gemm terms: bf16_gemm_nt(K_all, q_all, out_transposed)
# where K_all (N, D) and q_all (M, D) → out (N, M) = K @ q^T

q_fp8, k_cache, weights, seq_lens, block_table = make_inputs("cuda")
B, H, D = 8, 64, 128; PAGE_SZ = 64

all_sl = [seq_lens[b].item() for b in range(B)]
total = sum(all_sl)
k_raw = k_cache.view(-1, 132)

all_flat = []
for b in range(B):
    sl = all_sl[b]
    t = torch.arange(sl, device="cuda")
    pg = block_table[b, t//PAGE_SZ]; off = t % PAGE_SZ
    all_flat.append(pg * PAGE_SZ + off)
all_flat_t = torch.cat(all_flat)
K_fp8 = k_raw[all_flat_t, :D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)
Ks = k_raw[all_flat_t, D:D+4].contiguous().view(torch.float32).view(total)
Gids = all_flat_t.int()

K_bf16 = K_fp8.float().bfloat16().contiguous()
q_bf16 = q_fp8.reshape(B*H, D).float().bfloat16().contiguous()

M = B*H  # 512
M_a = ((M+127)//128)*128
N_a = ((total+127)//128)*128

# Transposed GEMM: K(total,D) @ q^T(D, B*H) = (total, B*H)
# K is (total, D), q is (B*H, D). We want K @ q^T = (total, B*H).
# bf16_gemm_nt(A, B, C): A(M,K) @ B^T(N,K) = C(M,N)
# Set A=K_pad (N_a,D), B=q_pad (M_a,D) → C(N_a, M_a) = K @ q^T
K_pad = torch.zeros(N_a, D, device="cuda", dtype=torch.bfloat16); K_pad[:total] = K_bf16
q_pad = torch.zeros(M_a, D, device="cuda", dtype=torch.bfloat16); q_pad[:M] = q_bf16
d_trans = torch.zeros(N_a, M_a, device="cuda", dtype=torch.bfloat16)

buf = torch.empty(512*1024*1024//4, dtype=torch.float32, device="cuda")
def bench(fn, w=5, n=30):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        buf.fill_(0.); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return ts[len(ts)//2]

def prof_span(fn, n=5):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    buf.fill_(0.); torch.cuda.synchronize()
    with P.profile(activities=[P.ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            buf.fill_(0.); fn()
    torch.cuda.synchronize()
    evts = prof.profiler.function_events
    total2 = sum(e.device_time for e in evts if e.device_time > 0 and "fill" not in e.name.lower())
    return total2 / n

deep_gemm.bf16_gemm_nt(K_pad, q_pad, d_trans)
torch.cuda.synchronize()
print(f"Transposed GEMM OK: d_trans shape {d_trans.shape}")

t = bench(lambda: deep_gemm.bf16_gemm_nt(K_pad, q_pad, d_trans))
p = prof_span(lambda: deep_gemm.bf16_gemm_nt(K_pad, q_pad, d_trans))
print(f"transposed GEMM: events={t:.2f}us, profiler={p:.2f}us")

# Post kernel reading transposed: d_trans[tok, b*H:(b+1)*H] for each batch b
# This is coalesced access! (H=64 consecutive elements per row)
CUDA_POST2 = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <torch/extension.h>
#define H_DIM 64
#define TOPK_K 2048
#define BSORT_N 4096
__global__ void post_topk(
    const __half* __restrict__ raw,  // (total_tokens, B*H) transposed: raw[tok, b*H+h]
    const float* __restrict__ ksc,
    const float* __restrict__ wt,
    const int* __restrict__ gids,
    const int* __restrict__ offsets,
    const int* __restrict__ sl_arr,
    int* __restrict__ out,
    int bh, int total  // bh = B*H = 512
){
    int b=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    int sl=sl_arr[b], off=offsets[b];
    extern __shared__ char smem[];
    float* sc_sh=(float*)smem;
    int* id_sh=(int*)(smem+BSORT_N*4);

    const float* wt_b=wt+b*H_DIM;
    int h_off=b*H_DIM;

    for(int t=tid; t<sl; t+=bsz){
        int tok=off+t;
        float ks=ksc[tok];
        // Read transposed row: raw[tok, h_off:h_off+H_DIM] -- coalesced!
        const __half* row=raw+(long)tok*bh+h_off;
        float score=0.f;
        for(int h=0; h<H_DIM; h++){
            float dot=__half2float(row[h])*ks;
            score+=(dot>0.f?dot:0.f)*wt_b[h];
        }
        sc_sh[t]=score; id_sh[t]=gids[tok];
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
torch::Tensor run_post2(
    torch::Tensor raw, torch::Tensor ksc, torch::Tensor wt,
    torch::Tensor gids, torch::Tensor offsets, torch::Tensor sls, int bh_in, int total_in
){
    int B=sls.size(0);
    auto out=torch::full({B,TOPK_K},-1,torch::TensorOptions().dtype(torch::kInt32).device(raw.device()));
    post_topk<<<B,256,BSORT_N*8>>>(
        (const __half*)raw.data_ptr(),ksc.data_ptr<float>(),wt.data_ptr<float>(),
        gids.data_ptr<int>(),offsets.data_ptr<int>(),sls.data_ptr<int>(),
        out.data_ptr<int>(),bh_in,total_in);
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_post2",&run_post2);}
"""
tmpdir = tempfile.mkdtemp(); src = os.path.join(tmpdir, "post2.cu")
open(src, "w").write(CUDA_POST2)
mod2 = ext.load(name="post2_ext", sources=[src], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

offsets_t = torch.tensor([0]+list(torch.cumsum(torch.tensor(all_sl),0).tolist())[:-1], device="cuda", dtype=torch.int32)
sls_t = seq_lens.int()
raw_trans = d_trans[:total, :M].contiguous()  # (total, 512)

t2 = bench(lambda: mod2.run_post2(raw_trans, Ks.contiguous(), weights.contiguous(), Gids.contiguous(), offsets_t, sls_t, M, total))
p2 = prof_span(lambda: mod2.run_post2(raw_trans, Ks.contiguous(), weights.contiguous(), Gids.contiguous(), offsets_t, sls_t, M, total))
print(f"post2_topk (coalesced, 256t): events={t2:.2f}us, profiler={p2:.2f}us")

def full2():
    deep_gemm.bf16_gemm_nt(K_pad, q_pad, d_trans)
    raw = d_trans[:total, :M].contiguous()
    return mod2.run_post2(raw, Ks.contiguous(), weights.contiguous(), Gids.contiguous(), offsets_t, sls_t, M, total)
t3 = bench(full2, w=3, n=20); p3 = prof_span(full2, n=5)
print(f"GEMM+post2: events={t3:.2f}us, profiler={p3:.2f}us")
PYEOF
