#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys, torch.utils.cpp_extension as ext, os, tempfile
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm, torch.profiler as P

# post2 is still 275 us. The bitonic sort of 2817 elements dominates.
# 
# KEY INSIGHT: deep_gemm bf16_gemm_nt profiler=4.6 us (close to ref 4.4 us).
# But we still need post-processing (scale + relu + weighted sum + topk).
# The post-processing alone is 275 us due to bitonic sort being slow.
#
# The bitonic sort problem: 78 __syncthreads() calls × 8 different batch-blocks.
# But all 8 blocks run on different SMs, so 78 syncs per block = 78 synchronization barriers.
# Each __syncthreads() on B200 takes ~few cycles (single-SM barrier, fast).
# 78 × (BSORT_N/256 iterations × compare+swap) should be fast.
# But we measured 275 us for sort_only before (after score computation).
# And sort_only alone (diag1) was 309 us.
# 
# Let me think: sort of 4096 elements with 256 threads.
# Each of 78 steps: 4096 elements, 256 threads → 4096/256 = 16 pairs per step/thread.
# Wait: the loop is `for(i=tid; i<BSORT_N; i+=bsz)` and only processes when `ixj>i`.
# Half of 4096 positions have ixj>i, so ~2048 active per step.
# 256 threads → 2048/256 = 8 iterations per __syncthreads().
# 78 syncs × 8 iterations = 624 iterations per thread.
# Each iteration: 2 shared mem reads + compare + 2 conditional writes = ~8 ops.
# 624 × 8 = 4992 ops per thread.
# At 1.7GHz with 1 op/cycle: 4992/1.7G = 2.9 us per block.
# With 8 blocks on 8 SMs in parallel: max = 2.9 us.
# 
# BUT WE MEASURED 275 us! That's 95x too slow.
# 
# The issue must be: with 8 blocks on 8 SMs, the smem for post2 kernel = BSORT_N*8=32KB.
# Plus: post2 does BOTH score computation AND sort in one kernel.
# Score takes ~292us on its own (measured earlier for score_only kernel).
# So 275+292 is wrong... let me re-check.
# 
# Wait: in diag30, raw_trans already has the GEMM results precomputed.
# So post2 only does: scale, relu, weighted sum, topk. NOT the full dot product.
# 
# For each token t in batch b: reads 64 consecutive bf16 values from raw_trans[tok].
# For batch 2: 2817 tokens × 64 values × 2 bytes = 361 KB reads.
# Plus 2817 × 4 bytes (ksc) + weights (B*H*4) + gids...
# Total ~400KB.
# At B200 bandwidth ~8TB/s (effective ~400GB/s per SM with 8 active SMs): 400KB/400GB/s = 1us.
# 
# Then bitonic sort on 2817 scores: should be ~2us.
# TOTAL post2: ~3us. But we're getting 275 us!
# 
# Something is MASSIVELY wrong with the post2 kernel.
# Let me add more diagnostics.

q_fp8, k_cache, weights, seq_lens, block_table = make_inputs("cuda")
B, H, D = 8, 64, 128; PAGE_SZ = 64
all_sl = [seq_lens[b].item() for b in range(B)]
total = sum(all_sl)
k_raw = k_cache.view(-1, 132)

all_flat = []
for b in range(B):
    sl = all_sl[b]
    t_ = torch.arange(sl, device="cuda")
    pg = block_table[b, t_//PAGE_SZ]; off = t_ % PAGE_SZ
    all_flat.append(pg * PAGE_SZ + off)
all_flat_t = torch.cat(all_flat)
Ks = k_raw[all_flat_t, D:D+4].contiguous().view(torch.float32).view(total)
Gids = all_flat_t.int()
offsets_list = [0] + list(torch.cumsum(torch.tensor(all_sl), 0).tolist())[:-1]
offsets_t = torch.tensor(offsets_list, device="cuda", dtype=torch.int32)

# Make pre-computed raw_trans: (total, B*H) random data
raw_trans = torch.randn(total, B*H, device="cuda", dtype=torch.bfloat16).contiguous()

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
    total2 = sum(e.device_time for e in evts if e.device_time>0 and "fill" not in e.name.lower())
    return total2/n

# Test: score-only part (no sort)
CUDA = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <torch/extension.h>
#define H_DIM 64
#define TOPK_K 2048
#define BSORT_N 4096

__global__ void score_from_raw(
    const __half* __restrict__ raw,  // (total, B*H) row-major
    const float* __restrict__ ksc,
    const float* __restrict__ wt,
    const int* __restrict__ offsets,
    const int* __restrict__ sl_arr,
    float* __restrict__ sc_out,
    int bh, int total
){
    int b=blockIdx.x, tid=threadIdx.x, bsz=blockDim.x;
    int sl=sl_arr[b], off=offsets[b];
    const float* wt_b=wt+b*H_DIM;
    int h_off=b*H_DIM;
    for(int t=tid; t<sl; t+=bsz){
        int tok=off+t;
        float ks=ksc[tok];
        const __half* row=raw+(long)tok*bh+h_off;
        float score=0.f;
        for(int h=0; h<H_DIM; h++){
            float dot=__half2float(row[h])*ks;
            score+=(dot>0.f?dot:0.f)*wt_b[h];
        }
        sc_out[(long)b*BSORT_N+t]=score;
    }
}

__global__ void sort_bsort_n(
    float* __restrict__ sc,    // (BSORT_N,) in smem via global ptr
    int* __restrict__ id,
    int sl,
    int* __restrict__ out_b
){
    int tid=threadIdx.x, bsz=blockDim.x;
    extern __shared__ char smem[];
    float* sc_sh=(float*)smem;
    int* id_sh=(int*)(smem+BSORT_N*4);
    for(int i=tid;i<sl;i+=bsz){sc_sh[i]=sc[i];id_sh[i]=id[i];}
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

torch::Tensor run_score_only(
    torch::Tensor raw, torch::Tensor ksc, torch::Tensor wt,
    torch::Tensor offsets, torch::Tensor sls, int bh, int total
){
    int B=sls.size(0);
    auto sc=torch::zeros({B,BSORT_N},torch::dtype(torch::kFloat32).device(raw.device()));
    score_from_raw<<<B,256>>>((const __half*)raw.data_ptr(),ksc.data_ptr<float>(),wt.data_ptr<float>(),
        offsets.data_ptr<int>(),sls.data_ptr<int>(),sc.data_ptr<float>(),bh,total);
    return sc;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_score_only",&run_score_only);}
"""
tmpdir = tempfile.mkdtemp(); src = os.path.join(tmpdir, "sc_only.cu")
open(src, "w").write(CUDA)
mod = ext.load(name="sc_only_ext", sources=[src], extra_cuda_cflags=["-O3","--use_fast_math"], verbose=False)

t = bench(lambda: mod.run_score_only(raw_trans, Ks.contiguous(), weights.contiguous(), offsets_t, seq_lens.int(), B*H, total))
p = prof_span(lambda: mod.run_score_only(raw_trans, Ks.contiguous(), weights.contiguous(), offsets_t, seq_lens.int(), B*H, total))
print(f"score_from_raw: events={t:.2f}us, profiler={p:.2f}us")
PYEOF
