#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, torch.utils.cpp_extension as ext, os, tempfile
import sys; sys.path.insert(0,".")
from bench_all import make_inputs

# The ref achieves 4.4 us for this workload.
# Our best so far: ~80 us (score) + ~40 us (topk) = ~120 us.
# The ref must use tensor cores differently.
#
# Key: q is SHARED across all tokens of batch b.
# q[b]: (H=64, D=128) fp8. For all N tokens, we compute q @ K^T = (H, N).
# This is a GEMM: (64 x 128) @ (128 x N) = (64 x N).
# For batch 2: N=2817. This is a medium GEMM.
#
# B200 FP8 GEMM throughput: 18 PFLOPS for full chip (132 SMs).
# FLOPs for batch 2: 2 * 64 * 128 * 2817 = 46M. Time: 46M/18P = 2.6 ns.
# But in practice, overhead dominates for small GEMMs.
#
# The ref achieves 4.4 us total for B=8 batches simultaneously.
# Total FLOPs: 2 * 64 * 128 * 4061 = 66M. At 18P: 3.7 ns. Overhead = 4.4 us.
# So the ref is heavily overhead-bound, not compute-bound!
#
# The overhead comes from: kernel launch + memory access + top-k.
# The ref's single kernel likely:
# 1. One block per batch (8 blocks total on 8 SMs)
# 2. Uses WGMMA to do q @ K^T with tensor cores efficiently
# 3. Fuses score computation with top-k radix sort
#
# With only 8 blocks on 8 SMs, the key is LOW LATENCY not HIGH THROUGHPUT.
# 
# Let's try: convert q and k to float16, do a single batched matmul,
# then topk. All in PyTorch (which calls cuBLAS/cuDNN with tensor cores).

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128; PAGE_SZ=64
k_raw=k_cache.view(-1,132)
all_sl=[seq_lens[b].item() for b in range(B)]
max_sl=max(all_sl)

# Pre-build K matrices (to avoid gather overhead in hot path)
# For a real implementation, K would be pre-built or gathered by a CUDA kernel
# Here we test the matmul + topk part assuming K is already gathered

# Build K: (B, max_sl, D) float16
K_f16=torch.zeros(B,max_sl,D,device="cuda",dtype=torch.float16)
K_scales=torch.zeros(B,max_sl,device="cuda",dtype=torch.float32)
K_gids=torch.zeros(B,max_sl,device="cuda",dtype=torch.int32)
for b in range(B):
    sl=all_sl[b]
    t=torch.arange(sl,device="cuda")
    pg=block_table[b,t//PAGE_SZ]; off=t%PAGE_SZ
    flat=pg*PAGE_SZ+off
    kd=k_raw[flat,:D].view(torch.uint8).view(torch.float8_e4m3fn).float().half()
    K_f16[b,:sl]=kd
    K_scales[b,:sl]=k_raw[flat,D:D+4].contiguous().view(torch.float32).view(sl)
    K_gids[b,:sl]=(pg*PAGE_SZ+off).int()

q_f16=q_fp8.float().half()  # (B,H,D)
w_f32=weights  # (B,H)

buf=torch.empty(512*1024*1024//4,dtype=torch.float32,device="cuda")
def bench(fn,w=5,n=30):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(n):
        buf.fill_(0.); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1000)
    ts.sort(); return ts[len(ts)//2]

# Test: batched matmul only
def mm_only():
    # (B,H,D) @ (B,D,max_sl) = (B,H,max_sl)
    return torch.bmm(q_f16, K_f16.transpose(1,2))
print(f"batched mm f16: {bench(mm_only):.2f} us")

# Test: mm + scale + relu + weighted sum + topk
def full_fast():
    dots=torch.bmm(q_f16, K_f16.transpose(1,2)).float()  # (B,H,max_sl)
    dots=dots*K_scales.unsqueeze(1)   # (B,H,max_sl) * (B,1,max_sl)
    relu_d=torch.relu(dots)
    # weighted sum: (B,H,1) * (B,H,max_sl) -> sum over H -> (B,max_sl)
    scores=(w_f32.unsqueeze(2)*relu_d).sum(1)  # (B,max_sl)
    _,topk_idx=torch.topk(scores,2048,dim=1)  # (B,2048)
    out=torch.gather(K_gids,1,topk_idx)
    return out
print(f"full_fast (prebuilt K): {bench(full_fast):.2f} us")

# Check if we can store K_f16 as fp8 and do batched mm
# Actually for sub-4us target we need to look at CUDA kernel
# The ref latency of 4.4 us = ~4400 ns means:
# - ~500 ns kernel launch overhead
# - ~400 ns memory loads (k data from L2)  
# - ~1000 ns compute (fp8 tensor cores, 8 SMs)
# - ~2500 ns top-k (radix sort 2817 elements)
# Total: 4400 ns
#
# Our multiblock score alone is 80 us = 80000 ns.
# The ref uses tensor cores which are 10x+ faster for small batched GEMMs.
# The key question: can we reproduce the ref's approach?
#
# Evidence: batched f16 mm takes only 27 us (from diag8).
# After relu+sum+topk: add ~10 us = 37 us total. Still 8x slower than ref.
#
# The ref must use a single fused kernel with warp-group MMA (WGMMA).
# This allows the tensor core GEMM and top-k to be fused, avoiding
# intermediate global memory writes.
#
# For a CUDA implementation that beats 4.4 us:
# We need a custom kernel using __mma or wgmma PTX instructions.
# This is complex but achievable.

print("\n--- Testing CUB-based approach ---")

CUDA_SRC2 = r"""
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <torch/extension.h>
using namespace nvcuda;

#define H_DIM 64
#define D_DIM 128
#define PAGE_SZ 64
#define K_STRIDE 132
#define TOPK_K 2048
#define WARP_SIZE 32

// Use WMMA for fp8: not directly supported in older APIs.
// Use mma.sync.aligned for fp8 on Blackwell:
// mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// This computes C = A*B + C where A(16x32 fp8) B(32x8 fp8) C(16x8 f32)

// For our problem: q(64x128 fp8) @ K^T(128xN fp8) = (64xN f32)
// Split into tiles of (16x32) @ (32x8) = (16x8) accumulating

// Actually let's use the simpler approach: convert to half and use
// wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
// q (64,128) = 4 row-tiles x 8 col-tiles of 16x16
// K^T (128,N) per token group

// Simpler: use PTX fp8 mma directly
// Or: use half-precision WMMA with fp8->half conversion in registers

// APPROACH: use WMMA with fp16 inputs (converted from fp8)
// Each warp computes a 16x16 tile of the output matrix

// For batch b:
// q: (64, 128) fp8 -> store as 4x8 tiles of 16x16 half
// K: (N, 128) fp8 per token -> need to process tiles of tokens
// Output: (64, N) f32 -> weighted sum -> (N,) score

// One SM per batch, using 4 warps (128 threads)
// Each warp computes a (16, N_tile) output tile using WMMA

// This is getting complex. Let me try the simplest possible fusion:
// One block per batch, 256 threads, using __mma_sync for fp16

// Actually, let me try using cutlass for this. But cutlass is complex.
// Alternative: use torch.compile or triton.

torch::Tensor placeholder(torch::Tensor x) { return x; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("placeholder",&placeholder);}
"""

# Instead let's try triton
try:
    import triton
    import triton.language as tl
    print(f"Triton version: {triton.__version__}")
    
    @triton.jit
    def score_kernel(
        q_ptr, k_ptr, w_ptr, sl_ptr, bt_ptr, sc_ptr, id_ptr,
        B, mp: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
        K_STRIDE: tl.constexpr, PAGE_SZ: tl.constexpr,
        BLOCK_T: tl.constexpr
    ):
        b = tl.program_id(0)
        pg_idx = tl.program_id(1)
        sl = tl.load(sl_ptr + b)
        t_base = pg_idx * PAGE_SZ
        if t_base >= sl: return
        
        page = tl.load(bt_ptr + b * mp + pg_idx)
        
        # Load q[b]: (H, D) fp8
        q_offs = b * H * D + tl.arange(0, H * D)
        q_fp8 = tl.load(q_ptr + q_offs).to(tl.float32)  # (H*D,)
        
        # Load weights[b]: (H,)
        w_offs = b * H + tl.arange(0, H)
        w = tl.load(w_ptr + w_offs)  # (H,) float32
        
        # Process tokens in this page
        for loc in range(PAGE_SZ):
            t = t_base + loc
            if t >= sl: break
            kp = (page * PAGE_SZ + loc) * K_STRIDE
            # Load k fp8 (D bytes) + scale (4 bytes)
            k_fp8 = tl.load(k_ptr + kp + tl.arange(0, D)).to(tl.float32)
            k_scale = tl.load(k_ptr + kp + D).to(tl.float32)
            # Score
            k_true = k_fp8 * k_scale
            # Compute 64 dot products
            q_mat = tl.reshape(q_fp8, (H, D))
            dots = tl.sum(q_mat * k_true[None,:], axis=1)  # (H,)
            relu_dots = tl.maximum(dots, 0.0)
            score = tl.sum(relu_dots * w)
            tl.store(sc_ptr + b * 4096 + t, score)
            tl.store(id_ptr + b * 4096 + t, (page * PAGE_SZ + loc).to(tl.int32))
    
    q_u8=q_fp8.view(torch.uint8)
    q_i8=q_u8.view(torch.int8).contiguous()  # triton needs int8 for fp8
    k_i8=k_cache.contiguous()  # already int8
    sc_out=torch.full((B,4096),-1e38,dtype=torch.float32,device="cuda")
    id_out=torch.full((B,4096),-1,dtype=torch.int32,device="cuda")
    
    # Test triton kernel
    try:
        score_kernel[(B,block_table.shape[1])](
            q_i8,k_i8,w_f32,seq_lens,block_table,sc_out,id_out,
            B=B,mp=block_table.shape[1],H=H,D=D,
            K_STRIDE=132,PAGE_SZ=64,BLOCK_T=64
        )
        torch.cuda.synchronize()
        print("Triton kernel ran successfully")
        def triton_bench():
            sc_out.fill_(-1e38); id_out.fill_(-1)
            score_kernel[(B,block_table.shape[1])](
                q_i8,k_i8,w_f32,seq_lens,block_table,sc_out,id_out,
                B=B,mp=block_table.shape[1],H=H,D=D,
                K_STRIDE=132,PAGE_SZ=64,BLOCK_T=64
            )
        print(f"triton score: {bench(triton_bench):.2f} us")
    except Exception as e:
        print(f"Triton kernel error: {e}")
except ImportError:
    print("Triton not available")
PYEOF
