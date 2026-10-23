#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys; sys.path.insert(0,".")
from bench_all import make_inputs

# Understand ref's approach: it does 4.4 us total for 4061 tokens, 64 heads, 128 dim.
# B200 FP8 tensor core GEMM throughput: let's measure actual small GEMMs.
# The ref uses ONE kernel covering ALL 8 batches with total 4061 tokens.
# It processes them all simultaneously.
#
# If the ref packs ALL batches' tokens into one K matrix and does a single GEMM:
# q_packed: (B, H, D) = (8, 64, 128) fp8
# K_packed: (sum(seq_lens)=4061, D=128) fp8
# GEMM: for each batch b: q[b] @ K[offset_b:offset_b+sl_b]^T = (64, sl_b)
# This is a batched GEMM with variable-length sequences.
# sgemm (fp8) for total work: 2 * 64 * 128 * 4061 = 66M flops

# CuBLAS grouped GEMM or CuTe can do this.
# But even simpler: torch.mm on fp16 takes 24 us for (64,128)@(128,2817).
# For the full problem (8 batches): 24 us * 8 = 192 us sequential.
# But ref is 4.4 us, meaning it's 44x faster.

# The key insight I'm missing: the ref uses a RADIX SORT top-k kernel that 
# FUSES score computation and sorting. Let's check if there's a fundamentally
# different algorithm.

# Actually - maybe I should re-examine the problem scale.
# seq_lens=[187,133,2817,336,135,149,130,174] sum=4061
# The ref's "AKO v8_radix_bt256" is specifically optimized for this pattern.
# 
# Let me think about what makes 4.4 us possible:
# Total tokens: 4061. At 4.4 us, that's 4061/4.4us = 923 tokens/us.
# Score per token: 64 heads * 128 dims = 8192 FP8 MACs.
# Total MACs: 4061 * 8192 = 33.3M FP8 MACs.
# B200 FP8 TC GEMM: 18 PFLOPS for full chip.
# At 8 SMs (one per batch): 8/132 * 18P = 1.09 TFLOPS.
# Time: 33.3M / 1.09T = 30.5 ns!
# The ref uses 4400 ns. So it only uses 30/4400 = 0.7% of peak FP8 TC.
# 
# The actual compute is tiny. The BOTTLENECK IS TOP-K SELECTION.
# 
# The radix sort in the ref: sort 2817 elements to find top-2048.
# Standard radix sort: multiple passes over data.
# In shared memory with 256 threads: very fast.
# 
# For my scalar kernel (80 us), the bottleneck is score computation.
# My 32 inner iterations * fp8 conversion overhead is the issue.
# 
# The ref likely uses WGMMA to compute all 64 head dot-products simultaneously
# using tensor cores, making score computation ~10x faster.
#
# Let me try a completely different approach: use cuBLAS GEMM for scoring,
# then a fast top-k.

# Approach: for each batch b, run cuBLAS fp16 GEMM: q[b] (64,128) @ K[b]^T (128,sl) = (64,sl)
# Then apply scale, relu, weighted sum, topk.
# Key: use streams to parallelize across batches.

# But we need K to be pre-gathered as float16.
# Let's measure gather + f16 GEMM per batch on separate streams.

q_fp8, k_cache, weights, seq_lens, block_table = make_inputs("cuda")
B, H, D = 8, 64, 128; PAGE_SZ = 64
k_raw = k_cache.view(-1, 132)

# Pre-build data for each batch (simulate what a fast gather kernel would do)
qs, Ks, Kscales, Kgids = [], [], [], []
for b in range(B):
    sl = seq_lens[b].item()
    t = torch.arange(sl, device="cuda")
    pg = block_table[b, t//PAGE_SZ]; off = t % PAGE_SZ
    flat = pg * PAGE_SZ + off
    K = k_raw[flat, :D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn).float().half()
    Ks.append(K)  # (sl, D)
    Kscales.append(k_raw[flat, D:D+4].contiguous().view(torch.float32).view(sl))
    Kgids.append((pg * PAGE_SZ + off).int())
    qs.append(q_fp8[b].float().half())  # (H, D)

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

# Sequential GEMM for all batches
def seq_gemm():
    scores = []
    for b in range(B):
        sl = seq_lens[b].item()
        dots = torch.mm(qs[b], Ks[b].T).float()  # (H, sl)
        dots = dots * Kscales[b].unsqueeze(0)
        sc = (weights[b].unsqueeze(1) * torch.relu(dots)).sum(0)
        scores.append(sc)
    return scores
print(f"seq gemm all batches: {bench(seq_gemm):.2f} us")

# Parallel GEMM using multiple CUDA streams
streams = [torch.cuda.Stream() for _ in range(B)]
def par_gemm():
    results = [None] * B
    for b in range(B):
        with torch.cuda.stream(streams[b]):
            sl = seq_lens[b].item()
            dots = torch.mm(qs[b], Ks[b].T).float()
            dots = dots * Kscales[b].unsqueeze(0)
            results[b] = (weights[b].unsqueeze(1) * torch.relu(dots)).sum(0)
    torch.cuda.synchronize()
    return results
print(f"par gemm (8 streams): {bench(par_gemm):.2f} us")

# Pack all tokens into one matrix for one big GEMM
# q: (B*H, D), K: (total_tokens, D) -> (B*H, total_tokens)
# But different batches use different q rows -> need grouped GEMM
# Simpler: pad K to (B, max_sl, D) and do batched GEMM
max_sl = max(seq_lens.tolist())
K_pad = torch.zeros(B, max_sl, D, device="cuda", dtype=torch.float16)
for b in range(B): K_pad[b, :Ks[b].shape[0]] = Ks[b]
q_stack = torch.stack(qs)  # (B, H, D)

def batched_gemm():
    dots = torch.bmm(q_stack, K_pad.transpose(1,2)).float()  # (B,H,max_sl)
    # ... (no scale/relu for timing)
    return dots
print(f"batched gemm (B,H,D)@(B,D,max_sl): {bench(batched_gemm):.2f} us")

# Full batched with scale, relu, sum, topk
Ks_pad = torch.zeros(B, max_sl, device="cuda", dtype=torch.float32)
for b in range(B): Ks_pad[b, :Kscales[b].shape[0]] = Kscales[b]

def full_batched():
    dots = torch.bmm(q_stack, K_pad.transpose(1,2)).float()  # (B,H,max_sl)
    dots = dots * Ks_pad.unsqueeze(1)  # scale
    sc = (weights.unsqueeze(2) * torch.relu(dots)).sum(1)  # (B,max_sl)
    _, idx = torch.topk(sc, 2048, dim=1)
    return idx
print(f"full batched gemm+topk: {bench(full_batched):.2f} us")

# Profiler kernel span
import torch.profiler as P
fn = full_batched
for _ in range(5): fn()
torch.cuda.synchronize()
buf.fill_(0.); torch.cuda.synchronize()
with P.profile(activities=[P.ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        buf.fill_(0.); fn()
torch.cuda.synchronize()
evts = prof.profiler.function_events
total = sum(e.device_time for e in evts if e.device_time > 0 and "fill" not in e.name.lower())
print(f"full_batched profiler kernel span: {total/5:.2f} us")
PYEOF
