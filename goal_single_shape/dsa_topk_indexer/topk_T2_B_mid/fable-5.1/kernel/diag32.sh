#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys
sys.path.insert(0,".")
from bench_all import make_inputs
import torch.profiler as P

# Let me rethink. We've used 30+ diagnostic rounds and haven't beaten 4.4 us.
# 
# Current best profiler spans:
# - deep_gemm bf16 GEMM (512,128)x(4096,128): 4.6 us (close to ref)
# - score_from_raw (reading GEMM output): 100 us
# - bitonic sort (2817 elements): 291 us
# - score+sort combined: 890 us
# 
# The ref is 4.4 us for a SINGLE KERNEL. We can't match it with multiple kernels.
# 
# NEW STRATEGY: Instead of beating the ref's algorithm, let me look at what
# "device time" measurement actually means in our profiler.
# 
# From bench_ref.py: profiler_kernel_us uses sum of all kernel device_times.
# If ref has 1 kernel at 4.4 us, we need our kernels to sum to ≤ 3.67 us.
# 
# With 2 kernels (GEMM + post): GEMM=4.6us + post=?. Even if post is 0, total = 4.6 us > 3.67.
# 
# What if we use a DIFFERENT, SMALLER GEMM?
# Instead of (512,128)x(4096,128), can we do the computation differently?
# 
# Actually, re-reading the measurement methodology:
# "device time口径 = GPU上kernel的实际执行跨度（首个kernel开始到最后一个kernel结束)"
# = SPAN from first kernel start to last kernel end = wall-clock GPU time.
# This is NOT the sum of kernel durations -- it's the span!
# 
# If 2 kernels overlap in time (different streams), their SPAN could be less than sum.
# With 2 overlapping kernels of 4.6 us and 2 us, the span could be max(4.6,2)=4.6 us.
# But to get span < 3.67 us, we need ONE kernel that runs in < 3.67 us.
# 
# Or: we use the same profiler method (sum of device_time for non-fill kernels).
# In that case: GEMM (4.6 us) + post (?) needs to sum to ≤ 3.67 us.
# This is IMPOSSIBLE since GEMM alone is 4.6 us.
# 
# CONCLUSION: We need a SINGLE KERNEL that:
# 1. Computes all scores using tensor cores (WGMMA/MMA)
# 2. Does top-k selection in shared memory
# 3. Runs in ≤ 3.67 us
# 
# This requires writing a custom kernel using WGMMA PTX instructions.
# B200 supports: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
# 
# Design for WGMMA-based kernel:
# - 1 block per batch (B=8 blocks)
# - 4 warp groups (128 threads) per block
# - Each warp group computes a tile of the score matrix
# - Then all threads collaborate on top-k
# 
# For batch 2 (2817 tokens, 64 heads, 128 dims):
# WGMMA shape: m16n8k32 for fp8
# q: (64,128) = 4×4 tiles of (16,32)
# K_page: (64,128) per page (64 tokens × 128 dims)
# Output per page: (64,64) = (64 heads × 64 tokens)
# 
# With warp group MMA: 4 warps × (64/16) × (64/8) × (128/32) = 4 × 4 × 8 × 4 = 512 mma ops
# ... complex but achievable
# 
# Given time constraints (approaching 40 rounds), let me write the best
# single-kernel implementation I can using __mma_sync (legacy) or inline PTX.
# 
# Actually, let me reconsider the profiler measurement.
# In bench_ref.py, prof_span sums ALL kernel device_times (non-fill).
# Let me verify: does our deep_gemm call result in ONE kernel or multiple?

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
    all_flat.append(pg*PAGE_SZ+off)
all_flat_t = torch.cat(all_flat)
K_fp8 = k_raw[all_flat_t,:D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)
K_bf16 = K_fp8.float().bfloat16()
q_bf16 = q_fp8.reshape(B*H,D).float().bfloat16()
M_a=((B*H+127)//128)*128; N_a=((total+127)//128)*128
K_pad = torch.zeros(N_a,D,device="cuda",dtype=torch.bfloat16); K_pad[:total]=K_bf16
q_pad = torch.zeros(M_a,D,device="cuda",dtype=torch.bfloat16); q_pad[:B*H]=q_bf16
d_out = torch.zeros(N_a,M_a,device="cuda",dtype=torch.bfloat16)
import deep_gemm

buf = torch.empty(512*1024*1024//4,dtype=torch.float32,device="cuda")
for _ in range(5): deep_gemm.bf16_gemm_nt(K_pad,q_pad,d_out)
torch.cuda.synchronize()
buf.fill_(0.); torch.cuda.synchronize()
with P.profile(activities=[P.ProfilerActivity.CUDA],record_shapes=True) as prof:
    for _ in range(5):
        buf.fill_(0.)
        deep_gemm.bf16_gemm_nt(K_pad,q_pad,d_out)
torch.cuda.synchronize()

evts = prof.profiler.function_events
print("Kernels in deep_gemm call:")
for e in evts:
    if e.device_time > 0 and "fill" not in e.name.lower():
        print(f"  {e.name}: {e.device_time/1e3:.3f} us")
PYEOF
