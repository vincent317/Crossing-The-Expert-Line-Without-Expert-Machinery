#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm, torch.profiler as P

# deep_gemm kernel device_time = 5.44 us per call! NOT 0.005 us.
# The confusion was that it showed "0.005" earlier but that was the formatted/rounded display.
# Actual: 5.44 us kernel time per deep_gemm bf16 GEMM call.
# 
# So for target ≤ 3.67 us, even deep_gemm GEMM alone is 5.44 us -- too slow.
# 
# CONCLUSION: No approach without a SINGLE CUSTOM KERNEL using tensor cores will achieve <4.4 us.
# 
# Let me try a completely different approach for the GEMM:
# Use __hmma (half-precision mma.sync) directly in a CUDA kernel.
# Or: use the torch.nn.functional.linear (which calls cuBLAS and may be faster for tiny matrices).
#
# Actually, wait. Let me re-read the measurement in bench_all.py:
# prof_span sums device_time for all non-fill kernels.
# For our goal: new kernel device time ≤ ref / 1.2 = 4.4 / 1.2 = 3.67 us.
#
# What if we use 2 CONCURRENT kernels (different CUDA streams) that overlap in time?
# Then the SPAN is max(t1, t2) not t1+t2.
# 
# But our profiler code sums ALL device_times. So overlapping doesn't help for
# the sum metric.
# 
# ACTUAL: the task says "device time 口径 = GPU上kernel的实际执行跨度（首个kernel开始到最后一个kernel结束）"
# = SPAN from first kernel start to last kernel end.
# Our profiler code sums device_times (NOT span). This is the wrong metric!
# 
# Let me fix bench_all.py to compute actual SPAN (first start to last end),
# not sum of durations.
# 
# With the SPAN metric:
# - 2 concurrent kernels of 2 us each = SPAN of 2 us (if truly parallel)
# - With our multiblock score kernel (80 us single block) vs ref's 4.4 us
# 
# For ref: 1 kernel → span = duration = 4.4 us.
# For our approach with 2 overlapping kernels: span = max(duration1, duration2) if parallel.
# 
# BUT: if we just use a gather kernel (fast, <1us) + score+topk kernel:
# SPAN = time from gather kernel START to score+topk kernel END.
# = gather_duration + score_topk_duration (sequential, unless different streams)
# 
# For 2 kernels on same stream: SPAN = sum of durations.
# For 2 kernels on different streams (truly parallel): SPAN = max of durations.
# 
# INSIGHT: If we can run gather + score in PARALLEL (different streams with overlapping execution):
# - Gather kernel for all pages: very fast (<5 us)
# - Score kernel: fast
# - SPAN = max(gather, score) instead of sum
# 
# But gather must complete BEFORE score (score uses gathered data). Can't overlap.
# Unless... we fuse them.
# 
# This confirms: we NEED a single fused kernel.
# 
# NEW APPROACH: write a kernel that uses B200's warp-group MMA (WGMMA).
# B200 supports: UMMA (unified MMA) instructions with much higher throughput.
# 
# Let me try using torch.nn.functional.linear which may use cuBLAS with different scheduler.
# torch.mm on fp16 uses cuBLAS which may be faster than deep_gemm for small matrices.
# 
# Actually from diag8: "bfloat16 matmul batch2: 24.29 us" and "float32 matmul batch2: 23.42 us"
# torch.mm for (64,128)@(128,2817) = 24 us CUDA events. Profiler?

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
K_bf16 = K_fp8.float().bfloat16().contiguous()
q_bf16 = q_fp8.reshape(B*H,D).float().bfloat16().contiguous()

buf = torch.empty(512*1024*1024//4,dtype=torch.float32,device="cuda")
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

# Measure profiler span for torch.mm
p = prof_span(lambda: torch.mm(q_bf16, K_bf16.T))
print(f"torch.mm bf16 (512,128)x(128,4061): profiler_sum={p:.2f} us")

# torch.mm fp16
q_f16 = q_fp8.reshape(B*H,D).float().half().contiguous()
K_f16 = K_fp8.float().half().contiguous()
p2 = prof_span(lambda: torch.mm(q_f16, K_f16.T))
print(f"torch.mm f16 (512,128)x(128,4061): profiler_sum={p2:.2f} us")

# torch._scaled_mm fp8  
pad=(16-(total%16))%16
K_fp8_pad = torch.cat([K_fp8.view(torch.uint8), torch.zeros(pad,D,device="cuda",dtype=torch.uint8)]).view(torch.float8_e4m3fn) if pad>0 else K_fp8
q_all_fp8 = q_fp8.reshape(B*H,D).contiguous()
p3 = prof_span(lambda: torch._scaled_mm(q_all_fp8, K_fp8_pad.T.contiguous(), scale_a=torch.tensor(1.0,device="cuda"), scale_b=torch.tensor(1.0,device="cuda"), out_dtype=torch.float32))
print(f"_scaled_mm fp8 (512,128)x(128,{total+pad}): profiler_sum={p3:.2f} us")

# Try: deep_gemm bf16 with TN layout (might be faster)
q_T = q_bf16.T.contiguous()  # (D, B*H) = (128, 512)
K_T_cont = K_bf16.T.contiguous()  # (D, total) = (128, 4061)
M_a=((B*H+127)//128)*128; K_a=((D+127)//128)*128; N_a=((total+127)//128)*128
q_T_pad=torch.zeros(K_a,M_a,device="cuda",dtype=torch.bfloat16); q_T_pad[:D,:B*H]=q_T
K_T_pad=torch.zeros(K_a,N_a,device="cuda",dtype=torch.bfloat16); K_T_pad[:D,:total]=K_T_cont
d_tn=torch.zeros(M_a,N_a,device="cuda",dtype=torch.bfloat16)
try:
    deep_gemm.bf16_gemm_tn(q_T_pad,K_T_pad,d_tn)
    torch.cuda.synchronize()
    p4=prof_span(lambda: deep_gemm.bf16_gemm_tn(q_T_pad,K_T_pad,d_tn))
    print(f"deep_gemm bf16 TN: profiler_sum={p4:.2f} us")
except Exception as e:
    print(f"deep_gemm TN error: {e}")
PYEOF
