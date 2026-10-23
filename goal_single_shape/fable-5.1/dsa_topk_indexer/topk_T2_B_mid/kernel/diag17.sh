#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, torch.utils.cpp_extension as ext, os, tempfile
import sys; sys.path.insert(0,".")
from bench_all import make_inputs
import torch.profiler as P

# fp8 _scaled_mm (64x128)@(128,2832): profiler=6.98 us per call
# all_batches fp8 mm+topk (prebuilt K): profiler=256 us (8 sequential mm calls)
# 
# 256 us / 8 batches = 32 us per batch avg for mm+topk.
# Each mm call is ~7 us. 8 sequential = 56 us just for mm.
# But profiler says 256 us → overhead from Python loops.
#
# Key insight: the profiler kernel span includes ALL kernels.
# fp8 mm b2: 6.98 us (kernel span for mm)
# score b2 full: 19.90 us (mm + scale + relu + sum)
# all_batches prebuilt: 256 us (8 × ~32 us each)
#
# We need the TOTAL kernel span to be < 3.67 us.
# But just ONE mm call = 7 us! That's already > our target.
#
# Conclusion: PyTorch's fp8 mm has high overhead for small matrices.
# The actual GEMM compute for (64,128)@(128,2832) = 0.03 us at 18 PFLOPS.
# The 7 us is entirely kernel launch + runtime overhead.
#
# To get to 4.4 us (ref) or better, ALL 8 batches must be processed in 
# ONE single CUDA kernel call with NO Python overhead.
#
# The ref achieves this by:
# 1. ONE custom kernel covering all 8 batches simultaneously
# 2. Tensor core (WGMMA/MMA) for score computation within the kernel
# 3. Radix sort in shared memory for top-k
#
# For us to beat 4.4 us: we need ONE kernel call that:
# - Processes all 8 batches in parallel (8 SMs)
# - Uses WGMMA for fp8 GEMV (not scalar ops)
# - Has a fast top-k algorithm (faster than bitonic sort for our sizes)
#
# WGMMA requires ptx or cutlass. Let me try cutlass.
# The sglang image should have cutlass headers since it's a deep learning image.

# Let me check what's available
import subprocess
r=subprocess.run(["find","/usr","-name","cutlass.h","-type","f"],capture_output=True,text=True)
if r.stdout:
    print("cutlass found:", r.stdout[:200])
else:
    r2=subprocess.run(["find","/opt","-name","cutlass.h","-type","f"],capture_output=True,text=True)
    if r2.stdout:
        print("cutlass found:", r2.stdout[:200])
    else:
        print("cutlass NOT found in /usr or /opt")

r3=subprocess.run(["find","/","-name","cute","maxdepth","6","-type","d"],capture_output=True,text=True)
if r3.stdout:
    print("cute dir:", r3.stdout[:200])

# Check CUDA PTX wgmma support
print("\nNVCC version:")
r4=subprocess.run(["nvcc","--version"],capture_output=True,text=True)
print(r4.stdout[:200])

# Try using PTX mma instructions for fp8 dot product
# mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 is sm_89+
# B200 is sm_100 which should support this.
# But this requires distributing q/k fragments across warp registers.
# This is complex to write manually.

# ALTERNATIVE: Use torch.mm on float16 but process ALL batches in ONE batched call.
# Instead of 8 sequential mm calls, do one batched mm.
# Key issue: different seq_lens means we need to handle variable-length batches.
# 
# Idea: stack all tokens from all batches into one K matrix (total 4061 tokens).
# Then do one BIG mm: q_all (8*64, 128) @ K_all^T (128, 4061) = (8*64, 4061).
# But each batch's q only applies to its own tokens, not all tokens.
# So result[b*64:(b+1)*64, offset_b:offset_b+sl_b] is valid for batch b.
# Result at cross-batch positions is garbage but we ignore it.

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128; PAGE_SZ=64
k_raw=k_cache.view(-1,132)
all_sl=[seq_lens[b].item() for b in range(B)]
total_tokens=sum(all_sl)  # 4061

# Gather ALL tokens from all batches into one K tensor
# Also build q_all: (B, H, D)
all_pages=[]
all_offs=[]
all_batch_ids=[]
offsets_start=[0]*B
cur=0
for b in range(B):
    sl=all_sl[b]
    offsets_start[b]=cur
    t=torch.arange(sl,device="cuda")
    pg=block_table[b,t//PAGE_SZ]; off=t%PAGE_SZ
    all_pages.append(pg); all_offs.append(off)
    all_batch_ids.append(torch.full((sl,),b,device="cuda",dtype=torch.long))
    cur+=sl

all_flat=torch.cat([ap*PAGE_SZ+ao for ap,ao in zip(all_pages,all_offs)])  # (4061,)
K_all_fp8=k_raw[all_flat,:D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)  # (4061,128)
Ks_all=k_raw[all_flat,D:D+4].contiguous().view(torch.float32).view(total_tokens)  # (4061,)
Gids_all=all_flat.int()  # (4061,)

# Pad K_all to multiple of 16
pad_all=(16-(total_tokens%16))%16
if pad_all>0:
    K_all_fp8=torch.cat([K_all_fp8.view(torch.uint8),torch.zeros(pad_all,D,device="cuda",dtype=torch.uint8)]).view(torch.float8_e4m3fn)

# q_all: (B*H, D) fp8
q_all=q_fp8.reshape(B*H,D)  # (512, 128) fp8

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

# ONE big fp8 mm: (512,128) @ (128, 4061+pad) = (512, 4061+pad)
K_T=K_all_fp8.T.contiguous()  # (128, 4061+pad)
def big_fp8_mm():
    return torch._scaled_mm(q_all,K_T,
        scale_a=torch.tensor(1.0,device="cuda"),
        scale_b=torch.tensor(1.0,device="cuda"),
        out_dtype=torch.float32)
t=bench(big_fp8_mm)
p=prof_span(big_fp8_mm)
print(f"ONE big fp8 mm (512,128)@(128,4072): events={t:.2f} us, profiler={p:.2f} us")

# Full pipeline: big mm + per-batch score + topk
def full_big_mm():
    raw=torch._scaled_mm(q_all,K_T,
        scale_a=torch.tensor(1.0,device="cuda"),
        scale_b=torch.tensor(1.0,device="cuda"),
        out_dtype=torch.float32)[:,:total_tokens]  # (B*H, total_tokens)
    # Apply per-token k_scale: raw[b*H:(b+1)*H, offset_b:offset_b+sl_b] *= Ks_all[offset_b:offset_b+sl_b]
    raw=raw*Ks_all.unsqueeze(0)  # broadcast scale over all batches (hack: applies wrong scale to cross-batch)
    raw=torch.relu(raw)  # (B*H, total_tokens)
    out=torch.full((B,2048),-1,dtype=torch.int32,device="cuda")
    for b in range(B):
        sl=all_sl[b]; off=offsets_start[b]
        # scores for batch b: raw[b*H:(b+1)*H, off:off+sl] weighted sum
        dots_b=raw[b*H:(b+1)*H, off:off+sl]  # (H, sl)
        scores=(weights[b].unsqueeze(1)*dots_b).sum(0)  # (sl,)
        k_out=min(2048,sl)
        if sl<=2048:
            out[b,:sl]=Gids_all[off:off+sl]
        else:
            idx=torch.topk(scores,k_out).indices
            out[b,:k_out]=Gids_all[off:off+sl][idx]
    return out
t=bench(full_big_mm,w=3,n=20)
p=prof_span(full_big_mm,n=5)
print(f"full_big_mm pipeline: events={t:.2f} us, profiler={p:.2f} us")
PYEOF
