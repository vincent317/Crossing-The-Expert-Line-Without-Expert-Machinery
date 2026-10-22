#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, torch.utils.cpp_extension as ext, os, tempfile
import sys; sys.path.insert(0,".")
from bench_all import make_inputs
import torch.profiler as P

# Analysis:
# score_only profiler: 622 us (for B=8 blocks, all batches)
# sort_only profiler: 291 us (for batch 2, 2817 elements)
# total: ~911 us
#
# ref is 4.4 us = 140x faster than us.
# 
# The ONLY WAY to match ref is to use tensor cores for dot products.
# Our scalar dot4v approach: 622 us for score.
# With tensor cores: ~4.4 us total (ref). Score must be ~2-3 us.
#
# Key realization: the ref uses WGMMA (Warp Group Matrix Multiply Accumulate)
# which is Blackwell's native accelerator. It's 18x faster than scalar ops.
# 
# For each batch b, the ref computes q[b](64x128 fp8) @ K[b]^T(128xN fp8) = (64xN fp16).
# This is a GEMM that maps perfectly to tensor cores.
#
# With WGMMA (B200): MMA shape m16n8k32 for fp8.
# For q(64x128) @ K^T(128xN):
# - M=64 (heads), K=128 (dims), N=variable per batch
# 
# Using cutlass or PTX wgmma instructions would get us to 4.4 us.
#
# Alternative that might work: use flash_attn's CUDA infrastructure.
# The sglang image has flash-attn-4 installed!
# flash_attn_func from flash_attn.cute.interface uses CuTe DSL + WGMMA.
#
# But we need a custom op, not attention.
#
# Let me try a DIFFERENT APPROACH:
# Use cuda mma.sync (Ampere/Hopper) or wgmma PTX for fp8 matmul.
#
# Actually, let me check: does torch.mm with fp8 use tensor cores?
q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
print("torch version:", torch.__version__)
print("cuda version:", torch.version.cuda)

# Check if torch._scaled_mm (fp8 gemm) is fast enough
sl2=seq_lens[2].item()
t=torch.arange(sl2,device="cuda")
pages=block_table[2,t//64]; offs=t%64
flat=pages*64+offs
k_raw=k_cache.view(-1,132)
kd_u8=k_raw[flat,:128].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)
ks=k_raw[flat,128:132].contiguous().view(torch.float32).view(sl2)

# Pad to multiple of 16
pad=(16-(sl2%16))%16
if pad>0:
    kd_pad=torch.cat([kd_u8.view(torch.uint8),torch.zeros(pad,128,device="cuda",dtype=torch.uint8)]).view(torch.float8_e4m3fn)
else:
    kd_pad=kd_u8

q_b=q_fp8[2]  # (64,128) fp8

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

kd_T=kd_pad.T.contiguous()
def fp8_mm():
    return torch._scaled_mm(q_b,kd_T,
        scale_a=torch.tensor(1.0,device="cuda"),
        scale_b=torch.tensor(1.0,device="cuda"),
        out_dtype=torch.float32)
t=bench(fp8_mm)
p=prof_span(fp8_mm)
print(f"fp8 _scaled_mm (64x128)@(128,2832): events={t:.2f} us, profiler={p:.2f} us")

# Full score for batch 2 with fp8 scaled_mm
def score_b2_fp8():
    dots=torch._scaled_mm(q_b,kd_T,
        scale_a=torch.tensor(1.0,device="cuda"),
        scale_b=torch.tensor(1.0,device="cuda"),
        out_dtype=torch.float32)[:,:sl2]
    dots=dots*ks.unsqueeze(0)  # (64,sl2)
    scores=(weights[2].unsqueeze(1)*torch.relu(dots)).sum(0)
    return scores
t=bench(score_b2_fp8)
p=prof_span(score_b2_fp8)
print(f"fp8 score b2 (mm+scale+relu+sum): events={t:.2f} us, profiler={p:.2f} us")

# All 8 batches with fp8 mm (sequential, using pre-gathered K)
# Pre-gather
Ks_fp8=[]; Ks_scales=[]; Ks_gids=[]
for b in range(8):
    sl=seq_lens[b].item()
    t_=torch.arange(sl,device="cuda")
    pg=block_table[b,t_//64]; off=t_%64
    flat_=pg*64+off
    kd=k_raw[flat_,:128].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)
    pad_=(16-(sl%16))%16
    if pad_>0:
        kd=torch.cat([kd.view(torch.uint8),torch.zeros(pad_,128,device="cuda",dtype=torch.uint8)]).view(torch.float8_e4m3fn)
    Ks_fp8.append(kd.T.contiguous())
    Ks_scales.append(k_raw[flat_,128:132].contiguous().view(torch.float32).view(sl))
    Ks_gids.append((pg*64+off).int())

def all_batches_fp8():
    out=torch.full((8,2048),-1,dtype=torch.int32,device="cuda")
    for b in range(8):
        sl=seq_lens[b].item()
        dots=torch._scaled_mm(q_fp8[b],Ks_fp8[b],
            scale_a=torch.tensor(1.0,device="cuda"),
            scale_b=torch.tensor(1.0,device="cuda"),
            out_dtype=torch.float32)[:,:sl]
        dots=dots*Ks_scales[b].unsqueeze(0)
        scores=(weights[b].unsqueeze(1)*torch.relu(dots)).sum(0)
        k_out=min(2048,sl)
        if sl<=2048:
            out[b,:sl]=Ks_gids[b]
        else:
            idx=torch.topk(scores,k_out).indices
            out[b,:k_out]=Ks_gids[b][idx]
    return out
t=bench(all_batches_fp8,w=3,n=20)
p=prof_span(all_batches_fp8,n=5)
print(f"all_batches fp8 mm+topk (prebuilt K): events={t:.2f} us, profiler={p:.2f} us")

# Most important: can we gather k on-the-fly AND do fp8 mm fast?
# Gather + fp8 mm for all batches
def gather_and_fp8mm():
    out=torch.full((8,2048),-1,dtype=torch.int32,device="cuda")
    for b in range(8):
        sl=seq_lens[b].item()
        t_=torch.arange(sl,device="cuda")
        pg=block_table[b,t_//64]; off=t_%64
        flat_=pg*64+off
        kd=k_raw[flat_,:128].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)
        pad_=(16-(sl%16))%16
        if pad_>0:
            kd=torch.cat([kd.view(torch.uint8),torch.zeros(pad_,128,device="cuda",dtype=torch.uint8)]).view(torch.float8_e4m3fn)
        ks2=k_raw[flat_,128:132].contiguous().view(torch.float32).view(sl)
        dots=torch._scaled_mm(q_fp8[b],kd.T.contiguous(),
            scale_a=torch.tensor(1.0,device="cuda"),
            scale_b=torch.tensor(1.0,device="cuda"),
            out_dtype=torch.float32)[:,:sl]
        dots=dots*ks2.unsqueeze(0)
        scores=(weights[b].unsqueeze(1)*torch.relu(dots)).sum(0)
        gids=(pg*64+off).int()
        k_out=min(2048,sl)
        if sl<=2048:
            out[b,:sl]=gids
        else:
            idx=torch.topk(scores,k_out).indices
            out[b,:k_out]=gids[idx]
    return out
t=bench(gather_and_fp8mm,w=3,n=10)
p=prof_span(gather_and_fp8mm,n=3)
print(f"gather+fp8mm+topk all batches: events={t:.2f} us, profiler={p:.2f} us")
PYEOF
