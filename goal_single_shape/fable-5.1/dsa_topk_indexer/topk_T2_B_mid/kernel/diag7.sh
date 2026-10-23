#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch
import sys; sys.path.insert(0,".")
from bench_all import make_inputs

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128
PAGE_SZ=64

# ks shape was (2817,1) but we need (2817,)
# Fix: use view instead of reshape

def gather_k(b, sl):
    t=torch.arange(sl,device="cuda")
    pages=block_table[b,t//PAGE_SZ]
    offs=t%PAGE_SZ
    flat=pages*PAGE_SZ+offs
    k_raw=k_cache.view(-1,132)
    k_fp8=k_raw[flat,:D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)
    # Extract float32 scale: 4 int8 bytes -> 1 float32
    k_scale=k_raw[flat,D:D+4].contiguous().view(torch.float32).view(sl)  # (sl,)
    return k_fp8, k_scale

def fp8_gemm(b):
    sl=seq_lens[b].item()
    kd,ks=gather_k(b,sl)
    pad=(16-(sl%16))%16
    if pad>0:
        kd=torch.cat([kd.view(torch.uint8),
            torch.zeros(pad,D,device="cuda",dtype=torch.uint8)]).view(torch.float8_e4m3fn)
    q_b=q_fp8[b]  # (H,D) fp8
    dots=torch._scaled_mm(q_b,kd.T.contiguous(),
        scale_a=torch.tensor(1.0,device="cuda"),
        scale_b=torch.tensor(1.0,device="cuda"),
        out_dtype=torch.bfloat16).float()[:,:sl]  # (H,sl)
    dots=dots*ks.unsqueeze(0)     # (H,sl) * (1,sl) = (H,sl)
    scores=(weights[b].unsqueeze(1)*torch.relu(dots)).sum(dim=0)  # (sl,)
    return scores

# Quick sanity
s=fp8_gemm(2)
print(f"batch2 scores: {s.shape} [{s.min().item():.3f},{s.max().item():.3f}]")

buf=torch.empty(512*1024*1024//4,dtype=torch.float32,device="cuda")
def bench(fn,w=5,n=20):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(n):
        buf.fill_(0.); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1000)
    ts.sort(); return ts[len(ts)//2]

print(f"batch2 fp8_gemm: {bench(lambda:fp8_gemm(2)):.2f} us")

def all_batches_gemm():
    out=[]
    for b in range(B):
        out.append(fp8_gemm(b))
    return out
print(f"all_batches fp8_gemm: {bench(all_batches_gemm,w=3,n=10):.2f} us")

# Full pipeline: gather + gemm + topk
def full_pipe():
    topk_out=torch.full((B,2048),-1,dtype=torch.int32,device="cuda")
    for b in range(B):
        sl=seq_lens[b].item()
        k=min(2048,sl)
        if k==0: continue
        t=torch.arange(sl,device="cuda")
        pages=block_table[b,t//PAGE_SZ]
        offs=t%PAGE_SZ
        flat=pages*PAGE_SZ+offs
        k_raw=k_cache.view(-1,132)
        kd=k_raw[flat,:D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)
        ks=k_raw[flat,D:D+4].contiguous().view(torch.float32).view(sl)
        pad=(16-(sl%16))%16
        if pad>0:
            kd=torch.cat([kd.view(torch.uint8),torch.zeros(pad,D,device="cuda",dtype=torch.uint8)]).view(torch.float8_e4m3fn)
        dots=torch._scaled_mm(q_fp8[b],kd.T.contiguous(),
            scale_a=torch.tensor(1.0,device="cuda"),
            scale_b=torch.tensor(1.0,device="cuda"),
            out_dtype=torch.bfloat16).float()[:,:sl]*ks
        scores=(weights[b].unsqueeze(1)*torch.relu(dots)).sum(0)
        gids=(pages*PAGE_SZ+offs).int()
        if sl<=2048:
            topk_out[b,:sl]=gids
        else:
            idx=torch.topk(scores,k).indices
            topk_out[b,:k]=gids[idx]
    return topk_out

print(f"full_pipe (gather+gemm+topk): {bench(full_pipe,w=3,n=10):.2f} us")
PYEOF
