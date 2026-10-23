#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch
import sys; sys.path.insert(0,".")
from bench_all import make_inputs
# Profile where time goes in full_pipe
# gather vs gemm vs topk independently
q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128; PAGE_SZ=64

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

# Gather for batch2 only
sl2=seq_lens[2].item()
t2=torch.arange(sl2,device="cuda")
pages2=block_table[2,t2//PAGE_SZ]; offs2=t2%PAGE_SZ; flat2=pages2*PAGE_SZ+offs2
k_raw=k_cache.view(-1,132)

def gather2():
    kd=k_raw[flat2,:D].contiguous()
    ks=k_raw[flat2,D:D+4].contiguous()
    return kd,ks
print(f"gather batch2: {bench(gather2):.2f} us")

# Pre-gathered data
kd_b2,ks_b2=gather2()
kd_fp8=kd_b2.view(torch.uint8).view(torch.float8_e4m3fn)
ks_f=ks_b2.view(torch.float32).view(sl2)
pad=(16-(sl2%16))%16
kd_pad=torch.cat([kd_b2.view(torch.uint8),torch.zeros(pad,D,device="cuda",dtype=torch.uint8)]).view(torch.float8_e4m3fn)
kd_T=kd_pad.T.contiguous()

def just_gemm():
    return torch._scaled_mm(q_fp8[2],kd_T,
        scale_a=torch.tensor(1.0,device="cuda"),
        scale_b=torch.tensor(1.0,device="cuda"),
        out_dtype=torch.bfloat16).float()[:,:sl2]
print(f"just gemm batch2: {bench(just_gemm):.2f} us")

# What about using bfloat16 matmul?
q_bf=q_fp8[2].float().bfloat16()
kd_bf=kd_fp8.float().bfloat16()
def just_gemm_bf():
    return torch.mm(q_bf, kd_bf.T)[:,:sl2]
print(f"bfloat16 matmul batch2: {bench(just_gemm_bf):.2f} us")

# What about float32 matmul?
q_f=q_fp8[2].float()
kd_f=kd_fp8.float()
def just_gemm_f32():
    return torch.mm(q_f, kd_f.T)
print(f"float32 matmul batch2: {bench(just_gemm_f32):.2f} us")

# What about: process all batches together (pad and batch the gemm)?
# Build K for all batches as one tensor
# Max seq = 2817, use topk on full 2817 padded tensor
all_sl=[seq_lens[b].item() for b in range(B)]
max_sl=max(all_sl)
print(f"\nmax_sl={max_sl}")
# Gather all k into (B, max_sl, D) -- padding with zeros
# Then batched gemm: q(B,H,D) @ K^T(B,D,max_sl) = (B,H,max_sl)
all_kd=torch.zeros(B,max_sl,D,device="cuda",dtype=torch.uint8)
all_ks=torch.zeros(B,max_sl,device="cuda",dtype=torch.float32)
for b in range(B):
    sl=all_sl[b]
    t=torch.arange(sl,device="cuda")
    pg=block_table[b,t//PAGE_SZ]; off=t%PAGE_SZ
    flat=pg*PAGE_SZ+off
    all_kd[b,:sl]=k_raw[flat,:D].contiguous().view(torch.uint8)
    all_ks[b,:sl]=k_raw[flat,D:D+4].contiguous().view(torch.float32).view(sl)
all_kd_f8=all_kd.view(torch.float8_e4m3fn)

def gather_all():
    kd2=torch.zeros(B,max_sl,D,device="cuda",dtype=torch.uint8)
    ks2=torch.zeros(B,max_sl,device="cuda",dtype=torch.float32)
    for b in range(B):
        sl=all_sl[b]
        t=torch.arange(sl,device="cuda")
        pg=block_table[b,t//PAGE_SZ]; off=t%PAGE_SZ
        flat=pg*PAGE_SZ+off
        kd2[b,:sl]=k_raw[flat,:D].contiguous().view(torch.uint8)
        ks2[b,:sl]=k_raw[flat,D:D+4].contiguous().view(torch.float32).view(sl)
    return kd2,ks2
print(f"gather all batches: {bench(gather_all,w=3,n=10):.2f} us")

# Batched float16 matmul
q_f16=q_fp8.float().half()  # (B,H,D)
kd_f16=all_kd_f8.float().half()  # (B,max_sl,D)
# batched mm: (B,H,D) @ (B,D,max_sl) = (B,H,max_sl)
def batched_mm():
    return torch.bmm(q_f16, kd_f16.transpose(1,2))
print(f"batched float16 mm (B,H,D)@(B,D,max_sl): {bench(batched_mm):.2f} us")
PYEOF
