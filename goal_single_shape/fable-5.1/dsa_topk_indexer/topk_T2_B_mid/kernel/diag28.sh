#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys, inspect
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm
import torch.profiler as P

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128; PAGE_SZ=64
num_sms=deep_gemm.get_num_sms()

# fp8_gemm_nt is actually fp8_fp4_gemm_nt. So deep_gemm.fp8_gemm_nt requires fp4?
# No wait -- fp8_fp4_gemm_nt is "A_fp8 @ B_fp4 = C". 
# Checking: what's the real fp8-only GEMM?
# Looking at the mapping: deep_gemm maps fp8_gemm_nt -> fp8_fp4_gemm_nt (their naming)
# This suggests deep_gemm supports fp8 A and fp4 B.
# Let's try bf16_gemm_nt which is simpler.

all_sl=[seq_lens[b].item() for b in range(B)]
total=sum(all_sl)
k_raw=k_cache.view(-1,132)

# Gather all tokens
all_flat=[]
for b in range(B):
    sl=all_sl[b]
    t=torch.arange(sl,device="cuda")
    pg=block_table[b,t//PAGE_SZ]; off=t%PAGE_SZ
    all_flat.append(pg*PAGE_SZ+off)
all_flat_t=torch.cat(all_flat)
K_fp8=k_raw[all_flat_t,:D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)
Ks=k_raw[all_flat_t,D:D+4].contiguous().view(torch.float32).view(total)
Gids=all_flat_t.int()

# Convert to bf16 for bf16_gemm_nt
K_bf16=K_fp8.float().to(torch.bfloat16)  # (total,D)
q_all=q_fp8.reshape(B*H,D).float().to(torch.bfloat16)  # (512,D)
q_all=q_all.contiguous()
K_bf16=K_bf16.contiguous()

# Pad for alignment
M,K_=q_all.shape; N_=total
M_a=((M+127)//128)*128; N_a=((N_+127)//128)*128
q_p=torch.zeros(M_a,K_,device="cuda",dtype=torch.bfloat16); q_p[:M]=q_all
K_p=torch.zeros(N_a,K_,device="cuda",dtype=torch.bfloat16); K_p[:N_]=K_bf16
d_out=torch.zeros(M_a,N_a,device="cuda",dtype=torch.bfloat16)

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

try:
    deep_gemm.bf16_gemm_nt(q_p,K_p,d_out)
    torch.cuda.synchronize()
    print(f"bf16_gemm_nt OK: {d_out.shape}")
    t=bench(lambda: deep_gemm.bf16_gemm_nt(q_p,K_p,d_out))
    p=prof_span(lambda: deep_gemm.bf16_gemm_nt(q_p,K_p,d_out))
    print(f"deep_gemm bf16 (512,128)x(4096,128): events={t:.2f}us profiler={p:.2f}us")
except Exception as e:
    print("bf16_gemm_nt error:", e)

# Also try torch.mm bf16 for same shapes
def torch_mm():
    return torch.mm(q_p[:M].contiguous(), K_p[:N_].T.contiguous())
t2=bench(torch_mm); p2=prof_span(torch_mm)
print(f"torch.mm bf16 (512,128)x(128,4061): events={t2:.2f}us profiler={p2:.2f}us")

# Full pipeline with torch.mm
offsets=[0]+list(torch.cumsum(torch.tensor(all_sl),0).tolist())
def full_torchm():
    raw=torch.mm(q_p[:M],K_p[:N_].T).float()  # (512, total)
    raw=raw*Ks.unsqueeze(0)
    raw=torch.relu(raw)
    out=torch.full((B,2048),-1,dtype=torch.int32,device="cuda")
    for b in range(B):
        sl=all_sl[b]; off=offsets[b]
        dots=raw[b*H:(b+1)*H, off:off+sl]
        scores=(weights[b].unsqueeze(1)*dots).sum(0)
        k_out=min(2048,sl)
        if sl<=2048:
            out[b,:sl]=Gids[off:off+sl]
        else:
            idx=torch.topk(scores,k_out).indices
            out[b,:k_out]=Gids[off:off+sl][idx]
    return out
t3=bench(full_torchm,w=3,n=20); p3=prof_span(full_torchm,n=5)
print(f"full torch.mm pipeline: events={t3:.2f}us profiler={p3:.2f}us")
PYEOF
