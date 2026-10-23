#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm
import torch.profiler as P

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128; PAGE_SZ=64
num_sms=deep_gemm.get_num_sms()

# Try deep_gemm.fp8_gemm_nt directly:
# A(M,K) fp8_e4m3fn, B(N,K) fp8_e4m3fn -> C(M,N) bf16
# with scales. Let's figure out exact API.
import inspect
src=inspect.getsource(deep_gemm.fp8_gemm_nt)
print("fp8_gemm_nt source:", src[:500])

# fp8_gemm_nt: (a, b, d, recipe=None, compiled_dims=None, disable_ue8m0_cast=False)
# where a = (data, scale) or just data, similar for b
# d = output tensor (M,N) bf16

all_sl=[seq_lens[b].item() for b in range(B)]
total=sum(all_sl)
k_raw=k_cache.view(-1,132)

# Gather all K tokens
all_flat=[]
for b in range(B):
    sl=all_sl[b]
    t=torch.arange(sl,device="cuda")
    pg=block_table[b,t//PAGE_SZ]; off=t%PAGE_SZ
    all_flat.append(pg*PAGE_SZ+off)
all_flat_t=torch.cat(all_flat)  # (total,)
K_all_fp8=k_raw[all_flat_t,:D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)  # (total,D)
Ks_all=k_raw[all_flat_t,D:D+4].contiguous().view(torch.float32).view(total)  # (total,)

q_all=q_fp8.reshape(B*H,D)  # (512,128)
M,K_=q_all.shape; N_=total

# Align M and N to 128 for deep_gemm
M_a=((M+127)//128)*128; N_a=((N_+127)//128)*128
q_pad=torch.zeros(M_a,K_,device="cuda",dtype=torch.float8_e4m3fn)
q_pad[:M]=q_all
K_pad=torch.zeros(N_a,K_,device="cuda",dtype=torch.float8_e4m3fn)
K_pad[:N_]=K_all_fp8
d_out=torch.zeros(M_a,N_a,device="cuda",dtype=torch.bfloat16)

# Scales: deep_gemm uses per-token scaling (128 elements)
# Per-token scale for q_all: q doesn't have a scale (it's directly fp8)
# Per-token scale for K: Ks_all
# deep_gemm expects UE8M0 or float32 scales? Let's check per_token_cast_to_fp8
q_sf=torch.ones(M_a,1,device="cuda",dtype=torch.float32)  # per-row scale for q
# K_sf: per-token scale = Ks_all
K_sf=torch.ones(N_a,1,device="cuda",dtype=torch.float32)
K_sf[:N_,0]=Ks_all  # apply actual k scales

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
    deep_gemm.fp8_gemm_nt((q_pad,q_sf),(K_pad,K_sf),d_out)
    torch.cuda.synchronize()
    print(f"fp8_gemm_nt OK, out shape: {d_out.shape}")
    t=bench(lambda: deep_gemm.fp8_gemm_nt((q_pad,q_sf),(K_pad,K_sf),d_out))
    p=prof_span(lambda: deep_gemm.fp8_gemm_nt((q_pad,q_sf),(K_pad,K_sf),d_out))
    print(f"deep_gemm fp8_gemm_nt (512,128)x(4096,128): events={t:.2f}us profiler={p:.2f}us")
    
    # Full pipeline: deep_gemm + per-batch score + topk
    offsets=[0]+list(torch.cumsum(torch.tensor(all_sl),0).tolist())
    def full_deep_gemm():
        deep_gemm.fp8_gemm_nt((q_pad,q_sf),(K_pad,K_sf),d_out)
        # d_out: (512, N_a) bf16, valid at d_out[b*H:(b+1)*H, offset_b:offset_b+sl_b]
        out=torch.full((B,2048),-1,dtype=torch.int32,device="cuda")
        for b in range(B):
            sl=all_sl[b]; off=offsets[b]
            dots=d_out[b*H:(b+1)*H, off:off+sl].float()  # (H,sl)
            scores=(weights[b].unsqueeze(1)*torch.relu(dots)).sum(0)
            k_out=min(2048,sl)
            gids=all_flat_t[off:off+sl].int()
            if sl<=2048:
                out[b,:sl]=gids
            else:
                idx=torch.topk(scores,k_out).indices
                out[b,:k_out]=gids[idx]
        return out
    t=bench(full_deep_gemm,w=3,n=20)
    p=prof_span(full_deep_gemm,n=5)
    print(f"full_deep_gemm pipeline: events={t:.2f}us profiler={p:.2f}us")
except Exception as e:
    import traceback; traceback.print_exc()
PYEOF
