#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, torch.utils.cpp_extension as ext, os, tempfile
import sys; sys.path.insert(0,".")
from bench_all import make_inputs
import torch.profiler as P

# deep_gemm is available! This library provides highly optimized FP8 GEMM for Blackwell.
# It likely uses the WGMMA/CuTe approach that the ref uses.
# Let's try deep_gemm for our fp8 matmul.

try:
    import deep_gemm
    print("deep_gemm available, version:", getattr(deep_gemm,'__version__','unknown'))
    print("deep_gemm dir:", [x for x in dir(deep_gemm) if not x.startswith('_')])
except ImportError as e:
    print("deep_gemm import error:", e)

# Also try tilelang
try:
    import tilelang
    print("tilelang available:", tilelang.__version__ if hasattr(tilelang,'__version__') else 'yes')
except ImportError as e:
    print("tilelang import error:", e)

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128; PAGE_SZ=64
k_raw=k_cache.view(-1,132)
all_sl=[seq_lens[b].item() for b in range(B)]
total_tokens=sum(all_sl)  # 4061

# Try deep_gemm for fp8 gemm
try:
    # deep_gemm API: deep_gemm.gemm(A, B) -> C
    # A: (M, K) fp8_e4m3fn, B: (N, K) fp8_e4m3fn, C: (M, N) bf16
    # Let's try with ALL batches together: q_all (512, 128) @ K_all^T (4061, 128)
    
    # Gather K
    all_flat=[]
    for b in range(B):
        sl=all_sl[b]
        t=torch.arange(sl,device="cuda")
        pg=block_table[b,t//PAGE_SZ]; off=t%PAGE_SZ
        all_flat.append(pg*PAGE_SZ+off)
    all_flat_t=torch.cat(all_flat)
    K_all=k_raw[all_flat_t,:D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)  # (4061,128)
    Ks_all=k_raw[all_flat_t,D:D+4].contiguous().view(torch.float32).view(total_tokens)
    Gids_all=all_flat_t.int()
    
    q_all=q_fp8.reshape(B*H,D)  # (512,128) fp8
    
    # deep_gemm expects specific alignment. Try:
    print("\nTrying deep_gemm.gemm_fp8_fp8_bf16_nt...")
    # Function signature varies by version. Check available functions:
    funcs=[f for f in dir(deep_gemm) if 'gemm' in f.lower()]
    print("deep_gemm gemm functions:", funcs)
    
    if hasattr(deep_gemm, 'gemm_fp8_fp8_bf16_nt'):
        # Likely: A(M,K) fp8, B(N,K) fp8 (N-transposed), scales, out(M,N) bf16
        # Pad to 128-aligned
        M,K=q_all.shape; N=K_all.shape[0]
        M_p=((M+127)//128)*128; N_p=((N+127)//128)*128
        q_pad=torch.zeros(M_p,K,device="cuda",dtype=torch.float8_e4m3fn)
        q_pad[:M]=q_all
        K_pad=torch.zeros(N_p,K,device="cuda",dtype=torch.float8_e4m3fn)
        K_pad[:N]=K_all
        out=torch.zeros(M_p,N_p,device="cuda",dtype=torch.bfloat16)
        # scales: per-token for both A and B
        scale_a=torch.ones(M_p,1,device="cuda",dtype=torch.float32)
        scale_b=torch.ones(N_p,1,device="cuda",dtype=torch.float32)
        deep_gemm.gemm_fp8_fp8_bf16_nt((q_pad,scale_a),(K_pad,scale_b),out)
        torch.cuda.synchronize()
        print("deep_gemm test OK, output shape:", out.shape)
        
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
        
        def deep_mm():
            deep_gemm.gemm_fp8_fp8_bf16_nt((q_pad,scale_a),(K_pad,scale_b),out)
        t=bench(deep_mm)
        print(f"deep_gemm (512,128)@(4096,128): {t:.2f} us CUDA events")
        
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
        
        p=prof_span(deep_mm)
        print(f"deep_gemm profiler span: {p:.2f} us")
    else:
        print("gemm_fp8_fp8_bf16_nt not found, trying other functions...")
        for f in funcs[:5]:
            print(f"  {f}: {getattr(deep_gemm,f)}")
except Exception as e:
    print("deep_gemm error:", e)
PYEOF
