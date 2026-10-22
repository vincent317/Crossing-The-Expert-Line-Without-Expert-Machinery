#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys, glob, os
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128; PAGE_SZ=64
num_sms=deep_gemm.get_num_sms()

# Look for the actual source of fp8_paged_mqa_logits in /deepgemm
cpp_src = []
for root,dirs,files in os.walk('/deepgemm'):
    for f in files:
        if f.endswith(('.hpp','.cpp','.h','.cu')):
            cpp_src.append(os.path.join(root,f))

print("C++ files:", len(cpp_src))
for f in cpp_src:
    try:
        txt=open(f).read()
        if 'paged_mqa_logits' in txt:
            print(f"\n=== {f} ===")
            idx=txt.find('paged_mqa_logits')
            print(txt[max(0,idx-50):idx+600])
    except: pass

# Also try the deepgemm package tests
for root,dirs,files in os.walk('/opt/sglang/lib/python3.12/site-packages/deep_gemm'):
    for f in files:
        if 'test' in f.lower() and f.endswith('.py'):
            path=os.path.join(root,f)
            txt=open(path).read()
            if 'paged' in txt:
                print(f"\n=== {path} ===")
                print(txt[:3000])
PYEOF
