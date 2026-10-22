#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys, glob, os
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm

# Find testing examples
for f in glob.glob('/opt/sglang/lib/python3.12/site-packages/deep_gemm/**', recursive=True):
    if 'test' in f.lower() and f.endswith('.py'):
        try:
            txt=open(f).read()
            if 'paged_mqa' in txt:
                print(f"==> {f}")
                # find all occurrences
                idx=0
                while True:
                    idx=txt.find('paged_mqa',idx)
                    if idx<0: break
                    print(txt[max(0,idx-100):idx+400])
                    print("---")
                    idx+=1
        except: pass

# Check deepgemm C source
for f in glob.glob('/deepgemm/**/*.hpp', recursive=True):
    try:
        txt=open(f).read()
        if 'fp8_paged_mqa_logits' in txt:
            idx=txt.find('fp8_paged_mqa_logits')
            print(f"==> {f}")
            print(txt[max(0,idx-100):idx+600])
    except: pass
PYEOF
