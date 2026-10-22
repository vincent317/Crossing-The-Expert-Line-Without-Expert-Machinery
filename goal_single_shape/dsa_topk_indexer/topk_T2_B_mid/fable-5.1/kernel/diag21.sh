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

# context_lens must be 2D. Try (B, 1) or (1, B)?
# block_kv is PAGE_SZ=64
try:
    meta=deep_gemm.get_paged_mqa_logits_metadata(
        seq_lens.int().unsqueeze(1),  # (B,1)
        PAGE_SZ, num_sms)
    print("meta shape:", meta.shape if hasattr(meta,'shape') else type(meta))
    print("meta:", meta)
except Exception as e:
    print("(B,1) failed:", e)

try:
    meta=deep_gemm.get_paged_mqa_logits_metadata(
        seq_lens.int().unsqueeze(0),  # (1,B)
        PAGE_SZ, num_sms)
    print("(1,B) meta:", meta)
except Exception as e:
    print("(1,B) failed:", e)

# Try with block_table (B, max_pages) as block_kv
try:
    meta=deep_gemm.get_paged_mqa_logits_metadata(
        seq_lens.int().unsqueeze(1),  # (B,1)
        block_table.int(), num_sms)
    print("with block_table meta:", meta)
except Exception as e:
    print("with block_table failed:", e)

# Check attention.hpp lines
import os, glob
csrc=[f for f in glob.glob('/opt/sglang/lib/**/*.hpp',recursive=True) if 'attention' in f]
for f in csrc:
    try:
        txt=open(f).read()
        if 'context_lens' in txt:
            lines=txt.split('\n')
            for i,l in enumerate(lines):
                if 'context_lens' in l and i<210:
                    print(f"{f}:{i}: {l}")
    except: pass
PYEOF
