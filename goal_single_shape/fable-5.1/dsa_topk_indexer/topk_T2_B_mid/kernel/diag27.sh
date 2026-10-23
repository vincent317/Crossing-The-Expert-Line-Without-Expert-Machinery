#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys, inspect
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128; PAGE_SZ=64

# Correct API for fp8_gemm_nt
src=inspect.getsource(deep_gemm.fp8_gemm_nt)
print("fp8_gemm_nt:\n",src)
src2=inspect.getsource(deep_gemm.per_token_cast_to_fp8)
print("per_token_cast_to_fp8:\n",src2)
PYEOF
