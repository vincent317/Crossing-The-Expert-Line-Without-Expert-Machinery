#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys, os, tempfile
sys.path.insert(0,".")
from bench_all import make_inputs
import torch.profiler as P
import deep_gemm

# deep_gemm has fp8_gemm_nt: A(M,K) fp8, B(N,K) fp8 -> C(M,N) bf16
# Also: fp8_paged_mqa_logits -- this might be EXACTLY what we need!
# "paged_mqa_logits" = query attention logits over paged KV cache
# Let's inspect fp8_paged_mqa_logits

import inspect
try:
    src=inspect.getsource(deep_gemm.fp8_paged_mqa_logits)
    print("fp8_paged_mqa_logits source:\n", src[:2000])
except:
    print("fp8_paged_mqa_logits: ", deep_gemm.fp8_paged_mqa_logits)
    import importlib.util
    spec=importlib.util.find_spec("deep_gemm")
    print("deep_gemm location:", spec.origin if spec else "not found")

# Also check fp8_mqa_logits (non-paged version)
try:
    src2=inspect.getsource(deep_gemm.fp8_mqa_logits)
    print("\nfp8_mqa_logits source:\n", src2[:2000])
except Exception as e:
    print("fp8_mqa_logits inspect error:", e)
PYEOF
