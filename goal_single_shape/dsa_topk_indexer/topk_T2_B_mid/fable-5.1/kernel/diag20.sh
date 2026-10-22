#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys, os, inspect
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm
import torch.profiler as P

# fp8_paged_mqa_logits: q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len
# This sounds EXACTLY like our operator! Let's understand the signature:
# - q: query tensor (fp8)
# - kv_cache: paged KV cache (fp8)
# - weights: per-head weights
# - context_lens: sequence lengths
# - block_table: page indices
# - schedule_meta: scheduling metadata
# - max_context_len: maximum sequence length
# Returns: logits (dot products summed with weights?)
# This might return just the dot products (logits), not our weighted sum + relu

# Get the C++ extension source or help
try:
    import deep_gemm._C as _C
    print("_C module dir:", [x for x in dir(_C) if not x.startswith('_')][:20])
except:
    pass

# Check get_paged_mqa_logits_metadata
src=inspect.getsource(deep_gemm.get_paged_mqa_logits_metadata)
print("get_paged_mqa_logits_metadata:\n", src[:2000])
PYEOF
