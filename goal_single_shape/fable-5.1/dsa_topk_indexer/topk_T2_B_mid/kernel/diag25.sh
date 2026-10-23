#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm

q_fp8,k_cache,weights,seq_lens,block_table=make_inputs("cuda")
B,H,D=8,64,128; PAGE_SZ=64
num_sms=deep_gemm.get_num_sms()

# layout.hpp:40 says t.dim()==N fails. 
# The function probably expects 4D tensors: (B, num_heads, seq, head_dim) or similar
# For attention logits, standard shape is:
# q: (B, num_heads_q, seq_q, head_dim)
# k: (B, num_heads_k, seq_k, head_dim)
# logits: (B, num_heads_q, seq_q, seq_k)
#
# Since we use MQA (multi-query attention), num_heads_k=1 typically.
# But we have 64 heads with equal q and k.
#
# Try q: (B, H, 1, D) -- (batch, heads, query_len=1, head_dim)
# K_cat: (B, 1, seq_k, D) -- (batch, heads=1, seq, head_dim)

all_sl=[seq_lens[b].item() for b in range(B)]
max_sl=max(all_sl)
k_raw=k_cache.view(-1,132)

# Build padded K: (B, max_sl, D) fp8
K_pad=torch.zeros(B,max_sl,D,device="cuda",dtype=torch.uint8)
K_scale_pad=torch.zeros(B,max_sl,device="cuda",dtype=torch.float32)
for b in range(B):
    sl=all_sl[b]
    t=torch.arange(sl,device="cuda")
    pg=block_table[b,t//PAGE_SZ]; off=t%PAGE_SZ
    flat=pg*PAGE_SZ+off
    K_pad[b,:sl]=k_raw[flat,:D].contiguous().view(torch.uint8)
    K_scale_pad[b,:sl]=k_raw[flat,D:D+4].contiguous().view(torch.float32).view(sl)
K_pad_fp8=K_pad.view(torch.float8_e4m3fn)

# Try q: (B, H, 1, D), K: (B, 1, max_sl, D)
q_4d=q_fp8.unsqueeze(2)  # (B, H, 1, D)
K_4d=K_pad_fp8.unsqueeze(1)  # (B, 1, max_sl, D)
print(f"q_4d: {q_4d.shape}, K_4d: {K_4d.shape}")

try:
    result=deep_gemm.fp8_mqa_logits(
        q_4d,
        (K_4d, K_scale_pad.unsqueeze(1).unsqueeze(2)),  # scale: (B,1,max_sl)
        weights,
        torch.zeros(B,dtype=torch.int32,device="cuda"),
        seq_lens.int())
    print(f"mqa_logits 4D SUCCESS: {result.shape}")
except Exception as e:
    print(f"mqa_logits 4D error: {e}")

# More experiments: check what dimension is expected
# The error is about tensor dim check. Let's see if q alone needs to be 4D
try:
    result=deep_gemm.fp8_mqa_logits(
        q_4d, (K_pad_fp8, K_scale_pad),
        weights,
        torch.zeros(B,dtype=torch.int32,device="cuda"),
        seq_lens.int())
    print(f"mqa_logits (K 3D) SUCCESS: {result.shape}")
except Exception as e:
    print(f"mqa_logits (K 3D) error: {e}")
PYEOF
