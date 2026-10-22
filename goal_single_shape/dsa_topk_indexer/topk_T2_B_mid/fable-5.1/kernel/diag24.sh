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

# Try fp8_mqa_logits (non-paged version) to understand the interface
# fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=False, max_seqlen_k=0)
# ks, ke = start, end indices of k sequences
# Let's try it with our data
# q: (B, H, D) fp8
# kv: (total_tokens, 2, H, D) or similar
# weights: (B, H)
# ks: (B,) start indices
# ke: (B,) end indices

# Build contiguous kv tensor from all tokens
all_sl=[seq_lens[b].item() for b in range(B)]
offsets=[0]+list(torch.cumsum(torch.tensor(all_sl),0).tolist())
total=offsets[-1]

k_raw=k_cache.view(-1,132)
all_k_fp8=[]
for b in range(B):
    sl=all_sl[b]
    t=torch.arange(sl,device="cuda")
    pg=block_table[b,t//PAGE_SZ]; off=t%PAGE_SZ
    flat=pg*PAGE_SZ+off
    kd=k_raw[flat,:D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)
    all_k_fp8.append(kd)

K_cat=torch.cat(all_k_fp8,dim=0)  # (total_tokens, 128)
print(f"K_cat shape: {K_cat.shape}")

# fp8_mqa_logits expects kv with both K and V -- try (total, H, D) or (total, D)
# Let's try different formats
ks=torch.tensor(offsets[:-1],device="cuda",dtype=torch.int32)
ke=torch.tensor(offsets[1:],device="cuda",dtype=torch.int32)

# try (total, D) -- MQA means single head K and V
# Or try with scale tensor as tuple
K_scales=[]
for b in range(B):
    sl=all_sl[b]
    t=torch.arange(sl,device="cuda")
    pg=block_table[b,t//PAGE_SZ]; off=t%PAGE_SZ
    flat=pg*PAGE_SZ+off
    ks_b=k_raw[flat,D:D+4].contiguous().view(torch.float32).view(sl)
    K_scales.append(ks_b)
K_scales_cat=torch.cat(K_scales)  # (total,)
print(f"K_scales_cat shape: {K_scales_cat.shape}")

# Per-token scaling: convert to per-tensor or pack into scale tensor
# deep_gemm fp8 expects (data, scale_factor) tuples typically
# Try passing (K_cat, K_scales_cat.unsqueeze(1)) as the kv tuple

try:
    result=deep_gemm.fp8_mqa_logits(
        q_fp8,  # (B,H,D) fp8
        (K_cat, K_scales_cat.unsqueeze(1)),  # kv as tuple with scale
        weights,  # (B,H)
        ks, ke,
        clean_logits=True)
    print(f"fp8_mqa_logits SUCCESS: {result.shape}, {result.dtype}")
    print(f"result b=2 sample: {result[2,:5]}")
except Exception as e:
    print(f"fp8_mqa_logits error: {e}")

# Try without tuple
try:
    result=deep_gemm.fp8_mqa_logits(q_fp8, K_cat, weights, ks, ke)
    print(f"fp8_mqa_logits (no tuple): {result.shape}")
except Exception as e:
    print(f"fp8_mqa_logits (no tuple) error: {e}")
PYEOF
