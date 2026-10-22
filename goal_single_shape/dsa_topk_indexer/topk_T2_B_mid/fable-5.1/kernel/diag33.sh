#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys
sys.path.insert(0,".")
from bench_all import make_inputs
import deep_gemm, torch.profiler as P

# deep_gemm kernel device_time=0.005 us per call (5 iterations → 5×0.005=0.025 us total)
# prof_span = sum / 5 = 0.005 us? But we reported 4.53-5.15 us profiler before!
# WAIT: device_time is in microseconds. 0.005 us per kernel × 5 calls = 0.025 us total / 5 = 0.005 us.
# But the profiler reported 4.53 us earlier. These don't match!
# 
# The issue: the profiler function_events .device_time may be in NANOSECONDS (not us).
# Let me recheck the units.

q_fp8, k_cache, weights, seq_lens, block_table = make_inputs("cuda")
B, H, D = 8, 64, 128; PAGE_SZ = 64
all_sl = [seq_lens[b].item() for b in range(B)]
total = sum(all_sl)
k_raw = k_cache.view(-1, 132)
all_flat = []
for b in range(B):
    sl = all_sl[b]
    t = torch.arange(sl, device="cuda")
    pg = block_table[b, t//PAGE_SZ]; off = t % PAGE_SZ
    all_flat.append(pg*PAGE_SZ+off)
all_flat_t = torch.cat(all_flat)
K_fp8 = k_raw[all_flat_t,:D].contiguous().view(torch.uint8).view(torch.float8_e4m3fn)
K_bf16 = K_fp8.float().bfloat16()
q_bf16 = q_fp8.reshape(B*H,D).float().bfloat16()
M_a=((B*H+127)//128)*128; N_a=((total+127)//128)*128
K_pad = torch.zeros(N_a,D,device="cuda",dtype=torch.bfloat16); K_pad[:total]=K_bf16
q_pad = torch.zeros(M_a,D,device="cuda",dtype=torch.bfloat16); q_pad[:B*H]=q_bf16
d_out = torch.zeros(N_a,M_a,device="cuda",dtype=torch.bfloat16)

buf = torch.empty(512*1024*1024//4,dtype=torch.float32,device="cuda")
for _ in range(5): deep_gemm.bf16_gemm_nt(K_pad,q_pad,d_out)
torch.cuda.synchronize()
buf.fill_(0.); torch.cuda.synchronize()
with P.profile(activities=[P.ProfilerActivity.CUDA],record_shapes=True) as prof:
    for _ in range(5):
        buf.fill_(0.)
        deep_gemm.bf16_gemm_nt(K_pad,q_pad,d_out)
torch.cuda.synchronize()

evts = prof.profiler.function_events
for e in evts:
    if e.device_time > 0 and "fill" not in e.name.lower():
        print(f"  {e.name[:60]}: device_time={e.device_time}")
        print(f"    (if us: {e.device_time:.3f} us per event)")
        print(f"    note: device_time unit is microseconds in PyTorch profiler")
        break

# Also check via torch.cuda.Event
s=torch.cuda.Event(enable_timing=True); e2=torch.cuda.Event(enable_timing=True)
buf.fill_(0.); torch.cuda.synchronize()
s.record()
for _ in range(5):
    deep_gemm.bf16_gemm_nt(K_pad,q_pad,d_out)
e2.record()
torch.cuda.synchronize()
t_ms = s.elapsed_time(e2)
print(f"\nCUDA events for 5 calls: {t_ms:.3f} ms = {t_ms*1000/5:.2f} us per call")
PYEOF
