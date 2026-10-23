#!/bin/bash
set -e
cd "$WORK_DIR"
pip install safetensors -q 2>&1 | grep -v "already" || true
python -u - << 'PYEOF'
import torch, sys, os, time
sys.path.insert(0, ".")
from bench_all import make_inputs

q_fp8, k_cache, weights, seq_lens, block_table = make_inputs("cuda")
from topk_triton import run_triton_score, run

# Warmup compile
_ = run_triton_score(q_fp8, k_cache, weights, seq_lens, block_table)
torch.cuda.synchronize()
print("Triton compiled OK")

buf = torch.empty(512*1024*1024//4, dtype=torch.float32, device="cuda")
def bench(fn, w=5, n=30):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        buf.fill_(0.); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return ts[len(ts)//2]

t = bench(lambda: run_triton_score(q_fp8, k_cache, weights, seq_lens, block_table))
print(f"triton score: {t:.2f} us")
t2 = bench(lambda: run(q_fp8, k_cache, weights, seq_lens, block_table))
print(f"triton full (score+topk): {t2:.2f} us")
PYEOF
