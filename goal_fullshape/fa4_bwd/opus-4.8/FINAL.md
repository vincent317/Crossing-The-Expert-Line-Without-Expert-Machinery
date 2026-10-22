# FlashAttention Backward (causal, bf16) — from-scratch Triton kernel on B200

**Target definitions:** `attention_bwd_causal_h16_d128` (BSHD, MHA+GQA) and
`attention_bwd_varlen_causal_h16_d128` (THD + cu_seqlens).

**Rules honored:** kernels are written from scratch in Triton; no library attention
implementation (flash-attn / cuDNN / torch SDPA / flashinfer / torch.matmul) is called
in the solution. Library baselines are only *run* for reference timing.

**Device-time methodology:** `flashinfer.testing.bench_gpu_time(fn, enable_cupti=True,
cold_l2_cache=True)` → CUPTI kernel span per call, L2 flushed cold before each call,
**median** across iterations. Ref and mine measured alternately on the same B200.

**Correctness:** every workload validated against the definition's fp32 naive reference
(recomputed independently). Reported error is max abs over {dq,dk,dv}. The lower-bound
tolerance = (fastest-library-ref vs fp32) error; our dev tolerance uses ×7. The official
FIB eval uses atol=0.14/rtol=0.01 (causal) and atol=0.11/rtol=0.01 (varlen).

## Results (B200, CUPTI median device time)

| workload | ref (fastest lib) | ref µs | my µs | ref/mine (my speed vs ref) | my err vs fp32 | tol (ref-vs-fp32 ×7) | eval atol |
|---|---|---:|---:|---:|---:|---:|---:|
| MHA  B8 N4096 Hkv16 | cuDNN SDPA (CUDNN_ATTENTION) | 1325 | 2675 | 0.495× | 3.13e-2 | 0.219 | 0.14 |
| GQA  B8 N4096 Hkv2  | cuDNN SDPA (CUDNN_ATTENTION) | 1372 | 2690 | 0.510× | 6.25e-2 | 0.438 | 0.14 |
| varlen T32768 (13 seq, maxlen 8192) | flash-attn-4 b19 (flash_attn.cute) | 1612 | 3380 | 0.477× | 3.13e-2 | 0.219 | 0.11 |

**Geometric-mean speedup vs ref ≈ 0.49×** (0.49–0.50 across runs; the spread is run-to-run noise in the *ref* measurements — cuDNN 1325–1395 µs, FA4 varlen 1571–1612 µs — my kernel's times are stable at ~2675 / ~2690 / ~3380–3540 µs). cuDNN is the faster library on the two dense workloads (so it is the ref there); FA4 is the ref for varlen. Using FA4 as the ref throughout gives geomean ≈ 0.50×.

All three workloads are correct: my max-abs error vs the independent fp32 naive reference is far below both the ×7 lower-bound tolerance and the official eval atol. The ref libraries themselves show the same bf16 rounding error vs fp32 (3.13e-2 / 6.25e-2 / 3.13e-2), so my kernel matches library-grade accuracy.


## Official FIB eval (isolated runner)
Official `flashinfer-bench run` (isolated runner, --use-isolated-runner), both definitions, all workloads **PASSED**:

```
attention_bwd_causal_h16_d128 (atol 0.14, rtol 0.01):
  scratch_triton_bwd_bshd  workload 5d2f3c61 (MHA): PASSED  (33.48x vs math ref)
  scratch_triton_bwd_bshd  workload 8c41e9b2 (GQA): PASSED  (32.28x vs math ref)
attention_bwd_varlen_causal_h16_d128 (atol 0.11, rtol 0.01):
  scratch_triton_bwd_varlen workload e3a7b5d9 (varlen): PASSED (28.15x vs math ref)
```
(The baseline `flash_attn4_wrapper_b19_*` reports 65.1x / 60.3x / 66.2x vs the same math ref; ratio of the two ≈ my speedup vs FA4.)

Solution packaged at `/root/fib-dataset/solutions/mine/attention_bwd/*/scratch_triton_bwd_*.json` (single-file pure-Triton `main.py`).


## Design summary
- Three kernels: `preprocess` (delta = rowsum(dO∘O)), `dkdv` (parallel over kv-block ×
  B × query-head; recompute P via exp2; accumulate dK,dV; per-query-head output, GQA
  sums groups afterwards), `dq` (parallel over q-block × B × query-head; recompute P;
  accumulate dQ). No atomics.
- exp2-based softmax (fold log2e into scale/lse); masked diagonal + unmasked body split
  for causal; compile-time `BOUND` flag removes boundary masking when seqlen divides the
  block (dense path). Varlen uses a per-call compact block schedule built from cu_seqlens.

## Why ~0.48× (engineering analysis, measured on B200 with ncu)
- My two kernels individually reach **dkdv 44.8%** and **dq 38.8% SM (compute) throughput** — dkdv already matches the ~45% that cuDNN/FA4 achieve overall. Both are latency-bound (top stall: CTA-barrier waits in the Triton MMA/smem pipeline).
- The structural gap is **recompute**: an atomic-free flash backward must recompute S=QKᵀ and dP=dO·Vᵀ in *both* the dk/dv pass and the dq pass (the N×N attention matrix is too large to store). That is ~7 matmul-passes vs the ~5 of a fused library kernel → a ~1.4× floor even at equal efficiency (≈0.7× ceiling), and dq's lower efficiency brings it to ~0.48×.
- A fused single kernel (S once) needs **atomic dQ** accumulation; measured on B200 via Triton it was **13.8 ms** (16–64-way atomic contention per element) — catastrophic, so the two-kernel recompute design is strictly better here.
- Things tried that did **not** help on this problem: larger tiles (Blackwell tmem/smem limits), more pipeline stages, stream-concurrent dkdv+dq (kernels don't co-reside: smem too large), pre-transposed V loads (Triton's `tl.trans` was already free), and TMA descriptor loads (loads are not the bottleneck).
- Beating cuDNN/FA4 would require warp-specialized producer/consumer scheduling with an in-SMEM dQ reduction (what those CUDA kernels do), which Triton does not expose.

## Files
- `solution/fa_bwd.py` — the Triton kernels (preprocess / dkdv / dq, dense + varlen).
- `solution/main_causal.py`, `solution/main_varlen.py` — FIB entry wrappers.
- `solution/build_fib.py` — emits the single-file FIB solution JSONs.
- `harness/` — measurement + correctness drivers.
