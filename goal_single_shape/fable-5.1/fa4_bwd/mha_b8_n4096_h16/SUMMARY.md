# FlashAttention causal MHA backward on B200 — result and bottleneck

Problem: `b=2, s=16384, h=16, d=128`, bf16, causal, `softmax_scale = 1/sqrt(128)`,
backward only. Reference: `flash_attn.cute.interface.flash_attn_func` backward
(flash-attn-4 4.0.0b19, image `model-align/sglang:v0.5.19`).

## Outcome against the completion conditions

| # | Condition | Status |
|---|-----------|--------|
| 1 | ref device time measured on B200 | **met** — 4.3552 ms kernel-sum, 4.4326 ms span, 3 kernels (CUDA-events median 4.7014 ms) |
| 2 | fp32 naive + `E_ref`, `tol = 7 E_ref` | **met** — `E_REF = 1.526976e-02`, `TOL = 1.068883e-01` |
| 3 | new kernel error ≤ tol | **met** — `E_MINE = 4.553023e-02 ≤ 1.068883e-01` PASS |
| 4 | device time ≤ ref/1.2 | **NOT met** — 8.1864 ms vs target 3.6293 ms, **0.5320x** |
| 5 | sources committed | **met** — commits `98c4046`, `6156a2a`, `b4ff0a9`, `1fb0f0d`, `1e3f67b` |

Best full backward: **8.1864 ms** (671.6 TFLOP/s) vs reference **4.3552 ms**
(1262.3 TFLOP/s). Accuracy passes with 2.3x margin; speed misses by 2.26x.

Composition of the 8.1864 ms: `_gluon_dkdv` 4.658 ms + `_split_dq` 3.440 ms +
`_bwd_preprocess` 0.088 ms.

## What was built

Five formulations, all numerically correct, measured on B200 (`bench.py`, `final_bench.py`):

| Formulation | Device time | Notes |
|---|---|---|
| Triton fused, 5 GEMMs, atomic dQ | 10.08 ms | `kernel.py` `fa_backward` |
| Triton split, 7 GEMMs, zero atomics | 10.24 ms | `kernel.py` `fa_backward_split` |
| Gluon warp-specialized dK/dV, un-pipelined | 7.86 ms | dK/dV only |
| Gluon warp-specialized dK/dV, software-pipelined | 4.55 ms | dK/dV only |
| **Gluon dK/dV + Triton dQ (best full bwd)** | **8.19 ms** | `final_bench.py` |

All use the transposed orientation `S^T = K Q^T`, which puts the K/V block on the MMA row
axis so `P^T` and `dS^T` emerge from the accumulator already in the layout the next MMA
consumes as its A operand. No accumulator is ever transposed.

## Bottleneck: measured, not inferred

**1. The dQ cross-program reduce-add costs ~4.6 ms.** Ablation on the fused kernel
(`kernel.py`, `ABL` switches):

| variant | time | delta |
|---|---|---|
| full | 10.35 ms | — |
| no dQ at all (4 GEMMs) | 4.95 ms | −5.40 ms |
| dQ dot + plain store | 9.58 ms | −0.77 ms |
| no exp2 | 8.76 ms | −1.59 ms |
| no elementwise | 6.47 ms | −3.88 ms |

The dQ *dot* is only 0.77 ms; the remaining ~4.6 ms is fp32 atomic traffic — 4.3 G atomic
elements at ~0.9 T/s. This is an atomic **rate** limit, not bandwidth (17 GB in 4.6 ms is
3.7 TB/s, well under HBM). The volume is `S²·D·4·BH/(2·BN)`, independent of `BLOCK_M`, so
no `BLOCK_M` tuning touches it. Replacing it with in-TMEM dQ accumulation removes the
atomics entirely but costs 2 extra GEMMs (7 instead of 5) — both land at ~10.2 ms.

**2. Softmax elementwise work does not overlap the MMA pipeline in plain Triton** (3.88 ms,
from the ablation above). Every `warp_specialize=True` configuration fails to compile:
`TritonGPUHoistTMEMAlloc` asserts, or TMEM overflows (576/704/768 > 512 columns). Gluon
fixes this — see below.

**3. The decisive limit: the bf16 tensor-core floor is above the reference.**
`ceiling.py` issues the real kernel's MMA shapes and causal loop with **no softmax and no
stores** — a pure tensor-core measurement:

| | 1 GEMM | 2 | 3 | 4 |
|---|---|---|---|---|
| bf16 | 3.073 ms | 3.821 | 4.336 | **4.820 ms** (1.205 ms/GEMM) |
| fp8 | 2.404 ms | 2.789 | 3.063 | **3.320 ms** (0.830 ms/GEMM) |

**4 bf16 GEMMs with zero softmax and zero stores already cost 4.820 ms, above the
reference's 4.3552 ms for 5 GEMMs plus softmax.** The reference runs at ~0.84 ms per
GEMM-pass — essentially the bf16 tcgen05 roofline with softmax fully hidden. So no bf16
formulation, however perfectly pipelined, can reach `ref/1.2 = 3.63 ms`. This closes the
bf16 route by measurement rather than by argument.

**4. fp8 is fast enough but not accurate enough.** fp8 e4m3 is ~1.45x faster per GEMM
(0.830 vs 1.205 ms), which would put 5 GEMMs near 3.3 ms — inside the target. Per-GEMM
error attribution (`fp8_mixed.py`, tol = 1.069e-1):

| fp8 applied to | max err | vs tol |
|---|---|---|
| nothing (all bf16) | 1.03e-2 | 10% |
| Q/K/V/dO | 2.56e-1 | 239% FAIL |
| P (dV only) | 1.28e-1 | 120% FAIL |
| dS per-tile (dQ, dK) | 1.32e-1 | 123% FAIL |
| dS per-row, dQ only | 1.04e-1 | **97% PASS** |
| everything | 2.81e-1 | 263% FAIL |

Only one variant passes, at 97% of tolerance with no margin, and it converts just 1 of 5
GEMMs — far too little to matter. `P ∈ [0,1]`, the most favorable possible operand, still
fails at 120% across every scale tried (1/448, 1/128, 1/32), so the loss is the e4m3
3-bit mantissa, not scale selection.

Scaling granularity here is **forced by the contraction**, not a free choice:
`dV[n,d] = Σ_m P^T[n,m] dO[m,d]` and `dK[n,d] = Σ_m dS^T[n,m] Q[m,d]` both sum over `m`,
so any scale varying along `m` cannot be factored out of the MMA — `P^T` and `dS^T` admit
only a per-tile scalar. `dQ[m,d] = Σ_n dS[m,n] K[n,d]` sums over `n`, so a per-row-`m`
scale on `dS` *is* legal there. That asymmetry is exactly why `dsrow8` is the only passing
variant.

## What worked: Gluon

Plain Triton cannot express the needed overlap. Gluon (`triton.experimental.gluon`) can,
and `gluon_kernel.py` proves out the whole machinery on B200:

* `tcgen05_mma` + `tcgen05_commit` + `mbarrier` verified against fp32 (err 1.5e-5)
* `warp_specialize` driving tcgen05 from a worker partition while the default partition
  consumes the accumulator — verified numerically
* the dK/dV pass: TMA-loaded Q/dO ring, MMAs in a worker partition, softmax in the
  default partition, dK/dV resident in TMEM for the whole causal range, zero atomics

Two findings from making it fast:

* **Software pipelining is worth 1.7x.** Issuing block `i+1`'s `S^T`/`dP^T` before
  blocking on block `i`'s softmax took dK/dV from 7.86 → 4.55 ms. Without it the two
  partitions strictly ping-pong and the MMA pipe idles through every softmax.
* **Two `warp_specialize` regions desynchronize mbarrier phases.** Splitting the masked
  diagonal range from the unmasked interior range hangs the kernel: each region restarts
  its iteration counter at 0 while the barriers retain the phase the previous region left.
  Merging into one region with the mask as a runtime predicate fixes it.

## Conclusion

The target is unreachable on this hardware with these two precisions. The reference
already runs at the bf16 tcgen05 roofline, so beating it by 1.2x requires either fewer
FLOPs than the 5-GEMM backward (none available) or a faster precision — and fp8 e4m3, the
only faster option, exceeds the error tolerance on every split that would yield a
meaningful speedup.

The remaining untested idea is reducing dQ reduce-add traffic so the 5-GEMM fused form
becomes viable: reduce in bf16 (halves the bytes) or give each program a strip of several
K blocks so `dQ^T` accumulates across the strip before one flush (divides atomic count by
the strip length). Even at zero reduce-add cost, the bf16 ceiling of ~1.205 ms/GEMM puts
5 GEMMs at ~6.0 ms — still short of 3.63 ms. This is why the bf16 line of attack was
stopped rather than tuned further.

## Files

| file | role |
|---|---|
| `common.py` | problem definition, deterministic inputs, CUDA-events + chrome-trace kernel-only timing |
| `naive.py` | fp32 chunked non-flash backward — accuracy ground truth |
| `kernel.py` | Triton fused (atomic dQ) and split (zero-atomic) backward, with ablation switches |
| `gluon_kernel.py` | Gluon warp-specialized tcgen05 dK/dV pass + tcgen05/warp-spec smoke tests |
| `ceiling.py`, `probe_ceiling.py` | MMA-only tensor-core floor, bf16 and fp8 |
| `fp8_feasible.py`, `fp8_mixed.py` | fp8 e4m3 accuracy and per-GEMM error attribution |
| `bench.py` | ref vs Triton variants, config sweeps |
| `final_bench.py` | final head-to-head, same data and methodology for both |
| `ref_bench.py` | standalone reference measurement |

Reproduce the headline numbers with:

```
python3 /Users/cyj/cyj/Goal-test/goal-gpu-run.py --timeout 30 -- "python -u final_bench.py"
```
