# Kimi Delta Attention FWD — h96_mixed on B200: results and bottleneck

Case: `T=8192` tokens packed into `N=6` sequences (1300/547/2048/963/271/3063),
`H=96`, `D=128`, bf16. Layout `q/k/v/g (1,8192,96,128)`, `beta (1,8192,96)`,
`A_log (96)`, `dt_bias (96,128)`, `state (6,96,128,128)` updated in place,
`cu_seqlens (7) int64` → `out (1,8192,96,128)`.

All numbers measured on one **NVIDIA B200 (148 SMs)** in the `aiforai-b200`
pool via `cctl job create`, torch 2.7.0a0+nv25.04, Triton 3.4.0,
flashinfer-python 0.6.18.post1.

## Status against the completion criteria

| # | Criterion | Result |
|---|-----------|--------|
| 1 | ref device time measured on B200 | **DONE** — 0.3993 ms (CUDA events) / **0.3904 ms** (profiler kernel duration) |
| 2 | fp32 naive + `E_ref`, `tol = 7·E_ref` | **DONE** — `E_ref = 2.040848e-03`, `tol = 1.428594e-02` |
| 3 | new kernel error ≤ tol | **DONE** — `1.5918e-03` = 0.111 × tol |
| 4 | new kernel ≤ ref/1.2 = 0.3327 ms | **NOT MET** — best 1.5917 ms (0.250× ref) |
| 5 | sources committed | **DONE** — see files below |

## 1. The reference

The reference is CAKE's `recurrent_kda`, shipped upstream in **FlashInfer**
(`pip install flashinfer-python`). It is used strictly as a black box
(install → import → call); no reference source was read.

Getting the CAKE B200 backend to dispatch was the hard part. `flashinfer.recurrent_kda`
silently falls back to a slow CuTe DSL path unless the call is *eligible* for the
frozen SM100-family FlashKDA backend. `seq_order` is rejected by every other path,
so passing it works as an eligibility oracle; a 32-combination scan
(`refprobe.py`) found exactly one accepted family:

```
lower_bound = -5.0          # safe gate: lower_bound * sigmoid(exp(A_log)*(g+dt_bias))
beta_is_logit = True        # beta given as pre-sigmoid logits
initial_state : bfloat16, layout [N, H, V, K]   (not [N,H,K,V])
use_gate_in_kernel = True, use_qk_l2norm_in_kernel = True
```

The dispatched kernel is **`kernel_flashkda_bf16_fused_m128`** — the M128 variant
named in the task. Measured:

```
REF device time (CUDA events, median of 5 x 50 iters) = 0.3993 ms
REF kernel duration (profiler, single kernel)          = 0.3904 ms
   390.4 us  kernel_flashkda_bf16_fused_m128
```

which matches the quoted 0.391 ms. Without `seq_order` the dispatcher picks
`kernel_flashkda_bf16_persistent_m128` (0.3546 ms profiler, but with a noisy
host-side sort); the fused M128 number above is the one used as the reference.

## 2. fp32 naive and the tolerance

`kda_common.kda_naive_fp32` is a token-by-token fp32 implementation of

```
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T     (DK x DV)
o_t = S_t^T q_t
alpha_t = exp(lower_bound * sigmoid(exp(A_log_h) * (g_t + dt_bias_h)))
q, k L2-normalised; q scaled by 1/sqrt(128)
```

```
naive fp32 |out|max = 0.307953   rms = 0.046729
E_ref = max|CAKE_ref_out - naive_fp32_out| = 2.040848e-03
E_ref(state)                               = 1.839864e-02
tol   = 7 * E_ref                          = 1.428594e-02
```

## 3. The new kernel — correctness

`kda_triton.py` is written from the recurrence above (chunked/UT-transform form
derived independently; no reference code consulted). One Triton program owns one
`(sequence, head)` pair, keeps the `128x128` state resident, and walks the
sequence in `BC`-token blocks:

```
Lb = cumsum(log alpha)        (inclusive, inside the block)
Ka = k*exp(Lb)                X  = beta*(V - Ka H)
M  = tril(k_i·k_j exp(Li-Lj), -1)        U = (I + diag(beta) M)^-1 X
Mq = tril(q_i·k_j exp(Li-Lj),  0)        O = (q*exp(Lb)/sqrt(D)) H + Mq U
H' = Diag(exp(L_BC)) H + (k*exp(L_BC-Lb))^T U
```

`M`/`Mq` are factored as `exp(Li-r)·exp(r-Lj)` with the per-channel **value
midpoint** `r = L_BC/2`, which minimises `max|Li-r|`. Because the reference's
safe gate bounds `log alpha` in `(-5, 0)`, `BC=32` gives `|Li-r| <= 80`, inside
the fp32/bf16 exponent range with **no clamping error at all**. (A `CAP` clamp is
kept as a guard for the unbounded softplus gate, where its error is provably
`<= 2·exp(-CAP)` per entry.) The `16x16`/`32x32` unit-triangular inverse uses the
nilpotent doubling `(I+A)(I+A^2)(I+A^4)...`.

```
MINE vs fp32 naive: out max_abs_err = 1.591802e-03   (tol 1.428594e-02)
                    state max_abs_err = 1.292e-02
RESULT: PASS  (0.111x of tol)
```

## 4. The new kernel — performance (not met)

Best configuration found (Triton 3.4.0):

```
BC=32, BV=128, num_warps=4, num_stages=2
  1.5917 ms (CUDA events)   1.5451 ms (profiler)
  speedup vs ref = 0.250x   (need >= 1.200x)
```

Selected sweep (same data, same measurement as the reference):

| BC | BV | warps | stages | ms (events) | regs | spills |
|----|----|-------|--------|-------------|------|--------|
| 32 | 128 | 4 | 2 | **1.5917** | 255 | 150 |
| 32 | 128 | 8 | 4 | 1.8875 | 255 | 48 |
| 32 | 128 | 8 | 2 | 1.8956 | 255 | 42 |
| 16 | 128 | 8 | 3 | 2.0668 | 255 | 0 |
| 16 | 128 | 8 | 2 | 2.2478 | 255 | 0 |
| 32 | 64  | 8 | 2 | 3.1755 | 255 | 0 |
| 32 | 128 | 16 | 2 | 30.46 | 64 | 1552 |

## Bottleneck analysis

Roofline for this shape: q/k/v/g are 4 x 201 MB read, out 201 MB written, so the
HBM floor is ~1.05 GB ≈ **160 us** at ~6.5 TB/s. The chunked algorithm at BC=32
needs ~5.5e10 MACs ≈ 110 GFLOP, i.e. ~**50 us** at B200 bf16 peak. The reference
lands at 390 us — about 2.4x the memory floor. The target 333 us is therefore
physically reachable; the 4.8x gap is entirely a code-generation problem.

What the measurements say about where it goes:

1. **Occupancy is pinned at one block per SM.** The resident `128x128` fp32 state
   alone is 64 registers/thread at 256 threads; the full body compiles to 255
   registers with spills. With 576 programs on 148 SMs and a single block
   resident, every shared-memory round trip between the ~15 `tl.dot`s per block
   iteration is fully exposed. `num_warps=4` (two 128-thread blocks per SM, two
   independent instruction streams) is the single biggest win measured
   (1.89 → 1.59 ms), which is the signature of latency exposure rather than
   throughput saturation.
2. **Splitting DV to raise occupancy backfires.** BV=64 doubles the per-token
   non-state work (gate, L2 norm, cumsum, `M`/`Mq`, triangular inverse) and is
   1.6-2x *slower* (3.18 ms at BV=64/nw=8 vs 1.90 ms at BV=128/nw=8); BV=32 is
   4.8 ms. So the kernel is dominated by per-token scalar/elementwise/small-dot
   work, not by the three big state GEMMs.
3. **Ablations**: removing the triangular inverse buys ~8%; replacing the gate
   with a constant buys ~0%. Neither the MUFU traffic (4 transcendentals per
   element) nor the inverse is the primary cost — the cost is the serialized
   chain of small `tl.dot`s with SMEM staging, with no second block to hide it.
4. **Toolchain matters a lot.** Triton 3.2.0 (the image default) spills 626
   registers and needs 11.0 ms for the same kernel, and its compiler segfaults on
   some tile shapes. 3.3.1 gives 2.17 ms, 3.4.0 gives 1.93 ms for the same
   configuration. `num_warps=16` is catastrophic (1552 spills, 30 ms).
5. **Precision headroom is unused.** The error is 0.111x of tol, and moving the
   resident state to bf16 (halving its register footprint on paper) did not help
   (1.6375 ms) because Triton still materialises an fp32 accumulator for
   `tl.dot`.

What the reference does that Triton 3.4 cannot express here: the CAKE kernel is a
hand-written CUTLASS/`tcgen05` M128 kernel with warp specialisation (separate
producer/consumer warpgroups), TMA loads, and a TMEM accumulator, so the state
never occupies the register file that the math needs, and the MMA pipeline stays
full across block iterations. Closing the remaining 4.8x would require the same:
a CUDA kernel with `tcgen05.mma` M128 tiles, TMA-driven double buffering of
q/k/v/g, and the recurrent state held in TMEM/SMEM rather than registers. That is
a rewrite outside the Triton programming model, not a tuning exercise.

## Files

| File | Purpose |
|------|---------|
| `kda_triton.py` | the new fused KDA forward kernel (kernel source) |
| `kda_common.py` | h96_mixed input generation, fp32 naive reference, timing helpers |
| `refwrap.py` | black-box wrapper pinning the CAKE M128 dispatch configuration |
| `refprobe.py` | eligibility scan that discovered that configuration |
| `final.py` | end-to-end evidence run (ref timing, E_ref/tol, accuracy, benchmark) |
| `check.py`, `check_mine.py`, `bench.py` | accuracy / benchmark drivers |
| `run_*.sh` | cctl batch-job entry scripts |

Reproduce with:

```
pip install flashinfer-python triton==3.4.0
python final.py 32:5:128:4:2:0
```
