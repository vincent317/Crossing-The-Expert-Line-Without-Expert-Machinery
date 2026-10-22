# GDN FWD (long sequences) on B200 — result summary

Workload: Gated DeltaNet forward (prefill), `T=8192` tokens packed into `N=20`
variable-length sequences `[753, 371, 161, 41, 105, 66, 23, 31, 19, 15, 2433,
39, 22, 3371, 20, 42, 219, 15, 45, 401]`, `head_dim=128`, 4 q/k heads, 8 v
heads (GVA), bf16 `q/k/v/a/b`, fp32 k-last state.

All numbers below were measured on one **NVIDIA B200** (148 SMs, driver
580.105.08) in the `aiforai-b200` pool, image `cybertron/pytorch:26.07-py3`
(torch 2.13.0a0+nv26.07, CUDA 13.3), Triton 3.5.0, job `1039619`.

## Reference

`flashinfer.gdn_prefill.chunk_gated_delta_rule` — the CuTe-DSL GDN prefill
kernel upstreamed into FlashInfer (`flashinfer-python 0.6.18.post1` +
`nvidia-cutlass-dsl[cu13]==4.7.1` + `apache-tvm-ffi`).  Used strictly as a
black box: installed, imported, its documented public signature introspected,
and called.  Its source was never opened.

Two environment facts had to be discovered to run it at all:

* the Blackwell (SM100) path refuses to run on CUDA 12 (`NotImplementedError:
  Blackwell GDN prefill is only supported on CUDA 13+`), so the default
  `infra/nvidia-pytorch:latest` image (CUDA 12.9) cannot host it — a CUDA 13
  image is required;
* its `g` argument is **alpha**, the multiplicative forget gate in `(0, 1]`
  (its own docs: *"Forget gate (alpha) ... Defaults to all ones when None"*,
  and the SM90 entry point names the same argument `alpha`), **not** the log
  decay.  Feeding it log-space `g` returns all-NaN, because the kernel takes
  the logarithm internally.  Verified by bisection: `g = 0` → finite output,
  `g = -0.05` → 100% NaN.

| metric | value |
| --- | --- |
| ref device time, per-call CUDA-event span (median of 100) | **116.83 us** (min 115.74) |
| ref device time, amortized over 200 back-to-back calls | **111.39 us** |
| ref device time incl. the gate (softplus/sigmoid) computation | 140.74 us / 134.69 us |
| published figure for the contest artifact | 125.5 us |

The measured 116.8 us sits right next to the published 125.5 us, which
corroborates that this is the same kernel family.

## Tolerance

`gdn_naive.py` is the fp32 anchor: the scalar recurrence, no chunking, fp32
throughout.

| quantity | value |
| --- | --- |
| `E_ref(out)`   = max&#124;ref − naive&#124; | **0.00078304** |
| `tol(out)`     = 7 × E_ref | **0.00548129** |
| `E_ref(state)` = max&#124;ref_state − naive_state&#124; | **0.00637358** |
| `tol(state)`   = 7 × E_ref | **0.04461509** |

## Our kernel

`gdn_triton.py`, best configuration `C=64, split=4, nsq=2, st_scan=3`:

| quantity | value | verdict |
| --- | --- | --- |
| max&#124;ours − naive&#124; (out) | **0.00078270** | ≤ 0.00548129 → **PASS** |
| max&#124;ours − naive&#124; (state) | **0.00712210** | ≤ 0.04461509 → **PASS** |
| device time (same events, same data, same process) | **195.05 us** amortized / 200.51 us per-call | needs ≤ 93.16 us → **NOT MET** |
| best device time observed over the whole run | 179.96 us | 0.62× the reference |

Reproduced by the committed scripts in job `1039647`:

```
[2] reference vs fp32 naive
    E_ref(out)   = 0.00078304      tol(out)   = 7 * E_ref = 0.00548129
    E_ref(state) = 0.00637358      tol(state) = 7 * E_ref = 0.04461509
[3] our Triton kernel (C=64, split=4) vs fp32 naive
    max_err(out)   = 0.00078270 <= 0.00548129 : PASS
    max_err(state) = 0.00712210 <= 0.04461509 : PASS
ref (kernel only)   : per-call = 116.86 us   amortized = 111.79 us
ours (C=64,split=4) : per-call = 200.51 us   amortized = 195.05 us
  .. wy   91.70 us      .. scan  114.18 us
```

Accuracy is essentially identical to the reference's own (7.83e-4 for the
reference vs 7.83e-4 for ours against the same fp32 anchor), which confirms the
algorithm reproduces the reference semantics exactly — the gap is purely
performance.

## Algorithm (derived from the recurrence, not copied)

See `README.md`.  Chunked WY representation: per chunk, `T = (I+M)^{-1}` for the
unit-lower-triangular `I + M` via nilpotency doubling
(`(I-M)(I+M²)(I+M⁴)…`, i.e. `2·log2(C)` matmuls); `W`, `Uv`, `Obase`, `Qeff`
are all state independent and computed fully in parallel; the sequential pass
is reduced to three `D × D`-class matmuls per chunk
(`O += Qeff S₀ᵀ`, `U = Uv − W S₀ᵀ`, `S = exp(G_C) S₀ + (diag(e)U)ᵀK`).

Optimisations that measurably helped:

* every row of the state along `d_v` evolves independently of the others, so
  `d_v` is sliced across `SPLIT` CTAs — this shortens the critical path of the
  3371-token sequence with no redundant arithmetic (split 1 → 4 was the single
  largest win, 402 us → 260 us on the first working version);
* the state is kept **transposed** as `[d_k, d_v_slice]` so it is always the
  right-hand dot operand and no register tile is transposed through shared
  memory inside the sequential loop;
* `W` and `Qeff` share one `(2C, D)` workspace tile so the sequential pass
  issues one 32 KB contiguous load instead of two 16 KB loads;
* the row factor `exp(G_C − G)` is folded into the stored `W` and `Uv`, so the
  loop needs neither that vector nor an `exp2`;
* all exponentials in the log2 domain (`ex2.approx` hardware path);
* tensor liveness in the prep kernel is kept short (load `v` and `q` late,
  keep `W`/`Uv` in bf16) because an extra pair of fp32 `(64,128)` tiles alive
  across the attention block costs 2×32 registers per thread and drops the
  occupancy from 2 CTAs/SM to 1 — worth ~10% end to end;
* `nsq` truncates the doubling expansion.  With this workload
  `‖M‖₂ ≈ 0.4`, so stopping after `M⁷` leaves a residual below `1e-4` — far
  under bf16 — and the tolerance check confirms it (error unchanged at
  7.83e-4).  `nsq=None` restores the exact full expansion.

Also measured and rejected: manual double buffering in the scan loop (register
pressure r158 → r247, scan 91 → 159 us), fusing the two scan matmuls into one
doubled-row tile (scan 124 → 158 us), `num_warps=16/32` for the prep kernel
(spills: 30–48 slots, wy 98 → 161 us), `C=32` (prep 92 us but scan 193 us),
`C=128` (scan exceeds the 232 KB shared-memory limit), splitting the prep into
two kernels (adds a 56 us attention kernel and reloads `W`/`Uv`).

## Bottleneck

Both kernels are limited by **Triton's fixed cost per `tl.dot` on Blackwell**,
not by arithmetic or bandwidth:

* The algorithm needs ~14 GFLOP total.  A one-CTA microbenchmark of chained
  `(64,128)x(128,128)` bf16 `tl.dot`s with shared-memory-resident operands
  reaches **6.7 TFLOP/s per SM and 933 TFLOP/s over 148 CTAs** with Triton
  3.5 — so 14 GFLOP is ~20 us of tensor-core time.  We are 10× off that.
* In the sequential pass each iteration costs ~1.7–2.3 us for three matmuls and
  ~48 KB of loads, i.e. **~0.6–0.7 us per `tl.dot`** (≈1300 cycles) at
  0.9 TFLOP/s per CTA — independent of the tile shape.  Fitting
  `per_iter = a/split + b` over split ∈ {1,2,4,8} gives `b ≈ 1.6 us`, and `b`
  is the same for `C=32` (26 KB/iteration) as for `C=64` (52 KB/iteration), so
  `b` is **fixed latency, not bandwidth**.  `num_stages` 2→3 recovers only
  ~10%; the per-chunk output store keeps Triton's pipeliner from covering the
  loop, and manual prefetching costs more in registers than it buys.
* In the prep kernel the ten-to-twelve matmuls of the triangular inverse form a
  strictly sequential register-fed chain at ~1.75 us per dot, with only
  2 CTAs/SM to hide it.  Occupancy cannot be raised: 128 registers × 256
  threads already fills half the register file, and it is the live
  `(64,64)`/`(64,128)` tiles that set that, not the warp count.
* Halving the dot count by batching two v-heads into one block-diagonal
  `128×128` inverse was analysed and rejected: it halves the dots but doubles
  the per-CTA register footprint, so `total_dots / CTAs_per_SM` — the quantity
  that actually sets the time — does not move.

The reference closes exactly this gap: a CuTe-DSL kernel drives `tcgen05` with
warp-specialised TMA pipelines and keeps the MMA unit fed across chunk
boundaries, so its effective per-matmul overhead is one to two orders of
magnitude lower.  Reaching 1.2× **faster** than it would require ~2× on both
kernels simultaneously, which needs that same level of control — realistically
a hand-written CUDA/CuTe implementation rather than Triton.

Two further structural options were designed but not implemented (each needs a
~2× win on one half only, so neither alone reaches the target):

1. Per-sequence streams so the prep kernel of the short sequences overlaps the
   long sequence's scan — models out at ~150–160 us.
2. Precomputing each chunk's `128×128` affine transition `(M_c, B_c)` in
   parallel and composing groups of four, which cuts the sequential critical
   path from 159 matmuls to ~22 — models out at ~130–145 us end to end.

## Completion status against the stated conditions

1. reference device time measured on B200 and printed — **done** (116.83 us
   per-call span / 111.39 us amortized).
2. fp32 naive implemented, `E_ref` and `tol = 7·E_ref` printed — **done**
   (7.83e-4 → 5.48e-3 for the output; 6.37e-3 → 4.46e-2 for the state).
3. our kernel within tolerance of the fp32 naive, numbers printed — **done**
   (7.827e-4 ≤ 5.48e-3 and 7.122e-3 ≤ 4.46e-2, both PASS).
4. our device time ≤ ref / 1.2 — **not met**: 195.00 us (best observed
   179.96 us) against a 92.83 us requirement, i.e. 0.57–0.62× the reference.
5. kernel source, benchmark script and accuracy-check script saved here and
   committed — **done** (`gdn_triton.py`, `bench.py`/`final_run.py`,
   `check.py`).
