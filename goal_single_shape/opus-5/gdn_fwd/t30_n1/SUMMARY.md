# GDN FWD (T=30, N=1) on B200 — final result

Workload: q/k (30,4,128) bf16, v (30,8,128) bf16, a/b (30,8) bf16,
A_log/dt_bias (8,) fp32, state (1,8,128,128) fp32 **k-last**, cu_seqlens (2,)
→ output (30,8,128) bf16, new_state (1,8,128,128) fp32. The new kernel computes
the gates and the q/k l2norm itself (`g = -exp(A_log)·softplus(a+dt_bias)`,
`beta = sigmoid(b)`, `scale = 1/√128`), i.e. it does strictly more work than the
reference call it is compared against.

All numbers below were printed by `bench.py` on one clock-locked B200
(`nvidia-smi -lgc 1965`, SM clock 1852 MHz under load) via `cctl` batch jobs on
the `cybertron/pytorch:26.07-py3` image (torch 2.13 / CUDA 13.3 / triton 3.7.1).
Device time = kernel span from the CUPTI trace (last kernel end − first kernel
start), median of 5 profiles of 50 calls each.

## Final numbers

| | device time (span = cupti) | max err vs fp32 oracle (out / state) |
| --- | --- | --- |
| **reference** — flashinfer CuTe-DSL `chunk_gated_delta_rule_sm100` | **16.74 µs** (1 kernel, 5-run range 16.70–16.77) | 1.13e-03 / 6.97e-03 |
| reference — high-level `chunk_gated_delta_rule` | 169.4 µs span, 34.5 µs kernel-sum (4 kernels) | same |
| reference — `cp_delta_rule_dsl_sm100` | 170.3 µs span, 34.7 µs kernel-sum (4 kernels) | 1.13e-03 / 7.45e-03 |
| **new kernel** (`kernel.py`, Triton, BV=8, 4 warps, NLEV=2) | **4.288 µs** (1 kernel, range 4.096–4.288) | 9.66e-04 / 4.96e-03 |
| new kernel (`gdn_kernel.cu`, CUDA register scan) | 11.3 µs | 9.66e-04 / 3.58e-06 |

Accuracy verdict (identical data, identical oracle):

```
E_ref = max|ref - fp32_naive| : out 1.127362e-03  state 6.965339e-03  -> 6.965339e-03
tol   = 7 * E_ref             : out 7.891536e-03  state 4.875737e-02  -> 4.875737e-02
E_new = max|new - fp32_naive| : out 9.663403e-04  state 4.963994e-03
ACCURACY PASS  (both tensors inside their per-tensor tolerance)
```

Speed verdict:

* vs the **measured** reference path: 16.74 / 4.288 = **3.90×**, i.e. well
  inside `≤ ref/1.2` → PASS.
* vs the **published 4.9 µs gdn_small figure**: 4.288 µs vs the required
  4.083 µs → **FAIL by 5 %** (achieved 1.14×, required 1.20×).

## Conditions

| # | condition | status |
| --- | --- | --- |
| 1 | ref device time measured on B200 and printed | **met** — 16.74 µs (fastest valid CuTe-DSL prefill entry) |
| 2 | fp32 naive oracle, E_ref and tol = 7·E_ref printed | **met** — E_ref = 6.97e-03, tol = 4.88e-02 |
| 3 | max\|new − naive\| ≤ tol, printed | **met** — 4.96e-03 ≤ 4.88e-02 |
| 4 | new device time ≤ ref/1.2, same harness/data | **met vs the measured ref (3.90×)**, **not met vs the published 4.9 µs bar (1.14× < 1.20×)** |
| 5 | kernel source, benchmark, accuracy script committed | **met** |

## Why the reference measures 16.7 µs and not 4.9 µs

The published figure refers to a short-sequence (`gdn_small`) path. That path is
**not reachable through the reference package's public API**. Enumerating every
public GDN entry point (`probe12.py`, `probe13.py` in the job logs) gives:

* `gdn_prefill.chunk_gated_delta_rule_sm100` — generic chunked CuTe-DSL prefill,
  1 kernel, **16.7 µs**; the fastest valid path for this shape.
* `gdn_prefill.chunk_gated_delta_rule` (high-level) and
  `cp_delta_rule_dsl_sm100` — 4 kernels, 34.5 µs of kernel time, ~170 µs span.
* `chunk_gated_delta_rule_sm90` / `_sm120` — refuse to compile on this device
  (DSL internal error).
* `gdn_kernels.gated_delta_rule*`, `gdn_decode_*`, `run_*_decode` — the decode /
  MTP family; they assert on T=30 (they accept 1 token, or a few for MTP).

Getting the reference to produce *valid* numbers at all took four separate
fixes, all found by black-box probing (never by reading its source):

1. It needs CUDA 13 — on the `infra/nvidia-pytorch` image (CUDA 12.9) the SM100
   path raises `NotImplementedError: Blackwell GDN prefill is only supported on
   CUDA 13+`.
2. `pip install flashinfer-python[cu13]` (wheels only, no torch pin) is required
   so the `nvidia-cutlass-dsl-libs-cu13` companion wheels land too.
3. `gate` is the **multiplicative** forget gate `alpha = exp(g)`, exactly as the
   docstring says. Passing log-decay makes the kernel write NaN into every
   element of the output and the state — for every shape, dtype and gate
   magnitude, which is what made this hard to localise.
4. The arch-specific entry has **no l2norm flag**, so q/k must be normalised by
   the caller; the high-level wrapper's own l2norm path returns NaN in this
   build.

## Where the remaining 5 % is

Measured floors on this shape and grid (`floor.py`): an **empty** kernel with the
same launch geometry is **1.54 µs** of device time, and a kernel that only
*reads* q/k/v/state and *writes* out/new_state is **2.05 µs**. So of the new
kernel's 4.29 µs, 2.05 µs is launch + unavoidable traffic and ~2.2 µs is compute.

The compute is 9 `tl.dot`s (2 gram matrices, 2 Neumann-chain products, 2 state
matmuls, the W solve, the output matmul, the state update). Triton stages every
`tl.dot` operand through shared memory — the PTX shows ≈60 `bar.sync` and ≈265
`st.shared` per program — which puts a floor of ≈230 ns on each dot regardless of
its arithmetic. Measured consequences:

* Neumann depth: NLEV 5 → 2 → 4.83 → 4.07 µs, i.e. ~0.19 µs per pair of dots.
* Everything else tried was neutral or worse: BV 16/32 (+0.4/+1.1 µs), 2 or 8
  warps (+0.7 µs), fp16 or tf32 operands (+0.5 µs), hoisting the state load
  ahead of the chain (+0.45 µs — it lengthens the live range and costs
  occupancy), `num_stages=2` (neutral).
* A hand-written CUDA alternative (`gdn_kernel.cu`) avoids Triton's staging
  entirely and is numerically far better (state error 3.6e-06 vs 5.0e-03), but a
  register-resident scan is latency-bound at 4–8 warps/SM: 11.3 µs even after
  grouping RG tokens per step and vectorising its shared-memory passes. Its
  ablation (`NG` override) shows 1.66 µs fixed + ~1.0 µs per 4-token group.

Closing the last 5 % therefore needs one of: fewer dots (merging the two gram
matmuls and the two state matmuls needs a 64-row stacked q/k tile, which Triton
can only produce via a `tl.cat`/reshape that costs its own shared round trip), or
a CUDA/CUTLASS implementation that keeps MMA fragments in registers between
dependent matmuls — i.e. what the reference's own `gdn_small` kernel presumably
does.

## Files

`kernel.py` (new kernel, Triton) · `gdn_kernel.cu` + `kernel_cuda.py` (CUDA
alternative) · `common.py` (workload, fp32 oracle, timing) · `check.py`
(accuracy) · `bench.py` (device time) · `refmod.py` (reference as a black box) ·
`verify_math.py` (CPU algebra check, machine precision) · `floor.py` (launch/IO
floor + ablation) · `probe*.py` (reference-convention probes) · `submit.sh`,
`run_on_b200.sh`, `setup_ref*.sh` (B200 job plumbing).
