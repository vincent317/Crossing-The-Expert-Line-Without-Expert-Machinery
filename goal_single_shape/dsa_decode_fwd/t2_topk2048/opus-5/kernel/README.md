# DSA FWD decode kernel (sparse MLA), B200

Target op — long/short mixed decode batch:
`q_nope (2,16,512)`, `q_pe (2,16,64)`, `ckv_cache (8462,64,512)`,
`kpe_cache (8462,64,64)`, `sparse_indices (2,2048) int32`, bf16,
page size 64, top-k 2048, 6 / 337 valid KV rows (343 total).

## Result (NVIDIA B200, measured in one process on identical tensors)

| | device time |
|---|---|
| reference — KDA 0.5 CuTe-DSL contest kernel | **6.279 µs** |
| target = ref / 1.2 | 5.233 µs |
| this kernel (`dsa_kernel.cu`) | **5.117 µs — 1.227x** |

Accuracy, against an fp32 naive ground truth (gather → fp32 matmul → fp64
softmax → fp32 matmul):

| | max abs error |
|---|---|
| `E_ref` = reference vs fp32 naive | 7.780790e-03 |
| `tol` = 7 × E_ref | 5.446553e-02 |
| this kernel vs fp32 naive | **1.135790e-02 (PASS)** |

Device time = the profiler's CUDA kernel duration, i.e. the GPU-side execution
span with host launch overhead excluded. Both the reference and this kernel are
single-kernel ops, so the duration is also the "first kernel start to last
kernel end" span.

## Files

| file | role |
|---|---|
| `dsa_kernel.cu` | the kernel (SM100 CUDA C++ / inline PTX) |
| `dsa_cuda.py` | JIT build + callable wrapper |
| `dsa_triton.py` | a second, independent Triton implementation (correct, slower) |
| `check.py` | accuracy check — prints E_ref, tol, and our error |
| `bench.py` | device-time benchmark — prints the reference, the target and ours |
| `workload.py` | deterministic workload + fp32 naive ground truth |
| `timing.py` | measurement helpers |
| `ref_contest.py` | black-box driver for the reference kernel |
| `impls.py` | the configuration each entry point uses |
| `final_run.py` | the development sweep harness (config sweep + stage-by-stage costs) |

The reference is `mit-han-lab/mlsys2026-flashinfer-contest-solution` @ `7aad0429`,
`kernels/dsa_attention/kernel.py`. It is fetched verbatim at run time, imported
by path and only executed — never read or copied — and is not committed here
(see `.gitignore`). `ref_contest.py` holds the URL.

Run: `pip install nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi` (clear `PIP_CONSTRAINT`
first on images that ship one), fetch the reference file, then
`python -u check.py && python -u bench.py`.

## Design

Split-KV flash decoding in a single kernel.

* **CTA = one slice of index slots.** CTA `(s, hg, t)` owns index slots
  `[s*32, s*32+32)` of request `t` (plus one more strided slice, `tpp=2`) and
  heads `[hg*8, hg*8+8)`. The slice is small on purpose: one SM only sustains
  ~25 GB/s on this row gather, so the KV bytes have to be spread across SMs for
  the memory time to hide behind the kernel-launch floor. A slice with no valid
  index exits before touching KV, which also balances the work when the valid
  indices are a short prefix. Exactly one KV tile in flight per step, so the
  softmax is one-pass with no accumulator rescaling.
* **Both GEMMs on bf16 tensor cores** via `mma.sync.m16n8k16`. QK: heads are the
  M dimension, `ldmatrix.x4` for Q and `ldmatrix.x2` for K — the non-transposed
  ldmatrix distribution *is* the m16n8k16 B layout when the KV rows are the
  matrix rows. PV: `ldmatrix.x4.trans` for V. Four independent mma accumulators
  break the 36-long dependent chain in the k loop.
* **Reduction without a partial buffer.** Every CTA exponentiates against a fixed
  reference of 0, which makes partial numerators and denominators directly
  addable, and reduces them straight into the output with
  `red.global.add.noftz.bf16x2`. One release-ordered arrive/wait per request
  follows, then each CTA divides its own slice of the output and zeroes it for
  the next launch (CUDA-graph safe — the generation comes from the ticket the
  CTA draws, so nothing is reset between launches).
* Shared-memory row strides are padded so every ldmatrix / LDS access is
  bank-conflict free; the per-request arrival counters are padded to separate
  cache lines (sharing one costs ~1 µs of polling interference).

Ceiling worth knowing: the fixed exponent reference is exact while every logit
is ≤ 80; above that the clamp degrades instead of producing inf. Attention
logits are O(1)–O(10) in practice. `fused=False` selects the exact per-split
max + log-sum-exp path instead (a second kernel, ~7 µs).

## What the cost actually is

Stage-by-stage, measured by compiling the kernel with a runtime `stop` switch
(`final_run.py`, STAGE_D2), for the `bn=32 w8 hg=2` configuration:

| stage | cumulative |
|---|---|
| launch only | 1.24 µs |
| + read this CTA's index slice | 1.78 µs |
| + stage KV and Q (cp.async) | 2.49 µs |
| + QK mma | 3.15 µs |
| + softmax and P | 3.17 µs |
| + PV mma | 3.39 µs |
| + reduce into the output | 4.14 µs |
| + arrive/wait and divide | 5.12 µs |

So the op is dominated by fixed costs — a ~1.2 µs kernel-launch floor plus two
dependent global round trips (indices, then the KV rows they point at) — not by
bandwidth: the 343 KV rows are only ~395 KB and stage in ~0.5 µs.
