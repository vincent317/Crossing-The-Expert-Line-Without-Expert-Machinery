# DSA sparse MLA decode — final benchmark report

**Definition:** `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
**GPU:** NVIDIA B200 (sm_100), `/opt/sglang` torch 2.13+cu130, Triton.
**Timing口径:** CUPTI kernel span (first kernel start → last kernel end), cold L2 before each call, median over iterations — identical to `kda05-ref --bench` (via `flashinfer.testing.bench_gpu_time(enable_cupti=True, cold_l2_cache=True)`). Reference and solution measured on the same card, alternating.

**Reference choice:** For every workload the reference is **KDA-0.5** (`kda05-ref --op dsa --inputs in.pt --bench`), the fastest of the candidates on this card. The FlashInfer trtllm sparse-MLA library baseline (`flashinfer_wrapper_5af199`) runs here but is far slower than KDA on the CUPTI metric and ~20× slower than this solution in the official FIB harness, so it is never the fastest reference.

**Correctness method:** A fp32 naive implementation (exact transcription of the definition's mathematical reference) is the gold value. Per-workload tolerance is derived as `tol = 7 × max|naive − KDA|` (floored at 2e-3 for output, 2e-2 for lse), then the solution is required to be within `tol` of the fp32 naive. All 23 workloads pass this, and the official FIB harness reports **all 23 PASSED** (`flashinfer-bench run … --definitions dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`), 110–128× vs. the mathematical reference versus the library baseline's 4–7×.

## Solution

A single from-scratch **Triton** kernel (`solution/main.py`), no call to any existing attention/MLA library:

- **Split-KV flash decode, fused into one launch.** Grid = `(num_tokens × NS)` with `NS=16` splits. Each program streams a contiguous slice of its token's `sparse_indices`, gathering the paged KV once and sharing it across all 16 heads (bf16 tensor-core `tl.dot` for both `q_nope·ckv` and `q_pe·kpe`), running an online (base-2) softmax and writing a per-split partial `(m, l, acc)`.
- **Padding (`-1`) handled generally** — masked gathers + `-inf` logits — so the kernel is correct for arbitrary data, not just the contiguous-prefix layout of the dataset.
- **Combine fused via an atomic "last-block" reduction.** A per-token completion counter (`atomic_add`, `acq_rel`) lets the last split-block of each token reduce the active splits (`atomic_max`-tracked) into the final `output` (bf16) and 2-based `lse` (f32). This keeps everything in **one kernel launch**, avoiding the ~17 µs CPU inter-kernel gap that a second Triton launch incurs on the CUPTI span.
- bf16 I/O, fp32 accumulation; `num_warps=8`, pipelined (`SKIP=False`) phase-1. Per-split partials (`po`) are stored in **bf16** to halve the combine's read traffic (fp32 `m`/`l` are kept for exact softmax rescale), which is safe because each partial is normalized by `l` in the combine so the bf16 relative error (~2⁻⁸) carries through to a ~0.015 absolute output error — well inside tolerance.

## Results

| workload (uuid) | T | ref=KDA µs | mine µs | ref/mine | out err | out tol | lse err | lse tol | PASS |
|---|---|---|---|---|---|---|---|---|---|
| 0c23b10c | 1 | 3.968 | 16.288 | 0.244 | 0.0136 | 0.0545 | 0.00000 | 0.02000 | ✅ |
| 05f6de65 | 2 | 11.040 | 19.904 | 0.555 | 0.0151 | 0.0544 | 0.00001 | 0.02000 | ✅ |
| 0a63b87b | 2 | 7.360 | 16.672 | 0.441 | 0.0135 | 0.0568 | 0.00001 | 0.02000 | ✅ |
| 9d4a5f21 | 2 | 5.152 | 16.448 | 0.313 | 0.0138 | 0.0544 | 0.00001 | 0.02000 | ✅ |
| 9f3f891b | 2 | 10.912 | 20.639 | 0.529 | 0.0141 | 0.0667 | 0.00001 | 0.02000 | ✅ |
| b7668cfd | 2 | 6.976 | 16.256 | 0.429 | 0.0142 | 0.0544 | 0.00001 | 0.02000 | ✅ |
| e6b849f2 | 2 | 10.880 | 17.440 | 0.624 | 0.0133 | 0.0539 | 0.00001 | 0.02000 | ✅ |
| f77df5ce | 2 | 5.344 | 15.840 | 0.337 | 0.0138 | 0.0544 | 0.00001 | 0.02000 | ✅ |
| fc85411e | 2 | 5.152 | 15.840 | 0.325 | 0.0129 | 0.0541 | 0.00001 | 0.02000 | ✅ |
| 68d6817d | 6 | 5.856 | 16.288 | 0.360 | 0.0142 | 0.0917 | 0.00001 | 0.02000 | ✅ |
| d57eb9e1 | 6 | 16.288 | 29.792 | 0.547 | 0.0136 | 0.0542 | 0.00001 | 0.02000 | ✅ |
| ddfa9e34 | 6 | 15.632 | 27.744 | 0.563 | 0.0151 | 0.0642 | 0.00001 | 0.02000 | ✅ |
| 2207f0fd | 7 | 15.809 | 30.368 | 0.521 | 0.0129 | 0.0542 | 0.00001 | 0.02000 | ✅ |
| 38389961 | 7 | 15.088 | 25.152 | 0.600 | 0.0150 | 0.0545 | 0.00001 | 0.02000 | ✅ |
| ae4219a9 | 7 | 15.680 | 30.816 | 0.509 | 0.0155 | 0.0546 | 0.00001 | 0.02000 | ✅ |
| 02d6ae9c | 8 | 16.000 | 29.472 | 0.543 | 0.0149 | 0.0547 | 0.00001 | 0.02000 | ✅ |
| 232ed014 | 8 | 15.072 | 25.760 | 0.585 | 0.0142 | 0.0544 | 0.00001 | 0.02000 | ✅ |
| 385742b2 | 8 | 15.024 | 25.248 | 0.595 | 0.0152 | 0.1067 | 0.00002 | 0.02000 | ✅ |
| 4c46a94b | 8 | 13.248 | 23.872 | 0.555 | 0.0140 | 0.0547 | 0.00001 | 0.02000 | ✅ |
| 5096e459 | 8 | 16.320 | 30.656 | 0.532 | 0.0129 | 0.0547 | 0.00001 | 0.02000 | ✅ |
| 564007ac | 8 | 16.689 | 30.688 | 0.544 | 0.0155 | 0.0667 | 0.00001 | 0.02000 | ✅ |
| 78b2e11c | 8 | 16.592 | 29.568 | 0.561 | 0.0147 | 0.0544 | 0.00001 | 0.02000 | ✅ |
| 7a389715 | 8 | 15.392 | 28.480 | 0.540 | 0.0151 | 0.0547 | 0.00001 | 0.02000 | ✅ |

**Geometric-mean speedup (ref/mine) over all 23 workloads = 0.480**  (i.e. KDA-0.5 is ~2.1× faster on the CUPTI metric)
**All 23 correct vs. fp32 naive within tolerance = True; official FIB harness: all 23 PASSED.**

## Analysis / notes

- The op is tiny and latency/occupancy-bound, not bandwidth-bound: even the heaviest workload gathers ≈5 MB of KV (<1 µs at HBM peak), so the wall-clock is dominated by kernel launch, occupancy and the split→combine reduction, not raw bandwidth.
- Measured structural facts that shaped the design: a single Triton kernel launch floors at ~1.9 µs; a **second** Python-launched Triton kernel adds ~17 µs of CPU-induced gap to the CUPTI span (2 kernels = 19 µs span for two no-ops), which is why the solution is fused into one launch via the atomic combine rather than a separate combine kernel. CUDA Graphs would hide that gap but require data-independent captured pointers (the 554 MB KV cache makes copy-in infeasible), so they are not used.
- `ncu` on the fused kernel (heaviest T=8 workload) confirms the bound: 34.9 µs at **12.5% achieved warp occupancy, 2.1% DRAM throughput, 6.9% SM throughput** — purely latency/occupancy-bound, not compute- or bandwidth-bound. With `T·NS = 8·16 = 128` blocks on 148 SMs (≈1 block/SM) the memory/MMA latency is not hidden. Raising occupancy requires more splits (but the single-launch serial combine then dominates — measured worse), a separate parallel combine (but the ~7–18 µs Triton inter-kernel launch gap then dominates — measured worse), or a cooperative grid-sync combine (CUDA-only). Standalone phase-1 is ~14 µs; the single-block-per-token combine is the remaining ~20 µs.
- Honest gap to the reference: KDA-0.5 is a hand-tuned CUDA kernel (~15 µs on the heavy T=8 cases, ~4 µs on the tiny T=1 case). This Triton solution's phase-1 (tensor-core, ~14 µs on the heaviest) is competitive, but the fused single-launch combine and the ~19 µs floor on the small/low-occupancy cases keep it ~2× behind KDA on the CUPTI metric. Several from-scratch CUDA-core designs (warp-per-head, lane-per-entry, smem-staged) were implemented and measured but were all 5–15× slower than the Triton kernel because, without tensor cores, the per-entry dot-product reductions become shuffle/latency-bound. A from-scratch **WMMA (tensor-core) kernel** was also written and made numerically correct (`experiments/wmma_dsa.cu`), but ran ~100× slower than the Triton kernel (per-tile shared-memory round-trips + low warp-occupancy); even multi-warp occupancy would not approach Triton's auto-generated pipeline. The conclusion is that Triton's compiler (software pipelining, register allocation, warp scheduling) already produces a near-optimal tensor-core phase-1, and the remaining ~2.2× to KDA-0.5 is expert hand-tuned CUDA (fused parallel combine via a cooperative grid, which Triton cannot express) rather than something reachable by a from-scratch kernel in this environment.
- Relative to the FlashInfer library baseline that the FIB harness also runs, this solution is ~20× faster (110–128× vs. mathematical reference, versus 4–7× for the library).
- **Arbitrary-data correctness** was stress-tested (`harness/stress.py`, 60 cases over T∈{1,2,6,7,8}): small-valid, **non-contiguous `-1` holes**, full-2048-valid, **all-`-1` (empty) tokens**, ×3-scaled queries, and random indices across the full `num_pages·64` range — all pass vs. fp32 naive with worst output error 0.015 (bf16 level), worst lse error 6e-5, and zero inf-pattern mismatches. The kernel masks `-1` generically and does not rely on the dataset's contiguous-prefix layout.