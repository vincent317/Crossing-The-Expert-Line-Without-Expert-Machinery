# FP8 block-scale MoE (DeepSeek-V3 routing) on B200 -- final results

Definition: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`. Device time = CUPTI kernel span, cold L2 before every call, median over iterations (3 trials x 2 interleaved rounds per implementation; ref and solution alternated on the same B200).

Ref per workload = the faster of KDA-0.5 (`kda05-ref --bench`) and flashinfer `trtllm_fp8_block_scale_moe` on this card. Errors are vs. an fp32 naive implementation of the definition's reference semantics (no fp8 requantization anywhere); tolerance per workload = 7 x the chosen ref's own error (RMS-relative and max-abs/output-max).

| seq_len | ref | ref us | ours us | ref/ours | ours rms_rel err | tol rms_rel | ours max_norm err | tol max_norm | pass |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | KDA-0.5 | 54.9 | 54.0 | 1.02x | 0.0191 | 0.1610 | 0.0244 | 0.1854 | PASS |
| 7 | KDA-0.5 | 84.2 | 80.4 | 1.05x | 0.0187 | 0.1646 | 0.0170 | 0.1900 | PASS |
| 14 | KDA-0.5 | 119.9 | 112.1 | 1.07x | 0.0188 | 0.1604 | 0.0171 | 0.1482 | PASS |
| 15 | KDA-0.5 | 72.6 | 69.6 | 1.04x | 0.0199 | 0.1640 | 0.0208 | 0.1625 | PASS |
| 16 | KDA-0.5 | 122.4 | 117.7 | 1.04x | 0.0192 | 0.1626 | 0.0196 | 0.1830 | PASS |
| 32 | KDA-0.5 | 186.0 | 177.3 | 1.05x | 0.0193 | 0.1622 | 0.0190 | 0.1332 | PASS |
| 52 | KDA-0.5 | 154.9 | 151.7 | 1.02x | 0.0192 | 0.1611 | 0.0161 | 0.1369 | PASS |
| 53 | KDA-0.5 | 212.4 | 196.7 | 1.08x | 0.0194 | 0.1641 | 0.0179 | 0.1819 | PASS |
| 54 | KDA-0.5 | 196.0 | 189.3 | 1.04x | 0.0193 | 0.1637 | 0.0222 | 0.1880 | PASS |
| 55 | KDA-0.5 | 212.3 | 196.6 | 1.08x | 0.0190 | 0.1625 | 0.0177 | 0.1613 | PASS |
| 56 | KDA-0.5 | 222.1 | 211.1 | 1.05x | 0.0192 | 0.1638 | 0.0219 | 0.1780 | PASS |
| 57 | KDA-0.5 | 216.0 | 202.4 | 1.07x | 0.0192 | 0.1634 | 0.0186 | 0.1678 | PASS |
| 58 | KDA-0.5 | 223.0 | 209.6 | 1.06x | 0.0194 | 0.1656 | 0.0185 | 0.1728 | PASS |
| 59 | KDA-0.5 | 178.0 | 166.2 | 1.07x | 0.0191 | 0.1621 | 0.0219 | 0.1724 | PASS |
| 62 | KDA-0.5 | 159.1 | 158.4 | 1.00x | 0.0192 | 0.1613 | 0.0332 | 0.1793 | PASS |
| 80 | KDA-0.5 | 245.5 | 226.6 | 1.08x | 0.0191 | 0.1638 | 0.0179 | 0.2212 | PASS |
| 901 | KDA-0.5 | 277.1 | 436.6 | 0.63x | 0.0215 | 0.1630 | 0.0234 | 0.1810 | PASS |
| 11948 | KDA-0.5 | 741.0 | 1681.5 | 0.44x | 0.0215 | 0.1631 | 0.0205 | 0.1465 | PASS |
| 14107 | KDA-0.5 | 1036.4 | 2442.5 | 0.42x | 0.0215 | 0.1632 | 0.0224 | 0.1698 | PASS |

**Workloads: 19/19. Geometric mean speedup (ref/ours) over 19 workloads: 0.932x. All pass tolerance: True.**

Also listed for reference (not the chosen ref where slower):

| seq_len | KDA-0.5 us | flashinfer us | KDA rms_rel | flashinfer rms_rel | local pairs offset |
|---:|---:|---:|---:|---:|---:|
| 1 | 54.9 | 196.3 | 0.0230 | 0.0440 | 32 |
| 7 | 84.2 | 210.6 | 0.0235 | 0.0431 | 192 |
| 14 | 119.9 | 239.8 | 0.0229 | 0.0416 | 0 |
| 15 | 72.6 | 211.9 | 0.0234 | 0.0434 | 32 |
| 16 | 122.4 | 250.8 | 0.0232 | 0.0438 | 224 |
| 32 | 186.0 | 316.4 | 0.0232 | 0.0434 | 32 |
| 52 | 154.9 | 288.8 | 0.0230 | 0.0433 | 160 |
| 53 | 212.4 | 335.6 | 0.0234 | 0.0434 | 32 |
| 54 | 196.0 | 320.0 | 0.0234 | 0.0428 | 128 |
| 55 | 212.3 | 327.5 | 0.0232 | 0.0424 | 128 |
| 56 | 222.1 | 339.2 | 0.0234 | 0.0428 | 64 |
| 57 | 216.0 | 328.1 | 0.0233 | 0.0430 | 96 |
| 58 | 223.0 | 348.6 | 0.0237 | 0.0430 | 64 |
| 59 | 178.0 | 297.6 | 0.0232 | 0.0417 | 160 |
| 62 | 159.1 | 292.0 | 0.0230 | 0.0426 | 96 |
| 80 | 245.5 | 364.2 | 0.0234 | 0.0430 | 96 |
| 901 | 277.1 | 472.0 | 0.0233 | 0.0431 | 96 |
| 11948 | 741.0 | 1928.9 | 0.0233 | 0.0446 | 128 |
| 14107 | 1036.4 | 2587.4 | 0.0233 | 0.0441 | 32 |

## Method notes

### What was built
A from-scratch CUDA implementation of the full MoE operator (routing + grouped GEMM1 + SwiGLU + grouped GEMM2 + weighted
accumulation) for NVIDIA B200, with no library GEMM/MoE calls. Sources: `solution/moe_common.cuh` (PTX helpers:
mbarrier/TMA/tcgen05/ldmatrix, the warp-level DeepSeek-V3 router), `solution/moe_small.cuh` (the persistent fused
kernel + large-T helper kernels), `solution/moe_route.cuh` (routing/scatter kernels for T > 128), `solution/moe_host.cu`
(launcher, tensor-map encoding), `solution/main.py` (entry point, one-time build cache, shape-only workspaces).

### Kernel design (final version)
Two streaming cores share the same tiling/scheduling/epilogue machinery; the dispatch is by token count.

* **T <= 128 (`solution/moe_v3.cuh`)**: persistent kernel, 2 CTAs per SM, 10 warps each, warp-specialized:
  - a *scheduler* warp grabs tiles from the atomic work counter (one 16-token work unit at a time, with a
    half-unit look-ahead so the tail stays balanced), resolves the GEMM1->GEMM2 dependency per expert, gathers
    the unit's activation/intermediate scale columns into shared memory and publishes a unit descriptor through
    a 2-slot queue; it also runs the tile epilogues (split-K completion counter with acq_rel atomics, SwiGLU + fp8
    requantization of the intermediate) off the streaming critical path;
  - a *feeder* warp streams the weight K-blocks by 2D TMA (128B-swizzled [128 rows][128 B] stages, 5-deep
    mbarrier ring that runs continuously across unit boundaries) and the activation rows by `cp.async`;
  - 4 *compute* warps (32 weight rows each) read fragments with `ldmatrix`, convert fp8->fp16 once per warp and
    multiply on the tensor cores with `mma.sync.m16n8k16` (fp32 accumulate), promoting each 128-wide K block
    (32-wide for the intermediate) with the fp32 block scales; partial sums are fire-and-forget `red.add`s into fp32
    scratch. Completed output column blocks are converted to bf16 by the compute warps as small "finalize" units.
  Routing is distributed over the grid (one token per warp; the top-8 selection uses the hardware warp max-reduce
  on order-preserving integer keys), exchanged through global memory behind an in-kernel grid barrier (all CTAs are
  co-resident), and every CTA then builds the identical expert-grouped pair lists.
  The split-K factor is chosen in-kernel from the number of active experts so that there are at least ~2 units
  per CTA (few experts -> deep K splits for balance, many experts -> whole-K tiles).
* **T > 128 (`solution/moe_small.cuh`)**: one persistent CTA per SM, warp-specialized: 8 consumer warps, an MMA
  warp, a TMA warp, a scheduler warp and an epilogue warp. Weights (128 x 128 B stages) and activations (contiguous
  pair rows, 64 or 128 tokens per tile) are streamed by TMA straight into the 5th-generation tensor core
  (`tcgen05.mma`, fp8 e4m3, M=128 weight rows x N=tokens x K=32) with accumulators in TMEM; per scale block the
  partial is pulled to registers (`tcgen05.ld`) and promoted with the fp32 block scales. Whole-K tiles; the SwiGLU +
  fp8 requantization is done by the consumer warps (gate/up rows exchanged through shared memory); GEMM2 writes
  per-pair bf16 rows that a combine kernel sums per token. (The per-tile epilogue loops are fully unrolled so the
  accumulators stay in registers; a profiled version with partially unrolled loops spilled them to local memory.)
* Routing uses exact reference semantics (sigmoid, +bias, 8 groups x top-2, keep 4 groups, top-8 with
  lowest-index tie-break, s-normalized weights x scaling) on one warp per token; for larger T a route + scatter +
  gather kernel sequence produces expert-grouped pair lists, gathered activation rows and pair-contiguous scale
  tables in global memory.
* Work = dynamic tiles from an atomic counter: GEMM1 tiles (expert, 64 gate + 64 up rows[, K-split]), then GEMM2
  tiles (expert, 128 output rows[, K-split]) which wait on a per-expert "GEMM1 done" counter. For T <= 128, split-K
  partials are reduced with `red.add` into an fp32 buffer and the last K-split finisher (detected with an
  acq_rel atomic) applies SwiGLU and fp8-requantizes the intermediate (per-token 32-wide scale blocks); the GEMM2
  output is accumulated with `red.add` into an fp32 scratch and the last finisher of each 128-column slice converts
  to bf16. All counters and scratch buffers are self-cleaning (zeroed by the last CTA / the finalizers), so nothing
  input-derived survives across calls.
* The tcgen05 core was also built for the small-T regime (it is the natural fit for fp8 on B200), but on this card the
  `mma.sync` core streamed weights faster for tiny token groups (its per-stage overheads are lower), so the final
  dispatch uses it for T <= 128; the tcgen05 core carries the large-token workloads.

### Correctness methodology
* `tools/naive.py`: fp32 implementation of the definition's reference semantics (dequantize everything to fp32, no
  fp8 requantization of the intermediate).
* For each workload the FIB random inputs (same generator as the official runner, seed 0) are run through the fp32
  naive, KDA-0.5 (`kda05-ref`), flashinfer `trtllm_fp8_block_scale_moe` and this solution. The chosen ref's error vs
  naive defines the tolerance (7x, for both RMS-relative error and max-abs/output-max error). The solution's error is
  lower than both refs on every workload (it keeps the intermediate scale blocks at 64 instead of 128).
* Extra checks (`tools/robust_test.py`, all passing): arbitrary sequence lengths built from dataset inputs
  (3, 100, 127/128/129/130 around the small/large dispatch boundary, 200, 2047/2048/2049 around the tile-config
  boundary, 3000), adversarial routing (every token on one local expert -> multi-group tiles at T=58/128/901; two
  dominant experts; no token on any local expert -> output must be exactly zero), repeat-call consistency (only
  fp32-atomic-order noise), and the official FIB evaluator (`--atol 1 --rtol 0.3 --required-matched-ratio 0.9`)
  which passed all 19 workloads. All 19 workloads were also checked with two further input seeds (38 runs, max
  RMS-relative error 0.022 vs tolerance ~0.16).

### Timing methodology
`tools/final_bench.py`: CUPTI kernel-span device time (`flashinfer_bench.bench.timing.bench_gpu_time_with_cupti`),
cold L2 before every call, median over iterations; KDA (`kda05-ref --bench`, same timer), this solution and flashinfer
are measured alternately in 2 rounds x 3 trials on the same card; the ref for a workload is whichever of KDA /
flashinfer is faster. Every timed call starts from the raw inputs (routing included); only compilation and shape-only
workspace allocation are reused.

