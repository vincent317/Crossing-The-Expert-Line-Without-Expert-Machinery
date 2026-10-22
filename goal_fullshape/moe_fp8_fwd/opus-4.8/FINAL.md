# FINAL — MoE FP8 block-scale (DeepSeek-V3 routing) custom Triton kernel

Definition: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

Device time metric: CUPTI kernel span, cold L2, median — identical to `kda05-ref --bench` and `flashinfer.testing.bench_gpu_time`.

ref = fastest of {KDA-0.5 agent, flashinfer trtllm_fp8_block_scale_moe library} measured on THIS B200.

Correctness gate = element-wise matched-ratio vs the semantic reference under the official tolerance (atol=1, rtol=0.3) ≥ 0.9 (the same check the official FIB eval uses). The `tol (7×floor)` column is an auxiliary bound: 7× the max |KDA−fp32_naive| error (the fp8 quantization floor); mean-err stays far below it everywhere.


| seq_len | offset | ref | ref µs | my µs | ref/my | matched% (atol1,rtol0.3) | max err vs fp32 | mean err vs fp32 | tol (7×floor) | correct |
|--------:|-------:|:----|-------:|------:|-------:|-------------------------:|----------------:|-----------------:|--------------:|:-------:|
| 1 | 32 | KDA-0.5 | 61.6 | 94.0 | 0.66 | 97.01 | 11.00 | 2.264 | 77.9 | ✅ |
| 7 | 192 | KDA-0.5 | 94.3 | 152.1 | 0.62 | 99.10 | 16.00 | 0.701 | 98.0 | ✅ |
| 14 | 0 | KDA-0.5 | 130.8 | 175.4 | 0.75 | 98.87 | 13.88 | 0.825 | 87.5 | ✅ |
| 15 | 32 | KDA-0.5 | 83.4 | 146.7 | 0.57 | 99.30 | 16.50 | 0.562 | 103.2 | ✅ |
| 16 | 224 | KDA-0.5 | 130.3 | 187.5 | 0.69 | 98.52 | 13.50 | 1.045 | 101.9 | ✅ |
| 32 | 32 | KDA-0.5 | 200.9 | 278.8 | 0.72 | 98.87 | 15.25 | 0.859 | 102.4 | ✅ |
| 52 | 160 | KDA-0.5 | 164.5 | 251.8 | 0.65 | 99.25 | 11.00 | 0.513 | 77.0 | ✅ |
| 53 | 32 | KDA-0.5 | 224.5 | 287.1 | 0.78 | 98.81 | 17.00 | 0.868 | 119.0 | ✅ |
| 54 | 128 | KDA-0.5 | 204.4 | 281.0 | 0.73 | 98.91 | 13.50 | 0.799 | 87.5 | ✅ |
| 55 | 128 | KDA-0.5 | 224.6 | 282.1 | 0.80 | 98.73 | 19.00 | 0.919 | 105.0 | ✅ |
| 56 | 64 | KDA-0.5 | 236.1 | 297.5 | 0.79 | 98.50 | 13.50 | 1.058 | 94.5 | ✅ |
| 57 | 96 | KDA-0.5 | 229.6 | 290.0 | 0.79 | 98.69 | 15.00 | 0.953 | 98.0 | ✅ |
| 58 | 64 | KDA-0.5 | 233.1 | 298.3 | 0.78 | 98.41 | 18.00 | 1.159 | 122.5 | ✅ |
| 59 | 160 | KDA-0.5 | 188.3 | 261.1 | 0.72 | 98.98 | 14.00 | 0.711 | 94.5 | ✅ |
| 62 | 96 | KDA-0.5 | 170.3 | 260.7 | 0.65 | 98.84 | 14.00 | 0.856 | 96.2 | ✅ |
| 80 | 96 | KDA-0.5 | 258.4 | 354.4 | 0.73 | 98.38 | 16.50 | 1.204 | 115.5 | ✅ |
| 901 | 96 | KDA-0.5 | 298.5 | 473.7 | 0.63 | 98.27 | 20.00 | 1.252 | 137.1 | ✅ |
| 11948 | 128 | KDA-0.5 | 842.6 | 2377.8 | 0.35 | 98.83 | 302.00 | 0.835 | 150.5 | ✅ |
| 14107 | 32 | KDA-0.5 | 1170.5 | 3213.8 | 0.36 | 98.51 | 253.00 | 1.067 | 154.0 | ✅ |

**Geometric mean ref/mine over all 19 workloads: 0.658**
**All workloads correct (matched ≥ 0.9 vs semantic reference): True**

## Notes
- ref timings for both candidates were measured on this card; KDA-0.5 was fastest on every workload, so ref=KDA-0.5 throughout.
- fp32 naive reference (`dev/common.py::ref_fp32`) implements the definition's semantic reference exactly in float32.
- The solution computes from raw inputs every call (routing → grouped GEMM1 → SwiGLU → GEMM2 → weighted accumulate); the only reused state is the compiled CUDA graph / Triton binaries (shape+pointer keyed), never any data-derived result.
- For T=11948/14107 the `max err vs fp32` (302/253) exceeds 7×floor because <0.1% of tokens sit on routing near-ties where fp32-sigmoid rounding flips the 8th-vs-9th expert; those few token rows diverge (the same happens between KDA-0.5 and the reference). Mean err (0.8–1.1) and matched-ratio (98.5–98.8%) stay well within tolerance, so the official gate passes.
- Official evaluation: `flashinfer-bench run ... --use-isolated-runner ... --atol 1 --rtol 0.3 --required-matched-ratio 0.9` reports PASSED for all 19 workloads.

## Performance analysis & ceiling (why large-T is ~0.35)
- The small/mid-T workloads (T≤901) land at 0.57–0.80. The two large-T workloads (11948/14107) are the only real drag at 0.35–0.36 and dominate the geomean.
- Both GEMMs are compute-bound there. ncu on gemm1 (T=14107, config 64×128, 4 warps, 3 stages): 168 registers/thread → occupancy limited to 3 blocks/SM (shared memory co-limits to 3 as well) → 17.3% achieved warp occupancy; DRAM at 21% (not memory-bound); the legacy HMMA counters read n/a, confirming Triton emits Blackwell **tcgen05** (5th-gen tensor-core) MMA. Achieved throughput ≈ 19% of B200 fp8 peak vs KDA-0.5 (CUTLASS) ≈ 23%.
- The remaining ≈1.2–1.5× gap on large-T is CUTLASS's warp-specialized *persistent* tcgen05+TMA pipeline, which sustains high MMA utilization at low occupancy — a structure Triton does not expose. The block scales here are 128×128 arbitrary fp32, which cannot use the hardware block-scaled MMA path (`tl.dot_scaled` only supports MX: group-32, e8m0 power-of-2 scales), so both this kernel and CUTLASS must rescale per 128-K-block in-SM; the difference is purely the pipeline.
- Optimization space explored and ruled out (see `dev/AUTOTUNE_RECORD.md`): BM∈{64,128}/BN∈{64,128,256}/num_warps∈{4,8}/num_stages∈{2,3,4,5,6} autotune; L2 program-id swizzle (GROUP_M); SwiGLU fusion into the gemm1 epilogue (occupancy collapse from a 2nd accumulator); split-K (fixed per-block overhead × SPLIT + atomics); bf16/fp16 accumulator (tcgen05 keeps fp32 TMEM regardless → no register saving); warp_specialize; block_ptr/TMA; pre-gather. None beat the shipped config.
- Robustness: correct (matched ≥ 0.9 vs fp32 naive) across 158 off-dataset configs — 108 random (routing/scale/offset) + 50 adversarial edge (all-equal-logit ties, tiling-boundary token counts 63/64/65/127/128/129/256/257, group-dominant routing, extreme scales).

## Implementation summary (solution/main.py, from scratch in Triton)
1. `route_kernel` — fused DeepSeek-V3 no-aux routing (sigmoid+bias, per-group top-2 → top-4 groups → global top-8, normalized weights ×routed_scaling_factor) one program/token; also counts local-expert assignments via atomics.
2. `offsets_kernel` — one program/block builds the padded per-expert offsets + block→expert map (replaces a ~6-op torch cumsum/searchsorted chain).
3. `place_kernel` — scatters each (token,expert) pair into the grouped/padded token buffer.
4. `gemm1_kernel` — grouped fp8 block-scale GEMM (hidden × W13ᵀ) with 128×128 block dequant, bf16/fp8 tensor-core dot, fp32 accumulate → bf16 intermediate.
5. `swiglu_kernel` — silu(up)·gate then per-128-block fp8 requantization.
6. `gemm2_kernel` — grouped fp8 block-scale GEMM (× W2ᵀ), ×routing-weight, atomic-add scatter straight into the bf16 output.
The whole pipeline is captured once per (shape, input-pointers) into a CUDA graph and replayed (falling back to eager when the caller is itself capturing). No cuBLAS/CUTLASS/flashinfer/torch-matmul GEMM is used.