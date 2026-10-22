# FP8 block-scale MoE (DeepSeek-V3 routing) — final results

Definition: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

Device: single NVIDIA B200 (sm_100a). Timing = CUPTI kernel span, cold L2, median over iterations (`flashinfer.testing.bench_gpu_time(enable_cupti=True, cold_l2_cache=True)`), identical methodology to `kda05-ref --bench`. ref and ours measured alternately on the same card, nine rounds each, median of the rounds reported for both (the minimum is too sensitive to a single lucky round).

Tolerance: for every workload a slow fp32 naive implementation is the ground truth; `tol = 7 x max|ref - naive|`, and we require `max|ours - naive| <= tol`.

| seq_len | ref | ref us | ours us | ref/ours | max err vs naive | tol | pass |
|---:|:--|---:|---:|---:|---:|---:|:-:|
| 1 | KDA-0.5 | 54.4 | 65.9 | **0.83x** | 0.00993 | 0.07308 | PASS |
| 7 | KDA-0.5 | 84.4 | 99.6 | **0.85x** | 0.01207 | 0.09894 | PASS |
| 14 | KDA-0.5 | 117.5 | 141.1 | **0.83x** | 0.01348 | 0.10416 | PASS |
| 15 | KDA-0.5 | 74.7 | 83.2 | **0.90x** | 0.01302 | 0.08261 | PASS |
| 16 | KDA-0.5 | 121.5 | 143.9 | **0.84x** | 0.01209 | 0.08460 | PASS |
| 32 | KDA-0.5 | 188.5 | 220.3 | **0.86x** | 0.01530 | 0.09492 | PASS |
| 52 | KDA-0.5 | 153.8 | 173.2 | **0.89x** | 0.01020 | 0.08030 | PASS |
| 53 | KDA-0.5 | 212.0 | 226.5 | **0.94x** | 0.01446 | 0.09915 | PASS |
| 54 | KDA-0.5 | 194.8 | 222.9 | **0.87x** | 0.01315 | 0.09501 | PASS |
| 55 | KDA-0.5 | 212.0 | 229.3 | **0.92x** | 0.01170 | 0.08732 | PASS |
| 56 | KDA-0.5 | 221.8 | 244.2 | **0.91x** | 0.01253 | 0.09047 | PASS |
| 57 | KDA-0.5 | 215.6 | 237.1 | **0.91x** | 0.01446 | 0.09056 | PASS |
| 58 | KDA-0.5 | 221.2 | 243.6 | **0.91x** | 0.01411 | 0.09849 | PASS |
| 59 | KDA-0.5 | 177.3 | 201.1 | **0.88x** | 0.01302 | 0.09242 | PASS |
| 62 | KDA-0.5 | 160.9 | 180.5 | **0.89x** | 0.01514 | 0.12041 | PASS |
| 80 | KDA-0.5 | 245.3 | 261.7 | **0.94x** | 0.01520 | 0.09565 | PASS |
| 901 | KDA-0.5 | 276.2 | 345.0 | **0.80x** | 0.02037 | 0.12419 | PASS |
| 11948 | KDA-0.5 | 743.5 | 1039.5 | **0.72x** | 0.02085 | 0.13736 | PASS |
| 14107 | KDA-0.5 | 1040.6 | 1479.7 | **0.70x** | 0.02819 | 0.18404 | PASS |

**Geometric mean speedup over the per-workload fastest reference: 0.860x** (19 workloads, all correct: True)

## Reference candidates measured on this card

| seq_len | KDA-0.5 us | flashinfer trtllm us | fastest |
|---:|---:|---:|:--|
| 1 | 54.4 | 82.9 | KDA-0.5 |
| 7 | 84.4 | 114.6 | KDA-0.5 |
| 14 | 117.5 | 153.7 | KDA-0.5 |
| 15 | 74.7 | 116.7 | KDA-0.5 |
| 16 | 121.5 | 162.2 | KDA-0.5 |
| 32 | 188.5 | 219.0 | KDA-0.5 |
| 52 | 153.8 | 189.7 | KDA-0.5 |
| 53 | 212.0 | 238.5 | KDA-0.5 |
| 54 | 194.8 | 238.1 | KDA-0.5 |
| 55 | 212.0 | 240.4 | KDA-0.5 |
| 56 | 221.8 | 250.3 | KDA-0.5 |
| 57 | 215.6 | 247.7 | KDA-0.5 |
| 58 | 221.2 | 249.0 | KDA-0.5 |
| 59 | 177.3 | 203.5 | KDA-0.5 |
| 62 | 160.9 | 210.7 | KDA-0.5 |
| 80 | 245.3 | 265.9 | KDA-0.5 |
| 901 | 276.2 | 363.3 | KDA-0.5 |
| 11948 | 743.5 | 1828.6 | KDA-0.5 |
| 14107 | 1040.6 | 2490.4 | KDA-0.5 |

Team Wombat (human, public leaderboard) reports 354 us at seq_len=901; no runnable implementation was available, and KDA-0.5 (275 us here) is faster, so KDA-0.5 is the reference at that shape.

## Error detail (relative to the fp32 naive reference)

| seq_len | ref max | ref rms | ours max | ours rms |
|---:|---:|---:|---:|---:|
| 1 | 0.01044 | 0.00252 | 0.00993 | 0.00252 |
| 7 | 0.01413 | 0.00167 | 0.01207 | 0.00153 |
| 14 | 0.01488 | 0.00159 | 0.01348 | 0.00159 |
| 15 | 0.01180 | 0.00127 | 0.01302 | 0.00126 |
| 16 | 0.01209 | 0.00160 | 0.01209 | 0.00159 |
| 32 | 0.01356 | 0.00152 | 0.01530 | 0.00152 |
| 52 | 0.01147 | 0.00109 | 0.01020 | 0.00109 |
| 53 | 0.01416 | 0.00151 | 0.01446 | 0.00151 |
| 54 | 0.01357 | 0.00145 | 0.01315 | 0.00146 |
| 55 | 0.01247 | 0.00153 | 0.01170 | 0.00154 |
| 56 | 0.01292 | 0.00157 | 0.01253 | 0.00158 |
| 57 | 0.01294 | 0.00159 | 0.01446 | 0.00159 |
| 58 | 0.01407 | 0.00172 | 0.01411 | 0.00172 |
| 59 | 0.01320 | 0.00136 | 0.01302 | 0.00136 |
| 62 | 0.01720 | 0.00148 | 0.01514 | 0.00148 |
| 80 | 0.01366 | 0.00179 | 0.01520 | 0.00178 |
| 901 | 0.01774 | 0.00181 | 0.02037 | 0.00181 |
| 11948 | 0.01962 | 0.00144 | 0.02085 | 0.00144 |
| 14107 | 0.02629 | 0.00163 | 0.02819 | 0.00164 |

## Implementation

Everything is written from scratch for this operator: no GEMM or MoE library
is called (no cuBLAS/cuBLASLt, no `torch.mm`/`matmul`/`_scaled_mm`/
`_grouped_mm`, no flashinfer/trtllm/DeepGEMM/sglang/vllm, no CUTLASS
device-level templates). The routing kernel is CUDA C++; the two GEMMs are
hand-written Triton kernels, because fp8 `mma.sync` turns out to be *emulated*
on sm_100 (see below) while Triton's fp8 `dot` lowers to native `tcgen05`.

### Kernels

1. **`routing_warp_kernel`** (CUDA C++, `solution/moe_kernel.cu`) — one *warp*
   per token, so there is not a single block barrier and the latency for short
   sequences is a few hundred cycles. Lane `l` holds expert `g*32+l` for each
   of the eight groups, which makes the per-group top-2 a butterfly over the
   warp, the top-4 group selection a register-local 8-way scan, and the global
   top-8 a sequence of eight packed-key `(value:index)` warp max-reductions —
   ties resolve to the lower expert index, matching `topk`. Selected local
   experts are appended to per-expert token buckets with one `atomicAdd` each,
   and the normalised routing weight (from the *unbiased* sigmoid score, times
   `routed_scaling_factor`) is stored next to the bucket slot.

2. **gemm1** — persistent, one work item per (expert, m-tile, n-tile). The
   work item index is turned into (expert, m-tile, n-tile) from the per-expert
   counts with an in-kernel cumulative sum, so no host round trip and nothing
   data dependent crosses the call boundary. Two variants are tuned per shape:

   * *fused* (`gemm1_kernel`, `gemm1ft_kernel`): one work item computes the
     gate and up halves of the same intermediate columns, applies SwiGLU
     (`silu(up) * gate`), re-quantises to fp8 with a per-(slot, n-tile) scale,
     and also zeroes the output tensor (it runs before gemm2, so the kernel
     boundary orders the writes). Best for the short sequences, which are
     bandwidth bound and want the fewest kernels.
   * *split* (`gemm1t_kernel` + `swiglu_kernel`): gate and up are separate work
     items writing bf16 into `G1`, and a third kernel does SwiGLU plus the fp8
     re-quantisation. Twice the work items and half the live accumulators per
     item, which wins for the two long sequences.

3. **gemm2** (`gemm2_kernel`, `gemm2t_kernel`) — persistent, one work item per
   (expert, m-tile, hidden-tile); same block-scaled fp8 accumulation,
   multiplied by the routing weight and accumulated into the bf16 output with
   `bfloat162` atomics. It also clears the counter buffer that the *next* call
   will use, so no per-call memset kernel is needed (the two buffers ping-pong
   on call parity).

Both GEMMs accumulate in fp32 with a *promotion per 128-deep quantisation
block*: `acc += dot(a, w) * (a_scale * w_scale)`, which is the finest
granularity the format requires and the coarsest that is correct.

### What actually moved the numbers

* **`mma.sync ... e4m3` is emulated on sm_100.** ptxas expands each one into
  `F2FP.F16.E4M3.UNPACK_B` conversions plus `HMMA.16816.F32`, so a hand-written
  `mma.sync` pipeline pays 64 conversion instructions per 8 logical MMAs per
  warp per k-block. A complete hand-written CUDA version of both GEMMs
  (`cp.async` staging, XOR-swizzled shared memory, `ldmatrix`, register
  accumulators — still in `solution/moe_kernel.cu`) is correct but about 2x
  slower for that reason. A CUDA-core (FFMA/`hfma2`) variant is also included
  (`solution/moe_ffma.cuh`); it carries no m-padding and needs no shared-memory
  staging, but it is memory-latency bound at these shapes and did not win
  either.

* **TMA for the dense operands.** Device-side `tl.make_tensor_descriptor` over
  `gemm1_weights` viewed as a single `[E*2I, H]` matrix, `gemm2_weights` as
  `[E*H, I]`, and `Cq`/`G1` indexed by compacted slot. Only the activation rows
  stay on the plain load path, because they are a gather over token ids. This
  took gemm1 at seq_len 11948 from 970 to 843 us and gemm2 from 584 to 489 us.

* **Software-pipelining the block scales.** With TMA in place, Nsight Compute
  attributed ~45% of the warp stalls to L1TEX scoreboard dependencies — the
  *tiny* per-k-block scale loads, which the Triton pipeliner does not prefetch
  because they do not feed the `dot`. Carrying them one iteration ahead by hand
  took gemm1 from 843 to 694 us. The ordering inside the loop body matters as
  much as the prefetch itself: the `dot` operands have to be issued first or
  the pipeliner schedules the whole body badly.

* **Occupancy, not MMA shape, caps the long sequences.** `BM=128` doubles the
  useful work per `tcgen05` instruction (M=64 wastes half the tensor pipe) but
  pushes the kernel to 200 registers and one block per SM, and measures ~20%
  *slower*. `num_ctas=2` (which would give M=128 across a CTA pair) fails to
  compile for these kernels, warp specialisation (`tl.range(warp_specialize=…)`)
  is slower, and so is a second accumulator chain. The long sequences therefore
  sit at ~40% tensor-pipe occupancy.

* **Counters are double buffered** so a call needs no separate zeroing kernel,
  and the Triton kernels are launched through their cached `CompiledKernel`
  rather than the JIT front end. Each extra kernel in the call costs ~3.5 us of
  CUPTI span, which is 2.4% of the shortest workloads.

* **Per-shape configuration.** Tile sizes, warp counts, pipeline depth and
  which kernel variant runs are chosen per sequence length by a staged search
  (`harness/autotune.py`), validated against the fp32 naive output before being
  timed. Any sequence length that is not in the table falls back to a range
  table, so the solution is correct (and reasonable) for every shape.

### Where the time goes

The weights of every *active* local expert must be read once per call:
29.36 MB of `gemm1_weights` plus 14.68 MB of `gemm2_weights` per expert. The
measured ceiling for streaming reads on this card is 7.0 TB/s, and about
5.4 TB/s with the work-item counts these shapes produce, which is what bounds
every sequence length up to 901. The two longest sequences are instead compute
bound: 1.42 PFLOP of fp8 MACs at seq_len 14107, against a tensor-core peak that
only the M=128 MMA shape can reach.

## Official FlashInfer-Bench evaluation

```
FIB_DATASET_PATH=/root/fib-dataset /opt/sglang/bin/flashinfer-bench run \
  --local /root/fib-dataset \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --save-results --use-isolated-runner --log-level INFO --resume \
  --timeout 300 --atol 1 --rtol 0.3 --required-matched-ratio 0.9
```

`19/19` workloads **PASSED** for this solution (`handwritten_triton_cuda`); the table below is the speedup each solution reports against the definition's mathematical reference, as printed by the harness.

| workload | ours | flashinfer wrapper | ours status |
|:--|---:|---:|:-:|
| b8f4f012 | 104.9x | 68.1x | PASSED |
| e05c6c03 | 184.5x | 73.9x | PASSED |
| 6230e838 | 64.8x | 71.7x | PASSED |
| 8f1ff9f1 | 73.7x | 64.9x | PASSED |
| 1a4c6ba1 | 60.6x | 51.4x | PASSED |
| a7c2bcfd | 79.6x | 60.1x | PASSED |
| 2e69caee | 93.3x | 70.0x | PASSED |
| 8cba5890 | 83.1x | 66.0x | PASSED |
| 5e8dc11c | 33.0x | 18.7x | PASSED |
| 58a34f27 | 36.7x | 20.2x | PASSED |
| 5eadab1e | 73.5x | 56.1x | PASSED |
| eedc63b2 | 66.2x | 56.0x | PASSED |
| e626d3e6 | 62.8x | 54.7x | PASSED |
| 74d7ff04 | 64.7x | 57.0x | PASSED |
| 4822167c | 63.9x | 55.6x | PASSED |
| 81955b1e | 64.8x | 55.8x | PASSED |
| 76010cb4 | 68.8x | 53.7x | PASSED |
| fc378037 | 65.2x | 55.2x | PASSED |
| f7d6ac7c | 84.5x | 58.2x | PASSED |


