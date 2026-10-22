# MoE FP8 block-scale, DeepSeek routing — /goal-mode kernels (B200)

Final kernels produced by interactive Claude Code `/goal` sessions, each written from
scratch without reading the reference kernel. Only the operator code (kernel + launcher)
is included; benchmarks, probes and test harnesses are omitted.

Operator: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` — DeepSeek-V3
routing (sigmoid + bias, top-2 per group, 4 of 8 groups kept, top-8 experts,
`routed_scaling_factor` 2.5) over 256 global experts of which 32 are local, hidden 7168,
intermediate 2048, FP8 e4m3 weights with 128x128 block scales and activations with
128-wide scales along K, two GEMMs with SwiGLU in between, bf16 output `[T, 7168]`.
The three cases are flashinfer-trace workloads `e05c6c03` (`t1`, decode, local expert
offset 32), `1a4c6ba1` (`t901`, offset 96) and `5e8dc11c` (`t14107`, offset 32).

The reference is the KDA 0.5 MoE kernel from the MLSys 2026 FlashInfer contest solution
(`mit-han-lab/mlsys2026-flashinfer-contest-solution`, commit `7aad0429`; published
49.05 / 278.29 / 1042.65 µs), installed on the session's B200 as a black-box command
(`kda05-ref`) that could only be run. It was the fastest available implementation on
all three cases (FlashInfer `trtllm_fp8_block_scale_moe`: 94+ / 377 / 2484+ µs measured
or published; the human champion's kernel was not available). The goal was 1.2x the
reference's device time.

Each case ran the same prompt with three models at the same time, each on its own B200,
with a clean Claude Code config (no skills, no memory) and a guard hook that refused
reads of the reference, of library MoE/GEMM sources and of the acceptance tool. The
prompt also forbade caching anything derived from the inputs across calls (weights
included) and calling library GEMM/MoE kernels.

Device time and the verdict come from a separate acceptance tool (`moe-accept`, written
by us, run on the session's node; the session could run it but not read it): CUPTI
kernel span with cold L2 (`flashinfer.testing.bench_gpu_time`), median per trial; the
reference and the candidate are timed alternately, 2 rounds x 3 trials, on the same card,
and every timed candidate call gets freshly allocated routing/hidden buffers.
Correctness is checked against an fp32 naive implementation with the tolerance set per
case to 7x the reference's own max-abs and mean-abs error against that naive, over
several routing cases (workload routing, 0/2/8 hit local experts, random), and again
after every input, weights included, is overwritten in place. `vs reference` =
reference / candidate. The numbers below are our re-runs of `moe-accept` after the
sessions ended. Develop time is from the session's start to the version kept here.

| case | model | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|---|
| `t1` 单 token decode (T=1; the workload hits 3 local experts) | Fable-5.1 | `t1/fable-5.1/moe_kernel.cu` (+ ctypes launcher `kernel.py`) | 50.44 µs | 60.75 µs | 1.204x ¹ | 4.1 h |
| | Opus-5 | `t1/opus-5/moe19.cu` (prebuilt `moe.cubin`, build flags in `mkout.sh`; driver-API launcher `kernel.py`) | 44.14 µs | 54.22 µs | 1.228x | 2.4 h |
| | Opus-4.8 | `t1/opus-4.8/kernel.py` (Triton) | 44.34 µs | 54.00 µs | 1.218x | 1.9 h |
| `t901` 中等 batch (T=901) | Fable-5.1 | `t901/fable-5.1/moe.cu` + `common.cuh` (raw PTX tcgen05/TMA; JIT launcher `kernel.py`) | 266.8 µs | 271.9 µs | 1.019x | 8.9 h |
| | Opus-5 | `t901/opus-5/moe_kernels.cu` (raw PTX tcgen05 + CuTe headers; JIT launcher `kernel.py`) | 318.9 µs | 276.4 µs | 0.867x | 8.0 h |
| | Opus-4.8 | `t901/opus-4.8/kernel.py` (Triton) | 621.6 µs | 276.7 µs | 0.445x | 2.1 h ² |
| `t14107` 大 prefill (T=14107) | Fable-5.1 | `t14107/fable-5.1/kernel.py` (Triton + Gluon tcgen05, inline PTX) | 884.8 µs | 1033.3 µs | 1.168x ³ | 7.5 h |
| | Opus-5 | `t14107/opus-5/kernel.py` (Triton GEMMs; CUDA routing `route.cu` + `route_host.cu`) | 915.2 µs | 1033.4 µs | 1.129x | 9.9 h |
| | Opus-4.8 | `t14107/opus-4.8/kernel.py` (Triton) | 2069.3 µs | 1034.8 µs | 0.500x | 2.0 h ² |

All kernels pass the correctness check. Only the three `t1` rows reached the 1.2x goal.

¹ On the line, not a stable pass: three further re-runs gave 1.195 / 1.200 / 1.197
(the candidate is steady at ~50.5 µs; the reference on that card is bimodal, 56–62 µs).

² The Opus-4.8 `t901` and `t14107` sessions concluded the goal was out of reach and
idled; they were stopped about 2.5 h after their last improvement.

³ The session itself reported 1.239x PASS, which is void: its card was power-throttled
(idle SM clock 1155 MHz at 155 W), which slowed the reference to ~1540 µs. The row is a
re-run on the Opus-5 session's (normal) card.

Entry point (all nine): `kernel.py` defines `run(routing_logits, routing_bias,
hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights,
gemm2_weights_scale, local_expert_offset, routed_scaling_factor)` returning bf16
`[T, 7168]`; inputs are logits f32 `[T,256]`, bias bf16 `[256]`, hidden fp8 `[T,7168]`,
hidden scale f32 `[56,T]`, W13 fp8 `[32,4096,7168]` with f32 scales `[32,32,56]`, W2 fp8
`[32,7168,2048]` with f32 scales `[32,56,16]`.

Notes

- `t1`: the workload routes to only 3 local experts, so all three kernels are HBM
  streaming of those experts' weights. Fable-5.1 is one persistent warp-specialized
  kernel (TMA ring, f16 `mma.sync` after `cvt` from e4m3, K-split partials published as
  tagged records with a per-launch epoch); Opus-5 is one cooperative kernel with no tensor
  cores (`__hfma2` dot products, per-expert-group spin barrier between the GEMMs);
  Opus-4.8 is three Triton GEMV-style kernels captured into a CUDA graph.
- `t1/opus-5/kernel.py` loads `moe.cubin` from its own directory and launches it with
  `cuLaunchCooperativeKernel`; `mkout.sh` documents how it was built
  (`nvcc -arch=sm_100a -cubin -O3 -DNT=512 -DBPSM=2 -DRSEL=3 -DWIN=7`) but hard-codes the
  session's directory layout. Its engine uses device 0.
- `t1/opus-4.8` re-captures its CUDA graph on every call (host-side, outside the timed
  span) and returns a persistent output buffer, so each call overwrites the previous
  result.
- The two `t1` CUDA sessions locked the node's GPU clocks while they worked (1500 MHz and
  1965 MHz); reference and candidate were always timed on the same card under the same
  clocks.
- `t901/fable-5.1` is a single persistent kernel (148 CTAs x 640 threads) with in-kernel
  routing, an atomic work queue of GEMM1/GEMM2 tiles, a 5-stage TMA pipeline, tcgen05
  `kind::f8f6f4` M128 N256 K32 MMAs into double-buffered TMEM, and DeepGEMM-style fp32
  promotion of the block scales. Built with `torch.utils.cpp_extension.load`
  (`-gencode arch=compute_100a,code=sm_100a`).
- `t901/opus-5` issues six kernels with programmatic dependent launch (routing, permute,
  grouped tcgen05 GEMM1 with 2-CTA A-tile multicast, SwiGLU + fp8 requant, GEMM2, combine).
  It needs CUTLASS/CuTe headers at `$CUTLASS_DIR` (default `/root/cutlass/include`). Its
  best score was 0.890x at one point, but that version had an mbarrier phase-lapping race
  that could deadlock; the race-free version is the one kept here.
- `t901/opus-4.8` and `t14107/fable-5.1` cache their CUDA graphs keyed on the weights'
  `data_ptr()`; the graphs read the weight values live, so in-place overwrites are still
  handled (the acceptance tool checks this). `t901/opus-4.8` does a host sync per call to
  pick a graph bucket.
- `t14107/fable-5.1` regex-parses its own compiled PTX at first launch to find mbarrier and
  TMEM-slot shared-memory offsets; it was written against Triton 3.7.1 and is fragile
  across Triton versions. It returns outputs from a 4-deep ring of persistent buffers.
- `t14107/opus-5` captures everything after routing into a CUDA graph keyed only on the two
  scalars, reading the inputs through a device pointer table that the eager routing kernel
  fills on every call; it returns the same persistent output buffer on every call.
- Both `t14107` tcgen05/Triton kernels accumulate the output with bf16 atomics, so results
  are not bit-deterministic (well inside the tolerance).
- `t14107/opus-4.8` allocates every intermediate per call and has no CUDA graph, so its
  span includes the launch gaps of ~12 kernels.
- The prompts (in Chinese) are identical within a case except for the GPU node's ID.
