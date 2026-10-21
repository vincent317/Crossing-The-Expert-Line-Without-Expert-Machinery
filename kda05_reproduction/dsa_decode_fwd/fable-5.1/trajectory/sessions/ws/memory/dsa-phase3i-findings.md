---
name: dsa-phase3i-findings
description: "Phase 3i (2026-10-21) of the DSA kernel: g51 (SPREAD_UNCOND) kept = official v26 103.73x; R1 register-resident tiny-tile path rejected after 3 iterations; F2b and KG_BIG=2 rejected; measured sm_100a latencies (mma.sync 22 cycles, ldmatrix 45, shfl 31, LDS.128 45, barrier 29, ex2 18; QK with 8 warps is smem-port bound at 65 cycles per 2-ldmatrix+mma step); the tiny-tile tail is a ~15-stage dependent chain that restructuring does not shorten"
metadata:
  node_type: memory
  type: project
  originSessionId: phase3i
---

Phase 3i of the DSA sparse-attention kernel (worktree `worktree-dsa-attn-dev`, 2026-10-21, after v25 = g50):

- **g51 = `DSA_SPREAD_UNCOND 1`** (commit a232482): the spread-mode tile-0 slot loads under the `spread_ok` branch had been
  sunk below the first barrier by the compiler (a second DRAM round trip for the sibling CTAs); issuing them unconditionally
  is -0.15 us mean, 22/23 faster. **Official v26 = g51: 103.73x / 6.01 us, 23/23 (best author score)**, packed_solution.json = v26.
- **Latency calibration** (`scripts/micro_mma_chain.py`, B200 sm_100a, cycles per dependent step): mma.sync m16n8k16 22
  (issue ~8 cycles per mma per SM sub-partition), ldmatrix.x4 -> mma 67, shfl 31, LDS.128 45, `__syncthreads` (8 warps) 29,
  ex2 18. The kernel's QK pattern (2 ldmatrix.x4 + 1 mma per k-step) = 24 cycles/step with 1 warp but 65 with 8 warps:
  16 x 512 B per step saturates the 128 B/clk shared-memory port. Empty-kernel launch under the evaluator's CUPTI timer is
  1.1 us (T=1 at 3.58 us = 1.1 launch + 0.16 first-CTA offset + 0.74 index DRAM latency + 0.7 gather latency + ~0.9 tail).
- **R1 rejected** (register-resident tiny-tile path for <= 2 / 4 n-tiles: warp w = dim-block w for all n-tiles, one smem
  reduction of 8 partials, warp-local softmax, P fragments built in registers as the PV A operand (C layout of n-tiles 2k,
  2k+1 = A layout of k-step k), Q fragments preloaded during the gather latency; archived `docs/kernel_r1_register_tiny.cu.txt`):
  T=1 3.57-3.66 vs 3.58, 3-n-tile tokens +0.1..0.27, and the NT<=2 build +0.2..0.4 us on big workloads that never execute
  it (fourth occurrence of "compiled-in code slows other paths"). Traces: the post-arrival tail is ~15 dependent stages
  (cp.async wait, ldmatrix, mma chain, barrier, LDS, shuffles, ex2, stmatrix, LDS, STG), each costing ~2x its instruction
  latencies with 2 warps per scheduler; moving work between warps or swapping a barrier for a register reduction changes
  the total by < 0.05 us. NCU record `profile/v26_g51` (T=1 base vs R1: 82.4K vs 83.2K instructions, warp cycles per issued
  instruction 8.66 vs 9.14).
- **F2b rejected**: pairing F2 warps (w, w+1) across two sub-partitions changed T=1 by 0.00 (the tensor pipe is not the
  limiter). **KG_BIG=2 rejected**: +0.3 us on every full-tile workload despite half the Q ldmatrix traffic (full-tile QK is
  not purely smem-bandwidth bound: 8-k-step chains over 4 accumulators + 2-block gather ownership lose more).
- **Gather floor confirmed** (`scripts/micro_gather2.py`, cold L2, 73 KB tile per CTA, indices staged in smem): cp.async with
  256 / 512 / 1024 threads, a TMA-bulk + cp.async hybrid and TMA bulk rows issued one lane per warp are all within 0.1-0.4 us
  of each other at 8 and 32 CTAs (cp.async 256 threads is best or tied); a micro whose issue loop loads the index per chunk
  shows TMA 2.5 us "faster" - an artifact. **Cross-SM visibility** (`scripts/micro_visibility.py`, all 148x148 SM pairs):
  st.relaxed.gpu -> ld.relaxed.gpu spin = 416 ns median (256-640), identical for every SM-id block / distance (no die or
  GPC locality on this B200); st.release + atom.add polling 896 ns. The kernel's level-2 cost (1.1-1.3 us) is visibility
  + a 0.6 us round trip for the 8.4 KB poll + phase alignment.
- Not started: S8b pair mode for 2-tile tokens (only e6b849f2, ~-0.25 us, new merge topology); one-barrier fused softmax
  (estimated <= -0.03 us).

**Why:** ~4 hours of A/B runs; the latency numbers and the "tail is a dependent chain" conclusion rule out further
re-partitioning of the tiny-tile compute.
**How to apply:** start from g51 (a232482; HEAD of `worktree-dsa-attn-dev` has the same kernel tree); do not re-split the
tiny-tile QK/softmax across warps; any new code path must be measured on all 23 workloads for the "compiled-in" penalty.
See [[dsa-phase3h-findings]], [[dsa-kernel-findings]], [[dsa-devspace-workflow]].
