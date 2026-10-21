---
name: moe-project-state
description: "Where the MoE FP8 block-scale kernel work stands (best candidate, official score, what is next) — updated session 8"
metadata:
  node_type: memory
  type: project
  originSessionId: c336ce7b-a67f-47aa-b2e9-9c8e4589a464
  modified: 2026-10-21T08:00:00.000Z
---

MoE task (`moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`), worktree branch `worktree-moe-fable51`
under `.claude/worktrees/moe-fable51`. Session 1 v22 (1.580 official), session 2 v54 (1.864), session 3 v63b (2.025).
**Session 4 (2026-10-20): official `fable51_moe_v67` 19/19 PASS, mean 2.076** (commit b828cc4: v66 CTA-wide
in-kernel routing + v67 early weight issue); the confirmation run `fable51_moe_v69` of the final source state (v67 +
v69 adaptive A box, dormant switches; commit 0abbf2c+) passed 19/19 with mean 2.072 (noise vs 2.076) - shipped candidate.
Dev numbers (v67): T=1 43.2 / 7 68 / 16 108 / 32 165 / 53 180 / 80 208 / 901 279 / 11948 680 / 14107 968.

Session-4 patch scripts `scripts/patch_v65.py` .. `patch_v72.py` (env switches in `main.py`): v65 work-conserving
two-queue scheduler + split targets (MOE_SCHED_WC / MOE_S1_MULT / MOE_S2_MULT, off), v66 MOE_ROUTE_CTA (on), v67
MOE_EARLY_W (on), v68 MOE_SPEC_PF (off), v69 MOE_A_BOX (on, neutral), v70 2-SM UMMA regenerated from the 1-SM kernel
(MOE_U2, off, correct but slower), v71 128x128 GEMM2 tiles / 4 TMEM buffers (MOE_G2_N128, off, slower), v72 per-warp
acc_empty release (-DMOE_ACC_WARP_RELEASE, neutral). Microbenchmark `scripts/mc_bench.cu` (cluster TMA feed).

**Session 5 (2026-10-20, Phase 7):** kept v81 (redundant CTA-wide routing for T=1, no grid barrier: 43.5 -> 41.9 us)
and v80 (large-path output zero-fill moved into UMMA GEMM1's idle warp 3: 14107 967 -> 956, 11948 682 -> 672);
**official `fable51_moe_v82` 19/19 PASS, mean 2.0784** (v67 2.0761; commit 59320c2) = shipped candidate; T=1 41.3 / 11948 674 / 14107 960. Rejected with evidence:
v73 GEMM2 wait-time L2 prefetch + prefetch-ahead, v74 warp-3 B-line prefetch (all mechanisms), v75 bulk-store GEMM2
epilogue, v76 coalesced GEMM2 epilogue, v77 next-unit scale prefetch, T=901 on the fused path (941 us); neutral: v79
early TMEM release, v82 scatter unroll. Tools added: `scripts/umma_chain.cu` (chain microbench with modes),
`scripts/ffma_bench.cu`, ablation defines `MOE_ABL_BHOT/AHOT/G1NOEPI/G2NOEPI`, `patch_v73.py`…`patch_v81.py`.

**Session 6 (2026-10-21, Phase 8):** v83 hardware block-scaled UMMA large path (`umma_mx_kernel`, `MOE_MX` switch, default off; commit 758ef10): correct but 1140/1469 us at 14107 vs 960 -> rejected, see [[moe-b200-perf-findings]]. Tools added: `scripts/mx_test.cu`, `mx_test2.cu` (standalone block-scaled pipeline with ablations and throughput mode), `mx_numerics.py` (torch numerics-budget simulation with the official metric), `mx_debug*.py`, `dg_calib.py` (DeepGEMM calibration), `ncu_helpers/pipes_stalls.py`. v84 TMA gather4 A tiles (`MOE_A_GATHER=2`): correct but GEMM1 2.8x slower -> rejected. Tiny-T split sweep: all slower. Shipped candidate still v82 (2.078); official confirmation run `fable51_moe_v84` of the final source: 19/19 PASS, mean 2.0790 (commit fe312d0 + records).

**Session 7 (2026-10-21, Phase 9):** v85 continuous unit stream, v86 speculative first wave, 901 with 128-column GEMM2 tiles - all rejected (commit d84981c records them; MOE_SPEC_W default 0).

**Session 8 (2026-10-21, Phase 10):** kept v90 (UMMA GEMM1 epilogue SwiGLU once: 901 280 -> 275, 11948 672 -> 663, 14107 955 -> 942) and v92 (UMMA GEMM2 epilogue 256-bit stores: 11948 -5, 14107 -6..-8); rejected with evidence: v87/v87c/v87d grid-fit and GEMM2 split-K at tiny T (dormant `MOE_SPLIT_FIT` bits), v88 GEMM2 epilogue smem handoff to warps 2-3 (`MOE_G2_HANDOFF`), v89 output cache hints (`MOE_OUT_STREAM`), v91 tail split-K (`MOE_TAIL_SPLIT=0`), suspend-hint waits (`MOE_MBAR_HINT`). **Shipped candidate: v94 = session-6 source (fe312d0) + v90 + v92, official `fable51_moe_v94` 19/19 PASS, mean 2.0848** (commit 0ae288c+). Lesson: dormant switch code in the fused tiny-T kernel cost 0.5-1.5 us per small workload (official v92 2.059, v93 2.074) -> rejected experiments must stay out of the shipped source (archived in git 3d31a26 / scripts/kernel_s8_experimental.cu.txt). Tools: whole-grid fused traces (`-DMOE_TRACE_CTAS=512`), `bw_bench x`, ablation defines `MOE_ABL_G2META/NORED/G2ONECHUNK/G2SAMEADDR`, `scripts/remote_s8_ptxas.sh`.

Records: `benchmark.csv`, `solutions.jsonl`, `docs/draft.md` (Phase 6 + final state), `ncu/README.md` (session-4
table), `docs/writeup.md` (results = v67 official).

**Why:** the next session should not retry: tiny-T split-K restructuring (grid-fit / GEMM2 splits), GEMM2 epilogue handoff / cache hints / tail split-K (session 8); the hardware block-scaled (MX) UMMA path with residual folding, TMA gather4 for the A rows, finer split-K / two-queue scheduling for tiny T, speculative weight
prefetch, 2-SM UMMA, smaller UMMA tiles with more TMEM buffers, barrier arrival-count tweaks — all measured.

**How to apply:** remaining ideas with evidence behind them: warp-3 smem prefetch of the GEMM2 epilogue row metadata (G2META ablation says -24/-33 us at 11948/14107 on the v94 source; register hoisting v95 failed); copy the A rows inside UMMA GEMM1 (per-m-tile readiness
counters) to overlap the ~45 us scatter copy at 11948/14107 (+0.01 mean); T=901 host-launch gaps (7.6 us) via a CUDA
graph of the large path (+0.002); a 2-SM 256x128 UMMA tile with 4 TMEM buffers (untested, high risk); fusing UMMA
GEMM1+GEMM2 (tail overlap, ~2 %). Do not retry: any L2 prefetch, epilogue store restructuring, scale prefetch,
release-point changes, T=901 on the fused path. The large path's per-K-block cost is
structural (see [[moe-b200-perf-findings]]); see [[moe-devspace-environment]] for the tooling.
