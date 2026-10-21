---
name: moe-fp8-b200-state
description: State and ceiling analysis of the B200 MoE FP8 block-scale kernel project
metadata:
  node_type: memory
  type: project
  originSessionId: 15dbccec-a4e6-425f-88ce-cc4a9aeb6d74
  modified: 2026-10-21T00:39:54.802Z
---

MoE kernel `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` on B200/CUDA 13.
Work lives in the git worktree `.claude/worktrees/moe-fp8-b200` (branch
`worktree-moe-fp8-b200`), NOT on main. Dev loop: edit `solution/python/{main.py,moe_kernels.cu}`
→ `bash scripts/sync.sh` → devspace `tsh ssh root@gpu-node-1295754`
`/root/moe_dev` (deployed files flattened to `solution/{main.py,moe_kernels.cu}`;
`fib_dev.py test dev|all`, `run_official.sh`, `extract_results.py`, `profile_real.py`).
`/opt/sglang/bin/python`, flashinfer 0.6.18. Long runs in tmux. On macOS host there is no
`timeout` cmd and the Bash tool caps at 120s — poll remote tmux logs instead of `sleep`-ing.

**CURRENT BEST: official score 1.6030 (solution content md5 main.py a58949f0 /
moe_kernels.cu e4441300, committed at c72426d; later benchmark.csv-only commits do not
change solution), ALL 19 PASS, 19/19 beat baseline.** Solution `moe_p3_fastrt`.
Progression: 1.1672 → D7 1.5438 → D8 1.5921 → D10+D11 1.6028 → reconfirm 1.6030.

**BREAKTHROUGH that overturned the old "near-ceiling 1.1672" conclusion (D7):** the
prior analysis only ever profiled our OWN 8-kernel decomposed graph and (wrongly)
concluded it was at the fp8 weight-BW floor. The winning lever: **CUDA-graph-capture the
baseline's own fully-fused kernel** `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`
(routing+GEMM1+SwiGLU+GEMM2+combine all internal; baseline runs it EAGERLY). Capturing it
for T≤2048 and replaying beats our decomposition 0.54–0.87× device time/workload, is
graph-capturable (no internal host sync), correctness ZERO-RISK (identical kernel →
output==baseline → passes). The vendor emits routed indices consumed directly by the GEMM,
avoiding ~39us of our gather/prefix/scatter/gemm-arg glue.

**Architecture now:** `T≤GRAPH_MAX(2048)` → capture `trtllm_fp8_block_scale_moe` in a
CUDA graph keyed by (T, input data_ptrs, offset, rsf), 5-iter warmup, replay+out.clone()
(clone needed: replay reuses the fixed output buffer). `T>2048` (only 11948/14107) → eager
D4 path (custom compaction + 2 `group_gemm_fp8_nt_groupwise` + fused bf16 scatter-accumulate
finalize), where the HBM roundtrip of the g1[R,4096] GEMM1 output pays off only at these
very large shapes; at 901 the fused/graph path still wins on both device time and host
dispatch. GRAPH_MAX=2048 cleanly/optimally splits (no workloads between 80/901 or 901/11948).

**GENERAL LESSON:** before concluding "near-ceiling", always test graph-capturing / directly
reusing the VENDOR baseline primitive, not just your own reconstruction — a leaner vendor
kernel + CUDA graph can beat a hand-decomposed pipeline with zero correctness risk.

**Eager large-path glue rewrites (all KEPT, in moe_kernels.cu):**
- **D8:** atomic-free `finalize_pertoken` (per-token inverse list, one block/token, each out
  row written once, no bf16 atomics). finalize 14107 0.875→0.178ms. → 1.5921.
- **D10:** `swiglu_quant_warp` — one warp per (row,kb) 128-col fp8 group, warp-shuffle max,
  no shared/syncs (old paid 7 syncs/row). swiglu 14107 0.238→0.171, 11948 0.155→0.097.
- **D11:** warp-shuffle two-level leftmost-argmax top-8 in `routing_kernel` (replaces 8×
  256-wide tree reductions). Provably exact leftmost-tie. route_gather 14107 0.29→0.215ms.
  → new best 1.6028/1.6030.

**ALL LEVERS CLOSED (every optimization frontier explored & rejected with evidence):**
- GEMM: `group_gemm_fp8_nt_groupwise` exposes only trtllm/cutile; cutile invalid for these
  shapes, mma_sm=2 slower. The 2 eager GEMMs sit at the vendor trtllm grouped-GEMM floor
  (~64% of 14107). Eager large path NCU: all custom glue kernels compute-pipe-bound @ high
  occ or HBM-bound, none sync-bound.
- **D9 autotune REJECTED:** flashinfer `autotune()` on eager GEMMs — de-risk predicted +3.4%
  (isolated timing) but FULL official REGRESSED (11948 1.578→1.446). Lesson: validate
  GEMM-scheduling levers on the FULL official set + BOTH large shapes, never isolated timing.
- **fp4-weight (D6):** fails correctness.
- **DeepGEMM:** SM100 fp8 needs UE8M0 power-of-2 scales, but task supplies arbitrary fp32
  block scales → rounding to 2^k injects up to ~41% per-block error, fails even rtol 0.3.
- **cutlass_fused_moe:** needs EXTERNAL token_selected_experts/token_final_scales + EP
  sharding + unclear deepseek-block-scale quant_scales format (integration/correctness risk);
  runs the SAME sm100 CUTLASS fp8 grouped GEMM class as trtllm (same floor); its only edge
  (fused-SwiGLU epilogue removing the g1 roundtrip) is sub-noise vs ~1.4GB weight traffic/pass.
- **tiny/mid custom fusion:** our decomposed glue is ~2× SLOWER than the graphed-vendor
  6-kernel chain at seq1; same trtllm GEMM underneath → no path to beat graphed-vendor.
- **Only remaining frontier:** a hand-written tcgen05/CuTe-DSL FP8 block-scale grouped GEMM
  to beat vendor CUTLASS Sm100. Near-zero ROI (only 2/19 workloads have headroom, already
  ~1.6; a 10% GEMM win moves mean <1% under the ±5% large-workload noise) + high risk of
  regressing validated 1.6030. NOT pursued.

**NOISE:** official metric has ±5% run-to-run noise on the two large workloads (14107 seen
1.48–1.76ms, 11948 1.55–1.65 across runs). Sub-5% device wins must be judged on the CUDA-event
phase profiler (`scripts/derisk_largephases.py` / `derisk_graph_baseline.py`), not a single
official run. Tiny/mid (17 workloads) = graphed identical vendor kernel + launch savings, near
ceiling.

**FIRSTHAND RE-VERIFICATIONS (continuations 1–10, all 2026-10-21, Opus 4.8):** every session
confirmed deployed(/root/moe_dev/solution)==committed==local (a58949f0 / e4441300), GPU idle,
dev-8 all PASS live on CUDA 13.0 with latencies matching recorded within ±5% large noise
(seq1 ~0.056, 901 ~0.350, 14107 1.48–1.66ms), maxabs unchanged. Each independently re-derived
both dispatch paths optimal and re-closed one or more levers above (RECONFIRM full-19 run =
1.6030; GRAPH_MAX=2048 boundary re-tested = eager-D4 −24% at 901; DeepGEMM UE8M0; GEMM1+SwiGLU
fusion = the tcgen05/CuTe frontier; cutlass_fused_moe). No code change in any; benchmark.csv
rows p3-recheck..p3-recheck9.

**CONSOLIDATED CADENCE (continuations 11–19, all 2026-10-21, Opus 4.8):** steady state reached.
Each session did an ARTIFACT-LEVEL md5 check ONLY — no GPU cycle, no code change, no verify-commit
(a near-duplicate verify-commit IS the churn to avoid). Every check confirmed: main worktree clean
(only `.claude/` untracked), worktree `.claude/worktrees/moe-fp8-b200` HEAD d2cbb94,
`solution/python/{main.py,moe_kernels.cu}` = a58949f0e66cbf8372581471bdc676fd /
e4441300192b8a5bae39b76933d53a08 = exact match to validated-1.6030 artifact, intact. Sessions are
NON-INTERACTIVE ("Do not ask questions"; AskUserQuestion returns "nobody to ask") — do NOT attempt
to ask, just decide + report. Future continuations: keep this consolidated line, do NOT append a
new per-continuation entry (that tail is the churn); only split out an entry if something actually
CHANGES.

**Project at validated ceiling — do not churn the solution; any further CUDA lever requires
out-engineering vendor CUTLASS Sm100 grouped GEMM for <1% mean gain at high regression risk.
Next continuation: even a firsthand verify adds ~nothing; artifact-level md5 check (cheap) is
sufficient to confirm intactness. Just report the ceiling state unless given a genuinely new
direction.**
