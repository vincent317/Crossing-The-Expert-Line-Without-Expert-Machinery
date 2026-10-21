---
name: attn-bwd-fa4-progress
description: Current progress/status of the B200 attention-backward kernel optimization
metadata:
  node_type: memory
  type: project
  originSessionId: 37f282fb-802c-4d02-b19d-1132af0f8028
  modified: 2026-10-21T01:10:57.154Z
---

**TASK COMPLETE — beats FA4 on all 3 workloads. See [[attn-bwd-fa4-task]] for setup/eval.**

Worktree `worktree-attn-bwd-b200` at `.claude/worktrees/attn-bwd-b200`, clean @ 4b7b72a
(final doc commit; best kernel = commit a39e6ac `cute_fabwd_v6_gridreorder`).

**Final official results (speedup = FA4_baseline / ours, all PASSED, 0 failing elements):**
- MHA 5d2f3c61: 1.350 ms vs 1.380 → **1.022×**
- GQA 8c41e9b2: 1.249 ms vs 1.474 → **1.180×**
- varlen e3a7b5d9: 1.568 ms vs 1.572 → **1.002×**
correctness max_abs_err 0.0625(MHA)/0.0313(varlen) ≪ atol 0.14/0.11 (huge max_rel_err is at
near-zero refs, harmless under the abs-AND-rel rule).

**Verified state (fresh session 2026-10-21 #23, re-confirmed #24–#80):** worktree clean @ 4b7b72a; local canonical
(kernel bd8c2a06, fmha_bwd f1eddd1c) == remote /root/work/cute_dev (kernel_cute.py+fmha_bwd.py md5 match, byte-identical;
note remote dev kernel file is named kernel_cute.py not kernel.py);
both defs deployed in /root/fib-dataset/solutions/mine/attention_bwd; docs/draft.md + docs/writeup.md
+ benchmark.csv + solutions.jsonl + 4 ncu records present; B200 reachable + idle (0%/0MiB).
Byte-identical + deterministic + fixed inputs ⇒ official numbers stand; did NOT re-run eval. No
positive-EV lever remains.

**Prior verifications (#4–#22, all 2026-10-21):** every session re-confirmed worktree clean @
4b7b72a, local canonical md5 (bd8c2a06/f1eddd1c) == remote /root/work/cute_dev, both defs deployed,
all deliverables present, B200 idle. Sessions #4/#5 did live correctness re-runs of all 3 workloads
= ALL OK fail=0 (MHA dv≤0.0313–0.0625, GQA dv≤0.0625, varlen dv≤0.0313 ≪ atol 0.14/0.11). helpers
66dcf1be/5d7562bc, solutions.jsonl 10-entry DAG. Latency not re-run (clocks drift; official stands).

**Solution** = CuTe DSL SM100, adapted from NVIDIA CUTLASS `BlackwellFusedMultiHeadAttentionBackward`
(BSD-3-Clause): warp-spec tcgen05 UMMA + TMEM + TMA, fuses dQ/dK/dV; causal/varlen/GQA/d128/bf16.
Canonical: `solution/cute/{kernel.py,fmha_bwd.py,helpers/}`; submission folders
`attention_bwd_{bshd,varlen}/` kept in sync. Dataset solutions via
`scripts/make_solutions_cute.py --name-suffix vN`. Remote dev `/root/work/cute_dev/`.

**Key wins (cumulative):**
- v5c **direct bf16 TMA reduce-add into dQ** (`cp.reduce.async.bulk.tensor.add.bf16`, sm_scale
  folded) kills fp32 dQ_acc scratch + convert kernel; sdQ SMEM halved → main bwd 1.359→1.229 ms.
  dQ-zero fused into sum_OdO pre-pass (standalone 134MB memset is ~12× penalized by official
  metric). Gated `self.direct_dq` (default True) in fmha_bwd.py.
- v6 **heavy-first CTA grid reorder** for causal load balance: launch (H_K*B, nkblk, 1) so heavy
  KV-block-0 CTAs dispatch first. Gated `self.reorder_grid=(not varlen) and (nkblk*H_K*B/148<=8)`
  → ON for GQA only (3.46 waves, tail-bound → 1.447→1.249, −13.6%); MHA (28 waves) & varlen lose
  L2 locality (forcing on: MHA 1.40→1.72, varlen 1.63→2.05).

**Wall (why MHA/varlen stuck at ~1.0×):** main bwd kernel latency-bound on long_scoreboard
4.8 cyc/inst (TMEM-read serialization); SM 60%, tensor 56%, DRAM <9%. All resources packed:
TMEM 512/512 cols (dK+dV+dQ/dP+S), SMEM 211KB (~16KB free < one load stage), regs 128 → 1 CTA/SM.
Occupancy + pipeline-stage levers BLOCKED. Scheduling slack already harvested (per-SM active-cyc
tight across waves for MHA/varlen → no load-balance gain there). NCU: `docs/ncu/p3_cute_v5c_mha_full.txt`
etc.

**Only remaining lever = 2-SM tcgen05 rewrite — assessed INFEASIBLE in budget** (commit 4b7b72a,
docs/draft.md): it is the right/state-of-art direction (FA4 bwd uses 2-CTA MMA + TMEM-retained
intermediates → lower per-SM latency wall) but the vendored CUTLASS bwd is hardcoded cta_group=ONE,
cluster(1,1), 5 fixed MMAs + packed TMEM 512/512 + coupled TMA/mbarrier/epilogue; same-CTA-group
rule forbids partial migration → wholesale 3.7k-line rewrite under zero-fail bar, no CUTLASS
FMHA-bwd 2-SM reference in KernelWiki to adapt. Negative-EV (risks the working 3/3-winning
solution). Rejected per explore-≤5-iters rule. Cheaper variants also rejected (2 CTA/SM SMEM-bound;
N-tile 128→64 doubles KV trips). **No positive-EV lever remains.**

**Old Triton (superseded, `solution/triton/kernel.py`)**: FA2 2-pass, plateaued ~0.28×.
</content>
</invoke>
