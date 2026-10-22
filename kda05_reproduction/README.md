# KDA 0.5 setup, reproduced with Opus-4.8 / Opus-5 / Fable-5.1 (B200)

A reproduction of the agent setup behind KDA 0.5 (the MLSys 2026 FlashInfer contest
submission `mit-han-lab/mlsys2026-flashinfer-contest-solution`), run once with each of
the three models used elsewhere in this repository, so that the `/goal` results can be
compared against what the same model achieves inside KDA's own harness. Each directory
holds the solutions that produced the reported numbers; only the operator code is
included.

Setup, aligned with KDA 0.5 where it is public:

- **Harness.** humanize2 (`hmz` 0.2.0) `flame_chase`: two chasers take turns, each turn a
  fresh Claude Code session that shares only the git workspace with the others. KDA 0.5
  pairs two different models; here both chasers are the same model (effort `high`), and
  Claude Code's helper-model calls are pointed at that model too. Skills installed:
  KernelWiki, `ncu-report-skill` and `iket-profiling`, each pinned to a commit before
  2026-08-31.
- **Task.** The KDA 0.1 contest prompts (published with the contest repository; phases
  1–3: correct kernel, optimize, full evaluation), with only the planning step of the first
  humanize version removed and the evaluator command filled in. The
  scope is a whole FlashInfer-Bench definition (all of its workloads), not one case. No
  reference kernel and no speed target are given; the only baseline is the contest's
  FlashInfer wrapper solution. MoE is restricted to CUDA C++ / CuTe DSL as in the contest.
  KDA 0.1/0.5 have no attention-backward track, so the FA4 backward task was written in the
  same form from the DSA prompt, with a FlashInfer-Bench definition we built
  (`attention_bwd_causal_h16_d128`, `attention_bwd_varlen_causal_h16_d128`; tolerance
  atol 0.14 / 0.11, rtol 0.01 = 7x FA4's own error) and calls into flash-attn or any other
  attention library forbidden.
- **Evaluator.** The official `flashinfer-bench` (main @ `40e6ca7`, the contest's
  evaluator: CUPTI kernel span, cold L2, median) with the contest dataset, run by the agents
  on their own B200; the reported numbers are the best PASSED trace per workload over the
  run.
- **Guard.** A PreToolUse hook refused fetches of the KDA and contest
  winner repositories and reads outside the arm's own workspace.
- **Budget.** 24 h; all twelve runs were stopped at the same moment, after 23.1–23.2 h
  (DSA, GDN, MoE) and 15.7 h (FA4, started later).

Device time in µs, best over the run, on the three workloads that the rest of this
repository uses (KDA 0.5 column: the contest submission's published figures):

| operator | workload | Opus-4.8 | Opus-5 | Fable-5.1 | KDA 0.5 |
|---|---|---|---|---|---|
| FA4 backward | MHA `5d2f3c61` (B8 N4096 H16) | 1347 | 1405 | 1204 | — |
| | GQA `8c41e9b2` (H_kv 2) | 1250 | 1440 | 1166 | — |
| | varlen `e3a7b5d9` (13 seqs, T 32768) | 1565 | 1673 | 1384 | — |
| DSA decode | 重负载 `564007ac` (T=8) | 12.45 | 8.53 | 7.64 | 10.3 |
| | 小请求 `9d4a5f21` (T=2) | 5.66 | 4.60 | 3.74 | 3.4 |
| | 长短混合 `05f6de65` (T=2) | 8.92 | 6.55 | 5.34 | 6.8 |
| GDN prefill | 长序列 `a9540651` (T 8192, N 20) | 1007 | 126 | 234 | 126 |
| | 单条短序列 `62447caf` (T 30, N 1) | 15.74 ¹ | 4.93 | 4.86 | 6 |
| | 少数几条长序列 `07aa7922` (T 5709, N 2) | 902 | 138 | 313 | 105 |
| MoE FP8 | 单 token `e05c6c03` (T=1) | (56.4) ² | 61.5 | 41.2 | 49.05 |
| | 中等 batch `1a4c6ba1` (T=901) | (350) ² | 393 | 283 | 278.29 |
| | 大 prefill `5e8dc11c` (T=14107) | (1446) ² | 1632 | 941 | 1042.65 |

Not independently re-verified: these are the official evaluator's numbers as the runs
recorded them, on four different B200s, with the two chasers of a run sharing one GPU.

¹ The best recorded value for this cell was 12.37 µs, but it came from a variant the run
itself reverted (it failed 5 of the 100 GDN workloads, one with errors of 1e16) and was
measured with the agent's development script rather than the official evaluator. 15.74 µs
is the kept solution's official result.

² Not an agent kernel: for T ≤ 2048 this solution replays a CUDA graph of FlashInfer's
`trtllm_fp8_block_scale_moe`, the same library kernel as the baseline, and for larger T it
calls FlashInfer's grouped FP8 GEMMs. The task asked for CUDA C++ or CuTe DSL. The files are
kept as the run produced them.

## Contents

FlashInfer-Bench solutions are JSON files that embed their sources. Each `<solution>/`
directory has those sources written out with their relative paths, plus `solution.json`
(the spec — language, entry point, target — with the source contents removed). Where a
name was re-registered with different code during the run, the version that produced the
number was recovered from the run's git history; directories with a `__<suffix>` were
renamed to keep three versions of one name apart.

| op / model | solution → rows |
|---|---|
| `fa4_bwd/opus-4.8` | `cute_fabwd_bshd_v5c_fusedzero` → MHA; `cute_fabwd_bshd_v6_gridreorder` → GQA; `cute_fabwd_varlen_v5c_fusedzero` → varlen |
| `fa4_bwd/opus-5` | `d20b_bshd` → MHA; `final6_bshd` → GQA; `final5_varlen` → varlen |
| `fa4_bwd/fable-5.1` | `fable51_attn_bwd_v009_bshd` → MHA, GQA; `fable51_attn_bwd_v009_varlen` → varlen |
| `dsa_decode_fwd/opus-4.8` | `dsa_triton_v15` → 564007ac; `dsa_triton_v11` → 9d4a5f21; `dsa_triton_v14` → 05f6de65 |
| `dsa_decode_fwd/opus-5` | `dsa_b200_cluster16_dsmem_v25` → 564007ac; `dsa_cluster_split16_v33` → 9d4a5f21; `dsa_b200_cluster16_dsmem` → 05f6de65 |
| `dsa_decode_fwd/fable-5.1` | `fable51_dsa_attn_v23` → 564007ac; `fable51_dsa_attn_v26` → 9d4a5f21; `fable51_dsa_attn_v20` → 05f6de65 |
| `gdn_fwd/opus-4.8` | `gdn-prefill-opus48-v1` → all three |
| `gdn_fwd/opus-5` | `gdn_prefill_b200_cuda__phase6_c082ff2` → a9540651; `__phase5_fa377b7` → 62447caf; `__phase7_final` → 07aa7922 |
| `gdn_fwd/fable-5.1` | `gdn-prefill-fused-fable51` → all three |
| `moe_fp8_fwd/opus-4.8` | `moe_p3_fastsw` → e05c6c03; `moe_p3_d9tune` → 1a4c6ba1; `moe_p3_fastrt` → 5e8dc11c |
| `moe_fp8_fwd/opus-5` | `moe_b200_p1_splitk` → e05c6c03; `moe_b200_tc_v2` → 1a4c6ba1; `moe_b200_tc_v1` → 5e8dc11c |
| `moe_fp8_fwd/fable-5.1` | `fable51_moe_v84` → e05c6c03; `fable51_moe_v94` → 1a4c6ba1; `fable51_moe_v93` → 5e8dc11c |

Notes

- These workspaces sat under the operator's home directory, so every session loaded the
  operator's personal `~/.claude/CLAUDE.md` as project instructions (general working
  rules, unrelated to the task). Among other things it told the agent to use a fresh git worktree
  for every task, which is why the runs worked in `ws/.claude/worktrees/*`. This differs
  from the `/goal` sessions, which ran with a clean config.
- FA4: the Opus-4.8 and Fable-5.1 solutions start from NVIDIA CUTLASS's CuTe DSL Blackwell
  FMHA backward example (BSD-3 headers kept; Fable-5.1's is a 5k-line rewrite). Opus-4.8's
  `helpers/fmha_helpers.py` is that example's unmodified helper under NVIDIA's proprietary
  CUTLASS DSL EULA and is not redistributed (see `helpers/NOT_INCLUDED.md`). Opus-5's
  kernel is its own CuTe DSL design.
- Lower-precision accumulation, all within the tolerance: dQ in fp16 or bf16 in all three
  FA4 runs (Fable-5.1 justifies fp16 by the workloads' dQ magnitudes); fp16 / bf16 partials
  in all DSA runs; bf16 GEMMs with `--use_fast_math` in GDN Opus-5 (max error 8.2e-3 against
  atol 1e-2).
- DSA Opus-4.8 caches a CUDA graph keyed on the inputs' `data_ptr`s, with its split
  schedule computed from the first call's `sparse_indices`; new index contents at the same
  addresses could drop keys. It also never returns `lse`, which the official evaluator did
  not catch. DSA Opus-4.8 and Opus-5 (`dsmem`, `v25`) assume the valid top-k indices form a
  prefix of each row.
- FA4 Opus-5's varlen entry returns slices of a persistent workspace, so a second call
  overwrites the first call's gradients.
- Several solutions tune their dispatch thresholds to the benchmark's shapes; this affects
  speed only.
- Many solutions read environment variables for ablations (`DSA_*`, `GDN_*`, `MOE_*`,
  `FMHA_BWD_*`); leave them unset to get the evaluated configuration.
