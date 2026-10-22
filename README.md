# Crossing the Expert Line Without Expert Machinery

GPU kernels written end-to-end by Claude models running in Claude Code's native
`/goal` mode — a single session given the task and a stopping condition, with no
custom agent harness and no hand-written knowledge base.

Each session was given one operator case, a B200 GPU, and the goal of beating an
expert-written reference kernel for that case, under three rules:

- write the kernel from scratch; the reference kernel's source may not be read or copied
  (it can only be called as a black box for timing and correctness);
- correctness is checked against an fp32 naive implementation, with the tolerance set
  from the reference's own error;
- speed is device time measured against the reference on the same machine.

The same case is run with several models (Opus-4.8, Opus-5, Fable-5.1) so the
results can be compared across model generations.

The FA4 backward varlen case additionally has three Fable-5.1 sessions that vary the
prompt rather than the model (KernelWiki pointer, NCU profiling, correctness clause
removed); see [`fa4_bwd/README.md`](fa4_bwd/README.md). A later ablation round (FA4
varlen, GDN `t8192_n20`, DSA `t8_topk2048`) adds sessions with the NCU skill installed,
and the same prompts run under humanize2's two-agent `flame_chase` flow instead of
`/goal`; they are documented in the corresponding operator READMEs. GDN `t8192_n20` also
has an earlier (2026-09-21) Fable-5.1 `/goal` session pointed at KernelWiki, with its full
trajectory; see [`gdn_fwd/README.md`](gdn_fwd/README.md).

[`kda05_reproduction/`](kda05_reproduction/) is a separate experiment: a reproduction of
the KDA 0.5 agent setup (humanize2 `flame_chase`, the KDA 0.1 contest task prompts, the
FlashInfer-Bench evaluator) with each of the same three models, on the DSA, GDN, MoE and
FA4-backward definitions. [`goal_fullshape/`](goal_fullshape/) runs the `/goal` setup at that
same scope (one solution per whole definition) with the same three models, and compares
the two on one GPU.

For most sessions the repository contains only the final operator code (kernel +
launcher). Sessions run from 2026-10-20 on (except `goal_fullshape/`), and the 2026-09-21
GDN KernelWiki session, also include their full trajectory; see
[Trajectories](#trajectories).

## Paper

The paper draft is at [`paper.pdf`](paper.pdf).

## Contents

| directory | operator | cases |
|---|---|---|
| [`kda_fwd/`](kda_fwd/) | Kimi Delta Attention forward (B200) | `h64_fixed8192`, `h96_uniform`, `h96_mixed` |
| [`gdn_fwd/`](gdn_fwd/) | Gated DeltaNet prefill forward (B200) | `t8192_n20`, `t30_n1`, `t5709_n2` |
| [`dsa_decode_fwd/`](dsa_decode_fwd/) | DeepSeek Sparse Attention paged MLA decode forward (B200) | `t2_topk2048`, `t8_topk2048`, `t2_topk2048_short` |
| [`fa4_bwd/`](fa4_bwd/) | FlashAttention-4 backward (B200) | `mha_b8_n4096_h16`, `gqa_b8_n4096_h16_kv2`, `varlen_t32768_n13_h16` |
| [`gqa_paged_decode/`](gqa_paged_decode/) | GQA paged-KV decode (B200) | `b1_kv173`, `b16_kv2841`, `b64_kv58134` |
| [`mla_paged_prefill/`](mla_paged_prefill/) | MLA paged prefill, causal (B200) | `b1_q33_kv34`, `b1_q199_kv203`, `b18_q15883_kv15937` |
| [`moe_fp8_fwd/`](moe_fp8_fwd/) | MoE FP8 block-scale, DeepSeek routing (B200) | `t1`, `t901`, `t14107` |
| [`kda05_reproduction/`](kda05_reproduction/) | KDA 0.5 setup reproduced with the three models (B200) | DSA, GDN, MoE and FA4-backward definitions |
| [`goal_fullshape/`](goal_fullshape/) | `/goal` on whole definitions, three models (B200) | DSA, GDN, MoE and FA4-backward definitions |

See each directory's README for the per-case results.

## Trajectories

Sessions run from 2026-10-20 on (`moe_fp8_fwd/`, the `fable-5.1-ncu-skill`,
`fable-5.1-ncu-prompt` and `fable-5.1-flame-chase` directories, and
`kda05_reproduction/`), and `gdn_fwd/t8192_n20/fable-5.1-kernelwiki` (2026-09-21), carry a
`trajectory/` directory next to their kernel:

- `/goal` sessions: `prompt.txt` (the `/goal` prompt, in Chinese; `prompt.v1.txt` where it
  was revised mid-run), `guard_hook.py` (the PreToolUse hook the session ran under), and
  `session/` — the Claude Code transcript (`<session-id>.jsonl.xz`, one JSON record per
  line), the large tool outputs Claude Code stored out of line
  (`<session-id>/tool-results/`), and the session's own memory files (`memory/`).
  The 2026-09-21 KernelWiki session has no guard hook; it has `BRIEF.md`, `node/`,
  `LOG.md` and `supervisor.log` instead (see [`gdn_fwd/README.md`](gdn_fwd/README.md)).
- humanize2 `flame_chase` runs: `TASK.md` (the task prompt), `hmz-console/` (humanize2's
  console output, one file per supervisor attempt), `hmz-epics/` (humanize2's own record of
  each run and its turns), and `sessions/<workspace>/` — one Claude Code transcript per
  chaser turn, grouped by the working directory (main workspace or a git worktree the
  agent created).

Files over 16 KB are xz-compressed (`xz -d` or `xzcat`). The transcripts are verbatim
except for these replacements, made before publication: internal GPU host names
(`gpu-node-<id>`) and cluster, project and resource-pool names, one IP address and one
domain, an internal git host name, the account e-mail address, the content of the
operator's personal `~/.claude/CLAUDE.md`, and a few tool outputs that printed
NVIDIA-proprietary-licensed source (replaced by a note giving their length). That file was loaded as project
instructions into every `kda05_reproduction/` session, because those workspaces sat under
the operator's home directory; the `/goal` and `fable-5.1-flame-chase` sessions ran from
`/Users/Shared` with a clean config and did not see it, and the 2026-09-21 KernelWiki
session ran with `--safe-mode`, which does not load it. No credentials appear in the
transcripts.

