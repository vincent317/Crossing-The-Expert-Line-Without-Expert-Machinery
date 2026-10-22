# Crossing the Expert Line Without Expert Machinery

GPU kernels written end-to-end by Claude models running in Claude Code's native
`/goal` mode — a single session given the task and a stopping condition, with no
custom agent harness and no hand-written knowledge base.

Each session was given an operator, a B200 GPU, and the goal of beating an
expert-written reference kernel, under three rules:

- write the kernel from scratch; the reference kernel's source may not be read or copied
  (it can only be called as a black box for timing and correctness);
- correctness is checked against an fp32 naive implementation, with the tolerance set
  from the reference's own error;
- speed is device time measured against the reference on the same machine.

The same task is run with several models (Opus-4.8, Opus-5, Fable-5.1) so the
results can be compared across model generations.

This repository contains only the operator code (kernels and their launchers) that the
sessions produced, with per-directory READMEs giving the setup, the measured results and
how to call each kernel. Benchmarks, test harnesses and session transcripts are not
included.

## Paper

The paper draft is at [`paper.pdf`](paper.pdf).

## Contents

The kernels are grouped into four experiments.

| directory | experiment |
|---|---|
| [`goal_single_shape/`](goal_single_shape/) | `/goal` on one specified shape: each session targets a single operator case (one workload) |
| [`goal_full_shape/`](goal_full_shape/) | `/goal` on a whole FlashInfer-Bench definition: one solution has to cover every workload (shape) of the definition |
| [`kda05_reproduction/`](kda05_reproduction/) | The KDA 0.5 agent setup (humanize2 `flame_chase`, the KDA 0.1 contest task prompts, the FlashInfer-Bench evaluator) reproduced with the same three Claude models, at the whole-definition scope |
| [`ablation/`](ablation/) | Prompt and harness ablations on single shapes with Fable-5.1: NCU skill or NCU prompt, KernelWiki pointer, correctness clause removed, and the same prompt under humanize2 `flame_chase` instead of `/goal` |

Operators covered in each:

| operator | `goal_single_shape/` | `goal_full_shape/` | `kda05_reproduction/` | `ablation/` |
|---|---|---|---|---|
| Kimi Delta Attention forward | [`kda_fwd`](goal_single_shape/kda_fwd/) | | | |
| Gated DeltaNet prefill forward | [`gdn_fwd`](goal_single_shape/gdn_fwd/) | [`gdn_fwd`](goal_full_shape/gdn_fwd/) | [`gdn_fwd`](kda05_reproduction/gdn_fwd/) | [`gdn_fwd`](ablation/gdn_fwd/) |
| DeepSeek Sparse Attention paged MLA decode | [`dsa_decode_fwd`](goal_single_shape/dsa_decode_fwd/) | [`dsa_decode_fwd`](goal_full_shape/dsa_decode_fwd/) | [`dsa_decode_fwd`](kda05_reproduction/dsa_decode_fwd/) | [`dsa_decode_fwd`](ablation/dsa_decode_fwd/) |
| FlashAttention-4 backward | [`fa4_bwd`](goal_single_shape/fa4_bwd/) | [`fa4_bwd`](goal_full_shape/fa4_bwd/) | [`fa4_bwd`](kda05_reproduction/fa4_bwd/) | [`fa4_bwd`](ablation/fa4_bwd/) |
| MoE FP8 block-scale, DeepSeek routing | [`moe_fp8_fwd`](goal_single_shape/moe_fp8_fwd/) | [`moe_fp8_fwd`](goal_full_shape/moe_fp8_fwd/) | [`moe_fp8_fwd`](kda05_reproduction/moe_fp8_fwd/) | |
| GQA paged-KV decode | [`gqa_paged_decode`](goal_single_shape/gqa_paged_decode/) | | | |
| MLA paged prefill, causal | [`mla_paged_prefill`](goal_single_shape/mla_paged_prefill/) | | | |
| MLA paged decode | [`mla_paged_decode`](goal_single_shape/mla_paged_decode/) | | | |
| FlashAttention-4 forward | [`fa4_fwd`](goal_single_shape/fa4_fwd/) | | | |

See each directory's README for the per-case results.

## Layout

```
goal_single_shape/<operator>/<case>/<model>/          kernel + launcher
goal_full_shape/<operator>/<model>/<definition>/      solution.json + its sources
kda05_reproduction/<operator>/<model>/<solution>/     solution.json + its sources
ablation/<operator>/<case>/fable-5.1-<arm>/           kernel + launcher
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE). Files adapted from NVIDIA
CUTLASS examples keep their BSD-3-Clause headers and license; they are listed in
[`NOTICE`](NOTICE).
