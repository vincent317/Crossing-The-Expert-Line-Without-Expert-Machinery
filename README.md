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

This repository contains only the final operator code from each session (kernel +
launcher). Benchmarks, probes and session logs are not included.

## Paper

The paper draft is at [`paper.pdf`](paper.pdf).

## Contents

| directory | operator | cases |
|---|---|---|
| [`kda_fwd/`](kda_fwd/) | Kimi Delta Attention forward (B200) | `h64_fixed8192`, `h96_uniform`, `h96_mixed` |
| [`gdn_fwd/`](gdn_fwd/) | Gated DeltaNet prefill forward (B200) | `t8192_n20`, `t30_n1`, `t5709_n2` |
| [`dsa_decode_fwd/`](dsa_decode_fwd/) | DeepSeek Sparse Attention paged MLA decode forward (B200) | `t2_topk2048` |

See each directory's README for the per-case results.
