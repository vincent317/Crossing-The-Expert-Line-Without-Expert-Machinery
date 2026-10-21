Develop a kernel that minimizes latency while preserving numerical correctness. The target machine is NVIDIA B200, and the software environment is CUDA 13.0. This task does not restrict the implementation language: CUDA C++, CuTe DSL, Triton, Python-wrapped CUDA extensions, or any other contest-allowed approach may be used.

## Kernel Information

- Definition name: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
- Baseline solution name: `flashinfer_wrapper_5af199`
- Operation type: `dsa_paged`
- Workload count: 23
- Constant axes:
  - `num_qo_heads = 16`
  - `head_dim_ckv = 512`
  - `head_dim_kpe = 64`
  - `page_size = 64`
  - `topk = 2048`
- Variable axes:
  - `num_tokens`
  - `num_pages`

The kernel receives query tensors `q_nope` and `q_pe`, compressed KV cache tensors `ckv_cache` and `kpe_cache`, sparse top-k indices, and a scalar softmax scale. It must return:

- `output` with shape `[num_tokens, 16, 512]` and dtype `bfloat16`
- `lse` with shape `[num_tokens, 16]` and dtype `float32`

The reference computation is:

1. Flatten the paged KV cache from `[num_pages, 64, dim]` to token-level storage.
2. For each token, use `sparse_indices[t]` to gather up to 2048 valid KV tokens; `-1` indicates padding.
3. Compute logits as `(q_nope @ ckv.T) + (q_pe @ kpe.T)` for 16 query/output heads.
4. Apply `sm_scale`.
5. Compute 2-based log-sum-exp for `lse`.
6. Compute softmax attention and multiply by `ckv` to produce `output`.

## Official Acceptance

The solution must pass the official FlashInfer benchmark correctness checks for `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`. Use the official FlashInfer benchmark/starter-kit evaluator and consult the FAQ when dependency or rule questions are unclear:

```text
https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/FAQ.md
```

On this task's machine the official evaluator is `flashinfer-bench` (main @ 40e6ca7), with the contest dataset for this definition at `/root/fib-dataset` (definition, all of its workloads, and the FlashInfer baseline solution). A full validation run is the official command for this definition:

```bash
FIB_DATASET_PATH=/root/fib-dataset flashinfer-bench run --local /root/fib-dataset --definitions dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 --save-results --use-isolated-runner --log-level INFO --resume --timeout 300
```

During development, use eight representative workloads before running all 23 workloads:

| UUID | num_tokens | num_pages |
|---|---:|---:|
| `0c23b10c7b7645719517828c12eaa1d2` | 1 | 8462 |
| `9d4a5f21268e484ea05a2f2af91d9fa7` | 2 | 8462 |
| `b7668cfd194c4b95ab600feb205ebac6` | 2 | 8462 |
| `ddfa9e340b264f76abe7418692faa876` | 6 | 8462 |
| `3838996164a94d728710f913477feba8` | 7 | 8462 |
| `385742b2717e4f02b918c7349dde23d8` | 8 | 8462 |
| `4c46a94ba2364dc7ab476286dee8dce3` | 8 | 8462 |
| `02d6ae9c64ab42ff93f05c23c53bcb7d` | 8 | 8462 |

After any major performance improvement, run the full 23-workload evaluation.

## Machine

- The workspace is this directory, a copy of the official FlashInfer starter kit (https://github.com/flashinfer-ai/flashinfer-bench-starter-kit). Work only inside it.
- Your GPU is the devspace `tasks/1295825` (one NVIDIA B200), already running: `tsh ssh root@gpu-node-1295825`. Use only this machine; do not create, stop or use any other devspace.
- On it, the Python environment is `/opt/sglang/bin/python` (torch 2.13 + CUDA 13.0, flashinfer, flashinfer-bench, CuTe DSL 4.6.0, cupti-python); `flashinfer-bench` is `/opt/sglang/bin/flashinfer-bench` (not on the PATH of a non-interactive `tsh ssh`).
- Do not ask questions.

## Workflow Requirements

- Record every performance-related commit in `benchmark.csv`.
- Record every candidate in `solutions.jsonl` and maintain parent links as a DAG.
- Keep NCU profiling records for each major optimization direction.
- Actively evaluate and use as many relevant B200 and CUDA 13.0 features as possible, including TMA, TMEM, `tcgen05`, warp specialization, persistent scheduling, wide vectorized memory operations, and coalesced memory access when they fit the kernel.
- Use KernelWiki for research on Blackwell/B200, CUDA 13.0, CuTe DSL, Triton, sparse attention, MLA/DSA, paged KV cache access, BF16 attention, softmax/LSE, TMA, TMEM, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Do not copy final released submission code into the starting workspace.

## Phases

Proceed through the phases below in order, continuing from the state recorded in this workspace.

### Phase 1 Goal

Research existing sparse attention and MLA/DSA implementations and produce the first correct B200 implementation. Focus on understanding the sparse index format, paged-cache memory layout, numerical requirements for softmax and 2-based LSE, and a simple correct implementation strategy. Performance matters, but correctness and a clean baseline design are the priority for this phase.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```

### Phase 2 Goal

Start from the best correct Phase 1 implementation. Phase 2 is an exploration phase: use NCU profiling, KernelWiki, and public documentation to identify as many plausible optimization directions as possible, then explore them systematically.

The draft must list the candidate optimization directions, rank them by expected benefit and implementation risk, and split each direction into concrete subtasks. Consider sparse KV gather efficiency, BF16 dot-product throughput, softmax/LSE cost, output accumulation over `head_dim_ckv=512`, reuse across 16 heads, invalid-index handling, occupancy, register pressure, memory stalls, and opportunities to use B200-specific features such as TMA, TMEM, and `tcgen05`.

Explore each optimization direction for at most five iterations. If a direction cannot be implemented cleanly, fails correctness, or does not show a credible path to improvement after those iterations, record the evidence and move to the next ranked direction.

For each explored direction, collect before/after benchmark results on the eight representative workloads and enough NCU evidence to justify whether the change should be kept, revised, or rejected.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```

### Phase 3 Goal

Analyze the full workload distribution and specialize the implementation for the observed shape groups. The primary variable axis is `num_tokens`, with workloads concentrated at 1, 2, 6, 7, and 8 tokens. Use the eight representative workloads during development, then use the full 23-workload set for promotion decisions.

Design dispatch logic and specialized kernels only where the measured win justifies the added complexity. Consider separate paths for single-token decode, two-token batches, and larger 6-8 token groups if profiling shows different bottlenecks. Preserve correctness for all 23 workloads.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```
