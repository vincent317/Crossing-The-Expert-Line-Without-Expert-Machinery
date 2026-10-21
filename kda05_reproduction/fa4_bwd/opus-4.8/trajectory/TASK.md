Develop a kernel that minimizes latency while preserving numerical correctness. The target machine is NVIDIA B200, and the software environment is CUDA 13.0. This task does not restrict the implementation language: CUDA C++, CuTe DSL, Triton, Python-wrapped CUDA extensions, or any other contest-allowed approach may be used.

## Kernel Information

This task has two definitions, one attention backward in two layouts; each definition needs its own solution.

- Definition names: `attention_bwd_causal_h16_d128` and `attention_bwd_varlen_causal_h16_d128`
- Baseline solution names: `flash_attn4_wrapper_b19_bshd` and `flash_attn4_wrapper_b19_varlen` (flash-attn-4 4.0.0b19)
- Operation type: `attention_bwd`
- Workload count: 3 (2 + 1)
- `attention_bwd_causal_h16_d128`, layout `(batch, seq, heads, dim)`:
  - Constant axes: `num_qo_heads = 16`, `head_dim = 128`
  - Variable axes: `batch_size`, `seq_len`, `num_kv_heads` (16 for MHA, 2 for GQA 8:1)
- `attention_bwd_varlen_causal_h16_d128`, layout `(tokens, heads, dim)`:
  - Constant axes: `num_heads = 16`, `head_dim = 128`
  - Variable axes: `total_tokens`, `num_seqs`, `len_cu_seqlens`

Each kernel receives the attention inputs `q`, `k`, `v` (bfloat16), the forward results `out` (bfloat16) and `lse` (float32, natural-log log-sum-exp of the scaled, masked logits; `[batch_size, 16, seq_len]` or `[16, total_tokens]`), the output gradient `dout` (bfloat16), and a scalar softmax scale `sm_scale`; the variable-length definition also receives `cu_seqlens` (int32, starting at 0) and the scalar `max_seqlen`. It must return `dq`, `dk` and `dv` with the shapes of `q`, `k` and `v` and dtype `bfloat16`.

The reference computation is, per batch entry (or per sequence given by `cu_seqlens`) and per query head:

1. Compute logits `S = sm_scale * q @ k.T` with the key/value head that serves this query head (query head `h` uses key/value head `h // (num_qo_heads // num_kv_heads)`).
2. Apply the causal mask: query position `i` attends to key positions `j <= i` within its own sequence.
3. Compute `P = softmax(S)` along the key axis.
4. Compute `dV = P.T @ dout` and `dP = dout @ v.T`.
5. Compute `dS = P * (dP - rowsum(P * dP))`.
6. Compute `dQ = sm_scale * dS @ k` and `dK = sm_scale * dS.T @ q`.
7. For grouped-query attention, sum `dK` and `dV` over the query heads that share a key/value head.

The baseline solution calls the backward entry of flash-attn-4 4.0.0b19 (`flash_attn.cute`), which is installed on the machine. Your solution must implement the backward pass itself: it must not call flash-attn or another library's attention kernels, and flash-attn's source code is off limits (the baseline may be run, not read).

## Official Acceptance

The solution must pass the official FlashInfer benchmark correctness checks for both definitions. Use the official FlashInfer benchmark/starter-kit evaluator and consult the FAQ when dependency or rule questions are unclear:

```text
https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/FAQ.md
```

On this task's machine the official evaluator is `flashinfer-bench` (main @ 40e6ca7), with the dataset for these definitions at `/root/fib-dataset` (both definitions, all of their workloads, and the flash-attn-4 baseline solution). A full validation run is the official command for each definition:

```bash
FIB_DATASET_PATH=/root/fib-dataset flashinfer-bench run --local /root/fib-dataset --definitions attention_bwd_causal_h16_d128 --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 --atol 0.14 --rtol 0.01
FIB_DATASET_PATH=/root/fib-dataset flashinfer-bench run --local /root/fib-dataset --definitions attention_bwd_varlen_causal_h16_d128 --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 --atol 0.11 --rtol 0.01
```

There are three workloads in all; use them during development and for every evaluation:

| UUID | definition | shape |
|---|---|---|
| `5d2f3c61-0a4e-4f7b-9a6d-1b8e2c4f7a01` | `attention_bwd_causal_h16_d128` | MHA: batch_size 8, seq_len 4096, num_kv_heads 16 |
| `8c41e9b2-7f3d-4a15-b0c8-6e2d9f1a3b02` | `attention_bwd_causal_h16_d128` | GQA: batch_size 8, seq_len 4096, num_kv_heads 2 |
| `e3a7b5d9-2c6f-4e81-9d4b-0f5a8c7e1d03` | `attention_bwd_varlen_causal_h16_d128` | 13 sequences, total_tokens 32768 (4096, 217, 8192, 1500, 513, 2800, 6144, 700, 1777, 257, 3500, 1024, 2048), max_seqlen 8192 |

After any major performance improvement, run the full evaluation for both definitions.

## Machine

- The workspace is this directory, a copy of the official FlashInfer starter kit (https://github.com/flashinfer-ai/flashinfer-bench-starter-kit). Work only inside it.
- Your GPU is the devspace `tasks/1305349` (one NVIDIA B200), already running: `tsh ssh root@gpu-node-1305349`. Use only this machine; do not create, stop or use any other devspace.
- On it, the Python environment is `/opt/sglang/bin/python` (torch 2.13 + CUDA 13.0, flash-attn-4 4.0.0b19 for the baseline, flashinfer, flashinfer-bench, CuTe DSL, cupti-python); `flashinfer-bench` is `/opt/sglang/bin/flashinfer-bench` (not on the PATH of a non-interactive `tsh ssh`).
- Do not ask questions.

## Workflow Requirements

- Record every performance-related commit in `benchmark.csv`.
- Record every candidate in `solutions.jsonl` and maintain parent links as a DAG.
- Keep NCU profiling records for each major optimization direction.
- Actively evaluate and use as many relevant B200 and CUDA 13.0 features as possible, including TMA, TMEM, `tcgen05`, warp specialization, persistent scheduling, wide vectorized memory operations, and coalesced memory access when they fit the kernel.
- Use KernelWiki for research on Blackwell/B200, CUDA 13.0, CuTe DSL, Triton, attention backward, FlashAttention, causal masking, grouped-query attention, variable-length sequence batching, BF16 attention, softmax/LSE, TMA, TMEM, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Do not copy final released submission code into the starting workspace.

## Phases

Proceed through the phases below in order, continuing from the state recorded in this workspace.

### Phase 1 Goal

Research existing attention backward implementations and produce the first correct B200 implementation. Focus on understanding the backward computation from `out`, `lse` and `dout`, the causal mask, grouped-query head mapping, variable-length batching through `cu_seqlens`, numerical requirements for the recomputed softmax and the BF16 gradients, and a simple correct implementation strategy. Performance matters, but correctness and a clean baseline design are the priority for this phase.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```

### Phase 2 Goal

Start from the best correct Phase 1 implementation. Phase 2 is an exploration phase: use NCU profiling, KernelWiki, and public documentation to identify as many plausible optimization directions as possible, then explore them systematically.

The draft must list the candidate optimization directions, rank them by expected benefit and implementation risk, and split each direction into concrete subtasks. Consider BF16 tensor-core throughput for the backward matrix products, recomputation of the attention probabilities, the `rowsum(P * dP)` term, accumulation of `dQ` across key blocks, causal-mask work skipping and load balance, the reduction of `dK`/`dV` across grouped query heads, scheduling of variable-length sequences, occupancy, register pressure, memory stalls, and opportunities to use B200-specific features such as TMA, TMEM, and `tcgen05`.

Explore each optimization direction for at most five iterations. If a direction cannot be implemented cleanly, fails correctness, or does not show a credible path to improvement after those iterations, record the evidence and move to the next ranked direction.

For each explored direction, collect before/after benchmark results on the three workloads and enough NCU evidence to justify whether the change should be kept, revised, or rejected.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```

### Phase 3 Goal

Analyze the three workloads and specialize the implementation for them: dense MHA (batch 8, seq_len 4096, 16 key/value heads), dense GQA 8:1 (2 key/value heads), and variable-length MHA with 13 packed sequences from 217 to 8192 tokens. Use all three workloads for development and for promotion decisions.

Design dispatch logic and specialized kernels only where the measured win justifies the added complexity. Consider separate paths for MHA, GQA and the variable-length layout if profiling shows different bottlenecks. Preserve correctness for all three workloads.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```
