Develop a kernel that minimizes latency while preserving numerical correctness. The target machine is NVIDIA B200, and the software environment is CUDA 13.0. This task does not restrict the implementation language: CUDA C++, CuTe DSL, Triton, Python-wrapped CUDA extensions, or any other contest-allowed approach may be used.

## Kernel Information

- Definition name: `gdn_prefill_qk4_v8_d128_k_last`
- Baseline solution name: `flashinfer_wrapper_123ca6`
- Operation type: `gdn`
- Workload count: 100
- Constant axes:
  - `num_q_heads = 4`
  - `num_k_heads = 4`
  - `num_v_heads = 8`
  - `head_size = 128`
- Variable axes:
  - `total_seq_len`
  - `num_seqs`
  - `len_cu_seqlens`

The kernel receives variable-length Gated Delta Net prefill inputs in k-last state layout. It must return:

- `output` with shape `[total_seq_len, 8, 128]` and dtype `bfloat16`
- `new_state` with shape `[num_seqs, 8, 128, 128]` and dtype `float32`

The reference computation is:

1. Compute `g = exp(-exp(A_log) * softplus(a + dt_bias))`.
2. Compute `beta = sigmoid(b)`.
3. Expand 4 query/key heads to 8 value heads by grouped-value attention mapping.
4. For each sequence defined by `cu_seqlens`, scan tokens in order and update the recurrent state.
5. Compute `output = scale * q @ state` at each token.
6. Return the final state for every sequence in k-last layout `[N, H, V, K]`.

## Official Acceptance

The solution must pass the official FlashInfer benchmark correctness checks for `gdn_prefill_qk4_v8_d128_k_last`. Use the official FlashInfer benchmark/starter-kit evaluator and consult the FAQ when dependency or rule questions are unclear:

```text
https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/FAQ.md
```

On this task's machine the official evaluator is `flashinfer-bench` (main @ 40e6ca7), with the contest dataset for this definition at `/root/fib-dataset` (definition, all of its workloads, and the FlashInfer baseline solution). A full validation run is the official command for this definition:

```bash
FIB_DATASET_PATH=/root/fib-dataset flashinfer-bench run --local /root/fib-dataset --definitions gdn_prefill_qk4_v8_d128_k_last --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 --warmup-runs 1 --iterations 5 --num-trials 3
```

During development, use eight representative workloads before running all 100 workloads:

| UUID | total_seq_len | num_seqs | len_cu_seqlens |
|---|---:|---:|---:|
| `77daf91d-0660-4c4b-8c32-336a69281cd9` | 6 | 1 | 2 |
| `d8f4a9ae-d391-4f70-8069-f5deb510a2d1` | 35 | 1 | 2 |
| `ba08a83e-e151-4e16-bc70-abee6851604c` | 134 | 1 | 2 |
| `ce832e76-c4c2-421f-ad75-ebda2032c401` | 461 | 2 | 3 |
| `c5257f65-c411-4dbb-9dc1-ad4abcf00254` | 983 | 2 | 3 |
| `aaa378be-7365-4381-b2d3-35951ef88c7a` | 1800 | 3 | 4 |
| `4b6143dd-0e5f-499f-93cb-076d9635bcd0` | 4124 | 15 | 16 |
| `d49df0b2-7838-4865-9a51-7ec7877fe27f` | 8192 | 48 | 49 |

After any major performance improvement, run the full 100-workload evaluation.

## Machine

- The workspace is this directory, a copy of the official FlashInfer starter kit (https://github.com/flashinfer-ai/flashinfer-bench-starter-kit). Work only inside it.
- Your GPU is the devspace `tasks/1295863` (one NVIDIA B200), already running: `tsh ssh root@gpu-node-1295863`. Use only this machine; do not create, stop or use any other devspace.
- On it, the Python environment is `/opt/sglang/bin/python` (torch 2.13 + CUDA 13.0, flashinfer, flashinfer-bench, CuTe DSL 4.6.0, cupti-python); `flashinfer-bench` is `/opt/sglang/bin/flashinfer-bench` (not on the PATH of a non-interactive `tsh ssh`).
- Do not ask questions.

## Workflow Requirements

- Record every performance-related commit in `benchmark.csv`.
- Record every candidate in `solutions.jsonl` and maintain parent links as a DAG.
- Keep NCU profiling records for each major optimization direction.
- Actively evaluate and use as many relevant B200 and CUDA 13.0 features as possible, including TMA, TMEM, `tcgen05`, warp specialization, persistent scheduling, wide vectorized memory operations, and coalesced memory access when they fit the kernel.
- Use KernelWiki for research on Blackwell/B200, CUDA 13.0, CuTe DSL, Triton, Gated Delta Net prefill, recurrent state scans, BF16 math, variable-length sequence batching, warp/block scheduling, TMA, TMEM, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Do not copy final released submission code into the starting workspace.

## Phases

Proceed through the phases below in order, continuing from the state recorded in this workspace.

### Phase 1 Goal

Research existing GDN prefill and linear-attention implementations and produce the first correct B200 implementation. Focus on understanding the recurrence, k-last state layout, variable-length batching through `cu_seqlens`, numerical requirements, and a simple correct implementation strategy. Performance matters, but correctness and a clean baseline design are the priority for this phase.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```

### Phase 2 Goal

Start from the best correct Phase 1 implementation. Phase 2 is an exploration phase: use NCU profiling, KernelWiki, and public documentation to identify as many plausible optimization directions as possible, then explore them systematically.

The draft must list the candidate optimization directions, rank them by expected benefit and implementation risk, and split each direction into concrete subtasks. Consider chunking the recurrence, parallelizing within and across sequences, short-sequence fast paths, long-sequence persistent kernels, state layout access, BF16 vectorized loads, gate precomputation, output/state writeback cost, occupancy, register pressure, memory stalls, and opportunities to use B200-specific features such as TMA, TMEM, and `tcgen05`.

Explore each optimization direction for at most five iterations. If a direction cannot be implemented cleanly, fails correctness, or does not show a credible path to improvement after those iterations, record the evidence and move to the next ranked direction.

For each explored direction, collect before/after benchmark results on the eight representative workloads and enough NCU evidence to justify whether the change should be kept, revised, or rejected.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```

### Phase 3 Goal

Analyze the full workload distribution and specialize the implementation for different shape groups. The primary variable axes are `total_seq_len` and `num_seqs`; the workload set includes very short single-sequence cases, medium prefill cases, and long `total_seq_len=8192` multi-sequence cases.

Design dispatch logic and specialized kernels only where the measured win justifies the added complexity. Consider separate paths for tiny single-sequence prefill, medium sequences, and long multi-sequence batches if profiling shows different bottlenecks. The final candidate should be evaluated on the full 100-workload set, not only on the eight development workloads.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```
