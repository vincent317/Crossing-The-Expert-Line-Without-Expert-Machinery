# `/goal` on whole FlashInfer-Bench definitions (B200)

The `/goal` sessions elsewhere in this repository each target one operator case (one
shape). Here the same `/goal` setup is given a whole FlashInfer-Bench definition: one
solution has to cover every workload of the definition — DSA decode 23, GDN prefill 100,
MoE FP8 19, FA4 backward 3 (two definitions) — which is the scope the
[`kda05_reproduction/`](../kda05_reproduction/) runs work at. One session per operator and
model: 4 operators × Opus-4.8 / Opus-5 / Fable-5.1 = 12 sessions.

Each directory holds the final solution as packaged for FlashInfer-Bench
(`<definition>/solution.json`, with its sources unpacked next to it) and the session's own
report (`FINAL.md`). Trajectories are not included.

## Setup

The prompt is the same one used for the single-case `/goal` sessions (in Chinese; clean
Claude Code config, no skills, no memory; work from scratch; never read or copy the
reference's implementation, only run it as a black box; fp32 naive implementation with a
per-workload tolerance derived from the reference's own error, ×7; correct for any input,
not only the dataset's), with these changes:

- **Scope.** All workloads of the definition, listed in the prompt; one solution per
  definition, which may dispatch or specialize by shape.
- **Reference.** Per workload, the fastest of the candidates measured on the session's own
  B200: KDA 0.5 (the contest submission, provided as a black-box command that runs and times
  its kernels) or the FlashInfer library baseline for DSA, GDN and MoE; flash-attn-4
  (`4.0.0b19`) or cuDNN through torch SDPA for FA4 backward.
- **Rules.** No runtime calls to GEMM/attention/MoE libraries from the solution (cuBLAS,
  `torch.mm`/`matmul`/`_scaled_mm`/`_grouped_mm`, FlashInfer, TensorRT-LLM, DeepGEMM,
  flash-attn, CUTLASS device-level GEMM templates, …). Timing = CUPTI kernel span, cold L2,
  median; no input-derived result may be cached across calls.
- **Acceptance.** The final solution also had to pass the official evaluator
  (`flashinfer-bench` main @ `40e6ca7`) on every workload, with each definition's official
  flags.
- **Budget.** Until a fixed deadline (about 26 h). Most sessions had stopped changing
  their kernels hours earlier and were waiting for the deadline; all twelve were stopped
  together after 19.8–20.3 h, about 6 h before it.

## Results, re-measured on one GPU

All numbers below were re-measured after the runs on a single B200, with the official
evaluator (`flashinfer-bench` main @ `40e6ca7`: CUPTI kernel span, cold L2, median; each
definition's official flags), in at least two alternating rounds, together with the
references. Only the three workloads per operator that the rest of this repository uses were
timed; the solutions passed the evaluator on all workloads of their definitions.

Device time in µs (lower is better):

| operator | workload | Opus-4.8 | Opus-5 | Fable-5.1 | KDA 0.5 | library |
|---|---|---|---|---|---|---|
| FA4 backward | MHA `5d2f3c61` (B8 N4096 H16) | 2675 | 1702 | 1932 | — | 1371 (FA4) |
| | GQA `8c41e9b2` (H_kv 2) | 2701 | 1647 | 1565 | — | 1456 (FA4) |
| | varlen `e3a7b5d9` (13 seqs, T 32768) | 3426 | 1985 | 2110 | — | 1572 (FA4) |
| DSA decode | `564007ac` (T=8) | 19.3 | 9.58 | 13.3 | 10.13 | 585 ¹ |
| | `9d4a5f21` (T=2) | 10.0 | 4.02 | 4.73 | 3.36 | 661 ¹ |
| | `05f6de65` (T=2) | 13.5 | 6.90 | 9.30 | 6.92 | 652 ¹ |
| GDN prefill | `a9540651` (T 8192, N 20) | 1126 | 213 | 178 | 124.9 | — |
| | `62447caf` (T 30, N 1) | 13.96 | 5.96 | 5.36 | 4.81 | — |
| | `07aa7922` (T 5709, N 2) | 1863 | 198 | 203 | 107.7 | — |
| MoE FP8 | `e05c6c03` (T=1) | 294 | 118 ² | 53.7 | 52.0 | 149 |
| | `1a4c6ba1` (T=901) | 551 | 361 | 431 | 275.8 | 435 |
| | `5e8dc11c` (T=14107) | 2703 | 1379 | 2431 | 1028.5 | 2515 |

¹ The FlashInfer baseline solution of the contest dataset, which concatenates its inputs
on every call; it is not the speed of the underlying library kernel.
² Bimodal across rounds (63–122 µs); median shown.

Geometric mean of reference time / solution time over the three workloads (higher is
better; reference = KDA 0.5 for DSA, GDN and MoE, flash-attn-4 for FA4), next to the
[`kda05_reproduction/`](../kda05_reproduction/) runs of the same models, re-measured on the
same GPU — at the end of their run (23.1–23.2 h; 15.7 h for FA4; 13.8 h for the MoE
Opus-4.8 rerun) and at about 16 h (the version each run had then; the best per workload):

| operator | model | `/goal`, whole definition | KDA 0.5 setup, at ~16 h | KDA 0.5 setup, final |
|---|---|---|---|---|
| FA4 backward | Opus-4.8 | 0.50 | 1.07 | 1.07 |
| | Opus-5 | 0.83 | 0.96 | 0.96 |
| | Fable-5.1 | 0.79 | 1.18 | 1.18 |
| DSA decode | Opus-4.8 | 0.45 | 0.73 ³ | 0.73 ³ |
| | Opus-5 | 0.96 | 0.92 | 1.01 |
| | Fable-5.1 | 0.74 | 1.16 | 1.20 |
| GDN prefill | Opus-4.8 | 0.13 | 0.18 | 0.18 |
| | Opus-5 | 0.64 | 0.90 | 0.92 |
| | Fable-5.1 | 0.69 | 0.55 | 0.58 |
| MoE FP8 | Opus-4.8 | 0.32 | 0.63 ⁴ | 0.64 ⁴ |
| | Opus-5 | 0.63 | 0.67 | 0.69 |
| | Fable-5.1 | 0.64 | 1.10 | 1.11 |

³ That solution returns only the attention output, not the log-sum-exp; the DSA
evaluator accepts this, the `/goal` prompt and KDA 0.5 compute both.
⁴ Fails the stricter check below on two workloads.

## Correctness, re-checked

Beyond the official evaluator, every solution was checked against the definition's
reference implementation computed in fp32 (the shipped reference rounds its outputs to
bf16), on three input seeds per workload, with tolerance = 7 × KDA 0.5's own error (7 ×
flash-attn-4's for FA4), separately for max and mean absolute error and for each output.
All twelve solutions pass on every timed workload except:

- **DSA Opus-4.8 and Opus-5**, workload `9d4a5f21`: the log-sum-exp exceeds the tolerance
  by 1.4–1.7×. KDA 0.5's log-sum-exp is almost exact there (tokens with 11–18 valid keys),
  so the tolerance is tiny; the absolute error is below 1e-5, at fp32 rounding level.
- **MoE Opus-4.8**, T=14107: 3 of 14107 token rows are wrong as a whole, which looks like
  near-ties in expert selection (the 8th and 9th expert scores almost equal); the mean
  error equals KDA 0.5's. The official evaluator passes it through its 90 % matched-ratio
  rule.

## Self-reported numbers

`FINAL.md` is each session's own report, measured on its own B200 against its own
measurement of the references. These numbers are generally more favorable than the table
above: the references were often slower on the sessions' GPUs (KDA 0.5 DSA T=8 measured at
12.6–16.7 µs by the sessions, 10.13 µs above), and three reports are wrong:

- **GDN Opus-4.8** reports a geometric mean of 185×: its benchmark read a millisecond
  timer as microseconds. Measured, the solution is 0.13× KDA 0.5.
- **MoE Opus-4.8** replays a CUDA graph captured per set of input pointers; its own
  benchmark reused the same tensors and hit the graph, while the evaluator passes fresh
  tensors and the solution runs eagerly (294 µs at T=1 instead of the reported ~94 µs).
- **MoE Opus-5** timed itself on inputs it generated, with routing unlike the dataset's
  (122 µs at T=1 in the evaluator vs ~64 µs in its report).

## Layout

```
goal_fullshape/<operator>/<model>/
  FINAL.md                      the session's report
  <definition>/solution.json    FlashInfer-Bench solution as evaluated
  <definition>/<sources>        the same sources, unpacked
```

FA4 backward has two definitions (`attention_bwd_causal_h16_d128` for the MHA and GQA
workloads, `attention_bwd_varlen_causal_h16_d128` for varlen), built for the
[`kda05_reproduction/`](../kda05_reproduction/) runs.
