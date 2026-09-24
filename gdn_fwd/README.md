# Gated DeltaNet prefill forward — /goal-mode kernels (B200)

Final kernels produced by `/goal` sessions, each written from scratch without reading
the reference kernel. Only the operator code is included; benchmarks, probes and test
harnesses are omitted.

The reference is the KDA 0.5 CuTe-DSL Gated DeltaNet prefill kernel from the MLSys 2026
FlashInfer contest solution (`mit-han-lab/mlsys2026-flashinfer-contest-solution`, commit
`7aad0429`). The workloads are the contest's own, pinned to HuggingFace dataset
`flashinfer-ai/mlsys26-contest` revision `5e832ce8`.

Device time is whole-call CUDA-graph replay time (50 iterations per trial, 3 trials,
minimum taken), measured for the candidate and the reference back to back on the same
idle B200 in the same process. `vs reference` = reference / candidate; > 1 means faster
than the reference. Develop time is the agent's *active* time — wall clock minus the
waits for the subscription's rate-limit window to reset; wall clock was 2.1-3.1x larger.

| case | model | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|---|
| `t8192_n20` 多条混合长度 (T=8192 in N=20 seqs, q/k [T,4,128] v [T,8,128] bf16, D=128) | Opus-4.8 | `t8192_n20/opus-4.8/gdn_triton.py` | 155.6 µs | 124.7 µs | 0.801x | 9.0 h |
| `t30_n1` 单条短序列 (T=30, N=1, same layout) | Opus-4.8 | `t30_n1/opus-4.8/gdn_cuda.py` | 9.16 µs | 4.12 µs | 0.450x | 13.1 h |
| `t5709_n2` 少数几条长序列 (T=5709 in N=2 seqs: 5637/72, same layout) | Opus-4.8 | `t5709_n2/opus-4.8/gdn_triton.py` | 233.8 µs | 100.8 µs | 0.431x | 10.3 h |

Shapes and config, common to all three cases: GVA with 4 q/k heads mapped onto 8 v heads,
head dim D=128, varlen packing addressed by `cu_seqlens`, recurrent state `[N,8,128,128]`
fp32 in k-last layout, bf16 q/k/v. Entry point `gdn_prefill_fwd(q, k, v, state, A_log, a,
dt_bias, b, cu_seqlens, scale)` returning `(out, new_state)`.

Notes

- `t30_n1/opus-4.8/gdn_cuda.py` JIT-builds its CUDA source through
  `torch.utils.cpp_extension.load_inline` with `-arch=sm_100a`; the other two are Triton.
- `t5709_n2/opus-4.8/gdn_triton.py` runs its phases on three side CUDA streams, so its
  device time depends on stream overlap being preserved by the caller.
- Correctness was checked against an fp32 naive implementation of the operator (the
  contest definition's own torch reference), with the tolerance taken from the reference
  kernel's error against that same naive: a candidate passes only if its relative L2 error
  is within 7x the reference's. All three passed on six input families (the frozen
  workload, fresh random, correlated keys, high/low beta, and long decay).
- These sessions were run headless (`claude -p`) rather than through the interactive
  `/goal` command: a Stop hook held the session open until the goal was met, a supervisor
  resumed it across the subscription's rate-limit windows, and the reference kernel lived
  only on a separate judge node that scored candidates shipped to it — so the agent never
  had the reference source on its own machine.
