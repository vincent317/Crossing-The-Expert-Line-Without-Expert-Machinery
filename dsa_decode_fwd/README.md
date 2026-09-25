# DeepSeek Sparse Attention — paged MLA sparse decode forward — /goal-mode kernels (B200)

Final kernels produced by `/goal` sessions, each written from scratch without reading the
reference kernel. Only the operator code is included; benchmarks, probes and test
harnesses are omitted.

The reference is the KDA 0.5 CuTe-DSL sparse MLA decode kernel from the MLSys 2026
FlashInfer contest solution (`mit-han-lab/mlsys2026-flashinfer-contest-solution`, commit
`7aad0429`). The workload is the contest's own, pinned to HuggingFace dataset
`flashinfer-ai/mlsys26-contest` revision `5e832ce8`.

Device time is kernel-span time: the sum of `self_device_time_total` over the call's
kernels from `torch.profiler`, divided by the iteration count. The reference is not
CUDA-graph capturable here (its capture comes back empty), so both sides are timed this
way. `vs reference` = reference / candidate; > 1 means faster than the reference.
Develop time is the agent's *active* time, excluding rate-limit waits; wall clock was 3.8x
larger.

| case | model | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|---|
| `t2_topk2048` 长短混合 (2 decode rows, H=16, ckv 512, kpe 64, top-k 2048, page 64, pool 8462, bf16) | Opus-4.8 | `t2_topk2048/opus-4.8/dsa.cu` (+ JIT launcher `dsa_load.py`) | 19.75 µs | 6.53 µs | 0.331x | 7.2 h |

Entry point `dsa_decode_fwd(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)`
returning `(out, lse)`. The two decode rows select 6 and 337 pages respectively out of the
top-k budget, so the kernel has to stay efficient across a 50x spread in work per row.

Notes

- `dsa_load.py` JIT-builds `dsa.cu` via `torch.utils.cpp_extension.load` from its own
  directory; the architecture is left to torch's detection (sm_100 on B200).
- Correctness was checked against an fp32 naive implementation of the operator, with the
  tolerance taken from the reference kernel's error against that same naive: a candidate
  passes only if its relative L2 error is within 7x the reference's. This kernel passed on
  both the frozen workload and fresh random inputs, with error roughly 30x *below* the
  reference's own.
- These sessions were run headless (`claude -p`) rather than through the interactive
  `/goal` command: a Stop hook held the session open until the goal was met, a supervisor
  resumed it across the subscription's rate-limit windows, and the reference kernel lived
  only on a separate judge node that scored candidates shipped to it — so the agent never
  had the reference source on its own machine.

## Interactive `/goal` sessions (Opus-5, Fable-5.1)

Same reference kernel family as above (the KDA 0.5 contest kernel; published times
10.293 / 3.365 / 6.805 µs for the three workloads), but a different session setup and
timing method, so the rows are kept separate. These sessions ran through the interactive
`/goal` command with the reference installed on their own machine as a black box; their
transcripts were audited and contain no read of the reference source.

Device time follows the contest's official protocol (`compare_human_best`): CUPTI
kernel span with the L2 flushed (256 MB) before every iteration, 3 warm-up / 50 timed
iterations, median per trial, mean of 3 trials; candidate and reference back to back on
the same idle B200. `vs reference` = reference / candidate.

| case | model | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|---|
| `t8_topk2048` 重负载 (8 decode rows selecting 288/4/1884/21/136/2048/42/335 pages, same H/ckv/kpe/page layout) | Opus-5 | `t8_topk2048/opus-5/dsa_kernel.cu` (+ ctypes launcher `dsa_run.py`) | 9.755 µs | 10.379 µs | 1.064x | 9 h |
| `t2_topk2048_short` 小请求 (2 decode rows selecting 18/11 pages) | Opus-5 | `t2_topk2048_short/opus-5/dsa_kernel.cu` (+ JIT launcher `mykernel.py`) | 3.286 µs | 3.451 µs | 1.050x | 4 h |
| `t2_topk2048` 长短混合 | Fable-5.1 | `t2_topk2048/fable-5.1/dsa_decode.cu` (+ JIT launcher `dsa_mine.py`) | 5.952 µs | 6.853 µs | 1.151x | 1.5 h |

Entry points: `run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)`
(`mykernel.py`, `dsa_mine.py`); `Runner(T, variant=150, so=...)(q_nope, q_pe, ckv_cache,
kpe_cache, sparse_indices, sm_scale)` for `t8_topk2048`, whose library is built with
`nvcc -O3 -std=c++17 -arch=sm_100a --shared -Xcompiler -fPIC -o libdsa.so dsa_kernel.cu`.
All return `(out, lse)`.

Notes

- The cold-L2 protocol costs these kernels more than the reference: with a warm L2 the
  same runs give 1.177x / 1.132x / 1.267x. The numbers the sessions reported for themselves
  (warm L2, nsys) were higher still; the table uses the official cold-L2 figures.
- Correctness used the contest definition's thresholds (atol = rtol = 0.01, all elements
  matched) against its fp32 naive reference.
