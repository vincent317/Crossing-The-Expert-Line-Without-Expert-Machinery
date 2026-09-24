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
