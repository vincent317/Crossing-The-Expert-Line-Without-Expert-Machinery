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

## NCU `/goal` sessions and humanize2 ablation — `t8_topk2048`, Fable-5.1 (2026-10-20)

Three more Fable-5.1 sessions on `t8_topk2048` 重负载 (workload `564007ac`), from a
harness ablation. Their full trajectories are included (`<arm>/trajectory/`, see the
top-level README).

- `fable-5.1-ncu-skill` — interactive `/goal` session whose prompt asks it to use the
  NCU skill (`mit-han-lab/ncu-report-skill` at `c7980a4`);
- `fable-5.1-ncu-prompt` — the same `/goal` prompt, but with no skill installed and the
  NCU line reduced to "please use NCU for tuning analysis";
- `fable-5.1-flame-chase` — the same prompt without the NCU line and the `/goal` prefix,
  run under humanize2's `flame_chase` flow (two Fable-5.1 chasers taking turns, each turn
  a fresh session sharing the git workspace), 24 h budget, stopped after 13 h.

The reference (the KDA 0.5 kernel) was installed on each session's machine as a run-only
command. Device time and the verdict come from our acceptance tool (`dsa-accept`, run on
the node; the sessions could run it but not read it): CUPTI kernel span with the L2
flushed every call, reference and candidate alternated (2 x 3 trials each, median), every
timed candidate call gets freshly allocated `q_nope` / `q_pe` / `sparse_indices` buffers,
and the inputs are overwritten in place after timing and re-checked. Tolerance is 7x the
reference's worst error against an fp32 naive. The goal was 1.2x; none of the three
reached it.

| arm | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|
| NCU skill (`/goal`) | `fable-5.1-ncu-skill/dsa_kernel.cu` (mma.sync + cp.async; build `build.sh` to a cubin, driver-API launcher `kernel.py`) | 8.912 µs | 10.144 µs | 1.138x | 8.3 h |
| NCU prompt (`/goal`) | `fable-5.1-ncu-prompt/dsa.cu` (2-CTA clusters, DSMEM exchange; JIT launcher `kernel.py`) | 8.704 µs | 10.136 µs | 1.164x ¹ | 6.6 h |
| flame chase | `fable-5.1-flame-chase/dsa_kernel.cu` (TMA row gathers; `load_inline` launcher `kernel.py`) | 9.672 µs | 10.144 µs | 1.049x | 12.1 h (of 13 h) |

¹ Measured on another node with a normal reference time. On its own node the session
recorded PASS at 1.202x (reruns 1.198–1.202), but that node's reference ran at 12.67 µs
against ~10.14 µs elsewhere, so the PASS does not hold.

Entry point (all three): `kernel.py` defines `run(q_nope, q_pe, ckv_cache, kpe_cache,
sparse_indices, sm_scale)` returning `(out, lse)`.

Notes

- All three split the work into small key chunks, one per CTA (512 CTAs at T=8, all
  co-resident), and merge the partials in the same launch: the CTAs then combine another
  request's partials by polling tagged (m, l) / partial words, so no grid sync or second
  kernel is needed. The two `/goal` kernels hide a per-launch tag in the low bits of the
  bf16 partials, which raises the output error (NCU skill: max-abs 4.2e-2 against a
  tolerance of 8.9e-2).
- NCU skill: an earlier version scored 1.206x under the first version of the acceptance
  tool, but it built an input-dependent work schedule on the host from `sparse_indices` and
  cached it under the indices tensor's `data_ptr()`, which that tool did not catch because
  it reused one buffer. The tool was fixed (fresh buffers per call), a rule against caching
  input-derived results was added to the prompt (`trajectory/prompt.v1.txt` is the
  original), and the kernel kept here is the best one under the fixed tool.
- The NCU-skill and flame-chase kernels only take their fast path when T x 64 fits in the
  resident CTAs (T ≤ 9 on 148 SMs); NCU skill asserts, flame chase falls back to a slow
  torch path. The NCU-prompt kernel has a generic path for other T and a hand-tuned
  mapping for T = 8.
- `build.sh` (NCU skill) hard-codes `/usr/local/cuda/bin/nvcc`; the NCU-prompt launcher
  needs `cuda-python` and writes its cubin cache into `_build/` next to itself.
