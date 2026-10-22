# DeepSeek Sparse Attention decode — harness ablation, `t8_topk2048`, Fable-5.1 (B200)

Three Fable-5.1 sessions on `t8_topk2048` 重负载 (8 decode rows selecting
288/4/1884/21/136/2048/42/335 pages, H=16, ckv 512, kpe 64, top-k 2048, page 64, bf16;
workload `564007ac`), from a harness ablation (2026-10-20). They vary the prompt and the
harness, not the model. Only the operator code (kernel + launcher) is included.

- `fable-5.1-ncu-skill` — interactive `/goal` session whose prompt asks it to use the
  NCU skill (`mit-han-lab/ncu-report-skill` at `c7980a4`);
- `fable-5.1-ncu-prompt` — the same `/goal` prompt, but with no skill installed and the
  NCU line reduced to "please use NCU for tuning analysis";
- `fable-5.1-flame-chase` — the same prompt without the NCU line and the `/goal` prefix,
  run under humanize2's `flame_chase` flow (two Fable-5.1 chasers taking turns, each turn
  a fresh session sharing the git workspace), 24 h budget, stopped after 13 h.

The reference (the KDA 0.5 kernel, see
[`goal_single_shape/dsa_decode_fwd/`](../../goal_single_shape/dsa_decode_fwd/)) was
installed on each session's machine as a run-only command. Device time and the verdict
come from our acceptance tool (`dsa-accept`, run on the node; the sessions could run it but
not read it): CUPTI kernel span with the L2 flushed every call, reference and candidate
alternated (2 x 3 trials each, median), every timed candidate call gets freshly allocated
`q_nope` / `q_pe` / `sparse_indices` buffers, and the inputs are overwritten in place after
timing and re-checked. Tolerance is 7x the reference's worst error against an fp32 naive.
The goal was 1.2x; none of the three reached it.

| arm | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|
| NCU skill (`/goal`) | `t8_topk2048/fable-5.1-ncu-skill/dsa_kernel.cu` (mma.sync + cp.async; build `build.sh` to a cubin, driver-API launcher `kernel.py`) | 8.912 µs | 10.144 µs | 1.138x | 8.3 h |
| NCU prompt (`/goal`) | `t8_topk2048/fable-5.1-ncu-prompt/dsa.cu` (2-CTA clusters, DSMEM exchange; JIT launcher `kernel.py`) | 8.704 µs | 10.136 µs | 1.164x ¹ | 6.6 h |
| flame chase | `t8_topk2048/fable-5.1-flame-chase/dsa_kernel.cu` (TMA row gathers; `load_inline` launcher `kernel.py`) | 9.672 µs | 10.144 µs | 1.049x | 12.1 h (of 13 h) |

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
  input-derived results was added to the prompt mid-run, and the kernel kept here is the
  best one under the fixed tool.
- The NCU-skill and flame-chase kernels only take their fast path when T x 64 fits in the
  resident CTAs (T ≤ 9 on 148 SMs); NCU skill asserts, flame chase falls back to a slow
  torch path. The NCU-prompt kernel has a generic path for other T and a hand-tuned
  mapping for T = 8.
- `build.sh` (NCU skill) hard-codes `/usr/local/cuda/bin/nvcc`; the NCU-prompt launcher
  needs `cuda-python` and writes its cubin cache into `_build/` next to itself.
