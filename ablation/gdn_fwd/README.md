# Gated DeltaNet prefill — harness ablation, `t8192_n20`, Fable-5.1 (B200)

Three Fable-5.1 sessions on `t8192_n20` 多条混合长度 (T=8192 in N=20 seqs, q/k [T,4,128]
v [T,8,128] bf16, D=128, fp32 state `[N,8,128,128]` k-last; workload `a9540651`). They vary
the prompt and the harness, not the model. The reference is the KDA 0.5 GDN prefill kernel
(see [`goal_single_shape/gdn_fwd/`](../../goal_single_shape/gdn_fwd/)). Only the operator
code (kernel + launcher) is included.

## NCU skill and humanize2 `flame_chase` (2026-10-20)

- `fable-5.1-ncu-skill` — interactive `/goal` session whose prompt asks it to use the
  NCU skill (`mit-han-lab/ncu-report-skill` at `c7980a4`); the reference was installed on
  its machine as a run-only command;
- `fable-5.1-flame-chase` — the same prompt without the NCU line and the `/goal` prefix,
  run under humanize2's `flame_chase` flow (two Fable-5.1 chasers taking turns, each turn
  a fresh session sharing the git workspace), 24 h budget, stopped after 13 h.

Both numbers are the sessions' own acceptance runs on their own B200: CUPTI kernel span
with the L2 flushed every call (`flashinfer.testing.bench_gpu_time`), 5 alternating rounds
of reference and candidate, median. Correctness is the 7x-reference rule against an fp32
naive on output and new state. Not independently re-verified.

| arm | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|
| NCU skill (`/goal`) | `t8192_n20/fable-5.1-ncu-skill/gdn.cu` (mma.sync + cp.async.bulk, PDL; build `build.sh`, ctypes launcher `gdn.py`) | 86.56 µs | 125.22 µs | 1.447x | 2.0 h |
| flame chase | `t8192_n20/fable-5.1-flame-chase/gdn_kernel.cu` (tcgen05 UMMA + TMA, PDL; ctypes launcher `gdn_cuda.py`) | 97.09 µs | 125.20 µs | 1.290x | 6.9 h (of 13 h) |

Entry points: `gdn.run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)` and
`gdn_cuda.gdn_forward(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)`, both
returning `(out, new_state)` and loading `libgdn.so` from their own directory. Build:
`nvcc -O3 -std=c++17 -arch=sm_100a --shared -Xcompiler -fPIC -o libgdn.so gdn.cu`
(`-gencode arch=compute_100a,code=sm_100a` for `gdn_kernel.cu`).

Notes

- Both are two kernels per call: a chunk-parallel prepass (gates, cumulative decay,
  KKᵀ/QKᵀ, T=(I+A)⁻¹ in the WY form, 64-token chunks) and a per-(sequence, v-head, v-slice)
  recurrence with the fp32 state in registers, launched with programmatic dependent launch.
  The flame-chase kernel reformulates each chunk step as two back-to-back matmuls on UMMA.
- NCU skill: once the session's Stop hook rejected an acceptance run that timed the
  reference as a subprocess, the session inspected the reference launcher (file header,
  `ps`/`/proc` to see which script it starts) and then ran that entry script in-process
  with `runpy` to time both sides in one process. It did not read the reference source,
  but this goes beyond calling the command.
- Flame chase: its acceptance script timed the reference in a separate process, although
  the prompt asked for alternating timing in one process. Its best kernel was committed
  6.9 h in; the remaining ~6 h were ~125 short turns that only re-ran acceptance.

## KernelWiki `/goal` session (2026-09-21)

An earlier Fable-5.1 session whose prompt pointed it at KernelWiki
(`mit-han-lab/KernelWiki` at `b6b4301`, copied into its working directory) as reference
notes. It ran through the interactive `/goal` command with `--safe-mode` (no user skills,
memory, hooks or `CLAUDE.md` loaded; `/goal`'s own Stop hook still applies). Unlike the
sessions above, the reference kernel was not on its machine at all: its source was deleted
before the session started, and the session got only the reference's outputs on the target
workload and its measured time (124.93 µs).

Device time is from the harness provided on its machine: the sum of all CUDA kernel
durations of one call from the CUPTI trace, L2 flushed with a 512 MB write before every
call and the flush's own time subtracted, 10 warm-up / 50 timed calls per trial, median of
5 trials; B200 clock locked at 1965 MHz. The reference time was measured with the same
function on the same machine before its source was removed; the two were not timed in one
process. Correctness is a 7x-reference rule against the contest definition's fp32 naive,
on max abs error and relative RMS of both output and new state, and passed on the target,
the four other contest workloads with this definition, and 14 inputs the session generated
(single token, single sequence, lengths not a multiple of the chunk, gates and betas at
both ends, zero and large state).

| arm | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|
| KernelWiki (`/goal`) | `t8192_n20/fable-5.1-kernelwiki/gdn_triton.py` | 103.81 µs | 124.93 µs | 1.2035x | 4.3 h (4.9 h wall) |

Entry point `gdn_triton.run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)`
returning `(out, new_state)`; pure Triton, no build step.

Notes

- The 103.81 µs is the median of three separate processes (103.81 / 103.83 / 103.81 µs);
  a re-run by the operator after the session gave 103.81 / 103.87 / 103.79 µs. The margin
  over the 1.2x target (104.1 µs) is only 0.3 µs, and without same-process paired timing
  against the reference it is within what a re-anchored reference could move.
- Design (five launches): a table-building kernel, a chunk-parallel bulk kernel (gates,
  cumulative decay, WY-form inverse, small per-chunk factors), a pipelined chain over
  fixed-length segments of each sequence with the state held as `[V, K]`, a short chain
  that stitches the segments of multi-segment sequences, and an output pass for the later
  chunks. The session's versions went v0 621 µs → v23 128.6 µs → v58 103.8 µs.
- At 2 h 58 min, on v23 (0.972x), the session wrote that it saw no credible path to 1.2x
  and tried to finish; the `/goal` Stop hook kept it going, and a supervisor script sent
  it a fixed "keep going" message after 10+ minutes of inactivity (5 times in total). It
  reached the target 1 h 56 min later and wrote that its earlier "flat optimum ~128 us" and
  "floor ~106 us" estimates were wrong.
- What it read: KernelWiki pages on Blackwell hardware, languages and optimization
  patterns, plus the wiki's `gated-delta-net.md` page (which holds no recurrence
  implementation); a CuTe-DSL GEMM tutorial kept in KernelWiki; library sources installed
  on the node (Triton, the CUTLASS CuTe-DSL package, FlashAttention's CuTe kernels); and, at
  3 h 27 min, about 110 lines of the Mamba2 SSD example that ships with the CUTLASS
  CuTe-DSL examples in FlashInfer's package — a chunked state-space scan, not a Gated
  DeltaNet kernel, and outside the prompt's ban list. It ran no web search or fetch, and
  read no Gated DeltaNet implementation.
- Early on, `num_stages > 1` silently produced wrong results in one of its Triton chain
  kernels (output rms error 0.13) and the session switched to `num_stages=1` everywhere;
  it later re-enabled `num_stages=3` on the segment chain, which the final kernel uses
  and where the full correctness battery passes.
