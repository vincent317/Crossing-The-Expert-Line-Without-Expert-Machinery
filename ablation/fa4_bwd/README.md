# FlashAttention-4 backward — harness ablation, `varlen_t32768_n13_h16`, Fable-5.1 (B200)

Five Fable-5.1 sessions on the varlen case (varlen MHA, 13 sequences packed into
T=32768, H=16, D=128, causal, bf16), all given the same `/goal` prompt as the
`varlen_t32768_n13_h16` Fable-5.1 row of
[`goal_single_shape/fa4_bwd/`](../../goal_single_shape/fa4_bwd/), except for one change
each. They vary the prompt and the harness, not the model. Only the operator code (kernel +
launcher) is included. All of them wrote the kernel from scratch and called the reference
(flash-attn-4 `4.0.0b19`) only as a black box.

## Prompt ablation

- `fable-5.1-kernelwiki` — the prompt additionally points at the KernelWiki repository;
- `fable-5.1-ncu` — the prompt additionally asks for NCU profiling;
- `fable-5.1-no-accuracy-constraint` — the correctness clause (fp32 naive + 7x tolerance)
  is removed from the prompt.

Timing caliber differs from `goal_single_shape/fa4_bwd/`: these rows were **not**
re-measured on a fresh idle node under the `ncu` all-kernel protocol. Each number is the
session's own measurement on its own B200 — device-kernel time summed over preprocess,
main kernel and postprocess (torch profiler, 20 iterations, median of repeated runs; the
no-accuracy-constraint row is a CUDA-graph replay total) — against a reference the same
session measured on the same machine. The three self-measured references disagree
(1633 / 1544.6 / 1775 µs, against 1639 µs measured separately), so `vs reference` is
meaningful within a row but not across rows.

| arm | files | device time | reference (self-measured) | vs reference | develop time |
|---|---|---|---|---|---|
| Kernel Wiki | `varlen_t32768_n13_h16/fable-5.1-kernelwiki/fa_bwd_kernel.py` (CuTe DSL; launcher `fa_bwd.py` with Triton pre/post-processing) | 1619 µs | 1633 µs | 1.009x | 13 h |
| NCU | `varlen_t32768_n13_h16/fable-5.1-ncu/fa4bwd_varlen_gluon.py` (Gluon) | 1286.6 µs | 1544.6 µs | **1.200x** | 16 h |
| no accuracy constraint | `varlen_t32768_n13_h16/fable-5.1-no-accuracy-constraint/fa4_bwd_varlen_gluon.py` (Gluon) | 1766 µs | 1775 µs | 1.005x | 8 h |

Entry points: `FaBwd(q, k, v, o, do, lse, cu_seqlens)`, then `.compile()` and `.run()`
(Kernel Wiki); `VarlenBwd(cu_seqlens_cpu, H, T, device)(q, k, v, o, do, lse, scale)` (NCU);
`flash_bwd_varlen(q, k, v, o, do, lse, cu_seqlens, max_seqlen)` (no accuracy constraint).
All return `(dq, dk, dv)`.

Notes

- The Kernel Wiki and NCU arms followed the fp32-naive + 7x-tolerance protocol themselves.
  The no-accuracy-constraint arm did not — with the clause dropped from its prompt it only
  compared against FA4 directly, with no fp32 gold standard — so its kernel was audited
  afterwards under the standard protocol on an idle B200: dq/dk/dv max-abs error vs the
  fp32 naive is 0.015203/0.017755/0.017920 against 7x-FA4 tolerances
  0.106418/0.124287/0.125443, i.e. identical to FA4's own error. Removing the correctness
  requirement bought no speed here.
- That arm also produced an fp8 path (1340.6 µs, 1.28x) which is excluded: it assumes the
  bf16 to fp8 cast is fused into the forward pass and thus leaves the cast out of the
  measurement, and it reports gradient errors of 7% and above.
- The Kernel Wiki arm stopped at 1.009x with a structural argument: on its machine the
  1.2x goal (<= 1361 µs for the whole call) is below FA4's own main kernel alone
  (1459 µs), so the arm has to beat FA4's main kernel *and* absorb the pre/post-processing
  cost. The no-accuracy-constraint arm reached the same conclusion from the other side: its
  compute floor with the dQ epilogue removed is 1433 µs, leaving no room for the 2.4 GB
  fp16 read-modify-write of the dQ reduction.
- The NCU arm crossed the goal by keeping K/V resident in smem per work unit, a 3-stage
  Q/dO TMA pipeline and an fp16 TMA reduce-add into a per-document 64-aligned dQ
  accumulator; its remaining limits are a ~72% active tensor pipe (N=64 issue cadence) and
  a power-capped main kernel (~1.55 GHz at 148 CTAs).
- Develop times are the sessions' wall times.

## NCU skill and humanize2 `flame_chase` (2026-10-20)

- `fable-5.1-ncu-skill` — interactive `/goal` session whose prompt asks it to use the
  NCU skill (`mit-han-lab/ncu-report-skill` at `c7980a4`, installed in the session's
  otherwise empty Claude Code config);
- `fable-5.1-flame-chase` — the same prompt with the NCU line and the `/goal` prefix
  removed, run instead under humanize2's `flame_chase` flow: two Fable-5.1 chasers take
  turns, each turn a fresh Claude Code session sharing only the git workspace. It had a
  24 h budget and was stopped after 13 h.

Both numbers are the sessions' own measurements on their own B200 (SM clock capped at
1.5 GHz), warm L2, not re-measured on a fresh node and not under the `ncu` protocol of
`goal_single_shape/fa4_bwd/`: `fable-5.1-ncu-skill` sums the call's kernel durations from
`torch.profiler` (30 iterations) and compares against a reference time the same session
measured the same way in a separate run; `fable-5.1-flame-chase` times a CUDA-graph replay
of the whole call (100 replays) with the reference in the same process. Develop time runs
from session start to the version kept here.

| arm | files | device time | reference (self-measured) | vs reference | develop time |
|---|---|---|---|---|---|
| NCU skill (`/goal`) | `varlen_t32768_n13_h16/fable-5.1-ncu-skill/fa4bwd.cu` + `sm100_utils.cuh` (raw PTX tcgen05/UMMA/TMA + CuTe headers; build `build.sh`, ctypes launcher `fa4bwd.py`) | 1443.6 µs | 1776.5 µs | 1.231x | 9.4 h |
| flame chase | `varlen_t32768_n13_h16/fable-5.1-flame-chase/fa_bwd_v4.py` (Gluon; `cluster_launch.py` patches Triton's launcher for 2-CTA clusters) | 2734.3 µs | 1766.1 µs | 0.646x | 4.6 h (of 13 h) |

Entry points: `fa4_bwd(q, k, v, o, do, lse, cu_seqlens, softmax_scale)` with q/k/v/o/do
`(T, H, 128)` bf16 and lse `(H, T)` fp32 (NCU skill; loads `libfa4bwd.so` from its own
directory); `FaBwd4(cu_seqlens, T, H)(q, k, v, o, do, lse, scale)` (flame chase; imports
`kernel.cluster_launch`, so it expects to live in a package named `kernel`). Both return
`(dq, dk, dv)`.

Notes

- Both pass the fp32-naive + 7x-FA4 tolerance (NCU skill dq/dk/dv max-abs
  0.0172/0.0133/0.0179 against 0.106/0.124/0.125).
- NCU skill: a persistent 148-CTA kernel over (doc, head, 128-key block) items, longest
  doc first, K/V resident in smem, Q/dO streamed through 4- and 3-stage TMA rings, the lse
  and delta subtractions folded into the S and dP MMAs as an extra K=16 step, P and dS kept
  in TMEM for the dV/dK MMAs, dQ accumulated in fp16 with `red.global.add.noftz.f16x2`.
  `build.sh` hard-codes the CUTLASS headers shipped inside the image's flashinfer package
  and `/usr/local/cuda/bin/nvcc`.
- Flame chase: the later chaser rewrote the kernel in CUDA C++ and reported 0.685x for an
  uncommitted version minutes before the stop; that exact version was overwritten by a
  further edit and is lost, so the last committed and accepted version (0.646x) is kept.
