# Kimi Delta Attention forward — /goal-mode kernels (B200)

Final kernels produced by native Claude Code `/goal` sessions, each written from
scratch without reading the reference kernel. Only the operator code (kernel +
its launcher) is included; benchmarks, probes and test harnesses are omitted.

The reference is the CAKE `recurrent_kda` B200 backend (FlashKDA, called through
`flashinfer.recurrent_kda`) for the same case. Device time and the ratio to the
reference are the numbers each session measured itself on B200; ratio > 1 means
faster than the reference. Device time is the sum of the kernels' device time for
one call (torch.profiler, or CUPTI with cold L2 for the two Fable-5.1 `h96` rows),
except where marked ¹.

¹ `h64_fixed8192` / Opus-4.8: timed with CUDA events around 500 back-to-back calls,
so the figure includes launch gaps between its four Triton kernels and is not a pure
device time; the ratio is against the published CAKE reference time (0.470 ms), not
a reference measured in the same session.

² `h96_mixed` / Opus-4.8: the wall time is the agent's *active* time, excluding the waits
for the subscription's rate-limit window to reset; wall clock was 23.6 h. Its ratio is
against a reference measured in the same process on the same idle GPU (383.0 µs, the CAKE
`persistent_m128_h96_lpt` route), not a published figure. The session was run headless (`claude -p`) rather than
through the interactive `/goal` command, with a Stop hook holding it open until the goal
was met and the reference living only on a separate judge node.

| case | model | files | device time | vs reference | wall time |
|---|---|---|---|---|---|
| `h64_fixed8192` 单条定长长序列 (B=1, T=8192, H=64, D=128) | Opus-4.8 | `h64_fixed8192/opus-4.8/kda_triton12.py` (+ `kda_triton8.py`) | 1280 µs¹ | 0.37x¹ | 20 h |
| | Opus-5 | `h64_fixed8192/opus-5/kda_k2.cu` (+ ctypes launcher `kda_cuda2.py`) | 520.2 µs | 0.916x | 6 h |
| | Fable-5.1 | `h64_fixed8192/fable-5.1/kda_v21.py` (Gluon) | 467.1 µs | 1.02x | 10.53 h |
| `h96_uniform` 等长打包 (T=8192 = 8×1024, H=96, D=128) | Opus-4.8 | `h96_uniform/opus-4.8/triton_kda_best.py` | 1644.5 µs | 0.247x | 14 h |
| | Opus-5 | `h96_uniform/opus-5/kda_fwd.cu` + `kda_common.cuh` (+ ctypes launcher `kda_cuda.py`) | 876.3 µs | 0.47x | 16 h |
| | Fable-5.1 | `h96_uniform/fable-5.1/kda_cuda49.cu` (+ JIT launcher `kda_cuda49.py`) | 505.7 µs | 0.83x | 11 h |
| `h96_mixed` 混合长度打包 (T=8192 = 6 seqs, H=96, D=128) | Fable-5.1 | `h96_mixed/fable-5.1/kda_fwd_best_legacy_mma.cu` (+ JIT launcher `mine.py`) | 688 µs | 0.567x | 6 h |
| | Opus-4.8 | `h96_mixed/opus-4.8/kda_triton.py` | 1362.0 µs | 0.281x | 3.4 h² |

Notes

- CUDA kernels target `sm_100a` (`-gencode=arch=compute_100a,code=sm_100a`).
- `h96_uniform/opus-5/kda_cuda.py` loads a prebuilt `libkda.so` from a hard-coded
  path (`/root/kda/libkda.so`); build `kda_fwd.cu` into a shared library there first.
- `h96_mixed/fable-5.1/mine.py` imports `make_case` from the session's input
  generator (`common.py`, not included); only `run_mine` is needed to call the kernel.
