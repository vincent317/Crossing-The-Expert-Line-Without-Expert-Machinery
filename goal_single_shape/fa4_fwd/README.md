# FlashAttention-4 forward — /goal-mode kernels (B200)

Final kernels produced by interactive Claude Code `/goal` sessions (Fable-5.1, run with
`--safe-mode`), each written from scratch without reading the reference kernel. Only the
operator code (kernel + launcher) is included; benchmarks, probes and test harnesses are
omitted.

Operator: attention forward, bf16, head dim 128, layout `(B, S, H, D)`, softmax scale
1/√128, causal masks bottom-right aligned (S_q = S_kv). Twelve cases: batch 8 × seq 4096 or
batch 1 × seq 32768; MHA (16/16 heads) or GQA (32 q heads over 4 or 8 kv heads, the
Qwen3-30B-A3B and Qwen3-8B attention layouts); non-causal or causal.

The reference is FlashAttention-4 (`flash_attn.cute`), installed on each session's B200 and
callable only as a black box. Goal: 1.2x FA4's device time. Device time is each session's
own measurement on its own B200, not re-measured on a common node: mostly the sum of the
call's CUDA kernel durations from `torch.profiler` (warm L2), with FA4 timed by the same
session — in some sessions in the same process, in others in a separate process, as
marked. Because the reference was re-timed differently per session (FA4 itself varies by a
few percent between runs and nodes), compare `vs reference` within a row rather than across
rows. Develop time is the session's active time, excluding the waits for the subscription's
rate-limit window to reset (wall clock 14–20 h for the 2026-09-25 batch).

| case | files | device time | FA4 | vs reference | develop time |
|---|---|---|---|---|---|
| `mha_b8_n4096_h16_noncausal` | `fa_fwd_v6.py` (CuTe DSL) | 751.1 µs | 935.5 µs | **1.245x** ¹ | 4.2 h |
| `mha_b8_n4096_h16_causal` | `fa_fwd.cu` + `dev_common.h` + `tmem_ops.h` (ctypes launcher `fa_lib.py`) | 538.5 µs | 460.0 µs | 0.854x ² ⁵ | 8.0 h |
| `mha_b1_n32768_h16_noncausal` | `fa_fwd2.py` (CuTe DSL) | 6.582 ms | 5.935 ms | 0.902x ¹ | 7.8 h |
| `mha_b1_n32768_h16_causal` | `fa_fwd2.cu` + `ptx.cuh` + `softmax_tile.cuh` (ctypes launcher `fa_mine.py`) | 4.083 ms | 3.169 ms | 0.776x ² | 8.1 h |
| `gqa_b8_n4096_h32_kv4_noncausal` | `fa_fwd_b200.cu` + `tmem_helpers.inc` (ctypes launcher `myfa.py`) | 2200.0 µs | 1558.2 µs | 0.708x ³ ⁶ | 8.2 h |
| `gqa_b8_n4096_h32_kv4_causal` | `fa_v9.py` (CuTe DSL + inline PTX) | 894.3 µs | 969.0 µs | 1.083x ⁴ ⁷ | 8.1 h |
| `gqa_b1_n32768_h32_kv4_noncausal` | `attn_v12.cu` + `tc.cuh` | 11.85 ms | 11.18 ms | 0.943x ⁴ ⁸ | 6.7 h |
| `gqa_b1_n32768_h32_kv4_causal` | `kernel/fa_fwd.cu` + `kernel/ptx.cuh` (build `kernel/build.sh`, ctypes launcher `fa.py`) | 6.496 ms | 5.754 ms | 0.886x ¹ | 4.1 h |
| `gqa_b8_n4096_h32_kv8_noncausal` | `fa_fwd.py` (CuTe DSL) | 1498.6 µs | 1406.4 µs | 0.938x ¹ | 11.5 h |
| `gqa_b8_n4096_h32_kv8_causal` | `fa2cta_v7e.cu` + `tmem_ops.h` (ctypes launcher `fa_lib.py`) | 1011.5 µs | 929.4 µs | 0.919x ² | 10.0 h |
| `gqa_b1_n32768_h32_kv8_noncausal` | `fa3_v12a.cu` + `common.cuh` + `tmem_ops.cuh` (torch extension, `build.py`) | 14.29 ms | 12.00 ms | 0.839x ² | 10.6 h |
| `gqa_b1_n32768_h32_kv8_causal` | `fa_fwd.cu` + `ptx.cuh` + `tmap.cuh` (ctypes launcher `mine.py`) | 6.40 ms | 6.03 ms | 0.942x ³ ⁷ | 5.8 h |

Timing of each row:
¹ FA4 and the kernel alternated in one process (several rounds; `mha_b8_n4096_h16_noncausal`:
median of 8 rounds, minimum ratio 1.223x; `mha_b1_n32768_h16_noncausal` and
`gqa_b8_n4096_h32_kv8_noncausal`: best round, median ratio 0.90–0.95x and 0.967x).
² FA4 timed in a separate process (for `mha_b8_n4096_h16_causal`, once at the start of the
session).
³ Same process, FA4 first, not alternated.
⁴ CUDA events over a sustained loop; FA4 in a separate process (`gqa_b8_n4096_h32_kv4_causal`)
or the same process (`gqa_b1_n32768_h32_kv4_noncausal`, minimum of cold starts).

Correctness was checked against an fp32 naive implementation with the tolerance set to 7x
FA4's own error against that naive. Unless noted, only the session's main input (randn,
seed 0) and a few smaller shapes were tested; `mha_b8_n4096_h16_noncausal` and
`gqa_b8_n4096_h32_kv8_noncausal` also passed adversarial inputs (growing, spiked and
stepped score ranges), and `mha_b1_n32768_h16_noncausal` passed with its rescale path
forced on. Two rows (⁷) pass only because the test inputs never exercise a shortcut they
take — see the notes.

Notes

- ⁵ `mha_b8_n4096_h16_causal`: the session's directory did not survive; the files were
  restored verbatim from the commands in its transcript that wrote them (`tmem_ops.h` by
  re-running the session's own string generator), at the moment the 538.5 µs version was
  built and timed. Later versions of that session were slower (best 568 µs).
- ⁶ `gqa_b8_n4096_h32_kv4_noncausal`: Fable-5.1 declined a number of requests in this
  session and Claude Code fell back to Opus-4.8 for about two hours, during which the kernel
  was restructured (Q back in shared memory, a 3-stage S pipeline, the issue order of the QK
  and PV MMAs, a cross-tile deadlock fix). Fable-5.1 then made small changes (wait hints,
  defaults). The kernel kept here is therefore partly written by Opus-4.8. It also leaves
  the emulated `exp2` unclamped (`EXP_CLAMP=0`), assuming exponents ≥ −126.
- ⁷ **Correctness shortcuts.** `gqa_b8_n4096_h32_kv4_causal` takes each row's reference
  maximum from its first key tile and never updates it (no online rescale; the file header's
  "lazy rescaling" does not match the code), emulates `exp2` without clamping and truncates P
  to bf16: scores more than about 88 above the first tile's maximum give wrong results. The
  measured configuration also needs `FA_TRUNC=1` (the default is 0).
  `gqa_b1_n32768_h32_kv8_causal` takes the row maximum from the diagonal block only, with no
  online rescale and `X_CLAMP=11`, accumulates row sums in fp16 and converts K/V to fp16
  (|s| must stay below 65504); attention-sink-like inputs give errors 40–50x over the
  tolerance. Both pass the tolerance on randn inputs only.
- ⁸ `gqa_b1_n32768_h32_kv4_noncausal`: the team results sheet lists 11.71 ms; that figure
  came from a parameter sweep of an earlier version (v9); the version the session delivered
  (v12, kept here) measures 11.83–11.88 ms. It scales K by `scale·log2e` and converts K and Q
  to fp16 (range risk for large inputs), and uses a static global scratch buffer (not
  thread-safe).
- `gqa_b8_n4096_h32_kv4_causal`: the session reported 1.134x for a further-truncated
  configuration against an FA4 time measured 14 h earlier; the same-session comparison gives
  the 1.083x above.
- Several kernels emulate a fraction of the `exp2` calls with an FMA polynomial to offload
  the MUFU unit, and several convert P to bf16 by truncation with a distribution-dependent
  bias correction (`mha_b8_n4096_h16_noncausal`), within the tolerance on the tested inputs.
- What the sessions read: none read FA4's source or SASS. Some read CUTLASS headers on a
  shared disk for UMMA descriptor / TMEM PTX syntax, `mha_b8_n4096_h16_noncausal` saw two
  lines of the CUTLASS Hopper `fmha.py` example in a `grep` result, and
  `gqa_b8_n4096_h32_kv8_noncausal` read the AVO paper's abstract.

Entry points and builds

- `mha_b8_n4096_h16_noncausal`: `flash_fwd(q, k, v, num_ctas=None)`; MHA only, S a multiple
  of 512; CuTe DSL JIT (`nvidia-cutlass-dsl` 4.6.2).
- `mha_b8_n4096_h16_causal`: `fa_lib.build()` runs nvcc (`-O3 -std=c++17`, sm_100a,
  `-shared -lcuda`), then `fa_lib.fa_fwd(q, k, v, G=8, grid=0)`; MHA, causal only; needs a
  zeroed int32[2] work counter (allocated by `fa_lib`).
- `mha_b1_n32768_h16_noncausal`: `attention_fwd(q, k, v, scale=None)`; B=1, MHA; CuTe DSL
  JIT (`nvidia-cutlass-dsl` 4.4.2). The `FA_*` environment switches default to the measured
  configuration; `FA_SOFTMAX_MODE=0` and `FA_NO_TMA_KV=1` are timing-only switches that give
  wrong results.
- `mha_b1_n32768_h16_causal`: build `nvcc -O3 -gencode arch=compute_100a,code=sm_100a
  -shared -Xcompiler -fPIC -DEMUL_EVERY=3 -o libfa_fwd.so fa_fwd2.cu` next to `fa_mine.py`,
  then `fa_fwd(q, k, v, causal=True)`. Launches on the legacy default stream.
- `gqa_b8_n4096_h32_kv4_noncausal`: `build.sh` (it `cd`s into the session's directory
  `/root/fa4work`; run the nvcc line from this directory instead), then
  `my_attn(q, k, v, scale)`.
- `gqa_b8_n4096_h32_kv4_causal`: `attn(q, k, v)` with `FA_EMUL=32 FA_TRUNC=1`; CuTe DSL JIT.
- `gqa_b1_n32768_h32_kv4_noncausal`: `extern "C" int attn_fwd(q, k, v, out, B, S, HQ, HKV,
  softmax_scale, emul_every, stream, dbg)` (S a multiple of 256); build `nvcc -O3 -gencode
  arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared -maxrregcount=96 attn_v12.cu`; no
  Python launcher (the session's lived in its benchmark script). `emul_every >= 100` is a
  diagnostic mode with wrong results.
- `gqa_b1_n32768_h32_kv4_causal`: `kernel/build.sh`, then `fa.fa_fwd(q, k, v)`.
- `gqa_b8_n4096_h32_kv8_noncausal`: `flash_fwd(q, k, v)`; non-causal, S a multiple of 256;
  CuTe DSL JIT; hard-codes 148 SMs.
- `gqa_b8_n4096_h32_kv8_causal`: `nvcc -gencode arch=compute_100a,code=sm_100a -O3
  -std=c++17 -shared -Xcompiler -fPIC -lcuda -o libfa.so fa2cta_v7e.cu`, then
  `fa_lib.fa_fwd(q, k, v)`; Hq/Hkv even, S a multiple of 128.
- `gqa_b1_n32768_h32_kv8_noncausal`: `build.build(src="fa3_v12a.cu")` (torch
  `cpp_extension.load`, `-lcuda`; its default source and build paths point at the session's
  `/root/work/`), then `fa_fwd(q, k, v, softmax_scale, num_ctas, prof)`.
- `gqa_b1_n32768_h32_kv8_causal`: build with `nvcc -O3 -gencode
  arch=compute_100a,code=sm_100a -shared -Xcompiler -fPIC -DUSE_FMAX3 -DMBAR_HINT=10000
  -DNPOLY=0 -DNO_RESCALE -o libfa.so fa_fwd.cu` (**these flags are the measured
  configuration**; the source defaults differ), then `mine.fa_fwd(q, k, v)` (loads
  `./libfa.so` or `$FA_LIB`).
- Debug / trace buffers and compile-time ablation switches are left in the sources, off by
  default.
