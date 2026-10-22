# MLA paged decode — /goal-mode kernels (B200)

Final kernels produced by interactive Claude Code `/goal` sessions (Fable-5.1), each
written from scratch without reading the reference kernel's device code. Only the operator
code is included; benchmarks, probes and test harnesses are omitted.

Operator: DeepSeek-V3 MLA decode (TP8 slice) — H=16, d_ckv=512, d_kpe=64, page size 1,
bf16, inputs `q_nope (B,16,512)`, `q_pe (B,16,64)`, `ckv_cache (P,1,512)`,
`kpe_cache (P,1,64)`, `kv_indptr (B+1)`, `kv_indices`, `sm_scale` → `out (B,16,512)` bf16,
`lse (B,16)` f32. The three cases are flashinfer-trace workloads `951de642`, `c4e14a53`
and `5bef8d88`.

The reference is the fastest library kernel the prompt named, FlashInfer's MLA decode
(which falls back to its `fa2` backend on B200); FlashInfer's native `trtllm-gen` backend
was not measured. Device time is each session's own measurement on its own B200: the sum
of the call's kernel durations from `torch.profiler` / CUPTI, the same inputs called in a
loop (warm L2), reference measured on the same machine. `vs reference` = reference /
candidate. Develop time is the session's wall time to the version kept here.

| case | files | device time | reference (same machine) | vs reference | develop time |
|---|---|---|---|---|---|
| `b1_kv408` 单请求中等长度 (B=1, kv 408) | `b1_kv408/fable-5.1/mla_decode_v4.cu` (kernel + C launcher) | 5.42 µs | 13.55 µs | 2.50x ¹ | 0.5 h |
| `b16_kv9657` 中批次混合长度 (B=16, kv 9657 total) | `b16_kv9657/fable-5.1/mla_decode.cu` (+ ctypes launcher `mine.py`) | 13.17 µs | 27.25 µs | 2.07x ² | 1.0 h |
| `b64_kv75145` 大批次 (B=64, kv 510–9197, 75145 total) | `b64_kv75145/fable-5.1/mla_decode.cu` (kernel + C launcher) | 46.71 µs | 72.39 µs | 1.55x ³ | 0.8 h |

¹ CUDA-graph replay: 6.48 µs vs 15.13 µs. The team results sheet lists 6.48 µs and 2.5x
against the prompt's quoted 15.98 µs.

² Median of 5 alternating rounds in one process; CUDA-graph replay 14.36 µs vs 28.70 µs
(2.00x). The sheet's 2.37x is against the prompt's quoted 31.25 µs.

³ Measured once; the same kernel family measured 47.5–48.4 µs in other runs. The sheet's
1.66x is against the prompt's quoted 77.50 µs.

Entry points

- `b1_kv408`: `extern "C" int mla_decode_launch(q_nope, q_pe, ckv_cache, kpe_cache,
  const int* kv_indptr, const int* kv_indices, float sm_scale, void* out, float* lse,
  int batch, cudaStream_t stream)`. Build: `nvcc -O3 -std=c++17 -arch=sm_100a -Xcompiler
  -fPIC -shared`; load with ctypes. `lse` is written in base 2.
- `b16_kv9657`: `Mine(64, 8)(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
  sm_scale)` returning `(out, lse)`. `mine.py` builds the `.cu` with
  `-gencode arch=compute_100a,code=sm_100a -DT_MAX=64 -DCL=8`; it hard-codes the session's
  paths (`/workspace/mla_fresh/...`, override the source with `MLA_SRC`) and imports the
  constants `H` and `DC` from the session's `common.py`, which is not included (it held the
  input generator and timing code).
- `b64_kv75145`: `extern "C" int mla_decode_run(q_nope, q_pe, ckv, kpe, kv_indptr,
  kv_indices, out, lse, part_o, part_m, part_l, counter, g_prefix, sm_scale, B, grid,
  stream)`, two kernels per call (persistent decode + combine). The caller allocates
  `part_o` f32 `[total/256 + B + 1, 16, 512]`, `part_m` / `part_l` f32 `[same, 16]`, a
  zeroed int32 `counter[1]` (reset by the combine kernel) and int32 `g_prefix[B+1]`.
  Build: `nvcc -O3 -std=c++17 -arch=sm_100a --shared -Xcompiler -fPIC -DTILE=64
  -DSTAGES=2 -DCHUNK=256 -DPD=0`. **`-DPD=0` is required to get the measured version**:
  the file's default `PD=3` enables an L2-prefetch path added in the session's last minute
  and measured at ~52–53 µs.

Notes

- Correctness was checked against an fp32 naive implementation, with the tolerance set to
  7x FlashInfer's own error against that naive. `b1_kv408`: target workload plus 12 shapes
  x 2 seeds (B up to 4, lengths 1–2047). `b16_kv9657`: target workload on 3 seeds plus 8
  edge cases (lengths 1–8, chunk boundaries, a single 8191-token row, zero-length rows,
  large `sm_scale`). `b64_kv75145`: the target workload only, one seed.
- Design: `b1_kv408` runs one 16-CTA cluster per row with split-KV and merges the fp16
  partial outputs over DSMEM in a single launch (fp16 partials can overflow for very large
  inputs); `b16_kv9657` uses 8-CTA clusters, spills long rows across clusters and merges
  through a global scratch with a self-resetting release/acquire counter; `b64_kv75145` is
  a persistent split-KV kernel with an atomic work queue and a separate combine kernel
  (its shared-memory prefix table caps B at 512, unchecked).
- `b1_kv408` keeps the version the sheet reports (`mla_decode_v4.cu`); the session went on
  to faster versions (3.7–4.6 µs CUPTI) before it was stopped, not included here. In
  `b64_kv75145` an earlier configuration measured 42.32 µs and passed, but the session did
  not keep it.
- The `b64_kv75145` session read FlashInfer's Python host dispatch code for the decode
  wrapper through `inspect.getsource`; no device code of the reference was read.
- Debug macros left in the sources (`PROFILE`, `PROF`, `NO_LOAD` / `NO_COMPUTE` / `NO_MMA`)
  are off by default.
