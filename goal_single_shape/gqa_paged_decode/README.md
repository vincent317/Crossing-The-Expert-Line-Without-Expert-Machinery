# GQA paged decode — /goal-mode kernels (B200)

Final kernels produced by interactive Claude Code `/goal` sessions, each written from
scratch without reading any reference kernel. Only the operator code (kernel + launcher)
is included; benchmarks, probes and test harnesses are omitted.

Common shape: 32 q heads / 8 kv heads, head dim 128, page size 1, bf16, inputs
`q (B,32,128)`, `k_cache / v_cache (pool,1,8,128)`, `kv_indptr (B+1)`, `kv_indices`,
outputs `out (B,32,128)` bf16 and `lse (B,32)` fp32.

There was no fixed reference: each session benchmarked the available library kernels on
its own B200 (FA4 `flash_attn.cute` varlen + page table, FlashInfer 0.7.0 decode wrappers,
cuDNN, SDPA) and took the fastest as the reference. Device time is the sum of the call's
CUDA kernel durations from `torch.profiler` in a warm-L2 loop, measured for candidate and
reference in the same session. `vs reference` = reference / candidate; > 1 means faster.
Develop time is the session's wall time as recorded in the team results sheet.

| case | model | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|---|
| `b1_kv173` 单请求 (B=1, kv_len 173, pool 223) | Fable-5.1 | `b1_kv173/fable-5.1/gqa_decode_kernel.cu` (+ launcher `gqa_ext.py`) | 3.96 µs | 7.26 µs (FA4) | 1.83x (1.56x with L2 flushed) | 1.4 h |
| `b16_kv2841` 中批次短序列 (B=16, kv_len 118–455, 2841 total, pool 2868) | Opus-5 ¹ | `b16_kv2841/opus-5/gqa_decode.cu` (+ launcher `gqa_decode.py`) | 7.92 µs | 12.07 µs (FlashInfer tensor-core decode) | 1.52x | 1.5 h |
| `b64_kv58134` 大批次 (B=64, kv_len 496–3159, 58134 total, pool 67405) | Opus-5 ¹ | `b64_kv58134/opus-5/gqa_paged_decode.py` (Triton) | 49.5 µs | 74.8 µs (FA4, best of `num_splits` / `pack_gqa` sweep) | 1.51x | 0.6 h |

¹ These two sessions were launched with Fable-5.1, but its safeguard flagged the first
message and Claude Code fell back to Opus-5 for the entire session, so every turn was
written by Opus-5.

Entry points: `gqa_ext.build()` returns the compiled op (Fable-5.1, `b1_kv173`);
`run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, plan)` with a `Plan`
(`b16_kv2841`); `GQAPagedDecode` (`b64_kv58134`).

Notes

- Correctness was checked against an fp32 naive implementation, with the tolerance set
  to 7x the reference's own error against that naive; all three pass, including extra
  kv lengths and batch sizes beyond the target case.
- The two CUDA kernels are JIT-built with `torch.utils.cpp_extension.load` for
  `sm_100a`; the `b1_kv173` kernel uses an 8-CTA thread-block cluster per kv head and
  merges the split partials over DSMEM.
