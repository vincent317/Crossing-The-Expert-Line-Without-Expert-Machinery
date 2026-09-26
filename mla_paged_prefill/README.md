# MLA paged prefill, causal — /goal-mode kernels (B200)

Final kernels produced by interactive Claude Code `/goal` sessions, each written from
scratch without reading the reference kernel. Only the operator code (kernel + launcher)
is included; benchmarks, probes and test harnesses are omitted.

Operator: `mla_paged_prefill_causal_h16_ckv512_kpe64_ps1` — H=16, d_ckv=512, d_kpe=64,
page size 1 over a 989669-page pool, bf16, causal (bottom-right aligned), inputs
`q_nope (Nq,H,512)`, `q_pe (Nq,H,64)`, `ckv_cache (P,1,512)`, `kpe_cache (P,1,64)`,
`qo_indptr`, `kv_indptr`, `kv_indices`, `sm_scale` → `out (Nq,H,512)` bf16,
`lse (Nq,H)` f32. The three cases are flashinfer-trace workloads `55b51e96`, `4eeb9b51`
and `fe63f292`.

All three sessions requested Fable-5.1, were blocked, and ran end to end as Opus-5, so
they are filed under `opus-5/`.

Timing caliber differs from the other operator directories: these rows were **not**
re-measured on a fresh idle node, and candidate and reference were **not** measured back
to back. Device time is each session's own measurement on its own B200 devspace — CUPTI
kernel duration per call, summed over all kernels of one call, 100 warm-up calls and 300
iterations, median of 7 repeats. The reference figures come from the team's per-workload
candidate list (measured separately from these sessions); only the `b1_q199_kv203`
session additionally measured its FlashInfer reference in-process. `vs reference` =
reference / candidate; > 1 means faster than the reference.

| case | model | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|---|
| `b1_q33_kv34` (B=1, q 33, kv 34) | Opus-5 | `b1_q33_kv34/opus-5/mla_kernel.cu` (raw PTX/CUDA, single kernel; `torch.utils.cpp_extension` launcher `mla.py`) | 3.936 µs | 22.38 µs (FlashInfer MLA fa2) | 5.69x | 1.9 h |
| `b1_q199_kv203` (B=1, q 199, kv 203) | Opus-5 | `b1_q199_kv203/opus-5/mla_prefill_triton.py` (Triton) | 13.02 µs | 26.46 µs (FA4 4.0.0b19) | 2.03x | 1.0 h |
| `b18_q15883_kv15937` (B=18, q 15883, kv 15937, mixed 4–4086 per request) | Opus-5 | `b18_q15883_kv15937/opus-5/mla.cu` (CuTe tcgen05 2-SM UMMA; ctypes launcher `mlarun.py`) | 729.3 µs | 709.25 µs (FA4 4.0.0b32) | 0.97x | 3.3 h |

Entry points: `mla_paged_prefill(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr,
kv_indices, sm_scale)` (`b1_q33_kv34`, with a `run(inp)` adapter);
`mla_paged_prefill_causal(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr,
kv_indices, sm_scale, **cfg)` (`b1_q199_kv203`, default `cfg` `BN=128, num_warps=8,
num_stages=2`, with an `mla(inp)` adapter); `mlarun.build()` then
`mlarun.Runner(inp, Q_LENS, KV_LENS).run()` (`b18_q15883_kv15937`). The first two return
`(out, lse)`; the `Runner` writes into its own `.out` / `.lse`.

Notes

- Correctness was checked against an fp32 naive implementation, with the tolerance set to
  7x the deviation of a bf16 reference model (bf16 operands, fp32 accumulation, online
  softmax, P rounded to bf16) against that same naive.
  - `b1_q33_kv34`: 8 seeds, worst deviation 0.012076 — exactly equal to the bf16
    reference model's own worst deviation, against a tolerance of 0.084530; output is
    bit-deterministic across runs.
  - `b1_q199_kv203`: 5 seeds plus a contiguous-index case, out max-abs 0.015625 against
    a tolerance of 0.109375, lse max-abs 1.9e-06 against 0.088059; its in-process
    FlashInfer reference scored the same out max-abs and a worse lse (0.00154).
  - `b18_q15883_kv15937`: out max-abs 0.009405 against a tolerance of 0.064214, rmse
    0.000233 against 0.001582 (1.025x / 1.030x of the bf16 model's own error).
- `b1_q33_kv34` runs the whole call in a single kernel, compiled with `NWARPS_CFG=8`,
  `QKWARPS_CFG=5`; `mla.py` JIT-builds the `.cu` with
  `-gencode arch=compute_100a,code=sm_100a`.
- `b18_q15883_kv15937` is the one row below 1x, and the one unfinished session. The
  included kernel is the 8-compute-warp, tile-per-CTA variant (384 threads per CTA over a
  2-SM cluster), built and measured with `MLA_ABL=0 MLA_PAIR=1 MLA_NS=2 MLA_NP=2` —
  `mlarun.py`'s own defaults (`NS=4`, `NP=4`, `PAIR=2`) are not the measured
  configuration. The session ended mid-conversion to a persistent-CTA variant with an
  unresolved multi-tile correctness bug; that work-in-progress is not included.
- `mla.cu` carries an `ABL` compile-time ablation switch. `ABL>=5` compiles out the
  softmax and epilogue warps, which is where the ~498 µs / 1336 TFLOPS figures in that
  session's logs come from; they are an ablation floor, not a working kernel. The 729.3 µs
  above is `ABL=0`, the complete kernel, and is the only configuration that passes the
  correctness check.
- Develop times are the sessions' wall times as recorded in the team results sheet.
