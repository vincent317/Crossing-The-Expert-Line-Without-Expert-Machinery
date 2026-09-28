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

The two `b1_*` sessions requested Fable-5.1, were blocked, and ran end to end as Opus-5,
so they are filed under `opus-5/`. `b18_q15883_kv15937` was rerun later with the same
prompt once Fable-5.1 was available, and that run — filed under `fable-5.1/` — is the one
kept here; the earlier Opus-5 attempt on this case stalled below the reference and is not
included.

Timing caliber differs from the other operator directories: these rows were **not**
re-measured on a fresh idle node, and candidate and reference were **not** measured back
to back. Device time is each session's own measurement on its own B200 devspace — CUPTI
kernel duration per call, summed over all kernels of one call, 100 warm-up calls and 300
iterations, median of 7 repeats; `b18_q15883_kv15937` is a median of 5 runs of 50
iterations. The reference figures come from the team's per-workload candidate list
(measured separately from these sessions); `b1_q199_kv203` and `b18_q15883_kv15937`
additionally measured their FlashInfer reference in-process, both landing within 0.5% of
the candidate list's figure. `vs reference` = reference / candidate; > 1 means faster
than the reference.

| case | model | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|---|
| `b1_q33_kv34` (B=1, q 33, kv 34) | Opus-5 | `b1_q33_kv34/opus-5/mla_kernel.cu` (raw PTX/CUDA, single kernel; `torch.utils.cpp_extension` launcher `mla.py`) | 3.936 µs | 22.38 µs (FlashInfer MLA fa2) | 5.69x | 1.9 h |
| `b1_q199_kv203` (B=1, q 199, kv 203) | Opus-5 | `b1_q199_kv203/opus-5/mla_prefill_triton.py` (Triton) | 13.02 µs | 26.46 µs (FA4 4.0.0b19) | 2.03x | 1.0 h |
| `b18_q15883_kv15937` (B=18, q 15883, kv 15937, mixed 4–4086 per request) | Fable-5.1 | `b18_q15883_kv15937/fable-5.1/mla_kernel_final.cu` (raw tcgen05/TMA for sm_100a, header `umma.cuh`; ctypes launcher `mla_prefill.py`) | 580 µs | 709.25 µs (FA4 4.0.0b32) | 1.22x | 12 h |

Entry points: `mla_paged_prefill(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr,
kv_indices, sm_scale)` (`b1_q33_kv34`, with a `run(inp)` adapter);
`mla_paged_prefill_causal(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr,
kv_indices, sm_scale, **cfg)` (`b1_q199_kv203`, default `cfg` `BN=128, num_warps=8,
num_stages=2`, with an `mla(inp)` adapter); `mla_prefill(q_nope, q_pe, ckv_cache,
kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, nclusters=0)`
(`b18_q15883_kv15937`). All return `(out, lse)`.

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
  - `b18_q15883_kv15937`: tolerance from its in-process FlashInfer reference instead of a
    bf16 model — out max-abs 1.56e-2 against a tolerance of 1.09e-1 (identical to
    FlashInfer's own, i.e. bf16 output rounding), out mean-abs 1.49e-4 against 8.3e-4,
    lse max-abs 1.24e-5 against 1.45e-2 (FlashInfer's own is 2.07e-3). Single-position,
    ragged, tiny and partial-tile cases also pass.
- `b1_q33_kv34` runs the whole call in a single kernel, compiled with `NWARPS_CFG=8`,
  `QKWARPS_CFG=5`; `mla.py` JIT-builds the `.cu` with
  `-gencode arch=compute_100a,code=sm_100a`.
- `b18_q15883_kv15937` decomposes as a 569 µs main kernel plus an 11 µs paged-KV gather.
  Per cluster one `cta_group::2` pair owns 8 query positions x 16 heads = 128 rows against
  256-wide kv tiles; `S = QK^T` via 2-SM tcgen05 MMAs at N=256, P written to SMEM as a
  swizzled bf16 operand, O accumulated entirely in TMEM, online softmax with lazy
  rescaling, K/V streamed through a 3x32 KB ring plus a kpe side buffer on spin-polled
  mbarriers, half the output stored by TMA queued behind the next q-tile's loads and half
  from registers, PV sub-blocks past `kv_end` skipped exactly, and an LPT schedule
  precomputed on the host. Build:
  `nvcc -O3 -gencode arch=compute_100a,code=sm_100a -std=c++17 -Xcompiler -fPIC -shared
  -o libmla_final_c2.so mla_kernel_final.cu -lcuda`; `mla_prefill.py` loads that `.so` and
  takes `H` / `D_CKV` from the session's bench harness, which is not included here.
- The decisive step for that case was cluster placement, not arithmetic: 4-CTA clusters
  fit only 33 at a time on this 148-SM B200 (132 SMs), leaving 16 SMs idle for the whole
  run. Two-CTA clusters place 74 clusters across all 148 SMs; the cross-pair TMA multicast
  is lost and L2 reads double to ~12 TB/s, which L2 sustains.
- That row's margin is thin and is reported as measured: the 1.2x goal is 591 µs, the five
  runs were 579.5 / 580.1 / 580.2 / 581.0 / 590.8 µs, so one run landed on the goal line
  and the median clears it by ~11 µs (2%).
- No runnable FA4 4.0.0b32 was available on that session's machine — the image's b5
  rejects head dims (576, 512) on SM100 and b32 trips its own TMEM-allocation assert — so
  709.25 µs stays a candidate-list figure. Its in-process FlashInfer 0.5.3 measurement
  (2320 / 2332 µs, i.e. 4.0x slower) is the part of that row measured back to back.
- Develop times are the sessions' wall times as recorded in the team results sheet.
