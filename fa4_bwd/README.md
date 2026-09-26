# FlashAttention-4 backward — /goal-mode kernels (B200)

Final kernels produced by interactive Claude Code `/goal` sessions, each written from
scratch without reading the reference kernel. Only the operator code (kernel + launcher)
is included; benchmarks, probes and test harnesses are omitted.

The reference is the FlashAttention-4 backward (`flash_attn.cute`, flash-attn-4
`4.0.0b19`, as shipped in the `model-align/sglang:v0.5.19` image) for the same case.
Its package was installed on the session's machine and could only be called as a black
box; the session transcripts were audited afterwards and contain no read of its source.

Device time is `ncu` `gpu__time_duration.sum` summed over **all** kernels of one backward
call (pre/post-processing kernels included), with `--profile-from-start off`, 100 warm-up
calls before the profiled one, `--cache-control none --clock-control none`; 3 rounds,
median taken. Candidate and reference were measured back to back on the same idle B200
(a node that never ran a goal session). `vs reference` = reference / candidate; > 1 means
faster than the reference. Develop time is the session's wall time as recorded in the
team results sheet.

| case | model | files | device time | reference | vs reference | develop time |
|---|---|---|---|---|---|---|
| `mha_b8_n4096_h16` (B=8, N=4096, H=16, D=128, causal, bf16, layout (B,N,H,D)) | Fable-5.1 | `mha_b8_n4096_h16/fable-5.1/fa_bwd49.cu` (raw PTX tcgen05/TMA; ctypes launcher `fa_bwd49.py`, header `tc_common.cuh`) | 1195.5 µs | 1386.0 µs | 1.159x | 24.5 h |
| | Opus-5 | `mha_b8_n4096_h16/opus-5/cute_bwd.py` (CuTe DSL + Triton) | 1398.0 µs | 1394.1 µs | 0.997x | 30 h |
| `gqa_b8_n4096_h16_kv2` (B=8, N=4096, H_q=16, H_kv=2, D=128, causal, bf16, layout (B,N,H,D)) | Fable-5.1 | `gqa_b8_n4096_h16_kv2/fable-5.1/gqa_bwd_final.py` (Gluon) | 1175.5 µs | 1459.8 µs | 1.242x | 4 h |
| | Opus-5 | `gqa_b8_n4096_h16_kv2/opus-5/hand224.cu` (+ ctypes launcher `hand224.py`) | 1312.7 µs | 1459.8 µs | 1.112x | 38 h |
| `varlen_t32768_n13_h16` (varlen MHA, 13 sequences packed into T=32768, H=16, D=128, causal, bf16) | Fable-5.1 | `varlen_t32768_n13_h16/fable-5.1/varlen_bwd.py` (Gluon) | 1520.9 µs | 1581.5 µs | 1.040x | 12 h |
| | Opus-5 | `varlen_t32768_n13_h16/opus-5/cute_bwd.py` (+ Triton helpers `tri_bwd.py`) | 1832.0 µs | 1581.5 µs | 0.863x | 39 h |

Entry points: `fa_bwd(q, k, v, o, do, lse, scale, ws)` with `ws = Workspace(B, N, H, D, device)`
(Fable-5.1 MHA; results in `ws.dq`, `ws.dk`, `ws.dv`); `attention_backward(q, k, v, o, lse, do, scale)` (Opus-5 MHA);
`gqa_bwd(q, k, v, o, do, lse, sm_scale)` (Fable-5.1 GQA);
`attn_bwd(q, k, v, out, do, lse, softmax_scale, causal=True)` (Opus-5 GQA);
`VarlenBwd` (Fable-5.1 varlen, schedule built from the sequence lengths);
`triton_bwd(q, k, v, o, do, lse, cu_seqlens, sm_scale)` (Opus-5 varlen). All return
`(dq, dk, dv)`.

Notes

- Correctness was checked against an fp32 naive implementation (forward written out,
  gradients from `torch.autograd`), with the tolerance set to 7x the reference's own
  relative-L2 error against that same naive. All six pass on fresh seeds.
- Both GQA kernels trade some dQ precision for speed, within the tolerance:
  the Fable-5.1 kernel accumulates dQ across kv blocks in fp16 (packed `red.f16x2`) and
  forms dS from bf16-rounded operands; the Opus-5 kernel accumulates dQ with bf16 atomics
  and also forms dS from bf16-rounded operands (FA4 keeps both in fp32). dK/dV match the
  reference's error level.
- `hand224.py` JIT-builds `hand224.cu` with `nvcc -gencode arch=compute_100a,code=sm_100a`
  and uses CUTLASS headers from the `tilelang` package of the same image.
- The two Opus-5 rows below 1x (MHA 0.997x, varlen 0.863x) are included for the
  cross-model comparison; they did not beat the reference on the same machine.
- Varlen sequence lengths (in packing order): 4096, 217, 8192, 1500, 513, 2800, 6144,
  700, 1777, 257, 3500, 1024, 2048.
- MHA Fable-5.1 accumulates dQ across kv blocks with TMA `cp.reduce.async.bulk` add directly into
  the bf16 output (FA4 keeps an fp32 accumulator); dq/dk/dv max-abs error vs the fp32 naive is
  0.0196/0.0173/0.0214, against 7x-FA4 tolerances 0.095/0.128/0.150. `fa_bwd49.py` JIT-builds the
  `.cu` with `nvcc -gencode arch=compute_100a,code=sm_100a`.
- MHA Fable-5.1 was re-measured on the B200 its session had used (idle at the time), not on a fresh node;
  its sheet-baseline ratio is 1387 / 1195.5 = 1.160x. During the session the model was blocked once and fell
  back to Opus-4.8 for about 1 h (198 replies) before returning to Fable-5.1; continuation turns only said
  "continue". Earlier versions of this session measured 0.947x.
