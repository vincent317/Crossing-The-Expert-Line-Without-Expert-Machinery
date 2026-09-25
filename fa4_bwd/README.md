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
| `gqa_b8_n4096_h16_kv2` (B=8, N=4096, H_q=16, H_kv=2, D=128, causal, bf16, layout (B,N,H,D)) | Fable-5.1 | `gqa_b8_n4096_h16_kv2/fable-5.1/gqa_bwd_final.py` (Gluon) | 1175.5 µs | 1459.8 µs | 1.242x | 4 h |
| | Opus-5 | `gqa_b8_n4096_h16_kv2/opus-5/hand224.cu` (+ ctypes launcher `hand224.py`) | 1312.7 µs | 1459.8 µs | 1.112x | 38 h |
| `varlen_t32768_n13_h16` (varlen MHA, 13 sequences packed into T=32768, H=16, D=128, causal, bf16) | Fable-5.1 | `varlen_t32768_n13_h16/fable-5.1/varlen_bwd.py` (Gluon) | 1520.9 µs | 1581.5 µs | 1.040x | 12 h |

Entry points: `gqa_bwd(q, k, v, o, do, lse, sm_scale)` (Fable-5.1 GQA);
`attn_bwd(q, k, v, out, do, lse, softmax_scale, causal=True)` (Opus-5 GQA);
`VarlenBwd` (Fable-5.1 varlen, schedule built from the sequence lengths). All return
`(dq, dk, dv)`.

Notes

- Correctness was checked against an fp32 naive implementation (forward written out,
  gradients from `torch.autograd`), with the tolerance set to 7x the reference's own
  relative-L2 error against that same naive. All three pass on fresh seeds.
- Both GQA kernels trade some dQ precision for speed, within the tolerance:
  the Fable-5.1 kernel accumulates dQ across kv blocks in fp16 (packed `red.f16x2`) and
  forms dS from bf16-rounded operands; the Opus-5 kernel accumulates dQ with bf16 atomics
  and also forms dS from bf16-rounded operands (FA4 keeps both in fp32). dK/dV match the
  reference's error level.
- `hand224.py` JIT-builds `hand224.cu` with `nvcc -gencode arch=compute_100a,code=sm_100a`
  and uses CUTLASS headers from the `tilelang` package of the same image.
- Same-case sessions not included here did not beat the reference on the same machine:
  MHA (B=8, N=4096, H=16) Fable-5.1 0.947x (still running) and Opus-5 0.997x;
  varlen Opus-5 0.857x.
