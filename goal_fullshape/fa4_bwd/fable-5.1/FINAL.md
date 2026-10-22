# FINAL REPORT: causal attention backward (bf16, H=16, D=128) on NVIDIA B200

Generated 2026-10-21T18:07:28 on gpu-node-1321213 (B200, torch 2.13+cu130, triton 3.7.1 Gluon).

Timing: CUPTI kernel span (first kernel start to last kernel end), cold L2 before every call, median; ref and my solution measured alternately on the same card (3 rounds, median of medians). Every timed call computes everything from the raw inputs (preprocess + both passes); nothing data-dependent is cached.

Solution: `solution/attn_bwd_gluon7.py` (+ `solution/main.py` FIB entry). Tolerance = 7 x the FA4 error vs the fp32 naive reference, per workload and output.

| workload | ref (fastest library on this card) | ref us | mine us | ref/mine | correctness vs fp32 naive |
|---|---|---:|---:|---:|---|
| MHA B=8 N=4096 H=16 Hkv=16 (attention_bwd_causal_h16_d128) | cuDNN (torch SDPA CUDNN_ATTENTION backend) | 1515.7 | 1878.8 | **0.807x** | dq: max_abs 0.0140 (tol 0.0982), rel_fro 0.00245 (tol 0.01714) OK; dk: max_abs 0.0180 (tol 0.1259), rel_fro 0.00243 (tol 0.01701) OK; dv: max_abs 0.0204 (tol 0.1428), rel_fro 0.00231 (tol 0.01618) OK |
| GQA B=8 N=4096 H=16 Hkv=2 (attention_bwd_causal_h16_d128) | cuDNN (torch SDPA CUDNN_ATTENTION backend) | 1558.1 | 1628.9 | **0.957x** | dq: max_abs 0.0153 (tol 0.1069), rel_fro 0.00245 (tol 0.01715) OK; dk: max_abs 0.0223 (tol 0.1636), rel_fro 0.00243 (tol 0.01700) OK; dv: max_abs 0.0363 (tol 0.2538), rel_fro 0.00231 (tol 0.01617) OK |
| varlen 13 seqs, 32768 tokens, max 8192 (attention_bwd_varlen_causal_h16_d128) | flash-attn-4 (flash_attn.cute 4.0.0b19) | 1793.4 | 2087.7 | **0.859x** | dq: max_abs 0.0155 (tol 0.1088), rel_fro 0.00247 (tol 0.01732) OK; dk: max_abs 0.0186 (tol 0.1304), rel_fro 0.00246 (tol 0.01721) OK; dv: max_abs 0.0174 (tol 0.1218), rel_fro 0.00230 (tol 0.01612) OK |

**Geometric mean speedup over the 3 workloads: 0.872x**

## All library candidates measured (median us)

- mha: flash-attn-4 (flash_attn.cute 4.0.0b19): 1565.7, cuDNN (torch SDPA CUDNN_ATTENTION backend): 1515.7; mine: 1878.8 (rounds: 1850.8, 1880.7, 1878.8)
- gqa: flash-attn-4 (flash_attn.cute 4.0.0b19): 1663.4, cuDNN (torch SDPA CUDNN_ATTENTION backend): 1558.1; mine: 1628.9 (rounds: 1628.9, 1627.3, 1636.8)
- varlen: flash-attn-4 (flash_attn.cute 4.0.0b19): 1793.4; mine: 2087.7 (rounds: 2087.7, 2082.9, 2124.3)

## Correctness

All three workloads pass the per-workload tolerances above; random-shape tests (odd lengths, GQA groups, empty/short varlen sequences, large logits) pass with the same tolerances (bench/check.py --random).

## Official FIB evaluation (flashinfer-bench run --use-isolated-runner, --atol 0.14/0.11 --rtol 0.01)

Solutions packaged as /root/fib-dataset/solutions/mine/attention_bwd/<definition>/gluon_bwd_{bshd,varlen}_v2.json.

- Solution 'gluon_bwd_bshd_v2' for workload 5d2f3c61-0a4e-4f7b-9a6d-1b8e2c4f7a01: PASSED with 59.13x speedup vs. mathematical reference
- Solution 'gluon_bwd_bshd_v2' for workload 8c41e9b2-7f3d-4a15-b0c8-6e2d9f1a3b02: PASSED with 63.33x speedup vs. mathematical reference
- Solution 'gluon_bwd_varlen_v2' for workload e3a7b5d9-2c6f-4e81-9d4b-0f5a8c7e1d03: PASSED with 59.66x speedup vs. mathematical reference
