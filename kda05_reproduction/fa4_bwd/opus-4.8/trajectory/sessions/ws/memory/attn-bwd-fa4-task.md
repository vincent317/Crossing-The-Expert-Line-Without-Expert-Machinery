---
name: attn-bwd-fa4-task
description: "Setup, eval workflow and baselines for the B200 attention-backward kernel task"
metadata:
  node_type: memory
  type: project
  originSessionId: 37f282fb-802c-4d02-b19d-1132af0f8028
  modified: 2026-10-20T10:08:52.631Z
---

Task: beat flash-attn-4 b19 latency on two B200 attention_bwd definitions, keep correctness.
Workspace: `arms/kda-mx-fa4-opus48/ws` (FlashInfer starter kit). Work in git worktree.

**Definitions** (dataset at `/root/fib-dataset` on devspace):
- `attention_bwd_causal_h16_d128` (BSHD [B,N,H,D]): `run(q,k,v,out,dout,lse,sm_scale)`→(dq,dk,dv). MHA(kv16)+GQA(kv2), b8 s4096. atol0.14 rtol0.01.
- `attention_bwd_varlen_causal_h16_d128` (THD [T,H,D]): `run(q,k,v,out,dout,lse,cu_seqlens,max_seqlen,sm_scale)`. 13seq 32768tok maxseq8192. atol0.11 rtol0.01.
- Hq=16, D=128, sm_scale=1/sqrt(128)=0.08838834764831845. lse: BSHD [B,H,N], varlen [H,T], natural-log.

**Correctness rule**: element fails iff abs>atol AND rel>rtol; required_matched_ratio defaults to **1.0** → zero failing elements allowed. Inputs are fixed safetensors; reference is exact fp32. bf16 output rounding gives rel≈0.4%<1% so most elements pass on rtol. flash-attn baseline uses delta=rowsum(dO*O) and passes, so that trick is fine.

**Remote**: GPU devspace `tsh ssh root@gpu-node-1305349` (one B200, already running — use ONLY this one, don't use cctl here, task overrides the global cctl rule). Python `/opt/sglang/bin/python`, `flashinfer-bench` at `/opt/sglang/bin/flashinfer-bench`. Work dir `/root/work`. `tsh scp` works for file sync. Avoid heredoc over `tsh ssh` from the worktree (git guard blocks it) — write scripts locally and scp.

**Eval workflow**: `scripts/make_solutions.py` reads kernel.py, writes solution JSONs into `/root/fib-dataset/solutions/mine/...` (additive). Then run `scripts/run_eval_{bshd,varlen}.sh` in tmux (sessions evalbshd/evalvarlen). Results land in `/root/fib-dataset/traces/<author>/attention_bwd/<def>.jsonl` with `evaluation.performance.latency_ms`. The CLI "speedup vs mathematical reference" is NOT the contest metric; contest speedup = baseline_latency/mine. Fast local iteration: `tests/harness.py bshd|varlen [--bench]` (compares vs embedded fp32 ref, reports fail counts + latency).

**Baselines (flash-attn-4 b19, official latency_ms)**: MHA 1.380, GQA 1.474, varlen 1.572.

**Phase 1 result (triton_*_v1, FA2 2-pass Triton, BLK64)**: MHA 6.027 (0.229x), GQA 6.412 (0.230x), varlen 7.295 (0.216x). All correct. See [[attn-bwd-fa4-progress]] for latest status.
