---
name: gdn-prefill-baseline-unrunnable
description: "FlashInfer GDN-prefill baseline can't run on the devspace; track absolute latency instead"
metadata:
  node_type: memory
  type: project
  originSessionId: d0404407-3b29-4c05-ac87-3112e08c89f3
  modified: 2026-10-20T03:44:10.483Z
---

For `gdn_prefill_qk4_v8_d128_k_last`, the official FlashInfer baseline
`flashinfer_wrapper_123ca6` **cannot run on devspace tasks/1295739**: its CuTe
DSL kernel calls `cutlass.cute.arch.ProxyKind`, which does not exist in the
installed CuTe DSL 4.6.0 (fails during `chunk_gated_delta_rule` main_loop).

**Why:** scoring is speedup vs this baseline, but it errors locally, so true
speedup can't be measured on the devspace (the real eval image is CUDA 13.2 with
a newer DSL where it works).

**How to apply:** track **absolute GPU latency** (cupti, via `scripts/dev_bench.py
--no-baseline` or `scripts/latbench.py`) as the optimization metric; lower = higher
speedup since the baseline is fixed. Do NOT try to patch the baseline. See
[[gdn-prefill-solution-state]].
