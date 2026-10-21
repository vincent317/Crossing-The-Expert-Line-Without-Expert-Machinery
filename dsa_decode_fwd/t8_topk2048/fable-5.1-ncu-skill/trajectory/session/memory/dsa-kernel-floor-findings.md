---
name: dsa-kernel-floor-findings
description: "What was measured and rejected while optimizing the B200 DSA sparse decode kernel (sol18, 1.138x vs 1.20 target); saves re-trying dead ends"
metadata:
  node_type: memory
  type: project
  originSessionId: 54608829-509a-4af0-abc0-ef4276f35438
  modified: 2026-10-20T10:40:24.523Z
---

DSA decode kernel on devspace tasks/1301991 (B200): the rule-7 compliant sol18 reached harness=v2 speedup 1.138
(8.91 us vs ref 10.14 us) on 2026-10-20; target 1.20 (<= 8.45 us) was not reached by the 2026-10-21 14:00 deadline.

**Why:** the single-launch design is latency-bound: index load 0.9 us (DRAM), gather 1.6 us (DRAM tail latency),
compute 2.0 us (mma.sync throughput, not chain latency: 8 warps/CTA did not help), SM doubling +0.5 us, then a ~3 us
combine tail (poll detection, data round trip 0.5-0.8 us, weights/accumulate ~1 us). L2 round trips under load
are 0.45-0.8 us.

**How to apply:** do not re-try: pipelined/replicated polling, speculative data probes, cp.async progressive
loads (higher latency than LDG), per-lane online merges, single poller + named barriers, parameter/constant
schedule tables (0.1-0.3 us prologue latency), takeover rebalancing (holders land on busy SMs), alternative static
layouts (identity is best), early publish before stores. Wins were: tag-validated partials (no producer fence),
early Q loads, per-warp copy-out, rotating combine roles to another request's blocks. Watch ptxas scoreboard
sharing: "+r" predicated-load asm inside a loop makes the compiler wait for all in-flight loads at the loop top
(inspect with the sbdec.py control-bit decoder in tools/). Related: [[dsa-goal-rules]].
