---
name: uniform-break-in-unrolled-loops
description: A uniform break in a compile-time-unrolled CUDA loop pays when it skips a whole barrier-delimited stage and loses when it sits inside a scheduled dependency chain
metadata:
  node_type: memory
  type: feedback
  originSessionId: ba4245a3-3aeb-4f46-9839-0dcdd7969d8e
  modified: 2026-10-20T17:36:34.817Z
---

Skipping work with a CTA-uniform `break` inside a `#pragma unroll` loop (rather
than putting the bound in the loop condition, which makes the index a runtime
value and kills the unroll) measured both ways in the same kernel:

* **Worth 1.09x**: skipping whole triangular-inverse *levels* whose band is
  empty. Each level is two matmuls and two `__syncthreads()`, so the break
  removes barrier-delimited stages wholesale.
* **6-13 % slower**: stopping a 15-step, 120-FFMA forward-substitution chain
  early. That loop is fast *because* full unrolling lets ptxas hoist all 120
  broadcast shared loads across the chain; a per-iteration branch gives that
  up. Splitting it into a fully-unrolled and an early-exit instantiation did not
  recover it either — the extra code cost ~2 % even on the shapes that take the
  unchanged path.

**Why:** the break is free when what it skips is a *stage* (its own barriers,
its own loads). It is expensive when it interrupts a region the scheduler was
software-pipelining. Before adding one, ask which of the two the loop is.

See [[gdn-prefill-b200-project]].
