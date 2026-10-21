---
name: precompute-pays-per-consumer
description: An algebraic pre-computation pays in proportion to how many consumers share it; removing work pays in proportion to its share of the kernel
metadata:
  node_type: memory
  type: feedback
  modified: 2026-10-21T23:05:00.000Z
---

Two rules that between them decided eleven directions in
[[gdn-prefill-b200-project]], and that are worth applying *before* building:

**An algebraic pre-computation pays in proportion to how many consumers share
it.** The identical change -- reassociating `O = P·(Tm·rhs)` to `(P·Tm)·rhs`, so
a serial pair of gemms fuses and a barrier comes off the chain -- was **+3 %**
in the two-kernel path, where the product is computed once per `(chunk, head)`
in a parallel kernel and read by the 8 V-slice CTAs that share that head, and
**5-10 % slower** in the fused kernel, where each CTA is its own V-slice and so
recomputes it 8 times *on the serial chain*. The fused version even removed
*two* barriers instead of one. Count the consumers and ask which kernel pays,
not how many instructions the fold saves.

**Removing work pays in proportion to the share of its kernel that the work
is** -- not in proportion to how much of the work is wasted. Balancing a gemm's
warp loads cut its critical warp's issue slots by 23 % and measured **0.9991**,
because the gemm was 14 % of the kernel. Three more rejections in the same phase
were the same arithmetic (a 0.8 % stage, a 6.2 % stage, and a balanced map whose
largest stage paid for its two smallest). **So price the stage with an ablation
before designing the optimisation**, not after -- the ablation is an hour and
the implementation is a day.

See [[gdn-prefill-ceilings]] for the per-stage numbers and
[[fib-bench-per-process-bimodality]] for how the measurements were made.
