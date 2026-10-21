---
name: gdn-prefill-ceilings
description: "Measured ceilings for the GDN prefill B200 kernel: which optimisation directions are closed and by what number"
metadata:
  node_type: memory
  type: project
  modified: 2026-10-21T23:05:00.000Z
---

Phases 5 and 6 of [[gdn-prefill-b200-project]] spent most of their budget
bounding directions rather than gaining time. The numbers, so none of these gets
re-opened on an instruction count:

**The rule Phase 6 ended on, which predicts most of the list below: an
algebraic pre-computation pays in proportion to how many consumers share it, and
removing work pays in proportion to the share of its kernel that the work is.**
Three directions agree on the first half -- the pre-kernel split (8 V-slice CTAs
share it, kept), head-pair Gram sharing (2 consumers, rejected), the `P.Tm` fold
(8 in the split path, 1 in the fused path: kept and 5-10% slower respectively).
Four Phase 6 rejections are the second half: the Gram warp-balance was -23% on
the critical warp and measured 0.9991 because the Gram gemm is 14% of the
pre-kernel; the gate block is 0.8% of it; `inv_mm_levels` is 6.2%; and the
main-kernel balanced map makes stage 1 (11.5%) pay for stages 2+3 (9.3%).

**Where the time is** (cfg 705, `8192/48`, CUPTI, `dev/x_bal4.sh` +
`dev/x_bal5.sh`). Pre-kernel 23.9 us: Gram 14%, triangular inverse 13.5%
(`inv_diag16` 7.6 + `inv_mm_levels` 6.2), `k`/`q` loads 10%, publish 8%,
`exp(E)` 7%, `A<-I` 2%, gates 0.8%, rest 44%. Main kernel 139.5 us: stage 1
11.5%, stage 4 6.4%, stage 2 5.1%, stage 3 4.2% (all four gemms 27.2%), `k`/`q`
loads 6.7%, `Tm`/`P` loads 5.5%, `new_state` store 2.6%, `output` store 2.2% --
the remaining 56% is the fp32 epilogues and the barriers.

* **Tile loads are at their floor.** `cp.async` *issue* is the biggest single
  line in the main kernel (11.9% of pcsamp samples, 3249 copies per CTA-chunk),
  but per K tile per CTA: `cp.async` 519 ns, `cp.async.bulk` per row 1217 ns
  padded and **1215 ns unpadded** (so destination stride is irrelevant; the cost
  is ~19 ns fixed per TMA transaction), one 16 KB bulk transfer 476 ns. The
  tensor form's best case is therefore 1.09-1.16x on the transfer alone = ~1.02x
  end to end. `dev/bulktest.cu`.
* **All four gemms made free is worth 1.10-1.37x.** `GDN_ABL=31457280` removes
  every `mma.sync`/`ldmatrix` from the main kernel. That caps `tcgen05`, which
  additionally cannot touch stage 4 (`M = VB`) or any `C = 32` config, keeps the
  fp32 epilogues, and adds a TMEM handshake per link of a 3-deep chain. See
  [[tcgen05-shape-and-layout-constraints]].
* **Communicating to avoid duplicated work does not pay below `T ~ 245`.** Three
  independent measurements: the split-path threshold, head-pair Gram sharing,
  and a cluster+DSMEM version that shares the state-independent half (41% of a
  fused chunk by ablation) and still loses -- 2.9 us fixed + 0.7 us/chunk of
  communication against 1.2 us/chunk saved, break-even at nchunk ~ 12. Cluster
  *launch* itself is free (0.992/1.000 for the unchanged kernel with an unused
  cluster dim), so the cost is all communication.
* **The register-to-register epilogue handoff is unavailable.** An `mma.sync`
  16x8 accumulator tile is exactly the A-operand layout for cols 0-7, so a
  transposed chain would make all three handoffs register-direct -- but warp `w`
  then needs all `k` for its m-tiles, forcing `NWARP <= VB/16`, and every such
  `(VB, NWARP)` is already measured worse (halving warps halves SM throughput
  because CTAs/SM stays smem-limited; `VB=64` was 1.2-1.5x slower).
* **Occupancy is irrelevant** -- retired three times. Phases 3 and 4 padded
  smem to *drop* a level. Phase 6 probed the reverse for free: `SmemMain` does
  not depend on `NWARP`, so cfg 706 (`C=64, VB=16`, 8 warps, same smem, same
  1 CTA/SM) doubles resident warps at no cost, and it is **5-6% worse** on every
  shape. So shrinking `SmemMain` below 114 KB to reach 2 CTAs/SM has nothing to
  buy -- more resident warps contend for the MIO path they would be hiding.
  (At `VB=32`, cfg 726 beats 725 by 0.5-1.7%: the one place more warps help.)
* **The evaluator's run-to-run spread is +-0.47%, not +-0.35%.** A third reading
  of the Phase 5 build gave 18.259 against its 18.089 and 18.151. Decide
  sub-1% changes with `dev/paircfg.py` / `dev/pairab.py`, never by comparing
  official runs.

The one thing that did pay: **masked-output gemms waste provably-dead work.**
`W`/`P` are masked to the lower triangle, so a warp whose whole output block is
in the strict upper triangle writes zeros regardless -- exactly 1 warp in 4 for
every instantiated `(C, NWARP)`. A warp-uniform guard *outside* the k loop (not
in its bound, which would kill the `#pragma unroll`) gave +0.80% geomean and cut
registers 254->193. The same holds with more force for the main kernel's stages
2-3 (`Tm` and `P` are lower triangular, 37.5% dead) but the useful k-count is
warp-dependent there, and stage 3 must share stage 1's warp map because it
accumulates onto stage 1's `oacc`. See [[uniform-break-in-unrolled-loops]].
