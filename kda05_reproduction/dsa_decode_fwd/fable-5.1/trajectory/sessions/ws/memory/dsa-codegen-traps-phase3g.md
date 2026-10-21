---
name: dsa-codegen-traps-phase3g
description: "Phase 3g (2026-10-21) findings for the DSA kernel: G7 Q-through-TMA kept (g44 = official v23 101.47x); D32 (32-key tiles, flat 16-cluster grid, probe-derived work list) and the probe-driven launch order (g47a) rejected with measurements; nvcc/ptxas traps found on the way"
metadata:
  node_type: memory
  type: project
  originSessionId: 8daac9cd-cdd8-456c-88f4-7ecde2773386
  modified: 2026-10-20T19:31:03.934Z
---

Phase 3g of the DSA sparse-attention kernel (worktree `worktree-dsa-attn-dev`, 2026-10-21):

- **Kept: g44 = Q tile through 32 `cp.async.bulk` row copies** (padded [16][1040 B] / [16][144 B] rows, one mbarrier,
  expect_tx = the 18432 payload bytes, NOT the padded footprint — counting the padding deadlocks the kernel at 100 %
  GPU). All-23 6.034 vs 6.117 us (g43), wins on all 23; late issue (after the ballots) +0.13 us. Commit 82f8616;
  **official v23 = 101.47x / 6.04 us, 23/23** (packed_solution.json = v23, commit b071151). HEAD of the worktree
  (8e9f2e6) has the g44 kernel; the D32 / g47a experiment kernels are archived under docs/*.cu.txt.
- **Stage A (cell-distributed level 2, 8-segment masks on the old grid, commit-less WIP g45c)**: correct, all-23 6.068 vs
  6.031 (g44 same session) — neutral/slightly worse; the first version with ownership-chunked polls was +3..6 us on tokens
  whose last segment has 1-2 active ranks (one CTA owning 8 slices = 8 sequential poll round trips). Any code added to
  the 3a/3b topology step is on the critical path (the kpe block lands only ~0.3 us later): 64-bit mask loops + two
  integer divisions cost +0.2 us on EVERY workload; fixed with `__vsetne4` byte masks and a reciprocal division under
  `if (n_seg >= 2)`.
- **Stage B (D32: flat min(4T,16)-cluster grid, probes of slot 512, 5 candidate slot loads, 32-key tiles for the unique big
  token, entry loop / continuation for the fallback)**: correct on rep8 but SLOWER everywhere (T=1 4.25 vs 3.87, 02d6ae9c
  7.96 vs 7.09). Measured causes: (1) big token: kpe block lands 1.3 us after issue in 32-mode (0.7 in 64-mode) and the
  level-2 exchange with 8 segments x 64 CTAs returns 2.0-2.6 us after the publish even when the first poll succeeds
  (the old 4-segment elected merger: 0.7-0.9) — the cross-SM L2 exchange cost scales with the number of strong
  stores/loads in flight, not with bytes; (2) small tokens: +0.4 us of instruction-fetch stalls ("no_instructions" in
  ncu warp sampling, `--warp-sampling-interval 0`) spread over the fragmented prologue, vs none after kernel start in g44.
- **g47a, probe-driven launch order on g44** (positions >= T = the j-th big token in priority order, 4 candidate slot
  loads, row + Q of prio[j] speculative, reload after the gather on a mismatch): correct, 93 regs, but rep8 6.23 vs 5.81 —
  +0.3 even on unchanged mappings (match code between the index loads and the gather issue) and +1.2 where the reload
  hits (it costs as much as the wave-2 penalty it removes). The static order 2,5,6,4 is the best the 15-cluster wave
  allows for T=8 (8 + 3 + 3 of 15 slots). Any decision logic in the prologue of this kernel shows up 1:1 in latency.
- **nvcc/ptxas traps** (B200, CUDA 13.0): a lambda body called inside a loop makes loop-invariant address math hoist
  above the gather issue (~200 IMAD/LOP3 on the critical path, 229 registers); code shared by two branches (e.g. an
  activity() call in both the inactive-exit and the main path) is hoisted above the branch; ordinary loads whose first
  use is far away get sunk to the use (two DRAM round trips instead of one) — `asm volatile` loads fix that; a chain of
  ternaries on a kernel argument becomes a constant-bank-2 table (`LDC c[0x2]`), a cold constant-cache miss at kernel
  start ahead of the index loads — use mask selects; passing `const Params&` to a `__noinline__` function copies the
  kernel params to a 104-byte stack frame; a runtime index into a small register array spills it; capping registers
  (`__launch_bounds__(256, 2)`) made the kernel +0.2 us slower.
- Devspace hygiene: a hung run (deadlocked kernel) keeps the GPU at 100 % and poisons later timings — check
  `ps aux | grep "[d]ev_bench"` / `nvidia-smi` before benchmarking; kill by pid, never `pkill -f` a pattern that matches
  the ssh command line.

**Why:** these cost ~4 hours of experiments and would be re-discovered otherwise.
**How to apply:** D32 needs (a) a level-2 exchange with far fewer L2 transactions for 8 segments and (b) a prologue as
compact as g44's before it can pay; the per-SM gather lever measured by micro_tilescale did not materialize in the
kernel (32-key tiles landed later, not sooner). See [[dsa-kernel-findings]] and [[dsa-devspace-workflow]].
