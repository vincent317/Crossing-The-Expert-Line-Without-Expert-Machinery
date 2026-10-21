---
name: dsa-phase3j-findings
description: "Phase 3j (2026-10-21) of the DSA kernel: dual merger segments (M2), late Q/row loads for segments >= 1, and the ML-first level-2 poll all rejected with all-23 measurements; measured facts: the level-2 merger already polls once (0.7-0.85us), the inactive-cluster drain of T=8 ends at ~3.2us (not critical), 16-CTA clusters launch at the same cost but only 7 are co-resident and 16 live CTAs cost +0.14us; a new live prologue value can push the kernel from 119 to 173 registers (+0.13us everywhere)"
metadata:
  node_type: memory
  type: project
  originSessionId: phase3j
  modified: 2026-10-21T00:23:26.421Z
---

Phase 3j of the DSA sparse-attention kernel (worktree `worktree-dsa-attn-dev`, 2026-10-21, after v26 = g51 103.73x):

- **M2 rejected** (`DSA_MERGE2`, two merger segments each polling 8 heads so the per-CTA poll stays under the
  outstanding-load budget): correct 23/23 but 100.56x vs 103.80x same session. Cause 1: 173 registers vs 119 (`dual`,
  `my_heads`, `mh0` live from 3a to 7d + a third publish path inside the unrolled merge lambda) → +0.13 us on EVERY
  workload incl. T=1 where the path never runs. Cause 2: the g51 merger already polls ONCE (trace: merged → polled
  0.70-0.85 us, polls=1 on all 8 CTAs of 02d6ae9c) — the Phase 3d "1.35 us for 3 pending segments" is obsolete since
  g25/g26; the earlier-launched second merger polls before the data is visible (polls=2, +0.45).
- **Late loads for segments >= 1 rejected**: `DSA_QTMA_LATE 2` (Q after the ballots for segs 1-3 only) 103.42x — T<=2
  unchanged but the big token's segments wait for Q (+0.1..0.2 on 5 workloads); `DSA_ROW_LATE` (index row after the
  ballots for segs 1-3, never for inactive CTAs) 101.82x — +0.1 even on T=1 (predicated row loads in the prologue);
  `DSA_QTMA_LATE 1` on all 23: 101.98x. The 0.3-0.5 us index-row lag of the big token under T=8 load is not caused by
  the inactive CTAs' start-window traffic (5 MB of Q + row requests).
- **ML-first poll rejected** (`DSA_POLL_ML_FIRST`: tight poll of the (M, L) records, then the 8 KB of words):
  +0.25..0.4 us on all 13 big-token workloads — two sequential round trips vs one combined poll.
- **Drain fact** (trace `--all`): inactive CTAs live 0.77-0.99 us in wave 1 and 0.5-0.55 us later (L2-warm rows), with
  0.4 us placement gaps; on T=8 workloads (21-22 inactive clusters) the drain ends at 3.1-3.2 us while the merger ends
  at 5.9-6.1 → the drain is NOT on the critical path (the g45c "drain" explanation was wrong). T=8 critical path = the
  big token: index 1.5-1.7 → kpe 2.2 → QK 2.9 → softmax 3.4 → PV 4.1 → pushed 4.3 → level-1 landed 4.7 → merged 4.9 →
  polled 5.6-5.9 → end 5.9-6.1 (+1.1 launch).
- **Cluster-16 facts** (`scripts/micro_cluster16.py`): launch cost identical to cluster 8 (1.47 us empty grid of 32;
  the empty-grid floor grows to 2.1 us at 256 CTAs); a cluster of 16 live CTAs costs +0.14 us vs 8 on a T=1-like grid;
  exactly 7 16-clusters are co-resident (vs 15 of 8) and cluster 8+ starts only when a wave-1 cluster exits → a T x 2
  grid of 16-clusters would start one mid-size token ~3 us late on T=7/8. C16 deferred (level-2 gain now <= 0.3 us).

**Why:** ~3 hours of A/B runs; the obsolete poll figure and the wrong drain story would otherwise motivate the same
experiments again.
**How to apply:** the level-2 exchange is at its floor for the tagged-word mechanism (visibility ~0.4 + one round
trip ~0.3 + alignment lottery); any new prologue-live value must be checked against the register count (119) before
benchmarking. Start from the HEAD of `worktree-dsa-attn-dev` (options off = g51 kernel). See [[dsa-phase3i-findings]],
[[dsa-kernel-findings]], [[dsa-devspace-workflow]].
