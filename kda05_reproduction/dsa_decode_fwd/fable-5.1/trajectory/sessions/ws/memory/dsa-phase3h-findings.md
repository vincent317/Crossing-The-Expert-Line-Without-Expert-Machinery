---
name: dsa-phase3h-findings
description: "Phase 3h (2026-10-21) of the DSA kernel: S8 spread mode (siblings of a single-tile token each store one output slice), F2, static PV loops (dynamic break in an unrolled ldmatrix/mma loop cost 0.3-0.4us; switch -> constant-bank jump table +0.09us), S8K rejected, machine drift, official v24"
metadata:
  node_type: memory
  type: project
  originSessionId: phase3h
---

Phase 3h of the DSA sparse-attention kernel (worktree `worktree-dsa-attn-dev`, 2026-10-21, after v23 = g44):

- **Score leverage**: the official score is the mean of per-workload speedups, so 0.1 us on a 3.7 us T=1/T=2 workload
  is worth ~4.5x more than 0.1 us on an 8 us T=8 workload (9 of the 23 workloads are T<=2).
- **S8 spread mode (g48, `DSA_SPREAD_T 6`)**: for num_tokens <= 6, the 8 segment-0 CTAs of a single-tile token all
  load tile 0's slots (2 extra index loads per lane for the siblings), gather / QK / softmax the tile redundantly, and
  CTA c computes + stores only output slice c. The single CTA's PV (0.2-0.45 us) and 16 KB output store (0.29 us: a
  SM drains stores at ~32 B/clk) leave the critical path: -0.4..-0.85 us on every single-tile T<=2 workload, 68d6817d
  (T=6, all small) -0.48. For T=7/8 the siblings holding their SM slots ~2 us longer delays the inactive-cluster
  drain (+0.05..0.09 on 6 workloads) -> threshold 6. Siblings decide from the ballots (own tile empty) and 3a
  (`cta_act == 1`); a wrong speculation (non-prefix-packed rows) just exits at 3a.
- **F2**: fused QK of <= 4 n-tiles split over warp pairs (warp w+4 = blocks 4-7, 16-B fragment exchange + named
  barrier `bar.sync 1+w, 64`): -0.06..-0.08 us on tokens of <= 32 keys.
- **Static loops (g49/g50, the biggest win of the day, -0.37 us mean)**: `if (kstep >= n_ksteps) break;` inside the
  unrolled PV loop stopped ptxas from overlapping the next k-step's ldmatrix.trans with the mma chain -> template
  instances `pv_loop<NK, NP>` dispatched once (-0.28 us, faster on all 23). The `switch (n_ksteps)` dispatch became a
  constant-bank-2 jump table (cold constant-cache miss): an if-else chain is another -0.09 us. The same trick on the
  config-4 QK (`DSA_QK_STATIC`) was +0.04 (off). Rule: no data-dependent break in an unrolled ldmatrix/mma loop;
  dispatch with if-else, not switch.
- **S8K rejected** (docs/kernel_s8k_ksplit.cu.txt): K-split spread (CTA c gathers 8 rows + dim-block c, S columns
  exchanged over DSMEM st.async/mbarrier): the exchange round trip ~0.6 us and the fused NTW=2 softmax 0.55 us eat the
  gather saving (the 17 KB lands no sooner than the 73 KB tile's first block). Compiled in but never executed it cost
  every workload +0.1..0.28 us (codegen, not code size: removing all merge code via `DSA_ABL_NOMERGE` changes nothing).
- **Machine drift**: this session the devspace ran ~10 % slower (g44 all-23 6.717 vs 6.034 in the morning; GPU ~195 W
  at idle, host load average 44). Official v24 (= g48) 96.74x / 6.52 us vs v23 101.47x is drift, not regression; v25 (= g50) 101.56x / 6.15 us is the best author score;
  only same-session A/B (`remote_ab.sh` or one tmux batch) is trustworthy. In-kernel trace stamps inside the PV
  region (a divergent `threadIdx.x == 0` globaltimer read) distort the PV timing by ~0.5 us - don't stamp inside
  mma loops.
- Trace tool notes: `dev_bench --scramble --no-time` ends with a harmless StatisticsError; `scripts/remote_ab.sh`
  variant flags must not contain '_' substitutions (use a comma separator and `${f//,/ }`).

**Why:** these took ~5 hours of A/B runs; the static-loop lesson applies to every unrolled mma loop in this kernel.
**How to apply:** start from the latest commit on `worktree-dsa-attn-dev` (g50 = commit a8e1398); check other
unrolled loops for data-dependent breaks before any new direction. See [[dsa-kernel-findings]],
[[dsa-codegen-traps-phase3g]], [[dsa-devspace-workflow]].
