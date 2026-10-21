---
name: dsa-kernel-findings
description: "Measured B200 facts and rejected directions for the DSA sparse-attention kernel (per-SM gather limit, DSMEM rates, tcgen05 at N=16, L2 round trips, store visibility bound on level 2, softmax/QK floors, codegen traps, launch-order/wave facts, prologue costs, Phase 3d-3f results through g41)"
metadata:
  node_type: memory
  type: project
  originSessionId: bfd01c5f-cdbd-4cc8-ba07-5355b17bf52a
  modified: 2026-10-20T12:11:42.234Z
---

Hard-won measurements on the B200 devspace for the DSA sparse attention kernel (2026-10-20):

- Per-SM gather of random 1152-B KV rows is throughput-limited at ~32 GB/s cold / ~60 GB/s from L2
  regardless of mechanism (cp.async 16 B, ld.global.v4, TMA bulk rows) and regardless of locality
  (not TLB). Spreading a 2048-key token over 32 SMs (64 keys per CTA) is what made it fast.
- DSMEM: pushes run at ~25-30 GB/s per SM; `cp.async.bulk.shared::cluster.shared::cta` beats
  `st.async` by ~0.35 µs for 7 x 2 KB; 16 bulk copies issued by lanes of ONE warp serialize (0.5 µs) —
  issue them from one lane per warp (0.17 µs). CTAs that exited after arriving count as arrived in
  later cluster-barrier phases (tested).
- tcgen05.mma at N=16 heads costs ~30 ns per instruction — slower than mma.sync over 8 warps (mma.sync
  itself runs at ~16 cycles per m16n8k16 per sub-partition: QK ≈ PV ≈ 0.5 µs per 64-key tile).
  tvm-ffi needs `TVM_FFI_CUDA_ARCH_LIST=10.0a` or it also compiles a plain sm_100 target.
- Shared-memory fp32 atomicAdd reductions across 8 warps are ~10x slower than partial buffers.
- 2 CTAs/SM (95 KB smem) was slower than 1 CTA/SM (SM sharing of active CTAs > second-wave cost).
- Cross-SM communication through L2: a strong-load poll costs ≈ 0.6 µs + 0.04 µs/KB (one L2 round
  trip dominates), a relaxed atomic ~0.3 µs; `ld.global.cg` compiles to the same STRONG.GPU SASS.
  Self-validating (data, 32-bit call tag) 8-byte words let publishers use plain relaxed stores and
  mergers poll (no release fence / acquire reload) — this is what v7+ use. TMA bulk copies from L2
  (cp.async.bulk global->smem) are NOT faster for ≤ 16 KB polls (1.2 µs vs 0.77 per poll).
- Codegen traps (watch SASS / `cuobjdump`): a value check between loads serializes them (one
  round trip each); unrolled loops over "owned" items with predicates execute every item's work;
  > ~10 KB of outstanding loads per CTA stalls issue; dynamic indexing of register arrays spills.
  Latency-bound phases run at ~3.5 cycles/instruction with 2 warps per scheduler, so 100
  instructions on the critical path ≈ 0.2 µs. SM clock is at 1852 MHz during the benchmark.
  Runtime integer divisions on the index-load path cost ~0.05-0.1 µs (v12 removed them).
- Hardware waves (clusters of 8, 168 KB smem): 15 clusters co-resident; wave starts observed at
  0.0-0.2 / 1.54-1.8 / 2.9-3.4 / 4.3-4.5 µs (inactive cluster lifetime ≈ 1.1 µs + 0.45 placement).
  The cluster launch ORDER decides which clusters hit wave 3: v12 = first 3 tokens token-major,
  then seg 0 of the rest, seg 1 of the rest, (seg 2, seg 3) pairs per token: +2.3 speedup points.
  Pure token-major hurts T=6 all-small (tokens 4-5 in wave 2); FULL_TOKENS=4 overflows wave 1.
- Single-CTA epilogue: per-warp `stmatrix` staging in the dead Q region + 16-B stores of full
  128-B rows (g8) beats the register quad-transpose (-0.15 µs); smem-staged + CTA barrier or a
  TMA bulk store are +0.3-0.4 µs (completion wait / barrier on the critical path). The per-SM
  store path drains 16 KB in ≈ 0.3 µs.
- Prologue: per-warp direct loads of the tile's 64 slots (lane l: slots l, l+32) + two ballots let
  the gather issue ~0.3 µs earlier than the smem-hop/CTA-barrier path (g1: -0.25..-0.3 µs everywhere).
  Index row DRAM latency ≈ 0.86 µs from CTA start, gather first block +0.6 (light load) / +1.2 (load).
- Early (m, l) exchange with st.async right after the softmax (second mbarrier) lets owners compute
  merge weights during the slice pushes (g5: level-1 merge math 0.58 -> 0.2 µs). Carrying (m, l) in
  the slice padding instead (g2) was neutral.
- Remaining big-token critical path (02d6ae9c, wave 1): index 1.0, gather 1.2-1.4 + tile 0.5-1.0,
  softmax 0.5-0.7, PV 0.4-0.5, staging+issue 0.45, push wait 0.5-0.65, merge 0.2, L2 poll 0.8-1.0,
  tail 0.3. Compute phases are mma.sync-issue bound; gather per-SM bound; poll L2-latency bound.
- Phase 3c (2026-10-20) rejected the whole "software scheduling" family (persistent 15-cluster kernel
  with speculative first item = v9t 77.8x official; software waves via counter / L2 table / claims /
  deterministic takeover = v10-v11, rep8 7.86-10 µs): barrier.cluster.wait ~0.3 µs when it has to
  wait for peers, DSMEM/mbarrier exchange 0.3-0.5 µs, atomic 0.3-0.5 µs; 39 concurrent gathering CTAs
  take 2.4-2.7 µs per 73 KB tile vs 1.6-1.9 with ~23; nothing about other tokens is knowable before
  the index rows land. Max active segments per workload = 14; all 23 workloads are prefix-packed
  (valid slots first); the big token sits at index 2 in 11 of 13 large workloads.
- Correctness tooling: `scripts/dev_bench.py --scramble` permutes the top-k rows (holes) - the kernel
  must stay correct; `compute-sanitizer --tool synccheck` finds divergent barriers.
- Results: v8 79.13x -> v12 81.42x (launch order) -> v13 86.47x (g1+g4+g5), 23/23 official. g8
  (stmatrix epilogue) rep8 7.07 µs pending all-23.

- Phase 3d part 3 (g16-g25, 2026-10-20): the softmax of a 64-key tile was 0.84 µs of latency-bound
  scalar code (236 SASS instr: full-range exp2f 22, log2f 30, a divergent (m,l) send loop 64) → approx
  ex2/lg2 + loop-free send = g16 (-0.1 µs all-23). QK split as KG key groups x DG dim groups (big tiles
  2 n-tiles x 4 blocks per warp, 3 partials; KG=8 neutral) = g17 (-0.07); QK is at the mma.sync issue
  floor (~0.55 µs per full tile after the first block). Level-2 poll: bounded by outstanding load
  TRANSACTIONS per CTA (1/2/3 polled segments = 0.65/0.8/1.35 µs), so loading the (M,L) record once per
  head group + shuffle broadcast = g25 (-0.02..-0.1); publish stores become visible ~1.1-1.3 µs after
  issue regardless of L2 residency (prefetch hints, real-load warming, 8-byte words, zero-reset words,
  staggered double polls all neutral or worse; a second in-flight poll competes with the first).
  barrier.cluster.wait after the gather costs 0.00 µs; before the kpe wait (cp.async in flight)
  +0.2-0.45 µs. A non-cluster launch is 0.1-0.17 µs SLOWER than the cluster launch. Shared-memory
  bank conflicts on the S-partial stores were ~42 cycles per CTA (irrelevant). Duplicate writers of a
  flag word or multiple readers of a zero-reset word are correctness traps (scramble mode finds them).

- g26-g29 (2026-10-20): level-2 words of 6 fp16 + 32-bit tag (11 per head row) = g26 (-0.03 us); the
  level-1 merge indexed by position among the active ranks and specialized on 2/4/8 sources = g28
  (-0.16 us all-23: a 2-rank merge was 0.83 us with 8-source masked loops, now 0.29; partial last
  segments of big tokens publish earlier). Rejected: staggered double poll (every time, +0.2),
  atom.exch.b128 publish (+0.08). After g28 the big-token path is: index 0.9, gather-to-kpe 1.0,
  QK 0.9 (mma floor), softmax 0.5, PV 0.45, push 0.3 + wait 0.5, merge 0.25, poll 0.75 (polls=1
  mostly), tail 0.3, launch ~1.0 outside the CTAs (CUPTI 7.3 vs span 6.3 on 02d6ae9c).

- Phase 3e (2026-10-20): the level-2 critical path is the publishers' store visibility (publish issue at
  ~5.0-5.3 us + ~0.5 us), not the poll volume (polling 1 or 4 heads instead of 16 changes the poll by <= 0.1 us
  and just returns before the data is there); a distributed level-2 merge would gain <= 0.1. The sm_100a
  redux.sync.max.f32 over 16 lanes is slower than 4 shfl+fmax (+0.07 us/tile). 16 per-row 128-B
  cp.async.bulk stores per warp cost +0.45 us vs 4 st.global.v4 per lane. A 2-rank level-1 exchange spent
  half its DSMEM volume on self-copies (fixed in g33b: -0.15 us on e6b849f2, -0.3 on ddfa9e34/7a389715).
  Splitting the PV into dim halves with early pushes drains the mma pipeline (+0.3 us PV). Level-2 words of
  4 fp16 (16 per row) are +0.1-0.2 us vs 6 fp16 (11 per row); 32-byte words change nothing. A 64-key
  single-CTA tile costs only ~0.6 us more in-kernel than a 2-key one (gather ~73 GB/s per SM when few CTAs
  gather), so re-tiling small tokens over more CTAs cannot pay for a level-1 merge (~1.1 us vs 0.45 epilogue).
  Big-token gather is DRAM-contention bound (32 CTAs x 73 KB): kpe block lands 1.1-1.4 us after issue.

**Why:** these took hours of experiments; they constrain any future redesign of this kernel.
**How to apply:** do not retry rejected directions without a new reason; start from the latest commit on
`worktree-dsa-attn-dev`.
See also [[dsa-devspace-workflow]].

- Phase 3f (2026-10-21, g38-g41, dev all-23 6.49 -> 6.11 us): the prologue was software-bound, not DRAM-bound: the
  9 per-thread `cp.async.mbarrier.arrive` cost 0.15-0.25 us of issue and the gather loop 0.42 us even with no copies
  (inactive CTAs lived 1.3 us doing nothing after their index landed at 0.55); the 1152 x 16-B Q cp.async burst stalls
  issue for 0.45 us (hidden by the index latency, ~0.15 us of gather time). Fixes: inactive CTAs exit right after their
  own ballots (G1), per-warp gather ownership with cp.async commit groups + wait_group (G2: a lane mapping MUST keep 8
  lanes on one 128-B line - 16 B per lane = 8x the L2 requests; per-copy address math in the issue loop costs 0.3 us -
  precompute per-row bases), fused per-warp QK+softmax for tiles <= 4 n-tiles (G4: S fragment in registers, quad
  shuffles; for > 4 n-tiles the 8x Q smem reads lose). Hardware co-residency at 166 KB smem: 33 x 4, 15 x 8, 7 x 16
  clusters; a late cluster starts 0.45 us after a slot frees. Launch order G3 (segment 0 of every token, then segments
  1-3 in a per-T priority 2,5,6,4 / 2,6,5,4 / 2,4,5,3 matching the dataset's large tokens) put every large token's
  full segments in wave 1: 5096e459 9.5 -> 7.9, 564007ac 9.6 -> 7.9 us. compute-sanitizer memcheck reports nothing for
  a kernel that spins on an uninitialized mbarrier (macro defined after its first use) - check macro order.
  Later in 3f (rejected): fused softmax for config-4 tiles (+0.15 softmax, +0.13 QK: 2 CTA barriers + a partial
  exchange lose to the 3-partial softmax), register-resident Q for full tiles (neutral), row-major gather issue
  (DRAM page locality lands the tile 0.4 us sooner but losing the per-block QK overlap costs more), n-tile-major
  gather + per-n-tile QK (neutral). Warm kernel code is worth <= 0.05 us (instruction fetch is not a bottleneck);
  prefetching the code address range (device function pointer) faults -> 20-180 us. Code compiled in but not
  executed costs the other paths ~0.05 us (keep experiments compiled out). v21 official 100.23x, v22 (g43) 100.47x.
