---
name: gdn-prefill-solution-state
description: GDN prefill solution = Triton recurrent + chunked + chunk3(3-kernel) with sync-free dispatch; full-100 ~35.4ms (official harness, locked clocks), 100/100, after P7/P8/P10/P11; -11% vs P6's 39.75. CONVERGED through P14: scan reg-reduction (P13), dispatch re-tune (P13), and chunk3 chunk-size C64 (P14) all rejected with fresh NCU/fastlat (scan=122 regs, 4-block reg ceiling, latency-bound; xl occ-bound at shallow depth not depth-bound). P15: OFFICIAL CLI + isolated-runner acceptance 100/100 confirmed at final code (commit 357a65b)
metadata:
  node_type: memory
  type: project
  originSessionId: d0404407-3b29-4c05-ac87-3112e08c89f3
  modified: 2026-10-20T18:00:09.633Z
---

**P7-P11 (2026-10-20 opus48 session): 39.75 -> ~35.4 ms (-11%), 100/100
(authoritative full-100 harness dev_bench --no-profile-baseline, locked 1830;
worst state abs 5.53e-3).** KEPT changes:
(1) **P7 chunk3 scan/out BV decouple** (commit ba0f536, the biggest win): Sc/Wc
are stored full-V, so the occupancy-starved serial scan (NCU after P5: occ 10%,
233 regs, 58% of chunk3) uses a SMALL BV=8 (occ 17%, 122 regs) while the TC-bound
parallel out uses a LARGE BV=64; -20% on the 15 chunk3 workloads (envs
GDN_C3_SCAN_BV=8, GDN_C3_OUT_BV=64).
(2) **P8 chunk BV=64 for high-N xl** (55f98f2; T>=7000 & N>=45 = the 3
chunk-routed xl N=48/56/57): enough seqs to fill at BV=64, bigger MMA tiles win,
beats chunk3 for them.
(3) **P10 chunk3 lower-T dispatch 3500->3000** (GDN_T_LO): after P7 sped chunk3
~20% its break-even T dropped (3271N2, 3028N5 flip to chunk3).
(4) **P11 recurrent threshold 25->16** (GDN_AVG_THRESH): fused chunk is a FLAT
~0.016ms for any short seq while recurrent grows with max_seqlen, so route all
but the tiniest (<=~15 tok) to chunk.
**P9 REJECTED (critical negative result): fused-chunk inverse tf32x3->tf32 FAILS
5/100** (state max_abs > atol; f5619793 diverges to 2.84e16). The nilpotent-
doubling (I+L)^-1 needs L^C==0 EXACTLY; tf32 rounding breaks nilpotency and the
residual diverges for ill-conditioned (I+L). Data-dependent, not depth-dependent
-> a 6-deep-case spot check PASSED and misled. **The state-path UT inverse is the
one MMA that must stay tf32x3; gate every precision change on the FULL 100.**
Exhausted (no reliable further win): num_warps/num_stages (noise), chunk size C
(32 best), prepare-inverse (same tf32 failure), fused-chunk occupancy.
**P12 (2026-10-20, commit 692dbf4, NO code change — stays P11): reconfirmed the
chunk3 serial-scan (the dominant bottleneck: 58% of xl path, xl=57% of full-100)
is at a local optimum.** Full-100 fastlat reproduced at 36.14ms. Same-session A/B
on the 8 slowest chunk3-routed workloads: scan num_warps 4->8 (GDN_NWS=8) +8%
subset (scan MMAs [128,8]/[32,8] too small — isolates P7's combined nw8 reject);
scan/out num_stages 1->2 (GDN_NSS/NSO) neutral-to-worse (reg+smem pressure drops
the already-starved occupancy). **Only credible remaining win = a log-depth
PARALLEL associative scan over chunk-states** (S_{i+1}=decay_i*S_i+delta_i is a
1st-order linear recurrence, combine (a2,b2)∘(a1,b1)=(a1*a2,a2*b1+b2)); helps only
the ~3 very-low-N deep cases (N<=2, est 3-5% full-100), big rewrite + new
correctness surface, judged not worth it at this maturity.
**P13 (2026-10-20, commit a5f2e89, NO code change — stays P11): two more low-risk
NCU-grounded directions REJECTED.** Fresh full-100 fastlat=35.74ms, 100/100
reconfirmed on all 3 paths (worst abs 2.49e-3). Route split: chunk3 59.4%
(21.24ms, 13 xl T8192 N20-43 dominate ~17ms), chunk 40.4%, rec 0.2%. **Fresh NCU
corrected the stale '233 regs': current chunk3 scan (BV8) = 122 regs, Block-Limit-
Registers=4 blocks/SM, theo-occ 25% ach 17.6%, latency-bound (DRAM 3.8% SM 18.6%);
a 5th block needs <=102 regs.** (A) Precompute the scan's gate (Kd f32 + egc_last)
into prepare so the scan LOADS them instead of recomputing 16x/vblock (bit-
identical; padded rows exactly 0 so stores drop the mask): regs 122->106, **4
SHORT of the cliff**, occ unchanged; removing the gate 'filler' EXPOSED mem latency
the 4-block occ can't hide -> scan 582->648us, xl-13 16.43->17.35ms (+5.6%); nw8/2
worse. REVERTED. The scan is at a hardware wall (tiny dependent [32,8] MMAs, 4-block
reg ceiling). (B) Dispatch re-tune: xl chunk3->chunk-bv64 noise-level/non-monotonic
(two T8192N32 disagree -> no sync-free T,N rule); low-N chunk->chunk3 chunk wins all
16 (7.92 vs 11.75). Confirms P8/P10. **NCU insight: the 13 xl are occ-bound at
SHALLOW depth (~5-6 chunks/seq), NOT serial-depth-bound — so the associative-scan
idea would only help the 4 genuinely-deep low-N chunk3 cases (~4ms). CONVERGED.**
**P11-confirm (2026-10-20, NO code change): fresh authoritative full-100 reproduced
100/100 PASSED, sum=35.554ms, worst abs=5.535e-3 (5d3fc66a T8192N34), top-5 latency
all T=8192 chunk3 (1.35-1.38ms) — exactly matches the converged record. Reconfirmed
the --no-profile-baseline gotcha the hard way: WITHOUT it the harness cupti-times the
O(T) reference ~18x/workload incl all T=8192 and stalls 50+ min with zero output;
WITH it the full 100 finishes in ~3 min.**
**P14 (2026-10-20, commit 0e69e10, NO code change — stays P11): swept the chunk3
chunk size C (GDN_C3), the one depth lever P12/P13 named but never directly A/B'd
for chunk3 (only the fused chunk's C was). C=64 (halves serial depth = ceil(seqlen/C))
net 4.7% WORSE on the deep+xl subset (SUM 6.395->6.698): xl regress +6-12%
(5d3fc66a 1.341->1.420, 26244fb4 1.307->1.463) because bigger C -> bigger per-chunk
MMAs + CxC prepare inverse -> more regs -> lowers the already 4-block-starved xl
occupancy; shallow low-N 109addb1 +20% (can't amortize 64x64 inverse). Only the
single deepest N=2 case 07aa7922 wins (-2.7%, ~0.03ms on full-100 = sub-noise).
C must be power-of-2 (nilpotent-doubling inverse) so only 32<->64 exist. REJECTED
-> chunk3 C=32 is a genuine local optimum, directly confirming P12/P13: the
serial-depth direction does not pay because the dominant xl are occupancy-bound,
not depth-bound. CONVERGED.**
**P15 (2026-10-20, commit 357a65b, NO code change — stays P11): OFFICIAL
ACCEPTANCE confirmed at final code.** Ran the task's stated acceptance command via
the official `flashinfer-bench` CLI **with --use-isolated-runner** (multi-process
IsolatedRunner, unlike P11-confirm which used the in-process Benchmark machinery),
scoped `--solutions gdn-prefill-opus48-v1 --no-profile-baseline`, locked 1830 MHz,
in tmux: **100/100 PASSED, 0 FAIL/ERROR, rc=0, 100 traces saved, ~13.5 min.** This
closes the last validation gap — CLI + isolated-runner acceptance was previously
only confirmed at P3's 55ms code. Freshly re-packed solution.json is byte-identical
to the registered `/root/fib-dataset/solutions/mine/.../gdn-prefill-opus48-v1.json`.
Launcher `/root/run_official.sh` on devspace; log `/root/gdnws/official_full100.log`.
Solution **CONVERGED + OFFICIALLY ACCEPTED** at ~35.4-35.6ms, 100/100.
**P16–P23 (2026-10-20→21, eight separate sessions, NO code change — stays P11):
repeated fresh-session handoff re-verifications, all identical outcome.** Each
confirmed: devspace tasks/1295739 reachable (B200, CUDA13.0, torch2.13.0+cu130,
triton3.7.1); artifact chain byte-identical — live /root/gdnws/solution/triton/
kernel.py sha256[:16]=`ec7764da62a5b3e8` == registered `gdn-prefill-opus48-v1.json`
sources[0] (BYTE-IDENTICAL==True → officially-accepted artifact == current code);
registered-json FULL-FILE sha256=`3ac07201…bdfc1395` (the embedded-code sha256[:16]
is the `ec7764da…` one — don't confuse them). dev8 `dev_bench.py dev8 --no-baseline
--no-profile-baseline` @ locked/stock = **8/8 PASSED**, sum 2.95–3.09ms across the
runs (P16 2.983, P17 3.012, P18 3.017, P19 3.092, P20 3.023, P21 3.015, P22
2.955, **P23 (2026-10-21 Opus-4.8 session) 3.032ms**), worst abs 6.14e-4 (c5257f65) /
d49df0b2 5.51e-4 ≪ atol 0.01, rc=0. **P22 FLAG LESSON (re-hit the hard way):
`--no-baseline` is NOT enough — it only drops the competing baseline solution; the
harness still BUILDS + cupti-TIMES the O(T) reference (`run_ref`→`build_baseline`→
`bench_gpu_time_with_cupti`), which stalls 10+ min at 100% CPU / 0% GPU on the
T=8192 workload. MUST also pass `--no-profile-baseline` → dev8 finishes in ~1 min.**
Earlier runs also: clocks reset
`-rgc` after; Mac deliverables worktree clean @ c3fd322; baseline still
RUNTIME_ERROR (documented). P16 also did an independent code audit (recurrent +
chunk3 scan/out) — documented structure confirmed, no missed inefficiency. No code
change warranted in any of these. **CONVERGED + OFFICIALLY ACCEPTED.** (Future
sessions: handoff is stable; a single artifact-sha + optional dev8 check suffices —
no need to append another near-identical P-entry unless something actually changes.)

**fastlat (scripts/fastlat.py) — the key iteration tool.** The harness computes a
slow O(T) correctness reference per workload (minutes for the 18 T=8192). fastlat
replicates the EXACT harness LATENCY recipe
`mean_over_trials(median_over_iters(flashinfer.testing.bench_gpu_time_with_cupti(
run, args, cold_l2_cache=True, use_cuda_graph=False)))` (bench/evaluators/
default.py + bench/timing.py) on run() alone, materializing inputs via
`load_safetensors`+`gen_inputs` — skipping the reference. Validated within 0.1%
of the P6 harness json on the rep9 sum (per-workload within ~3%, ~10% on the
deepest low-N). Use it for ALL A/B sweeps; confirm finalists + 100/100 on the
real harness. NOTE the old warning "cupti-direct over-sends to chunk3" was about
a DIFFERENT (per-kernel / warm) probe; fastlat times the FULL run() with
cold_l2+no-graph exactly like the harness, so it matches. Measurement noise on
chunk3 HBM-streaming kernels is ~8% run-to-run — only trust >5-10% deltas; warp/
stage micro-tuning and most dispatch-boundary flips are lost in it.
---

Solution for `gdn_prefill_qk4_v8_d128_k_last` (worktree branch
`worktree-gdn-prefill-dev`, dir `.claude/worktrees/gdn-prefill-dev`):
`solution/triton/kernel.py`, value-returning `run()`, config.toml `kernel.py::run`,
dps=false. **100/100 PASS**, full-100 **39.75 ms** (official flashinfer-bench
harness, cold-L2 cupti, GPU clocks LOCKED 1830 MHz). Was 49.76; **P5/P6 (2026-10-20
opus48 session)** cut it −20.6% via **tf32 precision right-sizing**: the chunk3
intermediates (U/W/S_c/W_c) are bf16 and all output-path MMAs feed only the bf16
`output`, so tf32x3 was pure waste (3 MMA passes + register pressure). Set
scan `w@S`, chunk3-out `Q@S_c`/`QK@W_c`, and fused-chunk-out `Q@S`/`QK@W` to tf32
(envs GDN_SCAN_WS_PREC/GDN_OUT_PREC/GDN_CHK_OUT_PREC, now default tf32). STATE-path
MMAs stay tf32x3 (feed fp32 new_state); state max_abs 5.5e-3 << atol 0.01. NCU
(`profile/chunk3_xl_t8192/`) found the chunk3 scan kernel is 58% of the xl path &
occupancy-bound (SM 11%, occ 10%, 233 regs) — tf32 cuts regs→occupancy. Rejected:
prepare u,w tf32 (state margin) and any dispatch widening (harness over-charges
chunk3's 3 launches: all-chunk3=75.8ms, high-N-xl→chunk3=40.7ms, both worse).

Three kernels + **sync-free** auto dispatch (uses only T,N — no cu read):
`avg<25 → recurrent`; `(T>=7000 & N<=44) or (T>=3500 & N<=13) → chunk3`;
else `chunk`.
- **recurrent**: fused per-token scan (BV16,NW8). tiny only.
- **chunk**: fused single-kernel chunked delta-rule (C=32, CBV adaptive 16 if
  N<=2 else 32, NW4). nilpotent-doubling (I+L)^-1, tf32x3 on inverse/W/Q@S/QK@W.
  Wins the broad middle (on-chip, 1 launch).
- **chunk3**: 3-kernel chunk-parallel (prepare || over all chunks: inverse+u+w;
  light serial state-scan storing S_c,W_c; parallel output). bf16 intermediates.
  Only for the ~12 deepest scans (long seqs).

**CRITICAL calibration lesson**: the official metric is the flashinfer-bench
harness which times **wall-per-iteration with cold_l2_cache=True, INCLUDING host
launch gaps**. So (1) any `cu.tolist()` sync in run()'s dispatch is counted
(~10ms/100 → dispatch MUST be sync-free), and (2) the harness penalizes chunk3's
3 launches + cold-L2 HBM traffic. A warm-L2 latbench oracle and a direct cupti
probe BOTH mis-calibrate (over-send to chunk3). Must fit dispatch on the harness
with **locked clocks** (`nvidia-smi --lock-gpu-clocks=1830`). Locked harness
all-100: all-chunk 52.58, auto 49.76 (-5.4%), oracle 48.65.

Dev loop: `scripts/sync_devspace.sh '<cmd>'` → `/root/gdnws`. `scripts/latbench.py`
(fast warm-L2 wall, tuning ONLY — not the real metric), `scripts/quick_test.py`
(correctness), `scripts/cupti_oracle.py` (per-workload per-mode cupti),
`scripts/dev_bench.py all --no-baseline --no-profile-baseline` (harness cupti;
`--no-profile-baseline` AVOIDS the reference-timing hang: the reference launches
O(T) tiny kernels that cupti chokes on for large T). Long runs in tmux. See
[[gdn-prefill-baseline-unrunnable]].
