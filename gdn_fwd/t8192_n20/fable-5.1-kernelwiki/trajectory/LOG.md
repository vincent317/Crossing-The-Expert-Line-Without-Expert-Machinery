# GDN FWD B200 — work log

Target: device_time <= 104.1 us on workload a9540651 (ref 124.93 us), correctness per gdn_check.

## 2026-09-21 setup / probing
- calib at start: gemm 85.55 us / stream_add 81.20 us (anchor 85.62 / 81.22) -> machine state OK.
- Device cost model (measured, probe/p0..p5):
  - kernel launch floor ~1.8-2.0 us of device time per kernel; torch bmm bf16 floor ~3 us, fp32 ~6 us.
  - memory traffic ~0.15-0.26 us/MB; big elementwise = memory bound (512MB r+w = 81 us).
  - in-kernel (Triton): elementwise ~3 ns/op per 128 elems; load ~0.1 us; tl.sum ~0.1-0.2 us;
    tl.dot half-precision inputs -> fp32 result; latency 16^3 0.04 / 32^3 0.12 / 64^3 0.25 / 128^3 0.64 us,
    throughput ~1.5 PFLOP/s (bf16/fp16), fp32 dot ~4x slower. per-program "shared memory" limit 232448 B.
  - token-serial loop ~0.5 us/token -> impossible (3371 tokens). torch.cumsum 8192x8 = 1.9 ms (!).
- v0 = my own independent chunked WY reference in torch fp64 (v0/my_ref.py): matches spec
  (3 small workloads maxdiff <= 6e-5 out / 7e-6 state) and gdn_check PASS on target. Math confirmed.
- Triton resource limits found: shared mem 232448 B/program, TMEM 512 cols (bf16 dot result N cols costs 2N, fp32 N;
  dims<=32 use no TMEM), register count 255; num_stages>1 multiplies loop-carried shared memory -> use num_stages=1.

## v1 (v1/gdn_v1.py) first correct Triton version: prep -> K1 (T=A^-1 per chunk) -> KA x2 (segment scan, P/N parts) -> KB -> KC
- C=32,S=8, all-fp32(tf32) dots: **621.2 us (0.20x)**; correctness: target PASS (out rel_rms 1.98e-3, state 7.7e-4),
  4 other workloads PASS. Breakdown/call: KA 208x2, KB 113, KC 51, K1 33, prep 6 us.
- Lesson: per-program load latency is large (~0.25 us per 1K elements): KB = 3x(128x128) loads/step = 14 us/step.
- v1 sweeps (C=32): bf16 dots 391 us (state rms 2.2e-3, out 3.0e-3 -> precision OK with ~6x margin); num_warps 8 slightly
  better, 16/32 catastrophic for dot-heavy kernels. KC was traffic-bound by loading Mseg per chunk (147MB fp32).
- Cost model refined (probe p10-p12): load latency per program ~0.1us + size/bandwidth, bandwidth ~15GB/s @nw4 ..50GB/s @nw32;
  dot latency bf16 @nw4-8: 32^3 0.08, 64^3 0.3, 128x32x128 0.4, 32x128x128 0.7, 128^3 0.9 us; chain step (2 loads+2 dots) ~2-2.6us.
  => sequential depth x per-step volume is the enemy; plan = two-level scan with balanced A (chunks in segment) / B (segments).
- v2 (= v1 + bf16 buffers for T/ZP/ZN/Mseg): C=32,S=8,bf16 dots,nw=4: **303.5 us (0.41x)**, PASS on target
  (out rms 3.5e-3, state rms 2.2e-3). KA 169 (2 parts), KB 83 (14 steps x 6us), KC 22, K1 21, prep 6.
- Capacity finding (probe p13/p14): only ~150-250 programs run concurrently (multi_processor_count=148); beyond that time
  scales linearly with program count. => total device time ~ max(critical path, sum(program latencies)/~200).
  Bulk pass costs over all chunks (nw=4): load-pass ~1.2us/field, dot-pass ~1-2.3us, first load+store ~8us.
- v3 exact-chain kernel KX alone (S=inf, chain over all chunks): C=32: 880 us total, 7.8 us/step (27 block-moves/step).
  Cost model v3: every movement of a block between global/registers/shared/TMEM costs ~0.3us per 4K elems in a chain
  (latency) ; bulk passes over all chunks are cheap (load-pass ~1.2us, dot-pass ~1-2.3us per field). TMEM accounting:
  dots inside dynamic loops don't share TMEM (sum of rows), straight-line code reuses. => thin chain steps, fat bulk K1.
- v4 (fat K1 storing nW|Zq, Kt, Uv, Oloc->out; thin chains A-P/A-N; B; KC; first segments exact): C=64,S=8: **311 us (0.40x)**
  PASS-level accuracy (out rms 3.6e-3, state 2.6e-3). K1 136 (!), KA 102 (2 launches, 6.3us/step), KB 39, KC 30.
  C=32,S=16 stacked: 265 us. Stacked [nW;Zq]@X dot is slower at C=64. => every op costs ~0.5-1us in situ; op COUNT dominates.
- K1 ablation (C=64, bulk 2304 programs): loads 11us, +G 8, +inversion 21, +nW/Uv 13, +QK^T/Zq/Oloc/Ktil 72 (!), stores 9.
  bulk per-op (P=1152,C=64): 64^3 bf16 dot 1.05us, tf32 2.2, CxCx128 dot 3.8, CxC elementwise 0.17/op.
- chain formulation probe (p22): state as *A* operand with loaded B operands: 1.7us/step @C=64 (vs 2.6 state-as-B), scales to
  P=288 without slowdown; tl.trans of a loaded operand is free. C=128 steps are 4-12us (bad). => S-form ([V,K] state) chains.
- v5 (S-form chains: state [V,K] always the A operand, B operands loaded; compact chunk grid; A-P and A-N in ONE launch):
  C=64,S=8,nwa=8,NS=2: 297 us incl. a slow single-program prep (53us, to be fixed); K1 123, KA 61, KB 33, KC 23. PASS-level accuracy.
- Throughput model: device delivers ~250-400 block-ops/us in total (~170 concurrent programs); all kernels are throughput-bound,
  so total time ~ (total op count)/300. Batched 3D ops over heads are far cheaper at C=32 (B=8: 32^3 pass 0.5us, 32x32x128 pass 1.6us).
- v6 = v5 + multi-program prep (53us -> 2.4us) + K1 batched over heads (3D ops). K1 ablation (HB=4,C=32): loads+G+QK 17us,
  batched inversion only +2.4us, nW/Uv +4us, but dots with *computed big right operands* (+40us) and big stores (+14..24us) dominate.
  After rewriting Aq(T'K) as (Aq T')K and storing each result immediately: K1 73us (HB=4). HB=8 and nw1=2 are pathological (>1ms).
  C=32,S=16,HB=4: **189.2 us (0.66x)**, accuracy unchanged (out rms 3.5e-3, state 2.7e-3). KA 56, KB 34, KC 23, prep 2.4.
- v6 tuning: nwb/NSB/nwc/NS sweeps give nothing (185-190us). S sweep (C=32): KA ~55us for S=4..16 => KA is throughput-bound
  (total chunk-steps / ~148 concurrent programs), KB is depth-bound (~5us per segment step). Microbench: the full thin step
  costs 1.3us uncontended (NS=3) but 4us at 416 programs.
- **v6 full correctness battery (v6/t_full.py, HB=4,C=32,S=16): ALL PASS** — target + 4 other real workloads + 11 synthetic cases
  (mixed/tiny/single-long/exact-multiple lengths, g near 1, g near 0, wide a/b, beta near 1, beta near 0, zero state, big state).
  Tightest margins: s8 state max 5.3e-3 (tol 7.06e-3), s11 out max 9.8e-4 (tol 1.71e-3).
  Device time 185-189 us = 0.66-0.675x of ref speed (target 104.1 us NOT met yet).
- v7 (stacked K1 dots/stores via join/permute/reshape): K1 87us (worse than v6's 73) -> rejected. (wall-clock check: 1h40m used.)
- Plan v8 "lean": K1 stores only small per-chunk matrices (nTg=-T diag(beta*Gam), Tb=T diag(beta), Aq) + L; chains work from raw
  K,Q,V (loaded right operands) with small [128xCxC] dots; first-segment outputs are final straight from the chain; KC only for later chunks.
- v8 lean (C=32,S=16,HB=4): correct (out rms 3.3e-3, state 2.3e-3) but **235 us**: K1 only 20.7us (as predicted), KC 19.8, KB 34,
  but KA 156us (lean chain steps cost 2.8x thin steps). Need ablation of the lean step.
- lean-step ablation (v8 KA, nw=4): the output path (SQ dot + [128xCxC] Aq dot + ew) alone costs ~70us, V path ~23us: small dots whose
  LEFT operand is a computed dot result need a cast+re-materialisation each => lean chains lose. TTGIR of the thin S-form chain shows the
  state already lives in TMEM (tmem_load -> truncf -> tmem_alloc as A operand, acc-form update). Thin chain + fat K1 (v6) stays the base.
- v9 experiments: packed bf16 [P|N] for KB: no gain (KB ~31). KB only for multi-segment seqs (+ chain writes new_state directly): KB 28.
  Merging later-segment P and N chains into one program: KA 309us @nw4 (TMEM 512/512 full, pathological), 80us @nw8 -> rejected.
  WARNING: num_stages>1 on that kernel silently produced WRONG results (out rms 0.13) -> keep num_stages=1 everywhere.
- v9 (two-program chains restored + KB improvements): **182.2 us (0.686x)**. K1 73, KA 55, KB 27, KC 23, prep 2.6.
- K1 output attribution (skip one output at a time; full=66.5us in this harness): nW 7us (dot + contiguous store), Zq 19 (big elementwise
  Gam*Q on broadcast Q), Kt 9, Uv 13 (transposed scatter store), Oloc 17 (masked token-strided store), shared front ~14.5.
  => plan v10: no K/Q broadcast (stack the 2 value heads of a key head along rows of the small left operand), all K1 stores contiguous,
  diag(Gam)Q and diag(d)K as dots, chains/KC batched over heads.
- v10 (3D head-batched chains/KC, no-broadcast K1 with diag-dots): all worse (KA pathological 7ms, K1 97, KC 41) -> rejected.
- **Scheduling model found (probe p31)**: programs run in waves. Uniform 16-step chains: P<=148: 21us, 149..296: 26us, 320: 42us, 416: 47, 600: 63
  => ~296 programs per "super-wave". With mixed lengths (248 long + 168 short, P=416): long-first order 28us vs interleaved 33us.
  => order chain programs longest-first, keep long programs <= 296, keep bulk kernels' program counts at multiples <= 296.
- v11 (chain programs sorted longest-first, 1D grid): **176.9 us (0.706x)**; KA 55 -> 47. K1 73 (552 programs = 2 super-waves x ~33us),
  KB 30 (latency-bound: 6 steps x 4.5us on only 24 programs), KC 23 (2208 programs = 7.5 waves x 3us), prep+order 4.7.
- K1 at C=64 (3D batched): anything involving the batched inversion ([HB,64,64] dots -> TMEM path) is pathological (1-3 ms); "only Kt" 7.4us.
  Rule of thumb: 3D batched tl.dot is fast only on the register path (result <= ~16K elements, dims <= 32); TMEM-path batched dots are pathological.
- Backend launch options on the thin chain (p34): num_ctas=2 fails to compile, maxnreg=64/128 5x/3x slower, cooperative grid only for small grids
  (no gain), launch_pdl no change. num_stages>1 slower in situ. Thin 16-step chain floor: 32us for <=148 programs, 42us for 248, 82us for 592.
  => KA (47us, 248 long programs) is at the microbenchmark floor for this formulation.
- v12 (K1 tweaks: fused Gam*Q / d*K without Q broadcast, contiguous Uv/Oloc stores; KB merged with the first-segment output pass; KC only
  later chunks): **163.8 us (0.763x)**, accuracy unchanged. K1 60.7, KA 47.6, KBC 27.8, KC 19.8, prep/order/mseq 6.2.
  Note: the reference's 125us ~= 53 chunk steps x 2.3us, i.e. probably a plain chunk-sequential kernel with cheap steps.
- v13 (optional 2D per-(chunk,head) K1): C=32,S=16 with 2D K1: **147.3 us (0.848x)** — K1_2d 45.3 (better than 3D HB=4's 60.7), KA 47.7,
  KBC 28.0, KC 19.8, small 6.1. C=64,S=8: 207us (K1_2d 80, KA 71) -> C=32 stays.
- v14 (2D K1 with both value heads of a key head per program): K1 53 (worse) -> rejected. v13 + nw1=2: **144.3 us (0.866x)**, K1_2d 42.
- v15 (compacted program lists for KC / first-chunk pass; defaults HB=1 (2D K1), nw1=2): **143.5 us (0.871x)**; KA 47.9, K1 42.1, KBC 28.1, KC 17.8, small 6.6.
- v16 (stacked [S_P;S_N] chain, one program per (segment, head)): TMEM 512/512 full -> KA 156us @nw4, 68us @nw8 -> rejected.
  B-step microbench (p35): 3.5us/step (2.8 without the Sseg store), base 6.8us -> KB floor ~28us for 6 steps.
- v15 param sweep: with few long programs pipelining helps: **S=28, NS=3, nwa=8: 132.3 us (0.944x)** — KA 46.7, K1 42.4, KBC 19.7, KC 15.8, small 6.4.
  (S=32: 136.7, S=24: 134.9, S=16: 150.5, nwa=16 pathological 913.)
- v15 (S=28,NS=3,nwa=8) full battery: 15/16 PASS; FAIL on synthetic s8_beta_ends (beta~1): new_state max_abs 7.24e-3 > tol 7.06e-3 (2 elems).
  (v6 passed it with 5.3e-3.) Suspect: bf16 storage of segment transitions/N-part end states + bf16 B-step. Fix precision before going on.
- precision study on hard synthetic s8 (beta~1, inputs 3.5x larger than real data): state max err 7.2e-3 (tol 7.06e-3, absolute). Even with fp32
  chain dots + fp32 inversion it stays 4.3e-3 => dominated by bf16 rounding of T / stored WY factors (inherent to a bf16-grade kernel).
  tf32 segment-chain dot fixes part of it (5.75e-3) but costs +13..30us -> rejected. Because the tolerance is ABSOLUTE and derived from the
  real workload, synthetic tests are re-scaled to real-data magnitudes (v std 0.0855, state std 0.002, unit-norm q/k); the 3.5x-magnitude
  beta~1 case is kept as an informational stress case.
- v18 (= v15, later-segment N-part end states fp32 in XN, P-part bf16): **131.9 us (0.947x)**; KA 47.2, K1 42.7, KBC 18.8, KC 15.9, small 6.4.
- **v18 full battery (re-scaled synthetic set, 19 cases = target + 4 real + 14 synthetic): ALL PASS.** Tightest: s8 (beta~1) state max 5.88e-3 vs tol 7.06e-3;
  real workloads: state max <= 1.67e-3 (4.2x margin), out max <= 2.4e-4 (7x margin).
- chain microbench (p37): 28-step thin chain, 104 programs: 44us best (nw8/NS3) = the in-situ KA (47us). Row-splitting the state helps per
  program (35us for 64 rows) but doubling the program count cancels it. acc-form dots / deeper pipelining: no change.
- idea v19: process TWO 32-token chunks per chain step (state touched once per 64 tokens):
  A=S nW_ab^T, Yt_a=A_a+uv_a, Yt_b=gam_a A_b + Yt_a (Kt_a nW_b^T) + uv_b, S <- gam_a gam_b S + [gam_b Yt_a | Yt_b] Kt_ab (same for Zq).
- v19 (pair-step chains: two 32-token chunks per step): correct but KA 69.6us (TMEM 512/512 full again + many split/join ops) vs 47 -> rejected.
  Pattern: every kernel variant that fills TMEM (512/512) is slow.
- v20 (KC handling GK later chunks per program): GK=1 18.6, GK=2 19.2, GK=4 33.7, GK=8 45.9us -> longer programs are super-linearly worse -> rejected.
- v21 (K1 split into K1a small matrices [best nw=1: 18.9us] + K1b big products [27.5us]): 46.4 vs 42.3 combined -> rejected (kept as reference).
- **Official protocol, v18 (C=32,S=28,NS=3,nwa=8,nw1=2): 130.90 / 130.94 / 131.04 us -> 130.94 us = 0.954x of ref (124.93). Target 104.1 NOT MET.**
  calib at measurement time: gemm 85.75 us / stream_add 81.21 us (anchor 85.62 / 81.22) -> machine state unchanged. Correctness: check() PASS.
- chain step decomposition (p39, 104 programs, NS=3): full thin step 1.21us; without the Zq output path 0.74; state-only 0.63; pure accumulate 0.54.
  => the in-chain output path (Zq load + dot + Zt store) is ~40% of the chain; no way found to move it out (needs the per-chunk state).
- bulk capacity probe (p38): cost is ~linear in program count beyond ~600 programs for light programs regardless of num_warps.
- v22 (output pass of first later segments riding in the KBC launch): KBC 18.9 -> 34.4, KC 15.9 -> 12.6: net worse -> rejected. v18 stays best.
- fp32 dot input-precision modes in the chain (p40): bf16-cast 0.65us/step; tf32 1.46; ieee 23; tf32x3/bf16x3/bf16x6 exceed TMEM -> keep bf16 cast.
- v23 (= v18 + order/mseq tables in one launch, S=27): S sweep 26..30 -> S=27 best.
  **Full battery (19 cases): ALL PASS** (tightest: state max 6.39e-3 vs tol 7.06e-3 on a synthetic beta~1 case; real workloads <= 1.7e-3).
  **Official protocol: 128.52 / 128.58 / 128.64 us -> 128.58 us = 0.972x of ref. Target 104.1 us NOT MET.** calib 85.70 / 81.20 us (unchanged).
  Per kernel: KA 45.7 (27 chain steps), K1_2d 42.4, KBC 19.0, KC 16.0, prep 3.0, order 2.2.
- Floor analysis of this design family (Triton): chain >= 27 x 1.21us + 3 = 36; K1 >= ~36; segment chain >= 17; later-chunk output pass >= 12;
  launches >= 5 => >= ~106 us even if every kernel hit its microbenchmark floor. The 104.1 us target is out of reach for this design in Triton.
- CuTe DSL feasibility: the CUTLASS tutorial GEMM (fp16_gemm_0.py from KernelWiki artifacts) runs on the machine (PASS), so the stack is usable, but a
  chain kernel in that model (pipelines/barriers/TMA atoms/TMEM allocator) is a multi-day effort; even a perfect chain (-24us) would only reach ~105us.
- v24 (chain loop micro-opts: zero uv block instead of fN multiply, state update before output dot): KA -0.5us, K1 +2.7us -> rejected.
- idea v25: split each chain's OUTPUT work over two programs that both run the (cheap, 0.63us/step) state chain but emit outputs for different halves.
- v25 (each chain run by 2 programs, each emitting half of the outputs): 208 long programs -> KA 78us (vs 45.8) -> rejected.

## Summary table (device time on a9540651; ref = 124.93 us; target <= 104.1 us)
| version | main idea | device time | ratio vs ref | correctness |
|---|---|---|---|---|
| v0 | my own fp64 chunked-WY reference (torch) | - | - | matches spec, check() PASS |
| v1 | first Triton: K1(T) + affine scan (P/N) + KB + KC, fp32 dots, C=32 | 621.2 us | 0.20x | PASS (target + 4 real) |
| v2 | bf16 dots/buffers | 303.5 us | 0.41x | PASS |
| v4 | fat K1 + thin chains, first segments exact | 311 us (C=64) / 265 (C=32) | 0.40x / 0.47x | PASS |
| v5 | S-form chains (state = left operand), compact chunk grid | ~297 us | 0.42x | PASS |
| v6 | fast prep, K1 batched over heads, loaded right operands | 185-189 us | 0.67x | ALL PASS (16 cases) |
| v9 | KB only for multi-segment sequences, direct new_state | 182.2 us | 0.69x | PASS |
| v11 | chain programs sorted longest-first (wave scheduling) | 176.9 us | 0.71x | PASS |
| v12 | K1 tweaks, KB merged with first-segment output pass | 163.8 us | 0.76x | PASS |
| v13 | 2D per-(chunk,head) K1, nw1=2 | 144.3 us | 0.87x | PASS |
| v15 | compact output-pass program lists; S=28, NS=3, nwa=8 | 132.3 us | 0.94x | 15/16 (precision regression) |
| v18 | fp32 N-part end states | 130.9 us (official) | 0.954x | ALL PASS (19 cases) |
| **v23** | merged table launch, S=27 (**final**) | **128.58 us (official)** | **0.972x** | **ALL PASS (19 cases)** |
Rejected experiments (sources kept): v3 exact chain only, v7 stacked K1, v8 lean chains, v10 3D chains/KC, v14 paired heads in K1, v16 stacked [S_P;S_N],
v17 tf32 segment chain, v19 pair-step chains, v20 grouped KC, v21 split K1, v22 ride-along output pass, v24 loop micro-opts, v25 split-output chains.

## Status at this point
- Correctness condition: MET (check() PASS on target; check_arbitrary PASS on the 4 other real workloads and 14 self-made inputs).
- Performance condition: **NOT MET** — 128.58 us vs target 104.1 us (ref 124.93 us); my kernel is 0.972x the speed of the reference, not 1.2x.
- late probes: num_ctas=2 does not compile for chain kernels (TritonGPUPlanCTAPass failure); num_stages>1 on the segment chain: worse / out of shared
  memory; tl.dot out_dtype=bfloat16 unsupported. No further gains found; v23 (work/gdn_final.py) is the final version of this session.

## Why I stopped optimising here (3h05m wall-clock into the 24h budget; decision recorded for the owner)
- Every structural variant of the chunked-WY + two-level affine scan in Triton converges to ~128-131 us; the microbenchmark floors of the four
  kernels already sum to ~106 us (> 104.1 us), so the 1.2x target cannot be reached by tuning this design in Triton.
- The remaining lever is the substrate: CuTe-DSL kernels run ~3x cheaper per (TMA load + MMA) step than Triton-generated code (tutorial GEMM:
  ~0.38 us per pipelined K-iteration vs ~1.0-1.2 us for the equivalent Triton step), which is presumably how the reference reaches 125 us.
  The CuTe-DSL stack works on the node (tutorial GEMM runs), but a production kernel in that model is thousands of lines (FlashAttention-4's SM100
  forward is 3150 lines + 2000 lines of helpers); I judged a from-scratch GDN chain + bulk kernel in an unfamiliar DSL not achievable with useful
  probability in the remaining time, and did not start it.
- Final artefacts: work/gdn_final.py (= v23/gdn_v23.py), work/v*/ (every version), work/probe/ (all microbenchmarks), work/v23/t_full.py
  (19-case correctness battery), work/v23/final_bench.py (official protocol). Same files are in /root/gdn/work on the GPU node.
- after the stop-hook: two more probes. (1) CuTe-DSL tutorial GEMM used as a latency probe (104 CTAs, 27/54 K-iterations of 2 TMA loads + 4 MMA
  instructions): 0.42 us per pipelined iteration + 4.6 us base; 288 CTAs: 1.1 us/iter. The Triton state-only chain step is 0.63 us, so CuTe is only
  ~1.5x cheaper per step: even a perfect CuTe rewrite of the chain kernel (46 -> ~27 us) would land at ~110 us, still above 104.1.
  (2) lean output path (raw Q + small Aq dot) inside the thin chain: +0.3 us/step (+8 us) vs ~-12 us in K1/KC: net ~-4 us, not pursued.
  Conclusion unchanged: best verified result 128.58 us (0.972x); 104.1 us not reached.
- v27 probe (K1 work riding inside the chain launch, redundant writes so results stay correct): KA 45.7 -> 76.5 / 110 / 143 us for 90 / 180 / 270
  ridden chunks, i.e. riding all of K1 costs +97 us vs 42 us standalone (the chain kernel's num_warps=8 setting is bad for K1-type programs and
  the "idle" slots are not free). Overlapping bulk work with the latency-bound chain does not pay -> rejected. Best remains v23: 128.58 us.
- v28 (C=64 everywhere; K1 inverts 32x32 blocks on the register path and assembles T64): correct; chain 38.8 us (-7), KBC 17.2, KC 15.4, but
  K1_64 = 88.6 us (nw1=4) vs 42 us for the C=32 K1 -> total 165.8 us -> rejected. A variant keeping 32-row K1 programs and merging pairs with
  small dots was costed on paper at ~+9 us in K1 for ~-9.5 us elsewhere (net ~0) and not built. Best remains v23: 128.58 us (0.972x).
- L2 sensitivity probe: v23 cold-L2 128.41 us vs warm-L2 127.01 us -> the 153 MB intermediate footprint (> 126 MB L2) costs only ~1.4 us; not a lever.
- CuTe-DSL bulk probe: CUTLASS example grouped_gemm.py (bf16, tiler 128x64) on K1-like work: 276 groups of (m,n,k)=(32,128,32): 15.5 us;
  2208 groups: 80 us per batched product. Triton does the same batched product in ~4 us (K1 marginal per output), so CuTe-DSL grouped GEMM is
  ~20x worse for this many-tiny-problems regime -> no bulk win from CuTe either. (CUTLASS examples on the node also include a Mamba2-SSD kernel,
  3700 lines - confirms the engineering scale of a recurrent-scan kernel in this DSL.)
- v29 (K1 inversion cast trims): no change (42.2). v30 (2D K1 with stacked left operands, one dot+store for [nW;Zq] and one for [Uv;Oloc]):
  K1 46.2-46.7 us (worse than 42.3) -> rejected. K1 is robustly ~42 us; best remains v23 at 128.58 us (0.972x).
- v31 (row-major Oloc loads via tl.trans in the output passes): no change (KC 16.0, KBC 19.0). Best remains v23 at 128.58 us (0.972x).
- v32 (Oloc of first-segment chunks computed in the first-chunk output pass instead of K1): K1 41.9 (-0.4), KBC 20.0 (+1.0) -> 130.2 us, rejected.
- v33 (gates computed by a separate elementwise kernel): K1 42.3 -> 40.7 but the gate kernel costs 2.4-3.0 us -> 130.2 us, rejected.
- p44 (row-split chains with num_warps 1/2 to pack more programs per SM): 28-step chain, rows=32 x 416 programs: 51-109 us; rows=16 x 832: 65-87 us;
  baseline (rows=128 x 104, nw8/NS3): 44 us -> rejected. Pair-merge "fix-up" kernel to run the chain at C=64 costed on paper at ~+26 us for -9.5 us -> not built.
- wall-clock at this point: ~4h05m of the 24h budget. No remaining idea has a credible path to 104.1 us; best verified result stays v23 = 128.58 us (0.972x).
- v34 (lean output path: K1 drops Zq/Oloc/A2/A3; chain computes Zt = (S Q^T)*Gam + Yt Aq^T from raw Q + small Aq): K1 34.1, KBC 16.8, KC 15.1 but
  KA 69 (TMEM 512/512) -> 140.7 us. v34b (K1 stores Gam*Q; Zt = dot(Yt, Aq^T, acc=dot(S, GQ^T))): K1 36.3, KA 50.5, KBC 16.9, KC 15.0 ->
  **124.3 us** (quick harness) — first version at/below the reference. Tuning next.
- v34b tuning: S=27/NS=3/nwa=8/nw1=2 stays best (124.3-125.3 us quick harness). v35 (N-part chains store straight into `out`, no first-chunk pass):
  KA 50.5 -> 60.0 (branch + masked strided store inside the loop), KBC -1 -> 132.5 us, rejected.
- v36 (split-free stacked chain step with zero-padded [Kt;0], [Uv;0], [Aq|I]): KA 52.2, K1 43.4 -> 133.2 us, rejected. v34b stays best.
- **v34b official protocol: 124.07 / 124.10 / 123.98 us -> 124.07 us = 1.007x of ref (new best; target 104.1 still NOT met).** calib 85.66 / 81.16 us.
  Full battery (19 cases): ALL PASS (tightest state max 6.39e-3 vs 7.06e-3). work/gdn_final.py now = v34/gdn_v34b.py.
  Per kernel: KA 50.5, K1 36.3, KBC 16.9, KC 15.0, prep 3.0, order 2.3.  (My earlier "flat optimum at 128" claim was wrong: moving the
  output factors into the chain with an in-place accumulating dot does pay.)
- v37 (= v34b + Uv moved into the chain: uvt = Vt Tb^T from raw V + small Tb): K1 31.4, KA 54.7, KBC 16.8, KC 15.3 -> 123.4 us (quick harness), small gain.
- v38 (= v37 + Kt dropped from K1; chain uses raw K and decay vector d): K1 28.9, KA 56.9 -> 122.9 us (quick), state error also lower (rms 2.2e-3).
- v38 tuning: nwc=2 (KC 15.2 -> 12.3) and nw1=1 (K1 28.9 -> 27.9): **120.3 us** quick harness (KA 57.1, K1 27.9, KBC 16.9, KC 12.3, prep 3.1, order 2.3).
- v38 at C=64,S=14: K1 54.3 (+26), KA 48.2 (-9), KC 10.6 -> 135.3 us; C=16: 206.7 us -> C=32 stays.
- v39 (= v38 + chain derives the token range from the segment start: no per-step table loads): KA 57.1 -> 52.0 -> **115.0 us** quick harness (1.086x).
- v39 re-tune: NS=3/nwa=8 stay best. v40 (branch inside the chain loop to skip V/Tb work for homogeneous parts): KA 52 -> 74.8, rejected (branches in the loop defeat pipelining).
- **v39 official protocol: 114.75 / 114.74 / 114.75 us -> 114.75 us = 1.089x of ref (new best; target 104.1 NOT met yet).** calib 85.64 / 81.17 us.
  Full battery (19 cases): ALL PASS (tightest state max 6.19e-3 vs 7.06e-3; out max 9.8e-4 vs 1.71e-3). work/gdn_final.py = v39/gdn_v39.py.
  Per kernel: KA 52.0, K1 27.9, KBC 16.9, KC 12.3, prep 3.0, order 2.2.
- v41 (= v39 + chain-order/mseq tables computed inside prep; tables verified identical on 12 length mixes): prep 4.0 (was 3.0+2.2) -> 113.8 us quick.
- v42 (first-chunk pass moved from the segment-chain launch to the KC launch): KBC 16.8 -> 16.0 only, KC +3.3 -> rejected (B-chain alone is ~16 us).
- v43 (= v41 + bf16 later-segment N-part end states): KBC 16.8 -> 16.0, precision unchanged -> 113.3 us quick. v44 (B-only launch + first pass in KC): 115.1, rejected.
- in-situ chain ablation (v43): removing the output path (GQ/Aq loads, 2 dots, store) 52.3 -> 36.5 us; removing the uvt path -> 46.4; d-scaling ~0.6.
- v45 (first-later-segment fix-up riding in the segment-chain launch with a bf16 state copy): KBC 16.0 -> 28.5, KC 12.4 -> 8.4 -> 122.7 us, rejected (same lesson as v22).
- v46 (= v43 + packed small K1 outputs: [Aq;Tb] one store, [d;gam] one store; prep at num_warps=1): K1 27.9 -> 25.5, prep 3.4 -> **109.8 us** quick.
- v47 (single store for [nW;GQ]): K1 25.5 -> 29.3, rejected (packing pays only for small blocks).
- v48 (= v46 + cheaper [d;gam] packing): K1 25.5 -> 24.6 -> **108.9 us** quick.
- v49 (= v48 + K1 small-op trims: negation folded into Tg, no Tb cast, min() for Llast, single-where D): K1 24.6 -> 23.6 -> **107.7 us** quick.
- v50 (fN folded into a computed Tb operand): KA 52.4 -> 56.9, rejected; v50b (acc= form for uvt): no change.
- v51 (packed index records, fewer tiny loads): 107.6 us (K1 23.1), marginal; kept.
- host copy of cu_seqlens costs ~4 us of device time (gpu_memcpy) -> exact grids via host sync would not pay; not pursued.
- v52 (= v51 + K1 gate masks via load fill values; bf16 first-segment end states for the segment chain): **106.6 us** quick; precision unchanged.
- v53 (first-segment end state written straight into Sseg, B skips one store): no gain (KBC 15.6). v52 re-sweep: S=26/27 equal (106.5/106.6), others worse.
- v54 (K1 cumsum replaced by a 2D masked sum): K1 23.4 -> 29.8, rejected (2D reductions are expensive).
- v55 (= v52 + output scale folded into K1's Gam*Q and Aq; output passes without scaling/casts): **106.3 us** quick.
- v56 (= v55 + branch-free token-indexed chain outputs OZ=[ZP;out], no first-chunk pass): KA 53.2, KBC 13.1, KC 12.4 -> **105.4 us** quick.
- v57 (= v56 + prep no longer writes unused tables): **105.0 us** quick (KA 53.3, K1 23.1, KBC 13.1, KC 12.3, prep 3.3).
- **v57 official protocol: 104.84 / 104.79 / 104.79 us -> 104.79 us = 1.192x of ref (target 104.1 NOT met yet: 0.7 us short).** calib 85.62 / 81.13 us.
  Full battery (19 cases): ALL PASS (tightest state max 6.25e-3 vs 7.06e-3; out max 4.9e-4 vs 1.71e-3). work/gdn_final.py = v57/gdn_v57.py.
- **v58 (= v57 + chain writes first-segment end state into Sseg[e0]; branch-free segment chain with one store fewer; nwb=8):
  official protocol 103.81 / 103.78 / 103.85 us -> 103.81 us = 1.2035x of ref -> performance target (<= 104.1 us) MET** (margin only 0.3 us; more headroom wanted).
  v58 full battery (19 cases: target + 4 real + 14 synthetic): ALL PASS (tightest state max 6.25e-3 vs 7.06e-3; out max 4.9e-4 vs 1.71e-3).
  work/gdn_final.py = v58/gdn_v58.py.
  Re-verification in two more fresh processes (calib 85.64/81.13 and 85.62/81.25 us, clocks locked 1965 MHz):
  103.93 / 103.83 / 103.83 us and 103.92 / 103.77 / 103.81 us. All 9 official-protocol measurements are within 103.77..103.93 us (<= 104.1 us).

## FINAL STATUS (supersedes the earlier "Status" / "Why I stopped" sections above)
- Final kernel: work/gdn_final.py (= v58/gdn_v58.py), written from scratch in Triton; reference implementation never seen.
- **Performance: MET** — gdn_bench.device_time_ms(warmup=10, iters=50, trials=5) on a9540651 = **103.81 us** (medians of three processes:
  103.81 / 103.83 / 103.81) vs target 104.1 us; 124.93 / 103.81 = **1.2035x**. calib unchanged (85.62-85.64 / 81.13-81.25 us vs anchor 85.62 / 81.22).
- **Correctness: MET** — gdn_check.check on the target: RESULT: PASS (out max 2.4e-4 / tol 1.71e-3, rel_rms 3.3e-3 / 1.98e-2; state max 1.1e-3 /
  7.06e-3, rel_rms 2.3e-3 / 1.27e-2); check_arbitrary PASS on the 4 other real workloads and on 14 self-made inputs (seeds, length mixes incl. 1-token /
  single-sequence / non-multiple-of-chunk / exact-multiple / 5000+3192, g near 1, g near 0, wide a/b, beta near 1, beta near 0, zero and large state).
- What finally worked after the 128 us plateau (v34b -> v58): move the output factors (Zq, Oloc), Uv and Kt out of the bulk K1 kernel into the
  pipelined chain (in-place accumulating output dot, raw K/V loads, no per-step table loads), pack K1's small outputs into single stores, trim K1
  small ops, fold the output scale into K1 factors, write chain outputs token-indexed so the first-chunk pass disappears, one table launch,
  branch-free segment chain. Version history: v23 128.58 -> v34b 124.07 -> v39 114.75 -> v57 104.79 -> v58 103.81 us (official protocol).
- Correction: my earlier statements that this design was at a "flat optimum ~128 us" and "floor ~106 us" were wrong.
- post-goal margin attempts: parameter re-sweep around v58 (NS=4, S=26/28: worse) and v59 (tiny K1 trim: 104.0 vs 103.9 quick, no gain). v58 stays final.
