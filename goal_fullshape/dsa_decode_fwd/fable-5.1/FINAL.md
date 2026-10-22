# DSA sparse attention decode (dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64) — final report

Hardware: NVIDIA B200 (devspace tasks/1321157), SM clock observed at 1155 MHz during all measurements (the GPU never left its idle clock state for these short kernels; both reference and solution were measured in this state).

Timer: CUPTI kernel span (first kernel start to last kernel end), L2 flushed before every call, median of 100 iterations, 3 alternating rounds (reference / solution) per workload, median of rounds.

Reference per workload: the fastest of KDA-0.5 (`kda05-ref --bench`) and the flashinfer trtllm sparse-MLA decode library call measured with the same timer (the FIB baseline flashinfer_wrapper_5af199 cannot run in this environment).

Correctness: error vs an independent fp32 naive implementation (naive/naive_fp32.py); tolerance = 7 x max(error of KDA-0.5 vs naive, 1 bf16 ulp at magnitude 1 / 1e-5 for lse), per workload.


| # | workload uuid | T | valid rows per token | ref (fastest candidate) | ref µs | KDA-0.5 µs | flashinfer trtllm µs | mine µs | ref/mine | out err (tol) | lse err (tol) | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0c23b10c7b7645719517828c12eaa1d2 | 1 | 2 | KDA-0.5 | 3.936 | 3.936 | 563.8 | 7.296 | 0.539 | 1.562e-02 (5.469e-02) | 9.537e-07 (7.000e-05) | PASS |
| 2 | 05f6de657db543ae9e4c46796522843a | 2 | 6,337 | KDA-0.5 | 10.607 | 10.607 | 610.4 | 15.040 | 0.705 | 1.562e-02 (5.469e-02) | 2.861e-06 (7.000e-05) | PASS |
| 3 | 0a63b87bb2e54e9db1ca3c4c53a1d521 | 2 | 63,9 | KDA-0.5 | 7.664 | 7.664 | 579.3 | 8.640 | 0.887 | 7.812e-03 (5.469e-02) | 2.861e-06 (7.000e-05) | PASS |
| 4 | 9d4a5f21268e484ea05a2f2af91d9fa7 | 2 | 18,11 | KDA-0.5 | 5.088 | 5.088 | 393.3 | 7.648 | 0.665 | 1.562e-02 (5.469e-02) | 3.815e-06 (7.000e-05) | PASS |
| 5 | 9f3f891bfbe24776adfcd5a579d093dd | 2 | 288,4 | KDA-0.5 | 10.768 | 10.768 | 616.2 | 14.560 | 0.740 | 1.562e-02 (5.469e-02) | 1.907e-06 (7.000e-05) | PASS |
| 6 | b7668cfd194c4b95ab600feb205ebac6 | 2 | 33,52 | KDA-0.5 | 7.136 | 7.136 | 544.6 | 8.128 | 0.878 | 1.562e-02 (5.469e-02) | 3.815e-06 (7.000e-05) | PASS |
| 7 | e6b849f2900446148c01d81152efae23 | 2 | 92,48 | KDA-0.5 | 10.881 | 10.881 | 604.8 | 13.984 | 0.778 | 1.562e-02 (5.469e-02) | 2.861e-06 (7.000e-05) | PASS |
| 8 | f77df5ce21634f7ca0586af7e3c13434 | 2 | 18,19 | KDA-0.5 | 5.280 | 5.280 | 578.2 | 7.712 | 0.685 | 1.562e-02 (5.469e-02) | 3.815e-06 (7.000e-05) | PASS |
| 9 | fc85411e250c41879b6d6a1edb80f0a7 | 2 | 17,13 | KDA-0.5 | 5.184 | 5.184 | 462.8 | 7.648 | 0.678 | 1.562e-02 (5.469e-02) | 2.861e-06 (7.000e-05) | PASS |
| 10 | 68d6817dcfd1433aa8d2ddeefd54b6ea | 6 | 19,20,32,12,25,3 | KDA-0.5 | 5.824 | 5.824 | 611.5 | 8.128 | 0.717 | 1.562e-02 (5.469e-02) | 3.815e-06 (7.000e-05) | PASS |
| 11 | d57eb9e19f0642e8af9bd76ad0823303 | 6 | 143,139,2013,142,306,539 | KDA-0.5 | 15.999 | 15.999 | 591.7 | 20.928 | 0.764 | 1.562e-02 (5.469e-02) | 3.815e-06 (7.000e-05) | PASS |
| 12 | ddfa9e340b264f76abe7418692faa876 | 6 | 6,9,9,14,1639,71 | KDA-0.5 | 15.296 | 15.296 | 407.4 | 17.729 | 0.863 | 1.562e-02 (5.469e-02) | 2.861e-06 (7.000e-05) | PASS |
| 13 | 2207f0fdc96347c59d106e4976cfad57 | 7 | 415,131,2011,148,263,169,462 | KDA-0.5 | 15.360 | 15.360 | 623.9 | 20.800 | 0.738 | 1.562e-02 (5.469e-02) | 2.861e-06 (7.000e-05) | PASS |
| 14 | 3838996164a94d728710f913477feba8 | 7 | 33,52,72,17,18,401,1089 | KDA-0.5 | 14.912 | 14.912 | 559.5 | 17.616 | 0.847 | 1.562e-02 (5.469e-02) | 3.815e-06 (7.000e-05) | PASS |
| 15 | ae4219a95f044f45bd10e17ab63c6e8f | 7 | 19,12,2048,21,26,46,136 | KDA-0.5 | 15.168 | 15.168 | 745.5 | 20.655 | 0.734 | 1.562e-02 (5.469e-02) | 3.338e-06 (7.000e-05) | PASS |
| 16 | 02d6ae9c64ab42ff93f05c23c53bcb7d | 8 | 63,9,2048,212,11,25,6,50 | KDA-0.5 | 15.488 | 15.488 | 588.6 | 18.688 | 0.829 | 1.562e-02 (5.469e-02) | 1.907e-06 (7.000e-05) | PASS |
| 17 | 232ed014bafc4835b9881bb308c659b0 | 8 | 35,54,74,19,20,403,1091,1 | KDA-0.5 | 14.784 | 14.784 | 637.2 | 19.936 | 0.742 | 1.562e-02 (5.469e-02) | 4.768e-06 (7.000e-05) | PASS |
| 18 | 385742b2717e4f02b918c7349dde23d8 | 8 | 92,48,1044,14,411,30,16,8 | KDA-0.5 | 14.639 | 14.639 | 579.2 | 17.151 | 0.854 | 1.562e-02 (5.469e-02) | 2.861e-06 (7.000e-05) | PASS |
| 19 | 4c46a94ba2364dc7ab476286dee8dce3 | 8 | 18,19,1002,31,11,316,24,2 | KDA-0.5 | 13.073 | 13.073 | 620.3 | 17.152 | 0.762 | 1.562e-02 (5.469e-02) | 4.768e-06 (7.000e-05) | PASS |
| 20 | 5096e459ce3f4cdf82773ce1a0c73c8a | 8 | 17,13,1887,16,180,1986,413,1 | KDA-0.5 | 15.921 | 15.921 | 598.4 | 19.855 | 0.802 | 1.562e-02 (5.469e-02) | 4.768e-06 (7.000e-05) | PASS |
| 21 | 564007ac354e4662a62cc4d6352dc494 | 8 | 288,4,1884,21,136,2048,42,335 | KDA-0.5 | 16.209 | 16.209 | 620.3 | 20.640 | 0.785 | 1.562e-02 (5.469e-02) | 2.861e-06 (7.000e-05) | PASS |
| 22 | 78b2e11c30004cceb84355722a7c6b0a | 8 | 18,11,2048,20,25,45,135,326 | KDA-0.5 | 16.160 | 16.160 | 622.3 | 20.000 | 0.808 | 1.562e-02 (5.469e-02) | 4.768e-06 (7.000e-05) | PASS |
| 23 | 7a389715dcc3479e9b0512309f9d1d56 | 8 | 8,11,11,16,1641,73,1,1 | KDA-0.5 | 15.168 | 15.168 | 582.2 | 17.504 | 0.867 | 1.562e-02 (5.469e-02) | 3.815e-06 (7.000e-05) | PASS |

**Geometric mean of ref/mine over all 23 workloads: 0.763**

Per-group geometric means: T=1: 0.539 (1 workloads), T=2: 0.748 (8 workloads), T=6: 0.779 (3 workloads), T=7: 0.771 (3 workloads), T=8: 0.805 (8 workloads)

Reference timings are the median over 3 alternating rounds of `kda05-ref --bench` (each itself the median of its iterations); solution timings use flashinfer.testing.bench_gpu_time_with_cupti (CUPTI kernel span, L2 flush before each call, 100 iterations, median), 3 rounds interleaved with the reference runs, median of rounds. Every timed call computes everything from the raw inputs; the only state kept across calls is a zero-initialised workspace and the compiled kernel.

All workloads within tolerance: True


## Official FIB evaluation

```
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 0c23b10c7b7645719517828c12eaa1d2: PASSED with 237.78x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 9d4a5f21268e484ea05a2f2af91d9fa7: PASSED with 273.76x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload b7668cfd194c4b95ab600feb205ebac6: PASSED with 277.95x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 0a63b87bb2e54e9db1ca3c4c53a1d521: PASSED with 256.49x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 05f6de657db543ae9e4c46796522843a: PASSED with 143.51x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload fc85411e250c41879b6d6a1edb80f0a7: PASSED with 320.76x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload e6b849f2900446148c01d81152efae23: PASSED with 145.89x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 9f3f891bfbe24776adfcd5a579d093dd: PASSED with 158.33x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload f77df5ce21634f7ca0586af7e3c13434: PASSED with 296.89x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 385742b2717e4f02b918c7349dde23d8: PASSED with 228.92x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 4c46a94ba2364dc7ab476286dee8dce3: PASSED with 230.39x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 3838996164a94d728710f913477feba8: PASSED with 194.92x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 02d6ae9c64ab42ff93f05c23c53bcb7d: PASSED with 220.04x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload ddfa9e340b264f76abe7418692faa876: PASSED with 176.17x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 78b2e11c30004cceb84355722a7c6b0a: PASSED with 189.42x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 68d6817dcfd1433aa8d2ddeefd54b6ea: PASSED with 374.58x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 564007ac354e4662a62cc4d6352dc494: PASSED with 204.57x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload ae4219a95f044f45bd10e17ab63c6e8f: PASSED with 174.22x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 232ed014bafc4835b9881bb308c659b0: PASSED with 190.72x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 7a389715dcc3479e9b0512309f9d1d56: PASSED with 212.91x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 5096e459ce3f4cdf82773ce1a0c73c8a: PASSED with 186.19x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload d57eb9e19f0642e8af9bd76ad0823303: PASSED with 157.15x speedup vs. mathematical reference
INFO flashinfer_bench.bench.benchmark: Solution 'dsa_fable51_v6_9' for workload 2207f0fdc96347c59d106e4976cfad57: PASSED with 175.11x speedup vs. mathematical reference

```


## Solution summary

# DSA sparse MLA decode kernel — design notes (solution/dsa_decode.cu, v5)

Problem: per token t, 16 heads, q = [q_nope(512) | q_pe(64)], top-2048 gathered KV rows
(ckv 512 + kpe 64, bf16; rows of 1024 B + 128 B), -1 = padding. Output softmax(QK^T*scale) V (bf16)
and base-2 LSE (fp32).

## Measured hardware facts that drove the design (B200, SM clock 1155 MHz in this environment)
* Empty-kernel floor under the CUPTI/cold-L2 timer: ~2.0-2.5 us. DRAM round trip ~1 us, L2 round trip ~0.5-1 us.
* Per-SM ingest is ~37-44 KB/us on every path (LSU 16-byte loads, cp.async, 1-D TMA bulk copies,
  tiled TMA gather4); 1-D TMA additionally costs ~50 ns per copy and ~1 us extra latency.
* A 16-CTA cluster at >110 KB smem allows only 7 resident clusters; DSMEM loads ~70 ns each, cluster
  barrier ~0.6-1 us (v1 used clusters and was abandoned).
* Instruction-level latency chains dominate short phases at this clock (~25-50 cycles per dependent
  step), so the kernel keeps dependent memory round trips to a minimum.

## Structure
* Work unit = (token, 32 or 64 consecutive raw positions of the top-k list). grid = T x NSPLIT, 256 threads.
  NB=2 (64-row units, 1 CTA/SM) when T*32 CTAs are co-resident, else NB=1 (32-row units, 2 CTAs/SM).
* Probe: a CTA reads only its own rows' indices; with no valid index it exits immediately (or, when
  the whole token is co-resident, becomes a combine helper).
* Workers load K rows straight from global memory into mma.sync m16n8k16 B fragments: warp w owns
  ckv dims [64w,64w+64) and kpe dims [8w,8w+8); 16-byte vector loads with a fixed permutation of the
  dims (applied identically to Q) form the fragments. The 8 warps' partial logits are summed through
  smem; softmax over the unit's rows is done by 4 warps; the same K registers become V fragments via
  movmatrix.trans for PV, so K/V bytes cross the SM boundary exactly once and never touch smem.
* The full index list (8 KB) is read by workers while their rows are in flight and only serves to
  learn which units of the token are workers (ballot/popcount); no compaction is needed because
  units are defined on raw positions and masked per column.
* Single worker: output written directly from registers. Several workers: each publishes a partial
  (bf16 O, fp32 m and l) and an epoch-stamped flag (st.release); warp 0 of each participant polls the
  flags (ld.acquire); then every participant finalises a round-robin subset of 8-dim output slices
  (weights exp2(m_r - M), batched loads, smem run reduction). No counters have to be reset.
* When the grid can exceed the co-resident capacity, CTAs take dynamic tickets so that token order
  equals start order, which makes the flag barrier deadlock-free; the host passes the ticket base
  and the epoch, so nothing is cached across calls except a zero-initialised workspace.
* Correctness: -1 anywhere, all -1 (zeros / -inf), out-of-range indices treated as -1, topk <= 2048
  (larger topk falls back to plain torch math), any T (ticket mode).

## Final kernel (v6.9) in one paragraph
Work unit = 64 raw positions of the token's top-k list (two 32-row batches), grid = T x 32 CTAs of 256
threads. Three builds of one kernel: T <= 4 -> 1 CTA/SM (190 registers); 5 <= T <= 9 -> 2 CTAs/SM
(register cap 128, tiny spill) so that every CTA is still co-resident and no tickets are needed;
larger T -> 1 CTA/SM with dynamic tickets. A CTA first probes only its own 64 indices (one 16-byte load
per thread per batch, tile row g of tile n being unit entry 4g+n); a CTA without any valid index exits
immediately (T >= 5) or becomes a combine helper (T <= 4, when all CTAs of a token are co-resident).
Workers stream their K rows straight into mma.sync B fragments, sum the eight warps' partial logits
through smem, run the softmax on four warps and reuse the K registers as V fragments via
movmatrix.trans; the full index list is read while the rows are in flight and only tells which units
are workers. A single worker writes the output directly; otherwise the workers publish bf16 partials
with fp32 (m, l), signal with epoch-stamped flags, and every participant finalises a round-robin subset
of 8-dim output slices (the m/l loads and the first slice group's partial loads are issued back to back
so the combine costs one L2 round trip after the barrier). Grids larger than the co-resident capacity
use dynamic tickets so that the flag barrier cannot deadlock.

## What was measured and why the design ended up here
* v1 (16-CTA clusters, K/V staged in smem through 1-D TMA bulk copies, DSMEM combine): correct, but
  15 us at T=1 and 28 us at T=8: 1-D bulk copies cost ~50 ns each per SM, 144 KB smem allowed only 7
  resident clusters (two waves at T=8), and the two DSMEM round trips plus two cluster barriers cost
  ~6 us.
* v2 (no clusters, TMA staging, global-memory combine): 10-30 us; the per-SM ingest rate is only
  ~40 KB/us on every path (LSU, cp.async, 1-D TMA, tiled TMA gather4), so wide splits are needed.
* v3-v5 (register-only K/V path, 32-row units, index compaction or raw units, counter / flag
  barriers): combine cost dominated; per-phase stamps showed ~1 us per dependent global round trip,
  ~0.7-1 us for a release fence, 1-2 us for polling, and that integer divisions, serialised loads
  and smem bank conflicts each added microseconds until removed.
* v6 (64-row units always): 2 us faster than 32-row units at 2 CTAs/SM on every large-T workload.
* v6.1: one 16-byte load per batch for the unit's indices (scalar index loads were costing ~0.3-0.5 us).
* v6.2 (skipping an empty second batch): -0.4 us at T=1 but +0.3 us on T=2 workloads; rejected.
* v6.3: combine loads issued in one round trip: -1.4 us on multi-worker T=2 workloads, neutral elsewhere.
* v6.4 (skip empty batches only in softmax/PV): same pattern as v6.2, rejected.
* v6.5: Q loads issued before the row loads, barrier after the worker list only on the combine path: -0.1 us on light workloads (final).
* v6.6 (no __launch_bounds__ register cap): the compiler chose fewer registers with spills, +0.3 us light / +3 us heavy; rejected.
* v6.7 (T<=4: 32-row units round robin over the 32 CTAs, no-combine fast path when all rows are in units 0-1): correct, but the mode decision needs the full list before the second batch and costs +1 us on light workloads with no gain on mid-size ones; rejected.
* v6.8/v6.9: a 2-CTA/SM build (register cap 128) for 5 <= T <= 9 makes all CTAs co-resident without tickets: +7.7% geomean over the 14 large-T workloads (T=6 light -14%, several T=8 -7..-21%, three within +3%); the T <= 4 build keeps 1 CTA/SM (final).
* v8 (one 16-CTA cluster per token, 32-row batches round robin, DSMEM push combine): correct, but
  slower on every workload (16 CTAs cannot ingest a 2048-row token as fast as 32-64 parallel CTAs,
  and packing two cluster CTAs per SM halves every phase); kept in exp/ for reference.
* Reference behaviour on this card (KDA-0.5): 3.9 us at T=1, ~5.1 us for light T=2, ~0.02-0.04 us per
  additional row, 13-16 us for T=7/8 with a 2048-row token; i.e. it is essentially at the launch +
  two-DRAM-round-trip floor for light tokens, which this design could not reach (its own in-kernel
  chain is probe -> rows -> ~2.5 us of exchange/softmax/PV latency chains).
