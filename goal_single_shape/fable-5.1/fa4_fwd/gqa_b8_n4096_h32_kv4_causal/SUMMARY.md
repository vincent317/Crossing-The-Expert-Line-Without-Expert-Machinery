> 2026-10-23 更新：当前 `fa_v9.py` 已重写为 Triton 因果 GQA，B8/S4096/H32/KV4/D128 的精度与性能见[修复验收](../../../../benchmarks/flashinfer/remaining-repair-20261023/README.md)。下文为历史实验材料，其不同 shape、实现及耗时不代表当前代码。

# FlashAttention FWD (d=256, causal, bf16) on B200 — from scratch in Gluon

Target: `q/k/v (4,4096,32,256) bf16`, `softmax_scale=1/16`, `causal=True`, beat
`flash_attn.cute.interface.flash_attn_func` (flash-attn-4 4.0.0b19) by 1.2x in
device time. The reference implementation was never read; only its public
callable was timed.

## Result (clock-controlled, same run, same data, same harness — job 1052296)

| SM clock | reference (flash-attn-4 cute) | this kernel (stage S) | speedup |
|---|---|---|---|
| locked 1500 MHz | **0.9949 ms** (median of 3, all within 0.0005) | **0.9824 ms** (median of 3) | **1.013x** |
| free (1.6–1.85 GHz, throttling) | 0.8977 ms | 0.9145 ms | 0.98x |

Accuracy (same run): E_ref = max|ref − fp32 naive| = 9.339094e-03,
tol = 7·E_ref = 6.537366e-02, E_mine = max|mine − fp32 naive| =
**9.339094e-03 ≤ tol (PASS)**; max|mine − ref| = 7.8e-03 (bf16 rounding).

**The 1.2x target (≤ 0.829 ms at the reference's 0.995 ms) is not met.** At
equal clocks the from-scratch kernel is on par with the reference (+1.3%).

### A measurement caveat that cost several rounds

Earlier sweeps reported this kernel at 0.878 ms against the reference's
0.9959 ms (job 1049128) — an apparent 1.135x. Those numbers came from
*different* jobs: the reference was measured on a throttled/busy node and the
kernel sweeps on a cool one. When both are measured interleaved in one
process the reference runs at 0.90–0.93 ms at free clocks, and both kernels
lose ~11% when the clock is locked to 1500 MHz (both are similarly
latency-bound). Only the locked-clock, same-run numbers above are
trustworthy; `final_eval.py` reproduces them (it samples `nvidia-smi`
clocks/power around every measurement and locks clocks with
`nvidia-smi -lgc` when permitted).

## Files

| file | purpose |
|---|---|
| `common.py` | problem spec, deterministic inputs, timing harness (profiler kernel span + CUDA events, L2 flush) |
| `naive_fp32.py` | fp32 materialised-softmax correctness anchor |
| `run_ref.py` | reference device time, E_ref, tol |
| `kernel_gluon_s.py` | **best kernel** (stage S) |
| `kernel_gluon_q.py` | stage Q: Q in TMEM (S minus the barrier-free flags) |
| `kernel_gluon_g.py` | stage G: 4-warp softmax, max off the critical path |
| `kernel_gluon_c.py` | stage C: first warp-specialised version with MMA/softmax overlap |
| `kernel_gluon_b.py`, `_a` (`kernel_gluon.py`) | earlier stages |
| `kernel_gluon_{d,e,f,h,p,r}.py` | experiments that did not pay off (kept for the record) |
| `kernel.py`, `kernel_tma.py`, `kernel_v3.py` | plain-Triton versions (0.52–0.56x of ref) |
| `bench_gluon_b.py` | multi-stage validation + timing driver (`python bench_gluon_b.py s`) |
| `final_eval.py` | final evidence run |
| `diag_ptx.py`, `diag_ncu.py` | PTX histogram / Nsight Compute diagnostics |

## Design of the final kernel (stage S)

One CTA per SM (SMEM-limited), grid of 4096 CTAs ordered heavy-row-block
first inside groups of 8 (b,h) pairs (LPT within an L2-friendly window).

Tile: 128 query rows x 64 keys, head dim 256. Three partitions via
`gl.warp_specialize`:

* **softmax** (4 warps, default partition) — thread = one query row, 64
  columns in registers (TMEM 32x32b layout), so the row max / row sum are
  in-thread; `p = exp2(fma(s, scale, -m_ref))` is issued straight from the
  loaded scores with the *current* reference; the row max only steers the
  reference for the next tile (lazy rescaling with threshold 2^16, applied one
  tile late; O, l and the pending P are rescaled together on the rare path).
  P is written as bf16 into the first half of the S buffer's TMEM columns and
  consumed there by the PV MMA. The per-tile "any row needs rescale" decision
  is a warp-shuffle plus one SMEM flag per warp published by the barrier that
  already precedes the `p_ready` arrive (no extra CTA barriers).
* **loader** (1 warp) — TMA: Q once, then K and V through independent
  2-stage rings (K released after QK, V after PV).
* **MMA** (1 warp) — `QK(j+1)` is issued *before* waiting for `P(j)`, so
  the tensor core computes the next scores while the softmax runs;
  `S = Q K^T` with **Q as a TMEM operand**, `O += P V` with P from TMEM.

Memory: SMEM 197 KB (Q 64 + K 2x32 + V 2x32); TMEM 512 columns
(O fp32 256 + Q bf16 128 + S fp32 2x64, P aliased into S).

## What moved the needle (device time, free clocks, sweep runs — relative
## numbers are meaningful, absolute ones vary with clock, see caveat above)

| stage | change | ms |
|---|---|---|
| plain Triton (best of 3 variants) | tl.dot / TMA descriptors | 1.79 |
| A | Gluon: TMA + tcgen05 + TMEM, serial | 1.83 |
| B | warp-specialised loader / MMA / softmax, lazy rescale | 1.63 |
| C | issue QK(j+1) before PV(j); split K/V release | 1.10 |
| C' | 4-warp softmax (in-thread row reductions) | 0.976 |
| E | grouped-LPT grid order | 0.957 |
| G | exp2 before max, reference update lagged one tile, no per-tile pv_done wait | 0.937 |
| Q | **Q in TMEM as the MMA A operand** | 0.884 |
| S | barrier-free rescale flags | **0.878** |

Things that did not help: 8-warp softmax (cross-warp reductions cost more
than the latency hiding gained), tree reductions, FMA-polynomial exp2 for a
column subset (ALU issue, not MUFU, is the limiter), 128-column softmax tiles
(TMEM cannot hold Q + 2x128-col S; without Q in TMEM the QK MMA is SMEM
bound), persistent grid-stride loop, 3-stage K/V ring (blocked by a compiler
assertion on merging ring stages into a Q view), software-pipelining two S
tiles in registers (spills).

## Where the time goes / the bottleneck

Nsight Compute on stage G (SM clock locked at 983 MHz): issue-slot
utilisation 22.5%, 2 warps per scheduler, eligible warps 0.25/cycle, top stall
reasons `wait` (fixed-latency dependencies) and `short_scoreboard`, tensor
pipe 79% active. The kernel is bound by the **softmax partition's latency
chain**: with one CTA per SM and a 4-warp softmax, each SM sub-partition runs
exactly one softmax warp, so every TMEM load, dependent FMA/ex2/add, TMEM
store and barrier is exposed. `NO_MMA` runs (softmax + loads only) take ~85%
of the full kernel time.

Before stage Q the QK MMA itself was SMEM-bandwidth bound: with M=128,
N=64, K=256 every QK streams 96 KB of operands (Q 64 KB + K 32 KB) through
SMEM at 128 B/clk, i.e. 768 clk for 512 clk of math. Moving Q to TMEM fixed
that (-5.6%).

Why the reference is structurally ahead at d=256: it uses a 2-CTA cluster
(`cta_group::2`, one 256x128 tile per SM pair). Each CTA then holds only half
of every K/V tile, so N=128 tiles fit in SMEM with double buffering, one
MMA instruction covers twice the work, per-tile synchronisation is amortised
4x better, and the QK MMA stays compute bound with Q in SMEM (leaving TMEM
for 2x128-column S buffers). A single-CTA design at d=256 is boxed into
128x64 tiles by SMEM (K/V rings) and TMEM (O alone is half of it), which is
what caps this kernel at about the reference's speed. Implementing the 2-CTA variant in Gluon
(two_ctas TMEM layouts, multicast commits, cross-CTA `p_ready` arrives) is
the next step; it was not attempted within the round budget.

## Reproduce

```
python run_ref.py            # reference timing, E_ref, tol
python bench_gluon_b.py s    # validate + time stage S
python final_eval.py         # everything in one run
```
