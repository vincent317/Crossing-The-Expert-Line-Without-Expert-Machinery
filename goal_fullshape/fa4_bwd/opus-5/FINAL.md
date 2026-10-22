# Causal FlashAttention backward (bf16) on NVIDIA B200 — from-scratch kernels

Target: the two FIB definitions `attention_bwd_causal_h16_d128` (bshd, 16 query
heads, head_dim 128, MHA or GQA) and `attention_bwd_varlen_causal_h16_d128`
(thd + cu_seqlens) — three workloads in total.

All numbers measured on the same B200 (devspace `tasks/1321278`): CUPTI
kernel-span timing, cold L2 before every call, reference and solution measured
back to back in the same process, median of **39 independent runs**. How that is
done, and what it took to make it trustworthy, is in "How these numbers were
measured" below.

## Results

| workload | ref (fastest lib) | ref us | mine us | ref/mine | my max abs err vs fp32 | tolerance (7x ref err) | ref max abs err |
|---|---|---|---|---|---|---|---|
| `attention_bwd_causal_h16_d128`, B=8 N=4096 H_kv=16 (MHA) | cuDNN (torch SDPA CUDNN_ATTENTION) | 1321.0 | 1733.1 | **0.762x** | 0.0208 | 0.1428 | 0.0204 |
| `attention_bwd_causal_h16_d128`, B=8 N=4096 H_kv=2 (GQA 8:1) | cuDNN (torch SDPA CUDNN_ATTENTION) | 1358.0 | 1663.2 | **0.817x** | 0.0419 | 0.3172 | 0.0453 |
| `attention_bwd_varlen_causal_h16_d128`, 13 seqs / 32768 tok, max_seqlen 8192 | flash-attn-4 4.0.0b19 (flash_attn.cute) | 1558.8 | 1998.2 | **0.780x** | 0.0231 | 0.1304 | 0.0186 |

Geometric mean of ref/mine over the three workloads: **0.786x**

`ref/mine > 1` would mean faster than the reference; these are still **slower**
than the fastest library on this card, at 75-80% of its speed. See "Where the
time goes".

Both library baselines, measured the same way (us):

| workload | flash-attn-4 b19 | cuDNN | mine |
|---|---|---|---|
| MHA | 1369 | 1321 | 1733 |
| GQA | 1478 | 1358 | 1663 |
| varlen | 1559 | n/a | 1998 |

### How these numbers were measured

All numbers measured on the same B200 (devspace `tasks/1321278`), CUPTI
kernel-span timing with a cold L2 before every call, median over the repeats
(`flashinfer.testing.bench_gpu_time(..., enable_cupti=True, cold_l2_cache=True)`),
reference and solution measured back to back in the same process. The table
reports the **median of thirty-nine independent such runs**, each of five
interleaved rounds. Per-workload spread across those thirty-nine is 1705-1784
us on MHA, 1652-1683 on GQA and 1965-2041 on varlen -- about 2-4.5%, and the
reference moves far less, so the ratio is condition-sensitive at roughly the 1%
level: a card left to settle to idle (30 C) before each run reads about 1%
better than one measured on a 100-second cadence. Both kinds of run are in the
pool rather than only the flattering ones, and the pooled figure moved by 0.004
across fifteen, twenty-one, twenty-seven and thirty-nine runs (0.782, 0.783,
0.785, 0.786), so it has converged.

A third caveat, for reading the sweep output in `scripts/reproduce.sh`: a
one-knob-at-a-time sweep run late in a long script is indicative only. In one
such run the preprocess tile size looked 4.8% better at BMp=64 on GQA, with the
whole baseline 80 us slower than normal -- the card had been under load for half
an hour. Re-measured on a settled card (31 C, idle, `bench/bmpab.py`, seven
interleaved rounds) all four values of BMp land within 0.6% of each other and
the shipped 32 is tied for best. Every shipped choice rests on a dedicated A/B
like that one, not on a position in a long sweep.

Two things the harness now enforces, both learned the hard way. It refuses to
time anything while another process holds the GPU (`require_idle_gpu`), after a
background job outlived its launcher and sat at 100% utilisation through a
measurement session -- interleaved A/B ratios survive that, absolute
microseconds do not. And it spins the candidates for 150 ms before timing,
because on an idle card the first milliseconds run below boost clock, which is
worth about 3% and does not land on every candidate equally; with the warm-up,
three independent runs now agree to a few tenths of a percent. Every warm-up
call computes from the raw inputs like any other, so nothing is cached.

## What is being compared

* **ref** for each workload is whichever allowed library baseline was actually
  fastest on this B200, measured in the same process as my kernels:
  * `flash_attn.cute._flash_attn_bwd` (flash-attn-4 4.0.0b19 — the FIB baseline
    `flash_attn4_wrapper_b19_*`), and
  * cuDNN through `torch.nn.functional.scaled_dot_product_attention` under
    `sdpa_kernel(SDPBackend.CUDNN_ATTENTION)` (dense layouts only).

  cuDNN wins both dense workloads; only flash-attn-4 covers the varlen layout.
  The library is only ever *called*, never read.
* **mine** is `solution/attn_bwd.py`, written from scratch: the dK/dV pass in
  **Gluon** (`triton.experimental.gluon`, Triton's low-level Blackwell dialect:
  explicit tensor-memory allocation, `tcgen05_mma`, mbarriers, TMA and warp
  specialisation) and the preprocess and dQ passes in Triton. No pre-existing
  attention implementation is used — the kernels are built from `tcgen05_mma` /
  `tl.dot` / `exp2` / TMA descriptors and friends only.

## Correctness

Tolerances follow the prescribed recipe: run the exact fp32 reference (the
semantic reference shipped with the definition, with
`torch.set_float32_matmul_precision("highest")`), measure the *reference
library's* own max absolute error against it, and take 7x that as the bar. My
error is at or near the reference's own error on every workload, i.e. well
inside the bar.

Layout handling is explicit rather than assumed: the kernels address q/k/v/dO
(and the outputs) as `(..., rows, heads, dim)` with `dim` contiguous, `head
stride == dim` and `row stride == heads*dim`. Anything else is staged through a
contiguous copy, which is checked against the fp32 reference for strided inputs,
a strided `out`, a strided `lse` and strided output buffers.

The kernels use no atomics, so results are bit-reproducible: five repeats of the
same input give bitwise-identical dq/dk/dv (checked in `bench/random_check.py`).

Beyond the three dataset workloads, `bench/random_check.py` checks 74 arbitrary
shape/data combinations against the fp32 reference: batch 1-4, seq_len in
{1, 2, 3, 7, 15, 16, 17, 31, 33, 63, 64, 65, 97, 127, 128, 129, 200, 255, 256,
333, 512, 513, 1000, 1024, 2047, 2048, 5000} (deliberately including 1, every
block boundary +-1, and a length longer than any dataset workload so the dS
workspace has to grow), `num_kv_heads` in {1, 2, 4, 8, 16}, input amplitudes
0.5/3.0/10.0 (the last drives logits into the thousands and gradients to ~4500,
exercising the exp2 range), and `sm_scale` in {1/sqrt(128), 1.0, 0.5, 0.01}
since the scale is an input the kernels fold into `lse2`, not a constant. Varlen
adds ragged batches: single-token sequences, thirteen 1-token sequences,
**zero-length** sequences, an 8192-token sequence, and a `max_seqlen`
deliberately looser than the true maximum. All cases pass; 0 failures, and all
outputs finite.

The shapes closest to the benchmark workloads were stressed separately with
fresh data (`bench/stress_big.py`: N up to 4096 dense, the varlen workload's own
sequence mixes): 12 cases, 0 failures, mine/ref error ratio 0.95-1.26 — on one
of them this kernel is *more* accurate than the library.

Degenerate inputs are short-circuited rather than left to launch a zero-sized
grid: `seq_len == 0`, `max_seqlen == 0`, an all-zero `cu_seqlens` and an empty
token array all return the output buffers untouched (`bench/edge_check.py`).

`bench/stress.py` then scores randomised cases the way the recipe actually
prescribes — against the *reference library's own* error on the same data:

```
STRESS: 280 cases, 0 FAILURES, worst mine/ref ratio 1.68 (bar is 7x)
STRESS: 598 cases, 0 FAILURES, worst mine/ref ratio 1.56 (bar is 7x)   # seed 777
STRESS: 949 cases, 0 FAILURES, worst mine/ref ratio 1.87 (bar is 7x)   # seed 31337
STRESS: 799 cases, 0 FAILURES, worst mine/ref ratio 1.65 (bar is 7x)   # seed 424242
STRESS: 547 cases, 0 FAILURES, worst mine/ref ratio 1.65 (bar is 7x)   # seed 5150
STRESS: 559 cases, 0 FAILURES, worst mine/ref ratio 1.59 (bar is 7x)   # seed 777/420
STRESS: 400 cases, 0 FAILURES, worst mine/ref ratio 1.58 (bar is 7x)   # seed 31337/300
STRESS: 679 cases, 0 FAILURES, worst mine/ref ratio 2.35 (bar is 7x)   # seed 20261023
STRESS: 599 cases, 0 FAILURES, worst mine/ref ratio 1.81 (bar is 7x)   # seed 11
STRESS: 600 cases, 0 FAILURES, worst mine/ref ratio 1.38 (bar is 7x)   # seed 2222
STRESS: 598 cases, 0 FAILURES, worst mine/ref ratio 1.56 (bar is 7x)   # seed 333333
STRESS: 600 cases, 0 FAILURES, worst mine/ref ratio 1.50 (bar is 7x)   # seed 44444444
... eight further seeds (5, 55, 555, 5555, 55555, 555555, 7777777,
    88888888), 668-670 cases each, 0 FAILURES, worst ratios 1.55-2.06
... ten further seeds (101 ... 1010), 668-670 cases each, 0 FAILURES,
    worst ratios 1.63-2.32
... three further seeds (1111, 2121, 3131), 669-670 cases each, 0 FAILURES,
    worst ratios 1.64-1.89
... five further seeds (4141, 5151, 6161, 7171, 8181), 669-670 cases each,
    0 FAILURES, worst ratios 1.43-1.89
```

**24610 randomised cases in total, no failures**, and the worst case anywhere is
2.35x the reference library's own error against the same fp32 gradients.

Shapes, batch, `num_kv_heads`, amplitude (0.25-6.0) and `sm_scale`
(1/sqrt(128), 1.0, 0.5, 0.125, 0.01) are drawn at random from a fixed seed, and
each case computes the exact fp32 gradients, flash-attn-4's error against them
and mine. This matters because an *absolute* bound is the wrong instrument on
adversarial data: with a large amplitude and a large `sm_scale` the logits reach
several hundred, the softmax becomes one-hot, and `dq = sum_k P_k(dP_k - delta)
K_k` turns into a difference of two large nearly-equal terms. Then |dq| is small
while the intermediates are huge, so *every* bf16 implementation has a large
error relative to |dq| — and on exactly those cases mine is **bit-for-bit as
accurate as flash-attn-4** (`bench/parity.py`, ratio 1.00 on eight peaked
configurations, and never worse than cuDNN).

The official FIB evaluation passes on all three workloads:

```
flashinfer-bench run --local /root/fib-dataset --definitions attention_bwd_causal_h16_d128 \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 --atol 0.14 --rtol 0.01
  -> attn_bwd_causal_h16_d128_opus5  workload 5d2f3c61...: PASSED
  -> attn_bwd_causal_h16_d128_opus5  workload 8c41e9b2...: PASSED

flashinfer-bench run --local /root/fib-dataset --definitions attention_bwd_varlen_causal_h16_d128 \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 --atol 0.11 --rtol 0.01
  -> attn_bwd_varlen_causal_h16_d128_opus5  workload e3a7b5d9...: PASSED
```

Packaged solutions live in
`/root/fib-dataset/solutions/opus5-scratch/attention_bwd/<definition>/`
(`scripts/package.py` builds them from `solution/`).

### One edge the workloads never reach

dS is addressed as `row * DSN + col`, and that product is computed in 32 bits,
which overflows once the key axis times the padded row stride passes 2^31 -- a
sequence of about 46k, whose dS workspace is ~70 GB and so is allocatable on
this card. Nothing in the dataset is within a factor of ten of it, but "correct
for arbitrary input" includes that shape, so the three kernels that index dS
select a 64-bit row offset from the shape alone (`_wide`, a compile-time
constexpr; every workload here keeps the 32-bit form, so it costs nothing).
`bench/wide_check.py` forces the wide form on at ordinary shapes and requires
bit-identical output, and checks that the selection fires where it should.

And then the shape itself gets run, because the card is big enough:
`bench/huge.py` goes up to **B=1, N=46592, 16 heads -- a 64.7 GB dS workspace,
the selector returning 1** -- and compares against flash-attn-4 on the same
data, since no fp32 reference is affordable there (the forward for it is
computed in fp32 in the harness, chunked). Agreement is 0.56% relative, the
same as at every smaller size:

| shape | dS workspace | 64-bit path | max rel. diff vs flash-attn-4 |
|---|---|---|---|
| N=8192, 16 kv heads | 2.0 GB | no | 0.52% |
| N=16384, 16 kv heads | 8.0 GB | no | 0.65% |
| N=24576, 8 kv heads | 18.0 GB | no | 0.61% |
| N=46592, 16 kv heads | 64.7 GB | **yes** | 0.56% |

Those are 11x the longest sequence in the dataset, and the key axis is 5.7x
longer than anything the workloads reach.

## The implementation

Per (batch, head) the backward is

```
S  = Q K^T * sm_scale   (causal)       P  = exp(S - lse)
dP = dO V^T                            delta = rowsum(dO * O)
dS = P * (dP - delta)
dQ = dS K * sm_scale   dK = dS^T Q * sm_scale   dV = P^T dO
```

Kernels:

0. **why Gluon for the dK/dV pass.** It carries 4 of the 5 GEMMs and 80% of the
   runtime, and in plain Triton it was stuck: 8 warps on one CTA per SM, with
   the softmax backward sitting in the dependency chain between the S^T/dP^T MMA
   and the dV/dK MMA of the same tile, so the tensor cores idled through it.
   Triton cannot widen that — `num_warps=16` compiles but falls off the tcgen05
   path (175 ms instead of 2.5 ms) and `warp_specialize=True` wants 576
   tensor-memory columns against a 512-column budget. Gluon exposes the
   machinery directly, so the pass is laid out by hand:

   * **four partitions**: one TMA warp, one MMA warp, and an 8-warp consumer
     group for the softmax backward. The consumer group is what plain Triton
     cannot have.
   * **tensor memory, exactly 512 columns**: S^T and dP^T double-buffered
     (64 each) plus the dK and dV accumulators (128 each). The MMA warp issues
     tile it+1's S^T/dP^T into the other slot before waiting for tile it's
     epilogue.
   * **dS^T leaves as a TMA store** out of the same shared-memory tile the dK
     MMA already needs, instead of a store from registers — the tensor-memory
     register layout gives each thread 32 bytes of a row, i.e. half a sector.
     Worth 500 us on MHA.
   * **lse/delta come through shared memory**, written by the TMA warp: read
     from global inside the consumer they sat on its critical path, and the
     vector-to-broadcast layout change cost a shuffle per warp. Worth 110-190 us.
   * **the (group, query tile) index is carried**, not recomputed: `it // nmbt`
     is a runtime integer division on every iteration of three partitions.

   Measured against the Triton version of the same pass: MHA 1995 -> 1195 us,
   GQA 1803 -> 1150 us, varlen 2324 -> 1404 us. Run in the same tile order it is
   **bit-identical** to the Triton pass -- `bench/bitid.py` compares them with
   `torch.equal` and dq, dk and dv are all exactly equal on all three workloads,
   which is how it was developed. The shipped configuration picks the descending
   tile order below, which sums the same terms in a different order: dq stays
   bit-identical (dS is unchanged) and dk/dv move by at most one bf16 ulp
   (0.016 on MHA, 0.063 on GQA).

1. **preprocess** — `delta = rowsum(dO*O)` and `lse2 = lse*log2(e) - log2(sm_scale)`.
   Folding `-log2(sm_scale)` into `lse2` makes `exp2` produce
   `Ptilde = sm_scale * P` directly, so `dS' = Ptilde*(dP - delta)` already *is*
   the `sm_scale * dS` that both dQ and dK want — one multiply per S element
   saved — and only the `[BN,D]` dV tile needs a final `1/sm_scale` rescale.
2. **dK/dV pass** (Gluon) — one CTA per (batch, kv head, key tile), looping
   every query head that shares the kv head and every query tile at or below the
   diagonal. It computes `S^T = K Q^T` and `dP^T = V dO^T`; that transposed
   orientation keeps `dV += P^T dO` and `dK += dS^T Q` as natural,
   transpose-free MMAs (the operand descriptor's `permute` supplies the
   "transposed" operand for free). Q/dO/K/V come in through **device-side TMA
   descriptors**, one per (batch) — or per (sequence, head) in varlen — so row
   tails clamp and zero-fill instead of bleeding into the next sequence; the
   measured cost of creating five of them per CTA is 13.6 us across the whole
   4096-CTA grid, i.e. nothing. The causal mask is applied only on tiles that
   straddle the diagonal.
3. **dQ pass** (Triton) — one CTA per (batch, query head, query tile); a single
   GEMM over the staged dS. It moves 2.48 GB in 422 us (5.9 TB/s) against
   6.95 TB/s for a read-dominated sequential stream on this card
   (`bench/readpeak.py`), and two attempts to close that 15% both failed: a
   Gluon rewrite with TMA-fed loads and a tensor-memory accumulator is
   bit-identical but *slower* (431 us at its best of six configurations,
   `bench/gldq.py`), and a blocked dS layout — which would turn its 256-byte
   strided rows into 16 KB contiguous blocks — is worth only 4% on a direct
   micro-benchmark of the two access patterns (`bench/dslayout.py`), nowhere
   near enough to justify restructuring both passes' addressing and masking.

Three structural choices carry most of the speed:

**5 GEMMs instead of 7.** The textbook two-pass backward recomputes S and dP in
the dQ pass, which costs 7 GEMMs. Instead the dK/dV pass *stages `dS^T` into
HBM* in bf16 — exactly the precision the MMA would have consumed, so the result
is bit-identical to recomputing it — and the dQ pass becomes the single GEMM
`dQ = dS K`. That is the theoretical minimum GEMM count for this backward, at
the cost of streaming ~2.1 GB of dS once:

| MHA, per kernel | 7-GEMM schedule | 5-GEMM, Triton | 5-GEMM, Gluon (shipped) |
|---|---|---|---|
| preprocess | 61 us | 43 us | 43 us |
| dK/dV | 1638 us | 1995 us (incl. dS stores) | 1182 us |
| dQ | 1307 us | 422 us | 418 us |
| **sum of kernels** | **3006 us** | **2460 us** | **1643 us** |

The other two workloads, same measurement: GQA 1142 + 416 + 43 = 1601 us,
varlen 1386 + 512 + 46 = 1944 us.

(Per-kernel figures are profiler device times; the headline 1733 us is the
CUPTI span of the whole call, which also covers the gaps between the three
launches.)

**Never mask the contiguous axis of the dS traffic.** The dS tile is
`[key, query]` with query contiguous. Masking that axis (the obvious thing to
do, since the last query tile can overhang the sequence) makes Triton fall back
to *scalar* stores: the staging cost went from 0.4 ms to 3.1 ms on the varlen
workload. Padding the dS row stride to 256 makes the overhang harmless, so only
the key axis is masked — which restores full-width vectorised stores and loads
and is what finally made the 5-GEMM schedule win on varlen too (3.3 ms -> 3.0 ms).
The shipped Gluon pass goes one better and stores dS with TMA, which has no
masking problem at all because the descriptor clamps; the key axis instead gets
padded per head (`NPAD`) so that a tail tile's overflow lands in that head's own
padding rather than in the next head's rows. Varlen needs no padding because
its descriptor is based at the sequence's own first row, so clamping drops the
overflow. The remaining care: if the dQ pass used a query tile wider than the
dK/dV pass' key tile it would read dS blocks that pass never wrote; everything
it could not have written is above the diagonal where dS is zero, so a causal
`tl.where` on exactly those blocks (compile-time selected, and inactive in the
shipped 128/128 configuration) covers it.

**Pick the CTA order and the loop direction separately.** Two independent knobs
decide how the dK/dV CTAs meet each other in L2, and the combination matters
more than either alone:

* *Grid order* — whether the key-tile index varies fastest (`GX=1`, so the CTAs
  resident together are the key tiles of one (batch, kv head)) or slowest
  (`GX=0`, so the longest CTAs are created first and the final wave holds only
  the cheapest ones).
* *Loop direction* — whether each CTA walks its query tiles upward from the
  diagonal (`DESC=0`) or downward from the end (`DESC=1`).

The full 2x2, whole-pipeline microseconds:

| | GX=0 DESC=0 | GX=0 DESC=1 | **GX=1 DESC=0** | GX=1 DESC=1 |
|---|---|---|---|---|
| MHA | 2124 | 2174 | **1750** | 2003 |
| GQA | **1692** | 1695 | 1834 | 1842 |
| varlen | 2502 | 2519 | **2045** | 2387 |

Key-tile-fastest *with ascending* query tiles is worth 13-14% on MHA and varlen
over the next best setting, and it is not the combination intuition suggests.
Walking downward makes all 32 CTAs of a (batch, head) start at the *same* query
tile and march in lockstep — maximal reuse, but also maximal contention: they
all miss on the same line set at the same instant and queue behind one fetch.
Ascending staggers them (the CTA owning key tile n starts at query tile 2n), so
at any moment they are spread across 32 different Q/dO tiles, and reuse still
happens a couple of steps later when a trailing CTA reaches a tile a leading one
already pulled in. On GQA's 512-CTA grid neither matters much and the
longest-first grid order wins instead, because there the tail dominates.
`_order` picks by CTA count — shape only, never data.

Three more order knobs were added and swept the same way, with interleaved A/B
runs so the ~2% run-to-run noise could not pick the winner: the dQ pass' grid
order (query tile fastest wins, 1747 vs 1839 us on MHA) and its query-tile and
key-tile directions (both indifferent — it is DRAM-bound, so the order of its
tiny K tiles does not matter), and for GQA the nesting of the group loop inside
the query loop (group *outermost* wins by 11%: the other way round the dS writes
jump between head slabs every iteration).

The dQ pass keeps its own order - tile index fastest, high to low, so the CTAs
resident together are adjacent query tiles of one (batch, head) that all sweep
the key tiles from 0 upwards and share K through L2; giving it the
longest-first order instead costs it 23%.

**GQA: walk the group inside the CTA.** One CTA owns a (batch, kv head, key
tile) and loops over all `GROUP` query heads that share the kv head, so K and V
are loaded once and stay in shared memory for the whole group, and the TMA/MMA
pipeline is filled once for `GROUP x` as much work. That is why GQA is the
*fastest* of the three workloads at 1663 us despite having exactly the
same FLOP count as MHA: 1160 us in the dK/dV pass against MHA's 1463.

Those two numbers used to differ by 300 us, which looked like MHA's per-CTA cost
(K/V load, pipeline fill, drain and dK/dV write-out, spread over loops that
average only 33 iterations under the causal mask). Most of it was the CTA order
instead: with the ordering fixed they are 1150 and 1195 us, within 4%. A
persistent variant that walks several work items per CTA (`bench/glp.py`,
amortising tensor-memory allocation and pipeline fill over ~5 items each) is
**slower**, re-measured in isolation against the shipped pass on an idle card
(`bench/pers.py`): 1.09x at 296 CTAs and 1.14x at 148 on MHA, 1.09x and 1.34x on
GQA. What it was meant to recover turns out to be small anyway:
`bench/tmemalloc.py` prices tensor-memory allocation at **tens of nanoseconds
per CTA**, and four 128-column allocations cost no more than two (an earlier
note in this report had it at ~1.7 us per CTA, i.e. ~47 us over MHA's 4096 CTAs,
which is wrong). What a persistent kernel would actually amortise is the
pipeline fill -- three TMA buffers and the K/V load, on CTAs that average 16.5
iterations -- and it does not come back, because fixing the CTA count also fixes
the order, and the hardware's own scheduling of 4096 short CTAs beats a
hand-written walk over the same work (the same effect the CTA-order sweep found,
from the other side).

The Triton fallback path takes the other route for GQA — one CTA per query head
with the dK/dV partials summed by a reduction kernel (`GSPLIT`) — because its
grid is otherwise only `32 x 8 x 2 = 512` CTAs and the tail dominates. There
`GSPLIT` is derived from the shape: the longest CTA does `GROUP/GSPLIT * N/BM1`
iterations and GSPLIT is the smallest power of two that keeps that under the
average per-SM load.

Block sizes/warps/stages were swept per kernel and then re-checked with
interleaved A/B runs (`bench/ab.py`, `bench/abv2.py`, `bench/tune.py`) so
thermal drift could not pick the winner. The shipped configuration is dK/dV
BM=64, BN=128, 8 consumer warps + 1 MMA + 1 TMA warp, 3 Q/dO buffers and 2
tensor-memory slots; dQ BM=128, BN=64, 4 warps, 3 stages. Every knob was swept
one at a time around that point and every other change is slower
(`bench/tune.py`, and `bench/finalsweep.py` re-prices all of them against the
final structure). Only the consumer warp count moved, and it moved for real:
re-measured on a settled card with seven interleaved rounds (`bench/nwab.py`),
8 consumer warps beat 16 by **1.7/1.2/1.9%** on MHA/GQA/varlen. Fewer warps at
the same work means less skew at the two intra-partition barriers per iteration,
which is where Nsight says the consumer stalls. The dQ pass' warp count is a
different story: 4 looked 0.3-0.5% better in the first sweeps but is level with
8 on a settled card (0.9858 against 0.9853 combined), so that one is noise and
either value would do.
In the Triton kernels `warp_specialize=True` is a loss everywhere — catastrophically
so for varlen, where a run-time loop bound turns the specialised dQ loop into
24 ms; `num_ctas=2` (2-SM MMA via clusters) fails to compile in Triton 3.7.1
(`TritonGPUPlanCTAPass` crash), `loop_unroll_factor > 1` miscompiles, and
capping registers with `maxnreg` only causes spills.

Staging dS in fp8 would cut the ~0.7 ms of dS traffic (260 us of stores plus the
dQ pass' 420 us of reads) roughly in half, and `bench/dsfp8.py` prices it: with
a per-tile scale, e4m3 (and e5m2) push the dq error from 0.0042 to 0.080 on a
single head - about 19x worse, and close enough to the harness' 0.14 atol that
other data could cross it. Not worth ~17% of runtime, so dS stays bf16, which
is bit-identical to recomputing it.

Things that were measured and rejected: cache modifiers (`.cs`/`.cg`/`.wt`) on
the dS traffic (no effect - it is not L2 pollution); a TMA (bulk) store for dS
*in the Triton kernel*, where it was ~3% slower than plain vectorised stores
because Triton has to stage the tile into shared memory just for the store —
the same TMA store is a 500 us *win* in the Gluon pass, where the tile is
already in shared memory for the dK MMA; a fully blocked dS layout, after a
micro-benchmark showed a contiguous read of the same volume is no faster than
the strided one; an "S orientation" dK/dV pass with BM=BN=128, which makes
all four MMAs 128x128 and reduces lse/delta to one scalar per thread but cannot
afford a second Q/dO pipeline stage in shared memory and ends up ~30% slower;
explicitly software-pipelining the S^T/dP^T dots one iteration ahead of the
dV/dK accumulation (Triton's own scheduler does better: 2749 us vs 2538 us);
and accumulating dK/dV with `acc += tl.dot(...)` instead of the three-operand
form, hoping Triton would keep them out of tensor memory and let two CTAs share
an SM (it does not, and it is ~2% slower).

Running the dQ pass concurrently with dK/dV on a second stream cannot help
either: the two kernels together need more than 512 tensor-memory columns, so
they can never be resident on the same SM. Measured (`bench/stream.py`): splitting
the batch across two streams gives 2530 us against 2539 us on one stream, i.e.
nothing.

The dS scratch is keyed by shape and reused across calls — it depends only on
the problem shape and the device, never on the input values, the same way the
JIT-compiled kernels are reused.

## Where the time goes (and why this is still behind the libraries)

Calibration on this card, under the same cold-L2 benchmark methodology
(`bench/peak.py`, `cuda/mmabench.cu`, `cuda/atombench.cu`):

* The dQ pass is at the copy rate and has nothing left in it. Its DRAM traffic
  is the 2.16 GB of dS below the diagonal plus the 0.27 GB of dQ written and
  K re-read through L2 — about 2.55 GB in the measured 418 us, i.e.
  **6.1 TB/s**, against ~6.1-6.4 TB/s for a pure copy. The 275 GFLOP it also
  does are free at that rate; the only way to speed this pass up is to give it
  less dS to read, and fp8 dS costs 19x the error (below).
* cuBLAS bf16 GEMM peak: **~1700 TFLOP/s**. The GPU is power-limited to 1000 W;
  a *sustained* GEMM throttles to 1320 MHz / ~1190 TFLOP/s, but the benchmark's
  L2 flush between iterations keeps clocks near boost, so ~1700 is the right
  ceiling for these measurements.
* `mma.sync` (legacy tensor-core path): only **556 TFLOP/s** — 1/3 of tcgen05.
  Any hand-written kernel has to use tcgen05 to be competitive.
* `ex2.approx.f32`: 16/SM/cycle -> 4.6 Top/s; fp32 FFMA: 118/SM/cycle.
* fp32 `atomicAdd` / `red.global.add`: **1.3-1.4 TB/s**, against 6.1 TB/s for
  plain stores. This is what rules out the "one fused kernel, dQ accumulated
  with atomics" shape: at BN=128 it needs ~4.2 GB of fp32 atomics, i.e. ~3 ms —
  which is why dS is *staged* rather than atomically reduced.

The causal backward needs 5 GEMMs of N x N x D with causal halving =
**1374 GFLOP** for the dense workloads, so ~808 us at 1700 TFLOP/s. The
libraries land at 1320-1442 us, i.e. ~55-60% of achievable GEMM throughput. My
dK/dV pass carries 4 of those 5 GEMMs (1100 GFLOP) in 1182 us = **931
TFLOP/s**, ~55% of that ceiling.

The softmax backward itself is no longer in the way. Ablating the shipped Gluon
pass, controlled (whole-pipeline times, the shipped kernel timed alongside and
tracking to within 15 us -- `bench/mo2.py`):

| MHA pipeline | MHA | GQA |
|---|---|---|
| as shipped | 1699 us | 1669 us |
| staging stores and the dS TMA store kept, drain and arithmetic dropped | 1625 us | 1616 us |
| consumer handing its slot straight back: no stores, no dS traffic | 1504 us | 1357 us |

The softmax backward's own work -- drain, exponential, mask, multiply -- is 74 us
on MHA and 54 on GQA, i.e. nearly but not entirely hidden behind the tensor
cores; the larger part of the gap is the dS staging store and the 2.1 GB the dQ
pass reads back. The same ablation on the Triton version of this pass showed the
epilogue costing a strictly additive ~600 us; that is what the rewrite bought.

What sets the floor is **how many operand bytes each MMA shape needs per FLOP**,
and that can be pinned down exactly. `bench/mmarate.py` times a bare loop of
tcgen05 MMAs -- operands fixed in shared memory, one tensor-memory accumulator,
no TMA, no epilogue -- and the achieved rate is a pure function of that ratio:

| MMA shape (M,N,K) | operand bytes | TFLOP/s | % of 2250 | implied B/cycle/SM |
|---|---|---|---|---|
| 128x128x256 | 128 KB | 1982 | 88.1% | 100 |
| 128x256x128 | 96 KB | 1946 | 86.5% | 99 |
| 128x128x128 | 64 KB | 1830 | 81.3% | 98 |
| 128x128x64 | 32 KB | 1617 | 71.9% | 87 |
| **128x64x128** | 48 KB | **1238** | **55.0%** | 97 |

The last column is the same number every time: ~95-100 bytes per cycle per SM of
operand fetch. The rate is not about M, N or K individually, it is `B * FLOPs /
operand_bytes`, and two of this kernel's four MMAs are the worst shape in the
table (N=64, the query tile) with the other two at K=64. That is what fixes the
pass' MMA floor at 782 us (and it is why the shapes, not the schedule, are the
thing one would want to change -- except that tensor memory forbids it, below).

What this does *not* explain is the distance from that floor to the measured
1182 us. The arithmetic below says 240 KB per iteration at ~95 B/cycle is
~1174 us, which looks like a perfect match and is not one: see the next
paragraph.

That bandwidth is **per SM, not per CTA**, so the usual escape of running more
CTAs does not exist here — the same probe at 2 and 4 CTAs per SM gets *less*
aggregate throughput, not more (128^3: 80.9% -> 68.5% -> 62.7%), because they
share one operand port and interleave badly.

Per query tile the four MMAs fetch 160 KB, the TMA writes 32 KB of Q/dO in, the
consumer writes 32 KB of staging and the dS store reads 16 KB back out: 240 KB
through the same port, which at ~95 B/cycle is ~2530 cycles, i.e. ~1174 us for
the whole pass -- against 1182-1195 us measured. **That agreement is a
coincidence, and an earlier draft of this report believed it.** Routing P^T and
dS^T through tensor memory removes 48 KB of those 240 KB and the kernel does not
move by a microsecond (see "Where the dK/dV pass' time goes"), so the port is
not what this loop is waiting on. What is true in this paragraph is the rest of
it: the tensor-core math is only about half the cycles (Nsight: 47% tensor-pipe
utilisation, 2.97 GB of DRAM traffic against the 6.77 GB the Triton pass moved),
removing the softmax-backward math changes nothing, and replacing the lse/delta
broadcasts with constants is *slower*.

The fix would be the shape at the top of that table, and **both** on-chip
budgets forbid it at once. The kernel already sits at 100% of each:

| resource | used | available |
|---|---|---|
| tensor memory | 512 columns (dK 128 + dV 128 + 2x(S^T 64 + dP^T 64)) | 512 |
| shared memory | 231.5 KB (K,V 64 + Q,dO 3x32 + staging 2x32 + lse/delta 1.5) | 232.4 KB |

Going to BM=128 doubles the Q/dO tiles. It can be made to fit -- one staging
buffer taking P^T and dS^T in turn instead of two, 231.4 KB with two Q/dO
buffers, and a [128,128] S^T plus dP^T then exactly fills tensor memory
alongside the dK and dV accumulators -- and the result is 17% slower, for a
reason that is about pipelining rather than capacity: **two Q/dO buffers are not
enough at any tile size** (4800 us against 1740), because the producer's refill
then waits on `bEmpty`, which the dK MMA signals, so TMA latency chains into the
MMA stream. Three 128-row buffers would need 290 KB. The same wall blocks the
cheaper half-measure of batching only the dV/dK MMAs to K=128 (worth
71.9% -> 81.3% on those two): their right-hand operand has to be two query
tiles' dO contiguous in shared memory, i.e. BM=128 tiles again. Nor can the
operands move out of shared memory: `tcgen05_mma` takes its left-hand side
there, and the "S orientation" that would put P and dS in tensor memory as
right-hand operands needs 640-768 columns.

So the ceiling for this schedule on one CTA per SM is set by the tensor cores at
the shapes it can afford, and both budgets that set those shapes are exactly
full.  What that costs, and what is actually left on the table, is measured in
"Where the dK/dV pass' time goes" below.

The obvious way out of a *per-CTA* budget is a **2-CTA (cta_group::2) MMA**,
which is how CUTLASS's Blackwell GEMMs reach peak, and Gluon does expose it
(`TensorMemoryLayout(two_ctas=True)`, CGA shared layouts, multicast TMA,
cluster mbarriers). Its semantics, reverse-engineered from the compiler's own
checks in `bench/mma2cta.py`, are exactly what this kernel would want: the
*result* is split along N across the pair (so each CTA still spends only half
the tensor-memory columns), the right-hand operand spans the full 2N, the
left-hand operand is shared, and the TMA can multicast one global read to both
CTAs. On paper that turns the N=64 MMA into an N=128 one at unchanged per-CTA
tensor memory.

In practice it does not pay, and it took three attempts to say that honestly.

The probe's result was wrong at first (relative error 1.0), so its rate meant
nothing -- an earlier draft of this report quoted it as evidence against
clusters, which it was not. Two diagnostics say what is actually happening.

Dumping the tile instead of scoring it (`bench/mma2cta_diag.py`): the **first N
columns are exact, to the bit**, and the rest of the output buffer is untouched
zeros. Then giving each CTA its own output slab, selected by `program_id(0) % 2`
(`bench/ctarank2.py`): **both slabs come back written, and both contain the same
thing** -- the same first N columns correct, the same zeros after them. So
`program_id` does distinguish the two CTAs of a cluster (an earlier draft of this
paragraph claimed no rank was reachable, which is wrong), but the pair is not
producing a 2N-wide result: each CTA reads back one CTA's worth of columns.

That also explains the rate. A pipelined loop of these MMAs measures **470-516
TFLOP/s** (the store is outside the timed loop, so the figure is the MMA work)
against 1249 for single-CTA 128x64x128 -- roughly half, which is what one CTA's
worth of work spread across two SMs would cost.

The obvious next guess, that the accumulator should be allocated at the per-CTA
tile while the layout's block stays cluster-wide, is refused outright: at
NC=64 the verifier says *"the allocation shape must be at least 128x128"*, and
at NC=128 *"at least 128x256"* -- i.e. the allocation has to match the
cluster-wide block, which is what the probe already does. So whatever this
configuration builds, it is not the 2x-wide MMA the mode exists for, and the
missing piece of the contract was not recoverable from the compiler's assertions
alone.

This matters because the orientation would otherwise be ideal. Computing S in
the untransposed orientation makes the 2-CTA split fall along the *key* axis, so
each CTA keeps its own kv block and its own dK/dV accumulators -- no cross-CTA
reduction -- while all four MMAs become 128x256x128, which would put the pass'
MMA floor at 564 us instead of 782. That is the largest single piece of headroom
this report can point at, and it is out of reach in this toolchain rather than
on the hardware.

Several constraints had to be discovered by reading compiler assertions -- CGA bases are in tile units, every shared layout in a
`num_ctas=2` kernel must declare one, `blockN` must equal the cluster-wide N
(`numRepN == 1`), and the cluster mbarriers need
`fence_init_release_cluster` + a relaxed cluster arrive before first use. So a cluster
rewrite would have had to start by reverse-engineering the rest of that
contract, on a mode whose only measurable behaviour so far is half the
single-CTA rate -- the wrong bet for the time left, and the one piece of
headroom this report cannot close.

### Where the dK/dV pass' time goes

(The diagnostics in this section were run at the 16-warp consumer configuration
that was shipped at the time, where the pass measured 1195 us; it is 1182 us at
the 8-warp configuration shipped now. The structure they establish is the same
-- all of them are A/B comparisons against that same baseline.)

Three models of this loop turned out to be wrong, each refuted by the
measurement it predicted. They are worth recording because they are the obvious
ones.

**It is not the shared-memory port.** The loop pushes 240 KB per iteration
through shared memory: 160 KB of MMA operands, 32 KB of TMA fill, 32 KB of
staging stores, 16 KB read back by the dS store. `tcgen05_mma` takes its *left*
operand from either shared or tensor memory (the docstring says the right one;
the verifier says otherwise, and `bench/tmemb.py` confirms the left one is exact
at `col_stride=1`), and P^T and dS^T are exactly those operands. Writing them
to tensor memory instead of to a staging buffer removes 48 KB of the 240 KB --
both the store and the MMA's read of it -- and the kernel does not move:
**1732.7 us against 1734.0 us** on MHA, bit-identical output. A fifth of the
traffic, for nothing.

**It is not the tensor-memory drain either.** `bench/tmemld.py` times the
consumer's two fp32 accumulator reads on their own: **430 B/cycle/SM**, 75 us
for the whole 55 MB per CTA, about 6% of the kernel.

**It is not tile shape.** `bench/mmapeak.py` measures the single-CTA tcgen05
ceiling with the operands already resident, which is the most favourable case a
CTA can get:

| shape | TFLOP/s | % of 2.25 PFLOP/s |
|---|---|---|
| 128x64x128 (S^T, dP^T) | 1249 | 55.5% |
| 128x128x64 (dV, dK) | 1608 | 71.5% |
| 128x128x128 | 1802 | 80.1% |
| 128x256x128 | 1951 | 86.7% |

This loop's mix -- two of the first shape and two of the second per iteration --
puts its MMA floor at **782 us** against 1182 us actual; all-128^3 would put it
at 610 us. So the shapes do leave 31-45% on the table and the operand arithmetic
that favours BM=128 is real -- it is just not what the kernel loses time to. A
128-row query tile now fits (one staging buffer taking P^T and dS^T in turn,
plus a barrier that releases both accumulator tiles as soon as the consumer has
them in registers) and it is **17% slower**: 2034 against 1733 us end to end,
1822 us even with one of its two handshakes removed.

**What it is: mostly the dS store, which the algorithm requires.** A three-rung
ablation out of one diagnostic build (`bench/attn_bwd_mo.py`, driven by
`bench/mo2.py`, with the shipped kernel timed alongside as a control, tracking
it to within 5-15 us):

| rung | MHA | GQA |
|---|---|---|
| the real thing | 1699 us | 1669 us |
| staging stores and the dS TMA store kept, drain and arithmetic dropped | 1625 us | 1616 us |
| slot handed straight back: no stores, no dS traffic | 1504 us | 1357 us |

So the softmax backward's own work -- the tensor-memory drain, the exponential,
the mask, the multiply -- costs **74 us on MHA and 54 on GQA** of non-hidden
time, and the rest of the gap, **121 us and 258 us**, is the staging store plus
the 2.1 GB of dS that the dQ pass has to read back. That part is not overhead to
be tuned away: it is what the 5-GEMM schedule pays for not having a third GEMM.
An earlier version of this section attributed the whole gap to handshake
latency; the middle rung disproves that.

The arithmetic really is nearly free, which is what the rest of the evidence
says too: `bench/consumer.py` times the phase standalone at **879 cycles** at
the shipped 8 warps (946 at 16) inside a ~1700-cycle MMA shadow, and Nsight
Compute puts the XU pipe (the exponential) at 23% and FMA at 10% while 32% of
the consumer's issue slots stall at barriers.

The pipeline depth is sharper than it looks, too. **NBUF=2 is catastrophic at
any tile size** -- 4800 us against 1740 -- because with only two Q/dO buffers
the producer's refill waits on `bEmpty`, which the dK MMA signals, so TMA
latency chains straight into the MMA stream; three buffers break that chain.
That is also the real reason BM=128 cannot win: 128-row Q and dO buffers are
64 KB apiece, and three of them plus K, V and staging need 290 KB of the 232 KB
available.

### How the Triton version was stuck, and why Gluon was the way out

This is the measurement trail that led to the Gluon rewrite, and it is still why
the pass is laid out the way it is. In the Triton version Nsight Compute named
the wall precisely: 8 warps on one CTA per SM — **2 warps per scheduler** — with
no eligible warp **76%** of cycles, at 32% compute throughput and 13% DRAM. One
CTA per SM is forced three times over: all **512** tensor-memory columns, 164 KB
of shared memory, and 201 registers per thread (51 456 of the SM's 65 536). Any
one of those alone rules out a second CTA, and that is true of every variant of
this algorithm.

The two standard fixes are both out of reach in Triton 3.7.1 proper:

* **More warps per CTA.** `num_warps=16` compiles and is numerically correct but
  runs at 175 ms instead of 2.5 ms (measured across six tile shapes) -- it drops
  off the tcgen05 path entirely.
* **Warp specialisation.** `warp_specialize=True` needs **576** tensor-memory
  columns at BM=64/BN=128 against a 512-column budget, so it only compiles at
  BM=32 (320 + 192 columns), where it is 57% slower than the BM=64 baseline.
  Hand-unrolling the query tile into two independent sub-chains inside one
  iteration -- the same overlap by other means -- is refused for the same reason
  (688 columns: Triton multi-buffers every sub-chain's accumulator).

Gluon is exactly those two fixes made expressible: the warp partition and the
tensor-memory allocation are written by hand, so a wide consumer group and a
512-column budget coexist. Two things that did *not* help, measured in Gluon:

* **Two consumer groups of 8 warps ping-ponging on alternate tiles**, so the
  group the MMA warp waits on is never the one computing: 1692 us against
  1649 us for a single 16-warp group at the same point in development. Same
  total epilogue throughput, more barriers.
* **A second MMA warp**, one issuing S^T/dP^T and one issuing dV/dK so the S^T
  stream can run ahead freely: 1927 us against 1737 us. The extra slot-release
  handshake costs more than the decoupling gains.
* **BM=128** (`bench/gl6.py`), which is the shape the operand arithmetic asks
  for -- 16 bytes of MMA operand per (key, query) pair instead of 20. A
  [128,128] S^T plus dP^T eats all 256 spare tensor-memory columns, so it has
  one slot instead of two and the dV/dK MMAs have to be split into query halves
  sharing one 16 KB staging tile (shared memory has no room for two). It is
  **1644 us against 1203 us** with the dS store taken out of both, i.e. the
  weaker pipelining costs far more than the better operand ratio buys; with the
  store it is 1826 us against 1463 us, because a single staging tile also forces
  a full `store_wait(0)` drain twice per tile. Tile shape is not the limit here,
  the dependency structure is.

Splitting the pass into separate kernels small enough in tensor memory for two
CTAs per SM does not pay either: dV alone (192 columns) runs at 756 us and
dK+dS alone (256 columns) at 1804 us, i.e. 2560 us for the pair against 1902 us
for the fused Triton kernel.

The same ablation on the CUDA version (switches in `cuda/attn_kernels.cuh`,
structurally the same kernel) splits its cost into MMA ~640 us, softmax-backward
math ~530 us, and ~1.6 ms of loop/synchronisation latency — the same starvation
seen from the other side.

### The hand-written tcgen05 path (in `cuda/`, not the shipped solution)

Because Triton tops out around 32%, I also wrote the backward from scratch in
CUDA C++ with inline PTX against Blackwell's 5th-gen tensor cores: `tcgen05.mma`
with shared-memory matrix descriptors, tensor-memory accumulators
(`tcgen05.alloc` / `tcgen05.ld` / `tcgen05.commit`), 5-D TMA loads, mbarriers,
and a warp-specialised producer/consumer variant using named barriers
(`cuda/tc_common.cuh`, `cuda/attn_kernels.cuh`, `cuda/attn_ws.cuh`). It is
numerically correct on all three workloads (errors identical to the Triton path).
Three things were worth learning the hard way:

* For an MN-major ("transposed") UMMA operand the descriptor's LBO and SBO
  fields swap roles relative to the K-major case (LBO = 128,
  SBO = K_extent*16). Found by brute-forcing the encoding in `cuda/t3_sweep.cu`.
* UMMA reads shared memory through the **async proxy**, so ordinary thread
  stores of P^T/dS^T need `fence.proxy.async.shared::cta` before the MMA;
  `__syncthreads()` alone silently yields garbage in whichever operand was
  written last.
* tcgen05 confines each warp to the tensor-memory lanes of its own
  sub-partition (`warp % 4`), so a warp-specialised kernel must keep the real
  warp id in the row index, not a renumbered consumer index.

Measured on the MHA workload, its best configuration is dK/dV 3.17 ms + dQ
2.25 ms + preprocess 0.06 ms = **~5.5 ms**, i.e. about 2.2x slower than the
Triton solution, so it is kept as documented exploration rather than shipped.
Nsight Compute blames barrier waits for 36% of its stall cycles, but that turns
out to be a symptom, not the disease -- three attacks on it all failed, each
measured back-to-back against the baseline on the same card:

* **Dropping the two redundant `__syncthreads()`** (the mbarrier waits already
  provide the ordering): 4% *slower*. The CTA barrier keeps the eight warps in
  lockstep, which the tmem/smem access pattern likes.
* **Double-buffering the P^T/dS^T staging buffers**, so the next iteration's
  epilogue no longer waits on the previous dV/dK MMA: 7% slower. It also forces
  NBUF 4 -> 3 (shared memory) and the Q/dO buffers then have to be tracked by
  their own mbarriers, whose wait reintroduces an MMA drain one iteration behind.
* **Hoisting the causal mask** out of the per-element loop (the only one that
  worked): 3.6%, 3.28 -> 3.17 ms. Instruction count is not the constraint --
  the kernel issues at 20% of its slots.

So the hand-written path is latency-bound in the same way the Triton one is, and
only a full FA3-style ping-pong (two consumer warpgroups alternating so one's
epilogue covers the other's MMA) would change that; the warp-specialised
kernels in `cuda/attn_ws.cuh` are an unfinished attempt at it -- they produced
correct results when they were measured but can deadlock in their current state.

### The dK/dV epilogue: badly coalesced, and better left that way

Masking the dK/dV stores off entirely (`bench/mo2.py`, MO=3) costs **68 us on
MHA** and ~9 on GQA -- about 2.5 us per CTA, and MHA runs 4096 of them. The
reason is the same one that moved dS^T to TMA: a tensor-memory load gives each
thread 32 bytes of a *row*, and consecutive lanes hold different rows, so a
store from registers leaves in 32-byte pieces.

Routing dK/dV out through TMA instead is **1.0-1.6% slower** (`bench/epi.py`).
The dS^T staging buffer is free by then and is exactly two [BN, BM] slots, so a
[BN, D] tile fits as two halves with no extra shared memory -- but four tiles
through two slots needs two full `store_wait` drains, plus the shared-memory
round trip and its fences, and that costs more than the coalescing it buys.
dS^T is the opposite case on both counts: eight times the volume, and its
staging buffer was already there because the dK MMA reads it.

It is also **wrong**, which took a 6700-case stress run to notice and is worth
recording. TMA clamps at the *tensor* end, and for varlen the dK/dV tensor is
the whole token array, not one sequence: a tail tile at the end of a sequence
therefore writes real rows belonging to the *next* sequence, where the register
store it replaced masks them off with `offn2 < N`. Whether that corrupts
anything depends on whether the next sequence's own CTA happens to write those
rows afterwards, so a one-shot bit-identity check passed while 21% of randomised
varlen cases failed. The register-store epilogue has no such problem, because it
masks on the sequence length directly.

So the epilogue stays as register stores, and its 68 us stays on the bill.

### Host-side ordering

The three kernels sum to 1643 us on MHA but the measured span is ~1724: about
38 us of that is genuine inter-kernel gap (the rest is the cold-L2 penalty the
timing protocol requires, which the per-kernel profile does not see). The gap is
host latency, and the preprocess kernel is the reason it matters -- it only runs
for 43 us, so any host work issued after its launch leaves the card idle before
the dK/dV launch lands. One call costs about 100 us of host time (`bench/cpucost.py`;
flash-attn-4 is 58, cuDNN 190), most of it Triton's argument marshalling.

So all of it moves ahead of the first launch: the workspace lookup, the dS
padding arithmetic, the CTA-order choice, both grids, the 64-bit selector. The
three launches then issue back to back. Worth **0.9%** end to end, measured as
the median of three interleaved runs (MHA 1741 -> 1724 us, GQA 1680 -> 1655,
varlen 2000 -> 1996). Nothing is cached between calls -- the work is redone
every call, just earlier in it.

## Verification log

`report/verification.log` is the verbatim output of `scripts/verify.sh` on the
shipped kernel -- it opens with the md5 of `solution/attn_bwd.py` so the log
says what produced it, and covers the arbitrary-shape correctness suite,
determinism, layout robustness, the interleaved timings, and both official
evaluations.

## Files

```
solution/attn_bwd.py      the shipped kernels: dK/dV in Gluon, preprocess and dQ
                          in Triton, plus the earlier all-Triton schedules kept
                          as the fallback and for the comparisons in this report
solution/main_dense.py    FIB entry point (bshd)
solution/main_varlen.py   FIB entry point (thd)
scripts/package.py        builds the FIB solution JSONs into the dataset
scripts/verify.sh         correctness + timings + official evaluation, one shot
scripts/reproduce.sh      every number in this report, with the commands
scripts/make_report.py    regenerates this report from report/final_bench.json
cuda/                     from-scratch tcgen05 / TMA CUDA implementation (exploration)
report/FINAL.md           this file

bench/final_bench.py      the interleaved ref-vs-mine timings in "Results"
bench/breakdown.py        per-kernel device times
bench/random_check.py     74 fixed shape/scale cases vs the fp32 reference
bench/stress.py           randomised cases scored against the library's own error
bench/parity.py           my error vs fa4/cuDNN on peaked-softmax cases
bench/layout_check.py     strided inputs/outputs; bench/edge_check.py degenerate ones
bench/bitid.py            Gluon vs Triton pass, bit-identity check
bench/mmarate.py          tcgen05 MMA issue rate by shape (the 95 B/cycle model)
bench/mmaceil.py          MMA-only floor of the dK/dV loop
bench/floor.py            the same loop with the consumer stubbed out
bench/glsweep.py          every Gluon knob, one at a time; bench/tune.py the Triton ones
bench/gl1-gl6.py          the Gluon development ladder (serial -> ping-pong -> shipped -> BM=128)
bench/mma2cta.py          the 2-CTA (cluster) MMA probe -- returns wrong answers
bench/peak.py             card calibration: GEMM peak, copy bandwidth, exp2, atomics

bench/mmapeak.py          single-CTA tcgen05 ceiling by shape (the 782 us floor)
bench/tmemld.py           tensor-memory drain rate
bench/consumer.py         the consumer phase, component by component, by warp count
bench/cpucost.py          host time per call, mine vs the libraries
bench/attn_bwd_mo.py      diagnostic copy of the kernel with the MO ablation ladder
bench/mo2.py              drives it: what the dS store and the softmax cost in situ
bench/epi.py              the TMA dK/dV epilogue (slower, and wrong for varlen)
bench/pers.py             persistent dK/dV against the shipped pass
bench/tmemb.py            which tcgen05 operand may live in tensor memory
bench/nwab.py             warp counts, interleaved, settled card
bench/bmpab.py            preprocess tile size, settled card
bench/finalsweep.py       every shipped knob re-priced on all three workloads
bench/wide_check.py       the 64-bit dS addressing path
bench/huge.py             shapes up to N=46592 (a 64.7 GB dS workspace)
scripts/sync.sh           push the committed tree to the node and compare md5s
report/verification.log   verbatim verify.sh output for the shipped kernel
```
