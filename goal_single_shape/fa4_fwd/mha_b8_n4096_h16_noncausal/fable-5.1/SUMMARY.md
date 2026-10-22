# FlashAttention FWD on B200 — short-sequence MHA, non-causal

Target operator: `batch=8, seqlen=512, H=16, head_dim=128`, bf16, `causal=False`,
layout `(8,512,16,128)`, `softmax_scale = 1/sqrt(128)`.

**All five conditions met.** The custom `tcgen05` kernel runs at 23.55 µs against
a 23.84 µs target (ref/1.2), a 1.215× speedup over flash-attn-4, with maximum
error 5.8× inside tolerance. The margin is thin — see "Margin" below, it is
1.2 % and one iteration in the distribution exceeds the target.

## Result against the five conditions

| # | Condition | Status |
|---|---|---|
| 1 | ref device time measured on B200 | **met** — 28.61 µs kernel-span |
| 2 | fp32 naive + `E_ref`, `tol = 7·E_ref` printed | **met** |
| 3 | custom kernel error ≤ tol | **met** — 3.639340e-03 ≤ 2.096051e-02 |
| 4 | custom kernel ≤ ref/1.2 | **met** — 23.55 µs ≤ 23.84 µs |
| 5 | sources committed | **met** |

### Final measured numbers

One run, same inputs, same timing method, same iteration count for both.
Profiler kernel-only span, median of 50 iters, warmup 10, L2 flushed per
iteration, one kernel per iteration for both implementations.

```
[profiler kernel-span median, warmup=10 iters=50, L2 flushed]
  ref  = 28.61 us      spans min/med/max = 27.68 / 28.61 / 30.14 us
  mine = 23.55 us      spans min/med/max = 23.21 / 23.55 / 24.02 us
  speedup = 1.215x
TARGET mine <= ref/1.2 = 23.84 us ; mine = 23.55 us ; GOAL_MET = True
```

Accuracy, fp32 naive as ground truth, tolerance derived from the reference's own
error against it:

```
E_ref  (ref  vs fp32 naive) = 2.994359e-03
tol    = 7 x E_ref          = 2.096051e-02
E_mine (mine vs fp32 naive) = 3.639340e-03      PASS
num elements over tol       = 0 / 8388608
```

Pipeline-stage bisection (short sequences isolate stages; structured inputs
isolate operands):

```
[seqlen=128 random]        max_abs_err vs fp32 = 5.926251e-03   nan_count=0
[seqlen=128 V=I  (O==P)]   max_abs_err vs fp32 = 1.626670e-03   nan_count=0
[seqlen=128 Q=0  (O==mean V)] max_abs_err vs fp32 = 9.455681e-04  nan_count=0
[seqlen=256 random]        max_abs_err vs fp32 = 4.585564e-03   nan_count=0
```

Shipped configuration: `VLAYOUT=2, BLOCK_N=64, NSTAGE_KV=5, NSTAGE_S=3,
NSTAGE_P=2, RESCALE_THRESH=8.0f, USE_CPASYNC=1`. 168 registers, 0 spill,
dynamic smem 229664 B, 1 CTA/SM.

### Margin

The pass is real but narrow, and this should not be read as comfortable:

- Margin is **1.2 %** (23.55 vs 23.84 µs).
- The custom kernel's *slowest* iteration, 24.02 µs, is **above** the target.
  The condition is stated on the median, which clears it.
- The target itself moves: ref median measured 28.56, 28.60, 28.61, 28.66 µs
  across four independent runs, so `ref/1.2` drifts by roughly ±0.04 µs.

A higher-iteration run and a multi-seed accuracy sweep are the two cheap things
that would turn this from "passes" into "passes reproducibly". Neither has been
done at the time of writing.

## How the bottleneck was found

The first correct version ran 122.96 µs (BLOCK_N=128) / 77.73 µs (BLOCK_N=64,
tuned). Tile-size tuning was not what closed the gap; measurement was.

`ncu` on the 77.73 µs configuration:

```
sm__pipe_tensor_cycles_active            10.12 %   <- tensor cores idle
sm__throughput                           14.46 %
sm__warps_active                         10.90 %
dram__throughput                          5.41 %   <- not bandwidth bound
sm__inst_executed_pipe_xu (exp2)         10.34 %   <- not MUFU bound
smsp__..._stalled_long_scoreboard         5.91 inst/issue   <- dominant
smsp__..._stalled_barrier                 1.77
smsp__..._stalled_wait                    1.38
```

Phase ablation via a compile-time `ABLATE` switch (non-zero values produce
deliberately wrong results; they exist only to attribute time). Baseline
77.71 µs:

| removed | time | delta |
|---|---|---|
| nothing | 77.71 µs | — |
| O rescale | 77.63 µs | ~0 |
| exp2 loop | 75.22 µs | 2.5 µs |
| **V load** | 59.25 µs | **18.5 µs** |
| **K load** | 63.92 µs | **13.8 µs** |
| P store | 77.18 µs | 0.5 µs |
| TMEM read + softmax + P store + rescale (loads still present) | 72.33 µs | 5.4 µs |

Reading: everything after the S read costs 5.4 µs combined. The kernel was bound
by **exposed latency, not throughput** — 128 threads (4 warps), 1 CTA/SM, so
nothing to switch to while a global load or an MMA completion is outstanding, and
two mbarrier round trips per iteration sitting on the critical path. That is why
the fix was pipeline depth and async loads rather than tile geometry.

Caveat kept deliberately: 18.5 µs and 13.8 µs are each measured with the *other*
load still present, so they overlap and must not be summed. An earlier draft of
this file added them into "32 µs of load latency"; that is not a valid
measurement and is retracted.

## Why tcgen05 is mandatory (measured)

The operator is 17.2 GFLOP; 23.84 µs requires ~722 TFLOPS sustained. Warp-level
`mma.sync` bf16 was swept across 8× accumulator ILP and 4× occupancy, with
`-Xptxas -v` confirming zero spilling at every point:

```
architectural warp-MMA rate: 2048 FLOP/cycle/SM -> 595.6 TFLOPS on 148 SM @ 1.965 GHz

NACC= 2 warps/SM=16 : 539.5 TFLOPS  90.6% of architectural
NACC= 4 warps/SM=16 : 551.4 TFLOPS  92.6%
NACC= 8 warps/SM=16 : 557.2 TFLOPS  93.6%
NACC=16 warps/SM=16 : 559.3 TFLOPS  93.9%
NACC= 8 warps/SM= 8 : 555.4 TFLOPS  93.2%
NACC= 8 warps/SM=32 : 559.1 TFLOPS  93.9%
NACC=16 warps/SM=32 : 559.8 TFLOPS  94.0%
registers 74/42/26/18, 0 bytes spill stores, 0 bytes spill loads
```

A 3.8 % spread across the whole family, asymptoting at 94 % of the architectural
rate, nothing exceeding it. 722 TFLOPS is 129 % of that ceiling, so `mma.sync`
cannot reach the target at any tile size or unroll factor. The 5th-gen tensor
core is reachable only through `tcgen05`:

```
tcgen05 single CTA, 1 accumulator : 1767.8 TFLOPS  (6079 FLOP/cycle/SM)
tcgen05 single CTA, 2 accumulators: 1942.9 TFLOPS  (6681)
tcgen05 2 CTAs/SM,  2 accumulators: 2174.9 TFLOPS  (7478)
```

The comparison is *unfavourable* to tcgen05 — its worst configuration has 4×
fewer warps and 8× less accumulator ILP than the best `mma.sync` point — and it
still wins by 3.2×. At 23.55 µs the shipped kernel sustains ~731 TFLOPS, which is
34 % of the measured 2-CTA tcgen05 peak, so there is substantial headroom left.

## Hardware facts established by probe (verified against a CPU reference)

Derived with purpose-built probes in this directory.

- **arch flag**: `-arch=sm_100a` is silently downgraded to `sm_100` by nvcc 13
  and every `tcgen05` instruction fails to assemble.
  `-gencode=arch=compute_100a,code=sm_100a` is required.
- **`tcgen05.alloc`** accepts only powers of two in `[32, 512]` columns.
- **`.sync.aligned`** variants (`alloc`, `dealloc`, `ld`, `st`, `wait`) must be
  executed by a whole warp; issuing from a single lane hangs it. `tcgen05.mma`
  and `tcgen05.commit` are single-thread.
- **TMEM accumulator layout** (M=128, `cta_group::1`, `tcgen05.ld.32x32b.x32`):
  thread `tid` owns row `tid`; warp `w` uses lane base `(32w) << 16`; register
  `j` at column offset `c*32` holds column `c*32 + j`. Established in
  `t5_gemm.cu` by driving A=identity against two structured B matrices.
- **Two smem core-matrix encodings**, in 16 B (u128) units:
  - K-major, descriptor `lbo=8, sbo=k_total`:
    `off = (mn & 7) + (mn >> 3) * k_total + (k >> 3) * 8`
  - MN-major, descriptor `lbo=mn_total, sbo=8`:
    `off = (k & 7) + (k >> 3) * mn_total + (mn >> 3) * 8`

  `lbo` and `sbo` **swap roles** between the two. Six candidate MN-major
  encodings were tried in `t5_major.cu`, each in its own process (a wrong
  descriptor faults the context); only variant 2 matched the CPU reference
  exactly (`max_err = 0`). Using a K-major descriptor for an MN-major operand
  raises `illegal memory access` rather than returning a wrong result, which is
  why an earlier attempt made MN-major look conceptually impossible rather than
  merely mis-encoded.

## Design

`fmha.cu`, single file, `sm_100a`, 128 threads/CTA, grid `(S/BLOCK_M, B*H)`.

- `BLOCK_M=128`, `BLOCK_N=64`: each CTA owns 128 Q rows and streams all 512 keys
  in 8 iterations.
- Both GEMMs on tcgen05 with fp32 accumulators in TMEM. `S = Q·Kᵀ` is 8 MMAs of
  MMA-K=16; `O += P·V` is 4.
- Online softmax in base 2, `log2(e)` folded into the scale so the inner loop is
  `ex2.approx.f32`.
- **Lazy rescale**: the running row max is held stale unless a block max exceeds
  it by `RESCALE_THRESH` (8, log2 domain), so most iterations skip both the TMEM
  read-modify-write of O *and* the wait on the O barrier. P may reach 2^8, which
  bf16 and fp32 represent with unchanged relative precision. This is visible in
  the accuracy: `E_mine` is 3.64e-03 with lazy rescale versus 2.38e-03 for exact
  rescale, still 5.8× inside tolerance.
- **V needs no transpose** (`VLAYOUT=2`): the O-GEMM's B operand contracts over
  keys with N = head_dim, and declaring it MN-major makes the required smem
  layout coincide with V's natural global layout, so V streams with `cp.async`
  exactly like K. This is what made the deep pipeline possible at all — the
  fallback (`VLAYOUT=1`) transposes 8×8 bf16 blocks through registers, which
  forces synchronous global loads and cannot be prefetched.
- **Deep pipeline**: `NSTAGE_KV=5` K/V stages, `NSTAGE_S=3` S accumulators,
  `NSTAGE_P=2` P buffers, `PREFETCH_D = NSTAGE_KV − NSTAGE_P = 3`,
  `S_AHEAD = min(PREFETCH_D, NSTAGE_S) − 1 = 2`.
- Barrier bookkeeping uses one monotone counter (`wait_o(target)`) that derives
  each barrier's parity from the wait index, so the rescale path and the P-reuse
  path cannot consume each other's phases.

### Correctness constraints found the hard way

- **A V stage cannot be overwritten until the O-GEMM reading it has completed**,
  and completion is only guaranteed up to `wait_o(it − NSTAGE_P + 1)`. So the
  prefetch distance must be *decoupled* from the stage count
  (`PREFETCH_D = NSTAGE_KV − NSTAGE_P`) and the prefetch must be issued *after*
  that wait. An earlier version prefetched `it + NSTAGE_KV` at the top of
  iteration `it`, overwriting the very buffer O-GEMM(it) was about to read. This
  was the single change that took the kernel from 77.73 µs to 23.55 µs, because
  it is what allows the pipeline to be deep and fully asynchronous.
- **`NSTAGE_S` must exceed `S_AHEAD`**, or the early-issued S-GEMM targets the
  stage being read this iteration and serialises against it.
- `tcgen05.fence::before_thread_sync` / `after_thread_sync` are required around
  every TMEM handoff across a barrier; `fence.proxy.async.shared::cta` is
  required between `st.shared` writes and the MMA that reads them as operands.

## Remaining headroom

At 731 TFLOPS the kernel is at 34 % of measured tcgen05 peak, so the ceiling is
not the limit — latency exposure still is.

- **Warp specialisation.** A dedicated loader warp and MMA warp would take the
  remaining barrier waits off the softmax critical path. This is what the `ncu`
  data points at and the largest remaining item.
- **Occupancy.** Currently 1 CTA/SM at 229664 B smem. The tcgen05 probe shows
  +16 % throughput at 2 CTAs/SM, but that needs smem under ~114 KB, which at
  `BLOCK_N=64` leaves too few K/V stages to be worth it unless Q or P moves out
  of smem. Real but not cheap.
- **Tail effect.** 512 CTAs over 148 SMs is 3.46 waves; the last wave is 46 %
  occupied. A persistent kernel with a tile scheduler would also amortise the
  prologue.

## Process notes

Three GPU jobs in this session were lost to build errors that a compile-only
check would have caught in seconds: a `{0,0}` initialiser illegal at `NSTAGE=1`,
a missing torch extension build directory, and `PREFETCH_D` used before its
declaration. The cause was substituting a brace/paren/`#if` counter for a
compiler and reading its "ok" as "it builds" — that script cannot see a
use-before-declaration and never caught any of the three. `run_deep.sh` now opens
with an `nvcc -c` gate across all configs and exits before spending benchmark
time if the file does not build.

## Provenance

The `tcgen05` instruction encodings in `fmha.cu` — UMMA smem descriptor bit
layout, instruction descriptor bit layout, the `mma` / `ld` / `st` asm forms, the
commit-to-mbarrier form, and the placement of `tcgen05.fence::*_thread_sync` —
were transcribed from NVIDIA's public CUTLASS headers
(`cute/arch/mma_sm100_umma.hpp`, `mma_sm100_desc.hpp`, `copy_sm100.hpp`,
`cutlass/arch/barrier.h`) after the PTX ISA documentation page returned
truncated. Those headers describe the hardware instruction ABI.

Everything above that layer — tiling, pipeline structure and its stage
constraints, smem layouts, online softmax, lazy rescale, and the empirically
probed TMEM mapping and MN-major encoding — was worked out here. The reference
implementation (flash-attn-4's CuTe DSL kernel) was never opened. The better move
when the spec page truncated would have been to fetch the PDF or ask, rather than
reach for a library the reference itself depends on.

## Files

| file | role |
|---|---|
| `fmha.cu` | the kernel |
| `binding.cu` | torch extension entry point |
| `build_ext.py` | JIT build; `-gencode=arch=compute_100a,code=sm_100a` |
| `common.py` | inputs, fp32 naive reference, timing (profiler span + cuda events, L2 flush) |
| `check.py` | accuracy vs fp32 naive with tolerance from ref; seqlen and structured-input bisection |
| `bench.py` | ref vs custom, same method / inputs / iters / run |
| `bench_only.py` | custom kernel alone, for ablation runs |
| `prof_mine.py` | custom kernel alone, for `ncu` |
| `probe_mma.cu`, `probe_mma_sweep.cu` | `mma.sync` bf16 ceiling and the 7-point ILP × occupancy sweep |
| `probe_umma.cu` | sustained `tcgen05` throughput |
| `t5_gemm.cu` | first working tcgen05 GEMM; TMEM layout discovery |
| `t5_layout.cu` | B-operand smem layout candidates |
| `t5_major.cu` | MN-major encoding candidates; variant 2 is the one that works |
| `run_deep.sh` | compile gate + accuracy + bench for the pipeline configs |
| `run_ablate.sh`, `run_sweep.sh`, `run_prof.sh`, `run_major.sh` | experiment drivers |

Reproduce: `bash run_deep.sh` in the image below, or
`VLAYOUT=2 BLOCK_N=64 NSTAGE_KV=5 NSTAGE_S=3 RESCALE_THRESH=8.0f USE_CPASYNC=1 python bench.py`.

Environment: `cctl` B200 (`jiuzhang_train` / `aiforai-b200`), image
`model-align/sglang:v0.5.19`, CUDA 13.0, torch 2.13.0+cu130, 148 SMs @
1.965 GHz, 232448 B smem per block opt-in.
