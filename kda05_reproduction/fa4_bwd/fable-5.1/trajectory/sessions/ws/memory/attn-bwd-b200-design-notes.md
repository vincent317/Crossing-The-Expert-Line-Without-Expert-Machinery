---
name: attn-bwd-b200-design-notes
description: "Non-obvious findings about the CuTe DSL attention backward kernel on B200 (what limits it, what did not work, DSL pitfalls, measurement pitfalls)"
metadata:
  node_type: memory
  type: project
  originSessionId: c1bbcd8f-eda7-465f-ba66-9359d562e054
  modified: 2026-10-21T08:00:00.000Z
---

Kernel: solution/python/fmha_bwd_sm100.py (derived from the BSD CUTLASS 4.5 fmha_bwd example), 1 CTA per
128-key block, 15 warps (4 reduce, 8 compute, 1 mma, 1 load), persistent (1 CTA/SM, atomic work counter).

Findings (2026-10-20/21):
- Tensor-pipe floor (scripts/mma_rate.py): full N=128 S/dP/dV/dQ/dK = 1.44 us per 128x128 tile; N=64 halves
  for dP/dQ = 1.70 us (+18%). Measured tile period ~2.7 us (v005), ~2.4 us in the IKET probe after D14.
- D14 (2026-10-21, kept): full-width dP/dQ through one 128-col TMEM region with V as a smem A operand
  (2 Q stages) beat the v005 layout (V in TMEM, N=64 halves, 3 Q stages): MHA 1.255 / GQA 1.203 /
  varlen 1.466 ms (dev). Reason: the in-order MMA warp's chain had two extra handshake hops per tile
  (dP half 0 release -> half 1, dQ half 0 drain -> half 1). After D14 the compute warps are the critical
  path (softmax 1.25 us: MUFU-bound 0.52 + issue; dS 0.53; waits ~0.4) and the MMA warp waits ~0.45 us
  per tile for P.
- fp16 TMEM accumulators do NOT save TMEM: a 16-bit accumulator still takes one 32-bit column per element
  (tcgen05.ld .pack::16b); and bf16 A/B require an fp32 accumulator anyway. GQA fp16 accumulation also
  fails one dV element (scripts/sim_f16acc.py). Dead end.
- Compute-warp micro-opts that were neutral while the MMA chain was critical (vectorized lse/delta loads,
  split P-store barriers, PRMT bf16 packing with a (1+2^-9) lse bias fold, exp2 FMA emulation): keep the
  knobs, re-test whenever the compute warps become critical (after D14 PRMT packing is worth ~1%).
- Rejected (all correct, all slower or neutral): ping-pong dP/dQ regions with halves, reduce warps computing
  half of dS, P^T in its own TMEM region, specialized causal mask, L2 prefetch, 2-CTA multicast, MMA reorders,
  double-buffered dO (Q2/dO2) with the halves layout, dO-first load order (slightly slower).
- TMA loads (Q/dO 32 KB) take 1.4-2.4 us from issue to landing in IKET, with or without the reduce-add
  traffic; the kernel at 1965 MHz takes 24% more SM cycles than the clock-locked (974 MHz) NCU run, so
  ~0.5 us per tile is absolute-latency bound. Keep Q/dO prefetch >= 2 tiles ahead; lse/delta pipelines need
  3 stages, otherwise the load warp is throttled (1-stage lse costs 2.4% after D14).
- Register budgets with the reduce warpgroup below 128 trap ("illegal instruction"); setmaxnreg configs
  whose increases exceed the decreases hang.
- DSL pitfalls: pipeline states referenced through a Python list are not seen as loop-carried (phases reset,
  deadlock); a PipelineState advanced inside a @cute.jit helper is not propagated back (return it);
  range_constexpr bounds must be plain names of Python ints; TMEM-side partition shapes include a 32-lane mode,
  register tensors must use the D-side (coordinate) partition shape; a variable assigned in two const_expr
  branches with different types fails the TYPE_UNSTABLE_JOIN check (use distinct names); inline PTX via
  llvm.inline_asm works for bit tricks (see fmha_mask.pack_bf16x2_rz); CUTE_DSL_KEEP=sass,cubin with
  CUTE_DSL_DUMP_DIR dumps SASS for inspection.
- Debugging a hung kernel: pinned host buffer passed as a raw gmem pointer (cute.runtime.make_ptr) that warp
  roles write progress markers into; poll it from the host (scripts/probe_hang.py).
- IKET per-tile timelines: scripts/iket_run.sh <tag> [ENV=..] then scripts/iket_timeline.py (one CTA,
  time window, selected warps) shows who waits on whom; summaries alone mislead (nested ranges double count).

Round 3 (2026-10-21), D16 "16 compute warps" rejected (1.43-1.48 ms vs v006 1.22):
- The decisive measurement is FMHA_BWD_DBG=nocomp (compute warps skip all math, handshakes kept):
  v6 1.228 -> 1.151 ms. The compute warps add only ~6%; the MMA-issue/load chain alone is ~2.27 us
  per tile vs the 1.44 us tensor floor (Q wait 0.17, dQ-drain/dP-region 0.18, S-region 0.12, dO 0.1,
  MMA issue). Do not optimize the softmax/dS code further unless that chain shrinks first.
- More warps per SM sub-partition slow the single-threaded MMA warp: 16 compute warps (4 per SMSP)
  halve the softmax math (0.33 us per warpgroup) but the kernel gets slower.
- With 2 Q stages, Q(i+1) is loaded only after dK(i-1) retires; TMA flight time 1.4-2.4 us means any
  schedule that issues S(i+1) < ~1.5 us after dK(i-1) stalls. A software-pipelined compute loop
  needs 3 Q + 2 dO stages (256 KB) and cannot fit.
- setmaxnreg: the compile cap is 65536/threads rounded down to 8 (576 threads -> 112, 704 -> 88);
  ptxas may compile below the cap (c16 used 96) and an increase that the same CTA's decreases cannot
  fund hangs. Budget 0 = skip the instruction (regCount must be 24..256).
- A cp.async.bulk wait_group.read right after issuing a TMA reduce-add costs ~0.8 us (the TMA engine
  reads smem late); place such waits a tile later. One staging buffer per producer then needs
  PipelineTmaStore with num_stages=1 (wait_group 0), otherwise the next write races the read.
- cute.struct: a conditional annotation (`if cond: field: ...`) inside the struct body is dropped by
  the DSL's source rewriter; declare fields unconditionally and size them by mode.
- A tiny Align[16] field between 1 KB-aligned buffers costs a full 1 KB of padding; the smem limit
  is 232448 B and 7 x 32 KB + 2 x 1 KB + 1 KB (barriers) fits exactly.
- IKET ranges cost ~0.03-0.06 us each; with ~25 ranges per tile the compute warps' per-tile time is
  inflated by ~0.8 us, so use timing-only DBG ablations for magnitudes and IKET for ordering only.
- Per-tile dependency loops of the v6 layout (measured, 2026-10-21): with compute the period is the
  single-dO-stage loop S 0.28 + softmax 1.25 + dV 0.28 + TMA flight ~1.4 - 0.84 = 2.37 us; without
  compute the 2-stage Q loop 0.84 + 1.4 = 2.24 us. The TMA flight time is the same for contiguous and
  strided rows and is L2 queueing (L2 ~68% busy: 64 KB loads + 32 KB reduce-add RMW per tile per SM).
  Rejected in-CTA fixes: S as N=64 halves + separate P^T region + Q half-tile ring (D19, 1.35 ms),
  dK-before-dQ (-4.5%), two dO stages via dQ staging in the dS buffer (D20, 1.41 ms: the reduce-add
  reads the staging ~0.8 us after issue). Recommended next: 2-CTA pairs with Q/dO multicast and a
  DSMEM pre-reduction of dQ (halves L2 traffic). Knobs for all of these remain in the code
  (FMHA_BWD_SHALF, FMHA_BWD_DO2, FMHA_BWD_DK_FIRST, FMHA_BWD_PF [not functional], FMHA_BWD_WARPS).
- cute.arch.prefetch needs an LLVM pointer; `tensor.iterator` of a sliced gmem tensor came back as
  a tuple in the load warp (unresolved); PipelineAsync.consumer_get_barrier(state) exposes the
  consumer-release (empty) mbarrier so other warps can wait on "both MMAs retired".
- Measured TMA flight time (load warp waiting on its own Q load, IKET loa_tma_Q, 2026-10-21): 32 KB
  tile median 0.83 us (p90 1.1, p99 2.2) on the full GPU, 0.70 us isolated/cold; 16 KB halves 0.70 us.
  The earlier "1.4-2.4 us" was inferred, not measured. L2 bandwidth is not the limiter (skipping the
  dQ reduce-add TMA gains 2.6%). The no-compute floor (~2.27 us/tile) is the single MMA warp's own
  issue stream: cute.gemm emits ~18-21 instructions (descriptor rebuild + 4x R2UR) per tcgen05.mma,
  40 MMAs per tile. Hoisting warp-uniform stage slices (cute.arch.make_warp_uniform on the pipeline
  stage index, slice once per MMA group) gained 0.5-1.5% (D22). Bigger gains need hand-built UMMA
  descriptors in uniform registers (inline PTX), or fewer MMA instructions per tile.

**Why:** each A/B costs minutes of GPU time and several of these contradict the naive pipeline model.
**How to apply:** always A/B in the same conditions (noise ~1-2%); never run torch.profiler in the same process
as the CUPTI timing; the Bash guard rejects compound commands containing "git"/"commit" (write patch scripts
with the Write tool); put remote loops in script files (remote.sh -t expands $vars too early; nested quotes
break tmux). See [[attn-bwd-b200-environment]].

Round 5 (2026-10-21): the 2.27 us/tile no-compute floor is NOT the MMA warp's issue stream
(D23 hand-built tcgen05.mma: floor unchanged) and NOT shared-memory bandwidth (mma_rate.py with
4-8 hog warps streaming 64-256 KB per tile: 1.438 us regardless). Unexplained; stop chasing it.
- The pre/post kernels were 12% of the total (preprocess 95 us, convert 53 us, main 1082 us for
  MHA; dev_bench --breakdown). Under the cold-L2 timing protocol (flush buffer leaves dirty L2
  lines) the torch floors are 40 us per 128 MB memset and 48 us per 256 MB copy, so standalone
  memory kernels cannot get much faster: v2 kernels (4 warps, 16 loads in flight per lane,
  16-lane shuffle reduce) only reach 84/51 us. Token-major mapping is slower (4-byte scattered
  delta/lse writes, zeroing scattered across heads); grid order is irrelevant.
- Fusing the dQ convert into the main kernel (D26c, per-tile arrival counters, last arriver
  converts + re-zeroes the fp16 tile so the workspace stays zero across calls; workspace
  allocated with torch.zeros) has a measured potential of -3.6% (skeleton ablation 1.166 ms MHA)
  but ANY gpu-scope fence (~0.2 us), acq_rel atomic round trip (~0.6 us) or DSL-indexed row
  loop on the reduce warps' per-tile path costs 0.1-1.5 ms: the reduce warps' slack per tile is
  small. Offload such work to the spare warp 14 (v6 layout has warps 14/15 idle at 40 regs;
  budget allows 96) through an mbarrier ring; `cp.async.bulk.wait_group N` (no .read) on the
  issuing warp is free when N covers the current tile's groups.
- Raw-pointer tensors in device code work and vectorize: cute.make_ptr(dtype, int64_addr,
  cute.AddressSpace.gmem, assumed_align=16) + make_tensor(ptr, make_layout(8)) -> .load()/.store()
  are single 16-byte accesses (rawptr_probe.py: 5.7 TB/s). Prefer them over indexing dynamic
  gmem layouts in hot loops.
- Pitfalls: workspace sub-regions must be offset from the uint8 workspace pointer (adding a byte
  count to an fp16-recast pointer doubles the offset -> silent out-of-bounds atomics that
  corrupt neighbouring tensors); a value assigned inside `with cute.arch.elect_one()` is not
  yielded out of the region (use `if lane == 0:`); cute.arch.atomic_add returns a raw IR value
  (wrap in Int32() before yielding/shuffling); declare loop-carried scalars before `while`.
- setmaxnreg is warpgroup-scoped: warps 12-15 (mma 80, load 80, spare 40, spare 40) share a
  warpgroup; giving warp 14 a budget of 96 hangs every CTA at its first tile (compute warps never
  see LSE) even if warp 14 does nothing, while 40 or 80 work. Keep extra roles in that warpgroup
  at <= 80. Diagnose hangs with scripts/probe_hang.py (slots 28-31 = finisher ring state).
- The CuTe DSL file cache can serve a stale module when only a @cute.jit helper changed: an ICE
  with identical IR numbers across source edits means a cache hit; run with
  CUTE_DSL_DISABLE_FILE_CACHING=True (or change the kernel body) before debugging the ICE.
- Stale deadlocked processes keep the GPU at 100% and block new launches; always check
  `pgrep -fa scripts/` and `nvidia-smi` before a run; pkill patterns must be anchored
  (`^/opt/sglang/bin/python scripts/...`) or the remote wrapper shell kills itself (exit 255).
