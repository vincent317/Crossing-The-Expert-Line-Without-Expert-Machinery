---
name: tcgen05-shape-and-layout-constraints
description: "tcgen05.mma kind::f16 requires M in {64,128,256} and canonical UMMA shared-memory layouts, which rules it out for small-M gemms and for buffers also read by ldmatrix"
metadata:
  node_type: memory
  type: reference
  originSessionId: ba4245a3-3aeb-4f46-9839-0dcdd7969d8e
  modified: 2026-10-20T17:36:24.128Z
---

Before planning a `tcgen05` port, check these three things — they rule it out
for a lot of kernels that look like good candidates on instruction count alone:

1. **M is constrained and unchecked.** `kind::f16` takes `M ∈ {64,128,256}`
   (PTX ISA; CUTLASS's `UMMA::InstrDescriptor::m_dim_` comment says "4 (M=64),
   8 (M=128), 16 (M=256)"). N is a multiple of 8 up to 256; K is 16 per
   instruction. M and N live in a *runtime* 32-bit descriptor, so `ptxas`
   accepts any value — a bad M is a silent run-time failure, not a build error.
2. **A and B must be in a canonical UMMA shared-memory layout** — swizzled
   32B/64B/128B, or, for `SWIZZLE_NONE`, an *interleaved core-matrix* layout
   (`((8,n),2):((1,SBO),LBO)` in uint128 units), which is not plain row-major.
   See `make_umma_desc()` in `cute/atom/mma_traits_sm100.hpp`
   ([[cutlass-headers-in-flashinfer]]). If the same buffer is also read by
   `ldmatrix`/`ldmatrix.trans` for a leftover `mma.sync` gemm, or written by a
   scattered fp32->bf16 epilogue, you need swizzle-aware addressing in all of
   those places or a duplicate buffer.
3. **The accumulator is in TMEM**, so each gemm in a dependent chain adds
   `tcgen05.commit` + an mbarrier wait + `tcgen05.ld` before its epilogue. For
   small N and K that handshake can cost more than the MMAs it replaces: at
   N=16..32, K=64..128 each gemm is only 4-8 instructions.

Also: it only assembles under `-gencode arch=compute_100a,code=sm_100a`, see
[[cuda13-arch-sm100a-flag]].
