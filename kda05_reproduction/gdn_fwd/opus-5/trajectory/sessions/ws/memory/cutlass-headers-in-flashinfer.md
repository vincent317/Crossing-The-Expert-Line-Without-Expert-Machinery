---
name: cutlass-headers-in-flashinfer
description: "A full CUTLASS/CuTe C++ tree ships inside the installed flashinfer package on the B200 devspace, usable as a reference for tcgen05/UMMA encodings"
metadata:
  node_type: memory
  type: reference
  originSessionId: ba4245a3-3aeb-4f46-9839-0dcdd7969d8e
  modified: 2026-10-20T17:36:11.333Z
---

On the B200 devspace there is a complete CUTLASS C++ source tree at

```
/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include
```

(not just the Python CuTe DSL, which is separately at
`.../nvidia_cutlass_dsl/dsl_packages/cutlass`). The files that matter for
writing `tcgen05` inline PTX by hand:

* `cute/arch/mma_sm100_desc.hpp` — `UMMA::SmemDescriptor` and
  `UMMA::InstrDescriptor` bitfields with per-bit comments, and
  `make_instr_desc()`
* `cute/atom/mma_traits_sm100.hpp` — `make_umma_desc()`, i.e. which shared-memory
  layouts are legal for A/B and how LBO/SBO/base-offset/swizzle are derived
* `cute/arch/mma_sm100.hpp`, `cute/arch/copy_sm100.hpp`,
  `cute/arch/tmem_allocator_sm100.hpp` — the PTX wrappers

Use it as a *reference*, not an include path: depending on a path inside another
package's data directory is not something a submitted solution should rely on.
