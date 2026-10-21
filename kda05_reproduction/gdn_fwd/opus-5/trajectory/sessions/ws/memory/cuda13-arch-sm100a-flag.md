---
name: cuda13-arch-sm100a-flag
description: "In CUDA 13.0 nvcc, -arch=sm_100a does not emit .target sm_100a; only -gencode arch=compute_100a,code=sm_100a enables tcgen05"
metadata:
  node_type: memory
  type: reference
  originSessionId: ba4245a3-3aeb-4f46-9839-0dcdd7969d8e
  modified: 2026-10-20T17:36:02.450Z
---

With CUDA 13.0's `nvcc` (V13.0.88) on a B200, `-arch=sm_100a` produces PTX with
`.target sm_100`, and `ptxas` then rejects every `tcgen05.*` instruction
("Instruction 'tcgen05.alloc' not supported on .target 'sm_100'"). Same for
`-arch=sm_100`. Only the explicit form works:

```
nvcc -gencode arch=compute_100a,code=sm_100a ...
```

Verified by compiling the same file with all three flags: 16 ptxas errors, 16,
and 0. The probe lives at `dev/tc5test.cu` in the gdn-prefill project.

Note the shape check cannot be done this way: `tcgen05.mma`'s M and N live in a
runtime 32-bit instruction descriptor, so ptxas accepts any value and an invalid
one fails (or silently misbehaves) at run time. See
[[tcgen05-shape-and-layout-constraints]].
