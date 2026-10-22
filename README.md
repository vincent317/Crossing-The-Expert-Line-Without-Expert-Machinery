# Crossing the Expert Line Without Expert Machinery

This repository contains only the operator (kernel) code generated in the experiments of
the paper ([`paper.pdf`](paper.pdf)):

- [`goal_single_shape/`](goal_single_shape/) — `/goal` experiments on a single specified shape
- [`goal_full_shape/`](goal_full_shape/) — `/goal` experiments on all shapes of an operator definition
- [`kda05_reproduction/`](kda05_reproduction/) — the KDA-0.5 reproduction
- [`ablation/`](ablation/) — the ablation experiments

## Environment

- GPU: NVIDIA B200 (`sm_100a`)
- CUDA: 13.0 (`nvcc` 13.0.88, driver 580.105.08). Two kernels point at a CUDA 13.1
  toolkit: `goal_single_shape/mla_paged_prefill/b1_q33_kv34/opus-5` and
  `goal_single_shape/gqa_paged_decode/b16_kv2841/opus-5`.
- CuTe DSL (`nvidia-cutlass-dsl`): 4.6.2. The DSA, GDN and MoE kernels in
  `kda05_reproduction/` use 4.6.0, the version pinned by the KDA-0.5 setup.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). Files adapted from NVIDIA
CUTLASS examples keep their BSD-3-Clause headers and license; they are listed in
[`NOTICE`](NOTICE).
