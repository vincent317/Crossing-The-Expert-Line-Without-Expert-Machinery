---
name: gdn-prefill-devspace-workflow
description: "How the GDN prefill contest work runs on devspace tasks/1295863 (sync, runner scripts, patched baseline, logs, microbenchmarks) and key evaluator facts"
metadata:
  type: project
---

GDN prefill kernel contest work (definition `gdn_prefill_qk4_v8_d128_k_last`), worktree
`.claude/worktrees/gdn-prefill-dev` (branch `worktree-gdn-prefill-dev`).

- GPU: `tsh ssh root@gpu-node-1295863` (B200, torch 2.13+cu130, cutlass DSL 4.6.0,
  nvcc/ncu in `/usr/local/cuda-13.0/bin`, python `/opt/sglang/bin/python`). Devspace has no rsync;
  `scripts/sync_devspace.sh` tars the worktree to `/root/work/gdn`. Logs go to `/root/work/logs`.
- `scripts/test_kernel.py --workloads <uuids>` = correctness vs reference (fast); `scripts/bench_dev.py --solution
  solution.json --no-profile-baseline [--workloads uuids]` = official evaluator numbers (run `pack_solution.py`
  first); `scripts/phase_timing.py` = per-phase clock64 breakdown (-DGDN_TIMING build, cold L2; GDN_EXTRA_FLAGS
  env adds -D flags); `bash scripts/run_full100.sh <tag>` = official CLI on all 100 (run in tmux, ~15 min).
  kernel.py honours GDN_EXTRA_FLAGS too (module name suffixed) -> A/B bisects on the official evaluator.
- Microbenchmarks in `scripts/dev/`: inv_bench.cu (inverse variants), load_bench.cu (tile load mechanisms),
  gates_bench.cu, launch_overhead.py. Build with nvcc -arch=sm_100a (-lcuda for load_bench).
- The dataset baseline `flashinfer_wrapper_123ca6` fails on this devspace (`cute.arch.ProxyKind` removed in DSL
  4.6.0). `scripts/patch_baseline.py` writes a working copy; baseline latencies: `results/baseline_all100.json`.
- Evaluator timing = CUPTI span per call with cold L2 (256 MB flush before each iteration, flashinfer
  `bench_gpu_time_with_cupti(cold_l2_cache=True)`); score = mean over 100 workloads of baseline/mine; ~50
  workloads have < 140 tokens, baseline is ~400 us on the 6-token workload.
- All 100 workloads have cu_seqlens[0] == 0 and cu_seqlens[-1] == T; 74 of 797 sequences are <= 16 tokens.

**Why:** these facts are not in the repo and were costly to discover.
**How to apply:** reuse the scripts; keep solutions to one kernel launch with no host-side device reads.
