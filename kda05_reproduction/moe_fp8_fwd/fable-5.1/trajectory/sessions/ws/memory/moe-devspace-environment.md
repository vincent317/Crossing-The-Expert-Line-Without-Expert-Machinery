---
name: moe-devspace-environment
description: "How to reach and use the B200 devspace for the MoE FP8 kernel task (paths, tools, transfer method, guard quirks, tmux pitfalls, benchmarking practice)"
metadata:
  node_type: memory
  type: project
  originSessionId: c336ce7b-a67f-47aa-b2e9-9c8e4589a464
  modified: 2026-10-20T12:01:47.838Z
---

Devspace for the MoE task: `tsh ssh root@gpu-node-1295872` (tasks/1295872, one B200, CUDA 13.0 driver 580). Reachable on 2026-10-20 (three sessions).

- Python: `/opt/sglang/bin/python` (torch 2.13+cu130, flashinfer 0.6.18, flashinfer-bench, cutlass DSL 4.6.0, deep_gemm, cupti-python). CLI: `/opt/sglang/bin/flashinfer-bench`. nvcc/ncu/compute-sanitizer at `/usr/local/cuda/bin` (export PATH first). nsys at `/opt/nvidia/nsight-systems-cli/bin`. `ncu_report` python module: `PYTHONPATH=/opt/nvidia/nsight-compute/2025.3.1/extras/python`.
- Dataset: `/root/fib-dataset` (definition, 19 workloads, baseline `flashinfer_wrapper_9sdjf3`). Scratch: `/root/ws` (code `/root/ws/moe`, logs `/root/ws/logs`, ncu `/root/ws/ncu`, nsys `/root/ws/nsys`, reference cache `/root/ws/ref_cache`). Official traces: `/root/fib-dataset/traces/{fable51,flashinfer}/moe/*.jsonl`; `scripts/official_mean.py <solution ...>` prints per-workload latency/speedup and the 19-workload mean.
- CUTLASS 4.5 headers: `/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include`; DeepGEMM SM100 sources under `.../site-packages/deep_gemm/include/deep_gemm/impls/`.
- Benchmark timing = CUPTI span from first kernel start to last kernel end, cold L2 (256 MB flush each iteration), median of the iterations, no CUDA graph. Inputs: `randn` for all random tensors (fp8 clamped to [-2,2]; scales N(0,1) incl. negatives).

**Why:** Re-discovering these costs ~20 min each session.

**How to apply:**
- File transfer: remote has no rsync; `scripts/sync.sh` pushes solution/, scripts/, config.toml with `tsh scp`; a hung scp leaves the remote partially updated — verify with a `grep -c` of a new symbol before launching.
- Run everything long in tmux through script files (`scripts/remote_s3_*.sh` pattern: wait on `tmux has-session -t "=name"`, `: > $LOG`, append `ALLDONE`); start with `tsh ssh host 'tmux new -s name -d /root/ws/moe/scripts/x.sh; tmux ls'`. `scripts/rctl.sh {ls,stop,start,log}` helper exists. Never prefix-overlapping session names. Do not sync solution/ while an official run is still packing (it packs at start; after "Running benchmark" appears it is safe).
- In the local zsh, never write an unquoted `echo ====` (zsh parses `=cmd`). Bash tool timeout is 10 min max — poll with short sleeps.
- Dev harness: `DEV_STATS=1 scripts/dev_bench.py --workloads 1,7,53 --iters 100 --trials 2 --tag <tag>` prints min/p10/med/p90/max; env switches `MOE_SCHED_TARGET`, `MOE_ROUTE_INKERNEL_T`, `MOE_ROUTE_COOP`, `MOE_COMBINE_ATOMIC`, `MOE_A_GATHER`, `MOE_ROUTE_SPLIT`, `MOE_NVCC_EXTRA=-DMOE_ABL_...` (with `--perf-always`). Traces: `MOE_TRACE=1 MOE_NVCC_EXTRA=-DMOE_TRACE scripts/trace_fused.py <T>` (fused small path, per-CTA phases) and `scripts/trace_dump.py <T>` (UMMA). Short runs (traces, microbenchmarks) execute at low clocks: compare only relative numbers, or make kernels ≥ 200 us.
- Official run: `scripts/official_run.sh <name>` (packs the solution, copies into the dataset, runs flashinfer-bench with the MoE tolerances; ~15 min for 19 workloads).
- Microbenchmarks: `scripts/bw_bench.cu` (modes k/u/v), `umma_test.cu`, `umma_test_f16.cu` (RAW/bench), `red_bench.cu`, `atomic_bench.cu`, `redux_bench.cu`, `route_bench.cu`.
