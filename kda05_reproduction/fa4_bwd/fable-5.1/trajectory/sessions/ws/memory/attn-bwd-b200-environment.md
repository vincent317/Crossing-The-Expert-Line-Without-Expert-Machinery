---
name: attn-bwd-b200-environment
description: "Devspace, dataset, evaluator and tooling facts for the attention_bwd B200 kernel task (flashinfer-bench, CuTe DSL 4.6.2, baseline numbers)"
metadata:
  node_type: memory
  type: project
  originSessionId: c1bbcd8f-eda7-465f-ba66-9359d562e054
  modified: 2026-10-21T09:00:00.000Z
---

Task workspace: kda-mx-fa4-fable51/ws (FlashInfer starter kit copy), development in worktree
`.claude/worktrees/attn-bwd-b200` (branch worktree-attn-bwd-b200). GPU box: `tsh ssh root@gpu-node-1305304`
(one B200, driver 580, CUDA 13.0 toolkit at /usr/local/cuda incl. ncu; python /opt/sglang/bin/python with torch 2.13+cu130,
nvidia-cutlass-dsl 4.6.2, flash-attn-4 4.0.0b19, flashinfer 0.6.18, flashinfer-bench 0.1.3.dev98, triton 3.7.1).
Dataset at /root/fib-dataset (definitions, 3 workloads, baseline solutions); traces saved under /root/fib-dataset/traces/<author>/attention_bwd/.
Remote scratch: /root/work/ws (synced copy of the worktree via scripts/sync_devspace.sh), logs in /root/logs, reference cache /root/work/ref_cache.

Baseline (flash-attn-4 b19, official command, 2026-10-20): MHA 1.372 ms, GQA 1.469 ms, varlen 1.559 ms (about 1 PFLOP/s).
Best state (2026-10-21 night, branch worktree-attn-bwd-b200, "v009" = v007 main kernel + D26a/b v2 pre/post
kernels with tuned block shapes, official): MHA 1.2041 ms, GQA 1.1660 ms, varlen 1.3838 ms (1.140x / 1.260x / 1.127x);
solution names fable51_attn_bwd_v009_{bshd,varlen} (config.toml name = fable51_attn_bwd_v009). v007 official was
1.2310 / 1.1941 / 1.4092; v008 (untuned v2 kernels) 1.2108 / 1.1742 / 1.3897.
Round-5 knobs (default off): FMHA_BWD_PRE=v1 (old pre/post kernels), FMHA_BWD_FUSE_DQ=1 (fused dQ convert, rejected),
FMHA_BWD_FZ*, FMHA_BWD_PRE_THREADS/STEPS, FMHA_BWD_FZSTATS=1 (+ scripts/fz_debug.py reads counters/stats back).
Round-3 experiment knobs (all default off = v007 behaviour): FMHA_BWD_WARPS=c16|c16r, FMHA_BWD_SHALF, FMHA_BWD_DO2,
FMHA_BWD_DK_FIRST, FMHA_BWD_PF (not functional), FMHA_BWD_VARLEN_ORDER, FMHA_BWD_DBG=nocomp|nos|nodk|nodp|nodv|nored|noredtma.
Records: benchmark.csv, solutions.jsonl (DAG baseline -> v001..v005), docs/draft.md (phases, every direction's numbers),
profile/<run>/ (NCU analysis text; .ncu-rep only on the devspace), experiments/patches (rejected variants as patch scripts).
Dev loop: scripts/dev_bench.py (CUPTI timing + fp32 reference check), scripts/ablate.sh, scripts/iket_run.sh,
scripts/bench_watchdog.sh (runs dev_bench under a Python watchdog; use it for anything that might deadlock),
scripts/probe_hang.py (host-mapped debug buffer, FMHA_BWD_DBGBUF=1, FMHA_BWD_MAX_CTAS caps the persistent grid),
scripts/sass_dump.sh <tag> (SASS of the bwd kernel + instructions between UTCHMMAs), scripts/iket_dist.py (duration
distribution of one IKET range; FMHA_BWD_IKET_TMAWAIT=1 makes the load warp wait on its Q TMA for flight times),
scripts/probe_c16.sh (hang probe + cuobjdump register/spill report). Long remote chains go through script files
(scripts/run_d22.sh pattern): inline multi-command tmux strings with env assignments failed silently once.

**Why:** these facts are needed every session and are not derivable from the repo.
**How to apply:** use `bash scripts/remote.sh '<cmd>'` (or `-t <tmux-session>` for long jobs) and `bash scripts/run_official.sh <name>`;
the Bash guard rejects compound commands containing the substring "git" (e.g. github URLs), so put such commands in script files.
Varlen definition must be value-returning (evaluator cannot size outputs from inputs: num_seqs is not an input dim).
CuTe DSL file cache (/tmp/root/cutlass_python_cache) makes recompiles take ~2 s. See [[attn-bwd-b200-design-notes]].
