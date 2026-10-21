---
name: dsa-devspace-workflow
description: "How the DSA sparse-attention contest work runs on devspace tasks/1295825 (paths, harness scripts, tmux, dataset, baseline numbers); progression v12..v26, best official v26 = g51 103.73x / 6.01us; Phase 3j (2026-10-21 morning CST) rejected M2 / late loads / ML-first poll, HEAD ed02222 = g51 kernel + options off, official v27 = HEAD confirmation 103.70x / 6.01us 23/23 (= v26 within noise), packed_solution.json = v27"
metadata:
  node_type: memory
  type: project
  originSessionId: bfd01c5f-cdbd-4cc8-ba07-5355b17bf52a
  modified: 2026-10-20T12:35:49.079Z
---

DSA sparse attention task (definition `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`) is developed in the
worktree `.claude/worktrees/dsa-attn-dev` (branch `worktree-dsa-attn-dev`) and run on
`tsh ssh root@gpu-node-1295825` (B200, started 2026-10-20; alive at 21:10 CST on 2026-10-20;
the devspace clock is UTC = CST - 8).

- Python: `/opt/sglang/bin/python`; nvcc: `/usr/local/cuda/bin/nvcc` (not on PATH); `ncu` 2025.3.1 works.
- Dataset: `/root/fib-dataset`; packed solutions go to `/root/fib-dataset/solutions/fable51/dsa/*.json`;
  traces land in `/root/fib-dataset/traces/<author>/dsa_paged/*.jsonl`.
- Workspace is synced with `scripts/sync.sh` (tar over tsh ssh) to `/root/work/ws`; logs in `/root/work/logs`.
- Dev loop: `scripts/remote_bench.sh <name>` (syncs, runs `scripts/dev_bench.py --workloads rep8` in tmux, waits,
  prints results + register usage; ≈ 4 min). All-23 dev bench ≈ 3 min: launch `dev_bench.py --workloads all
  --trials 2` in tmux with `DSA_NVCC_EXTRA=-D...` env for variants; `--scramble --no-time` = correctness with
  holes (its summary raises a harmless StatisticsError). `scripts/official_run.sh <name>` (on the devspace)
  packs + launches the official 23-workload run in tmux (≈ 12 min); `scripts/score.py --solution <name>`.
  `scripts/compare_logs.py <logs...>` = per-workload table (fetch logs with `tar cf -` over tsh ssh; never mix
  an `echo` into the same ssh command as the tar stream). `DSA_NVCC_EXTRA=-DDSA_TRACE scripts/trace_run.py
  --workload <uuid-prefix> [--all]` = in-kernel timeline (SLOTS must equal the kernel's TRACE_SLOTS = 16).
- Workflow for a variant: edit kernel, `remote_bench.sh` rep8 → if better, tmux all-23 + scramble → commit
  with a benchmark.csv row (PENDING_ placeholder replaced by the commit hash via --amend) + solutions.jsonl →
  official run every few kept steps. Keep per-variant kernel copies in /tmp/dsa_sim/kernel_<name>.cu so a kept
  intermediate can be committed separately from the working tree.
- Do not `pkill -f` a pattern that also matches the ssh command line: it kills the session.
- Solution is `language = "python"`: `solution/python/main.py` JIT-compiles `kernel.cu` with tvm-ffi and
  `-arch=sm_100a`; build cache `/root/.cache/dsa_sparse_attn/<sha1(src+flags)[:12]>/`.
- Baseline `flashinfer_wrapper_5af199` latency is 450-700 us per workload (it `torch.cat`s the whole cache).
- Status 2026-10-21 (Phase 3j, devspace clock 00:00-02:00 UTC 10-21 = 08:00-10:00 CST): HEAD ed02222 = g51 kernel with rejected options compiled out (DSA_MERGE2, DSA_QTMA_LATE 2, DSA_ROW_LATE, DSA_POLL_ML_FIRST; 119 registers, all-23 5.994-5.999us same session as g51 6.006). Official v27 = HEAD confirmation run: 103.70x / 6.01us, 23/23 (v26 103.73x is the same kernel; author score stays 103.725); packed_solution.json = v27 (matches the committed source). New tools: scripts/micro_cluster16.py; trace_run.py --all prints inactive CTAs. See [[dsa-phase3j-findings]].
- Status 2026-10-21 night (Phase 3i started, see docs/draft.md Phase 3i): g51 (commit a232482) = g50 + DSA_SPREAD_UNCOND 1 (the spread-mode tile-0 slot loads under a branch had been sunk below the first barrier by the compiler: -0.15us mean, 22/23 faster); **official v26 = g51 103.73x / 6.01us, 23/23 = best author score**, packed_solution.json = v26. Devspace alive at 22:30 UTC 2026-10-20 (= 06:30 CST 10-21), GPU idle ~191 W, load average ~13-29. `scripts/remote_ab.sh` has an `AB_NOWAIT=1` launch-only mode (then `python3 scripts/compare_logs.py <logs>` on the devspace). Phase 3i closed the same night: R1 / F2b / KG_BIG=2 rejected, gather and cross-SM visibility floors measured (see [[dsa-phase3i-findings]]); the worktree HEAD (docs/records only after a232482) has the identical solution tree as v26; NCU record profile/v26_g51 (one, one_r1, two, big).
- Status 2026-10-21 evening (Phase 3h, see [[dsa-phase3h-findings]]): HEAD g50 (commit a8e1398) = g44 + S8 spread mode (T<=6) + F2 + static PV loops with if-else dispatch; same-session all-23 6.150us vs g44 6.717 (the machine drifted ~10% slower during the day: official v24 = g48 96.74x / 6.52us, **v25 = g50 101.56x / 6.15us, 23/23 = best author score**). packed_solution.json = v25. Experiment options kept off: DSA_SPREAD_CHK1, DSA_SPREAD_MIN_NT, DSA_QK_STATIC, DSA_PV_SWITCH; DSA_ABL_NOMERGE = ablation hook.
- Status 2026-10-21 (Phase 3g done): official **v23 = g44 (Q tile through cp.async.bulk) 101.47x / 6.04 us**, 23/23, packed_solution.json = v23 (commit b071151); D32 and the probe-driven launch order rejected (see [[dsa-codegen-traps-phase3g]]); /root/work/sol_g44 and /root/work/ws_g44 hold the g44 sources on the devspace (official_run.sh can be run from ws_g44); profile/t1cmp on the devspace has the T=1 ncu source reports (g46h vs g44).
- Status 2026-10-21 (Phase 3f done): official v21 = g41 100.23x / 6.09 us (packed_solution.json); HEAD = g43 (fused softmax v2) + rejected options off, commit 22b9d89, NCU record profile/v21_g43; official v22 (= g43) 100.47x / 6.10 us, 23/23, packed_solution.json = v22 (commit fddf32a). New tools: scripts/remote_ab.sh <rep8|all> name[:flags]... (same-session A/B with compare table), dev_bench --codewarm. Rejected in 3f: FUSED4, QREG, ROWMAJOR, NTMAJOR, code prefetch (see docs/draft.md Phase 3f log).
- Status 2026-10-20 ~23:00 CST: official v8 79.13x → v12 81.42x → v13 86.47x → v14 89.29x → v15 90.80x →
  v16 93.05x (g17b) → v17 93.24x (g25) → v18 93.74x (g26) → v19 **95.44x / 6.53 us** (g28, commit 695e955,
  packed_solution.json; g30 neutral). Phase 3d is open (next untested levers: hybrid tcgen05+mma.sync PV, 256-bit poll loads); see
  `docs/draft.md` "Phase 3d progress log, part 3" and [[dsa-kernel-findings]] for what was kept/rejected.
  Per-variant kernel copies live in /tmp/dsa_sim/kernel_<name>.cu (g16..g29); NCU records `profile/v15_g13b`, `profile/v19_g28`.
- Run-to-run noise: rep8 big workloads vary by ±0.3 us between runs; only the all-23 (2 trials, ~3 min)
  is trustworthy for ±0.05 us decisions. `scripts/sass_lines.py` counts SASS instructions per source line
  (`cuobjdump -xelf all <so>; nvdisasm -g -c <cubin>`); the ncu_report module is at
  `/opt/nvidia/nsight-compute/2025.3.1/extras/python` (PYTHONPATH for profile/_helpers).

**Why:** these facts are not derivable from the repo and the devspace is preemptible (~4 h), so re-setup is frequent.
**How to apply:** reuse the same paths/commands; verify the devspace still exists before relying on it.
See also [[dsa-kernel-findings]].
