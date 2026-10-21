---
name: gdn-prefill-b200-project
description: "Where the gdn_prefill_qk4_v8_d128_k_last B200 kernel work lives, how to run it, and what it reached"
metadata:
  node_type: memory
  type: project
  originSessionId: ba4245a3-3aeb-4f46-9839-0dcdd7969d8e
  modified: 2026-10-21T23:05:00.000Z
---

FlashInfer MLSys-2026 `gated_delta_net` track, definition
`gdn_prefill_qk4_v8_d128_k_last`. Work lives in the git worktree
`ws/.claude/worktrees/gdn-prefill-b200` (branch `worktree-gdn-prefill-b200`).

**State as of 2026-10-21 (Phase 6):** official evaluator 100/100 PASSED, geomean
**17.951 and 17.916 us on two runs of the same build -- 19.44-19.47x** over the
`flashinfer_wrapper_123ca6` baseline (348.870 us); compute-sanitizer clean (56
checks) over all seven dispatchable configs. Phases 1-6 complete and documented.

Phase 6's one keep is the **`P.Tm` fold**, worth +1.29%: `O = P.(Tm.rhs) =
(P.Tm).rhs` and both factors are state-independent, so the pre-kernel publishes
`PT = P.Tm` in place of `P` and the main kernel's `U = Tm.rhs` / `O += P.U` stop
being serial -- one stage, shared B fragments, one fewer per-chunk barrier.
`U` is still computed; only its serialisation goes away. Paired 1.011-1.037 on
the split path with every shape winning. The dispatched split cfgs are now the
**8xx** ids (805/825/855/801 = the 7xx ones with `PTF=1`); the 7xx ids stay
instantiated so the comparison can be re-run.

The **same fold in the fused kernel removes two barriers and is 5-10% slower** --
see [[gdn-prefill-ceilings]] for the rule that explains both results.

**Running it:** everything executes on the devspace
`tsh ssh root@gpu-node-1295801`, python at `/opt/sglang/bin/python`.
`bash dev/sync.sh` pushes the worktree, `bash dev/rrun.sh <name> '<cmd>'` runs a
job in a detached tmux session, `bash dev/rlog.sh <name>` tails it. Put the
command in a `dev/x_*.sh` script rather than inlining it in `rrun.sh` -- quotes
in an inline command get mangled by the ssh/tmux layers and the job dies
silently. Never run a long job in the foreground. `bash dev/official.sh` packs
and runs the official evaluator (it clears stale traces first, because
`--resume` keys on the solution *name* and would otherwise no-op); a full
100-workload official run takes ~15 min.

**Measure with the paired protocol**, not across runs -- see
[[fib-bench-per-process-bimodality]]. The official evaluator's run-to-run spread
on this definition is about **+-0.35%** (Phase 4 build: 18.257 and 18.338;
Phase 5 build: 18.089 and 18.151), so a sub-1% change can only be resolved
paired, and anything under ~0.5% paired should probably be rejected on
cost/benefit rather than kept. Always take TWO official readings per build
before quoting a number. `dev/pairab.py <attr> 0 1 <rounds> ALL` pairs a runtime
switch over all 100 workloads with each taking its natural dispatch; that is the
measurement to trust. Make a knob a *launch-time* choice (a kernel argument, or
a `cfg + 10000` offset the entry point decodes) so one process can alternate it.

Record every performance commit in `benchmark.csv` and every candidate (kept
*and* rejected, with the measurement that decided it) in `solutions.jsonl`;
`docs/writeup.md` §7 explains the protocol and §5 lists the 20 rejected
directions.

Architectural shape, if returning to it: three kernels in
`solution/python/gdn_prefill.cu` -- a fused one for `total_seq_len < 245`, and a
`(pre, main)` pair above that, launched back to back from one `extern "C"`
entry point with a programmatic dependent launch. `solution/python/main.py`
JIT-compiles with nvcc at first call and picks one of eight configs from
`(total_seq_len, num_seqs)` only; reading `cu_seqlens` on the host would cost
the D2H sync that gives the baseline its 167 us floor.
