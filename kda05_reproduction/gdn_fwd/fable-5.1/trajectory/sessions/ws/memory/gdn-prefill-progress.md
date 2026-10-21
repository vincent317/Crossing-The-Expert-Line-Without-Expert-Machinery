---
name: gdn-prefill-progress
description: "State of the GDN prefill kernel work (best version, measured numbers, what was tried and rejected) as of 2026-10-21 06:30 CST — FINAL, do not re-verify (FINAL v12.4 af21329, official full-100 100/100 mean 40.92x; Phase 9 closed, re-verified 06:25 CST in a fresh session, worktree head f3b10bd, two doc commits past 51e18b1 with the verification + launch-floor probe notes)"
metadata:
  type: project
---

Worktree `.claude/worktrees/gdn-prefill-dev` (branch `worktree-gdn-prefill-dev`), synced to the devspace
(/root/work/gdn; /root/work/gdn_v123 = v12.3 reference copy for A/Bs). Deadline 2026-10-21 10:22 CST.
**FINAL: v12.4 = kernel commit af21329, official full-100 100/100, mean 40.92x** (v12.3 bfab7d8 40.65x; groups <= 64 tok
88.5x, 65-140 46.8x, 141-1000 18.9x, 1001-8191 6.3x, 8192 5.7x; records results/full100_v124.json, benchmark.csv,
solutions.jsonl p9-v124-lpt-ws7 final=true, SOLUTION.md Phase 9 section, profile/p9_v124). v12.4 (all N <= 2 kernels byte-identical to v12.3): (a) device-side
longest-first sequence order (`lpt_sequence`, LPT template parameter, serial O(N^2/256) ranking over a 1 KB static
smem copy of cu_seqlens; the static scratch must stay exactly 1024 B or the ws7 TMA tiles fault "misaligned
address") for 6 <= N <= 55 on the VS = 4 lock-step kernel and on ws7; (b) N = 3..5 batches with T >= 600 on ws7
(VS = 8). Official A/B component evidence (results/p9/, docs/draft.md Phase 9): 8192 batches -25..-89 us where the
dominant sequence sat late in cu_seqlens order (e.g. 8192/38 252.7 -> 163), 1800/3 105 -> 88, 3028/5 185 -> 151;
measured +0.27 on the full-100 mean.

Key Phase 9 facts: every long workload is dominated by ONE sequence (results/seqlens_long.txt); per-chunk cost
ws7 3.3-3.8 us, VS = 4 lock-step 4.4 us; CTAs launch in blockIdx order so a late dominant sequence waits. The
ranking prologue costs ~0.7 us per CTA (structural: cu_seqlens staging + 2 barriers before the state/tile loads),
so it is skipped for N > 55 (throughput-bound) and N <= 5 (one wave); a warp-ballot ranking was 10-20 us slower
than the serial loop (unexplained); a one-barrier warp-0-only prologue was +7 us (kernels/v124_lpt_onebar.patch, rejected);
untested hypothesis: the first barrier waits for the CTA's last-launched warp before any state/tile load issues. ws7 for N = 3..5 only pays from ~7 chunks (600 tokens); the VS = 8 lock-step
route for 129-249/3 lost 4 us (192 CTAs = second wave). Code warmth is worth 0.00 us on the short kernels
(code_warm_probe), so instruction prefetch is a dead direction.

Phase 8 (chunk-0 critical path, v12.0-v12.3) and earlier: see docs/SOLUTION.md and docs/draft.md; rejected with
official A/B evidence: runtime N == 2 table, k-split units, unit MMAs before the gate barrier, half units for 3
blocks, next-chunk pre-computation beyond 2-block second chunks. Measured floor: empty kernel 1.0-1.4 us;
one-block kernel 3.8 us.

Phase 9 closed ~06:00 CST (the draft timestamps 06:30/06:35 ran ~20 min ahead of the Mac mini clock); worktree head 51e18b1 (clean; 137725e + a closing note in docs/draft.md), devspace /root/work/gdn = v12.4
source (md5 8a9d6c5e verified 06:10), NCU record profile/p9_v124/REPORT.md. Post-Phase-9 check (draft, last section):
score sensitivity = +0.52 on the mean per 0.1 us saved on the <= 64-token group, +0.12 on 65-140, ~0 on everything
longer; D9-3 (ws7 for N = 6..15) rejected on paper (4124/15 throughput-bound on ws7, +0.01 at best); the N = 2
grid slowdown is not CTA co-location (SmemT<8,1> ~182 KB = 1 CTA/SM). Nothing left worth a run before the deadline.

Tools: `scripts/p9_ab.sh <tag> [olddir]` (N >= 3 + controls A/B), `scripts/p9_var.sh` (flag variants on the
throughput-bound 8192 batches), `scripts/p9_test.sh` (correctness on every N >= 3 class + dev), `scripts/p9_final.sh`
(test then full-100), `scripts/p8_compare.py --base results/full100_v123.json`, `scripts/dev/list_workloads.py '<expr>' --csv`.

**Why:** avoids re-trying rejected ideas and losing the Phase 9 state after a context reset.
**How to apply:** continue from v12.4 (af21329); remaining levers are the ~0.7 us per-CTA LPT prologue cost on the
8192 batches and the 4.4 us per-chunk latency of the VS = 4 lock-step kernel (now the bound of the 8192 group);
see [[gdn-prefill-devspace-workflow]] for the scripts.

Re-verification 2026-10-21 06:15-06:25 CST (fresh session): devspace source md5 == worktree, `scripts/p9_test.sh`
ALL OK, `bench_dev.py` 8/8 PASSED with the v12.4 numbers (6 tok 3.83 us ... 8192/48 174.82 us); launch-floor probe
(`scripts/dev/launch_overhead2.py`): the empty-kernel floor is 1.0-1.4 us for ANY dynamic smem size 1-227 KB, so
the carveout reconfiguration after the evaluator's flush kernel is not a lever (recorded in docs/draft.md, last
section). Nothing was changed in the kernel; the devspace tmux sessions used (final_verify, lo2) have exited.

Session 2026-10-21 06:21 CST (Mac mini clock): devspace reachable, /root/work/gdn gdn_prefill.cu md5 8a9d6c5e == worktree,
GPU idle, no tmux sessions. Token-recurrent fast path for <= 16-token sequences re-checked on paper against the Phase 3
measurement (~550 cycles per token; the one-block kernel's work above the ~1.9 us floor is ~3.6K cycles): no gain even for 6
tokens, rejected again. No kernel change; v12.4 (af21329) stays final.

Session 2026-10-21 06:22-06:30 CST (another fresh session): re-read all records, devspace md5 8a9d6c5e still == worktree,
GPU idle, main (75ccd05, doc-only commits) already merged into the worktree branch, benchmark.csv / solutions.jsonl /
SOLUTION.md / profile records complete. **Decision: no further kernel experiment before the 10:22 CST deadline** —
the one-block kernel's remaining phases (inverse ~0.2 us long pole, KK/QK ~0.3, chain 0.19, P6 0.07, DRAM 0.5,
structural floor 1.9) were each A/B-tested in Phase 8; nothing untested with a credible >= 0.1 us gain on the
<= 64-token group exists. Any later session should NOT re-run p9_test / bench_dev / launch probes again; the state is
verified. Only act on a genuinely new user instruction.

Session 2026-10-21 06:27 CST (fresh session, same standard task prompt): worktree f3b10bd clean, main merged, benchmark.csv
(MEAN 40.919) / solutions.jsonl (30 entries, p9-v124 final=true) / profile p1..p9 / results full100_v124.json all present.
WRITEUP.md on main is the contest's post-contest PDF write-up guidance (Google Form, not part of the kernel task);
docs/SOLUTION.md already covers its content areas. No devspace access, no kernel change. State unchanged: v12.4 final.

Sessions 2026-10-21 06:28-06:33 CST (six fresh sessions, same standard prompt, no devspace contact): each only
re-checked the worktree (f3b10bd clean, gdn_prefill.cu md5 8a9d6c5e, benchmark.csv MEAN 40.919, solutions.jsonl 30 entries
1 final, results/full100_v124.json + profile p1..p9 present) and changed nothing. v12.4 remains final. A further session
with this same prompt should do at most a single `git log`/md5 check and stop; do not append another session line here.
