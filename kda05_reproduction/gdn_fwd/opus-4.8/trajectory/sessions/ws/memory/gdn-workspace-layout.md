---
name: gdn-workspace-layout
description: Where the GDN prefill work actually lives — fresh Mac workspace vs the real devspace working dir
metadata:
  node_type: memory
  type: project
  originSessionId: 5409622b-a580-45a9-a932-297a1fbcc062
  modified: 2026-10-21T00:59:01.539Z
---

For `gdn_prefill_qk4_v8_d128_k_last` (gated_delta_net track, B200 devspace
`tasks/1295739`, `tsh ssh root@gpu-node-1295739`):

- The Mac repo `.../kda-mx-gdn-opus48/ws` is a **fresh starter-kit copy** (no
  solution, no ledgers at session start). All real work lives on the devspace.
- **Authoritative dev dir: `/root/gdnws`** (NOT a git repo) — `solution/triton/
  kernel.py` (~30KB, the converged 3-kernel solution), `scripts/` (fastlat.py,
  dev_bench.py, cupti_oracle.py, oracle_sweep.py, quick_test.py), `docs/`
  (draft.md, solution_writeup.md), NCU reports (`profile/`, `*.ncu-rep`),
  full-100 logs/JSONs. `full100_confirm.log` = 100/100, 35.554ms.
- Released submission registered as **`gdn-prefill-opus48-v1`** (author `opus48`)
  at `/root/fib-dataset/solutions/mine/gdn/.../gdn-prefill-opus48-v1.json`
  (byte-identical to `/root/gdnws` kernel). Baseline = `flashinfer_wrapper_123ca6`
  (CuTe-DSL, unrunnable here per [[gdn-prefill-baseline-unrunnable]]).
- Run scripts from `/root/gdnws` with `/opt/sglang/bin/python -I scripts/...`.
  Quick live check: `dev_bench.py dev8 --no-baseline --no-profile-baseline`
  (official correctness gate, skips slow O(T) reference profiling).
- Task rule: **do NOT copy the released kernel.py into the Mac workspace
  `solution/`**. Only reconstruct process/doc deliverables there (docs/draft.md,
  solution_writeup.md, profiling.md, benchmark.csv, solutions.jsonl, STATUS.md).
- 2026-10-21 re-verify (latest session): devspace reachable (B200, driver
  580.105.08), dev8 8/8 PASSED live (sum 2.952ms, max_abs 6.14e-4), kernel
  sha256[:16]=`ec7764da62a5b3e8` len 30345 still == registered
  `gdn-prefill-opus48-v1.json` embedded source (BYTE-IDENTICAL==True). Artifact
  intact; no code change warranted (converged).
  (NB: authoritative check is sha256[:16]=`ec7764da62a5b3e8`; the real byte
  length is 30345 — an earlier note said 30339, a stale/mistyped figure.)
- 2026-10-20 re-verify: dev8 8/8 PASSED live @ locked 1830 (sum 3.021ms,
  max_abs 6.14e-4), same sha; Mac deliverables worktree clean @ c3fd322.
  See [[gdn-prefill-solution-state]] for the full optimization chain.
- Two Mac worktrees under `.claude/worktrees/`: **`gdn-prefill-dev`** (branch
  `worktree-gdn-prefill-dev`, the full iteration history P1-P16, HEAD e4aeaea) and
  **`gdn-deliverables`** (branch `worktree-gdn-deliverables`, a clean curated
  deliverable set on top of main). 2026-10-20: the deliverables worktree is now
  COMMITTED at **c3fd322** (config.toml + docs/{draft,solution_writeup,profiling,
  STATUS}.md + benchmark.csv + solutions.jsonl). Main `solution/triton/kernel.py`
  correctly stays the starter STUB. Devspace kernel.py sha256[:16]=`ec7764da62a5b3e8`
  still matches registered `gdn-prefill-opus48-v1.json` — artifact intact.
