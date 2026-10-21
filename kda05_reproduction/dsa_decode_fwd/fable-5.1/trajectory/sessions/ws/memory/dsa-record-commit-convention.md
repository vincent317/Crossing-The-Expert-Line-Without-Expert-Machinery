---
name: dsa-record-commit-convention
description: How to record commit hashes in benchmark.csv / solutions.jsonl for the DSA task (never via --amend; the pre-amend hash dangles)
metadata:
  node_type: memory
  type: feedback
  originSessionId: a31468c5-54b9-4d22-84a0-c9d996bfd650
  modified: 2026-10-20T15:13:40.342Z
---

For the DSA sparse-attention worktree (`worktree-dsa-attn-dev`): commit the perf change first, then add the
benchmark.csv / solutions.jsonl rows that reference its hash in a separate follow-up commit. Never write a
`PENDING_` placeholder and replace it with the hash via `git commit --amend`: the amend creates a new commit,
so the recorded hash points at a dangling object (15 rows had to be repaired on 2026-10-20, commit 91246ef).

**Why:** the records must form a DAG of commits that exist in the branch history (task requirement).
**How to apply:** `git commit` the kernel change → `H=$(git rev-parse --short HEAD)` → append rows with `$H`
→ `git commit` the records ("docs: gNN rows"). Also use the 7 single-CTA workloads as a run-drift control
when judging all-23 differences ≤ 0.05 µs on merge-path changes.
See also [[dsa-devspace-workflow]].
