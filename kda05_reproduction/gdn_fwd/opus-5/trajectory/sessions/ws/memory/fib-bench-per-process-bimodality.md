---
name: fib-bench-per-process-bimodality
description: "flashinfer-bench latency on B200 is bimodal per process, so comparing two settings in separate runs measures allocation placement rather than the change"
metadata:
  node_type: memory
  type: reference
  originSessionId: ba4245a3-3aeb-4f46-9839-0dcdd7969d8e
  modified: 2026-10-21T23:05:00.000Z
---

On the B200 devspace, `bench_gpu_time_with_cupti(..., cold_l2_cache=True)` gives
results that are **bimodal per process** for kernels that use a large persistent
workspace tensor: the same build measured the `gdn_prefill` 294-token/3-sequence
workload at 17.9 us in one run and 20.4 us in another, stably within each run.
Kernels with no workspace were stable to ±0.5 %. The only process-level state
that moves is the workspace allocation, so placement is the likely cause.

Consequence: a 5-10 % difference measured in two separate processes can be
entirely noise. A programmatic-dependent-launch change looked like 1.10-1.14x
across runs and measured 1.010x when paired.

**How to measure instead:** make the knob a *runtime* switch (a launch
attribute, or an env var read per call) so one process can alternate it, then
alternate the two settings between measurements of the same workload in that
process and take the median per-round ratio. `dev/pairab.py` and
`dev/paircfg.py` in the project do this. Also score a change on the workloads it
touches (`dev/subset.py`), not on the full-set geomean — a change affecting ten
of 100 workloads moves the geomean by about the run-to-run spread — and take the
per-workload minimum over several full passes (`dev/minmerge.py`) for tracking
numbers, because single passes contain occasional 2-4x single-workload outliers
that alone move the geomean by over 1 %.

Scale of the official evaluator's own spread on this definition: three runs of
one build gave **18.089, 18.151 and 18.259 us**, i.e. **+-0.47 %**. Two readings
had suggested +-0.35 %; don't trust a two-reading estimate of a spread. Anything
under ~1 % has to be decided paired.

See [[gdn-prefill-b200-project]] and [[precompute-pays-per-consumer]].
