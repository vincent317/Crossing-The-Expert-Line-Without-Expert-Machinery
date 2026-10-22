# GDN prefill (Gated Delta Net, variable-length) — B200

Everything in `/root/gdn` was prepared for you before the session started.
Work in `/root/gdn/work`.  `/root/gdn/harness` and `/root/gdn/data` are inputs
— read them, import them, but do not change them.

## The task

Implement the FlashInfer-Bench definition `gdn_prefill_qk4_v8_d128_k_last`:

    run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
        -> (output, new_state)

    q, k        bf16 [total_seq_len, 4, 128]
    v           bf16 [total_seq_len, 8, 128]
    state       fp32 [num_seqs, 8, 128, 128]   (k-last: [N, H, V, K])
    A_log       fp32 [8]
    a, b        bf16 [total_seq_len, 8]
    dt_bias     fp32 [8]
    cu_seqlens  int64 [num_seqs + 1]
    scale       python float (0.08838834764831843 = 128**-0.5)

    output      bf16 [total_seq_len, 8, 128]
    new_state   fp32 [num_seqs, 8, 128, 128]   (k-last)

The full, authoritative semantics are in `data/definition.json`
(field `reference`), also extracted for you as
`harness/gdn_spec_reference.py`: a sequential fp32 recurrence, one token at a
time.  That file is the *task spec*, the same one every contest entrant gets.
It is slow on purpose; it is your numerical ground truth.

Note q/k have 4 heads and v has 8: each q/k head is shared by two v heads
(`repeat_interleave(2, dim=1)`), so the recurrence runs over 8 value heads.

## The number you have to beat

Target workload: `a9540651` — `total_seq_len=8192`, `num_seqs=20`, the 20
sequence lengths being
753 371 161 41 105 66 23 31 19 15 2433 39 22 3371 20 42 219 15 45 401.
Real tensors from the contest dataset, loaded by `harness/gdn_data.py`.

The kernel you have to beat is the KDA 0.5 CuTe-DSL GDN prefill kernel
(3rd place, MLSys 2026 FlashInfer contest GDN track).  It was measured on
**this GPU, in this container, with clocks locked at 1965 MHz**, using
exactly `harness/gdn_bench.py`:

    reference device time = 124.93 us  (cold-L2 protocol)
                            124.88 us  (warm-L2)

    five independent measurements: 124.971 / 124.937 / 124.937 / 124.922 us
    cold-L2, 124.877 / 124.880 us warm-L2  -> spread < 0.1 %.

**Your goal: 1.2x faster on device time, i.e. <= 104.1 us** under the same
`gdn_bench.device_time_ms(...)` call on workload `a9540651`.

`data/golden_ref.safetensors` holds that kernel's `output` and `new_state`
on this workload.

### The reference implementation is not on this machine

Only its *outputs* and its *timing* were kept.  The source was deleted before
your session started, and it is not on the machine that drives your session
either.  Do not try to reconstruct it: do not search the web for the KDA /
mit-han-lab / NVlabs kernel sources or for the contest solution repository,
and do not read any other project's Gated-Delta-Net kernel implementation.
Public documentation, papers, PTX/CUDA/CUTLASS/CuTe references, and the
KernelWiki notes are all fine — kernel implementations of *this* operator by
other people are not.  Write it yourself.

## Measuring

```python
import sys; sys.path.insert(0, "/root/gdn/harness")
import gdn_data, gdn_bench
args = gdn_data.args()                      # target workload a9540651
t_ms = gdn_bench.device_time_ms(lambda: my_run(*args),
                                warmup=10, iters=50, trials=5)
```

`device_time_ms` reports **device time**: the sum of all CUDA kernel
durations of one call, from CUPTI data via `torch.profiler`.  Host gaps are
not counted, and every kernel you launch is counted.  Cold-L2 is the default
and the cost of the L2 flush is measured separately and subtracted.

`gdn_bench.paired({...})` measures several implementations interleaved, which
is what you should use when comparing two of your own versions.

`harness/calib.py` times two fixed public workloads (a 4096^3 bf16 GEMM and a
256 MB stream add).  At anchor time it printed

    CALIB gemm_bf16_4096      85.623 us   (1605.2 TFLOP/s)
    CALIB stream_add_256MB    81.216 us   (6.61 TB/s)

Re-run it whenever you want to confirm the machine is still in the state the
reference anchor was measured in.  If those two numbers still hold, the
124.93 us anchor is still directly comparable to your measurements.

Clocks are locked (`nvidia-smi -lgc 1965`).  If anything resets them, lock
them again before trusting a measurement.

## Correctness

`harness/gdn_check.py` derives the tolerance from the data instead of
guessing it.  The reference kernel is a bf16-grade implementation already
accepted as correct for this task, so its distance from the fp32 spec is the
noise floor, and you get 7x that much room:

    floor_max = max | spec - ref |                    per output tensor
    floor_rms = rms( spec - ref ) / rms( spec )       per output tensor
    tol_*     = 7 * floor_*

    PASS  <=>  max | yours - spec | <= tol_max
          and  rms( yours - spec ) / rms(spec) <= tol_rms

Both are required: `tol_max` alone misses an error that is small everywhere
but present everywhere, `tol_rms` alone misses one very wrong element.  The
measured floors on this workload are

    output     floor_max 2.441e-04  -> tol 1.709e-03 ;  floor_rms 2.828e-03 -> tol 1.980e-02
    new_state  floor_max 1.009e-03  -> tol 7.060e-03 ;  floor_rms 1.811e-03 -> tol 1.268e-02

The contest's own criterion (atol=rtol=1e-2) is reported too, for
information; the 7x-floor rule is the one that decides.

```python
import gdn_check
gdn_check.check(out, new_state)                 # target workload
gdn_check.check_arbitrary(my_run, gdn_data.args("410794d4"))   # any other
```

`gdn_data.WORKLOADS` has five real workloads: the target plus
`77daf91d` (T=6, N=1), `fc7a2bcb` (T=139, N=3), `f4926229` (T=1377, N=1) and
`410794d4` (T=8192, N=39).  Check all of them, and also check some inputs you
generate yourself (vary seeds, sequence-length mixes, and the magnitude of
`a`/`b` so the gates `g = exp(-exp(A_log)*softplus(a+dt_bias))` and
`beta = sigmoid(b)` span their range) — a kernel that is only correct on one
shape is not correct.

## Environment

* B200 (SM100), 183 GB, clocks locked 1965 MHz, driver 580.105.08
* python: `/opt/sglang/bin/python` — torch 2.13.0+cu130, triton 3.7.1,
  nvidia-cutlass-dsl 4.6.2, cuda-python 13.3.1
* CUDA toolkit at `/usr/local/cuda` (13.0) -- **not on PATH by default**;
  `export PATH=/usr/local/cuda/bin:$PATH` gives you `nvcc`, `ncu`, `ptxas`,
  `nvdisasm`, `cuobjdump`, `compute-sanitizer`.  `nsys` is on PATH already.
* pypi is reachable; github.com and huggingface.co are **not** reachable from
  this node.  Anything you need from the web has to come in from the driving
  machine.
