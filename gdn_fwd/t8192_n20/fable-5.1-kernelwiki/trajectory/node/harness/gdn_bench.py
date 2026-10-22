"""Device-time measurement for the GDN prefill task.

`device_time_ms(fn)` returns the *device* time of one call of `fn`: the sum of
all CUDA kernel durations attributed to that call, measured with the CUPTI
kernel-span data that torch.profiler collects.  Host-side Python/launch gaps
are NOT counted, so a launch-bound implementation is not punished and a
multi-kernel implementation is charged for every kernel it launches.

Protocol (matches the contest's cold-L2 kernel-span protocol):
  * before every iteration the L2 is flushed by touching a 512 MB buffer;
  * the cost of the flush itself is measured in a second, otherwise identical
    profiled loop and subtracted, so it never lands in the reported number;
  * `trials` independent repeats, median reported.

Use `paired(fns)` to measure several implementations interleaved (A B A B ...)
so that clock/thermal drift cancels instead of favouring whoever ran first.
"""
import json
import os
import statistics
import tempfile

import torch

_L2_BYTES = 512 * 1024 * 1024
_flush_buf = None


def _flush():
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(_L2_BYTES // 4, dtype=torch.int32, device="cuda")
    _flush_buf.zero_()


def _kernel_us(prof):
    """Total CUDA kernel time (us) in a finished profile, from the trace."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            trace = json.load(f)
    finally:
        os.unlink(path)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    total = 0.0
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            total += float(e.get("dur", 0.0))
    return total


def _profiled_us(body, iters):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            body()
        torch.cuda.synchronize()
    return _kernel_us(prof)


def device_time_ms(fn, warmup=5, iters=50, trials=3, cold_l2=True):
    """Median device time of one `fn()` call, in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    def body_full():
        if cold_l2:
            _flush()
        fn()

    def body_flush():
        _flush()

    out = []
    for _ in range(trials):
        tot = _profiled_us(body_full, iters)
        if cold_l2:
            tot -= _profiled_us(body_flush, iters)
        out.append(tot / iters / 1000.0)
    return statistics.median(out)


def paired(fns, warmup=5, iters=50, trials=3, cold_l2=True):
    """fns: dict name -> callable.  Interleaved A/B/A/B measurement.

    Returns dict name -> median device time (ms).
    """
    for f in fns.values():
        for _ in range(warmup):
            f()
    torch.cuda.synchronize()

    samples = {n: [] for n in fns}
    for _ in range(trials):
        for name, f in fns.items():
            def body(f=f):
                if cold_l2:
                    _flush()
                f()
            tot = _profiled_us(body, iters)
            if cold_l2:
                tot -= _profiled_us(_flush, iters)
            samples[name].append(tot / iters / 1000.0)
    return {n: statistics.median(v) for n, v in samples.items()}, samples
