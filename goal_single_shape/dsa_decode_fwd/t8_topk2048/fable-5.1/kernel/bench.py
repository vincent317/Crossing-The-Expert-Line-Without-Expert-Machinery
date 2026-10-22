"""Device-time benchmark: reference kernel vs new kernel, same data, same method.

Method (bench_common.time_call): per iteration flush L2 with a 1 GiB memset,
then CUDA events around one call; median of 100 iterations after 10 warmups.
A CUDA-graph replay variant is reported as a cross-check when capturable.
"""

import os
import sys
import traceback

import torch

from bench_common import make_flush_buffer, report, time_call, time_call_graph
from dsa_data import describe_inputs, make_inputs

TARGET_SPEEDUP = 1.2
WARMUP = 10
ITERS = 100


def main():
    inp = make_inputs("cuda")
    describe_inputs(inp)
    args = (inp["q_nope"], inp["q_pe"], inp["ckv_cache"], inp["kpe_cache"], inp["sparse_indices"], inp["sm_scale"])
    flush = make_flush_buffer()

    import dsa_kernel

    print(f"[bench] new kernel gather mode: {dsa_kernel.GATHER_MODE} (0 = gather4, 1 = per-row tiles)")

    def new_fn():
        return dsa_kernel.run(*args)

    ref_fn = None
    try:
        sys.path.insert(0, ".")
        import ref_kernel  # KDA-0.5 CuTe-DSL kernel, imported only, never read

        def ref_fn():
            return ref_kernel.run(*args)

    except Exception as exc:
        print(f"[bench] reference kernel unavailable: {type(exc).__name__}: {exc}")

    med0, mn0, times0 = time_call(lambda: dsa_kernel.noop(296, 48 << 10), flush, warmup=WARMUP, iters=ITERS)
    report("NOOP empty kernel (timing-method fixed overhead)", med0, mn0, times0)

    results = {}
    for label, fn in (("REF", ref_fn), ("NEW", new_fn)):
        if fn is None:
            continue
        try:
            med, mn, times = time_call(fn, flush, warmup=WARMUP, iters=ITERS)
            report(f"{label} device time (cuda events, cold L2, direct call)", med, mn, times)
            results[label] = med
        except Exception:
            print(f"[bench] {label} direct timing FAILED:")
            traceback.print_exc()
        try:
            med_g, mn_g, times_g = time_call_graph(fn, flush, warmup=WARMUP, iters=ITERS)
            report(f"{label} device time (cuda events, cold L2, cuda graph)", med_g, mn_g, times_g)
            results[label + "_graph"] = med_g
        except Exception as exc:
            print(f"[bench] {label} cuda graph timing unavailable: {type(exc).__name__}: {exc}")

    if "REF" in results and "NEW" in results:
        ref_us, new_us = results["REF"], results["NEW"]
        target = ref_us / TARGET_SPEEDUP
        print(f"[bench] REF={ref_us:.3f} us  NEW={new_us:.3f} us  speedup={ref_us / new_us:.3f}x  "
              f"target NEW <= {target:.3f} us -> {'PASS' if new_us <= target else 'FAIL'}")
    if "REF_graph" in results and "NEW_graph" in results:
        ref_us, new_us = results["REF_graph"], results["NEW_graph"]
        print(f"[bench] (graph) REF={ref_us:.3f} us  NEW={new_us:.3f} us  speedup={ref_us / new_us:.3f}x  "
              f"target NEW <= {ref_us / TARGET_SPEEDUP:.3f} us -> {'PASS' if new_us <= ref_us / TARGET_SPEEDUP else 'FAIL'}")

    if os.environ.get("DSA_PROFILE", "0") == "1":
        dsa_kernel.profile(*args, flush_buf=flush)


if __name__ == "__main__":
    main()
