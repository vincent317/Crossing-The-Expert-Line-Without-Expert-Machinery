"""Benchmark: device time (profiler kernel span) of ref vs new kernel, same data, same method, same run."""
import sys
import traceback

import torch

import harness as H


def main():
    dev = "cuda"
    print(torch.cuda.get_device_name(), "torch", torch.__version__)
    q, kc, w, sl, bt = H.make_inputs(dev)
    flusher = H.L2Flusher(dev)

    ref_run = H.import_ref_run()
    ref_run(q, kc, w, sl, bt)  # JIT compile
    torch.cuda.synchronize()
    ref_span, ref_sum, ref_ev, _ = H.device_time_us(lambda: ref_run(q, kc, w, sl, bt), flusher, label="ref")

    new_span = None
    try:
        import topk_indexer
        topk_indexer.run(q, kc, w, sl, bt)
        torch.cuda.synchronize()
        new_span, new_sum, new_ev, _ = H.device_time_us(lambda: topk_indexer.run(q, kc, w, sl, bt), flusher,
                                                        label="new")
    except Exception:
        traceback.print_exc()

    print(f"REF device time = {ref_span:.2f} us")
    if new_span is not None:
        target = ref_span / 1.2
        print(f"NEW device time = {new_span:.2f} us   target (ref/1.2) = {target:.2f} us   "
              f"speedup = {ref_span / new_span:.3f}x  -> {'PASS' if new_span <= target else 'FAIL'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
