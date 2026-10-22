"""B200 job: naive fp32/fp64 reference, KDA-0.5 reference kernel accuracy + device time.

The reference kernel module (ref_kernel.py) is shipped alongside verbatim and
only imported; it is not part of this repository.
"""

import sys
import time
import traceback

import torch

from bench_common import make_flush_buffer, report, time_call, time_call_graph
from check_common import max_errors
from dsa_data import describe_inputs, make_inputs
from dsa_naive import naive_attention


def main():
    torch.cuda.init()
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
    inp = make_inputs("cuda")
    describe_inputs(inp)

    t0 = time.time()
    naive_out32, naive_lse32 = naive_attention(**inp, dtype=torch.float32)
    naive_out64, naive_lse64 = naive_attention(**inp, dtype=torch.float64)
    torch.cuda.synchronize()
    print(f"[naive] fp32/fp64 naive done in {time.time() - t0:.1f}s")
    e_out, e_lse, _ = max_errors((naive_out32, naive_lse32), naive_out64, naive_lse64)
    print(f"[naive] fp32-naive vs fp64-naive: max|d output|={e_out:.3e}  max|d lse|={e_lse:.3e}")

    flush = make_flush_buffer()

    try:
        sys.path.insert(0, ".")
        import ref_kernel  # KDA-0.5 CuTe-DSL kernel, imported only

        def ref_fn():
            return ref_kernel.run(
                inp["q_nope"], inp["q_pe"], inp["ckv_cache"], inp["kpe_cache"], inp["sparse_indices"], inp["sm_scale"]
            )

        t0 = time.time()
        res = ref_fn()
        torch.cuda.synchronize()
        print(f"[ref] first call (includes JIT) {time.time() - t0:.1f}s; result type={type(res).__name__}")
        e_out, e_lse, nans = max_errors(res, naive_out32, naive_lse32)
        print(f"[ref] E_ref vs fp32-naive: max|d output|={e_out:.6e}  max|d lse|={e_lse:.6e}  nan={nans}")
        print(f"[ref] tolerance tol_out=7*E_ref_out={7 * e_out:.6e}  tol_lse=7*E_ref_lse={7 * e_lse:.6e}")
        e_out64, e_lse64, _ = max_errors(res, naive_out64, naive_lse64)
        print(f"[ref] vs fp64-naive: max|d output|={e_out64:.6e}  max|d lse|={e_lse64:.6e}")

        med, mn, times = time_call(ref_fn, flush, warmup=10, iters=100)
        report("REF device time (cuda events, cold L2, direct call)", med, mn, times)
        try:
            med_g, mn_g, times_g = time_call_graph(ref_fn, flush, warmup=10, iters=100)
            report("REF device time (cuda events, cold L2, cuda graph)", med_g, mn_g, times_g)
        except Exception as exc:  # graph capture is optional
            print(f"[ref] cuda graph timing unavailable: {type(exc).__name__}: {exc}")
    except Exception:
        print("[ref] FAILED:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
