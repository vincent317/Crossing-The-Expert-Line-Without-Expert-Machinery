"""Measure the reference flash-attn-4 cute kernel device time on B200,
and establish the accuracy tolerance from the fp32 naive implementation.
"""

import torch

import common as C


def main():
    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
    import flash_attn
    print("flash_attn version", getattr(flash_attn, "__version__", "?"))
    from flash_attn.cute.interface import flash_attn_func

    q, k, v = C.make_inputs()
    print("shapes", tuple(q.shape), q.dtype, "softmax_scale", C.SOFTMAX_SCALE)

    def run_ref():
        return flash_attn_func(q, k, v, softmax_scale=C.SOFTMAX_SCALE, causal=False)

    out_ref = run_ref()
    if isinstance(out_ref, (tuple, list)):
        out_ref = out_ref[0]
    torch.cuda.synchronize()
    print("ref out", tuple(out_ref.shape), out_ref.dtype)

    ref_fp32 = C.attn_fp32_naive(q, k, v)
    e_ref = C.max_abs_err(out_ref, ref_fp32)
    tol = 7.0 * e_ref
    print(f"E_ref (ref vs fp32_naive max abs err) = {e_ref:.6e}")
    print(f"tol = 7 * E_ref = {tol:.6e}")

    ev_med, _ = C.time_cuda_events(run_ref)
    print(f"REF cuda_events median device time = {ev_med:.2f} us")

    pf_med, spans = C.time_profiler_kernel_span(run_ref, tag="ref")
    print(f"REF profiler kernel-span median device time = {pf_med:.2f} us")
    print(f"REF profiler spans min/median/max = {min(spans):.2f}/{pf_med:.2f}/{max(spans):.2f} us")
    print(f"TARGET (ref/1.2) = {pf_med / 1.2:.2f} us")


if __name__ == "__main__":
    main()
