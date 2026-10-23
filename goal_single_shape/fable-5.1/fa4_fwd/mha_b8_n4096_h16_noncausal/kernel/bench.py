"""Benchmark: ref (flash-attn-4 cute) vs custom tcgen05 kernel, same inputs,
same timing method (CUDA events and profiler kernel-only span), same iteration
count, same job.
"""

import torch

import common as C
from build_ext import build


def main():
    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
    mod = build(verbose=False)
    from flash_attn.cute.interface import flash_attn_func

    q, k, v = C.make_inputs()

    def run_ref():
        return flash_attn_func(q, k, v, softmax_scale=C.SOFTMAX_SCALE, causal=False)

    def run_mine():
        return mod.fwd(q, k, v, C.SOFTMAX_SCALE)

    # correctness gate first
    ref_fp32 = C.attn_fp32_naive(q, k, v)
    out_ref = run_ref()
    if isinstance(out_ref, (tuple, list)):
        out_ref = out_ref[0]
    out_mine = run_mine()
    torch.cuda.synchronize()
    e_ref = C.max_abs_err(out_ref, ref_fp32)
    e_mine = C.max_abs_err(out_mine, ref_fp32)
    tol = 7.0 * e_ref
    print(f"E_ref={e_ref:.6e} tol=7*E_ref={tol:.6e} E_mine={e_mine:.6e} "
          f"accuracy_pass={e_mine <= tol}")

    ev_ref, _ = C.time_cuda_events(run_ref)
    ev_mine, _ = C.time_cuda_events(run_mine)
    print(f"[cuda_events median, warmup={C.WARMUP} iters={C.ITERS}, L2 flushed] "
          f"ref={ev_ref:.2f} us  mine={ev_mine:.2f} us  speedup={ev_ref / ev_mine:.3f}x")

    pf_ref, sp_ref = C.time_profiler_kernel_span(run_ref, tag="ref")
    pf_mine, sp_mine = C.time_profiler_kernel_span(run_mine, tag="mine")
    print(f"[profiler kernel-span median, warmup={C.WARMUP} iters={C.ITERS}, L2 flushed] "
          f"ref={pf_ref:.2f} us  mine={pf_mine:.2f} us  speedup={pf_ref / pf_mine:.3f}x")
    print(f"ref  spans min/med/max = {min(sp_ref):.2f}/{pf_ref:.2f}/{max(sp_ref):.2f} us")
    print(f"mine spans min/med/max = {min(sp_mine):.2f}/{pf_mine:.2f}/{max(sp_mine):.2f} us")
    target = pf_ref / 1.2
    print(f"TARGET mine <= ref/1.2 = {target:.2f} us ; mine = {pf_mine:.2f} us ; "
          f"GOAL_MET = {pf_mine <= target and e_mine <= tol}")


if __name__ == "__main__":
    main()
