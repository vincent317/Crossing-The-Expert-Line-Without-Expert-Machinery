"""Accuracy across several inputs, not just one seed.

Lazy rescale (RESCALE_THRESH) lets P grow to 2^thresh before O is rescaled, so
the error depends on each row's running-max trajectory and therefore on the
data. A single seed does not establish that the kernel stays inside tolerance,
so this sweeps seeds and also amplifies q/k to force the row max to move more
often (more rescales, larger P).
"""

import torch

import common as C
from build_ext import build


def main():
    mod = build(verbose=False)
    from flash_attn.cute.interface import flash_attn_func

    worst_ratio, worst_case = 0.0, None
    ok = True
    for seed in (1234, 7, 20260923, 99991):
        for scale in (1.0, 4.0):
            q, k, v = C.make_inputs(seed=seed)
            if scale != 1.0:
                q = (q.float() * scale).to(q.dtype)
                k = (k.float() * scale).to(k.dtype)
            ref32 = C.attn_fp32_naive(q, k, v)
            out_ref = flash_attn_func(q, k, v, softmax_scale=C.SOFTMAX_SCALE,
                                      causal=False)
            if isinstance(out_ref, (tuple, list)):
                out_ref = out_ref[0]
            e_ref = C.max_abs_err(out_ref, ref32)
            tol = 7.0 * e_ref
            out = mod.fwd(q, k, v, C.SOFTMAX_SCALE)
            torch.cuda.synchronize()
            e = C.max_abs_err(out, ref32)
            nan = torch.isnan(out).sum().item()
            over = ((out.float() - ref32).abs() > tol).sum().item()
            ratio = e / tol
            passed = e <= tol and nan == 0 and over == 0
            ok &= passed
            if ratio > worst_ratio:
                worst_ratio, worst_case = ratio, (seed, scale)
            print(f"seed={seed:<9} qk_scale={scale:<4} E_ref={e_ref:.6e} "
                  f"tol={tol:.6e} E_mine={e:.6e} ratio={ratio:.3f} "
                  f"nan={nan} over_tol={over} {'PASS' if passed else 'FAIL'}")
    print(f"worst E_mine/tol = {worst_ratio:.3f} at seed/scale {worst_case}")
    print("SEED_SWEEP", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
