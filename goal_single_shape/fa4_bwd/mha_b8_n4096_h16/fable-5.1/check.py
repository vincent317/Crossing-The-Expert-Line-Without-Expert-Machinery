"""Accuracy check: my Triton backward vs the naive fp32 backward.

Tolerance comes from E_ref = max|ref_bwd - naive_fp32_bwd| measured in the
same process: tol = 7 * E_ref, per gradient tensor.
"""
import torch

from common import SCALE, make_inputs
from fa_bwd import Workspace, flash_attn_bwd, forward_o_lse
from naive_ref import max_abs_err, naive_fp32_backward


def run(cfg=None, verbose=True):
    q, k, v, do = make_inputs()

    # Reference gradients (opaque callable, public API only).
    from flash_attn.cute.interface import flash_attn_func
    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = flash_attn_func(qg, kg, vg, softmax_scale=SCALE, causal=True)
    if isinstance(out, (tuple, list)):
        out = out[0]
    rdq, rdk, rdv = (t.detach() for t in torch.autograd.grad(out, (qg, kg, vg), do))
    del out, qg, kg, vg
    torch.cuda.empty_cache()

    ndq, ndk, ndv = naive_fp32_backward(q, k, v, do)

    o, lse = forward_o_lse(q, k, v, SCALE, causal=True)
    mdq, mdk, mdv = flash_attn_bwd(q, k, v, o, do, lse, SCALE, causal=True, cfg=cfg)
    torch.cuda.synchronize()

    print("\n===== accuracy vs naive fp32 (tol = 7 x E_ref) =====")
    ok = True
    for name, r, n, m in (("dq", rdq, ndq, mdq), ("dk", rdk, ndk, mdk), ("dv", rdv, ndv, mdv)):
        e_ref = max_abs_err(r, n)
        e_mine = max_abs_err(m, n)
        tol = 7 * e_ref
        good = e_mine <= tol
        ok &= good
        print(f"  {name}: E_ref = {e_ref:.6e}  tol = {tol:.6e}  E_mine = {e_mine:.6e}  "
              f"{'PASS' if good else 'FAIL'}")
    print("ACCURACY " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    run()
