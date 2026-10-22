"""Accuracy check: fp32/fp64 naive vs reference kernel (E_ref, tol = 7*E_ref) vs new kernel.

Prints every number; exit code 1 if the new kernel misses the tolerance.
Reference kernel (ref_kernel.py) is optional: if absent, only the new kernel
vs naive numbers are printed.
"""

import sys
import time
import traceback

import torch

from check_common import max_errors
from dsa_data import describe_inputs, make_inputs
from dsa_naive import naive_attention

TOL_FACTOR = 7.0


def try_import_ref():
    try:
        sys.path.insert(0, ".")
        import ref_kernel  # KDA-0.5 CuTe-DSL kernel, imported only, never read

        return ref_kernel
    except Exception as exc:
        print(f"[check] reference kernel unavailable: {type(exc).__name__}: {exc}")
        return None


def recombine_check(dsa_kernel, inp, naive32):
    """Recombine the kernel's published partials on the host; localises errors to items vs reduction."""
    s = dsa_kernel._get_scratch(inp["q_nope"].shape[0], inp["q_nope"].device)
    m = s["part_m"].float()  # [T][C][H]
    l = s["part_l"].float()
    o = dsa_kernel.partial_o_thd(s).float()  # [T][C][H][D] normalised partials
    live = m > -float("inf")
    M = torch.where(live, m, torch.full_like(m, -float("inf"))).amax(dim=1, keepdim=True)  # [T][1][H]
    w = torch.where(live, torch.exp2(m - M) * l, torch.zeros_like(m))  # [T][C][H]
    L = w.sum(dim=1)  # [T][H]
    out = torch.einsum("tch,tchd->thd", w, torch.nan_to_num(o)) / L.unsqueeze(-1)
    lse = M.squeeze(1) + torch.log2(L)
    e_o, e_l, _ = max_errors((out, lse), *naive32)
    print(f"[check] host recombination of kernel partials vs fp32-naive: max|d output|={e_o:.6e} max|d lse|={e_l:.6e}; "
          f"live chunks per token={live[..., 0].sum(dim=1).tolist()}")


def main():
    inp = make_inputs("cuda")
    describe_inputs(inp)
    naive32 = naive_attention(**inp, dtype=torch.float32)
    naive64 = naive_attention(**inp, dtype=torch.float64)
    torch.cuda.synchronize()
    e_out, e_lse, _ = max_errors(naive32, *naive64)
    print(f"[check] fp32-naive vs fp64-naive: max|d output|={e_out:.6e}  max|d lse|={e_lse:.6e}")

    args = (inp["q_nope"], inp["q_pe"], inp["ckv_cache"], inp["kpe_cache"], inp["sparse_indices"], inp["sm_scale"])

    tol_out = tol_lse = None
    ref = try_import_ref()
    if ref is not None:
        try:
            t0 = time.time()
            res = ref.run(*args)
            torch.cuda.synchronize()
            print(f"[check] ref first call {time.time() - t0:.1f}s")
            e_ref_out, e_ref_lse, nans = max_errors(res, *naive32)
            tol_out, tol_lse = TOL_FACTOR * e_ref_out, TOL_FACTOR * e_ref_lse
            print(f"[check] E_ref (ref vs fp32-naive): max|d output|={e_ref_out:.6e}  max|d lse|={e_ref_lse:.6e}  nan={nans}")
            print(f"[check] tol = {TOL_FACTOR:g} * E_ref: tol_out={tol_out:.6e}  tol_lse={tol_lse:.6e}")
        except Exception:
            print("[check] reference kernel FAILED:")
            traceback.print_exc()

    import dsa_kernel

    t0 = time.time()
    mine = dsa_kernel.run(*args)
    torch.cuda.synchronize()
    print(f"[check] new kernel first call (includes build) {time.time() - t0:.1f}s")
    recombine_check(dsa_kernel, inp, naive32)
    # run again: monotonic counters must keep working and results must be stable
    mine2 = dsa_kernel.run(*args)
    torch.cuda.synchronize()
    same = torch.equal(mine[0], mine2[0]) and torch.equal(mine[1], mine2[1])
    print(f"[check] new kernel repeat-call bitwise identical: {same}")
    for name, res in (("launch1", mine), ("launch2", mine2)):
        e_o, e_l, _ = max_errors(res, *naive32)
        per_tok = (res[0].float() - naive32[0]).abs().amax(dim=(1, 2)).tolist()
        print(f"[check] {name}: max|d output|={e_o:.6e} max|d lse|={e_l:.6e} per-token={['%.2e' % v for v in per_tok]}")
    e_my_out, e_my_lse, nans = max_errors(mine, *naive32)
    print(f"[check] NEW vs fp32-naive: max|d output|={e_my_out:.6e}  max|d lse|={e_my_lse:.6e}  nan={nans}")
    e64_out, e64_lse, _ = max_errors(mine, *naive64)
    print(f"[check] NEW vs fp64-naive: max|d output|={e64_out:.6e}  max|d lse|={e64_lse:.6e}")

    ok = same and nans == 0
    if tol_out is not None:
        ok_out = e_my_out <= tol_out
        ok_lse = e_my_lse <= tol_lse
        print(f"[check] output: {e_my_out:.6e} <= tol {tol_out:.6e} -> {'PASS' if ok_out else 'FAIL'}")
        print(f"[check] lse:    {e_my_lse:.6e} <= tol {tol_lse:.6e} -> {'PASS' if ok_lse else 'FAIL'}")
        ok = ok and ok_out and ok_lse
    print(f"[check] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
