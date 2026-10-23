"""Accuracy check for the GDN forward kernel.

  E_ref = max |ref - fp32_naive|        reference vs the slow fp32 oracle
  tol   = 7 * E_ref
  PASS iff max |new_kernel - fp32_naive| <= tol   (checked per tensor)

Usage: python check.py [triton]
"""

import sys

import torch

import common as C


TRITON_CFG = dict(BV=16, num_warps=4)


def impl(name="cuda"):
    if name == "triton":
        import kernel
        return "triton", kernel.gdn_fwd, dict(TRITON_CFG)
    import kernel_cuda
    return "cuda", kernel_cuda.gdn_fwd, {}


def err(x, y):
    return (x.float() - y.float()).abs().max().item()


def run(inp=None, which="cuda", verbose=True):
    inp = inp or C.make_inputs()
    name, fwd, kw = impl(which)
    g, beta = C.gates_fp32(inp["a"], inp["b"], inp["A_log"], inp["dt_bias"])

    o_nav, s_nav = C.naive_fp32(inp)
    if verbose:
        print(f"fp32 naive oracle : out absmax {o_nav.abs().max():.6f}  "
              f"state absmax {s_nav.abs().max():.6f}")

    res = {}
    try:
        import refmod
        ref_fn, label = refmod.make_any_ref(inp, g, beta)
        o_ref, s_ref = ref_fn()
        torch.cuda.synchronize()
        res["ref_label"] = label
        res["E_ref_o"], res["E_ref_s"] = err(o_ref, o_nav), err(s_ref, s_nav)
        print(f"reference         : {label}")
        if res["E_ref_o"] > 0.05:      # convention mismatch, not rounding
            print("  !! reference disagrees with the oracle; probing conventions")
            o_mod, s_mod = C.naive_fp32(inp, qk_map="mod")
            o_ns, s_ns = C.naive_fp32(inp, scale=1.0)
            o_nl, s_nl = C.naive_fp32(inp, l2norm=False)
            for nm, (oo, ss) in {"qk_map=h//2 scale=1/sqrt(D)": (o_nav, s_nav),
                                 "qk_map=h%4  scale=1/sqrt(D)": (o_mod, s_mod),
                                 "qk_map=h//2 scale=1.0": (o_ns, s_ns),
                                 "qk_map=h//2 no l2norm": (o_nl, s_nl)}.items():
                print(f"     {nm:30s} out {err(o_ref, oo):.6e}  state {err(s_ref, ss):.6e}")
    except Exception as exc:  # noqa: BLE001
        print("!! reference unavailable:", type(exc).__name__, str(exc)[:400])
        res["E_ref_o"] = res["E_ref_s"] = float("nan")

    try:
        o_new, s_new = fwd(inp["q"], inp["k"], inp["v"], inp["a"], inp["b"],
                           inp["A_log"], inp["dt_bias"], inp["state"], **kw)
        torch.cuda.synchronize()
        res["E_new_o"], res["E_new_s"] = err(o_new, o_nav), err(s_new, s_nav)
    except Exception as exc:  # noqa: BLE001
        print(f"!! {name} kernel failed:", type(exc).__name__, str(exc)[:300])
        res["E_new_o"] = res["E_new_s"] = float("inf")

    res["E_ref"] = max(res["E_ref_o"], res["E_ref_s"])
    res["E_new"] = max(res["E_new_o"], res["E_new_s"])
    res["tol"] = 7.0 * res["E_ref"]
    res["tol_o"], res["tol_s"] = 7.0 * res["E_ref_o"], 7.0 * res["E_ref_s"]
    res["pass"] = bool(res["E_new_o"] <= res["tol_o"] and res["E_new_s"] <= res["tol_s"])

    if verbose:
        print(f"\n=== ACCURACY ({name} kernel; everything vs the same fp32 oracle, same data) ===")
        print(f"  E_ref  = max|ref  - naive| : out {res['E_ref_o']:.6e}   state {res['E_ref_s']:.6e}"
              f"   max {res['E_ref']:.6e}")
        print(f"  tol    = 7 * E_ref         : out {res['tol_o']:.6e}   state {res['tol_s']:.6e}"
              f"   max {res['tol']:.6e}")
        print(f"  E_new  = max|mine - naive| : out {res['E_new_o']:.6e}   state {res['E_new_s']:.6e}"
              f"   max {res['E_new']:.6e}")
        print(f"  out  : {res['E_new_o']:.3e} <= {res['tol_o']:.3e} ? {res['E_new_o'] <= res['tol_o']}")
        print(f"  state: {res['E_new_s']:.3e} <= {res['tol_s']:.3e} ? {res['E_new_s'] <= res['tol_s']}")
        print(f"  ACCURACY {'PASS' if res['pass'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    run(which="triton" if "triton" in sys.argv else "cuda")
