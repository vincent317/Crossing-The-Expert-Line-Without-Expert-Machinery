"""Accuracy check: fp32 naive (ground truth) vs CuTe-DSL reference vs Triton kernel.

Tolerance comes from the reference itself: tol = 7 * E_ref with
E_ref = max|ref - naive_fp32|, computed separately for the output tensor and the
final state (their magnitudes differ by ~10x).

usage: python check.py [seg] [m]
"""

import sys

import torch

from gdn_data import err, gates, make_inputs, naive_fp32
from gdn_kernel import Runner, build_plan
from ref_flashinfer import make_ref_call


def main():
    seg = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    mm = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    mode = sys.argv[3] if len(sys.argv) > 3 else "two"
    fa = (sys.argv[4] != "0") if len(sys.argv) > 4 else True
    inp = make_inputs()
    g, beta = gates(inp["a"], inp["b"], inp["A_log"], inp["dt_bias"])

    o_nv, s_nv = naive_fp32(inp)
    print("naive fp32 (ground truth): |o| mean %.6f max %.6f | |S| mean %.6f max %.6f"
          % (o_nv.abs().mean(), o_nv.abs().max(), s_nv.abs().mean(), s_nv.abs().max()),
          flush=True)

    out, as_is = None, True
    for extra in ({}, {"use_cp": True}, {"use_cp": False}):
        try:
            call, as_is = make_ref_call(inp, g, beta, extra=extra)
            out = call()
            print("ref variant used: %s" % (extra or "default",))
            break
        except Exception as e:
            print("ref variant %s failed: %r" % (extra or "default", repr(e)[:140]))
    if out is None:
        raise SystemExit("no working reference variant")
    o_rf = out[0].float().reshape(o_nv.shape)
    s_rf = out[1].float()
    s_rf = s_rf if as_is else s_rf.transpose(-1, -2)
    e_ref_o = err(o_rf, o_nv)
    e_ref_s = err(s_rf, s_nv)
    tol_o, tol_s = 7 * e_ref_o, 7 * e_ref_s
    print("E_ref(out)   = max|ref - naive_fp32|      = %.6e  -> tol_out   = 7*E = %.6e"
          % (e_ref_o, tol_o))
    print("E_ref(state) = max|ref_S - naive_fp32_S|  = %.6e  -> tol_state = 7*E = %.6e"
          % (e_ref_s, tol_s), flush=True)

    plan = build_plan(inp["cu_seqlens"], m=mm, seg=seg)
    run = Runner(inp["q"], inp["k"], inp["v"], inp["a"], inp["b"],
                 inp["A_log"], inp["dt_bias"], inp["state"], plan, mode=mode, fused_attn=fa)
    o_mi, s_mi = run()
    print("config: seg=%d m=%d mode=%s fused_attn=%s nblk=%d nseg=%d"
          % (seg, mm, mode, fa, plan["nblk"], plan["nseg"]))
    e_o = err(o_mi, o_nv)
    e_s = err(s_mi, s_nv)
    print("MINE(out)    = max|mine - naive_fp32|     = %.6e  (%.3fx of E_ref)"
          % (e_o, e_o / max(e_ref_o, 1e-30)))
    print("MINE(state)  = max|mine_S - naive_fp32_S| = %.6e  (%.3fx of E_ref)"
          % (e_s, e_s / max(e_ref_s, 1e-30)))
    ok = (e_o <= tol_o) and (e_s <= tol_s)
    print("ACCURACY %s: out %.3e <= %.3e  state %.3e <= %.3e"
          % ("PASS" if ok else "FAIL", e_o, tol_o, e_s, tol_s), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
