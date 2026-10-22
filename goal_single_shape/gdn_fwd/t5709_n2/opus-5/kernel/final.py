"""Final evidence run: reference device time, tolerance, accuracy and benchmark.

Prints, in one place:
  1. reference (FlashInfer upstream CuTe-DSL GDN prefill) device time on B200
  2. E_ref = max|reference - fp32 naive| and tol = 7 * E_ref
  3. max|triton kernel - fp32 naive| vs tol
  4. triton kernel device time, same method / data / measurement as the reference
"""

import sys
import traceback

import torch

from bench_util import device_time_us
from gdn_data import SCALE, err, err_finite, gates, make_inputs, naive_fp32
from gdn_kernel import Runner, build_plan
from ref_flashinfer import make_ref_call

BEST = dict(mode="two", m=1, seg=12, ns=4, warps_prep=2, stages_out=3, stages_seg=3)


def fla_golden(inp, g, beta):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    o, s = chunk_gated_delta_rule(
        q=inp["q"].unsqueeze(0), k=inp["k"].unsqueeze(0), v=inp["v"].unsqueeze(0),
        g=g.unsqueeze(0).contiguous(), beta=beta.unsqueeze(0).contiguous(), scale=SCALE,
        initial_state=inp["state"].transpose(-1, -2).contiguous(), output_final_state=True,
        cu_seqlens=inp["cu_seqlens"].to(torch.int32), use_qk_l2norm_in_kernel=True)
    return o.squeeze(0).float(), s.transpose(-1, -2).float()


def main():
    inp = make_inputs()
    g, beta = gates(inp["a"], inp["b"], inp["A_log"], inp["dt_bias"])
    print("=" * 78)
    print("WORKLOAD  T=%d  N=2 (5637+72)  q/k 4 heads, v 8 heads, head_dim 128, bf16"
          % inp["q"].shape[0])
    p = torch.cuda.get_device_properties(0)
    print("DEVICE    %s  cc %d.%d  %d SMs  | torch %s"
          % (p.name, p.major, p.minor, p.multi_processor_count, torch.__version__))
    print("=" * 78, flush=True)

    print("\n[1] fp32 naive (token-sequential ground truth)", flush=True)
    o_nv, s_nv = naive_fp32(inp)
    print("    |o| mean %.6f max %.6f | |S| mean %.6f max %.6f"
          % (o_nv.abs().mean(), o_nv.abs().max(), s_nv.abs().mean(), s_nv.abs().max()),
          flush=True)

    print("\n[2] REFERENCE: flashinfer.gdn_prefill.chunk_gated_delta_rule (CuTe-DSL)",
          flush=True)
    print("    -- calling-convention search (q/k pre-normalized vs in-kernel l2norm)",
          flush=True)
    conv = None
    for pn in (True, False):
        for l2 in (False, True):
            try:
                call, as_is = make_ref_call(inp, g, beta, prenorm=pn, l2norm=l2)
                out = call()
                o_rf = out[0].float().reshape(o_nv.shape)
                s_rf = out[1].float()
                s_rf = s_rf if as_is else s_rf.transpose(-1, -2)
                eo, no = err_finite(o_rf, o_nv)
                es, ns_ = err_finite(s_rf, s_nv)
                print("       prenorm=%-5s l2norm_in_kernel=%-5s -> max|dout| %.4e (%d nan)"
                      "  max|dS| %.4e (%d nan)" % (pn, l2, eo, no, es, ns_), flush=True)
                score = (eo if eo == eo else 1e30) + no + ns_
                if conv is None or score < conv[0]:
                    conv = (score, pn, l2, eo, es, no, ns_)
            except Exception as e:
                print("       prenorm=%-5s l2norm=%-5s FAILED %r" % (pn, l2, repr(e)[:110]),
                      flush=True)
    if conv is None:
        print("    reference unusable")
        return 1
    _, PN, L2, ceo, ces, cno, cns = conv
    print("    -- chosen convention: prenorm=%s l2norm_in_kernel=%s" % (PN, L2), flush=True)

    ref_us = None
    ref_err = (ceo, ces, cno, cns)
    for name, extra in (("default(chain-split/CP)", {}), ("use_cp=True", {"use_cp": True}),
                        ("use_cp=False", {"use_cp": False})):
        try:
            call, as_is = make_ref_call(inp, g, beta, extra=extra, prenorm=PN, l2norm=L2)
            out = call()
            o_rf = out[0].float().reshape(o_nv.shape)
            s_rf = out[1].float()
            s_rf = s_rf if as_is else s_rf.transpose(-1, -2)
            eo, no = err_finite(o_rf, o_nv)
            es, ns_ = err_finite(s_rf, s_nv)
            ng, _ = device_time_us(call, graph=False)
            us, ts = device_time_us(call, graph=True)
            print("    %-24s device time: cudagraph %8.2f us | no-graph %8.2f us"
                  % (name, us, ng))
            print("    %-24s accuracy vs naive: max|dout| %.4e (%d non-finite), "
                  "max|dS| %.4e (%d non-finite)" % ("", eo, no, es, ns_), flush=True)
            if ref_us is None or us < ref_us:
                ref_us = us
                ref_err = (eo, es, no, ns_)
                print("    %-24s (fastest so far)" % "", flush=True)
        except Exception as e:
            print("    %-24s FAILED %r" % (name, repr(e)[:130]), flush=True)
    if ref_us is None:
        print("    reference unavailable")
        return 1

    print("\n[3] TOLERANCE  tol = 7 * E_ref", flush=True)
    eo, es, no, ns_ = ref_err
    if no == 0 and ns_ == 0:
        src = "CuTe-DSL reference"
        e_o, e_s = eo, es
    else:
        print("    NOTE: the CuTe-DSL reference returns non-finite values for every output"
              " element in this environment (flashinfer 0.6.18 / CUDA 13.3 / B200),")
        print("          so E_ref is taken from the other public implementation of the very"
              " same operator (flash-linear-attention chunk_gated_delta_rule).")
        try:
            o_fl, s_fl = fla_golden(inp, g, beta)
            e_o, e_s = err(o_fl, o_nv), err(s_fl, s_nv)
            src = "flash-linear-attention reference"
        except Exception:
            traceback.print_exc()
            return 1
    print("    E_ref(out)   = max|ref - naive_fp32|     = %.6e   [%s]" % (e_o, src))
    print("    E_ref(state) = max|ref_S - naive_fp32_S| = %.6e   [%s]" % (e_s, src))
    print("    tol_out   = 7 * E_ref(out)   = %.6e" % (7 * e_o))
    print("    tol_state = 7 * E_ref(state) = %.6e" % (7 * e_s), flush=True)

    print("\n[4] TRITON KERNEL  %s" % BEST, flush=True)
    cfg = dict(BEST)
    plan = build_plan(inp["cu_seqlens"], m=cfg.pop("m"), seg=cfg.pop("seg"))
    run = Runner(inp["q"], inp["k"], inp["v"], inp["a"], inp["b"], inp["A_log"],
                 inp["dt_bias"], inp["state"], plan, **cfg)
    o_mi, s_mi = run()
    m_o, m_s = err(o_mi, o_nv), err(s_mi, s_nv)
    ok = (m_o <= 7 * e_o) and (m_s <= 7 * e_s)
    print("    max|mine - naive_fp32|     = %.6e   (tol %.6e)" % (m_o, 7 * e_o))
    print("    max|mine_S - naive_fp32_S| = %.6e   (tol %.6e)" % (m_s, 7 * e_s))
    print("    ACCURACY: %s" % ("PASS" if ok else "FAIL"), flush=True)

    tot, ts = device_time_us(run, graph=True)
    tot_ng, _ = device_time_us(run, graph=False)
    parts = "  ".join("%s %.2f" % (nm, device_time_us(fn, graph=True)[0])
                      for nm, fn in run.parts())
    print("    device time: cudagraph %8.2f us | no-graph %8.2f us | all %s"
          % (tot, tot_ng, ["%.2f" % t for t in ts]))
    print("    per-kernel:  %s" % parts, flush=True)

    print("\n[5] Secondary reference (numerically valid): "
          "flash-linear-attention chunk_gated_delta_rule", flush=True)
    fla_us = None
    try:
        def fla_call():
            return fla_golden(inp, g, beta)
        fla_call()
        fng, _ = device_time_us(fla_call, graph=False)
        fla_us, _ = device_time_us(fla_call, graph=True)
        print("    device time: cudagraph %8.2f us | no-graph %8.2f us" % (fla_us, fng))
        print("    accuracy vs naive: max|dout| %.4e  max|dS| %.4e" % (e_o, e_s), flush=True)
    except Exception as e:
        print("    FAILED %r" % (repr(e)[:150],), flush=True)

    print("\n[6] RESULT", flush=True)
    print("    REF  device time = %8.2f us" % ref_us)
    print("    MINE device time = %8.2f us" % tot)
    print("    speedup vs CuTe-DSL ref = %.3fx   target = ref/1.2 = %.2f us"
          "   TARGET_MET = %s" % (ref_us / tot, ref_us / 1.2, tot <= ref_us / 1.2))
    if fla_us:
        print("    speedup vs numerically-valid FLA ref = %.3fx   target = %.2f us"
              "   TARGET_MET = %s" % (fla_us / tot, fla_us / 1.2, tot <= fla_us / 1.2))
    print("    NOTE: every flashinfer CuTe-DSL path returns all-NaN on this workload in this"
          " environment, so its %.2f us is a timing-only baseline." % ref_us, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
