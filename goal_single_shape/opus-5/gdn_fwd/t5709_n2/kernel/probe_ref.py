"""Probe the FlashInfer CuTe-DSL GDN prefill reference: public API, calling
convention (validated against the fp32 naive ground truth) and device time."""

import inspect
import itertools
import sys
import traceback

import torch

from bench_util import device_time_us
from gdn_data import err, err_finite, gates, make_inputs, naive_fp32
from ref_flashinfer import get_fn, make_ref_call


def main():
    print("=== env ===", flush=True)
    p = torch.cuda.get_device_properties(0)
    print("torch", torch.__version__, "| gpu", p.name, "cc %d.%d" % (p.major, p.minor),
          "| sms", p.multi_processor_count)
    try:
        import triton
        print("triton", triton.__version__)
    except Exception as e:
        print("triton missing:", e)
    try:
        import flashinfer
        print("flashinfer", getattr(flashinfer, "__version__", "?"))
    except Exception:
        traceback.print_exc()
        return 1
    fn = get_fn()
    print("public signature: chunk_gated_delta_rule%s" % (inspect.signature(fn),), flush=True)

    import math
    base = make_inputs()
    gscale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    inp = dict(base)
    inp["A_log"] = base["A_log"] + math.log(gscale)
    g, beta = gates(inp["a"], inp["b"], inp["A_log"], inp["dt_bias"])
    print("gate scale %.3f -> g min/mean %.3f/%.3f | per-64-token cum decay min %.3e"
          % (gscale, g.min().item(), g.mean().item(),
             float(g[:64].sum(0).min().exp())), flush=True)
    o_nv, s_nv = naive_fp32(inp)
    print("naive fp32: |o| mean %.6f max %.6f | |S| mean %.6f max %.6f"
          % (o_nv.abs().mean(), o_nv.abs().max(), s_nv.abs().mean(), s_nv.abs().max()),
          flush=True)

    print("=== calling-convention probe (vs fp32 naive) ===", flush=True)
    best = None
    grid = itertools.product((True, False), (True, False), (True, False), ("fp32", "bf16"))
    for as_is, l2, cu32, gdt in grid:
        tag = ("state_as_is=%-5s l2norm=%-5s cu_int32=%-5s gate=%s"
               % (as_is, l2, cu32, gdt))
        try:
            call, _ = make_ref_call(inp, g, beta, state_as_is=as_is, l2norm=l2,
                                    cu_int32=cu32, gate_dtype=gdt)
            out = call()
            o_rf = (out[0] if isinstance(out, (tuple, list)) else out).float()
            s_rf = out[1].float() if isinstance(out, (tuple, list)) else None
            o_rf = o_rf.reshape(o_nv.shape)
            eo = err(o_rf, o_nv)
            es = float("nan")
            if s_rf is not None:
                s_cand = s_rf if as_is else s_rf.transpose(-1, -2)
                es = err(s_cand, s_nv)
            fo, no = err_finite(o_rf, o_nv)
            fs, ns_ = err_finite(s_cand, s_nv) if s_rf is not None else (float("nan"), -1)
            print("  %s -> max|out-naive| %.6e (finite %.3e, %d nan) | max|S-naive| %.6e "
                  "(finite %.3e, %d nan)" % (tag, eo, fo, no, es, fs, ns_), flush=True)
            score = (eo if eo == eo else 1e30) + (0.0 if es != es else es)
            if best is None or score < best[0]:
                best = (score, as_is, l2, eo, es, cu32, gdt)
        except Exception as e:
            print("  %s -> FAILED %r" % (tag, repr(e)[:200]), flush=True)

    if best is None:
        print("no working calling convention found")
        return 1
    _, as_is, l2, eo, es, cu32, gdt = best
    print("CHOSEN: state_as_is=%s l2norm=%s cu_int32=%s gate=%s" % (as_is, l2, cu32, gdt))
    print("E_ref(out)   = %.6e -> tol_out   = %.6e" % (eo, 7 * eo))
    print("E_ref(state) = %.6e -> tol_state = %.6e" % (es, 7 * es), flush=True)

    print("=== ref device time (all public dispatch paths) ===", flush=True)
    variants = [("default", {}), ("use_cp=True", {"use_cp": True}),
                ("use_cp=False", {"use_cp": False})]
    for vname, extra in variants:
        try:
            call, _ = make_ref_call(inp, g, beta, state_as_is=as_is, l2norm=l2,
                                    extra=extra, cu_int32=cu32, gate_dtype=gdt)
            out = call()
            o_rf = out[0].float().reshape(o_nv.shape)
            s_cand = out[1].float()
            s_cand = s_cand if as_is else s_cand.transpose(-1, -2)
            print("  variant %-13s err(out) %.3e err(state) %.3e"
                  % (vname, err(o_rf, o_nv), err(s_cand, s_nv)), flush=True)
        except Exception as e:
            print("  variant %-13s CALL FAILED %r" % (vname, repr(e)[:160]), flush=True)
            continue
        for graph in (False, True):
            try:
                med, ts = device_time_us(call, graph=graph)
                print("    REF %-13s graph=%-5s device time median %8.2f us  all %s"
                      % (vname, graph, med, ["%.2f" % t for t in ts]), flush=True)
            except Exception as e:
                print("    REF %-13s graph=%-5s timing FAILED %r"
                      % (vname, graph, repr(e)[:160]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
