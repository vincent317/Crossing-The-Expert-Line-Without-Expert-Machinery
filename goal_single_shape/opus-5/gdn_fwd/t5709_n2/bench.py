"""Device-time benchmark: CuTe-DSL reference vs Triton kernel, same data, same method.

Device time = GPU execution span of one call: CUDA events around back-to-back
replays of a captured CUDA graph, so host launch cost is excluded (the non-graph
number is printed too).  Both implementations are measured identically.

usage: python bench.py "[{'m':2,'seg':4}, ...]"
"""

import ast
import sys

import torch

from bench_util import device_time_us
from gdn_data import gates, make_inputs
from gdn_kernel import Runner, build_plan
from ref_flashinfer import make_ref_call


def main():
    cfgs = ast.literal_eval(sys.argv[1]) if len(sys.argv) > 1 else [{}]
    inp = make_inputs()
    g, beta = gates(inp["a"], inp["b"], inp["A_log"], inp["dt_bias"])
    ref_us, ref_ng, ref_tag = None, None, None
    for vname, extra in (("default", {}), ("use_cp=True", {"use_cp": True}),
                         ("use_cp=False", {"use_cp": False})):
        try:
            call, _ = make_ref_call(inp, g, beta, extra=extra)
            call()
            ng, _ = device_time_us(call, graph=False)
            us, ts = device_time_us(call, graph=True)
            print("REF(flashinfer cute-dsl gdn_prefill) %-13s device time: cudagraph %8.2f us"
                  " | no-graph %8.2f us | all %s"
                  % (vname, us, ng, ["%.2f" % t for t in ts]), flush=True)
            if ref_us is None or us < ref_us:
                ref_us, ref_ng, ref_tag = us, ng, vname
        except Exception as e:
            print("REF %-13s FAILED %r" % (vname, repr(e)[:160]), flush=True)
    if ref_us is None:
        print("WARNING: no working reference variant; reporting MINE only", flush=True)
    else:
        print("REF chosen (fastest variant) = %s : %.2f us (cudagraph)" % (ref_tag, ref_us),
              flush=True)

    best = None
    for cfg in cfgs:
        cfg = dict(cfg)
        seg = cfg.pop("seg", 4)
        mm = cfg.pop("m", 1)
        plan = build_plan(inp["cu_seqlens"], m=mm, seg=seg)
        tag = "m=%d seg=%d %s" % (mm, seg, " ".join("%s=%s" % kv for kv in sorted(cfg.items())))
        try:
            run = Runner(inp["q"], inp["k"], inp["v"], inp["a"], inp["b"], inp["A_log"],
                         inp["dt_bias"], inp["state"], plan, **cfg)
            tot, _ = device_time_us(run, graph=True)
            parts = " ".join("%s %6.2f" % (nm, device_time_us(fn, graph=True)[0])
                             for nm, fn in run.parts())
            print("MINE %-46s total %8.2f us  [%s]  %s"
                  % (tag, tot, parts,
                     ("%.3fx ref" % (ref_us / tot)) if ref_us else "(no ref)"), flush=True)
            if best is None or tot < best[0]:
                best = (tot, tag)
        except Exception as e:
            print("MINE %-46s FAILED %r" % (tag, repr(e)[:200]), flush=True)

    if best:
        print("\nBEST MINE: %.2f us (%s)" % (best[0], best[1]))
        if ref_us:
            print("REF device time %.2f us | target = ref/1.2 = %.2f us | speedup %.3fx | MET=%s"
                  % (ref_us, ref_us / 1.2, ref_us / best[0], best[0] <= ref_us / 1.2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
