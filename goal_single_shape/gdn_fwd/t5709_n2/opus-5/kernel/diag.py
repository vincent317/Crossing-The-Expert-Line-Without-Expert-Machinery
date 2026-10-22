"""Compiled-kernel diagnostics: registers, spills, shared memory, MMA instruction mix."""

import re
import sys

import torch

import gdn_kernel as gk
from gdn_data import make_inputs


def compiled_kernels(jitfn):
    """Yield CompiledKernel objects from a Triton JITFunction cache (version tolerant)."""
    found = []
    for attr in ("device_caches", "cache"):
        obj = getattr(jitfn, attr, None)
        if obj is None:
            continue
        for val in (obj.values() if hasattr(obj, "values") else []):
            cands = val if isinstance(val, (list, tuple)) else [val]
            for cand in cands:
                if hasattr(cand, "values"):
                    for ck in cand.values():
                        if hasattr(ck, "metadata"):
                            found.append(ck)
        if found:
            break
    return found


def report(name, jitfn):
    cks = compiled_kernels(jitfn)
    if not cks:
        print("%-12s <no compiled kernel found in cache>" % name)
        return
    for ck in cks:
        md = ck.metadata
        ptx = ""
        try:
            ptx = ck.asm.get("ptx", "") or ""
        except Exception:
            pass
        mma = {}
        for pat in ("tcgen05.mma", "wgmma.mma_async", "mma.sync.aligned.m16n8k16",
                    "mma.sync", "ldmatrix", "stmatrix", "cp.async"):
            n = ptx.count(pat)
            if n:
                mma[pat] = n
        print("%-12s regs=%-4s spills=%-4s shared=%-7s warps=%-2s  %s"
              % (name, getattr(md, "num_regs", "?"), getattr(md, "num_spills", "?"),
                 getattr(md, "shared", "?"), getattr(md, "num_warps", "?"), mma))


def main():
    seg = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    mm = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    inp = make_inputs()
    plan = gk.build_plan(inp["cu_seqlens"], m=mm, seg=seg)
    run = gk.Runner(inp["q"], inp["k"], inp["v"], inp["a"], inp["b"],
                    inp["A_log"], inp["dt_bias"], inp["state"], plan,
                    mode=(sys.argv[3] if len(sys.argv) > 3 else "two"))
    run()
    torch.cuda.synchronize()
    print("=== compiled kernel stats (seg=%d m=%d) ===" % (seg, mm))
    for nm in ("prep_kernel", "scan_kernel", "seg_scan_kernel", "top_scan_kernel",
               "attn_kernel", "out2_kernel", "out_kernel"):
        report(nm.replace("_kernel", ""), getattr(gk, nm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
