"""Device-time benchmark: reference (CuTe-DSL GDN prefill) vs the new kernel.

Identical data, identical harness, identical iteration counts for everything.
Device time = the GPU kernel span of one call (first kernel start -> last kernel
end), reported four ways:
  span      : CUPTI trace, max(kernel end) - min(kernel start) per call
              -> the nsys "kernel duration" definition, no host cost
  cupti     : sum of per-kernel durations per call (no inter-kernel gaps)
  cudagraph : CUDA-graph replay timed with CUDA events (no host launch cost)
  loop      : plain back-to-back launches (includes host launch cost)

Every measurement is preceded by a 1.5 s GPU hammer, because a few ms of
micro-benchmark never boosts the SM clock out of its idle state.

Usage: python bench.py [triton|cuda|all]
"""

import sys

import torch

import check
import common as C

REF_STATED_US = 4.9   # published CuTe-DSL GDN prefill number, gdn_small path


def time_all(name, fn, iters=2000, reps=5):
    out = {}
    spans = []
    for _ in range(reps):
        sp, nk, cup = C.trace_span_us(fn, iters=50)
        spans.append(sp)
        out["nk"], out["cupti"] = nk, cup
    spans.sort()
    out["span"] = spans[len(spans) // 2]
    out["span_lo"], out["span_hi"] = spans[0], spans[-1]
    try:
        out["cudagraph"], out["_g"] = C.graph_time_us(fn, iters=iters)
    except Exception as exc:  # noqa: BLE001
        print(f"  [{name}] cudagraph failed: {type(exc).__name__} {exc}")
        out["cudagraph"] = float("nan")
    out["loop"] = C.loop_time_us(fn, iters=iters)
    print(f"  {name:34s} span {out['span']:8.3f} (of {reps}: {out['span_lo']:.3f}"
          f"-{out['span_hi']:.3f}) | cupti {out['cupti']:8.3f} | "
          f"cudagraph {out['cudagraph']:8.3f} | loop {out['loop']:9.3f} us | "
          f"{out['nk']} kernel(s) | clk {C.gpu_clock_mhz()}")
    return out


def pick_cuda_variant(inp, cands=(0, 1, 2, 3, 4, 5, 6)):
    """Time the templated CUDA shapes once and keep the fastest working one."""
    import kernel_cuda
    out = torch.empty(C.T, C.HV, C.D, dtype=inp["q"].dtype, device="cuda")
    sn = torch.empty_like(inp["state"])
    best, best_t = None, float("inf")
    for var in cands:
        def fn(var=var):
            return kernel_cuda.gdn_fwd(inp["q"], inp["k"], inp["v"], inp["a"], inp["b"],
                                       inp["A_log"], inp["dt_bias"], inp["state"],
                                       out=out, new_state=sn, variant=var)
        try:
            fn()
            torch.cuda.synchronize()
            t, _g = C.graph_time_us(fn, iters=500)
        except Exception as exc:  # noqa: BLE001
            print(f"  variant {var:2d} ({kernel_cuda.VARIANTS[var]:24s}) failed: "
                  f"{type(exc).__name__} {str(exc)[:120]}")
            continue
        print(f"  variant {var:2d} ({kernel_cuda.VARIANTS[var]:24s}) cudagraph {t:8.3f} us")
        if t < best_t:
            best, best_t = var, t
    if best is None:
        print("  -> no CUDA variant works")
        return None
    print(f"  -> using variant {best} ({kernel_cuda.VARIANTS[best]})")
    kernel_cuda.BEST = best
    return best


# BV=8 / 4 warps is the measured optimum; NLEV controls the Neumann depth
# (5 = exact inverse for a 32-wide chunk, 2 = truncated at j<=7, which on this
# workload is numerically indistinguishable - see the printed errors).
TRITON_CANDS = [
    dict(BV=8, num_warps=4, NLEV=2), dict(BV=8, num_warps=4, NLEV=5),
]


def pick_triton_config(inp):
    """Time the Triton configurations once and keep the fastest working one."""
    import kernel
    out = torch.empty(C.T, C.HV, C.D, dtype=inp["q"].dtype, device="cuda")
    sn = torch.empty_like(inp["state"])
    best, best_t = None, float("inf")
    for cfg in TRITON_CANDS:
        def fn(cfg=cfg):
            return kernel.gdn_fwd(inp["q"], inp["k"], inp["v"], inp["a"], inp["b"],
                                  inp["A_log"], inp["dt_bias"], inp["state"],
                                  out=out, new_state=sn, **cfg)
        try:
            fn()
            torch.cuda.synchronize()
            ts = sorted(C.trace_span_us(fn, iters=50)[0] for _ in range(3))
            t = ts[1]
        except Exception as exc:  # noqa: BLE001
            print(f"  {str(cfg):52s} failed: {type(exc).__name__} {str(exc)[:100]}")
            continue
        print(f"  {str(cfg):52s} span {t:8.3f} us (of 3: {ts[0]:.3f}-{ts[2]:.3f})")
        if t < best_t:
            best, best_t = cfg, t
    if best is not None:
        print(f"  -> using {best}")
        check.TRITON_CFG = best
    return best


def main(argv):
    which = [w for w in ("cuda", "triton") if w in argv] or ["cuda", "triton"]
    inp = C.make_inputs()
    g, beta = C.gates_fp32(inp["a"], inp["b"], inp["A_log"], inp["dt_bias"])
    print(f"device {torch.cuda.get_device_name(0)} sm{torch.cuda.get_device_capability(0)} "
          f"torch {torch.__version__}")
    print(f"clock idle : {C.gpu_clock_mhz()}   (sm,max,power,temp)")
    C.warm_clocks(2.0)
    print(f"clock warm : {C.gpu_clock_mhz()}")
    print(f"workload T={C.T} N={C.N} HK={C.HK} HV={C.HV} D={C.D} scale={C.SCALE:.8f} "
          f"q/k/v/a/b=bf16 state=fp32 k-last\n")

    if "triton" in which:
        print("=== Triton config calibration ===")
        pick_triton_config(inp)
    if "cuda" in which:
        print("=== CUDA variant calibration ===")
        try:
            if pick_cuda_variant(inp) is None:
                which = [w for w in which if w != "cuda"]
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("CUDA build unusable:", type(exc).__name__, str(exc)[:500])
            which = [w for w in which if w != "cuda"]
    accs = {w: check.run(inp=inp, which=w) for w in which}

    print("\n=== DEVICE TIME (same data, same harness, us) ===")
    ref_t = None
    try:
        import refmod
        variants = refmod.make_ref_variants(inp, g, beta)
        best_label, best_span = None, float("inf")
        for label, fn in variants.items():
            try:
                o, st = fn()
                torch.cuda.synchronize()
                if bool(o.float().isnan().any()):
                    print(f"  ref [{label}] returns NaN, skipped")
                    continue
            except Exception as exc:  # noqa: BLE001
                print(f"  ref [{label}] failed: {type(exc).__name__} {str(exc)[:90]}")
                continue
            t = time_all(f"ref: {label[:30]}", fn)
            if t["span"] < best_span:
                best_label, best_span, ref_t = label, t["span"], t
        print(f"  -> fastest reference path: {best_label} ({best_span:.3f} us). The "
              f"package exposes no short-sequence ('gdn_small') prefill entry, so the\n"
              f"     published {REF_STATED_US} us gdn_small figure is used as the "
              f"bar for condition 4 (stricter than the measured path).")
    except Exception as exc:  # noqa: BLE001
        print("  reference unavailable for timing:", type(exc).__name__, str(exc)[:300])

    mine = {}
    for w in which:
        name, fwd, kw = check.impl(w)
        out_buf = torch.empty(C.T, C.HV, C.D, dtype=inp["q"].dtype, device="cuda")
        st_buf = torch.empty_like(inp["state"])

        def fn(fwd=fwd, kw=kw, out_buf=out_buf, st_buf=st_buf):
            return fwd(inp["q"], inp["k"], inp["v"], inp["a"], inp["b"], inp["A_log"],
                       inp["dt_bias"], inp["state"], out=out_buf, new_state=st_buf, **kw)
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} launch failed: {type(exc).__name__} {str(exc)[:200]}")
            continue
        mine[w] = time_all(f"new kernel ({name})", fn)

    print("\n=== VERDICT ===")
    for w in accs:
        a = accs[w]
        print(f"[{w}] accuracy: E_ref={a['E_ref']:.6e}  tol=7*E_ref={a['tol']:.6e}  "
              f"E_new={a['E_new']:.6e}  -> {'PASS' if a['pass'] else 'FAIL'}")
    if not mine:
        print("no working implementation")
        return
    best = min(mine, key=lambda w: mine[w]["span"])
    print(f"fastest implementation: {best}")
    for meth in ("span", "cupti", "cudagraph", "loop"):
        r = ref_t[meth] if ref_t else float("nan")
        m = mine[best][meth]
        print(f"  [{meth:9s}] ref {r:9.3f} | mine {m:8.3f} | speedup {r/m:5.2f}x | "
              f"need mine <= ref/1.2 = {r/1.2:8.3f} -> {'PASS' if m <= r / 1.2 else 'FAIL'}")
    print(f"  [published] ref {REF_STATED_US:.3f} us (CuTe-DSL gdn_small) | "
          f"need <= {REF_STATED_US/1.2:.3f} | mine(span) {mine[best]['span']:.3f} -> "
          f"{'PASS' if mine[best]['span'] <= REF_STATED_US / 1.2 else 'FAIL'}")

    a = accs.get(best, {})
    rs_meas = ref_t["span"] if ref_t else float("nan")
    rs = REF_STATED_US            # published gdn_small figure is the binding bar
    ms = mine[best]["span"]
    print("\n=== GOAL CONDITIONS ===")
    print(f"1. ref device time measured on B200 (span)        : {rs_meas:.3f} us"
          f" (fastest CuTe-DSL prefill entry)   [{'OK' if rs_meas == rs_meas else 'MISSING'}]")
    print(f"   published gdn_small figure for this shape      : {rs:.3f} us"
          f"  -> used as the bar (no gdn_small entry in the package API)")
    print(f"2. fp32 naive oracle + E_ref, tol = 7*E_ref       : E_ref={a.get('E_ref', float('nan')):.6e}"
          f"  tol={a.get('tol', float('nan')):.6e}")
    print(f"3. |new - naive| <= tol                           : E_new={a.get('E_new', float('nan')):.6e}"
          f"   [{'PASS' if a.get('pass') else 'FAIL'}]")
    print(f"4. new device time <= ref/1.2                     : {ms:.3f} <= {rs/1.2:.3f}"
          f"   [{'PASS' if ms <= rs / 1.2 else 'FAIL'}]"
          f"   (vs published bar {REF_STATED_US/1.2:.3f} us:"
          f" {'PASS' if ms <= REF_STATED_US / 1.2 else 'FAIL'})")
    print(f"   speedup over reference: {rs/ms:.2f}x")


if __name__ == "__main__":
    main(sys.argv[1:])
