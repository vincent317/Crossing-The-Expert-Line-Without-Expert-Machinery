"""Device-time benchmark: ref (KDA 0.5 CuTe-DSL) vs the new kernel on identical data.

Primary metric = CUPTI kernel span per call (first kernel start -> last kernel end on
the GPU timeline, no host launch overhead), captured with torch.profiler (CUPTI
activity API) and delimited by a marker kernel. Reported for the KDA 0.5 protocol
(cold L2: a >L2-sized buffer is overwritten between iterations) and for warm L2.
CUDA-event timing is printed as a secondary, launch-bound reference only.
"""
import collections
import json
import statistics
import sys

import torch
from torch.profiler import ProfilerActivity, profile

import dsa_ops
import run_ref
import workload

ITERS = 100
FLUSH_BYTES = 512 << 20  # B200 L2 is ~126MB
TARGET_SPEEDUP = 1.2
GPU_CATS = ("kernel", "gpu_memset", "gpu_memcpy")


def parse_trace(trace_path):
    """Per-iteration kernel spans (us) between marker kernels, kernel-name histogram,
    kernels/iter histogram and the marker kernel's own durations (empty-kernel floor)."""
    ev = json.load(open(trace_path))["traceEvents"]
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in GPU_CATS]
    gpu.sort(key=lambda e: e["ts"])
    groups, cur, marker_durs = [], None, []
    for e in gpu:
        n = e["name"]
        if "dsa_marker_kernel" in n:
            marker_durs.append(e["dur"])
            cur = []
            groups.append(cur)
            continue
        if ("dsa_l2_" in n and "flush" in n) or "dsa_spin_kernel" in n:
            cur = None
            continue
        if cur is not None:
            cur.append(e)
    spans, names, nk = [], collections.Counter(), collections.Counter()
    for g in groups:
        if not g:
            continue
        start = min(e["ts"] for e in g)
        end = max(e["ts"] + e["dur"] for e in g)
        spans.append(end - start)
        nk[len(g)] += 1
        for e in g:
            names[e["name"][:90]] += 1
    return spans, names, nk, marker_durs


def eager_spans(fn, iters, flush_mode, tools, flush_buf, flush_sink, trace_path):
    """flush_mode: None = warm L2, 0 = overwrite flush (dirty L2), 1 = stream-read flush (clean L2)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            if flush_mode is not None:
                tools.l2_flush(flush_buf, flush_mode, flush_sink)
            tools.marker()
            fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    return parse_trace(trace_path)


def cuda_event_us(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    return [st[i].elapsed_time(en[i]) * 1000.0 for i in range(iters)]


def stats(xs):
    xs = sorted(xs)
    return (f"median={statistics.median(xs):.3f} us  min={xs[0]:.3f}  mean={statistics.fmean(xs):.3f}  "
            f"p90={xs[int(0.9 * (len(xs) - 1))]:.3f}  n={len(xs)}")


def phase_breakdown(lib, args, tools, flush_buf):
    """Timing build: per-CTA phase stamps of thread 0 -> median ns / cycles per phase."""
    if "mma" in lib.config:
        names = ["idx+Q loads, spec cp.async issued", "compaction + wait copies + sync", "partial S (mma)",
                 "softmax + P (sync A..B)", "PV (mma)", "epilogue stores"]
    else:
        names = ["indices arrive (load latency)", "compaction", "rows arrive (gather latency)",
                 "softmax math (warp0)", "merge barrier", "final stores"]
    print(f"[phase] kernel '{lib.name}': {lib.config}")
    for cold in (True, False):
        reps_ns, reps_cy = [], []
        for _ in range(20):
            if cold:
                tools.l2_flush(flush_buf)
            lib.fwd(*args)
            ns, cy = lib.timing_read()
            reps_ns.append(ns)
            reps_cy.append(cy)
        ns = torch.stack(reps_ns).double()  # [rep, cta, stamp]
        cy = torch.stack(reps_cy).double()
        present = ns[0, :, 0] != 0  # CTAs that exist for this kernel (2 for the mma kernel, 32 otherwise)
        ns, cy = ns[:, present], cy[:, present]
        d_ns = (ns[:, :, 1:] - ns[:, :, :-1]).flatten(0, 1)
        d_cy = (cy[:, :, 1:] - cy[:, :, :-1]).flatten(0, 1)
        total_ns = (ns[:, :, -1] - ns[:, :, 0]).flatten()
        total_cy = (cy[:, :, -1] - cy[:, :, 0]).flatten()
        skew = (ns[:, :, 0].max(dim=1).values - ns[:, :, 0].min(dim=1).values)  # CTA start spread per rep
        last_end = (ns[:, :, -1].max(dim=1).values - ns[:, :, 0].min(dim=1).values)
        mode = "cold-L2" if cold else "warm-L2"
        print(f"[phase] {mode}: entry->stores per CTA median {total_ns.median():.0f} ns / {total_cy.median():.0f} cyc "
              f"(=> SM clock ~{total_cy.median() / max(total_ns.median(), 1):.2f} GHz); "
              f"CTA start skew median {skew.median():.0f} ns; first-entry->last-store {last_end.median():.0f} ns")
        for i in range(len(names)):
            print(f"[phase]   {names[i]:30s}: median {d_ns[:, i].median():.0f} ns / {d_cy[:, i].median():.0f} cyc  "
                  f"(p90 {d_cy[:, i].quantile(0.9):.0f} cyc)")


def main():
    torch.cuda.set_device(0)
    inp = workload.make_inputs()
    print("[bench]", workload.describe(inp))
    print(f"[bench] GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}")
    args = (inp["q_nope"], inp["q_pe"], inp["ckv_cache"], inp["kpe_cache"], inp["sparse_indices"], inp["sm_scale"])
    libs = dsa_ops.load_all()
    tools = next(lib for lib in libs if lib.has_tools)
    for lib in libs:
        print(f"[bench] kernel '{lib.name}': {lib.config}")
    flush_buf = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")

    impls = {}
    try:
        ref_run = run_ref.load_ref_run()
        ref_run(*args)  # JIT compile
        torch.cuda.synchronize()
        impls["ref"] = lambda: ref_run(*args)
    except Exception as exc:
        print("[bench] REFERENCE FAILED (benchmarking new kernels only):\n" + run_ref.format_exc_without_source(exc))
    for lib in libs:
        if lib.is_timing:
            continue
        impls[lib.name] = (lambda l: lambda: l.fwd(*args))(lib)
    sink = torch.zeros(64, dtype=torch.int32, device="cuda")
    impls["probe_chain1"] = lambda: tools.probe(1, inp["sparse_indices"], inp["ckv_cache"], sink)
    impls["probe_chain2"] = lambda: tools.probe(2, inp["sparse_indices"], inp["ckv_cache"], sink)
    impls["probe_grid"] = lambda: tools.probe(3, inp["sparse_indices"], inp["ckv_cache"], sink)
    mma_lib = next((lib for lib in libs if lib.has_mma_probe), None)
    if mma_lib is not None:
        fsink = torch.zeros(64, dtype=torch.float32, device="cuda")
        impls["probe_hmma_dep64"] = lambda: mma_lib.mma_probe(1, fsink)
        impls["probe_hmma_ind64"] = lambda: mma_lib.mma_probe(2, fsink)
    for fn in impls.values():  # warm up
        fn()
    torch.cuda.synchronize()

    for lib in libs:
        if lib.is_timing:
            phase_breakdown(lib, args, tools, flush_buf)

    med = {}
    modes = [("cold-L2", 0), ("cold-L2-readflush", 1), ("cold-L2-settled", 2), ("warm-L2", None)]
    floor = {mode: [] for mode, _ in modes}
    flush_sink = torch.zeros(4096, dtype=torch.int32, device="cuda")
    for mode, flush_mode in modes:
        for name, fn in impls.items():
            trace = f"/tmp/trace_{name}_{mode}.json"
            spans, names, nk, marker_durs = eager_spans(fn, ITERS, flush_mode, tools, flush_buf, flush_sink, trace)
            floor[mode] += marker_durs
            if not spans:
                print(f"[bench] {name} {mode}: no GPU kernels captured!")
                continue
            med[(name, mode)] = statistics.median(spans)
            tag = "[RESULT]" if name in ("ref", "new") else "[bench] "
            print(f"{tag} {name:18s} device time (CUPTI kernel span, {mode}): {stats(spans)}")
            if name in ("ref", "new") and mode == "cold-L2":
                print(f"[bench]  {name} kernels/iter={dict(nk)}; kernels: {dict(names)}")
    for mode, durs in floor.items():
        print(f"[bench] empty marker kernel duration ({mode}, launch floor, CUPTI): {stats(durs)}")

    for name in ("ref", "new"):
        if name in impls:
            print(f"[bench] {name:3s} CUDA-event per-call time (includes host launch gaps, secondary): "
                  f"{stats(cuda_event_us(impls[name], 200))}")

    if "ref" not in impls:
        return 3
    rc = 0
    for mode, _ in modes:
        if ("ref", mode) not in med or ("new", mode) not in med:
            continue
        r, n = med[("ref", mode)], med[("new", mode)]
        target = r / TARGET_SPEEDUP
        ok = n <= target
        rc |= 0 if ok else 2
        print(f"[RESULT] {mode}: ref median = {r:.3f} us, new median = {n:.3f} us, speedup = {r / n:.3f}x; "
              f"target new <= ref/1.2 = {target:.3f} us -> {'PASS' if ok else 'FAIL'}")
        for lib in libs[1:]:
            v = med.get((lib.name, mode))
            if v is not None and not lib.is_diagnostic:
                print(f"[bench]  variant {lib.name:9s} {mode}: median = {v:.3f} us, speedup vs ref = {r / v:.3f}x")
            elif v is not None:
                print(f"[bench]  diagnostic {lib.name:9s} {mode}: median = {v:.3f} us (not a valid kernel)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
