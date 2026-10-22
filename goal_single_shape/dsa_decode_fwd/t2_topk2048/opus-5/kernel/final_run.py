"""Full evidence run: reference (KDA 0.5 CuTe-DSL) vs fp32 naive vs our kernels.

One process, bit-identical tensors, one timing routine for reference and
candidates. Primary metric = profiler kernel duration ("device time": GPU-side
kernel execution, host launch overhead excluded), cross-checked with a
CUDA-graph + events measurement where the implementation is capturable.
"""

import traceback

import torch

import ref_contest
from timing import graph_time_us, kernel_breakdown, kernel_span_us
from workload import make_workload, max_abs_err, naive_fp32

torch.backends.cuda.matmul.allow_tf32 = False
DECLARED_REF_US = 6.8

p = torch.cuda.get_device_properties(0)
print(f"STAGE_A device = {p.name}  SMs = {p.multi_processor_count}  cc = {p.major}.{p.minor}", flush=True)
print("STAGE_A torch", torch.__version__, "cuda", torch.version.cuda, flush=True)

w = make_workload()
print("STAGE_A workload: q_nope %s q_pe %s ckv %s kpe %s idx %s scale %.8f"
      % (tuple(w["q_nope"].shape), tuple(w["q_pe"].shape), tuple(w["ckv_cache"].shape),
         tuple(w["kpe_cache"].shape), tuple(w["sparse_indices"].shape), w["sm_scale"]), flush=True)
print("STAGE_A valid KV rows per request:", (w["sparse_indices"] >= 0).sum(dim=1).tolist(),
      " total:", int((w["sparse_indices"] >= 0).sum()), flush=True)

o_naive = naive_fp32(w)
print("STAGE_A fp32 naive: mean|o| = %.6f  max|o| = %.6f"
      % (o_naive.abs().mean(), o_naive.abs().max()), flush=True)


def measure(tag, fn):
    """Returns (kernel_sum_us, span_us, graph_us). NaN where unavailable."""
    ksum, rows = float("nan"), []
    try:
        ksum, rows = kernel_breakdown(fn)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    span, nk = float("nan"), 0
    try:
        span, nk = kernel_span_us(fn)
    except Exception:  # noqa: BLE001
        pass
    graph = float("nan")
    try:
        g = graph_time_us(fn, inner=20)
        graph = g if g > 0.9 else float("nan")  # <1us means the graph captured nothing
    except Exception:  # noqa: BLE001
        pass
    print(f"  [{tag}] kernel-sum {ksum:.3f} us | span {span:.3f} us ({nk} kernels) | "
          f"cuda-graph {graph:.3f} us", flush=True)
    for name, cnt, us in rows:
        print(f"        {us:8.3f} us  x{cnt:.1f}  {name}")
    return ksum, span, graph


# ---------------- the reference -------------------------------------------
ref_us = float("nan")
E_ref = TOL = float("nan")
try:
    print("STAGE_B importing reference kernel (never read, only executed) ...", flush=True)
    run_ref = ref_contest.make_callable(w)
    ref_out = ref_contest.as_output_tensor(run_ref())
    torch.cuda.synchronize()
    print("STAGE_B ref output:", tuple(ref_out.shape), ref_out.dtype, flush=True)
    E_ref = max_abs_err(ref_out.reshape(o_naive.shape), o_naive)
    TOL = 7.0 * E_ref
    print(f"STAGE_B E_ref = max|ref - fp32_naive| = {E_ref:.6e}", flush=True)
    print(f"STAGE_B tol   = 7 * E_ref             = {TOL:.6e}", flush=True)
    print("STAGE_C timing reference ...", flush=True)
    r_sum, r_span, r_graph = measure("REF KDA0.5 CuTe-DSL", run_ref)
    ref_us = max(x for x in (r_sum, r_span) if x == x)
except Exception:  # noqa: BLE001
    traceback.print_exc()

if ref_us != ref_us:
    ref_us = DECLARED_REF_US
TARGET = ref_us / 1.2
print("=" * 88, flush=True)
print(f"REF device time = {ref_us:.3f} us   (task states 6.8 us)")
print(f"TARGET = ref/1.2 = {TARGET:.3f} us")
print("=" * 88, flush=True)

# ---------------- our kernels ---------------------------------------------
results = []
print("STAGE_D CUDA kernel sweep ...", flush=True)
first = True
CFGS = [(32, 8, 2, 2), (32, 16, 2, 2), (32, 8, 2, 4), (32, 8, 2, 1), (32, 8, 1, 2)]
for fused in (True, False):
    for bn, nw, hg, tpp in CFGS:
        if not fused and tpp != 1:
            continue
        try:
            from dsa_cuda import DsaCuda

            impl = DsaCuda(w, block_n=bn, num_warps=nw, head_groups=hg, tiles_per_cta=tpp,
                           fused=fused, verbose=first)
            out = impl().clone()
            torch.cuda.synchronize()
            err = max_abs_err(out, o_naive)
            print(f"STAGE_D {impl} smem={impl.smem_bytes()} err={err:.6e} "
                  f"{'PASS' if err <= TOL else 'FAIL'}", flush=True)
            if first:
                print("  ours [0,0,:4] ", [round(x, 5) for x in out[0, 0, :4].float().tolist()])
                print("  naive[0,0,:4] ", [round(x, 5) for x in o_naive[0, 0, :4].tolist()])
                print("  ours [1,9,:4] ", [round(x, 5) for x in out[1, 9, :4].float().tolist()])
                print("  naive[1,9,:4] ", [round(x, 5) for x in o_naive[1, 9, :4].tolist()],
                      flush=True)
                first = False
            tag = f"cuda bn{bn}w{nw}h{hg}t{tpp}{'f' if fused else ''}"
            results.append((tag, err) + measure(f"OURS {tag}", impl))
        except Exception:  # noqa: BLE001
            traceback.print_exc()

print("STAGE_D3 cluster / DSMEM variant ...", flush=True)
for bn in ():
    try:
        from dsa_cuda import DsaCudaCluster

        cimpl = DsaCudaCluster(w, block_n=bn)
        out = cimpl().clone()
        torch.cuda.synchronize()
        err = max_abs_err(out, o_naive)
        print(f"STAGE_D3 {cimpl} err={err:.6e} {'PASS' if err <= TOL else 'FAIL'}", flush=True)
        results.append((f"cluster bn{bn}", err) + measure(f"OURS cluster bn{bn}", cimpl))
    except Exception:  # noqa: BLE001
        traceback.print_exc()

print("STAGE_D2 stage-by-stage cost of the fused kernel (bn=32, w8, hg=2) ...", flush=True)
try:
    from dsa_cuda import DsaCuda

    pr = DsaCuda(w, block_n=32, num_warps=8, head_groups=2, tiles_per_cta=1, fused=True)
    STAGES = [(1, "index read + valid scan"), (2, "+ KV/Q staging"), (3, "+ QK mma"),
              (4, "+ softmax & P"), (5, "+ PV mma"), (6, "+ atomic accumulate"),
              (7, "+ threadfence"), (0, "+ barrier & divide (full)")]
    for st, label in STAGES:
        measure(f"STAGE{st} {label}", pr.staged(st))
except Exception:  # noqa: BLE001
    traceback.print_exc()

print("STAGE_E Triton kernel ...", flush=True)
for cfg in (dict(block_n=32, splits=64, num_warps=4, num_stages=2),
            dict(block_n=16, splits=128, num_warps=4, num_stages=2)):
    try:
        from dsa_triton import DsaTriton

        t_impl = DsaTriton(w, **cfg)
        out = t_impl().clone()
        torch.cuda.synchronize()
        err = max_abs_err(out, o_naive)
        print(f"STAGE_E {t_impl} err={err:.6e} {'PASS' if err <= TOL else 'FAIL'}", flush=True)
        tag = f"triton bn{cfg['block_n']}s{cfg['splits']}"
        results.append((tag, err) + measure(f"OURS {tag}", t_impl))
    except Exception:  # noqa: BLE001
        traceback.print_exc()

print("=" * 88)
print(f"{'impl':22s} {'err':>11s} {'tol':>11s} {'ksum_us':>9s} {'span_us':>9s} "
      f"{'graph_us':>9s} {'verdict':>8s}")
best = None
for name, err, ksum, span, graph in results:
    worst = max(x for x in (ksum, span, graph) if x == x)
    good = (err <= TOL) and (worst <= TARGET)
    if good and (best is None or worst < best[1]):
        best = (name, worst)
    print(f"{name:22s} {err:11.4e} {TOL:11.4e} {ksum:9.3f} {span:9.3f} {graph:9.3f} "
          f"{'PASS' if good else 'FAIL':>8s}")
print(f"ref = {ref_us:.3f} us ; target = {TARGET:.3f} us")
if best:
    print(f"BEST PASSING: {best[0]} at {best[1]:.3f} us -> speedup {ref_us / best[1]:.2f}x")
print("FINAL_RUN DONE", flush=True)
