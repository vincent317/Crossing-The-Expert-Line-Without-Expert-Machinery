"""Round 2a: what do other backends reach on this shape, and what tooling exists.

Gives an upper bound reality check before committing to a kernel-writing path.
"""

import glob
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing import cuda_events_median, kernel_breakdown, make_inputs, tflops, B, S, HQ, HKV, D

SCALE = D ** -0.5
print(f"shape B={B} S={S} Hq={HQ} Hkv={HKV} D={D} causal bf16", flush=True)
print(f"bwd matmul FLOPs = {__import__('timing').bwd_flops():.4e}", flush=True)

# ----------------------------------------------------------------- GEMM ceiling
print("=" * 70, flush=True)
print("GEMM CEILING (bf16 8192^3)", flush=True)
print("=" * 70, flush=True)
M = N = K = 8192
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b_ = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
gflop = 2.0 * M * N * K
med, lo, hi = cuda_events_median(lambda: torch.matmul(a, b_), 20, 50)
print(f"cuBLAS  {med:.4f} ms  -> {gflop/(med*1e-3)/1e12:.1f} TFLOPS", flush=True)

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _mm(A, Bp, C, Mm, Nn, Kk, sam, sak, sbk, sbn, scm, scn,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
        pid = tl.program_id(0)
        nm = tl.cdiv(Mm, BM)
        nn = tl.cdiv(Nn, BN)
        ng = GM * nn
        gid = pid // ng
        f = gid * GM
        gsz = min(nm - f, GM)
        pm = f + (pid % gsz)
        pn = (pid % ng) // gsz
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in range(0, tl.cdiv(Kk, BK)):
            ap = A + rm[:, None] * sam + (rk[None, :] + kk * BK) * sak
            bp = Bp + (rk[:, None] + kk * BK) * sbk + rn[None, :] * sbn
            av = tl.load(ap)
            bv = tl.load(bp)
            acc = tl.dot(av, bv, acc)
        cp = C + rm[:, None] * scm + rn[None, :] * scn
        tl.store(cp, acc.to(tl.bfloat16))

    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    best = None
    for BM, BN, BK, w, s in [(128, 256, 64, 8, 3), (128, 128, 64, 8, 4), (256, 128, 64, 8, 3),
                             (128, 256, 64, 4, 4), (256, 256, 64, 8, 3)]:
        try:
            grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
            f = lambda: _mm[grid](a, b_, c, M, N, K, a.stride(0), a.stride(1), b_.stride(0),
                                  b_.stride(1), c.stride(0), c.stride(1),
                                  BM=BM, BN=BN, BK=BK, GM=8, num_warps=w, num_stages=s)
            med, _, _ = cuda_events_median(f, 10, 30)
            tf = gflop / (med * 1e-3) / 1e12
            print(f"triton {BM}x{BN}x{BK} w{w} s{s}: {med:.4f} ms -> {tf:.1f} TFLOPS", flush=True)
            if best is None or tf > best:
                best = tf
        except Exception as e:
            print(f"triton {BM}x{BN}x{BK} w{w} s{s} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
    print(f"triton GEMM best = {best} TFLOPS", flush=True)
except Exception as e:
    print("triton gemm section failed:", repr(e)[:300], flush=True)

# ----------------------------------------------------------------- SDPA backends
print("=" * 70, flush=True)
print("SDPA BACKWARD BASELINES (B,H,S,D layout, enable_gqa)", flush=True)
print("=" * 70, flush=True)
from torch.nn.attention import SDPBackend, sdpa_kernel

qh = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
kh = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
vh = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
doh = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.bfloat16)

for name, backend in [("CUDNN", SDPBackend.CUDNN_ATTENTION),
                      ("FLASH", SDPBackend.FLASH_ATTENTION),
                      ("MEM_EFF", SDPBackend.EFFICIENT_ATTENTION)]:
    try:
        with sdpa_kernel(backend):
            o = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, is_causal=True, scale=SCALE, enable_gqa=True)
            f = lambda: torch.autograd.grad(o, (qh, kh, vh), doh, retain_graph=True)
            f()
            med, lo, hi = cuda_events_median(f, 10, 50)
            print(f"SDPA[{name}] bwd cuda-events median {med:.4f} ms (min {lo:.3f}) "
                  f"-> {tflops(med):.1f} TFLOPS", flush=True)
            kernel_breakdown(f, 15, label=name)
        del o
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"SDPA[{name}] FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        torch.cuda.empty_cache()

del qh, kh, vh, doh
torch.cuda.empty_cache()

# ----------------------------------------------------------- ref pack_gqa probe
print("=" * 70, flush=True)
print("REF with pack_gqa variants", flush=True)
print("=" * 70, flush=True)
from flash_attn.cute.interface import flash_attn_func

q, k, v, do = make_inputs()
for pg in [None, True, False]:
    try:
        out = flash_attn_func(q, k, v, softmax_scale=SCALE, causal=True, pack_gqa=pg)
        out = out[0] if isinstance(out, tuple) else out
        f = lambda: torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
        f()
        med, lo, hi = cuda_events_median(f, 10, 30)
        print(f"REF pack_gqa={pg}: {med:.4f} ms -> {tflops(med):.1f} TFLOPS", flush=True)
        del out
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"REF pack_gqa={pg} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
        torch.cuda.empty_cache()

del q, k, v, do
torch.cuda.empty_cache()

# ----------------------------------------------------------------- tooling scan
print("=" * 70, flush=True)
print("TOOLING SCAN", flush=True)
print("=" * 70, flush=True)
import cutlass
import cutlass.cute as cute
print("cutlass dsl root:", os.path.dirname(cutlass.__file__), flush=True)
print("cute top-level:", [x for x in dir(cute) if not x.startswith("_")], flush=True)
try:
    import cutlass.cute.nvgpu as nvgpu
    print("nvgpu:", [x for x in dir(nvgpu) if not x.startswith("_")], flush=True)
    from cutlass.cute.nvgpu import tcgen05
    print("tcgen05:", [x for x in dir(tcgen05) if not x.startswith("_")], flush=True)
    print("tcgen05.mma:", [x for x in dir(tcgen05.mma) if not x.startswith("_")]
          if hasattr(tcgen05, "mma") else "no .mma", flush=True)
except Exception as e:
    print("tcgen05 probe failed:", repr(e)[:300], flush=True)

root = os.path.dirname(cutlass.__file__)
pyfiles = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
print(f"dsl py files: {len(pyfiles)}", flush=True)
for p in pyfiles:
    r = os.path.relpath(p, root)
    if any(t in r for t in ("tcgen05", "sm100", "blackwell", "example", "tutorial")):
        print("   ", r, flush=True)

print("--- C++ cutlass headers on disk ---", flush=True)
print(subprocess.run(["bash", "-lc",
                      "find / -name 'mma_sm100*.hpp' -o -name 'cute' -maxdepth 6 -type d 2>/dev/null | head -20"],
                     capture_output=True, text=True).stdout, flush=True)
print(subprocess.run(["bash", "-lc",
                      "ls /usr/local/cuda/include/cute 2>&1 | head -3; "
                      "python -c 'import cutlass_library, os; print(os.path.dirname(cutlass_library.__file__))' 2>&1 | tail -1"],
                     capture_output=True, text=True).stdout, flush=True)
print("DONE-LANDSCAPE", flush=True)
