"""Probe the Gluon / tcgen05 API surface available in this image.

Most important question: is there a bulk TMA reduce-add (cp.reduce.async.bulk.tensor)
exposed anywhere, since per-thread fp32 atomics measured only ~3 TB/s device-wide
(job 1049524: 104.7 GB in 34.0 ms) which alone would exceed the time budget.
"""

import torch
import triton

print("triton", triton.__version__, flush=True)

import triton._C.libtriton as _lt

print("== libtriton submodules:", [m for m in dir(_lt) if not m.startswith("_")], flush=True)

GB = _lt.gluon_ir.GluonOpBuilder
names = [m for m in dir(GB) if any(k in m.lower() for k in
         ("tma", "reduce", "atomic", "bulk", "tcgen05", "tmem", "mbarrier", "copy", "desc"))]
print("== GluonOpBuilder relevant methods:", flush=True)
for m in sorted(names):
    doc = (getattr(GB, m).__doc__ or "").strip().split("\n")[0]
    print(f"   {m}   :: {doc}", flush=True)

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell as bw
from triton.experimental.gluon.language.nvidia.blackwell import tma as btma
from triton.experimental.gluon.language.nvidia.hopper import tma as htma

print("== gl.nvidia.blackwell:", [m for m in dir(bw) if not m.startswith("_")], flush=True)
print("== blackwell.tma:", [m for m in dir(btma) if not m.startswith("_")], flush=True)
print("== hopper.tma:", [m for m in dir(htma) if not m.startswith("_")], flush=True)
print("== gl has:", [m for m in ("exp2", "where", "atomic_add", "barrier", "num_programs",
                                 "static_range", "program_id", "convert_layout", "inline_asm_elementwise")
                     if hasattr(gl, m)], flush=True)

# regular-triton descriptor reduce (this is what v3 used and it worked) -- check the op name
import triton.language as tl
td = tl.tensor_descriptor if hasattr(tl, "tensor_descriptor") else None
print("== tl.tensor_descriptor methods:", [m for m in dir(td) if not m.startswith("_")] if td else None, flush=True)

# ---------------------------------------------------------------- smoke test
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout, mbarrier,
    tcgen05_mma, tcgen05_commit, fence_async_shared)


@gluon.jit
def _smoke(a_desc, b_desc, c_desc, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
           PERMUTE_A: gl.constexpr, num_warps: gl.constexpr):
    """a (M,K) @ b(N,K)^T -> c (M,N); if PERMUTE_A, a is stored (K,M) and permuted."""
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    htma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
    htma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    mbarrier.init(bar, count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout([M, N], col_stride=1)
    acc = allocate_tensor_memory(gl.float32, [M, N], tmem_layout)
    a = a_smem.permute((1, 0)) if PERMUTE_A else a_smem
    tcgen05_mma(a, b_smem.permute((1, 0)), acc, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)

    reg: gl.constexpr = get_tmem_reg_layout(gl.float32, (M, N), tmem_layout, num_warps)
    out = acc.load(reg)
    c_smem = gl.allocate_shared_memory(gl.float32, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(out)
    fence_async_shared()
    htma.async_copy_shared_to_global(c_desc, [0, 0], c_smem)
    htma.store_wait(pendings=0)


def smoke(M, N, K, permute_a):
    a = torch.randn((K, M) if permute_a else (M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    ashape = [K, M] if permute_a else [M, K]
    a_desc = TensorDescriptor.from_tensor(a, ashape, gl.NVMMASharedLayout.get_default_for(ashape, gl.bfloat16))
    b_desc = TensorDescriptor.from_tensor(b, [N, K], gl.NVMMASharedLayout.get_default_for([N, K], gl.bfloat16))
    c_desc = TensorDescriptor.from_tensor(c, [M, N], gl.NVMMASharedLayout.get_default_for([M, N], gl.float32))
    _smoke[(1,)](a_desc, b_desc, c_desc, M, N, K, permute_a, num_warps=8)
    ref = (a.T.float() @ b.T.float()) if permute_a else (a.float() @ b.T.float())
    err = (c - ref).abs().max().item()
    print(f"smoke M{M} N{N} K{K} permute_a={permute_a}: max err {err:.4e} "
          f"(|ref| {ref.abs().max().item():.2f})", flush=True)


for permute_a in (False, True):
    try:
        smoke(128, 64, 128, permute_a)
    except Exception as e:
        print(f"smoke permute_a={permute_a} FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)

# store through a permuted shared-memory view (needed to materialise dS from dS^T)
@gluon.jit
def _permstore(IN, OUT, M: gl.constexpr, N: gl.constexpr, num_warps: gl.constexpr):
    blk: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    om = gl.arange(0, M, gl.SliceLayout(1, blk))
    on = gl.arange(0, N, gl.SliceLayout(0, blk))
    x = gl.load(IN + om[:, None] * N + on[None, :])              # (M, N)
    lay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([N, M], gl.float32)
    sm = gl.allocate_shared_memory(gl.float32, [N, M], lay)
    sm.permute((1, 0)).store(x)                                   # write transposed
    blk2: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    on2 = gl.arange(0, N, gl.SliceLayout(1, blk2))
    om2 = gl.arange(0, M, gl.SliceLayout(0, blk2))
    y = sm.load(blk2)                                             # (N, M)
    gl.store(OUT + on2[:, None] * M + om2[None, :], y)


try:
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    y = torch.empty(128, 64, device="cuda", dtype=torch.float32)
    _permstore[(1,)](x, y, 64, 128, num_warps=4)
    print(f"permuted smem store: max err {(y - x.T).abs().max().item():.3e}", flush=True)
except Exception as e:
    print(f"permuted smem store FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)

print("DONE-PROBE", flush=True)
