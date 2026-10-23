"""Python entry point for the tcgen05 UMMA variant in dsa_kernel_tcgen05.cu (correct, but slower than dsa_kernel.cu).

run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale) -> (output, lse)
matches the flashinfer-bench definition dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.
profile(...) runs once with per-CTA %globaltimer phase stamps and prints a summary.
"""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_ext = None
_scratch = {}

NUM_HEADS = 16
HEAD_DIM_CKV = 512


def _load():
    global _ext
    if _ext is None:
        build_dir = _HERE / "build_tcgen05"
        build_dir.mkdir(exist_ok=True)
        # torch's JIT loader appends its own -gencode list from TORCH_CUDA_ARCH_LIST; the
        # container default includes pre-Ampere targets where bf16 mma does not exist.
        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
        _ext = load(
            name="dsa_fwd_tcgen05_ext",
            sources=[str(_HERE / "dsa_kernel_tcgen05.cu")],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"],
            build_directory=str(build_dir),
            verbose=os.environ.get("DSA_VERBOSE_BUILD", "0") == "1",
        )
    return _ext


def _get_scratch(num_tokens: int, device: torch.device):
    key = (num_tokens, str(device))
    if key not in _scratch:
        ext = _load()
        ext.set_per_sm(PER_SM_OVERRIDE)
        nchunk = ext.num_chunks()
        _scratch[key] = dict(
            part_o=torch.empty(num_tokens, nchunk, HEAD_DIM_CKV, NUM_HEADS, dtype=torch.bfloat16, device=device),
            part_m=torch.empty(num_tokens, nchunk, NUM_HEADS, dtype=torch.float32, device=device),
            part_l=torch.empty(num_tokens, nchunk, NUM_HEADS, dtype=torch.float32, device=device),
            arrive=torch.zeros(num_tokens * ext.counter_pad(), dtype=torch.int64, device=device),
            smctr=torch.zeros(ext.num_sms() * 32, dtype=torch.int32, device=device),
            prof=torch.zeros(ext.grid_for(num_tokens), ext.num_prof(), dtype=torch.int64, device=device),
        )
    return _scratch[key]


# 1 = one tiled TMA copy per row; 0/2/3/4 = tile::gather4 variants (box rows 4/1 x dst cluster/cta)
GATHER_MODE = int(os.environ.get("DSA_GATHER_MODE", "2"))
PER_SM_OVERRIDE = int(os.environ.get("DSA_PER_SM", "2"))  # CTAs per SM (the occupancy query reports 1 for TMEM kernels)
SLOT_MODE = int(os.environ.get("DSA_SLOT_MODE", "1"))  # 0 = per-SM atomic counter, 1 = TMEM column address
MMA_M64 = int(os.environ.get("DSA_M64", "1"))  # 1 = S^T with M=64 UMMAs
L2_PROMO = int(os.environ.get("DSA_L2_PROMO", "2"))  # CUtensorMapL2promotion: 0 none, 1 64B, 2 128B, 3 256B


def partial_o_thd(scratch):
    """Published partials as [T][chunk][head][dim] (stored [T][chunk][dim][head])."""
    return scratch["part_o"].permute(0, 1, 3, 2)


def _run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, do_prof):
    ext = _load()
    s = _get_scratch(q_nope.shape[0], q_nope.device)
    return ext.run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, float(sm_scale), GATHER_MODE, PER_SM_OVERRIDE,
                   SLOT_MODE, MMA_M64, L2_PROMO, s["part_o"], s["part_m"], s["part_l"], s["arrive"], s["smctr"],
                   s["prof"], do_prof)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    out, lse = _run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, False)
    return out, lse


def noop(grid=1, smem_bytes=0):
    """Launch an empty kernel (timing-method calibration)."""
    _load().run_noop(grid, smem_bytes)


def profile(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, flush_buf=None):
    """One profiled launch (after a warmup launch); prints per-phase statistics in µs.

    Stamps: 0 start, 1 indices loaded, 2 gather done, 3 S done, 4 PV done, 5 published,
    6 slice barrier passed, 7 slice reduced (end); [10] = SM id, [11] = busy flag.
    """
    _run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, False)
    torch.cuda.synchronize()
    scratch = _get_scratch(q_nope.shape[0], q_nope.device)
    scratch["prof"].zero_()
    if flush_buf is not None:
        flush_buf.zero_()
    _run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, True)
    torch.cuda.synchronize()
    num_tokens = q_nope.shape[0]
    nchunk = _load().num_chunks()
    raw_all = scratch["prof"].cpu()
    raw = raw_all[: num_tokens * nchunk]  # rows = chunk index i (token i % T, chunk i / T)
    smid = raw[:, 10]
    busy = raw[:, 11] > 0
    tmem_col = raw[:, 12] & 0xFFFF
    slot = raw[:, 12] >> 32
    prof = raw[:, :16].double() / 1000.0  # ns -> us
    t0 = prof[:, 0].min()
    rel = prof - t0
    has_slice = prof[:, 6] > 0
    end = torch.where(has_slice, rel[:, 7], rel[:, 5])
    print(f"[prof] kernel span (first CTA start -> last CTA end): {end.max():.3f} us; "
          f"CTA start spread: {rel[:, 0].max():.3f} us; CTAs: {prof.shape[0]}, busy: {int(busy.sum())}, "
          f"with slice: {int(has_slice.sum())}")
    sm_busy = torch.bincount(smid[busy], minlength=int(smid.max()) + 1)
    print(f"[prof] SMs used: {int((torch.bincount(smid) > 0).sum())}; SMs hosting 2+ busy CTAs: {int((sm_busy >= 2).sum())}; "
          f"max busy CTAs on one SM: {int(sm_busy.max())}; TMEM base columns seen: {sorted(set(tmem_col.tolist()))}; "
          f"slots: {torch.bincount(slot).tolist()}")

    def stats(name, d):
        d = d.flatten()
        if d.numel() == 0:
            return
        print(f"[prof]   {name:22s} median={d.median():.3f}  p90={d.quantile(0.9):.3f}  max={d.max():.3f} us")

    print("[prof] busy CTAs:")
    b = prof[busy]
    stats("start->indices", b[:, 1] - b[:, 0])
    stats("indices->gather", b[:, 2] - b[:, 1])
    stats("gather->S", b[:, 3] - b[:, 2])
    stats("  gather->S issued", b[:, 13] - b[:, 2])
    stats("  issued->S done", b[:, 14] - b[:, 13])
    stats("  S done->TMEM read", b[:, 15] - b[:, 3])
    stats("S->PV", b[:, 4] - b[:, 3])
    stats("PV->published", b[:, 5] - b[:, 4])
    stats("start->published", b[:, 5] - b[:, 0])
    print("[prof] empty CTAs:")
    e = prof[~busy]
    stats("start->published", e[:, 5] - e[:, 0])
    print("[prof] slice CTAs:")
    sl = prof[has_slice]
    stats("published->barrier", sl[:, 6] - sl[:, 5])
    stats("barrier->stats(m,l)", sl[:, 8] - sl[:, 6])
    stats("stats->accumulated", sl[:, 9] - sl[:, 8])
    stats("accumulated->stored", sl[:, 7] - sl[:, 9])
    stats("barrier->reduced", sl[:, 7] - sl[:, 6])
    stats("start->end", sl[:, 7] - sl[:, 0])
    print(f"[prof] last publish at {rel[:, 5].max():.3f} us (busy: {rel[busy][:, 5].max():.3f} us); "
          f"barrier release: earliest {rel[has_slice][:, 6].min():.3f} us, last {rel[has_slice][:, 6].max():.3f} us")
    tok = torch.arange(prof.shape[0]) % num_tokens  # chunk index i -> token i % T
    for tt in range(int(tok.max()) + 1):
        sel = (tok == tt) & busy
        sls = (tok == tt) & has_slice
        if sel.any():
            print(f"[prof] token {tt}: busy={int(sel.sum()):2d} last_publish={rel[sel][:, 5].max():.3f} "
                  f"barrier={rel[sls][:, 6].min():.3f} last_end={rel[sls][:, 7].max():.3f} us")
