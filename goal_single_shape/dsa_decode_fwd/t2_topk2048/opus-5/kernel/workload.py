"""Deterministic workload for DSA FWD decode (long/short mixed).

Shape spec (contest workload 05f6de65):
  T = 2 (decode batch), H = 16 heads, ckv dim 512, kpe dim 64,
  top-k = 2048, page size 64, page pool 8462 pages, bf16.
  Request 0 hits 6 KV rows, request 1 hits 337 KV rows (343 total).

Every script (reference timing, fp32 naive check, custom kernel bench) imports
this module so that all measurements use bit-identical inputs.
"""

import torch

T = 2
H = 16
D_CKV = 512
D_KPE = 64
TOPK = 2048
PAGE_SIZE = 64
NUM_PAGES = 8462
SEQ_LENS = (6, 337)
SM_SCALE = float((D_CKV + D_KPE) ** -0.5)


def make_workload(device="cuda", seed=1234):
    """Build the fixed workload tensors on `device`.

    sparse_indices holds *global slot ids* into the flattened page pool
    (slot = page_id * PAGE_SIZE + offset); -1 marks a padded / unused slot.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    q_nope = torch.randn(T, H, D_CKV, generator=g, device=device, dtype=torch.bfloat16)
    q_pe = torch.randn(T, H, D_KPE, generator=g, device=device, dtype=torch.bfloat16)
    ckv_cache = torch.randn(
        NUM_PAGES, PAGE_SIZE, D_CKV, generator=g, device=device, dtype=torch.bfloat16
    )
    kpe_cache = torch.randn(
        NUM_PAGES, PAGE_SIZE, D_KPE, generator=g, device=device, dtype=torch.bfloat16
    )

    # Paged layout: each request owns ceil(seq_len / PAGE_SIZE) distinct random
    # pages; token i of the request lives in page[i // 64] at offset i % 64.
    perm = torch.randperm(NUM_PAGES, generator=g, device=device)
    sparse_indices = torch.full((T, TOPK), -1, device=device, dtype=torch.int32)
    page_cursor = 0
    for t, seq_len in enumerate(SEQ_LENS):
        n_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        pages = perm[page_cursor : page_cursor + n_pages]
        page_cursor += n_pages
        pos = torch.arange(seq_len, device=device)
        slots = pages[pos // PAGE_SIZE] * PAGE_SIZE + (pos % PAGE_SIZE)
        sparse_indices[t, :seq_len] = slots.to(torch.int32)

    return dict(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        sparse_indices=sparse_indices,
        sm_scale=SM_SCALE,
    )


def naive_fp32(w):
    """Slow, obviously-correct fp32 reference (gather -> fp32 matmul -> softmax)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ckv_flat = w["ckv_cache"].reshape(-1, D_CKV)
    kpe_flat = w["kpe_cache"].reshape(-1, D_KPE)
    out = torch.zeros(T, H, D_CKV, device=w["q_nope"].device, dtype=torch.float32)
    for t in range(T):
        idx = w["sparse_indices"][t]
        ids = idx[idx >= 0].long()
        k_nope = ckv_flat[ids].float()          # (N, 512)
        k_pe = kpe_flat[ids].float()            # (N, 64)
        scores = (
            w["q_nope"][t].float() @ k_nope.T + w["q_pe"][t].float() @ k_pe.T
        ) * w["sm_scale"]                       # (H, N)
        probs = torch.softmax(scores.double(), dim=-1).float()
        out[t] = probs @ k_nope
    return out


def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def cuda_event_time_us(fn, warmup=25, iters=200):
    """Device time: CUDA events around one invocation, median over `iters`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def profiler_kernel_span_us(fn, warmup=25, iters=50):
    """Device time via the profiler: per-iteration span from first kernel start
    to last kernel end (nsys-style kernel duration, no host launch overhead)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    evts = []
    for e in prof.events():
        if str(getattr(e, "device_type", "")).endswith("CUDA") and e.device_time > 0:
            evts.append((e.time_range.start, e.time_range.end, e.key))
    if not evts:
        return float("nan"), 0
    total = sum(e[1] - e[0] for e in evts)
    return total / iters, len(evts) // iters
