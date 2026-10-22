"""Triton kernel for DSA top-k indexer.

Uses triton JIT to implement the score computation using warp-level parallelism.
"""

import torch
import triton
import triton.language as tl

H_DIM = 64
D_DIM = 128
PAGE_SZ = 64
K_STRIDE = 132  # 128 fp8 bytes + 4 float32 bytes
TOPK_K = 2048


@triton.jit
def score_kernel(
    q_ptr,      # (B, H, D) int8 (fp8 bytes)
    k_ptr,      # (num_pages*PAGE_SZ, K_STRIDE) int8
    w_ptr,      # (B, H) float32
    sl_ptr,     # (B,) int32
    bt_ptr,     # (B, max_pages) int32
    sc_ptr,     # (B, 4096) float32
    id_ptr,     # (B, 4096) int32
    B, max_pages,
    H: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,   # PAGE_SZ
    KS: tl.constexpr,   # K_STRIDE
    BLOCK_H: tl.constexpr,   # = H, parallelise over heads
):
    b = tl.program_id(0)
    pg_idx = tl.program_id(1)

    sl = tl.load(sl_ptr + b)
    t_base = pg_idx * PS
    if t_base >= sl:
        return

    page = tl.load(bt_ptr + b * max_pages + pg_idx)

    # Load q[b]: (H, D) as flat vector, then reshape
    q_off = b * H * D + tl.arange(0, H * D)
    q_flat = tl.load(q_ptr + q_off).to(tl.float32)   # (H*D,)

    # Load weights[b]: (H,)
    w_off = b * H + tl.arange(0, H)
    w_h = tl.load(w_ptr + w_off)                      # (H,) float32

    # Process each token in this page sequentially (mask out-of-bounds)
    for loc in tl.range(0, PS, 1):
        t = t_base + loc
        active = t < sl
        kp = (page * PS + loc) * KS
        # Load k fp8 (D bytes) as int8 -> float32
        k_off = tl.arange(0, D)
        k_fp8 = tl.load(k_ptr + kp + k_off, mask=active).to(tl.float32)
        # Load 4-byte float32 scale stored as 4 int8 bytes
        kp_scale = kp + D
        scale_raw = tl.load(tl.cast(k_ptr + kp_scale, tl.pointer_type(tl.int32)), mask=active)
        k_scale = scale_raw.to(tl.float32, bitcast=True)

        k_true = k_fp8 * k_scale   # (D,)

        # Compute H dot products: q_flat reshaped to (H, D)
        q_mat = tl.reshape(q_flat, (H, D))
        dots = tl.sum(q_mat * k_true[None, :], axis=1)   # (H,)
        relu_d = tl.maximum(dots, 0.0)
        score = tl.sum(relu_d * w_h)

        gid = page * PS + loc
        tl.store(sc_ptr + b * 4096 + t, score, mask=active)
        tl.store(id_ptr + b * 4096 + t, gid, mask=active)


def run_triton_score(q_fp8, k_cache, weights, seq_lens, block_table):
    """
    q_fp8: (B, H, D) float8_e4m3fn
    k_cache: (num_pages, PAGE_SZ, 1, K_STRIDE) int8
    weights: (B, H) float32
    seq_lens: (B,) int32
    block_table: (B, max_pages) int32
    Returns: scores (B, 4096) float32, ids (B, 4096) int32
    """
    B = q_fp8.shape[0]
    max_pages = block_table.shape[1]
    device = q_fp8.device

    # Reinterpret q as int8 for triton
    q_i8 = q_fp8.view(torch.uint8).view(torch.int8).contiguous()
    # Flatten k_cache to (num_pages * PAGE_SZ, K_STRIDE)
    k_flat = k_cache.reshape(-1, K_STRIDE).contiguous()

    scores = torch.full((B, 4096), -1e38, dtype=torch.float32, device=device)
    ids = torch.full((B, 4096), -1, dtype=torch.int32, device=device)

    grid = (B, max_pages)
    score_kernel[grid](
        q_i8, k_flat, weights, seq_lens, block_table,
        scores, ids,
        B, max_pages,
        H=H_DIM, D=D_DIM, PS=PAGE_SZ, KS=K_STRIDE,
        BLOCK_H=H_DIM,
    )
    return scores, ids


def run(q_fp8, k_cache, weights, seq_lens, block_table):
    """Full pipeline: score + topk."""
    B = q_fp8.shape[0]
    max_sl = seq_lens.max().item()

    scores, ids = run_triton_score(q_fp8, k_cache, weights, seq_lens, block_table)

    # Topk over [0, max_sl) for all batches
    sc_slice = scores[:, :max_sl]  # (B, max_sl)
    _, topk_idx = torch.topk(sc_slice, TOPK_K, dim=1)  # (B, 2048)
    topk_ids = torch.gather(ids[:, :max_sl], 1, topk_idx)  # (B, 2048)

    # Fix: for batches with seq_len < 2048, fill excess with -1
    for b in range(B):
        sl = seq_lens[b].item()
        if sl < TOPK_K:
            topk_ids[b, sl:] = -1

    return topk_ids
