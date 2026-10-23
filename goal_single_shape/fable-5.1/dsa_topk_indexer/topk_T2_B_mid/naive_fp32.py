"""FP32 naive implementation of DSA top-k indexer.

Reference semantics (no GPU-specific ops, pure Python/PyTorch):
  For each batch b, token t:
    page = block_table[b, t // 64]
    pg_off = t % 64
    k_fp8 = k_cache[page, pg_off, 0, 0:128]   (fp8_e4m3 stored as int8)
    k_scale = k_cache[page, pg_off, 0, 128:132] (fp32)
    k_true[d] = to_float(k_fp8[d]) * k_scale
    score = sum_h( relu(q[b,h] @ k_true) * weights[b,h] )
  Output: top min(2048, seq_len[b]) global token ids (page*64+pg_off), rest -1.
"""

import torch
import struct


def fp8e4m3fn_to_float32_table():
    """Build a lookup table for fp8_e4m3fn -> float32 conversion (256 entries)."""
    # We use PyTorch's native fp8 conversion via a small tensor
    idx = torch.arange(256, dtype=torch.uint8)
    # Reinterpret as fp8_e4m3fn, then cast to float32
    fp8_vals = idx.view(torch.float8_e4m3fn)
    return fp8_vals.float()  # shape (256,)


# Build LUT once at module load
_FP8_LUT = fp8e4m3fn_to_float32_table()  # cpu tensor, shape (256,)


def run_naive(q_fp8, k_cache, weights, seq_lens, block_table):
    """
    Args:
        q_fp8:       (B, 64, 128) torch.float8_e4m3fn
        k_cache:     (num_pages, 64, 1, 132) torch.int8
        weights:     (B, 64) torch.float32
        seq_lens:    (B,) torch.int32
        block_table: (B, max_pages) torch.int32
    Returns:
        topk_indices: (B, 2048) torch.int32
    """
    device = q_fp8.device
    lut = _FP8_LUT.to(device)

    B = q_fp8.shape[0]
    H, D = 64, 128
    PAGE_SZ = 64
    TOPK = 2048

    # Reinterpret q_fp8 as uint8 for indexing into LUT
    q_u8 = q_fp8.view(torch.uint8)  # (B, H, D)
    # Convert q to float32: (B, H, D)
    q_f32 = lut[q_u8.long()]  # (B, H, D)

    # Reinterpret k_cache for fp8 data and scale extraction
    # k_cache shape: (num_pages, PAGE_SZ, 1, 132) int8
    # First 128 bytes of last dim = fp8 data, last 4 bytes = float32 scale
    k_cache_u8 = k_cache.view(torch.uint8)  # (num_pages, PAGE_SZ, 1, 132)
    k_fp8_u8 = k_cache_u8[..., :D]  # (num_pages, PAGE_SZ, 1, D)

    # Extract float32 scale from bytes 128:132
    # k_cache is (num_pages, PAGE_SZ, 1, 132) contiguous int8
    # scale bytes at offset 128 per (page, offset) row
    num_pages = k_cache.shape[0]
    # Flatten to (num_pages * PAGE_SZ, 132)
    k_flat = k_cache.view(num_pages * PAGE_SZ, 132)
    # Extract scale: last 4 bytes reinterpreted as float32
    scale_bytes = k_flat[:, 128:132].contiguous()  # (N, 4) int8
    k_scales = scale_bytes.view(torch.float32).view(num_pages * PAGE_SZ)  # (N,)
    # Reshape back
    k_scales = k_scales.view(num_pages, PAGE_SZ)  # (num_pages, PAGE_SZ)

    # Convert k fp8 data to float32
    k_fp8_flat = k_fp8_u8.view(-1, D)  # (num_pages * PAGE_SZ, D)
    k_f32_flat = lut[k_fp8_flat.long()]  # (num_pages * PAGE_SZ, D)
    k_f32 = k_f32_flat.view(num_pages, PAGE_SZ, D)  # (num_pages, PAGE_SZ, D)
    # Multiply by per-token scale: k_true[page, off, d] = k_f32[page,off,d] * scale[page,off]
    k_true = k_f32 * k_scales.unsqueeze(-1)  # (num_pages, PAGE_SZ, D)

    topk_out = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)

    for b in range(B):
        seq_len = seq_lens[b].item()
        k_out = min(TOPK, seq_len)

        if seq_len == 0:
            continue

        # Gather k vectors for all tokens of batch b
        # token t -> page block_table[b, t//64], offset t%64
        t_indices = torch.arange(seq_len, device=device)
        pages = block_table[b, t_indices // PAGE_SZ]      # (seq_len,)
        offsets = t_indices % PAGE_SZ                       # (seq_len,)

        # k vectors: (seq_len, D)
        k_batch = k_true[pages, offsets]  # (seq_len, D)

        # Dot products: q_f32[b] (H, D) x k_batch.T (D, seq_len) -> (H, seq_len)
        dots = torch.mm(q_f32[b], k_batch.T)  # (H, seq_len)

        # relu + weighted sum
        relu_dots = torch.relu(dots)  # (H, seq_len)
        scores = (weights[b].unsqueeze(1) * relu_dots).sum(0)  # (seq_len,)

        # Global token ids
        global_ids = pages * PAGE_SZ + offsets  # (seq_len,)

        # Top-k selection
        if seq_len <= TOPK:
            topk_out[b, :seq_len] = global_ids.int()
        else:
            _, topk_idx = torch.topk(scores, k_out)
            topk_out[b, :k_out] = global_ids[topk_idx].int()

    return topk_out
