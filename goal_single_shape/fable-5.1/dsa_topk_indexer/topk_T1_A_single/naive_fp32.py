"""naive_fp32.py — Pure fp32 naive DSA top-k indexer (slow, for correctness)."""
import torch

PAGE_SIZE = 64
HEAD_DIM  = 128
K_STRIDE  = 132   # 128 fp8 bytes + 4 float32 bytes


def run(q_fp8, k_cache, weights, seq_lens, block_table, topk_size=2048):
    """
    For each batch b, token t < seq_lens[b]:
        page    = block_table[b, t // PAGE_SIZE]
        off     = t % PAGE_SIZE
        k_fp8   = k_cache[page, off, 0, :128].view(fp8_e4m3fn).float()
        scale   = k_cache[page, off, 0, 128:132].view(float32)
        k_f32   = k_fp8 * scale
        score_t = sum_h( weights[b,h] * relu( dot(q[b,h].float(), k_f32) ) )
    Returns topk_indices (B, topk_size) int32, -1 padded.
    """
    B      = q_fp8.shape[0]
    device = q_fp8.device
    out    = torch.full((B, topk_size), -1, dtype=torch.int32, device=device)
    q_f32  = q_fp8.float()   # (B, 64, 128)

    for b in range(B):
        sl = int(seq_lens[b].item())
        K  = min(topk_size, sl)
        if sl == 0:
            continue

        scores = torch.zeros(sl, device=device, dtype=torch.float32)

        for t in range(sl):
            pg  = int(block_table[b, t // PAGE_SIZE].item())
            off = t % PAGE_SIZE

            # fp8 key vector
            k_int8 = k_cache[pg, off, 0, :HEAD_DIM]            # (128,) int8
            k_fp8v = k_int8.view(torch.float8_e4m3fn)           # (128,) fp8
            k_f32  = k_fp8v.float()                             # (128,) f32

            # per-token scale (4 bytes → 1 float32)
            s_int8 = k_cache[pg, off, 0, HEAD_DIM:K_STRIDE]    # (4,) int8
            scale  = float(s_int8.view(torch.float32).item())

            k_scaled = k_f32 * scale                            # (128,) f32

            dots     = (q_f32[b] * k_scaled).sum(dim=-1)        # (64,)
            scores[t] = (weights[b] * dots.clamp(min=0.0)).sum()

        topk_vals, topk_lids = scores.topk(K)
        for ki in range(K):
            t   = int(topk_lids[ki].item())
            pg  = int(block_table[b, t // PAGE_SIZE].item())
            off = t % PAGE_SIZE
            out[b, ki] = pg * PAGE_SIZE + off

    return out
