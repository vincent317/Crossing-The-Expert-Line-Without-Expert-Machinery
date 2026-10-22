"""topk_ext.py — Python wrapper that JIT-compiles topk_kernel.cu."""
import os
import torch
from torch.utils.cpp_extension import load

_ext = None


def _get_ext(verbose=False):
    global _ext
    if _ext is None:
        cu = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topk_kernel.cu")
        _ext = load(
            name="dsa_topk_ext",
            sources=[cu],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100"],
            verbose=verbose,
        )
    return _ext


def run(q_fp8, k_cache, weights, seq_lens, block_table,
        topk_size=2048, verbose=False):
    """Run DSA TopK kernel, returns (B, topk_size) int32 indices, -1 padded."""
    ext   = _get_ext(verbose=verbose)
    max_sl = int(seq_lens.max().item())
    return ext.forward(q_fp8, k_cache, weights, seq_lens, block_table,
                       topk_size, max_sl)
