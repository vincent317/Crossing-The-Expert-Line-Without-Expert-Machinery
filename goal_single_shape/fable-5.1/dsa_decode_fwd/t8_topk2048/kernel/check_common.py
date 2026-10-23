"""Accuracy helpers: max abs error of a kernel result against the naive reference."""

import torch


def normalize_result(res):
    """Accept (output, lse) tuples/lists or dicts with 'output'/'lse' keys."""
    if isinstance(res, dict):
        return res["output"], res["lse"]
    out, lse = res[0], res[1]
    return out, lse


def max_errors(res, ref_out, ref_lse):
    out, lse = normalize_result(res)
    out = out.to(torch.float32)
    lse = lse.to(torch.float32)
    assert out.shape == ref_out.shape, (out.shape, ref_out.shape)
    assert lse.shape == ref_lse.shape, (lse.shape, ref_lse.shape)
    finite = torch.isfinite(ref_lse)
    err_out = (out - ref_out).abs().max().item()
    err_lse = (lse[finite] - ref_lse[finite]).abs().max().item() if finite.any() else 0.0
    nan_out = int(torch.isnan(out).sum().item())
    return err_out, err_lse, nan_out
