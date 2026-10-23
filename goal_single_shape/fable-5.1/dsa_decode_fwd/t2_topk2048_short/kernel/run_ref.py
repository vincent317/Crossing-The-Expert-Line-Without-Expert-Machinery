"""Black-box loader for the reference implementation (KDA 0.5 CuTe-DSL kernel,
mit-han-lab/mlsys2026-flashinfer-contest-solution @ 7aad0429).

The repository is cloned by job.sh; only its public `run(...)` entry point is
imported and called here. Its source is never read or copied by this project.
"""
import importlib.util
import inspect
import os
import sys
import traceback

import torch

REF_COMMIT = "7aad04296c3d40f0fc6beeafeffd7a548a26dfae"
REF_REPO = os.environ.get("DSA_REF_REPO",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_repo"))
REF_KERNEL = os.path.join(REF_REPO, "kernels", "dsa_attention", "kernel.py")


def load_ref_run():
    """Returns fn(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale) -> (output, lse)."""
    spec = importlib.util.spec_from_file_location("kda05_dsa_attention_kernel", REF_KERNEL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    params = list(inspect.signature(mod.run).parameters)
    dps = "output" in params and "lse" in params  # destination-passing style variant
    print(f"[ref] loaded {os.path.relpath(REF_KERNEL, REF_REPO)}; run params={params}")

    def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        if dps:
            T, H, D = q_nope.shape
            output = torch.empty((T, H, D), dtype=torch.bfloat16, device=q_nope.device)
            lse = torch.empty((T, H), dtype=torch.float32, device=q_nope.device)
            res = mod.run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)
            return (output, lse) if res is None else _unpack(res)
        return _unpack(mod.run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale))

    return run


def _unpack(res):
    if isinstance(res, dict):
        return res["output"], res["lse"]
    out, lse = res
    return out, lse


def format_exc_without_source(exc, limit=2000):
    """Exception summary that lists file:line only, never source text."""
    frames = [f"{os.path.basename(fr.filename)}:{fr.lineno} in {fr.name}"
              for fr in traceback.extract_tb(exc.__traceback__)]
    msg = f"{type(exc).__name__}: {str(exc)[:limit]}"
    return msg + "\n  at " + "\n  at ".join(frames)
