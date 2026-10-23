"""Single place that knows how to build every implementation under test."""

import torch

# Best configuration found by the sweep in final_run.py on B200:
#   block_n=32   KV index slots per CTA (11 active CTAs for a 337-row request)
#   num_warps=8  256 threads
#   head_groups=2  8 heads per CTA - halves the Q bytes and the output bytes
#   tiles_per_cta=2  halves the grid (shorter launch ramp, cheaper barrier)
#   fused=True   single kernel: atomic accumulate + one barrier + divide
BEST_CUDA = dict(block_n=32, num_warps=8, head_groups=2, tiles_per_cta=2, fused=True)
BEST_TRITON = dict(block_n=32, splits=64, num_warps=4, num_stages=2)


def build_reference(w):
    """The task's reference: the KDA 0.5 CuTe-DSL contest kernel, as a black box.

    Returns a zero-argument callable so the timed region holds nothing but the op.
    """
    import ref_contest

    return ref_contest.make_callable(w)


def build_cuda(w, verbose=False, **over):
    from dsa_cuda import DsaCuda

    cfg = dict(BEST_CUDA)
    cfg.update(over)
    return DsaCuda(w, verbose=verbose, **cfg)


def build_triton(w, **over):
    from dsa_triton import DsaTriton

    cfg = dict(BEST_TRITON)
    cfg.update(over)
    return DsaTriton(w, **cfg)


def build_all(w, verbose=False):
    impls = {}
    for name, fn in (("cuda", build_cuda), ("triton", build_triton)):
        try:
            impls[name] = fn(w, verbose=verbose) if name == "cuda" else fn(w)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {name} impl unavailable: {type(exc).__name__}: {exc}")
    return impls
