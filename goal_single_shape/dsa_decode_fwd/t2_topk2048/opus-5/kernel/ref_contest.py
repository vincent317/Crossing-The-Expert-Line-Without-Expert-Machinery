"""Black-box driver for the task's reference implementation.

The reference is the KDA 0.5 CuTe-DSL contest kernel
(mit-han-lab/mlsys2026-flashinfer-contest-solution @ 7aad0429,
kernels/dsa_attention/kernel.py). The file is fetched verbatim and imported by
path; its contents are never read, printed or copied - we only call its
published entry point:

    run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
"""

import importlib.util
import os

REF_FILENAME = "ref_dsa_kernel.py"
REF_URL = (
    "https://raw.githubusercontent.com/mit-han-lab/"
    "mlsys2026-flashinfer-contest-solution/7aad0429/kernels/dsa_attention/kernel.py"
)

_MOD = None


def ref_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), REF_FILENAME)


def load_ref():
    global _MOD
    if _MOD is None:
        path = ref_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing; fetch it from {REF_URL}")
        spec = importlib.util.spec_from_file_location("ref_dsa_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD = mod
    return _MOD


def make_callable(w):
    """Return a zero-argument callable that runs the reference on `w`."""
    mod = load_ref()
    q_nope = w["q_nope"]
    q_pe = w["q_pe"]
    ckv = w["ckv_cache"]
    kpe = w["kpe_cache"]
    idx = w["sparse_indices"]
    scale = w["sm_scale"]

    def run():
        return mod.run(q_nope, q_pe, ckv, kpe, idx, scale)

    return run


def as_output_tensor(ret):
    """Normalise whatever `run` returned into the (T, H, D) output tensor."""
    if isinstance(ret, (tuple, list)):
        ret = ret[0]
    return ret
