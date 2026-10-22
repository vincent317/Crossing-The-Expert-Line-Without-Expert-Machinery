"""ctypes wrapper for fa_fwd.cu (compiled to libfa_fwd.so in the job)."""
import ctypes
import os

import torch

import common as C

_lib = None


def _load():
    global _lib
    if _lib is None:
        path = os.environ.get("FA_FWD_SO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfa_fwd.so"))
        _lib = ctypes.CDLL(path)
        _lib.fa_fwd_launch.restype = ctypes.c_int
        _lib.fa_fwd_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                       ctypes.c_void_p]
    return _lib


QTILE = int(os.environ.get("FA_QTILE", "256"))
KTILE = int(os.environ.get("FA_KTILE", "64"))


def build_work_items(seqlens, H, qtile=QTILE, ktile=KTILE, device="cuda"):
    """One item per (doc, head, q_tile): (q_row, k_row, L, head, qi, nk), sorted by descending nk (LPT)."""
    items = []
    off = 0
    for L in seqlens:
        nq = (L + qtile - 1) // qtile
        for qi in range(nq):
            nk = min(((qi + 1) * qtile) // ktile, (L + ktile - 1) // ktile)
            for h in range(H):
                items.append((off + qi * qtile, off, L, h, qi, nk))
        off += L
    items.sort(key=lambda t: -t[5])
    t = torch.tensor(items, dtype=torch.int32)
    return t.to(device).contiguous(), len(items)


class FaFwd:
    def __init__(self, seqlens, H, device="cuda"):
        self.work, self.n = build_work_items(seqlens, H, device=device)
        self.H = H

    def __call__(self, q, k, v, out=None):
        if out is None:
            out = torch.empty_like(q)
        lib = _load()
        rc = lib.fa_fwd_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), self.work.data_ptr(),
                               self.n, q.shape[0], self.H, C.SOFTMAX_SCALE,
                               torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"fa_fwd_launch rc={rc}")
        return out
