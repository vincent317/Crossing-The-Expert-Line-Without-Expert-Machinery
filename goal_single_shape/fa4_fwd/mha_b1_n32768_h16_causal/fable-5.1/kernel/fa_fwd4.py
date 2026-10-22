import torch, ctypes, os
QTILE, KTILE = 128, 128

def build_work_items(seqlens, H, qtile=QTILE, ktile=KTILE, device="cuda"):
    items = []
    off = 0
    for L in seqlens:
        for h in range(H):
            for qi in range((L + qtile - 1) // qtile):
                nk = min(((qi + 1) * qtile) // ktile, (L + ktile - 1) // ktile)
                items.append((off + qi * qtile, off, L, h, qi, nk))
        off += L
    items.sort(key=lambda x: -x[5])
    w = torch.tensor([(x[0], x[1], x[2], x[3], x[4]) for x in items], dtype=torch.int32, device=device)
    return w

def fa_fwd(lib_path, q, k, v, cu_q, cu_k, scale):
    lib = ctypes.CDLL(lib_path)
    T, H, D = q.shape
    seqlens = [(cu_q[i+1] - cu_q[i]).item() for i in range(len(cu_q) - 1)]
    work = build_work_items(seqlens, H)
    out = torch.zeros_like(q)
    rc = lib.fa_fwd_launch(
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(work.data_ptr()),
        ctypes.c_int(work.shape[0]), ctypes.c_int(T), ctypes.c_int(H),
        ctypes.c_float(scale), ctypes.c_void_p(0)
    )
    if rc != 0:
        raise RuntimeError(f"fa_fwd_launch rc={rc}")
    return out
