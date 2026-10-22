"""Run fa_fwd (built with -DFA_MODE=9) and print per-phase cycle breakdown, summed over all CTAs."""
import ctypes
import numpy as np
import torch
import common as C
from fa_fwd import FaFwd, _load

q, k, v, cu = C.make_inputs()
fa = FaFwd(C.SEQLENS, C.H)
out = fa(q, k, v)
torch.cuda.synchronize()
lib = _load()
lib.fa_fwd_prof_dump.restype = ctypes.c_int
lib.fa_fwd_prof_dump.argtypes = [ctypes.c_void_p, ctypes.c_int]
N = 16
nb = fa.n
buf = (ctypes.c_ulonglong * (nb * 2 * N))()
rc = lib.fa_fwd_prof_dump(buf, nb * 2 * N)
a = np.frombuffer(buf, dtype=np.uint64).reshape(nb, 2, N).astype(np.float64)
mma = a[:, 0, :].sum(0); smx = a[:, 1, :].sum(0)
names_m = {0: "prologue(Q+K0 wait)", 2: "QK issue", 3: "K wait", 1: "sfree wait", 4: "sready wait(QK_j done)",
           5: "TMA issue", 6: "pv wait(PV_j-1 done)", 7: "pready wait(softmax_j)", 8: "V wait", 9: "PV issue"}
names_s = {11: "smx sready wait", 12: "smx tmem ld S", 13: "smx math", 14: "smx pv wait", 10: "smx P write+publish"}
tot = mma[15]
nsm = 148
print(f"sum over {nb} CTAs of MMA-thread wall cycles: {tot:.3e}  -> per-SM {tot/nsm:.3e} cyc = {tot/nsm/1.965e6:.3f} ms @1.965GHz")
print("MMA thread (sum cycles, % of MMA-thread wall):")
for i, n in names_m.items():
    print(f"  {n:28s} {mma[i]:12.3e}  {100*mma[i]/tot:5.1f}%")
st = sum(smx[i] for i in names_s)
print(f"softmax warp0 thread0 (sum cycles, % of softmax loop total {st:.3e}):")
for i, n in names_s.items():
    print(f"  {n:28s} {smx[i]:12.3e}  {100*smx[i]/st:5.1f}%")
print("PROF_DONE")
