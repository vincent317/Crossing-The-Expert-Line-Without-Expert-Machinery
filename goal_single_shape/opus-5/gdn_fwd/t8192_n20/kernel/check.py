"""Accuracy check: fp32 naive anchor -> E_ref from the reference -> tol = 7*E_ref
-> compare our Triton kernel against the fp32 naive result.

Run:  python -u check.py [C] [split]
"""
import sys, torch
sys.path.insert(0, ".")
from gdn_data import make_inputs, T, N, HQK, HV, D, SEQ_LENS
from gdn_naive import gdn_naive_fp32
from gdn_triton import GDN
from bench_util import maxerr

C = int(sys.argv[1]) if len(sys.argv) > 1 else 64
SPLIT = int(sys.argv[2]) if len(sys.argv) > 2 else 4

d = make_inputs()
args = (d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"],
        d["state"], d["cu_seqlens"])

print(f"workload: T={T} N={N} lens={SEQ_LENS}")
print(f"          q/k=({T},{HQK},{D}) v=({T},{HV},{D}) a/b=({T},{HV}) "
      f"state=({N},{HV},{D},{D}) bf16/fp32")

o_nv, s_nv = gdn_naive_fp32(*args)
torch.cuda.synchronize()
print(f"\n[1] fp32 naive recurrence: |o|max={o_nv.abs().max().item():.6f} "
      f"|state|max={s_nv.abs().max().item():.6f}")

E_ref_o = E_ref_s = None
try:
    from gdn_ref import Ref
    ref = Ref(HQK, HV, D)
    E_ref_o, E_ref_s = ref.calibrate(*args, o_nv, s_nv)
    print(f"\n[2] reference vs fp32 naive (variant gqa/l2/scale={ref.variant}):")
    print(f"    E_ref(out)   = {E_ref_o:.8f}")
    print(f"    E_ref(state) = {E_ref_s:.8f}")
    print(f"    tol(out)     = 7 * E_ref = {7*E_ref_o:.8f}")
    print(f"    tol(state)   = 7 * E_ref = {7*E_ref_s:.8f}")
except Exception as e:
    print(f"\n[2] reference unavailable: {type(e).__name__}: {str(e)[:300]}")

g = GDN(T, N, HQK, HV, D, C=C, split=SPLIT, nsq=2, st_scan=3)
o, s = g(*args)
torch.cuda.synchronize()
eo, es = maxerr(o, o_nv), maxerr(s, s_nv)
print(f"\n[3] our Triton kernel (C={C}, split={SPLIT}) vs fp32 naive:")
print(f"    max_err(out)   = {eo:.8f}")
print(f"    max_err(state) = {es:.8f}")
if E_ref_o is not None:
    ok = eo <= 7 * E_ref_o and es <= 7 * E_ref_s
    print(f"    out   {eo:.8f} <= tol {7*E_ref_o:.8f} : {'PASS' if eo<=7*E_ref_o else 'FAIL'}")
    print(f"    state {es:.8f} <= tol {7*E_ref_s:.8f} : {'PASS' if es<=7*E_ref_s else 'FAIL'}")
    print(f"\nACCURACY: {'PASS' if ok else 'FAIL'}")
