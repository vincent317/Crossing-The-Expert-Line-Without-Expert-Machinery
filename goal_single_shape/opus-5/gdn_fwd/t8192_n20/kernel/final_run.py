import sys, torch
sys.path.insert(0, ".")
from gdn_data import make_inputs, T, N, HQK, HV, D, SEQ_LENS
from gdn_naive import gdn_naive_fp32
from gdn_triton import GDN
from bench_util import device_time_ms, device_time_amortized_ms, maxerr
PUBLISHED_REF_MS = 0.1255
CFGS = eval(sys.argv[1]) if len(sys.argv) > 1 else [dict(C=32, split=4)]

d = make_inputs()
args = (d["q"], d["k"], d["v"], d["a"], d["b"], d["A_log"], d["dt_bias"],
        d["state"], d["cu_seqlens"])
print("GPU:", torch.cuda.get_device_name(0), "SMs",
      torch.cuda.get_device_properties(0).multi_processor_count, flush=True)
print(f"workload T={T} N={N} lens={SEQ_LENS}", flush=True)
import torch.nn.functional as _F
_g = -d["A_log"].float().exp() * _F.softplus(d["a"].float() + d["dt_bias"].float())
print(f"gate: g in [{_g.min().item():.4f}, {_g.max().item():.4f}]  "
      f"max |sum g| over any 64 consecutive tokens = "
      f"{_g.abs().cumsum(0).diff(n=64, dim=0).abs().max().item():.2f}  "
      f"beta in [{torch.sigmoid(d['b'].float()).min().item():.4f}, "
      f"{torch.sigmoid(d['b'].float()).max().item():.4f}]", flush=True)
o_nv, s_nv = gdn_naive_fp32(*args); torch.cuda.synchronize()
print(f"\n[1] fp32 naive anchor |o|max={o_nv.abs().max().item():.6f} "
      f"|s|max={s_nv.abs().max().item():.6f}", flush=True)
E_o = E_s = ref_med = None
import os
try:
    if os.environ.get("NO_REF"): raise RuntimeError("ref skipped by NO_REF")
    from gdn_ref import Ref
    ref = Ref(HQK, HV, D)
    E_o, E_s = ref.calibrate(*args, o_nv, s_nv)
    print(f"\n[2] REFERENCE variant={ref.variant}")
    print(f"    E_ref(out)={E_o:.8f}  tol(out)=7*E_ref={7*E_o:.8f}")
    print(f"    E_ref(state)={E_s:.8f}  tol(state)=7*E_ref={7*E_s:.8f}", flush=True)
    for nm, inc in (("ref kernel only", False), ("ref incl gate", True)):
        f = ref.make_callable(*args, include_gate=inc)
        med, mn, p10 = device_time_ms(f)
        am = device_time_amortized_ms(f)
        print(f"    {nm:16s} per-call span median={med*1e3:8.2f}us min={mn*1e3:8.2f}us | "
              f"amortized={am*1e3:8.2f}us", flush=True)
        if not inc: ref_med = min(med * 1e3, am * 1e3)
except Exception as e:
    import traceback; traceback.print_exc()

print("\n[3] OUR KERNEL", flush=True)
best = None
for cfg in CFGS:
    try:
        g = GDN(T, N, HQK, HV, D, **cfg)
        o, s = g(*args); torch.cuda.synchronize()
        eo, es = maxerr(o, o_nv), maxerr(s, s_nv)
        tot = device_time_ms(lambda: g(*args))[0] * 1e3
        amo = device_time_amortized_ms(lambda: g(*args)) * 1e3
        tot = min(tot, amo)
        twy = device_time_ms(lambda: g.wy(*args))[0] * 1e3
        tsc = device_time_ms(lambda: g.scan(*args))[0] * 1e3
        tat = 0.0 if g.fused else device_time_ms(lambda: g.attn(*args))[0] * 1e3
        st = g.stats()
        print(f"  {cfg} err_o={eo:.6f} err_s={es:.6f} TOTAL={tot:.2f}us (amort={amo:.2f}) "
              f"(wy={twy:.2f} attn={tat:.2f} scan={tsc:.2f}) " +
              " ".join(f"{k}:r{v[0]}/sp{v[1]}/sm{v[2]//1024}K" for k, v in st.items()),
              flush=True)
        okacc = (E_o is None) or (eo <= 7*E_o and (E_s != E_s or es <= 7*E_s))
        if okacc and (best is None or tot < best[0]):
            best = (tot, cfg, eo, es)
    except Exception as e:
        print(f"  {cfg} FAILED {type(e).__name__}: {str(e)[:500]}", flush=True)

print("\n[4] SUMMARY", flush=True)
if best:
    tot, cfg, eo, es = best
    print(f"best config            : {cfg}")
    print(f"our max_err(out)       : {eo:.8f}" + (f"  <= tol {7*E_o:.8f} -> {'PASS' if eo<=7*E_o else 'FAIL'}" if E_o else ""))
    print(f"our max_err(state)     : {es:.8f}" + (f"  <= tol {7*E_s:.8f} -> {'PASS' if es<=7*E_s else 'FAIL'}" if E_s else ""))
    print(f"our device time        : {tot:.2f} us")
    if ref_med:
        print(f"measured ref device t  : {ref_med:.2f} us  (/1.2 = {ref_med/1.2:.2f} us)  speedup = {ref_med/tot:.2f}x")
    print(f"published ref device t : {PUBLISHED_REF_MS*1e3:.2f} us  (/1.2 = {PUBLISHED_REF_MS*1e3/1.2:.2f} us)  speedup = {PUBLISHED_REF_MS*1e3/tot:.2f}x")
    print("TARGET MET" if tot <= PUBLISHED_REF_MS*1e3/1.2 else "TARGET NOT MET")
