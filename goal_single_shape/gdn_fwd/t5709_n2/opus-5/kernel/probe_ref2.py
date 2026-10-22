"""Systematic probe of the CuTe-DSL reference calling convention.

The reference runs (stable 113 us device time) but returns an all-NaN output for
our inputs, so this bisects the convention: preallocated output buffers, no
initial state, CPU cu_seqlens, single sequence, and a minimal smoke test whose
expected value is computed in closed form.
"""

import traceback

import torch
import torch.nn.functional as F

from gdn_data import D, HQ, HV, SCALE, gates, make_inputs
from ref_flashinfer import get_fn


def nanfrac(t):
    t = t.float()
    return float((~torch.isfinite(t)).float().mean())


def show(tag, out):
    if isinstance(out, (tuple, list)):
        o, s = out[0], out[1]
        print("  %-42s out nan%%=%.3f first=%s | state nan%%=%.3f first=%s"
              % (tag, 100 * nanfrac(o), ["%.4f" % x for x in o.flatten()[:3].float().tolist()],
                 100 * nanfrac(s), ["%.4f" % x for x in s.flatten()[:3].float().tolist()]),
              flush=True)
    else:
        print("  %-42s out nan%%=%.3f first=%s"
              % (tag, 100 * nanfrac(out),
                 ["%.4f" % x for x in out.flatten()[:3].float().tolist()]), flush=True)


def naive_small(q, k, v, g, beta, state, scale):
    """fp32 reference for one packed sequence, GQA expand by repeat_interleave."""
    T = q.shape[0]
    hv = v.shape[1]
    grp = hv // q.shape[1]
    qn = (F.normalize(q.float(), dim=-1) * scale).repeat_interleave(grp, dim=1)
    kn = F.normalize(k.float(), dim=-1).repeat_interleave(grp, dim=1)
    S = state.clone().float()
    o = torch.empty(T, hv, v.shape[-1], device=q.device, dtype=torch.float32)
    for t in range(T):
        S = S * torch.exp(g[t]).view(hv, 1, 1)
        u = beta[t].view(hv, 1) * (v[t].float() - torch.bmm(S, kn[t].unsqueeze(-1)).squeeze(-1))
        S = S + u.unsqueeze(-1) * kn[t].unsqueeze(-2)
        o[t] = torch.bmm(S, qn[t].unsqueeze(-1)).squeeze(-1)
    return o, S


def main():
    fn = get_fn()
    inp = make_inputs()
    g, beta = gates(inp["a"], inp["b"], inp["A_log"], inp["dt_bias"])
    q, k, v = inp["q"], inp["k"], inp["v"]
    st = inp["state"]
    cu32 = inp["cu_seqlens"].to(torch.int32)
    T = q.shape[0]

    base = dict(q=q, k=k, v=v, g=g, beta=beta, scale=SCALE, initial_state=st,
                output_final_state=True, cu_seqlens=cu32, use_qk_l2norm_in_kernel=True)

    print("=== full workload variants ===", flush=True)
    trials = [
        ("baseline", {}),
        ("initial_state=None", {"initial_state": None}),
        ("scale=None", {"scale": None}),
        ("cu on cpu", {"cu_seqlens": cu32.cpu()}),
        ("cu_seqlens=None(single seq)", {"cu_seqlens": None}),
        ("output_final_state=False", {"output_final_state": False}),
        ("backend=flashinfer", {"backend": "flashinfer"}),
        ("use_cp=False", {"use_cp": False}),
    ]
    for tag, over in trials:
        kw = dict(base)
        kw.update(over)
        try:
            show(tag, fn(**kw))
        except Exception as e:
            print("  %-42s FAILED %r" % (tag, repr(e)[:150]), flush=True)

    print("=== preallocated output buffers ===", flush=True)
    try:
        ob = torch.full((T, HV, D), float("inf"), dtype=torch.bfloat16, device=q.device)
        sb = torch.full_like(st, float("inf"))
        kw = dict(base)
        kw.update(output=ob, output_state=sb)
        out = fn(**kw)
        print("  returned-is-ob:", out[0].data_ptr() == ob.data_ptr() if isinstance(out, (tuple, list))
              else out.data_ptr() == ob.data_ptr())
        show("prealloc", (ob, sb))
    except Exception:
        traceback.print_exc()

    print("=== minimal smoke test (1 seq, T=64, zero gates) ===", flush=True)
    gen = torch.Generator(device="cpu").manual_seed(7)
    t2 = 64
    q2 = torch.randn(t2, HQ, D, generator=gen).cuda().bfloat16()
    k2 = torch.randn(t2, HQ, D, generator=gen).cuda().bfloat16()
    v2 = torch.randn(t2, HV, D, generator=gen).cuda().bfloat16()
    g2 = torch.zeros(t2, HV, device="cuda")
    b2 = torch.full((t2, HV), 0.5, device="cuda")
    s2 = torch.zeros(1, HV, D, D, device="cuda")
    cu2 = torch.tensor([0, t2], dtype=torch.int32, device="cuda")
    o_exp, s_exp = naive_small(q2, k2, v2, g2, b2, s2[0], SCALE)
    print("  naive small: |o| mean %.5f max %.5f" % (o_exp.abs().mean(), o_exp.abs().max()))
    for tag, over in (("smoke", {}), ("smoke l2norm=False", {"use_qk_l2norm_in_kernel": False}),
                      ("smoke state=None", {"initial_state": None})):
        kw = dict(q=q2, k=k2, v=v2, g=g2, beta=b2, scale=SCALE, initial_state=s2,
                  output_final_state=True, cu_seqlens=cu2, use_qk_l2norm_in_kernel=True)
        kw.update(over)
        try:
            out = fn(**kw)
            o = out[0].float().reshape(o_exp.shape)
            print("  %-24s nan%%=%.2f  max|o-naive|=%.4e" % (tag, 100 * nanfrac(o),
                  (o - o_exp).abs().max().item()), flush=True)
        except Exception as e:
            print("  %-24s FAILED %r" % (tag, repr(e)[:150]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
