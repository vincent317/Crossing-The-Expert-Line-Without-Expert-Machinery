"""Lean CAKE-ref probe with progress timestamps (diagnose flashinfer first-call stalls)."""

import os
import sys
import time

T0 = time.time()


def log(*a):
    print('[%7.1fs]' % (time.time() - T0), *a, flush=True)


import torch
log('torch', torch.__version__, torch.cuda.get_device_name(0))
import flashinfer
log('flashinfer', flashinfer.__version__)
log('cubin env', {k: v for k, v in os.environ.items() if 'FLASHINFER' in k})

from kda_common import make_inputs, kda_naive_fp32, max_abs_err, device_time_ms, profiler_kernel_ms

d = make_inputs()
q, k, v, g, beta = d['q'], d['k'], d['v'], d['g'], d['beta']
A_log, dt_bias, state0, cu = d['A_log'], d['dt_bias'], d['state'], d['cu_seqlens']
lens = cu[1:] - cu[:-1]
seq_order = torch.argsort(lens, descending=True).to(torch.int32).contiguous()
log('inputs ready; seq_order', seq_order.tolist())

o_nv, s_nv = kda_naive_fp32(q, k, v, g, beta, A_log, dt_bias, state0, cu)
torch.cuda.synchronize()
log('naive fp32 done |o|max=%.6f' % o_nv.abs().max().item())


def call(st, out=None, use_so=True, lb=None):
    return flashinfer.recurrent_kda(
        q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
        initial_state=st, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        lower_bound=lb, cu_seqlens=cu, output=out,
        seq_order=seq_order if use_so else None)


best = None
for layout in ('vk', 'kv'):
    for sdt in (torch.bfloat16, torch.float32):
        st = (state0 if layout == 'kv' else state0.transpose(-1, -2)).to(sdt).contiguous()
        log('--> calling CAKE ref layout=%s sdt=%s' % (layout, sdt))
        try:
            r = call(st.clone())
            torch.cuda.synchronize()
        except Exception as e:
            log('    EXC', repr(e)[:220])
            continue
        s = r[1].float()
        s = s if layout == 'kv' else s.transpose(-1, -2)
        oe, se = max_abs_err(r[0], o_nv), max_abs_err(s, s_nv)
        log('    out_err=%.5e state_err=%.5e |out|max=%.5f' % (oe, se, r[0].float().abs().max().item()))
        if best is None or oe < best[0]:
            best = (oe, se, layout, sdt)

assert best is not None, 'no working CAKE ref configuration'
E_ref, SE, LAYOUT, SDT = best
log('=' * 66)
log('CAKE ref config: layout=%s dtype=%s' % (LAYOUT, SDT))
log('E_ref = max|CAKE_ref - naive_fp32| = %.6e' % E_ref)
log('tol   = 7 * E_ref                  = %.6e' % (7 * E_ref))
log('state err                          = %.6e' % SE)

st = (state0 if LAYOUT == 'kv' else state0.transpose(-1, -2)).to(SDT).contiguous()
st0 = st.clone()
out_buf = torch.empty_like(v)
for use_so in (True, False):
    fn = lambda: call(st, out=out_buf, use_so=use_so)
    m, ts = device_time_ms(fn, warmup=20, iters=50, rounds=5, reset=lambda: st.copy_(st0))
    log('REF device time seq_order=%-5s : %.4f ms (events)  rounds=%s' % (use_so, m, ['%.4f' % x for x in ts]))
    pm, names = profiler_kernel_ms(fn)
    log('REF kernel-sum  seq_order=%-5s : %.4f ms' % (use_so, pm))
    for nm, (us, cnt) in sorted(names.items(), key=lambda kv: -kv[1][0])[:5]:
        log('      %8.1f us x%d  %s' % (us, cnt, nm[:80]))
log('done')
