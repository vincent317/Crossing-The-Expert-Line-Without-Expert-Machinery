"""Find the argument combination that makes flashinfer dispatch the frozen
SM100 (CAKE) FlashKDA prefill backend, then validate + time it.

`seq_order` is only accepted by that backend, so passing it acts as a cheap
eligibility oracle: ineligible combinations raise immediately instead of
falling through to the slow CuTe DSL path.
"""

import itertools
import time

import torch
import flashinfer

from kda_common import make_inputs, kda_naive_fp32, max_abs_err, device_time_ms, profiler_kernel_ms

T0 = time.time()


def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)


log('flashinfer', flashinfer.__version__, torch.cuda.get_device_name(0))
d = make_inputs()
q, k, v, g = d['q'], d['k'], d['v'], d['g']
beta, beta_logit = d['beta'], d['beta_logit']
A_log, dt_bias, state0, cu = d['A_log'], d['dt_bias'], d['state'], d['cu_seqlens']
seq_order = torch.argsort(cu[1:] - cu[:-1], descending=True).to(torch.int32).contiguous()
cu32 = cu.to(torch.int32)


def mk_state(layout, sdt):
    s = state0 if layout == 'kv' else state0.transpose(-1, -2)
    return s.to(sdt).contiguous()


def call(st, lb, bl, ofs, out=None, so=True, cus=None):
    return flashinfer.recurrent_kda(
        q=q, k=k, v=v, g=g, beta=(beta_logit if bl else beta),
        A_log=A_log, dt_bias=dt_bias, initial_state=st,
        output_final_state=ofs, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        lower_bound=lb, beta_is_logit=bl, cu_seqlens=(cu if cus is None else cus),
        output=out, seq_order=(seq_order if so else None))


log('=== eligibility scan (seq_order oracle) ===')
ok = []
seen_err = {}
for lb, bl, ofs, layout, sdt, cus in itertools.product(
        (None, -5.0), (False, True), (True, False), ('vk', 'kv'),
        (torch.bfloat16, torch.float32), (None, cu32)):
    st = mk_state(layout, sdt)
    tag = 'lb=%-5s blogit=%-5s ofs=%-5s layout=%s sdt=%-4s cu=%s' % (
        lb, bl, ofs, layout, 'bf16' if sdt == torch.bfloat16 else 'fp32',
        'i32' if cus is not None else 'i64')
    try:
        r = call(st.clone(), lb, bl, ofs, cus=cus)
        torch.cuda.synchronize()
        log('ACCEPT', tag)
        ok.append((lb, bl, ofs, layout, sdt, cus, tag))
    except Exception as e:
        msg = repr(e)[:110]
        if msg not in seen_err:
            seen_err[msg] = tag
            log('reject', tag, '::', msg)

log('accepted combos: %d' % len(ok))
if not ok:
    log('distinct errors:')
    for m, t in seen_err.items():
        log('   ', t, '::', m)
    raise SystemExit(1)

log('=== validate accepted combos against fp32 naive ===')
naive = {}
best = None
for (lb, bl, ofs, layout, sdt, cus, tag) in ok:
    if lb not in naive:
        t = time.time()
        naive[lb] = kda_naive_fp32(q, k, v, g, beta, A_log, dt_bias, state0, cu, lower_bound=lb)
        torch.cuda.synchronize()
        log('naive fp32 (lb=%s) %.1fs |o|max=%.6f' % (lb, time.time() - t, naive[lb][0].abs().max().item()))
    o_nv, s_nv = naive[lb]
    st = mk_state(layout, sdt).clone()
    r = call(st, lb, bl, ofs, cus=cus)
    torch.cuda.synchronize()
    sfin = (r[1] if (ofs and r[1] is not None) else st).float()
    if layout == 'vk':
        sfin = sfin.transpose(-1, -2)
    oe, se = max_abs_err(r[0], o_nv), max_abs_err(sfin, s_nv)
    log('  %s -> out_err=%.5e state_err=%.5e' % (tag, oe, se))
    if best is None or oe < best[0]:
        best = (oe, se, lb, bl, ofs, layout, sdt, cus, tag)

E_ref, SE, LB, BL, OFS, LAYOUT, SDT, CUS, TAG = best
log('=' * 70)
log('CAKE ref config: %s' % TAG)
log('E_ref = max|CAKE_ref_out - naive_fp32_out| = %.6e' % E_ref)
log('tol   = 7 * E_ref                          = %.6e' % (7 * E_ref))
log('state err                                  = %.6e' % SE)

st = mk_state(LAYOUT, SDT).clone()
st_keep = st.clone()
out_buf = torch.empty_like(v)
for so in (True, False):
    fn = lambda: call(st, LB, BL, OFS, out=out_buf, so=so, cus=CUS)
    m, ts = device_time_ms(fn, warmup=20, iters=50, rounds=5, reset=lambda: st.copy_(st_keep))
    log('REF device time seq_order=%-5s : %.4f ms  rounds=%s' % (so, m, ['%.4f' % x for x in ts]))
    pm, names = profiler_kernel_ms(fn)
    log('REF kernel-sum  seq_order=%-5s : %.4f ms' % (so, pm))
    for nm, (us, cnt) in sorted(names.items(), key=lambda kv: -kv[1][0])[:5]:
        log('      %8.1f us x%d  %s' % (us, cnt, nm[:80]))
log('done')
