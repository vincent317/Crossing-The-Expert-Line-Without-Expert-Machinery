"""End-to-end evidence run for the KDA forward benchmark (case h96_mixed, B200).

Prints, in one place:
  1. CAKE recurrent_kda (M128) reference device time
  2. E_ref = max|ref - fp32 naive|  and  tol = 7 * E_ref
  3. max|new kernel - fp32 naive|  vs tol
  4. new kernel device time, same data / same measurement as the reference
"""

import sys
import time

import torch

from kda_common import (make_inputs, kda_naive_fp32, max_abs_err,
                        device_time_ms, profiler_kernel_ms, kda_log_decay)
from refwrap import CakeRef, LOWER_BOUND
from kda_triton import kda_fwd

CFGS = sys.argv[1] if len(sys.argv) > 1 else '32:5:128:8:2:0'

p = torch.cuda.get_device_properties(0)
print('torch %s | device %s | SMs %d' % (torch.__version__, p.name, p.multi_processor_count), flush=True)
import triton
print('triton', triton.__version__, flush=True)

d = make_inputs()
q, k, v, g = d['q'], d['k'], d['v'], d['g']
beta, beta_logit = d['beta'], d['beta_logit']
A_log, dt_bias, state0, cu = d['A_log'], d['dt_bias'], d['state'], d['cu_seqlens']
print('shapes: q%s v%s beta%s A_log%s dt_bias%s state%s cu=%s' %
      (tuple(q.shape), tuple(v.shape), tuple(beta.shape), tuple(A_log.shape),
       tuple(dt_bias.shape), tuple(state0.shape), cu.tolist()), flush=True)
la = kda_log_decay(g, A_log, dt_bias, LOWER_BOUND)
print('gate: lower_bound=%s  log_alpha in [%.4f, %.6f], mean %.4f' %
      (LOWER_BOUND, la.min().item(), la.max().item(), la.mean().item()), flush=True)

# ---------------- 1. fp32 naive ----------------
t0 = time.time()
o_nv, s_nv = kda_naive_fp32(q, k, v, g, beta, A_log, dt_bias, state0, cu, lower_bound=LOWER_BOUND)
torch.cuda.synchronize()
print('[naive] fp32 token-by-token reference: %.1fs  |out|max=%.6f rms=%.6f' %
      (time.time() - t0, o_nv.abs().max().item(), o_nv.pow(2).mean().sqrt().item()), flush=True)

# ---------------- 2. CAKE reference ----------------
ref = CakeRef(q, k, v, g, beta_logit, A_log, dt_bias, cu)
st_ref = ref.prep_state(state0)
st_ref0 = st_ref.clone()
o_ref, s_ref = ref(st_ref.clone())[:2]
torch.cuda.synchronize()
E_ref = max_abs_err(o_ref, o_nv)
E_ref_state = max_abs_err(ref.post_state(s_ref), s_nv)
TOL = 7 * E_ref
print('=' * 78, flush=True)
print('[ref  ] CAKE recurrent_kda (flashinfer frozen SM100 FlashKDA, M128)', flush=True)
print('[ref  ] E_ref = max|ref_out - naive_fp32_out| = %.6e' % E_ref, flush=True)
print('[ref  ] E_ref(state)                          = %.6e' % E_ref_state, flush=True)
print('[ref  ] tol   = 7 * E_ref                     = %.6e' % TOL, flush=True)

out_buf = torch.empty_like(v)
fn_ref = lambda: ref(st_ref, out=out_buf, use_seq_order=True)
ref_ms, ref_rounds = device_time_ms(fn_ref, warmup=20, iters=50, rounds=5,
                                    reset=lambda: st_ref.copy_(st_ref0))
ref_pm, ref_names = profiler_kernel_ms(fn_ref)
print('[ref  ] device time (CUDA events, median of 5x50) = %.4f ms  rounds=%s'
      % (ref_ms, ['%.4f' % x for x in ref_rounds]), flush=True)
print('[ref  ] kernel duration (profiler sum)            = %.4f ms' % ref_pm, flush=True)
for nm, (us, cnt) in sorted(ref_names.items(), key=lambda kv: -kv[1][0])[:4]:
    print('[ref  ]     %8.1f us  %s' % (us, nm[:80]), flush=True)
print('[ref  ] TARGET for new kernel = ref/1.2 = %.4f ms' % (ref_ms / 1.2), flush=True)

# ---------------- 3/4. new Triton kernel ----------------
print('=' * 78, flush=True)
best = None
for cfg in CFGS.split(','):
    a = cfg.split(':')
    BC, CAP, BV, NW, NS, ABL = int(a[0]), float(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5])
    SB = bool(int(a[6])) if len(a) > 6 else False
    st = state0.clone()
    try:
        kda_fwd(q, k, v, g, beta, A_log, dt_bias, st, cu, out=out_buf, BC=BC, CAP=CAP, BV=BV,
                lower_bound=LOWER_BOUND, num_warps=NW, num_stages=NS, abl=ABL, sbf16=SB)
        torch.cuda.synchronize()
        kk = kda_fwd.last
        eo, es = max_abs_err(out_buf, o_nv), max_abs_err(st, s_nv)
        st2 = state0.clone()
        fn = lambda: kda_fwd(q, k, v, g, beta, A_log, dt_bias, st2, cu, out=out_buf, BC=BC,
                             CAP=CAP, BV=BV, lower_bound=LOWER_BOUND, num_warps=NW,
                             num_stages=NS, abl=ABL, sbf16=SB)
        ms, rounds = device_time_ms(fn, warmup=20, iters=50, rounds=5,
                                    reset=lambda: st2.copy_(state0))
        pm, _ = profiler_kernel_ms(fn)
        print('[mine ] cfg BC=%-3d BV=%-3d nw=%-2d ns=%d abl=%d sbf16=%d | err %.4e st %.4e | '
              '%.4f ms (events) %.4f ms (profiler) | regs=%s spills=%s smem=%s'
              % (BC, BV, NW, NS, ABL, SB, eo, es, ms, pm, kk.n_regs, kk.n_spills, kk.metadata.shared),
              flush=True)
        if ABL == 0 and (best is None or ms < best[0]):
            best = (ms, pm, eo, es, cfg)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('[mine ] cfg %s EXC %s' % (cfg, repr(e)[:220]), flush=True)

print('=' * 78, flush=True)
if best is None:
    print('RESULT: no working configuration', flush=True)
    raise SystemExit(1)
ms, pm, eo, es, cfg = best
print('SUMMARY (case h96_mixed: T=8192, N=6, H=96, D=128, bf16, B200)', flush=True)
print('  1) ref  device time (CAKE recurrent_kda M128) = %.4f ms  [profiler %.4f ms]' % (ref_ms, ref_pm), flush=True)
print('  2) E_ref = %.6e   tol = 7*E_ref = %.6e' % (E_ref, TOL), flush=True)
print('  3) mine vs fp32 naive: max_abs_err = %.6e  -> %s (%.3fx tol)'
      % (eo, 'PASS' if eo <= TOL else 'FAIL', eo / TOL), flush=True)
print('  4) mine device time = %.4f ms  [profiler %.4f ms]  cfg=%s' % (ms, pm, cfg), flush=True)
print('     speedup vs ref = %.3fx   (need >= 1.200x)  -> %s'
      % (ref_ms / ms, 'PASS' if ms <= ref_ms / 1.2 else 'FAIL'), flush=True)
