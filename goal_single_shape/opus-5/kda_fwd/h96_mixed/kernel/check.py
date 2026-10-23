"""Accuracy check: fp32 naive  vs  CAKE ref (E_ref, tol=7*E_ref)  vs  the new Triton kernel."""

import sys
import time

import torch

from kda_common import make_inputs, kda_naive_fp32, max_abs_err, kda_log_decay
from refwrap import CakeRef
from kda_triton import kda_fwd

LOWER_BOUND = None if len(sys.argv) < 2 or sys.argv[1] == 'none' else float(sys.argv[1])
BC = int(sys.argv[2]) if len(sys.argv) > 2 else 16
CAP = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

print('torch', torch.__version__, '| dev', torch.cuda.get_device_name(0),
      '| lower_bound', LOWER_BOUND, '| BC', BC, '| CAP', CAP, flush=True)

d = make_inputs()
q, k, v, g, beta = d['q'], d['k'], d['v'], d['g'], d['beta']
A_log, dt_bias, state0, cu = d['A_log'], d['dt_bias'], d['state'], d['cu_seqlens']

la = kda_log_decay(g, A_log, dt_bias, LOWER_BOUND)
print('log_alpha min %.3f max %.6f mean %.4f | frac(<-%.0f)=%.3e' %
      (la.min().item(), la.max().item(), la.mean().item(), CAP, (la < -CAP).float().mean().item()), flush=True)

t0 = time.time()
o_nv, s_nv = kda_naive_fp32(q, k, v, g, beta, A_log, dt_bias, state0, cu, lower_bound=LOWER_BOUND)
torch.cuda.synchronize()
print('fp32 naive: %.1fs  |out|max=%.6f rms=%.6f' %
      (time.time() - t0, o_nv.abs().max().item(), o_nv.pow(2).mean().sqrt().item()), flush=True)

print('--- probing CAKE ref state layout/dtype ---', flush=True)
ref = CakeRef(q, k, v, g, beta, A_log, dt_bias, cu, lower_bound=LOWER_BOUND)
E_ref, E_ref_state = ref.probe(state0, o_nv, s_nv)
TOL = 7 * E_ref
print('=' * 74, flush=True)
print('REF config: layout=%s dtype=%s lower_bound=%s' % (ref.layout, ref.sdt, LOWER_BOUND), flush=True)
print('E_ref   = max|CAKE_ref_out - naive_fp32_out| = %.6e' % E_ref, flush=True)
print('        (state)                              = %.6e' % E_ref_state, flush=True)
print('tol = 7 * E_ref                              = %.6e' % TOL, flush=True)

print('--- new Triton kernel ---', flush=True)
st = state0.clone()
o_tr = kda_fwd(q, k, v, g, beta, A_log, dt_bias, st, cu, BC=BC, CAP=CAP)
torch.cuda.synchronize()
kk = kda_fwd.last
print('kernel: n_regs=%s n_spills=%s shared=%s' %
      (getattr(kk, 'n_regs', '?'), getattr(kk, 'n_spills', '?'), getattr(kk, 'metadata', None) and kk.metadata.shared), flush=True)
e_out = max_abs_err(o_tr, o_nv)
e_st = max_abs_err(st, s_nv)
print('MINE vs naive_fp32:  out max_abs_err = %.6e   (tol %.6e)' % (e_out, TOL), flush=True)
print('MINE vs naive_fp32:  state max_abs_err = %.6e  (ref state err %.6e)' % (e_st, E_ref_state), flush=True)
print('RESULT: %s   (%.3fx of tol)' % ('PASS' if e_out <= TOL else 'FAIL', e_out / TOL), flush=True)
