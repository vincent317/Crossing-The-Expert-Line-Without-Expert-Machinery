"""Device-time benchmark: CAKE recurrent_kda reference vs the new Triton kernel.

Identical data, identical measurement (CUDA events around the op, median of rounds,
plus the profiler's summed CUDA-kernel duration).
"""

import sys

import torch

from kda_common import make_inputs, device_time_ms, profiler_kernel_ms
from refwrap import CakeRef
from kda_triton import kda_fwd

LOWER_BOUND = None if len(sys.argv) < 2 or sys.argv[1] == 'none' else float(sys.argv[1])
LAYOUT = sys.argv[2] if len(sys.argv) > 2 else 'vk'
SDT = torch.bfloat16 if (len(sys.argv) < 4 or sys.argv[3] == 'bf16') else torch.float32
CFGS = sys.argv[4] if len(sys.argv) > 4 else '16:10:8:2'

print('torch', torch.__version__, '| dev', torch.cuda.get_device_name(0),
      '| SMs', torch.cuda.get_device_properties(0).multi_processor_count, flush=True)
print('ref cfg: lower_bound=%s layout=%s sdt=%s' % (LOWER_BOUND, LAYOUT, SDT), flush=True)

d = make_inputs()
q, k, v, g, beta = d['q'], d['k'], d['v'], d['g'], d['beta']
A_log, dt_bias, state0, cu = d['A_log'], d['dt_bias'], d['state'], d['cu_seqlens']

ref = CakeRef(q, k, v, g, beta, A_log, dt_bias, cu, lower_bound=LOWER_BOUND)
ref.layout, ref.sdt = LAYOUT, SDT
st_ref = ref.prep_state(state0).clone()
st_ref0 = st_ref.clone()
out_buf = torch.empty_like(v)

for use_so in (True, False):
    fn = lambda: ref(st_ref, out=out_buf, use_seq_order=use_so)
    m, ts = device_time_ms(fn, warmup=20, iters=50, rounds=5, reset=lambda: st_ref.copy_(st_ref0))
    pm, names = profiler_kernel_ms(fn)
    print('REF  seq_order=%-5s : events %.4f ms | profiler-kernel-sum %.4f ms | rounds %s'
          % (use_so, m, pm, ['%.4f' % x for x in ts]), flush=True)
    for nm, (us, cnt) in sorted(names.items(), key=lambda kv: -kv[1][0])[:6]:
        print('        %8.1f us x%d  %s' % (us, cnt, nm[:80]), flush=True)
REF_MS = m if False else None

print('-' * 74, flush=True)
st_my = state0.clone()
st_my0 = st_my.clone()
best = None
for cfg in CFGS.split(','):
    BC, CAP, NW, NS = cfg.split(':')
    BC, CAP, NW, NS = int(BC), float(CAP), int(NW), int(NS)
    try:
        fn = lambda: kda_fwd(q, k, v, g, beta, A_log, dt_bias, st_my, cu, out=out_buf,
                             BC=BC, CAP=CAP, num_warps=NW, num_stages=NS)
        fn()
        torch.cuda.synchronize()
        kk = kda_fwd.last
        m2, ts2 = device_time_ms(fn, warmup=20, iters=50, rounds=5, reset=lambda: st_my.copy_(st_my0))
        pm2, _ = profiler_kernel_ms(fn)
        print('MINE BC=%-3d CAP=%-4.1f nw=%d ns=%d : events %.4f ms | profiler %.4f ms | regs=%s spills=%s | rounds %s'
              % (BC, CAP, NW, NS, m2, pm2, getattr(kk, 'n_regs', '?'), getattr(kk, 'n_spills', '?'),
                 ['%.4f' % x for x in ts2]), flush=True)
        if best is None or m2 < best[0]:
            best = (m2, pm2, cfg)
    except Exception as e:
        print('MINE cfg=%s -> EXC %s' % (cfg, repr(e)[:200]), flush=True)

if best is not None:
    print('=' * 74, flush=True)
    print('BEST MINE cfg=%s : %.4f ms (events) / %.4f ms (profiler)' % (best[2], best[0], best[1]), flush=True)
