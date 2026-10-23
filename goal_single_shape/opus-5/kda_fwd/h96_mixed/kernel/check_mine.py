"""fp32 naive vs the Triton kernel: correctness + device time + ablations."""
import sys, time
import torch
from kda_common import make_inputs, kda_naive_fp32, max_abs_err, kda_log_decay, device_time_ms
from kda_triton import kda_fwd

print('torch', torch.__version__, 'dev', torch.cuda.get_device_name(0),
      'SMs', torch.cuda.get_device_properties(0).multi_processor_count, flush=True)
import triton; print('triton', triton.__version__, flush=True)
d = make_inputs()
q,k,v,g,beta = d['q'],d['k'],d['v'],d['g'],d['beta']
A_log,dt_bias,state0,cu = d['A_log'],d['dt_bias'],d['state'],d['cu_seqlens']
o_nv, s_nv = kda_naive_fp32(q,k,v,g,beta,A_log,dt_bias,state0,cu)
torch.cuda.synchronize()
print('naive fp32 |o|max=%.6f rms=%.6f' % (o_nv.abs().max().item(), o_nv.pow(2).mean().sqrt().item()), flush=True)
out_buf = torch.empty_like(v)
best=None
for cfg in sys.argv[1].split(','):
    p = cfg.split(':')
    BC,CAP,BV,NW,NS,ABL = int(p[0]),float(p[1]),int(p[2]),int(p[3]),int(p[4]),int(p[5])
    st = state0.clone()
    try:
        tc=time.time()
        o = kda_fwd(q,k,v,g,beta,A_log,dt_bias,st,cu,out=out_buf,BC=BC,CAP=CAP,BV=BV,
                    num_warps=NW,num_stages=NS,abl=ABL)
        torch.cuda.synchronize(); kk=kda_fwd.last
        eo = max_abs_err(o,o_nv); es = max_abs_err(st,s_nv)
        st2 = state0.clone()
        m,ts = device_time_ms(lambda: kda_fwd(q,k,v,g,beta,A_log,dt_bias,st2,cu,out=out_buf,BC=BC,CAP=CAP,
                              BV=BV,num_warps=NW,num_stages=NS,abl=ABL),
                              warmup=10, iters=30, rounds=3, reset=lambda: st2.copy_(state0))
        print('cfg %-18s %5.1fs | out_err %.4e st_err %.4e | %7.4f ms | regs=%-3s spills=%-3s smem=%-6s grid=%d'
              % (cfg, time.time()-tc, eo, es, m, kk.n_regs, kk.n_spills, kk.metadata.shared,
                 kk.metadata.name and 0 or 0), flush=True)
        if ABL==0 and (best is None or m<best[0]): best=(m,cfg,eo)
    except Exception as e:
        import traceback; traceback.print_exc()
        print('cfg %s EXC %s' % (cfg, repr(e)[:250]), flush=True)
if best: print('BEST full cfg=%s %.4f ms err=%.3e' % (best[1],best[0],best[2]), flush=True)
