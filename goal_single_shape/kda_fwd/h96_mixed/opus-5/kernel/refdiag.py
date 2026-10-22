"""Diagnose where flashinfer's CAKE recurrent_kda first call blocks."""
import faulthandler, os, sys, time, threading
T0=time.time()
def log(*a): print('[%7.1fs]'%(time.time()-T0), *a, flush=True)
faulthandler.dump_traceback_later(240, exit=True)
for k in ('FLASHINFER_CUBIN_DIR','FLASHINFER_CUBIN_ARTIFACTORY','FLASHINFER_JIT_DIR'):
    log('env', k, os.environ.get(k))
import torch; log('torch ok')
import flashinfer; log('flashinfer', flashinfer.__version__)
try:
    import flashinfer.jit as fj
    log('jit dir', getattr(fj,'FLASHINFER_JIT_DIR',None), 'cubin dir', getattr(fj,'FLASHINFER_CUBIN_DIR',None))
except Exception as e: log('jit import', repr(e)[:150])
from kda_common import make_inputs
d = make_inputs()
q,k,v,g,beta = d['q'],d['k'],d['v'],d['g'],d['beta']
A_log,dt_bias,state0,cu = d['A_log'],d['dt_bias'],d['state'],d['cu_seqlens']
lens = cu[1:]-cu[:-1]
so = torch.argsort(lens, descending=True).to(torch.int32).contiguous()
log('inputs ready')
st = state0.transpose(-1,-2).to(torch.bfloat16).contiguous()
log('--> first recurrent_kda call')
r = flashinfer.recurrent_kda(q=q,k=k,v=v,g=g,beta=beta,A_log=A_log,dt_bias=dt_bias,
        initial_state=st, output_final_state=True, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, cu_seqlens=cu, seq_order=so)
torch.cuda.synchronize(); log('first call OK, out', tuple(r[0].shape), r[0].dtype)
faulthandler.cancel_dump_traceback_later()
log('done')
