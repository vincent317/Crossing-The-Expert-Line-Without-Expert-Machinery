"""Check library baselines against cached independent recurrence and gradients."""
import json
import os
from pathlib import Path
import unittest
import torch
from paired import Captured, prepare
import kernel_span


class LibraryBaselineTests(unittest.TestCase):
    def test_case(self):
        group, case = os.environ['LIBRARY_CASE'].split('/')
        root = Path(os.environ['W'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        record = dict(case=case, group=group, metrics=[], timing=[], complete=False,
                      gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        dest = root/'results'/(group+'-'+case+'.json');dest.parent.mkdir(exist_ok=True)
        for seed in (0,42):
            if group == 'kda':
                import flashinfer
                from run_kda import reference
                from metrics import compare
                data = reference(case, seed)
                args = tuple(data[k] for k in ('q','k','v','g','beta'))
                kwargs = dict(A_log=data['A_log'],dt_bias=data['dt_bias'],scale=data['scale'],
                              initial_state=data['state'].clone(),output_final_state=True,
                              use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,
                              lower_bound=data['lower_bound'],cu_seqlens=data['cu_seqlens'],beta_is_logit=True)
                fn = flashinfer.recurrent_kda
                def check(actual):
                    self.assertEqual(len(actual),2)
                    rows=[compare(a,b,b) for a,b in zip(actual,data['fp'])]
                    for r in rows:r['passed']=r['flashinfer_bench']['standard_pass']
                    return rows
            else:
                from run_fa4 import reference, LENGTHS
                from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
                from tridao import compare
                data = reference(case, seed)
                pair = torch.load(root/'tridao_cache'/f'{case}-{seed}.pt',map_location='cuda',weights_only=True)
                q,k,v=[data[x].detach().requires_grad_() for x in ('q','k','v')]
                options=dict(softmax_scale=128**-.5,causal=True,return_lse=True)
                if case=='varlen':
                    out,_=flash_attn_varlen_func(q,k,v,cu_seqlens_q=data['cu'],cu_seqlens_k=data['cu'],
                            max_seqlen_q=max(LENGTHS),max_seqlen_k=max(LENGTHS),**options)
                else:out,_=flash_attn_func(q,k,v,**options)
                def fn():return torch.autograd.grad(out,(q,k,v),data['do'],retain_graph=True)
                args=();kwargs={}
                def check(actual):
                    self.assertEqual(len(actual),3)
                    return [compare(a,b,c) for a,b,c in zip(actual,pair[0],pair[1])]
            call,reset,actual=prepare(Captured(fn,args,kwargs))
            actual=call()
            rows=check(actual);record['metrics'].append(dict(seed=seed,outputs=rows));dest.write_text(json.dumps(record))
            self.assertTrue(all(r['passed'] for r in rows),str(rows))
            if seed==0:
                for _ in range(3):record['timing'].append(kernel_span.trace(call,reset,repeats=50))
            reset();self.assertTrue(all(r['passed'] for r in check(call())))
        import importlib.metadata
        record['packages']={n:importlib.metadata.version(n) for n in ('torch','flashinfer-python')}
        record['complete']=True;dest.write_text(json.dumps(record))
        print('BASELINE_PASS',group,case,flush=True)


if __name__=='__main__':unittest.main()
