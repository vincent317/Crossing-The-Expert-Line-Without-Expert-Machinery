"""Recheck new single-shape kernels and matched baselines on identical inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
import torch
import official
import kernel_span
from paired import Captured, prepare, schedule, aggregate
from metrics import compare


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def outputs(value):
    return (value,) if isinstance(value, torch.Tensor) else tuple(value)


class LatestSingleTests(unittest.TestCase):
    def test_requested(self):
        root = Path(os.environ['W'])
        relative, role = os.environ['LATEST_CASE'], os.environ['LATEST_ROLE']
        folder = root/'repository'/relative
        group, shape = folder.parent.name, folder.name
        solution = json.loads((folder/'solution.json').read_text())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        sys.path.insert(0, str(folder)); os.chdir(folder)
        record = dict(path=relative, role=role, revision=os.environ['LATEST_REVISION'], complete=False,
                      source_sha256={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in folder.rglob('*') if p.suffix in ('.py','.cu','.cuh','.h','.json')},
                      gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), metrics=[], timing=[])
        dest = root/'results'/(group+'-'+shape+'-'+role+'.json'); dest.parent.mkdir(exist_ok=True)
        def save(): dest.write_text(json.dumps(record))
        save()
        for seed in ((0,) if role=='paired' else (0,42)):
            torch.manual_seed(seed)
            if group=='fa4_bwd':
                from run_fa4 import reference, LENGTHS
                from tridao import compare as gradient_compare
                from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
                case = 'varlen' if shape.startswith('varlen') else shape.split('_')[0]
                data = reference(case,seed)
                expected = torch.load(root/'tridao_cache'/f'{case}-{seed}.pt',map_location='cuda',weights_only=True)
                q,k,v,o,do,lse,cu = [data[x] for x in ('q','k','v','o','do','lse','cu')]
                args = [q,k,v,o,do,lse]+([cu,max(LENGTHS)] if case=='varlen' else [])+[128**-.5]
                templates = (q,k,v)
                def metrics(actual):
                    actual=outputs(actual);self.assertEqual(len(actual),3)
                    return [gradient_compare(a,b,c) for a,b,c in zip(actual,*expected)]
            else:
                names={'gdn_fwd':'gdn_prefill','dsa_decode_fwd':'dsa_attention','moe_fp8_fwd':'moe'}
                prefixes={'t30_n1':'62447caf','t5709_n2':'07aa7922','t8192_n20':'a9540651',
                          't2_topk2048_short':'9d4a5f21','t2_topk2048':'05f6de65','t8_topk2048':'564007ac',
                          't1':'e05c6c03','t901':'1a4c6ba1','t14107':'5e8dc11c'}
                if group=='gdn_fwd':official.DATA=Path('/jiuzhang_train/user/chenyaojian/kda-astra-high-20261019/data/flashinfer-trace')
                args=official.inputs(solution['definition'],prefixes[shape])
                expected=official.reference(solution['definition'],args);templates=expected
                record['workload_prefix']=prefixes[shape]
                def metrics(actual):
                    actual=outputs(actual);self.assertEqual(len(actual),len(expected))
                    rows=[compare(a,b,b,operator='moe' if group=='moe_fp8_fwd' else None) for a,b in zip(actual,expected)]
                    for row in rows:row['passed']=row['flashinfer_bench']['standard_pass']
                    return rows
            calls,resets={},{}
            for endpoint in (('baseline','candidate') if role=='paired' else (role,)):
                call_args=[a.detach().clone() if isinstance(a,torch.Tensor) else a for a in args]
                if endpoint=='candidate':
                    file,entry=solution['spec']['entry_point'].split('::')
                    module=load(folder/file,'latest_candidate');run=getattr(module,entry)
                    if solution['spec'].get('destination_passing_style'):
                        allocated=tuple(torch.full_like(x,float('nan')) for x in templates)
                        def fn(*a,_run=run,_out=allocated):
                            _run(*a,*_out)
                            return _out
                    else:fn=run
                elif group!='fa4_bwd':
                    path=Path(os.environ['BASELINE_REFERENCE'])/names[group]/'kernel.py'
                    record['baseline_source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                    fn=load(path,'latest_baseline').run
                else:
                    bq,bk,bv=[data[x].detach().clone().requires_grad_() for x in ('q','k','v')]
                    options=dict(softmax_scale=128**-.5,causal=True,return_lse=True)
                    if case=='varlen':bo,_=flash_attn_varlen_func(bq,bk,bv,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=max(LENGTHS),max_seqlen_k=max(LENGTHS),**options)
                    else:bo,_=flash_attn_func(bq,bk,bv,**options)
                    def fn(_out=bo,_q=bq,_k=bk,_v=bv,_do=do):return torch.autograd.grad(_out,(_q,_k,_v),_do,retain_graph=True)
                    call_args=[]
                    record['baseline_identity']='FlashAttention-4 cute autograd backward'
                call,reset,_=prepare(Captured(fn,call_args,{}))
                for repeat in range(10):
                    reset();rows=metrics(call())
                    record['metrics'].append(dict(seed=seed,endpoint=endpoint,repeat=repeat,outputs=rows));save()
                    self.assertTrue(all(r['passed'] for r in rows),str(rows))
                calls[endpoint],resets[endpoint]=call,reset
            if seed==0:
                for i in range(3):
                    if role=='paired':result=kernel_span._trace(calls,resets,schedule(50,i),schedule(10,i),set(calls))
                    else:result=kernel_span.trace(calls[role],resets[role],repeats=50)
                    record['timing'].append(result);save()
            for endpoint,call in calls.items():
                resets[endpoint]();rows=metrics(call())
                record['metrics'].append(dict(seed=seed,endpoint=endpoint,phase='after_timing',outputs=rows));save()
                self.assertTrue(all(r['passed'] for r in rows),str(rows))
        if role=='paired':record['paired']=aggregate(record['timing'])
        record['complete']=True;save()
        print('LATEST_PASS',relative,role,flush=True)


if __name__=='__main__':unittest.main()
