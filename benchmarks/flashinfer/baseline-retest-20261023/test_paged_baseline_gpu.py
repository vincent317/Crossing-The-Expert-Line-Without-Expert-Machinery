"""Validate historical paged-attention baselines on canonical audit fixtures."""
import ast
import json
import math
import os
from pathlib import Path
import unittest
import torch
import kernel_span
from metrics import compare
from paired import Captured,prepare


class PagedBaselineTests(unittest.TestCase):
    def test_case(self):
        spec=json.loads(os.environ['PAGED_SPEC']);group=spec['group'];case=spec['case']
        root=Path(os.environ['W']);filename=root/('run_gqa.py' if group=='gqa' else 'run_mla.py')
        tree=ast.parse(filename.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='one')
        if group=='gqa':
            removed=[n for n in fn.body if isinstance(n,ast.If) and ast.unparse(n.test)=='B == 1']
            self.assertEqual(len(removed),1);fn.body.remove(removed[0])
        loop=next(n for n in fn.body if isinstance(n,ast.For))
        anchors=[i for i,n in enumerate(loop.body) if
                 (group=='gqa' and isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='got' for t in n.targets)) or
                 (group=='mla' and isinstance(n,ast.Expr) and 'STAGE_CANDIDATE' in ast.unparse(n))]
        self.assertEqual(len(anchors),1)
        loop.body=loop.body[:anchors[0]]+ast.parse('return locals()').body
        namespace=dict(__name__='paged_fixture',__file__=str(filename))
        record=dict(spec=spec,metrics=[],timing=[],complete=False,gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        dest=root/'results'/(group+'-'+case+'-'+spec['backend']+'.json');dest.parent.mkdir(exist_ok=True)
        for seed in (0,42):
            loop.iter=ast.parse(repr([seed]),mode='eval').body
            exec(compile(ast.fix_missing_locations(tree),str(filename),'exec'),namespace)
            c=namespace['one'](('gqa_paged_decode/' if group=='gqa' else 'mla_paged_prefill/')+case+'/baseline')
            if spec['backend']=='flashinfer':
                if group=='gqa':
                    fn=c['wrapper'].run_return_lse;args=(c['q'],(c['k'],c['v']));kwargs={}
                else:fn=c['wrapper'].run;args=(c['q'],c['qp'],c['k'],c['kp']);kwargs=dict(return_lse=True)
            else:
                from flash_attn.cute.interface import flash_attn_varlen_func
                fn=flash_attn_varlen_func
                lengths=c['lengths'] if group=='gqa' else c['kl'];ptr=c['ptr'] if group=='gqa' else c['ki']
                table=torch.zeros(len(lengths),max(lengths),device='cuda',dtype=torch.int32)
                for i,n in enumerate(lengths):table[i,:n]=c['ids'][int(ptr[i]):int(ptr[i+1])]
                kwargs=dict(page_table=table,seqused_k=torch.tensor(lengths,device='cuda',dtype=torch.int32),
                            max_seqlen_k=max(lengths),softmax_scale=c['scale'],causal=group=='mla',
                            num_splits=1,pack_gqa=None,return_lse=True)
                if group=='gqa':
                    args=(c['q'],c['k'],c['v']);kwargs.update(cu_seqlens_q=torch.arange(len(lengths)+1,device='cuda',dtype=torch.int32),max_seqlen_q=1)
                else:
                    args=(c['qp'],c['kp'].unsqueeze(2),c['k'].unsqueeze(2));kwargs.update(qv=c['q'],cu_seqlens_q=c['qi'],max_seqlen_q=max(c['ql']))
            call,reset,actual=prepare(Captured(fn,args,kwargs))
            expected=(torch.stack(c['fp']),torch.stack(c['fl'])) if group=='gqa' else (c['fpout'],c['fplse'])
            def check(value):
                out,lse=value
                if spec['backend']=='flashinfer':lse=lse*math.log(2)
                elif lse.shape!=expected[1].shape:lse=lse.T
                return [compare(a,b,b) for a,b in zip((out,lse),expected)]
            rows=check(actual);record['metrics'].append(dict(seed=seed,outputs=rows));dest.write_text(json.dumps(record))
            self.assertTrue(all(r['flashinfer_bench']['standard_pass'] for r in rows),str(rows))
            if seed==0:
                for _ in range(3):record['timing'].append(kernel_span.trace(call,reset,repeats=50))
            reset();self.assertTrue(all(r['flashinfer_bench']['standard_pass'] for r in check(call())))
        record['complete']=True;dest.write_text(json.dumps(record));print('BASELINE_PASS',spec,flush=True)


if __name__=='__main__':unittest.main()
