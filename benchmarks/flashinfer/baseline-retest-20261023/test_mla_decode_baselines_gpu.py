"""Audit MLA decode library and archived AKO anchor on declared synthetic lengths."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import unittest
import torch
import flashinfer
import test_remaining_forward_base_gpu as fixture
from paired import Captured,prepare
from metrics import compare
import kernel_span


class MlaDecodeBaselineTests(unittest.TestCase):
    def test_shape(self):
        torch.backends.cuda.matmul.allow_tf32=False
        root=Path(os.environ['W']);case=os.environ['MLA_CASE']
        tree=ast.parse(textwrap.dedent(inspect.getsource(fixture.RepositoryForwardTests.mla)))
        method=tree.body[0]
        anchors=[i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source' for t in n.targets)]
        self.assertEqual(len(anchors),1)
        method.body=method.body[:anchors[0]]+ast.parse('return [q,qp,kv,kp,indptr,indices,scale],(expected,lse),lengths').body
        namespace=dict(fixture.__dict__);exec(compile(ast.fix_missing_locations(tree),'<canonical_mla_fixture>','exec'),namespace)
        source=root/'ako_reference/mla_decode/kernel.py'
        ako=fixture.load(source)
        record=dict(case=case,input_family='synthetic-balanced-lengths-original-distribution-unavailable',
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),implementations={},complete=False,
                    gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        dest=root/'results'/('mla-decode-'+case+'.json');dest.parent.mkdir(exist_ok=True)
        for label in ('ako_anchor','flashinfer_fa2'):
            row=dict(metrics=[],timing=[],complete=False);record['implementations'][label]=row
            for seed in (0,42):
                torch.manual_seed(seed)
                args,expected,lengths=namespace['mla'](self,Path(case)/'baseline')
                if label=='ako_anchor':fn=ako.run
                else:
                    wrapper=flashinfer.mla.BatchMLAPagedAttentionWrapper(torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8),backend='fa2')
                    q,qp,kv,kp,indptr,indices,scale=args;B=q.shape[0]
                    wrapper.plan(torch.arange(B+1,device='cuda',dtype=torch.int32),indptr,indices,
                                 torch.tensor(lengths,device='cuda',dtype=torch.int32),16,512,64,1,False,scale,q.dtype,kv.dtype)
                    def fn(q,qp,kv,kp,indptr,indices,scale):return wrapper.run(q,qp,kv,kp,return_lse=True)
                call,reset,_=prepare(Captured(fn,args,{}));actual=call()
                rows=[compare(a,b,b) for a,b in zip(actual,expected)];self.assertEqual(len(actual),len(expected))
                row['metrics'].append(dict(seed=seed,outputs=rows));dest.write_text(json.dumps(record))
                self.assertTrue(all(r['flashinfer_bench']['standard_pass'] for r in rows),str(rows))
                if seed==0:
                    for _ in range(3):row['timing'].append(kernel_span.trace(call,reset,repeats=50))
                reset();self.assertTrue(all(compare(a,b,b)['flashinfer_bench']['standard_pass'] for a,b in zip(call(),expected)))
            row['complete']=True
        record['complete']=True;dest.write_text(json.dumps(record));print('BASELINE_PASS',case,flush=True)


if __name__=='__main__':unittest.main()
