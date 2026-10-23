"""Check FA4 forward baselines with the existing independent input fixture."""
import ast
import inspect
import json
import os
from pathlib import Path
import textwrap
import unittest
import torch
import test_remaining_forward_base_gpu as fixture
from paired import Captured,prepare
from metrics import compare
import kernel_span


class ForwardBaselineTests(unittest.TestCase):
    def test_shape(self):
        from flash_attn.cute.interface import flash_attn_func
        torch.backends.cuda.matmul.allow_tf32=False
        # Reuse only the canonical fixture prefix, before any candidate loading.
        tree=ast.parse(textwrap.dedent(inspect.getsource(fixture.RepositoryForwardTests.attention)))
        method=tree.body[0]
        anchors=[i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='python_entries' for t in n.targets)]
        self.assertEqual(len(anchors),1)
        method.body=method.body[:anchors[0]]+ast.parse('return (q,k,v), expected, causal').body
        namespace=dict(fixture.__dict__)
        exec(compile(ast.fix_missing_locations(tree),'<canonical_attention_fixture>','exec'),namespace)
        case=os.environ['FORWARD_CASE'];folder=Path(case)/'baseline'
        record=dict(case=case,metrics=[],timing=[],complete=False,gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        dest=Path(os.environ['W'])/'results'/('fa4-forward-'+case+'.json');dest.parent.mkdir(exist_ok=True)
        for seed in (0,42):
            torch.manual_seed(seed)
            args,expected,causal=namespace['attention'](self,folder)
            call,reset,actual=prepare(Captured(flash_attn_func,args,dict(causal=causal,softmax_scale=128**-.5)))
            actual=actual[0] if isinstance(actual,tuple) else actual
            metric=compare(actual,expected,expected)
            record['metrics'].append(dict(seed=seed,output=metric));dest.write_text(json.dumps(record))
            self.assertTrue(metric['flashinfer_bench']['standard_pass'],str(metric))
            if seed==0:
                for _ in range(3):record['timing'].append(kernel_span.trace(call,reset,repeats=50))
            reset();after=call();after=after[0] if isinstance(after,tuple) else after
            self.assertTrue(compare(after,expected,expected)['flashinfer_bench']['standard_pass'])
        record['complete']=True;dest.write_text(json.dumps(record))
        print('BASELINE_PASS',case,flush=True)


if __name__=='__main__':unittest.main()
