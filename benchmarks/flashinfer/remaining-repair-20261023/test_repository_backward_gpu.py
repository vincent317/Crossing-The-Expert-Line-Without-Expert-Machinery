"""Check remaining backward entries on the existing frozen attention inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None


def load_package(path):
    spec = importlib.util.spec_from_file_location('backward_candidate', path,
                                                submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA required')
class RepositoryBackwardTests(unittest.TestCase):
    def test_requested_candidate(self):
        from run_fa4 import reference, LENGTHS
        from run_fa4_tridao import gradients
        from tridao import compare
        from paired import Captured, prepare
        import kernel_span
        root = Path(os.environ['W'])
        relative = os.environ['AUDIT_CASE']
        case = os.environ['AUDIT_SHAPE']
        folder = root/'repository'/relative
        sys.path.insert(0,str(folder))
        os.chdir(folder)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        record=dict(path=relative, case=case, input_family='existing-frozen-attention-inputs',
                    gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), metrics=[], timing=[], complete=False,
                    source_sha256={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.cu','.cuh','.h')
                                   and 'trajectory' not in p.parts})
        dest=root/'results'/(relative.replace('/','__')+'-'+case+'.json')
        dest.parent.mkdir(exist_ok=True)
        for seed in (0,42):
            data=reference(case,seed)
            cache=root/'tridao_cache'/f'{case}-{seed}.pt'
            if cache.exists():pair=torch.load(cache,map_location='cuda',weights_only=True)
            else:
                pair=gradients(data,case);cache.parent.mkdir(exist_ok=True);torch.save(pair,cache)
            q,k,v,o,do,lse,cu=[data[x] for x in ('q','k','v','o','do','lse','cu')]
            scale=128**-.5
            solution=folder/'solution.json'
            if solution.exists():
                spec=json.loads(solution.read_text())['spec']
                file,entry=spec['entry_point'].split('::')
                mod=load_package(folder/file)
                args=[q,k,v,o,do,lse]+([cu,max(LENGTHS)] if case=='varlen' else [])+[scale]
                run=getattr(mod,entry)
                if spec.get('destination_passing_style'):
                    outputs=tuple(torch.empty_like(x) for x in (q,k,v))
                    def fn(*a):
                        run(*a,*outputs)
                        return outputs
                else:fn=run
            elif case=='mha':
                mod=load_package(folder/'mha_bwd_v3.py')
                fn=mod.flash_bwd_v3;args=[q,k,v,o,do,lse,scale]
            elif case=='gqa':
                mod=load_package(folder/'mybwd.py')
                fn=mod.my_backward;args=[q,k,v,o,lse,do]
            else:
                mod=load_package(folder/'mine.py')
                op=mod.Bwd(q,k,v,cu,scale)
                def fn(q,k,v,o,do,lse):
                    q2=(q.float()*(scale*mod.LOG2E)).to(q.dtype)
                    return op.run(q2,k,v,do,lse.T.contiguous()*mod.LOG2E,mod.compute_delta(o,do))
                args=[q,k,v,o,do,lse]
            call,reset,got=prepare(Captured(fn,args,{}))
            self.assertIsInstance(got,(tuple,list));self.assertEqual(len(got),3)
            metrics=[compare(a,b,c) for a,b,c in zip(got,pair[0],pair[1])]
            record['metrics'].append(dict(seed=seed,outputs=metrics))
            dest.write_text(json.dumps(record))
            for metric in metrics:self.assertTrue(metric['passed'],str(metric))
            if seed==0:
                for _ in range(3):
                    record['timing'].append(kernel_span.trace(call,reset,repeats=50))
                    dest.write_text(json.dumps(record))
            reset()
            for a,b,c in zip(call(),pair[0],pair[1]):self.assertTrue(compare(a,b,c)['passed'])
        record['complete']=True;dest.write_text(json.dumps(record))
        print('MARK_CANDIDATE_DONE',relative,case,flush=True)


if __name__=='__main__':
    unittest.main()
