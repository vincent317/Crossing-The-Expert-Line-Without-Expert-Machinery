"""Recheck historical Top-k experts and current candidates on identical tensors."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
import torch
import official
import kernel_span
from paired import Captured, prepare, schedule, aggregate


class TopkBaselineTests(unittest.TestCase):
    def test_shape(self):
        root=Path(os.environ['W']);historical=root/'historical-topk'
        os.chdir(historical);sys.path.insert(0,str(historical))
        import bench_topk_ref as fixture
        from run_dsa import load
        name,uuid,batch,_=next(x for x in fixture.WORKLOADS if x[0]==os.environ['TOPK_SHAPE'])
        torch.backends.cuda.matmul.allow_tf32=False
        implementations={k:importlib.import_module(k+'.kernel').run for k in ('ref_ako_fs6','ref_ako')}
        dog=importlib.import_module('ref_dogacel.indexer_fused')
        def dog_run(*args):
            output=torch.empty(batch,2048,device='cuda',dtype=torch.int32)
            dog.kernel(*args,output)
            return output
        implementations['ref_dogacel']=dog_run
        relative={'A':'topk_T1_A_single','B':'topk_T2_B_mid','C':'topk_T3_C_large'}[name]
        folder=root/'repository/goal_single_shape/dsa_topk_indexer'/relative/'fable-5.1'
        file,entry={'A':('topk_triton.py','run_triton'),'B':('topk_cuda.py','run'),'C':('topk_indexer.py','run')}[name]
        implementations['candidate']=getattr(load(folder/'kernel'/file,'topk_candidate'),entry)
        record=dict(shape=name,uuid=uuid,historical_job='tasks/1074336',complete=False,distributions=[],
                    source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for directory in (historical,folder) for p in directory.rglob('*') if p.is_file() and p.suffix in ('.py','.cu','.cuh')},
                    gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        dest=root/'results'/('topk-'+name+'.json');dest.parent.mkdir(exist_ok=True)
        for distribution in ('historical_uniform_nonnegative','signed'):
            args=list(fixture.make_inputs(uuid,batch,torch.device('cuda')))
            if distribution=='signed':
                torch.manual_seed(42);args[2]=torch.randn_like(args[2])
            expected=official.reference('dsa_topk_indexer_fp8_h64_d128_topk2048_ps64',args)[0].sort(-1).values
            scores=fixture.final_scores(*args)
            result=dict(distribution=distribution,implementations={},paired=[])
            record['distributions'].append(result)
            calls={};resets={}
            for label,fn in implementations.items():
                call,reset,actual=prepare(Captured(fn,args,{}))
                actual=actual[0] if isinstance(actual,tuple) else actual
                exact=torch.equal(actual.sort(-1).values,expected)
                error=fixture.sorted_score_err(actual,scores,args[3])
                row=dict(exact_membership=exact,sorted_score_max_abs_error=error,timing=[])
                result['implementations'][label]=row;dest.write_text(json.dumps(record))
                # Preserve both checks; failed exact membership is not a passing result.
                row['repeat_exact_passes']=0
                for repeat in range(100):
                    reset();repeated=call();repeated=repeated[0] if isinstance(repeated,tuple) else repeated
                    row['repeat_exact_passes']+=int(torch.equal(repeated.sort(-1).values,expected))
                dest.write_text(json.dumps(record))
                if not exact or row['repeat_exact_passes']!=100:continue
                for _ in range(3):row['timing'].append(kernel_span.trace(call,reset,repeats=50))
                reset();after=call();after=after[0] if isinstance(after,tuple) else after
                row['post_timing_exact_membership']=torch.equal(after.sort(-1).values,expected)
                dest.write_text(json.dumps(record))
                if not row['post_timing_exact_membership']:continue
                calls[label]=call;resets[label]=reset
            if 'candidate' in calls:
                for label in calls:
                    if label=='candidate':continue
                    pair_calls={'baseline':calls[label],'candidate':calls['candidate']}
                    pair_resets={'baseline':resets[label],'candidate':resets['candidate']}
                    rounds=[kernel_span._trace(pair_calls,pair_resets,schedule(50,i),schedule(10,i),set(pair_calls)) for i in range(3)]
                    result['paired'].append(dict(baseline=label,rounds=rounds,**aggregate(rounds)))
            dest.write_text(json.dumps(record))
        record['complete']=True;dest.write_text(json.dumps(record))
        print('BASELINE_PASS_TOPK_AUDIT_COMPLETE',name,flush=True)


if __name__=='__main__':unittest.main()
