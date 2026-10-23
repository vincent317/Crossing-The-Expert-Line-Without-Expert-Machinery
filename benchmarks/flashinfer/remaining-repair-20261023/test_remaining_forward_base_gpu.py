"""Audit previously uncovered forward kernels without editing candidate sources."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None


def load(path):
    spec = importlib.util.spec_from_file_location('remainder_candidate', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compile_shared(source, output):
    subprocess.run(['nvcc', '-O3', '-std=c++17', '-gencode', 'arch=compute_100a,code=sm_100a',
                    '--expt-relaxed-constexpr', '-shared', '-Xcompiler', '-fPIC',
                    '-I'+os.environ['CUTLASS_INCLUDE'], str(source), '-o', str(output), '-lcuda'], check=True)
    return ctypes.CDLL(str(output))


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA required')
class RepositoryForwardTests(unittest.TestCase):
    def test_requested_candidate(self):
        import kernel_span
        from metrics import compare
        from paired import Captured, prepare
        root = Path(os.environ['W'])
        relative = os.environ['AUDIT_CASE']
        folder = root/'repository'/relative
        os.chdir(folder)
        sys.path.insert(0, str(folder))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        sources = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py', '.cu', '.cuh', '.h')
                   and 'trajectory' not in p.parts}
        record = dict(path=relative, source_sha256=sources, metrics=[], timing=[], complete=False,
                      gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        dest = root/'results'/(relative.replace('/', '__')+'.json')
        dest.parent.mkdir(exist_ok=True)
        seeds = (-1, 0, 42) if '/gdn_fwd/' in relative else (0, 42)
        for seed in seeds:
            torch.manual_seed(seed)
            if '/fa4_fwd/' in relative:
                fn, args, expected = self.attention(folder)
            elif '/gdn_decode/' in relative:
                fn, args, expected = self.gdn_decode(folder)
            elif '/gdn_fwd/' in relative:
                fn, args, expected = self.gdn_prefill(folder, seed)
            elif '/dsa_decode_fwd/' in relative:
                fn, args, expected = self.dsa(folder)
            elif '/mla_paged_prefill/' in relative:
                fn, args, expected = self.mla_prefill(folder)
                record['input_family'] = 'synthetic-lengths-original-distribution-unavailable'
            elif '/mla_paged_decode/' in relative:
                fn, args, expected = self.mla(folder)
                record['input_family'] = 'synthetic-balanced-lengths-original-distribution-unavailable'
            elif '/kda_fwd/' in relative:
                fn, args, expected = self.kda(folder,seed)
            elif '/dsa_topk_indexer/' in relative:
                fn, args, expected = self.topk(folder, seed)
            else:
                raise ValueError(relative)
            call, reset, _ = prepare(Captured(fn, args, {}))
            got = call()
            if isinstance(got, torch.Tensor):got = (got,)
            self.assertEqual(len(got), len(expected))
            def checked(a,b):
                if '/dsa_topk_indexer/' in relative:
                    passed=a.shape==b.shape and torch.equal(a.sort(-1).values,b.sort(-1).values)
                    if not passed:
                        print('TOPK_MISMATCH',[(int(torch.isin(x,y).sum()),x[:10].tolist(),y[:10].tolist()) for x,y in zip(a,b)],flush=True)
                        q,cache,weights,lens,table=args
                        flat=cache.view(torch.uint8).view(cache.shape[0],-1)
                        keys=flat[:,:8192].contiguous().view(torch.float8_e4m3fn).view(-1,64,128).float()*flat[:,8192:].contiguous().view(torch.float32).view(-1,64,1)
                        for bi in range(q.shape[0]):
                            count=int(lens[bi]);ids=table[bi,:(count+63)//64].long()
                            score=((q[bi].float()@keys[ids].reshape(-1,128)[:count].T).relu()*weights[bi,:,None]).sum(0)
                            if count>2048:
                                global_ids=(table[bi,torch.arange(count,device=q.device)//64].long()*64+torch.arange(count,device=q.device)%64)
                                missed=b[bi][~torch.isin(b[bi],a[bi])];extra=a[bi][~torch.isin(a[bi],b[bi])]
                                print('TOPK_SCALE',bi,[(int(t),float(flat[int(t)//64,8192:].contiguous().view(torch.float32)[int(t)%64]),bool(torch.isfinite(keys[int(t)//64,int(t)%64]).all())) for t in missed],flush=True)
                                print('TOPK_DIFFERENCE',bi,[(int(t),float(score[(global_ids==t).nonzero()[0,0]])) for t in missed],[(int(t),float(score[(global_ids==t).nonzero()[0,0]])) for t in extra],int(score.isnan().sum()),flush=True)
                                selected=torch.topk(score,2048).values
                                print('TOPK_BOUNDARY',bi,count,float(selected[-1]),int((score==selected[-1]).sum()),float(score.min()),float(score.max()),flush=True)
                    return {'flashinfer_bench':{'standard_pass':passed},'criterion':'exact_topk_membership_including_padding'}
                metric=compare(a,b,b)
                if not metric['flashinfer_bench']['standard_pass']:
                    bad=(a.float()-b.float()).abs()>(0.01+0.01*b.float().abs())
                    ix=bad.nonzero()[:8]
                    print('BAD_VALUES',ix.tolist(),[(float(a[tuple(i)]),float(b[tuple(i)])) for i in ix],flush=True)
                return metric
            metrics = [checked(a,b) for a,b in zip(got, expected)]
            record['metrics'].append(dict(seed=seed, outputs=metrics))
            dest.write_text(json.dumps(record))
            for metric in metrics:
                self.assertTrue(metric['flashinfer_bench']['standard_pass'], str(metric))
            if seed == seeds[0]:
                for _ in range(3):
                    record['timing'].append(kernel_span.trace(call, reset, repeats=50))
                    dest.write_text(json.dumps(record))
            reset()
            after = call()
            if isinstance(after, torch.Tensor):after = (after,)
            for a,b in zip(after, expected):
                self.assertTrue(checked(a,b)['flashinfer_bench']['standard_pass'])
        record['complete'] = True
        dest.write_text(json.dumps(record))
        print('MARK_CANDIDATE_DONE', relative, flush=True)

    def attention(self, folder):
        case = folder.parent.name
        match = re.fullmatch(r'(mha|gqa)_b(\d+)_n(\d+)_h(\d+)(?:_kv(\d+))?_(noncausal|causal)', case)
        self.assertIsNotNone(match)
        _, B, S, H, K, causal = match.groups()
        B,S,H = map(int,(B,S,H));K = int(K) if K else H
        causal = causal == 'causal'
        q = torch.randn(B,S,H,128,device='cuda',dtype=torch.bfloat16)
        k = torch.randn(B,S,K,128,device='cuda',dtype=torch.bfloat16)
        v = torch.randn_like(k)
        scale = 128**-.5
        expected = torch.empty_like(q)
        # Bound reference memory while preserving the full FP32 attention formula.
        keys = torch.arange(S,device='cuda')
        for b in range(B):
            for h in range(H):
                kh = h//(H//K)
                kk,vv = k[b,:,kh].float(),v[b,:,kh].float()
                for start in range(0,S,256):
                    stop = min(start+256,S)
                    scores = q[b,start:stop,h].float() @ kk.T * scale
                    if causal:
                        scores.masked_fill_(keys[None,:] > torch.arange(start,stop,device='cuda')[:,None], -float('inf'))
                    expected[b,start:stop,h] = scores.softmax(-1) @ vv
        python_entries = {
            'mha_b8_n4096_h16_noncausal': ('fa_fwd_v6.py','flash_fwd'),
            'mha_b1_n32768_h16_noncausal': ('fa_fwd2.py','attention_fwd'),
            'gqa_b8_n4096_h32_kv4_causal': ('fa_v9.py','attn'),
            'gqa_b8_n4096_h32_kv8_noncausal': ('fa_fwd.py','flash_fwd'),
        }
        if case in python_entries:
            file,entry = python_entries[case]
            fn = getattr(load(folder/file), entry)
        elif case == 'gqa_b1_n32768_h32_kv8_noncausal':
            from torch.utils.cpp_extension import load as extension
            mod = extension(name='audit_fa3_v12a',sources=[str(folder/'fa3_v12a.cu')],
                            extra_cuda_cflags=['-O3','-std=c++17','-gencode', 'arch=compute_100a,code=sm_100a','--expt-relaxed-constexpr'],
                            extra_ldflags=['-lcuda'],verbose=True)
            prof = torch.empty(0, device='cuda', dtype=torch.int64)
            fn = lambda q,k,v:mod.fa_fwd(q,k,v,scale,0,prof,0)
        elif case == 'gqa_b1_n32768_h32_kv4_noncausal':
            lib = compile_shared(folder/'attn_v12.cu',folder/'libaudit.so')
            lib.attn_fwd.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*4+[ctypes.c_float,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
            lib.attn_fwd.restype = ctypes.c_int
            def fn(q,k,v):
                out = torch.empty_like(q)
                rc = lib.attn_fwd(q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),B,S,H,K,scale,0,torch.cuda.current_stream().cuda_stream,None)
                if rc:raise RuntimeError(f'attn_fwd returned {rc}')
                return out
        else:
            entries = {
                'gqa_b1_n32768_h32_kv4_causal':('kernel/fa_fwd.cu','kernel/libfa.so','fa.py','fa_fwd'),
                'gqa_b1_n32768_h32_kv8_causal':('fa_fwd.cu','libfa.so','mine.py','fa_fwd'),
                'gqa_b8_n4096_h32_kv4_noncausal':('fa_fwd_b200.cu','libfa.so','myfa.py','my_attn'),
                'gqa_b8_n4096_h32_kv8_causal':('fa2cta_v7e.cu','libfa.so','fa_lib.py','fa_fwd'),
                'mha_b1_n32768_h16_causal':('fa_fwd2.cu','libfa_fwd.so','fa_mine.py','fa_fwd'),
                'mha_b8_n4096_h16_causal':('fa_fwd.cu','libfa_fwd.so','fa_lib.py','fa_fwd'),
            }
            source,so,file,entry = entries[case]
            compile_shared(folder/source,folder/so)
            mod = load(folder/file)
            if case == 'mha_b8_n4096_h16_causal':mod.build(str(folder/source),str(folder/so))
            fn = getattr(mod,entry)
            if entry == 'my_attn':fn = lambda q,k,v:mod.my_attn(q,k,v,scale)
        return fn,[q,k,v],(expected,)

    def gdn_decode(self, folder):
        B = int(folder.parent.name[1:])
        q = torch.nn.functional.normalize(torch.randn(B,1,4,128,device='cuda'),dim=-1).bfloat16()
        k = torch.nn.functional.normalize(torch.randn_like(q,dtype=torch.float32),dim=-1).bfloat16()
        v = torch.randn(B,1,8,128,device='cuda',dtype=torch.bfloat16)
        state = torch.randn(B,8,128,128,device='cuda',dtype=torch.float32)
        A = torch.randn(8,device='cuda');dt = torch.randn(8,device='cuda')
        a = torch.randn(B,1,8,device='cuda',dtype=torch.bfloat16);b = torch.randn_like(a)
        scale = 128**-.5
        kk = k[:,0].float().repeat_interleave(2,dim=1)
        qq = q[:,0].float().repeat_interleave(2,dim=1)*scale
        decay = (-A.exp()*torch.nn.functional.softplus(a[:,0].float()+dt)).exp()
        st = state*decay[:,:,None,None]
        delta = (v[:,0].float()-(st*kk[:,:,None,:]).sum(-1))*b[:,0].float().sigmoid()[:,:,None]
        final = st+delta[:,:,:,None]*kk[:,:,None,:]
        output = (final*qq[:,:,None,:]).sum(-1).bfloat16()[:,None]
        if B == 1:
            so = folder/'libgdn.so';compile_shared(folder/'gdn_decode_final.cu',so)
            os.environ['GDN_LIB'] = str(so);mod = load(folder/'mykernel.py')
            def fn(q,k,v,state,A,a,dt,b,scale):
                return mod.run(0,q,k,v,state,A,a,dt,b,scale,torch.empty_like(v),torch.empty_like(state))
        else:
            mod = load(folder/'gdn_decode.py')
            fn = mod.gdn_decode_qk4_v8_d128_k_last if B==16 else mod.run
        return fn,[q,k,v,state,A,a,dt,b,scale],(output,final)

    def gdn_prefill(self, folder, seed):
        from run_gdn import workload, DEFINITION
        import official
        prefix = {'t8192_n20':'a9540651','t5709_n2':'07aa7922','t30_n1':'62447caf'}[folder.parent.name]
        args = workload(prefix)
        print("GDN_INPUT", prefix, [(tuple(x.shape),str(x.dtype),float(x.float().abs().max())) for x in args if isinstance(x,torch.Tensor)], "QK_NORM",float(args[0].float().norm(dim=-1).mean()),float(args[1].float().norm(dim=-1).mean()),flush=True)
        if seed >= 0:
            q,k,v,st,A,a,dt,b,cu,scale = args
            q = torch.nn.functional.normalize(torch.randn_like(q,dtype=torch.float32),dim=-1).bfloat16()
            k = torch.nn.functional.normalize(torch.randn_like(k,dtype=torch.float32),dim=-1).bfloat16()
            args = [q,k,torch.randn_like(v),torch.randn_like(st),torch.randn_like(A),torch.randn_like(a),torch.randn_like(dt),torch.randn_like(b),cu,scale]
        expected = official.reference(DEFINITION,args)
        model = folder.name
        if model == 'fable-5.1':
            fn = load(folder/'kernel.py').gdn_prefill_fwd
        elif model == 'fable-5.1-ncu':
            so=folder/'libaudit.so';compile_shared(folder/'gdn_fwd_v3.cu',so)
            os.environ['GDN_LIB']=str(so);fn=load(folder/'mykernel.py').gdn_fwd
        elif model == 'fable-5.1-no-accuracy-constraint':
            so=folder/'libaudit.so';compile_shared(folder/'gdn_fwd_v8.cu',so)
            os.environ['GDN_SO']=str(so);fn=load(folder/'gdn_mine_v8.py').run
        elif folder.parent.name == 't30_n1':
            mod=load(folder/'kernel/kernel.py')
            fn=lambda q,k,v,st,A,a,dt,b,cu,scale:mod.gdn_fwd(q,k,v,a,b,A,dt,st,cu,scale)
        elif folder.parent.name == 't8192_n20':
            mod=load(folder/'kernel/gdn_triton.py')
            # Configuration published in this candidate's SUMMARY.md.
            op=mod.GDN(args[0].shape[0],args[8].numel()-1,4,8,128,C=64,split=4,nsq=2,st_scan=3)
            fn=lambda q,k,v,st,A,a,dt,b,cu,scale:op(q,k,v,a,b,A,dt,st,cu)
        else:
            mod=load(folder/'kernel/gdn_kernel.py')
            q,k,v,st,A,a,dt,b,cu,scale=args
            import ast
            tree=ast.parse((folder/'kernel/final.py').read_text())
            best=next(n.value for n in tree.body if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='BEST' for t in n.targets))
            config={kw.arg:ast.literal_eval(kw.value) for kw in best.keywords}
            plan=mod.build_plan(cu,m=config.pop('m'),seg=config.pop('seg'))
            op=mod.Runner(q,k,v,a,b,A,dt,st,plan,scale,**config)
            fn=lambda:op()
            args=[]
        return fn,args,expected

    def dsa(self, folder):
        import official
        name='dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64'
        prefix={'t8_topk2048':'564007ac','t2_topk2048':'05f6de65','t2_topk2048_short':'9d4a5f21'}[folder.parent.name]
        args=official.inputs(name,prefix)
        expected=official.reference(name,args)
        if folder.name in ('opus-4.8', 'fable-5.1-kernelwiki'):fn=load(folder/'kernel.py').run
        elif folder.parent.name=='t8_topk2048':
            sys.path.insert(0,str(folder/'kernel'))
            fn=load(folder/'kernel/dsa_kernel.py').run
        elif folder.parent.name=='t2_topk2048_short':
            src=folder/'kernel/dsa_kernel.cu';so=folder/'kernel/libdsa.so';compile_shared(src,so)
            fn=load(folder/'kernel/dsa_ops.py').DsaLib(str(so)).fwd
        else:
            mod=load(folder/'kernel/dsa_cuda.py')
            names=('q_nope','q_pe','ckv_cache','kpe_cache','sparse_indices','sm_scale')
            op=mod.DsaCuda(dict(zip(names,args)))
            fn=lambda:op();args=[]
        return fn,args,expected

    def kda(self, folder, seed):
        sys.path.insert(0,str(folder/'kernel'))
        common=load(folder/'kernel/kda_common.py')
        data=common.make_inputs(seed=seed)
        args=[data[n] for n in ('q','k','v','g','beta','A_log','dt_bias','state','cu_seqlens')]
        mod=load(folder/'kernel/kda_triton.py')
        import inspect
        lower_bound=inspect.signature(mod.kda_fwd).parameters['lower_bound'].default
        expected=common.kda_naive_fp32(*args, lower_bound=lower_bound)
        def fn(*a):
            out=mod.kda_fwd(*a)
            return out,a[7]
        return fn,args,expected

    def mla(self, folder):
        B,total=map(int,re.fullmatch(r'b(\d+)_kv(\d+)',folder.parent.name).groups())
        lengths=[total//B+(i<total%B) for i in range(B)]
        indptr=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int32)
        indices=torch.randperm(total,device='cuda',dtype=torch.int32)
        q=torch.randn(B,16,512,device='cuda',dtype=torch.bfloat16)
        qp=torch.randn(B,16,64,device='cuda',dtype=torch.bfloat16)
        kv=torch.randn(total,1,512,device='cuda',dtype=torch.bfloat16)
        kp=torch.randn(total,1,64,device='cuda',dtype=torch.bfloat16)
        scale=576**-.5
        expected=torch.empty_like(q);lse=torch.empty(B,16,device='cuda')
        for n,(start,end) in enumerate(zip(indptr.tolist(),indptr.tolist()[1:])):
            ids=indices[start:end].long()
            scores=(q[n].float()@kv[ids,0].float().T+qp[n].float()@kp[ids,0].float().T)*scale
            expected[n]=scores.softmax(-1)@kv[ids,0].float()
            lse[n]=scores.logsumexp(-1)/__import__('math').log(2)
        source=folder/('mla_decode_v4.cu' if B==1 else 'mla_decode.cu')
        lib=compile_shared(source,folder/'libaudit.so')
        if B==1:
            launch=lib.mla_decode_launch
            launch.argtypes=[ctypes.c_void_p]*6+[ctypes.c_float,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
            def fn(q,qp,kv,kp,indptr,indices,scale):
                out=torch.empty_like(q);ls=torch.empty(B,16,device='cuda')
                rc=launch(*[x.data_ptr() for x in (q,qp,kv,kp,indptr,indices)],scale,out.data_ptr(),ls.data_ptr(),B,torch.cuda.current_stream().cuda_stream)
                if rc:raise RuntimeError(f'MLA launch returned {rc}')
                return out,ls
        elif B==16:
            launch=lib.mla_decode_launch;launch.restype=ctypes.c_int
            launch.argtypes=[ctypes.c_void_p]*6+[ctypes.c_float,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
            lib.mla_decode_max_part.argtypes=[ctypes.c_int];lib.mla_decode_max_part.restype=ctypes.c_int
            mp=lib.mla_decode_max_part(total)+B
            po=torch.empty(B*mp*16*512,device='cuda');pm=torch.empty(B*mp*16*32,device='cuda')
            cnt=torch.zeros(B*16,device='cuda',dtype=torch.int32);prof=torch.zeros(4096*32,device='cuda',dtype=torch.int64)
            def fn(q,qp,kv,kp,indptr,indices,scale):
                out=torch.empty_like(q);ls=torch.empty(B,16,device='cuda')
                rc=launch(*[x.data_ptr() for x in (q,qp,kv,kp,indptr,indices)],scale,B,total,*[x.data_ptr() for x in (out,ls,po,pm,cnt)],mp,torch.cuda.current_stream().cuda_stream,prof.data_ptr())
                if rc:raise RuntimeError(f'MLA launch returned {rc}')
                return out,ls
        else:
            launch=lib.mla_decode_run;launch.restype=ctypes.c_int
            launch.argtypes=[ctypes.c_void_p]*13+[ctypes.c_float,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
            mp=(total+255)//256+B
            po=torch.empty(mp,16,512,device='cuda');pm=torch.empty(mp,16,device='cuda');pl=torch.empty_like(pm)
            cnt=torch.zeros(1,device='cuda',dtype=torch.int32);prefix=torch.zeros(B+1,device='cuda',dtype=torch.int32)
            def fn(q,qp,kv,kp,indptr,indices,scale):
                out=torch.empty_like(q);ls=torch.empty(B,16,device='cuda')
                rc=launch(*[x.data_ptr() for x in (q,qp,kv,kp,indptr,indices,out,ls,po,pm,pl,cnt,prefix)],scale,B,0,torch.cuda.current_stream().cuda_stream)
                if rc:raise RuntimeError(f'MLA launch returned {rc}')
                return out,ls
        return fn,[q,qp,kv,kp,indptr,indices,scale],(expected,lse)

    def topk(self, folder, seed):
        import official
        name='dsa_topk_indexer_fp8_h64_d128_topk2048_ps64'
        prefix={'topk_T1_A_single':'30cecff1','topk_T2_B_mid':'81a953ea'}[folder.parent.name]
        args=official.inputs(name,prefix)
        if folder.parent.name == "topk_T2_B_mid":
            # Generic int8 generation does not encode valid FP8 page scales.
            q, cache, weights, lens, table = args
            q = torch.randn(q.shape, device=q.device).to(q.dtype)
            keys = torch.randn((cache.shape[0], 64, 128), device=q.device).to(torch.float8_e4m3fn)
            scales = torch.rand((cache.shape[0], 64), device=q.device) + 0.01
            packed = torch.cat((keys.view(torch.uint8).reshape(cache.shape[0], -1),
                                scales.view(torch.uint8)), dim=1)
            cache = packed.view(torch.int8).reshape_as(cache)
            weights = torch.randn_like(weights)
            args = (q, cache, weights, lens, table)
        expected=official.reference(name,args)
        print('TOPK_DEFINITION',official._definition(name)[0].model_dump(),flush=True)
        print('TOPK_INPUT',[(tuple(x.shape),str(x.dtype)) for x in args],flush=True)
        if folder.parent.name=='topk_T1_A_single':fn=load(folder/'kernel/topk_triton.py').run_triton
        else:fn=load(folder/'kernel/topk_cuda.py').run
        return fn,args,expected


if __name__ == '__main__':
    unittest.main()
