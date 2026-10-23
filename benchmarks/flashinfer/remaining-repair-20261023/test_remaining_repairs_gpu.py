"""Regression adapters for the remaining repaired repository candidates."""
import ctypes
import os
from pathlib import Path
import sys
import unittest
import torch
import test_remaining_forward_base_gpu as forward

class RemainingRepairsTests(forward.RepositoryForwardTests):
    def dsa(self, folder):
        if 'kda05_reproduction' not in str(folder):
            return super().dsa(folder)
        import official
        definition='dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64'
        prefix={'dsa_triton_v11':'9d4a5f21','dsa_triton_v14':'05f6de65','dsa_triton_v15':'564007ac'}[folder.name]
        args=official.inputs(definition,prefix)
        return forward.load(folder/'main.py').run,args,official.reference(definition,args)

    def kda(self, folder, seed):
        if folder.parent.name=='h96_mixed':return super().kda(folder,seed)
        root=Path(os.environ['W']);case=folder.parent.name
        d=torch.load(root/'reference_cache'/f'kda-{case}-{seed}.pt',weights_only=True,map_location='cuda')
        args=[d[x] for x in ('q','k','v','g','beta','A_log','dt_bias','state')]
        scale=d['scale']
        if case=='h64_fixed8192' and folder.name=='opus-4.8':
            mod=forward.load(folder/'kda_triton12.py')
            def fn(q,k,v,g,beta,A,bias,state):
                qn=(q.float()*torch.rsqrt(q.float().square().sum(-1,keepdim=True)+1e-6)).bfloat16()
                kn=(k.float()*torch.rsqrt(k.float().square().sum(-1,keepdim=True)+1e-6)).bfloat16()
                gl=-5.*torch.sigmoid(A.exp()[None,None,:,None]*(g.float()+bias))
                return mod.kda_forward12(qn,kn,v,gl,beta.float().sigmoid(),scale)
        elif case=='h64_fixed8192':
            mod=forward.load(folder/'kda_cuda2.py');mod.NVCC=os.environ['NVCC']
            def fn(q,k,v,g,beta,A,bias,state):
                out,st=mod.kda_fwd(q,k,v,g,beta.float().sigmoid().bfloat16(),A,bias,scale,state.transpose(-1,-2).float().contiguous())
                return out,st.transpose(-1,-2)
        elif folder.name=='opus-4.8':
            mod=forward.load(folder/'triton_kda_best.py')
            def fn(q,k,v,g,beta,A,bias,state):
                inp=dict(d,q=q,k=k,v=v,g=g,beta=beta,A_log=A,dt_bias=bias,state=state,N=8,L=1024,H=96,K=128,V=128,T=8192)
                return mod.triton_kda(inp,lower_bound=-5.,scale=scale)
        else:
            lib=forward.compile_shared(folder/'kda_fwd.cu',folder/'libkda.so')
            lib.kda_fwd_launch.argtypes=[ctypes.c_void_p]*10+[ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
            mod=forward.load(folder/'kda_cuda.py');mod._lib=lib
            cu=d['cu_seqlens'].long()
            def fn(q,k,v,g,beta,A,bias,state):
                out=mod.kda_fwd(q,k,v,g,beta,A,bias,state,cu,lower_bound=-5.,scale=scale)
                return out,state
        return fn,args,d['fp']

    def mla_prefill(self,folder):
        import itertools
        import types
        common=types.ModuleType('common');common.H=16;common.D_CKV=512;sys.modules['common']=common
        timing=types.ModuleType('timing');timing.bench=None;sys.modules['timing']=timing
        forward.compile_shared(folder/'mla_kernel_final.cu',folder/'libmla_final_c2.so')
        fn=forward.load(folder/'mla_prefill.py').mla_prefill
        ql=[4086,4]+[737]*15+[738];kl=[n+3 for n in ql];pool=989669
        def rand(*shape):return torch.randn(shape,device='cuda',dtype=torch.bfloat16)
        q,qp,k,kp=rand(sum(ql),16,512),rand(sum(ql),16,64),rand(pool,1,512),rand(pool,1,64)
        qi=torch.tensor([0]+list(itertools.accumulate(ql)),device='cuda',dtype=torch.int32)
        ki=torch.tensor([0]+list(itertools.accumulate(kl)),device='cuda',dtype=torch.int32)
        ids=torch.randperm(pool,device='cuda')[:sum(kl)].int()
        fp=torch.empty_like(q,dtype=torch.float32);fl=torch.empty((sum(ql),16),device='cuda')
        for n,(nq,nk) in enumerate(zip(ql,kl)):
            a,b=int(qi[n]),int(qi[n+1]);ix=ids[ki[n]:ki[n+1]].long();ks=k[ix,0].float();ps=kp[ix,0].float()
            mask=torch.arange(nk,device='cuda')[None,:]>torch.arange(nq,device='cuda')[:,None]+nk-nq
            for h in range(16):
                scores=(q[a:b,h].float()@ks.T+qp[a:b,h].float()@ps.T)*576**-.5
                scores.masked_fill_(mask,-float('inf'));fp[a:b,h]=scores.softmax(-1)@ks;fl[a:b,h]=scores.logsumexp(-1)
        return fn,[q,qp,k,kp,qi,ki,ids,576**-.5],(fp,fl)

if __name__=='__main__':unittest.main()
