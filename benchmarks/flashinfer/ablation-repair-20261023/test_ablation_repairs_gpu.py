"""Validate repaired ablations using the pinned audit harness and frozen inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

@unittest.skipUnless(os.environ.get('ABLATION_REPAIR_CASE'), 'Requires the staged B200 audit harness')
class AblationRepairsTests(unittest.TestCase):
    def test_accuracy_repeated_calls_and_performance(self):
        import torch
        import official
        import kernel_span
        from metrics import compare
        from paired import Captured, prepare
        from tridao import compare as compare_gradient
        root = Path(os.environ['W'])
        relative = os.environ['ABLATION_REPAIR_CASE']
        folder = root / 'repository' / relative
        sys.path.insert(0, str(folder))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result = dict(path=relative, complete=False, metrics=[], timing=[],
                      source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in folder.iterdir() if p.suffix in ('.cu', '.cuh', '.py')})
        output = root / 'results' / (relative.replace('/', '__') + '.json')
        output.parent.mkdir(exist_ok=True)
        def load(name):
            spec = importlib.util.spec_from_file_location('repair_candidate', folder / name)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        mode = 'dsa' if '/dsa_decode_fwd/' in relative else 'fa4' if '/fa4_bwd/' in relative else 'gdn'
        flags = ['nvcc', '-O3', '-std=c++17', '-gencode', 'arch=compute_100a,code=sm_100a']
        if mode == 'dsa':
            subprocess.run(flags + ['-cubin', '-DKEYS_PER_CTA=32', str(folder/'dsa_kernel.cu'), '-o', str(folder/'dsa_s32.cubin')], check=True)
            os.environ['DSA_KEYS_PER_CTA'] = '32'
            candidate = load('kernel.py').run
        elif mode == 'fa4':
            subprocess.run(flags + ['--expt-relaxed-constexpr', '-DCUTE_ARCH_MMA_SM100A_ENABLED',
                           '-I'+os.environ['CUTLASS_INCLUDE'], '-shared', '-Xcompiler', '-fPIC',
                           str(folder/'fa4bwd.cu'), '-o', str(folder/'libfa4bwd.so'), '-lcuda'], check=True)
            candidate = load('fa4bwd.py').fa4_bwd
        else:
            subprocess.run(flags + ['--expt-relaxed-constexpr', '-shared', '-Xcompiler', '-fPIC',
                           str(folder/'gdn_fwd_v3.cu'), '-o', str(folder/'libgdn.so'), '-lcuda'], check=True)
            os.environ['GDN_LIB'] = str(folder/'libgdn.so')
            candidate = load('mykernel.py').gdn_fwd
        for seed in ((-1, 0, 42) if mode == 'gdn' else (0, 42)):
            torch.manual_seed(seed)
            if mode == 'dsa':
                definition = 'dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64'
                args = official.inputs(definition, '564007ac')
                expected = official.reference(definition, args)
            elif mode == 'gdn':
                from run_gdn import workload, DEFINITION
                args = workload('a9540651')
                if seed >= 0:
                    q,k,v,st,A,a,dt,b,cu,scale = args
                    norm = lambda x: torch.nn.functional.normalize(torch.randn_like(x,dtype=torch.float32),dim=-1).bfloat16()
                    args = [norm(q),norm(k),torch.randn_like(v),torch.randn_like(st),torch.randn_like(A),torch.randn_like(a),torch.randn_like(dt),torch.randn_like(b),cu,scale]
                expected = official.reference(DEFINITION, args)
            else:
                data = torch.load(root/'reference_cache'/f'fa4-varlen-{seed}.pt', weights_only=True, map_location='cuda')
                expected, pytorch = torch.load(root/'tridao_cache'/f'varlen-{seed}.pt', weights_only=True, map_location='cuda')
                args = [data[k] for k in ('q','k','v','o','do','lse','cu')] + [128**-.5]
            call, reset, _ = prepare(Captured(candidate, args, {}))
            for iteration in range(100):
                reset()
                got = call()
                torch.cuda.synchronize()
                self.assertEqual(len(got), len(expected))
                if iteration in (0, 99):
                    metrics = [compare_gradient(a,b,c) for a,b,c in zip(got,expected,pytorch)] if mode == 'fa4' else [compare(a,b,b)['flashinfer_bench'] for a,b in zip(got,expected)]
                    result['metrics'].append(dict(seed=seed, iteration=iteration, outputs=metrics))
                    output.write_text(json.dumps(result, allow_nan=False))
                    self.assertTrue(all(m['passed' if mode == 'fa4' else 'standard_pass'] for m in metrics), metrics)
            if seed == (-1 if mode == 'gdn' else 0):
                for _ in range(3):
                    result['timing'].append(kernel_span.trace(call, reset, repeats=50))
            output.write_text(json.dumps(result, allow_nan=False))
        result['complete'] = True
        output.write_text(json.dumps(result, allow_nan=False))
        print('ABLATION_REPAIR_PASS', relative, flush=True)

if __name__ == '__main__':
    unittest.main()
