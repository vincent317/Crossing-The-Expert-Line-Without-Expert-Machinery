"""Validate and time pinned expert baselines on complete official workloads."""
import hashlib
import json
import os
from pathlib import Path
import unittest
import torch
import official
from run_dsa import load
from paired import Captured, prepare
import kernel_span
from metrics import compare


class ExpertBaselineTests(unittest.TestCase):
    def test_workload(self):
        spec = json.loads(os.environ['BASELINE_SPEC'])
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        official.DATA = Path(spec['dataset'])
        folder = Path(os.environ['BASELINE_REFERENCE']) / spec['expert']
        sources = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in folder.rglob('*') if p.suffix in ('.py', '.cu', '.cuh', '.h')}
        module = load(folder/'kernel.py', 'baseline_expert')
        args = official.inputs(spec['definition'], spec['uuid'])
        expected = official.reference(spec['definition'], args)
        call, reset, _ = prepare(Captured(module.run, args, {}))
        record = dict(spec=spec, source_sha256=sources, metrics=[], timing=[], complete=False,
                      gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        destination = Path(os.environ['W'])/'results'/(spec['uuid']+'.json')
        destination.parent.mkdir(exist_ok=True)
        def check():
            reset()
            actual = call()
            actual = (actual,) if isinstance(actual, torch.Tensor) else actual
            self.assertEqual(len(actual), len(expected))
            rows = []
            for a, b in zip(actual, expected):
                row = compare(a, b, b, operator='moe' if spec['expert']=='moe' else None)
                row['passed'] = row['flashinfer_bench']['standard_pass']
                rows.append(row)
            record['metrics'].append(rows)
            destination.write_text(json.dumps(record))
            self.assertTrue(all(r['passed'] for r in rows), str(rows))
        check()
        for _ in range(3):
            record['timing'].append(kernel_span.trace(call, reset, repeats=50))
        check()
        record['complete'] = True
        destination.write_text(json.dumps(record))
        print('BASELINE_PASS', spec['expert'], spec['uuid'], flush=True)


if __name__ == '__main__':
    unittest.main()
