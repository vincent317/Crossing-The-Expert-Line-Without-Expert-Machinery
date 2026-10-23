"""Validate the newly published T3 kernel on its canonical shapes and page map."""
import os
import json
import unittest
from pathlib import Path
import torch
import test_remaining_forward_base_gpu as forward


class T3TopkTests(forward.RepositoryForwardTests):
    def test_requested_candidate(self):
        root = Path(os.environ['W'])
        relative = os.environ['AUDIT_CASE']
        for distribution in ('signed', 'nonnegative'):
            self.distribution = distribution
            super().test_requested_candidate()
            path = root / 'results' / (relative.replace('/', '__') + '.json')
            record = json.loads(path.read_text())
            record['weight_distribution'] = distribution
            record['repeated_calls_verified_per_seed'] = 100
            target = path.with_name(path.stem + '-' + distribution + '.json')
            target.write_text(json.dumps(record))
            path.unlink()

    def topk(self, folder, seed):
        import official
        definition = 'dsa_topk_indexer_fp8_h64_d128_topk2048_ps64'
        q, cache, weights, lengths, table = official.inputs(definition, '8635db8f')
        self.assertEqual(tuple(q.shape), (27, 64, 128))
        self.assertEqual(tuple(table.shape), (27, 91))
        self.assertEqual(int(lengths.max()), 5806)
        q = torch.randn(q.shape, device=q.device).to(q.dtype)
        keys = torch.randn((cache.shape[0], 64, 128), device=q.device, dtype=torch.bfloat16).float()
        scales = keys.abs().amax(-1) / 448.0
        quantized = (keys / scales[..., None]).to(torch.float8_e4m3fn)
        packed = torch.cat((quantized.view(torch.uint8).reshape(cache.shape[0], -1),
                            scales.contiguous().view(torch.uint8)), dim=1)
        cache = packed.view(torch.int8).reshape_as(cache)
        weights = torch.randn_like(weights)
        if self.distribution == "nonnegative":
            weights = weights.abs()
        args = (q, cache, weights, lengths, table)
        expected = official.reference(definition, args)
        self.assertEqual(len(expected), 1)
        module = forward.load(folder / 'kernel/topk_indexer.py')
        expected_members = expected[0].sort(-1).values
        for repeat in range(100):
            actual = module.run(*args)
            self.assertTrue(torch.equal(actual.sort(-1).values, expected_members),
                            f'Top-k membership mismatch at seed={seed}, repeat={repeat}')
            for scratch in module._scratch.values():
                self.assertEqual(int(torch.count_nonzero(scratch)), 0,
                                 f'Scratch not reset at seed={seed}, repeat={repeat}')
        print('T3_REPEAT_PASS', self.distribution, seed, 100, 'exact_membership_and_zero_scratch', flush=True)
        print('T3_INPUT', list(q.shape), list(cache.shape), lengths.tolist(), flush=True)
        print('T3_EXTENSION', module._ext.__file__, flush=True)
        return module.run, args, expected


if __name__ == '__main__':
    unittest.main()
