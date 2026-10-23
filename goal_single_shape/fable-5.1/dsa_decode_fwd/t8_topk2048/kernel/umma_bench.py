"""UMMA micro-benchmark: per-instruction cost of tcgen05.mma (N=16) on B200, printed in cycles."""

import sys

import torch

import dsa_kernel_tcgen05 as dsa_kernel

ext = dsa_kernel._load()
torch.cuda.init()
cases = [(m64, 1) for m64 in (0, 1)] + [(m64, iss) for iss in (2, 4) for m64 in (0, 1)]
for m64, issuers in cases:
    for n_mma, n_acc in ((1, 1), (36, 1), (36, 4), (36, 8), (144, 4)):
        try:
            for _ in range(2):  # warm
                ext.umma_bench(n_mma, n_acc, m64, issuers)
            r = ext.umma_bench(n_mma, n_acc, m64, issuers)
            print(f"[umma] M={'64' if m64 else '128'} issuers={issuers} n_mma={n_mma:3d} n_acc={n_acc}: "
                  f"issue={r[0]:6d} cyc ({r[0] / max(n_mma, 1):6.1f}/mma)  commit->done={r[1]:6d} cyc  tmem_ld={r[2]:5d} cyc",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[umma] M={'64' if m64 else '128'} issuers={issuers} n_mma={n_mma} n_acc={n_acc}: FAILED "
                  f"{type(exc).__name__}: {str(exc)[:100]}", flush=True)
            sys.exit(1)
