"""Profiler kernel-span timing of the custom kernel only (for ablation runs,
where results are intentionally wrong and accuracy is meaningless)."""

import torch

import common as C
from build_ext import build

mod = build(verbose=False)
q, k, v = C.make_inputs()


def run_mine():
    return mod.fwd(q, k, v, C.SOFTMAX_SCALE)


run_mine()
torch.cuda.synchronize()
med, spans = C.time_profiler_kernel_span(run_mine, tag="mine")
print(f"ABLATE_RESULT profiler_median={med:.2f} us  min={min(spans):.2f} max={max(spans):.2f}")
