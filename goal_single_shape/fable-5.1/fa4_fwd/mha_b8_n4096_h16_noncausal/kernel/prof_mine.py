"""Run the custom kernel alone, for ncu profiling."""

import torch

import common as C
from build_ext import build

mod = build(verbose=False)
q, k, v = C.make_inputs()
for _ in range(3):
    mod.fwd(q, k, v, C.SOFTMAX_SCALE)
torch.cuda.synchronize()
mod.fwd(q, k, v, C.SOFTMAX_SCALE)
torch.cuda.synchronize()
print("done")
