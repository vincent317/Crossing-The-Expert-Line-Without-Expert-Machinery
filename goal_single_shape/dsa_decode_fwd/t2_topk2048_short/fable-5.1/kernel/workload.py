"""Contest workload 9d4a5f21268e484ea05a2f2af91d9fa7 of
dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 (flashinfer-ai/mlsys26-contest).

Random inputs are regenerated from a fixed seed; sparse_indices is the exact tensor
shipped with the workload (18 and 11 valid rows for the two tokens).
"""
import json
import os
import struct

import numpy as np
import torch

T, H, D_CKV, D_KPE, TOPK, PAGES, PAGE_SIZE = 2, 16, 512, 64, 2048, 8462, 64
SM_SCALE = 0.1352337788608801
WORKLOAD_UUID = "9d4a5f21268e484ea05a2f2af91d9fa7"
_IDX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"sparse_indices_{WORKLOAD_UUID}.safetensors")


def load_sparse_indices():
    raw = open(_IDX_FILE, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    info = json.loads(raw[8:8 + n])["sparse_indices"]
    assert info["dtype"] == "I32" and info["shape"] == [T, TOPK], info
    s, e = info["data_offsets"]
    arr = np.frombuffer(raw[8 + n + s:8 + n + e], dtype=np.int32).reshape(T, TOPK)
    return torch.from_numpy(arr.copy())


def make_inputs(device="cuda", seed=20260823):
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(shape, generator=g, device=device, dtype=torch.bfloat16)

    return dict(
        q_nope=rnd(T, H, D_CKV),
        q_pe=rnd(T, H, D_KPE),
        ckv_cache=rnd(PAGES, PAGE_SIZE, D_CKV),
        kpe_cache=rnd(PAGES, PAGE_SIZE, D_KPE),
        sparse_indices=load_sparse_indices().to(device),
        sm_scale=SM_SCALE,
    )


def describe(inp):
    idx = inp["sparse_indices"]
    valid = [int((idx[t] != -1).sum()) for t in range(idx.shape[0])]
    return (f"workload {WORKLOAD_UUID}: T={idx.shape[0]} H={H} ckv={D_CKV} kpe={D_KPE} "
            f"topk={TOPK} pages={PAGES} bf16, valid rows per token={valid} (total {sum(valid)}), "
            f"sm_scale={inp['sm_scale']}")
