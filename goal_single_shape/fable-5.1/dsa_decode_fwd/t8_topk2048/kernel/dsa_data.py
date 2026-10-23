"""Workload 564007ac inputs for dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.

Random tensors follow the flashinfer-bench protocol (torch.randn in the target
dtype, definition input order); sparse_indices come from the official
safetensors blob shipped in spec/.
"""

import json
import struct
from pathlib import Path

import torch

SPEC_DIR = Path(__file__).resolve().parent / "spec"
INDICES_BLOB = SPEC_DIR / "sparse_indices_564007ac.safetensors"

NUM_TOKENS = 8
NUM_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
NUM_PAGES = 8462
TOPK = 2048
SM_SCALE = 0.1352337788608801


def load_sparse_indices(device: str) -> torch.Tensor:
    raw = INDICES_BLOB.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    info = header["sparse_indices"]
    assert info["dtype"] == "I32", info
    begin, end = info["data_offsets"]
    data = bytearray(raw[8 + header_len + begin : 8 + header_len + end])
    idx = torch.frombuffer(data, dtype=torch.int32).reshape(info["shape"]).clone()
    assert list(idx.shape) == [NUM_TOKENS, TOPK], idx.shape
    return idx.to(device)


def make_inputs(device: str = "cuda", seed: int = 20260823) -> dict:
    torch.manual_seed(seed)
    q_nope = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    sparse_indices = load_sparse_indices(device)
    return dict(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        sparse_indices=sparse_indices,
        sm_scale=SM_SCALE,
    )


def describe_inputs(inp: dict) -> None:
    idx = inp["sparse_indices"]
    valid = (idx != -1).sum(dim=1).tolist()
    print(f"[data] num_tokens={idx.shape[0]} topk={idx.shape[1]} valid_rows_per_token={valid} total={sum(valid)}")
    for k, v in inp.items():
        if torch.is_tensor(v):
            print(f"[data] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"[data] {k}: {v}")
