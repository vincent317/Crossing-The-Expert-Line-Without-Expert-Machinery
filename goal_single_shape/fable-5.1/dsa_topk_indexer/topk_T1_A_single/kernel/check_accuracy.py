#!/usr/bin/env python3
"""
check_accuracy.py
Compares ref, naive fp32, and new CUDA kernel outputs.
Prints E_ref, tol = 7*E_ref, E_new, and PASS/FAIL.
"""
import os, sys
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ref"))

PAGE_SIZE = 64
HEAD_DIM  = 128
K_STRIDE  = 132
NUM_PAGES = 11923


def load_data(device="cuda"):
    from safetensors import safe_open
    path = os.path.join(ROOT, "ref/data/30cecff1-7ea4-474b-90fc-7f4a87206d8e.safetensors")
    tensors = {}
    with safe_open(path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def find_tensors(data):
    seq_lens = block_table = None
    for k, v in data.items():
        kl = k.lower()
        if "seq" in kl and v.ndim == 1:
            seq_lens = v.to(torch.int32)
        elif ("block" in kl or "table" in kl) and v.ndim >= 2:
            block_table = v.to(torch.int32)
    return seq_lens, block_table


def gen_inputs(seq_lens, block_table, device="cuda"):
    B = seq_lens.shape[0]
    q_fp8   = torch.randn(B, 64, HEAD_DIM, device=device).to(torch.float8_e4m3fn)
    weights = torch.rand(B, 64, device=device, dtype=torch.float32) + 0.1

    k_bf16   = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM, device=device,
                            dtype=torch.bfloat16)
    k_absmax = k_bf16.abs().amax(dim=-1, keepdim=True).float()
    k_scale  = (k_absmax / 448.0).clamp(min=1e-8)
    k_fp8    = (k_bf16.float() / k_scale).to(torch.float8_e4m3fn)

    k_cache = torch.zeros(NUM_PAGES, PAGE_SIZE, 1, K_STRIDE,
                          dtype=torch.int8, device=device)
    k_cache[:, :, 0, :HEAD_DIM] = (
        k_fp8.view(torch.int8).reshape(NUM_PAGES, PAGE_SIZE, HEAD_DIM))
    k_scale_i8 = (k_scale.squeeze(-1).contiguous()
                         .view(torch.int8).reshape(NUM_PAGES, PAGE_SIZE, 4))
    k_cache[:, :, 0, HEAD_DIM:K_STRIDE] = k_scale_i8

    return q_fp8, k_cache, weights


def idx_error(a, b):
    va = a[a != -1].float()
    vb = b[b != -1].float()
    n  = min(len(va), len(vb))
    return float((va[:n] - vb[:n]).abs().max()) if n > 0 else 0.0


def main():
    torch.cuda.set_device(0)
    device = "cuda"

    data = load_data(device)
    print("Safetensors keys:", list(data.keys()))
    seq_lens, block_table = find_tensors(data)
    assert seq_lens is not None and block_table is not None, \
        f"Cannot find seq_lens/block_table in {list(data.keys())}"
    print(f"seq_lens={seq_lens}  block_table={block_table}")

    q_fp8, k_cache, weights = gen_inputs(seq_lens, block_table, device)

    # --- ref ---
    from ref_entry import run as ref_run
    ref_out = ref_run(q_fp8, k_cache, weights, seq_lens, block_table)
    torch.cuda.synchronize()
    print(f"ref     : {ref_out[0, :min(10, ref_out.shape[1])]}")

    # --- naive ---
    import naive_fp32
    naive_out = naive_fp32.run(q_fp8, k_cache, weights, seq_lens, block_table)
    print(f"naive   : {naive_out[0, :min(10, naive_out.shape[1])]}")

    E_ref = idx_error(ref_out, naive_out)
    tol   = 7.0 * E_ref
    print(f"\nE_ref = {E_ref:.2f}  (max abs diff ref vs naive, treating -1-padded int indices)")
    print(f"tol   = 7 * E_ref = {tol:.2f}")

    # --- new kernel ---
    import topk_ext
    our_out = topk_ext.run(q_fp8, k_cache, weights, seq_lens, block_table,
                           verbose=True)
    torch.cuda.synchronize()
    print(f"new     : {our_out[0, :min(10, our_out.shape[1])]}")

    E_new = idx_error(our_out, naive_out)
    ok    = (E_new == 0.0) or (E_new <= tol)
    print(f"\nE_new (new kernel vs naive) = {E_new:.2f}")
    print(f"tol                         = {tol:.2f}")
    print(f"ACCURACY: {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
