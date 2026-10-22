"""Official FlashInfer-Bench workloads for definition
`gdn_prefill_qk4_v8_d128_k_last` (MLSys 2026 contest dataset
flashinfer-ai/mlsys26-contest).  Inputs come straight from the dataset
safetensors blobs -- nothing is synthesised.

TARGET workload is `a9540651` (total_seq_len=8192, num_seqs=20).  That is the
only one performance is judged on.  The other four are there so you can check
that your kernel is correct in general and not just on one shape.
"""
import os
import torch
from safetensors.torch import load_file

DATA_DIR = os.environ.get("GDN_DATA_DIR", "/root/gdn/data")
SCALE = 0.08838834764831843  # "scale" scalar input, identical for all workloads

# short name -> (file stem, total_seq_len, num_seqs)
WORKLOADS = {
    "a9540651": ("a9540651-862b-44af-ad75-57e2d9caf37a", 8192, 20),   # TARGET
    "77daf91d": ("77daf91d-0660-4c4b-8c32-336a69281cd9", 6, 1),
    "fc7a2bcb": ("fc7a2bcb-25d4-4840-a095-959499223267", 139, 3),
    "f4926229": ("f4926229-e4ba-49a3-8b2f-df2ec79c333b", 1377, 1),
    "410794d4": ("410794d4-6f70-4bb4-aed1-4669be9a610f", 8192, 39),
}
TARGET = "a9540651"

_KEYS = ("q", "k", "v", "state", "A_log", "a", "dt_bias", "b", "cu_seqlens")


def load(name=TARGET, device="cuda"):
    """Return dict of the 9 tensor inputs (on `device`) plus the scalar scale."""
    stem = WORKLOADS[name][0]
    path = os.path.join(DATA_DIR, f"workload_{stem}.safetensors")
    t = load_file(path)
    out = {k: t[k].to(device) for k in _KEYS}
    out["scale"] = SCALE
    return out


def args(name=TARGET, device="cuda"):
    """Positional args in definition order:
    (q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)"""
    d = load(name, device)
    return (d["q"], d["k"], d["v"], d["state"], d["A_log"], d["a"],
            d["dt_bias"], d["b"], d["cu_seqlens"], d["scale"])


if __name__ == "__main__":
    for name in WORKLOADS:
        d = load(name)
        print(f"=== {name}  T={WORKLOADS[name][1]} N={WORKLOADS[name][2]}"
              f"{'   <-- TARGET' if name == TARGET else ''}")
        for k in _KEYS:
            print(f"    {k:11s} {tuple(d[k].shape)} {str(d[k].dtype):16s}"
                  f" absmax={d[k].float().abs().max().item():.4g}")
        print("    cu_seqlens", d["cu_seqlens"].tolist())
