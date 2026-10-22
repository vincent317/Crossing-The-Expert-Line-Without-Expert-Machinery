set -e
echo "=== build v4 (QTILE=128 KTILE=128) ==="
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -DFA_MODE=1 \
  -Xptxas -v --shared -Xcompiler -fPIC fa_fwd4.cu -o libfa_fwd4.so 2>&1 | \
  grep -E "error|warning|registers|spill|smem" | head -20
ls -la libfa_fwd4.so || exit 1

echo "=== smoke test v4 ==="
export PIP_CONSTRAINT=
pip install -q torch 2>&1 | grep -v "already satisfied" || true
python -u - <<'PY'
import torch, ctypes
torch.manual_seed(42)
lib = ctypes.CDLL("./libfa_fwd4.so")
q = torch.randn(512, 4, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(512, 4, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(512, 4, 128, dtype=torch.bfloat16, device="cuda")
out = torch.zeros_like(q)
cu = torch.tensor([0, 128, 512], dtype=torch.int32, device="cuda")
lib.fa_fwd_entry(
    ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
    ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(cu.data_ptr()),
    ctypes.c_void_p(cu.data_ptr()), ctypes.c_void_p(out.data_ptr()),
    ctypes.c_int(4), ctypes.c_float(1.0/11.31371)
)
torch.cuda.synchronize()
print("smoke: kernel launched, no crash")
err = torch.cuda.get_device_properties(0).name
print(f"device: {err}")
PY

echo "=== all done ==="
