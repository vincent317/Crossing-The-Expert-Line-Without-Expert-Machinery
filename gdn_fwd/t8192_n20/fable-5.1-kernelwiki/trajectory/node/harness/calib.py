"""Machine-state calibration.

Times two fixed, public workloads with exactly the same protocol as the task
benchmark.  Re-run this at any time: if the numbers still match the values
recorded in ANCHOR.md, the GPU is in the same state as when the reference
anchor was measured, and the anchor is still directly comparable.
"""
import sys
sys.path.insert(0, "/root/gdn/harness")
import torch, gdn_bench

torch.manual_seed(0)
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
c = torch.empty(4096, 4096, dtype=torch.bfloat16, device="cuda")
x = torch.randn(64 << 20, dtype=torch.float32, device="cuda")
y = torch.empty_like(x)

g1 = gdn_bench.device_time_ms(lambda: torch.mm(a, b, out=c), warmup=10, iters=50, trials=5)
g2 = gdn_bench.device_time_ms(lambda: torch.add(x, 1.0, out=y), warmup=10, iters=50, trials=5)
print(f"CALIB gemm_bf16_4096   {g1*1000:9.3f} us   ({2*4096**3/g1*1e-9:.1f} TFLOP/s)")
print(f"CALIB stream_add_256MB {g2*1000:9.3f} us   ({2*x.numel()*4/g2*1e-6/1000:.2f} TB/s)")
print("sm clock:", torch.cuda.get_device_name(0))
