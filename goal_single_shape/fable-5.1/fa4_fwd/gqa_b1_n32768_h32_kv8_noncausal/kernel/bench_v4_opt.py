import torch
import common as C
from fa_kernel_v4_opt import flash_attn_fwd_v4

def main():
    print(torch.cuda.get_device_name(0), flush=True)
    q, k, v = C.make_inputs()
    
    # warmup + quick timing
    o = flash_attn_fwd_v4(q, k, v, softmax_scale=C.SCALE)
    torch.cuda.synchronize()
    
    FLOPS = 4.0 * C.B * C.H * C.S * C.S * C.D
    fn = lambda: flash_attn_fwd_v4(q, k, v, softmax_scale=C.SCALE)
    med, lo, hi = C.cuda_event_median(fn, warmup=3, iters=10)
    print(f"v4_opt (maxnreg=160, no branches): {med:7.3f} ms  {FLOPS/(med*1e-3)/1e12:6.1f} TFLOP/s", flush=True)
    
    # profiler span
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(10):
        flash_attn_fwd_v4(q, k, v, softmax_scale=C.SCALE)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":
    main()
