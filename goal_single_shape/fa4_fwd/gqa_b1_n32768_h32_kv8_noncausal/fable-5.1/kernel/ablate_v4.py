"""Ablation timing for the Gluon v4 kernel: which partition bounds the pipeline?

MODE 0 full kernel | 1 no exp2 (linear stand-in) | 2 no softmax/rescale math (barriers only)
| 3 no tcgen05 MMA (barriers only) | 4 no TMA loads (barriers only).
Output is numerically wrong for MODE != 0; only device time matters here.
"""
import torch
import common as C
from fa_kernel_v4 import flash_attn_fwd_v4

FLOPS = 4.0 * C.B * C.H * C.S * C.S * C.D
RUNS = [
    ("full  MATH=0 (plain)      ", dict(MODE=0, MATH=0)),
    ("full  MATH=1 (f32x2+f16x2)", dict(MODE=0, MATH=1)),
    ("no-softmax-math           ", dict(MODE=2, MATH=0)),
    ("no-mma        MATH=1      ", dict(MODE=3, MATH=1)),
]


def main():
    print(torch.cuda.get_device_name(0), flush=True)
    q, k, v = C.make_inputs()
    # accuracy of the packed path vs the plain path on a 4K slice (fp32 sdpa reference)
    qs, ks, vs = (t[:, :4096].contiguous() for t in (q, k, v))
    ref_s = torch.nn.functional.scaled_dot_product_attention(
        qs.transpose(1, 2).float(), ks.transpose(1, 2).float(), vs.transpose(1, 2).float(),
        scale=C.SCALE).transpose(1, 2)
    for m in (0, 1):
        o = flash_attn_fwd_v4(qs, ks, vs, softmax_scale=C.SCALE, MATH=m)
        print(f"  S=4096 MATH={m}: max err vs fp32 = {C.max_err(o, ref_s):.3e}", flush=True)
    for tag, kw in RUNS:
        try:
            fn = lambda: flash_attn_fwd_v4(q, k, v, softmax_scale=C.SCALE, **kw)
            fn(); torch.cuda.synchronize()
            med, lo, hi = C.cuda_event_median(fn, warmup=3, iters=10)
            print(f"  {tag}: {med:8.4f} ms  ({FLOPS/(med*1e-3)/1e12:7.1f} TFLOP/s-equiv)  min {lo:.4f}",
                  flush=True)
        except Exception as e:
            print(f"  {tag}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
