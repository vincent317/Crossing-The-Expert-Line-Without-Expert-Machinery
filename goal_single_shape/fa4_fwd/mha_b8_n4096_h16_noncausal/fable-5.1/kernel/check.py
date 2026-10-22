"""Accuracy check: custom tcgen05 kernel vs fp32 naive, with the tolerance
derived from the reference implementation's own error against fp32 naive.
"""

import torch

import common as C
from build_ext import build


def _run_case(mod, label, q, k, v):
    ref = C.attn_fp32_naive(q, k, v)
    out = mod.fwd(q, k, v, C.SOFTMAX_SCALE)
    torch.cuda.synchronize()
    nan = torch.isnan(out).sum().item()
    print(f"[bisect {label}] max_abs_err vs fp32 = "
          f"{C.max_abs_err(out, ref):.6e}  nan_count={nan}")


def bisect_seqlens(mod):
    """Shorter sequences isolate pipeline stages: seqlen=128 is a single
    iteration (no rescale, no prefetch); 256 adds exactly one rescale.
    Structured inputs at seqlen=128 further localise a failure:
      V = identity  -> O == P, tests the P smem layout and softmax only
      Q = 0         -> P uniform, O == mean_k V, tests the V^T layout only
    """
    q, k, v = C.make_inputs(seqlen=128)
    _run_case(mod, "seqlen=128 random", q, k, v)
    v_eye = torch.zeros_like(v)
    eye = torch.eye(128, device=v.device, dtype=v.dtype)  # (key, d)
    v_eye[:] = eye[None, :, None, :]
    _run_case(mod, "seqlen=128 V=I (O==P)", q, k, v_eye)
    _run_case(mod, "seqlen=128 Q=0 (O==mean V)", torch.zeros_like(q), k, v)
    q, k, v = C.make_inputs(seqlen=256)
    _run_case(mod, "seqlen=256 random", q, k, v)


def main():
    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
    mod = build()
    bisect_seqlens(mod)
    q, k, v = C.make_inputs()

    ref_fp32 = C.attn_fp32_naive(q, k, v)

    from flash_attn.cute.interface import flash_attn_func
    out_ref = flash_attn_func(q, k, v, softmax_scale=C.SOFTMAX_SCALE, causal=False)
    if isinstance(out_ref, (tuple, list)):
        out_ref = out_ref[0]
    e_ref = C.max_abs_err(out_ref, ref_fp32)
    tol = 7.0 * e_ref
    print(f"E_ref (ref vs fp32_naive)      = {e_ref:.6e}")
    print(f"tol = 7 * E_ref                = {tol:.6e}")

    out_mine = mod.fwd(q, k, v, C.SOFTMAX_SCALE)
    torch.cuda.synchronize()
    print(f"nan_count = {torch.isnan(out_mine).sum().item()}")
    e_mine = C.max_abs_err(out_mine, ref_fp32)
    print(f"E_mine (mine vs fp32_naive)    = {e_mine:.6e}")
    print(f"PASS = {e_mine <= tol}  ({e_mine:.6e} <= {tol:.6e})")

    # extra diagnostics
    d = (out_mine.float() - ref_fp32).abs()
    print(f"mine vs fp32 mean abs err      = {d.mean().item():.6e}")
    print(f"mine vs ref  max abs err       = {C.max_abs_err(out_mine, out_ref):.6e}")
    bad = (d > tol).nonzero()
    print(f"num elements over tol          = {bad.shape[0]} / {d.numel()}")
    if bad.shape[0]:
        i = tuple(bad[0].tolist())
        print(f"first bad idx {i}: mine={out_mine[i].item():.6f} "
              f"fp32={ref_fp32[i].item():.6f} ref={out_ref[i].item():.6f}")
    return e_mine <= tol


if __name__ == "__main__":
    ok = main()
    print("RESULT", "PASS" if ok else "FAIL")
