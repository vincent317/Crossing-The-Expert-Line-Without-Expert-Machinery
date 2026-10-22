"""Correctness judge for the GDN prefill task (workload a9540651).

Two frozen golden files live in /root/gdn/data:

  golden_spec_fp32.safetensors
      Output of the official definition's fp32 sequential reference
      (gdn_spec_reference.run) on this workload.  Numerical ground truth.

  golden_ref.safetensors
      Output of the reference kernel you have to beat, on the same inputs.
      Its source is not on this machine; only its outputs and its device
      time are.

The tolerance is derived from the data, not guessed.  The reference kernel is
a bf16-grade implementation that is already accepted as correct for this task,
so the distance between it and the fp32 spec is the noise floor, and you get
7x that much room:

  floor_max = max | spec - ref |                       per output tensor
  floor_rms = rms( spec - ref ) / rms( spec )          per output tensor
  tol_*     = 7 * floor_*

  PASS  <=>  max | yours - spec | <= tol_max
        and  rms( yours - spec ) / rms(spec) <= tol_rms

Both are required.  `tol_max` alone would miss an error that is small
everywhere but present everywhere; `tol_rms` alone would miss a single very
wrong element.  The contest's own criterion for this definition
(atol=rtol=1e-2, required_matched_ratio=1.0) is reported as well, for
information -- but the 7x-floor rule is the one that decides.
"""
import os
import torch
from safetensors.torch import load_file

DATA_DIR = os.environ.get("GDN_DATA_DIR", "/root/gdn/data")
TOL_FACTOR = 7.0
_CONTEST_ATOL, _CONTEST_RTOL = 0.01, 0.01
NAMES = ("output", "new_state")

_spec = None
_ref = None


def goldens():
    """(fp32 spec golden, reference-kernel golden) for workload a9540651."""
    global _spec, _ref
    if _spec is None:
        _spec = load_file(os.path.join(DATA_DIR, "golden_spec_fp32.safetensors"))
        _ref = load_file(os.path.join(DATA_DIR, "golden_ref.safetensors"))
    return _spec, _ref


def _metrics(mine, truth):
    """(max_abs_err, relative_rms_err) in float64."""
    mine = mine.detach().float().double().cpu()
    truth = truth.detach().float().double().cpu()
    d = mine - truth
    rms_t = truth.pow(2).mean().sqrt().item()
    return d.abs().max().item(), d.pow(2).mean().sqrt().item() / max(rms_t, 1e-30)


def tolerances():
    """name -> dict(floor_max, tol_max, floor_rms, tol_rms)."""
    spec, ref = goldens()
    out = {}
    for name in NAMES:
        fmax, frms = _metrics(ref[name], spec[name])
        out[name] = dict(floor_max=fmax, tol_max=TOL_FACTOR * fmax,
                         floor_rms=frms, tol_rms=TOL_FACTOR * frms)
    return out


def _report(name, mine, truth, tol):
    if mine is None:
        print(f"  {name}: missing"); return False
    mine = mine.detach()
    truth = truth.detach()
    if tuple(mine.shape) != tuple(truth.shape):
        print(f"  {name}: SHAPE MISMATCH {tuple(mine.shape)} vs {tuple(truth.shape)}")
        return False
    if not torch.isfinite(mine.float()).all():
        print(f"  {name}: contains NaN/Inf")
        return False
    emax, erms = _metrics(mine, truth)
    d = (mine.float().cpu() - truth.float().cpu()).abs()
    nbad = int((d > tol["tol_max"]).sum())
    contest = torch.allclose(mine.float().cpu(), truth.float().cpu(),
                             atol=_CONTEST_ATOL, rtol=_CONTEST_RTOL)
    ok_max = emax <= tol["tol_max"]
    ok_rms = erms <= tol["tol_rms"]
    ok = ok_max and ok_rms
    print(f"  {name:10s} max_abs={emax:.4e} (tol {tol['tol_max']:.4e}"
          f" {'ok' if ok_max else 'OVER'}, {nbad} elems over)   "
          f"rel_rms={erms:.4e} (tol {tol['tol_rms']:.4e}"
          f" {'ok' if ok_rms else 'OVER'})   "
          f"contest_1e-2={'pass' if contest else 'FAIL'}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def check(output, new_state, verbose=True):
    """Judge a candidate result for the target workload a9540651."""
    spec, _ = goldens()
    tol = tolerances()
    if verbose:
        for n in NAMES:
            t = tol[n]
            print(f"  floor[{n}]: max={t['floor_max']:.4e} rel_rms={t['floor_rms']:.4e}"
                  f"  -> tol x{TOL_FACTOR:g}: max={t['tol_max']:.4e}"
                  f" rel_rms={t['tol_rms']:.4e}")
    got = {"output": output, "new_state": new_state}
    ok = all([_report(n, got[n], spec[n], tol[n]) for n in NAMES])
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def check_arbitrary(fn, args, verbose=False):
    """Judge `fn(*args)` against the fp32 spec on ANY inputs (other workloads,
    shapes or random data), using the same relative tolerances."""
    import gdn_spec_reference as spec_mod
    tol = tolerances()
    so, ss = spec_mod.run(*args)
    mo, ms = fn(*args)
    truth = {"output": so, "new_state": ss}
    got = {"output": mo, "new_state": ms}
    if verbose:
        print("  tolerances:", {n: (f"{tol[n]['tol_max']:.3e}",
                                    f"{tol[n]['tol_rms']:.3e}") for n in NAMES})
    ok = all([_report(n, got[n], truth[n], tol[n]) for n in NAMES])
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    for n, t in tolerances().items():
        print(f"{n:10s} floor_max={t['floor_max']:.6e} tol_max={t['tol_max']:.6e}"
              f"   floor_rms={t['floor_rms']:.6e} tol_rms={t['tol_rms']:.6e}")
