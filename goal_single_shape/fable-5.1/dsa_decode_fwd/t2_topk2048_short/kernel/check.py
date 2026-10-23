"""Accuracy check: ref (KDA 0.5 CuTe-DSL) and the new kernel against the fp32 naive truth.

tol = 7 * E_ref, where E_ref = max |ref - naive| on the contest workload.
"""
import sys

import torch

import dsa_ops
import naive_fp32
import run_ref
import workload

TOL_FACTOR = 7.0


def max_err(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    torch.cuda.set_device(0)
    inp = workload.make_inputs()
    print("[check]", workload.describe(inp))
    libs = [lib for lib in dsa_ops.load_all() if "skip_compute=1" not in lib.config]
    for lib in libs:
        print(f"[check] kernel '{lib.name}': {lib.config}")

    out_naive, lse_naive = naive_fp32.dsa_naive_fp32(**inp)
    torch.cuda.synchronize()
    print(f"[check] naive fp32: |out| max={float(out_naive.abs().max()):.4f} "
          f"lse range=[{float(lse_naive.min()):.4f}, {float(lse_naive.max()):.4f}]")

    new = {}
    for lib in libs:
        o, l = lib.fwd(**inp)
        torch.cuda.synchronize()
        new[lib.name] = (o, l)
        print(f"[check] {lib.name}: max|new - naive_fp32| output={max_err(o, out_naive):.6e} "
              f"lse={max_err(l, lse_naive):.6e}")
    out_new, lse_new = new["new"]
    e_new_out, e_new_lse = max_err(out_new, out_naive), max_err(lse_new, lse_naive)

    primary_ok = False
    tol_out, tol_lse = None, None
    try:
        ref_run = run_ref.load_ref_run()
        out_ref, lse_ref = ref_run(inp["q_nope"], inp["q_pe"], inp["ckv_cache"], inp["kpe_cache"],
                                   inp["sparse_indices"], inp["sm_scale"])
        torch.cuda.synchronize()
        print(f"[check] ref: output {tuple(out_ref.shape)} {out_ref.dtype}, lse {tuple(lse_ref.shape)} {lse_ref.dtype}")
        e_ref_out = max_err(out_ref, out_naive)
        e_ref_lse = max_err(lse_ref, lse_naive)
        tol_out = TOL_FACTOR * e_ref_out
        tol_lse = TOL_FACTOR * e_ref_lse
        print(f"[RESULT] E_ref(output) = max|ref - naive_fp32| = {e_ref_out:.6e}  -> tol_out = 7*E_ref = {tol_out:.6e}")
        print(f"[RESULT] E_ref(lse)    = max|ref - naive_fp32| = {e_ref_lse:.6e}  -> tol_lse = 7*E_ref = {tol_lse:.6e}")
        pass_out = e_new_out <= tol_out
        pass_lse = e_new_lse <= tol_lse
        print(f"[RESULT] E_new(output) = max|new - naive_fp32| = {e_new_out:.6e}  <= tol_out {tol_out:.6e} ? {'PASS' if pass_out else 'FAIL'}")
        print(f"[RESULT] E_new(lse)    = max|new - naive_fp32| = {e_new_lse:.6e}  <= tol_lse {tol_lse:.6e} ? {'PASS' if pass_lse else 'FAIL'}")
        print(f"[RESULT] new vs ref directly: max|out_new - out_ref| = {max_err(out_new, out_ref):.6e}, "
              f"max|lse_new - lse_ref| = {max_err(lse_new, lse_ref):.6e}")
        for name, (o, l) in new.items():
            if name != "new":
                print(f"[check] variant {name}: out err {max_err(o, out_naive):.3e} <= tol {tol_out:.3e} "
                      f"{'ok' if max_err(o, out_naive) <= tol_out else 'FAIL'}, lse err {max_err(l, lse_naive):.3e} "
                      f"{'ok' if max_err(l, lse_naive) <= tol_lse else 'FAIL'}")
        primary_ok = pass_out and pass_lse
    except Exception as exc:  # keep the new-kernel numbers even if the reference env is broken
        print("[check] REFERENCE FAILED:\n" + run_ref.format_exc_without_source(exc))
    if tol_out is None:
        tol_out, tol_lse = 4.0 / 256.0 * float(out_naive.abs().max()), 1e-4

    # Robustness of the new kernels on other index patterns (informational, vs naive fp32).
    g = torch.Generator(device="cuda").manual_seed(7)
    extra = {}
    extra["all_2048_valid"] = torch.randint(0, workload.PAGES * workload.PAGE_SIZE, (2, 2048), generator=g,
                                            device="cuda", dtype=torch.int32)
    sc = torch.randint(0, workload.PAGES * workload.PAGE_SIZE, (2, 2048), generator=g, device="cuda",
                       dtype=torch.int32)
    sc[torch.rand((2, 2048), generator=g, device="cuda") > 0.4] = -1
    extra["scattered_40pct"] = sc
    oe = inp["sparse_indices"].clone()
    oe[1] = -1
    extra["one_token_empty"] = oe
    t8 = torch.randint(0, workload.PAGES * workload.PAGE_SIZE, (8, 2048), generator=g, device="cuda",
                       dtype=torch.int32)
    counts = torch.randint(0, 2049, (8,), generator=g, device="cuda")
    ar = torch.arange(2048, device="cuda")[None, :]
    t8[ar >= counts[:, None]] = -1
    extra["T8_random_counts"] = t8
    extra_ok = True
    for name, idx in extra.items():
        Tn = idx.shape[0]
        gq = torch.Generator(device="cuda").manual_seed(11)
        case = dict(inp)
        case["sparse_indices"] = idx.contiguous()
        if Tn != workload.T:
            case["q_nope"] = torch.randn((Tn, 16, 512), generator=gq, device="cuda", dtype=torch.bfloat16)
            case["q_pe"] = torch.randn((Tn, 16, 64), generator=gq, device="cuda", dtype=torch.bfloat16)
        o_n, l_n = naive_fp32.dsa_naive_fp32(**case)
        for lib in libs:
            o_k, l_k = lib.fwd(**case)
            torch.cuda.synchronize()
            eo = max_err(torch.nan_to_num(o_k.float(), nan=1e9), torch.nan_to_num(o_n, nan=1e9))
            el = max_err(torch.nan_to_num(l_k, nan=1e9, neginf=-1e9), torch.nan_to_num(l_n, nan=1e9, neginf=-1e9))
            # scale-aware bound: bf16 output rounding is ~2^-8 relative, lse should match to fp32 noise
            bound_o = max(tol_out, 4.0 / 256.0 * float(o_n.abs().max()))
            good = eo <= bound_o and el <= max(tol_lse, 1e-4)
            extra_ok = extra_ok and good
            print(f"[check-extra] {lib.name:9s} {name:18s} T={Tn} valid/token={[int((idx[t] != -1).sum()) for t in range(Tn)]} "
                  f"max|out-naive|={eo:.3e} (bound {bound_o:.3e}) max|lse-naive|={el:.3e} {'ok' if good else 'MISMATCH'}")

    print(f"[RESULT] accuracy check on workload (new <= 7*E_ref): {'PASS' if primary_ok else 'FAIL'}; "
          f"extra index patterns: {'ok' if extra_ok else 'MISMATCH'}")
    return 0 if (primary_ok and extra_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
