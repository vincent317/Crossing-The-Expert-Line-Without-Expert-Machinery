"""Black-box driver for the reference GDN prefill kernel.

Reference = the CuTe-DSL GDN prefill implementation upstreamed into FlashInfer,
reachable as ``flashinfer.gdn_prefill.chunk_gated_delta_rule``.  Only import,
introspection of the documented public signature, and invocation are used;
the reference source is never opened or copied.

Public contract (from the function's own docstring):
    q    [total_seq_len, num_q_heads, head_size]
    k    [total_seq_len, num_k_heads, head_size]
    v    [total_seq_len, num_v_heads, head_size]
    g    [total_seq_len, num_sab_heads] float32  -- alpha, the multiplicative
         forget gate in (0, 1] (documented default: all ones), NOT its log
    beta [total_seq_len, num_sab_heads] float32
    initial_state [num_seqs, num_sab_heads, head_size, head_size] with the
        inner block contiguous as [H, V, K]  ->  k-last, same as our layout
    scale defaults to 1/sqrt(head_size); GQA and GVA are supported natively.
"""

import importlib
import inspect

import torch
import torch.nn.functional as F


def load_ref():
    mod = importlib.import_module("flashinfer.gdn_prefill")
    return getattr(mod, "chunk_gated_delta_rule")


class Ref:
    def __init__(self, HQK, HV, D):
        self.HQK, self.HV, self.D = HQK, HV, D
        self.fn = load_ref()
        self.names = set(inspect.signature(self.fn).parameters)
        self.variant = None
        self.out = None
        self.ost = None

    def gates(self, a, b, A_log, dt_bias, alpha_space=True):
        """The reference takes the forget gate as alpha (documented as
        "Forget gate (alpha) ... Defaults to all ones when None", and the
        SM90 entry point names the same argument `alpha`), i.e. the
        multiplicative decay in (0, 1] rather than its logarithm."""
        gl = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
        g = gl.exp() if alpha_space else gl
        return g.contiguous(), torch.sigmoid(b.float()).contiguous()

    def _alloc(self, q, v, state):
        if self.out is None:
            T, HV, D = v.shape
            self.out = torch.empty(T, HV, D, dtype=v.dtype, device=v.device)
            self.ost = torch.empty_like(state)

    def _kwargs(self, variant, q, k, v, g, beta, state, cu):
        kv_last, l2, scale, i32 = variant[:4]
        self._alloc(q, v, state)
        st = state if kv_last else state.transpose(-1, -2).contiguous()
        kw = dict(q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
                  g=g, beta=beta, initial_state=st,
                  cu_seqlens=cu.to(torch.int32) if i32 else cu)
        if "scale" in self.names:
            kw["scale"] = scale
        if "output_final_state" in self.names:
            kw["output_final_state"] = True
        if "use_qk_l2norm_in_kernel" in self.names:
            kw["use_qk_l2norm_in_kernel"] = l2
        if "output" in self.names:
            kw["output"] = self.out
        if "output_state" in self.names:
            kw["output_state"] = self.ost
        return {n: val for n, val in kw.items() if n in self.names}

    def _unpack(self, out, kv_last):
        if isinstance(out, (tuple, list)):
            o = out[0]
            sf = out[1] if len(out) > 1 else self.ost
        else:
            o = out if out is not None else self.out
            sf = self.ost
        if o is not None and o.dim() == 4:
            o = o[0]
        if sf is not None and not kv_last:
            sf = sf.transpose(-1, -2)
        return o, sf

    def calibrate(self, q, k, v, a, b, A_log, dt_bias, state, cu, o_nv, s_nv,
                  verbose=True):
        best = None
        for alpha_space in (True, False):
          g, beta = self.gates(a, b, A_log, dt_bias, alpha_space)
          for i32 in (True, False):
            for kv_last in (True, False):
                for l2 in (False,):
                    var = (kv_last, l2, self.D ** -0.5, i32, alpha_space)
                    tag = f"alpha={alpha_space} i32={i32} kvlast={kv_last} l2={l2}"
                    try:
                        out = self.fn(**self._kwargs(var, q, k, v, g, beta, state, cu))
                        torch.cuda.synchronize()
                        o, sf = self._unpack(out, kv_last)
                        eo = (o.float() - o_nv).abs().max().item()
                        es = ((sf.float() - s_nv).abs().max().item()
                              if sf is not None else float("nan"))
                    except Exception as e:
                        if verbose:
                            print(f"  ref {tag}: {type(e).__name__}: {str(e)[:200]}")
                        continue
                    if verbose:
                        print(f"  ref {tag}: err_o={eo:.8f} err_s={es:.8f}")
                    if best is None or eo < best[0]:
                        best = (eo, es, var)
          if best is not None and best[0] < 0.01:
              break
        assert best is not None, "reference did not run in any calling convention"
        self.variant = best[2]
        return best[0], best[1]

    def make_callable(self, q, k, v, a, b, A_log, dt_bias, state, cu,
                      include_gate=False):
        var = self.variant
        asp = var[4]
        if include_gate:
            def fn():
                g, beta = self.gates(a, b, A_log, dt_bias, asp)
                return self.fn(**self._kwargs(var, q, k, v, g, beta, state, cu))
        else:
            g, beta = self.gates(a, b, A_log, dt_bias, asp)
            kw = self._kwargs(var, q, k, v, g, beta, state, cu)

            def fn():
                return self.fn(**kw)
        return fn
