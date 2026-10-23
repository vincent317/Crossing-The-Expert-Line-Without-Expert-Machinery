"""Reference wrapper: FlashInfer's upstreamed CuTe-DSL GDN prefill kernel.

This module only *calls* the public API `flashinfer.gdn_prefill.chunk_gated_delta_rule`
(install / import / run).  Its implementation is treated as a black box.

The calling convention (whether the k-last state tensor is passed as-is and
whether q/k L2 normalization is done by the kernel) is established empirically
against the fp32 naive ground truth by probe_ref.py, and recorded here.
"""

import torch

from gdn_data import SCALE

# Established empirically (see probe_ref.py output): state is k-last, i.e. exactly
# the (N, HV, D_v, D_k) layout of the workload spec, and the kernel L2-normalizes
# q/k itself.
STATE_AS_IS = True
L2NORM_IN_KERNEL = True


def get_fn():
    """Import the public reference entry point.

    flashinfer's package __init__ pulls in an optional cuDNN-frontend backend we
    neither need nor have; if that import is what fails, it is stubbed out so the
    GDN prefill entry point can be imported and run.
    """
    try:
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
        return chunk_gated_delta_rule
    except Exception as exc:
        if "cudnn" not in repr(exc).lower():
            raise
        import sys
        from unittest.mock import MagicMock
        sys.modules["cudnn"] = MagicMock()
        for mod in [m for m in list(sys.modules) if m.startswith("flashinfer")]:
            del sys.modules[mod]
        print("note: stubbed optional cudnn backend to import flashinfer (%r)"
              % (repr(exc)[:120],))
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
        return chunk_gated_delta_rule


CU_INT32 = True
GATE_DTYPE = "fp32"


def make_ref_call(inp, g, beta, state_as_is=None, l2norm=None, extra=None,
                  cu_int32=None, gate_dtype=None, prenorm=False):
    """Return a zero-arg closure running one reference call on the given inputs."""
    import torch as _t
    fn = get_fn()
    state_as_is = STATE_AS_IS if state_as_is is None else state_as_is
    l2norm = L2NORM_IN_KERNEL if l2norm is None else l2norm
    cu_int32 = CU_INT32 if cu_int32 is None else cu_int32
    gate_dtype = GATE_DTYPE if gate_dtype is None else gate_dtype
    q = inp["q"].contiguous()
    k = inp["k"].contiguous()
    if prenorm:
        import torch.nn.functional as _F
        q = _F.normalize(q.float(), dim=-1).to(q.dtype).contiguous()
        k = _F.normalize(k.float(), dim=-1).to(k.dtype).contiguous()
    v = inp["v"].contiguous()
    gd = {"fp32": _t.float32, "bf16": _t.bfloat16}[gate_dtype]
    g1 = g.to(gd).contiguous()
    b1 = beta.to(gd).contiguous()
    st = inp["state"] if state_as_is else inp["state"].transpose(-1, -2).contiguous()
    st = st.contiguous()
    cu = inp["cu_seqlens"].to(_t.int32) if cu_int32 else inp["cu_seqlens"]
    kw = dict(extra or {})

    def call():
        return fn(q=q, k=k, v=v, g=g1, beta=b1, scale=SCALE, initial_state=st,
                  output_final_state=True, cu_seqlens=cu,
                  use_qk_l2norm_in_kernel=l2norm, **kw)

    return call, state_as_is


def to_k_last(state_out, state_as_is=None):
    """Normalize the reference final state to the workload's k-last layout."""
    state_as_is = STATE_AS_IS if state_as_is is None else state_as_is
    s = state_out.float()
    return s if state_as_is else s.transpose(-1, -2)
