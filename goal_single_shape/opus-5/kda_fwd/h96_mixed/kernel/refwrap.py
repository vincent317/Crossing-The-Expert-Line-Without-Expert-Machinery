"""Black-box wrapper around the CAKE recurrent_kda reference (flashinfer).

Only the public API is used: install / import / call.  No reference source is read.

The frozen SM100-family FlashKDA (CAKE) prefill backend -- kernel
``kernel_flashkda_bf16_fused_m128``, i.e. the M128 variant -- is only dispatched
for this shape when:
  * ``lower_bound`` is set (safe gate ``lower_bound * sigmoid(exp(A_log)*(g+dt_bias))``)
  * ``beta_is_logit=True`` (beta given as pre-sigmoid logits)
  * ``initial_state`` is bfloat16 with layout ``[N, H, V, K]``
Passing ``seq_order`` is rejected by every other path, so it doubles as the
eligibility oracle used to discover this configuration.
"""

import torch
import flashinfer

LOWER_BOUND = -5.0


class CakeRef:
    def __init__(self, q, k, v, g, beta_logit, A_log, dt_bias, cu_seqlens,
                 lower_bound=LOWER_BOUND):
        self.t = (q, k, v, g, beta_logit, A_log, dt_bias, cu_seqlens)
        self.lower_bound = lower_bound
        self.seq_order = torch.argsort(cu_seqlens[1:] - cu_seqlens[:-1],
                                       descending=True).to(torch.int32).contiguous()

    @staticmethod
    def prep_state(state_kv):
        """(N,H,K,V) fp32 -> (N,H,V,K) bf16 as required by the frozen backend."""
        return state_kv.transpose(-1, -2).to(torch.bfloat16).contiguous()

    @staticmethod
    def post_state(s):
        return s.float().transpose(-1, -2)

    def __call__(self, st, out=None, use_seq_order=True):
        q, k, v, g, beta_logit, A_log, dt_bias, cu = self.t
        return flashinfer.recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta_logit, A_log=A_log, dt_bias=dt_bias,
            initial_state=st, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            lower_bound=self.lower_bound, beta_is_logit=True,
            cu_seqlens=cu, output=out,
            seq_order=self.seq_order if use_seq_order else None)
