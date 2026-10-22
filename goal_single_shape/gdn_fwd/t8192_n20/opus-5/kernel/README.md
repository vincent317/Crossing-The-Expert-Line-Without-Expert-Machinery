# GDN FWD (long sequences) — from-scratch Triton kernel for B200

Target operator: Gated DeltaNet forward (prefill), `T=8192` tokens packed into
`N=20` variable-length sequences, `head_dim=128`, 4 q/k heads, 8 v heads,
bf16 q/k/v/a/b, fp32 k-last state.

## Files

| file | role |
| --- | --- |
| `gdn_data.py` | deterministic workload generator (fixed shapes and seq lengths) |
| `gdn_naive.py` | fp32 scalar recurrence — the numerical anchor |
| `gdn_triton.py` | the kernel (two Triton kernels + host wrapper) |
| `gdn_ref.py` | black-box driver for the reference (`flashinfer.gdn_prefill.chunk_gated_delta_rule`, the CuTe-DSL GDN prefill kernel upstreamed into FlashInfer) |
| `check.py` | accuracy: `E_ref` from the reference, `tol = 7*E_ref`, ours vs naive |
| `bench.py` / `final_run.py` | device-time benchmark (CUDA events) |
| `bench_util.py` | timing helpers |
| `SUMMARY.md` | measured results, tolerances and bottleneck analysis |

## How to run (B200, CUDA 13 image required)

The reference's Blackwell path refuses to run on CUDA 12, so use a CUDA 13
image (`cybertron/pytorch:26.07-py3`) and install
`apache-tvm-ffi`, `nvidia-cutlass-dsl[cu13]==4.7.1`, `nvidia-cudnn-frontend`,
`flashinfer-python`, `triton==3.5.0` (the shipped Triton 3.2 generates
usable but ~3x slower Blackwell code).  Then:

    python -u check.py          # E_ref, tol = 7*E_ref, ours vs fp32 naive
    python -u bench.py          # device time, reference vs ours
    python -u final_run.py '[dict(C=64,split=4,nsq=2,st_scan=3)]'   # both

## Algorithm (derived from the recurrence, not copied)

Per (sequence, v-head) with state `S` of shape `[d_v, d_k]`:

```
g_t  = -exp(A_log) * softplus(a_t + dt_bias)      (<= 0)
b_t  = sigmoid(b_t)
S   <- S * exp(g_t)
u_t  = b_t * (v_t - S k_t)
S   <- S + u_t k_t^T
o_t  = S (q_t * d_k^-0.5)
```

Split the sequence into chunks of `C` tokens.  With `G` the inclusive cumsum of
`g` inside the chunk and `S0` the state at chunk entry, unrolling gives a
triangular linear system for the per-token update vectors `U`:

```
M[j,l] = beta_j * exp(G_j - G_l) * (k_j . k_l),  l < j   (strictly lower)
(I + M) U = diag(beta) V - diag(beta * exp(G)) K S0^T
```

`I + M` is unit lower triangular, so `T = (I+M)^{-1}` exists and

```
W    = T diag(beta*exp(G)) K          (state independent)
Uv   = T diag(beta) V                 (state independent)
U    = Uv - W S0^T
S1   = exp(G_C) S0 + (diag(exp(G_C - G)) U)^T K
O    = Wattn Uv + (diag(exp(G)) Q*scale - Wattn W) S0^T
     = Obase + Qeff S0^T
Wattn[i,j] = exp(G_i - G_j) (q_i . k_j) * scale,  j <= i
```

`T` is obtained exactly (not by truncation) from the nilpotency of `M`:
`(I+M)^{-1} = (I-M)(I+M^2)(I+M^4)...(I+M^{C/2})`, i.e. `2*log2(C)` matmuls.

### Kernel split

1. `_wy` — one CTA per (chunk, v-head), fully parallel.  Builds `M`, inverts it,
   emits `W`, `Uv`, `Obase`, `Qeff`, `exp(G_C)` and `G_C - G`.
2. `_scan` — the only sequential part.  Per chunk it needs three matmuls:
   `O += Qeff S0^T`, `U = Uv - W S0^T`, `S = exp(G_C) S + (diag(e)U)^T K`.

### Why the sequential pass is not the wall

Each **row** of `S` along `d_v` evolves independently of the others (column `r`
of `U` depends only on row `r` of `S`).  So the `d_v` axis is sliced across
`SPLIT` CTAs, cutting the critical path of the longest sequence (3371 tokens =
53 chunks at C=64) by that factor with no redundant arithmetic.

The state is also kept **transposed** (`[d_k, d_v_slice]`) inside the loop so
that it is always the right-hand dot operand and no register tile has to be
transposed through shared memory per iteration.
