# Prompt and harness ablations (Fable-5.1, B200)

Sessions on single cases from [`../goal_single_shape/`](../goal_single_shape/) that keep
the model fixed (Fable-5.1) and change one thing in the prompt or the harness. Only the
operator code (kernel + launcher) is included.

| arm | change | DSA `t8_topk2048` | FA4 bwd `varlen_t32768_n13_h16` | GDN `t8192_n20` |
|---|---|---|---|---|
| `fable-5.1-ncu-skill` | `/goal` + NCU skill (`mit-han-lab/ncu-report-skill`) installed and asked for | 1.138x | 1.231x | 1.447x |
| `fable-5.1-ncu-prompt` | `/goal` + "please use NCU for tuning analysis", no skill | 1.164x | | |
| `fable-5.1-ncu` | `/goal` + prompt asks for NCU profiling | | 1.200x | |
| `fable-5.1-kernelwiki` | `/goal` + prompt points at KernelWiki | | 1.009x | 1.2035x |
| `fable-5.1-no-accuracy-constraint` | `/goal` with the correctness clause removed | | 1.005x | |
| `fable-5.1-flame-chase` | same prompt under humanize2 `flame_chase` (two chasers taking turns) instead of `/goal` | 1.049x | 0.646x | 1.290x |

`vs reference` ratios as reported in each operator's README; timing protocols differ
between rows (several are the sessions' own measurements), so compare within an operator
README rather than across this table.

| directory | operator |
|---|---|
| [`dsa_decode_fwd/`](dsa_decode_fwd/) | DeepSeek Sparse Attention paged MLA decode forward |
| [`fa4_bwd/`](fa4_bwd/) | FlashAttention-4 backward |
| [`gdn_fwd/`](gdn_fwd/) | Gated DeltaNet prefill forward |
