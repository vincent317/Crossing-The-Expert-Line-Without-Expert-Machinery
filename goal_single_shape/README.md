# `/goal` on one specified shape (B200)

Each session here was given one operator case — a single workload shape — and the goal of
beating an expert-written reference kernel on it. The same case is run with several models
(Opus-4.8, Opus-5, Fable-5.1) where available. Only the operator code (kernel + launcher)
is included.

| directory | operator | cases |
|---|---|---|
| [`kda_fwd/`](kda_fwd/) | Kimi Delta Attention forward | `h64_fixed8192`, `h96_uniform`, `h96_mixed` |
| [`gdn_fwd/`](gdn_fwd/) | Gated DeltaNet prefill forward | `t8192_n20`, `t30_n1`, `t5709_n2` |
| [`dsa_decode_fwd/`](dsa_decode_fwd/) | DeepSeek Sparse Attention paged MLA decode forward | `t2_topk2048`, `t8_topk2048`, `t2_topk2048_short` |
| [`fa4_bwd/`](fa4_bwd/) | FlashAttention-4 backward | `mha_b8_n4096_h16`, `gqa_b8_n4096_h16_kv2`, `varlen_t32768_n13_h16` |
| [`gqa_paged_decode/`](gqa_paged_decode/) | GQA paged-KV decode | `b1_kv173`, `b16_kv2841`, `b64_kv58134` |
| [`mla_paged_prefill/`](mla_paged_prefill/) | MLA paged prefill, causal | `b1_q33_kv34`, `b1_q199_kv203`, `b18_q15883_kv15937` |
| [`moe_fp8_fwd/`](moe_fp8_fwd/) | MoE FP8 block-scale, DeepSeek routing | `t1`, `t901`, `t14107` |
| [`mla_paged_decode/`](mla_paged_decode/) | MLA paged decode | `b1_kv408`, `b16_kv9657`, `b64_kv75145` |
| [`fa4_fwd/`](fa4_fwd/) | FlashAttention-4 forward, MHA and GQA, non-causal and causal | 12 cases: B8 × S4096 and B1 × S32768, 16/16, 32/4 and 32/8 heads |

Timing protocols differ between operators (and in places between session groups of one
operator); each README states the one its rows use. Ablations that vary the prompt or the
harness on some of these cases are in [`../ablation/`](../ablation/).
