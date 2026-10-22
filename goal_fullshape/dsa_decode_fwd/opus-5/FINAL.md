# DSA sparse MLA decode on B200 — final report

Definition: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` (16 heads, ckv 512 + kpe 64, page_size 64, topk 2048), all 23 workloads.


## Measurement protocol

* device time = CUPTI kernel span (first kernel start → last kernel end), L2 flushed cold before every call, median over 300 timed iterations after 30 warm-up iterations — the same definition `kda05-ref --bench` reports.
* `flashinfer.testing.bench_gpu_time(enable_cupti=True, cold_l2_cache=True)` is the timer for our kernel; `kda05-ref --op dsa --inputs … --bench` for the reference.
* ref and our kernel are measured on the same B200 (devspace tasks/1321236), interleaved workload by workload in one script run.
* profiling a `run()` loop confirms the measured span contains exactly one kernel launch per call: the only other kernel in the process is the one-time `torch.zeros` that clears the publication slots when a shape is first seen.
* every timed call computes the outputs from the raw inputs; nothing data dependent is hoisted out of the timed region and nothing derived from the inputs is cached between calls (only the shape-keyed scratch buffers and the one-time JIT compile are reused).

## Reference selection

For every workload the reference is the fastest of the two available candidates measured on this card:

* **KDA-0.5** — black-box `kda05-ref --op dsa` (the fastest agent implementation).
* **flashinfer trtllm sparse MLA decode** — `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`, called exactly the way the FIB library baseline `flashinfer_wrapper_5af199` calls it.

The flashinfer path measures 365–767 µs here: its wrapper has to `torch.cat` the 624 MB ckv cache with the 78 MB kpe cache on every call and needs a `seq_lens.max().item()` host sync, and both land inside the CUPTI span. KDA-0.5 is therefore the reference for all 23 workloads.


## Results

| # | workload (uuid) | T | valid keys / token | ref | ref µs | ours µs | ref/ours | err vs fp32-naive | tolerance | ok |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `0c23b10c7b7645719517828c12eaa1d2` | 1 | 2 | KDA-0.5 | 3.232 | 4.928 | **0.656x** | 7.812e-03 | 3.125e-02 | PASS |
| 2 | `9d4a5f21268e484ea05a2f2af91d9fa7` | 2 | 18,11 | KDA-0.5 | 4.272 | 5.184 | **0.824x** | 7.812e-03 | 3.125e-02 | PASS |
| 3 | `b7668cfd194c4b95ab600feb205ebac6` | 2 | 33,52 | KDA-0.5 | 5.664 | 6.048 | **0.937x** | 7.812e-03 | 3.125e-02 | PASS |
| 4 | `0a63b87bb2e54e9db1ca3c4c53a1d521` | 2 | 63,9 | KDA-0.5 | 5.777 | 6.112 | **0.945x** | 1.562e-02 | 3.125e-02 | PASS |
| 5 | `05f6de657db543ae9e4c46796522843a` | 2 | 6,337 | KDA-0.5 | 8.608 | 8.800 | **0.978x** | 1.562e-02 | 3.125e-02 | PASS |
| 6 | `fc85411e250c41879b6d6a1edb80f0a7` | 2 | 17,13 | KDA-0.5 | 4.128 | 5.280 | **0.782x** | 1.562e-02 | 3.125e-02 | PASS |
| 7 | `e6b849f2900446148c01d81152efae23` | 2 | 92,48 | KDA-0.5 | 8.287 | 8.640 | **0.959x** | 1.562e-02 | 3.125e-02 | PASS |
| 8 | `9f3f891bfbe24776adfcd5a579d093dd` | 2 | 288,4 | KDA-0.5 | 8.319 | 8.768 | **0.949x** | 1.562e-02 | 3.125e-02 | PASS |
| 9 | `f77df5ce21634f7ca0586af7e3c13434` | 2 | 18,19 | KDA-0.5 | 4.304 | 5.184 | **0.830x** | 1.562e-02 | 3.125e-02 | PASS |
| 10 | `385742b2717e4f02b918c7349dde23d8` | 8 | 92,48,1044,14,411,30,16,8 | KDA-0.5 | 11.392 | 9.856 | **1.156x** | 1.562e-02 | 6.250e-02 | PASS |
| 11 | `4c46a94ba2364dc7ab476286dee8dce3` | 8 | 18,19,1002,31,11,316,24,2 | KDA-0.5 | 10.176 | 9.888 | **1.029x** | 1.562e-02 | 3.125e-02 | PASS |
| 12 | `3838996164a94d728710f913477feba8` | 7 | 33,52,72,17,18,401,1089 | KDA-0.5 | 11.568 | 9.616 | **1.203x** | 1.562e-02 | 3.125e-02 | PASS |
| 13 | `02d6ae9c64ab42ff93f05c23c53bcb7d` | 8 | 63,9,2048,212,11,25,6,50 | KDA-0.5 | 12.032 | 10.912 | **1.103x** | 1.562e-02 | 3.125e-02 | PASS |
| 14 | `ddfa9e340b264f76abe7418692faa876` | 6 | 6,9,9,14,1639,71 | KDA-0.5 | 11.952 | 10.111 | **1.182x** | 1.562e-02 | 5.469e-02 | PASS |
| 15 | `78b2e11c30004cceb84355722a7c6b0a` | 8 | 18,11,2048,20,25,45,135,326 | KDA-0.5 | 12.576 | 11.551 | **1.089x** | 1.562e-02 | 5.469e-02 | PASS |
| 16 | `68d6817dcfd1433aa8d2ddeefd54b6ea` | 6 | 19,20,32,12,25,3 | KDA-0.5 | 4.959 | 5.568 | **0.891x** | 1.562e-02 | 3.125e-02 | PASS |
| 17 | `564007ac354e4662a62cc4d6352dc494` | 8 | 288,4,1884,21,136,2048,42,335 | KDA-0.5 | 12.608 | 11.744 | **1.074x** | 1.562e-02 | 5.469e-02 | PASS |
| 18 | `ae4219a95f044f45bd10e17ab63c6e8f` | 7 | 19,12,2048,21,26,46,136 | KDA-0.5 | 11.856 | 9.888 | **1.199x** | 1.562e-02 | 3.125e-02 | PASS |
| 19 | `232ed014bafc4835b9881bb308c659b0` | 8 | 35,54,74,19,20,403,1091,1 | KDA-0.5 | 11.552 | 9.441 | **1.224x** | 1.562e-02 | 3.125e-02 | PASS |
| 20 | `7a389715dcc3479e9b0512309f9d1d56` | 8 | 8,11,11,16,1641,73,1,1 | KDA-0.5 | 11.920 | 10.304 | **1.157x** | 1.562e-02 | 3.125e-02 | PASS |
| 21 | `5096e459ce3f4cdf82773ce1a0c73c8a` | 8 | 17,13,1887,16,180,1986,413,1 | KDA-0.5 | 12.448 | 10.848 | **1.147x** | 1.562e-02 | 5.469e-02 | PASS |
| 22 | `d57eb9e19f0642e8af9bd76ad0823303` | 6 | 143,139,2013,142,306,539 | KDA-0.5 | 12.288 | 9.952 | **1.235x** | 1.562e-02 | 3.125e-02 | PASS |
| 23 | `2207f0fdc96347c59d106e4976cfad57` | 7 | 415,131,2011,148,263,169,462 | KDA-0.5 | 12.000 | 10.016 | **1.198x** | 1.562e-02 | 3.125e-02 | PASS |

**Geometric mean of ref/ours over all 23 workloads: 1.0191x** (min 0.656x, max 1.235x).

All 23 workloads pass the tolerance check: **True**.


### lse accuracy

| workload | our lse err | lse tolerance |
|---|---|---|
| `0c23b10c7b7645719517828c12eaa1d2` | 6.676e-06 | 1.000e-03 |
| `9d4a5f21268e484ea05a2f2af91d9fa7` | 8.583e-06 | 1.000e-03 |
| `b7668cfd194c4b95ab600feb205ebac6` | 1.144e-05 | 1.000e-03 |
| `0a63b87bb2e54e9db1ca3c4c53a1d521` | 8.583e-06 | 1.000e-03 |
| `05f6de657db543ae9e4c46796522843a` | 1.144e-05 | 1.000e-03 |
| `fc85411e250c41879b6d6a1edb80f0a7` | 8.583e-06 | 1.000e-03 |
| `e6b849f2900446148c01d81152efae23` | 1.049e-05 | 1.000e-03 |
| `9f3f891bfbe24776adfcd5a579d093dd` | 1.144e-05 | 1.000e-03 |
| `f77df5ce21634f7ca0586af7e3c13434` | 8.583e-06 | 1.000e-03 |
| `385742b2717e4f02b918c7349dde23d8` | 1.526e-05 | 1.000e-03 |
| `4c46a94ba2364dc7ab476286dee8dce3` | 1.335e-05 | 1.000e-03 |
| `3838996164a94d728710f913477feba8` | 1.240e-05 | 1.000e-03 |
| `02d6ae9c64ab42ff93f05c23c53bcb7d` | 1.144e-05 | 1.000e-03 |
| `ddfa9e340b264f76abe7418692faa876` | 1.526e-05 | 1.000e-03 |
| `78b2e11c30004cceb84355722a7c6b0a` | 1.335e-05 | 1.000e-03 |
| `68d6817dcfd1433aa8d2ddeefd54b6ea` | 6.676e-06 | 1.000e-03 |
| `564007ac354e4662a62cc4d6352dc494` | 1.335e-05 | 1.000e-03 |
| `ae4219a95f044f45bd10e17ab63c6e8f` | 1.335e-05 | 1.000e-03 |
| `232ed014bafc4835b9881bb308c659b0` | 1.144e-05 | 1.000e-03 |
| `7a389715dcc3479e9b0512309f9d1d56` | 1.335e-05 | 1.000e-03 |
| `5096e459ce3f4cdf82773ce1a0c73c8a` | 9.537e-06 | 1.000e-03 |
| `d57eb9e19f0642e8af9bd76ad0823303` | 1.335e-05 | 1.000e-03 |
| `2207f0fdc96347c59d106e4976cfad57` | 1.717e-05 | 1.000e-03 |

## How the tolerance is derived

1. An fp32 naive implementation (`tools/naive.py`: gather, fp32 QK, float64 softmax, fp32 PV, round to bf16) is the ground truth.
2. `e_ref` = max |ref − naive| per workload, i.e. the error the reference itself shows. Multiplying it by 7 gives the per-workload tolerance band.
3. Because the op returns **bfloat16**, `e_ref` can be exactly 0 on the tiny workloads, which would make `7·e_ref` degenerate. A floor of 2 bf16 ULP at the output scale is therefore applied: with `M = max|naive|` and `u = 2^(floor(log2 M) - 7)` the exact bf16 spacing at that magnitude, `tol = max(7·e_ref, 2u)`. Two independent implementations that both round the final result to bf16 can legitimately differ by that much.
4. Our kernel is then compared against the fp32 naive implementation with that tolerance. The table above reports the raw error, the tolerance and the verdict for each workload.

## Correctness beyond these 23 workloads

The kernel has a fast path that assumes a row's `-1` padding is a suffix (which is what top-k selection produces, and what all 23 workloads have), and it *verifies* that inside the kernel on every call: a row is suffix-padded exactly when its highest valid slot is `nval-1`, which the prefix-scan barrier folds in as a per-warp max. A row that fails the check drains the speculative copies and falls back to the compacting gather, so the result does not depend on how the `-1` entries are arranged.

`tools/stress.py` checks that against the fp32 naive path on 72 adversarial cases: `num_tokens` in {1,2,3,5,6,7,8,11,16,32,64} (the last four force the grid-stride chunk and combine mapping) crossed with front-packed, randomly scattered, tail-only, duplicated, all-padding and completely full (2048-key) index rows; q/k scaled by 30, 1e-3 and 100/0.01; and `topk` of 512, 1024 and 2047 (the non-vector index path). All 72 pass, including the `-inf` lse for tokens with no valid key.

