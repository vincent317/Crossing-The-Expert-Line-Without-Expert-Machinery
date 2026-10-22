# gdn_prefill_qk4_v8_d128_k_last -- final results

Device: NVIDIA B200 (sm_100a). Timing: CUPTI kernel span, cold L2 before every
call, median over 11 iterations, median over 3 trials -- the same definition the
reference harness (`kda05-ref --bench`) prints. For every row the reference was
measured immediately before our kernel, in the same pass, on the same card. Both
sides move by a couple of percent between passes (the reference a little more than
ours), so the per-row ratios carry that much noise; the table is one pass, not a
best-of.

Reference per workload = the fastest runnable candidate on this card, which is
KDA-0.5 everywhere: the flashinfer library path is not usable in this
environment -- the installed flashinfer exposes no chunked gated-delta-rule
entry point, and the FIB baseline solution that wraps it (`flashinfer_wrapper_123ca6`)
fails with RUNTIME_ERROR on every workload in the official evaluation run below.

Tolerance: an independent fp32 naive implementation (literal per-token recurrence,
float32 throughout) is the ground truth. For each workload the tolerance is 7x the
reference's own max-abs deviation from that naive result, computed separately for
`output` and `new_state`; the solution must stay inside it.

| # | total_seq_len | num_seqs | plan | ref | ref us | mine us | ref/mine | err out | tol out | err state | tol state | pass |
|---:|---:|---:|:--|:--|---:|---:|---:|---:|---:|---:|---:|:--|
| 0 | 6 | 1 | fused/c16/v8/w8 | KDA-0.5 | 3.65 | 5.44 | 0.671 | 4.68e-05 | 3.14e-04 | 1.18e-03 | 8.61e-03 | PASS |
| 1 | 12 | 1 | fused/c16/v8/w8 | KDA-0.5 | 3.90 | 5.73 | 0.682 | 8.23e-05 | 6.01e-04 | 3.25e-04 | 2.80e-03 | PASS |
| 2 | 13 | 1 | fused/c16/v8/w8 | KDA-0.5 | 3.97 | 5.70 | 0.697 | 6.39e-05 | 2.74e-04 | 1.42e-03 | 8.58e-03 | PASS |
| 3 | 14 | 1 | fused/c16/v8/w8 | KDA-0.5 | 4.00 | 5.70 | 0.702 | 4.59e-05 | 6.01e-04 | 3.00e-04 | 7.99e-03 | PASS |
| 4 | 16 | 1 | fused/c16/v8/w8 | KDA-0.5 | 4.19 | 5.83 | 0.720 | 1.23e-04 | 1.14e-03 | 3.90e-04 | 2.43e-03 | PASS |
| 5 | 16 | 1 | fused/c16/v8/w8 | KDA-0.5 | 4.16 | 5.66 | 0.734 | 5.31e-05 | 3.38e-04 | 6.34e-03 | 1.84e-02 | PASS |
| 6 | 16 | 1 | fused/c16/v8/w8 | KDA-0.5 | 4.26 | 5.73 | 0.743 | 8.50e-05 | 8.62e-04 | 1.05e-03 | 4.83e-03 | PASS |
| 7 | 18 | 1 | fused/c32/v8/w8 | KDA-0.5 | 4.64 | 6.11 | 0.759 | 7.44e-05 | 4.98e-04 | 1.53e-03 | 9.10e-03 | PASS |
| 8 | 23 | 1 | fused/c32/v8/w8 | KDA-0.5 | 4.80 | 6.11 | 0.785 | 6.38e-05 | 8.62e-04 | 1.20e-03 | 6.00e-03 | PASS |
| 9 | 24 | 1 | fused/c32/v8/w8 | KDA-0.5 | 4.77 | 6.11 | 0.780 | 7.23e-05 | 8.62e-04 | 9.49e-04 | 5.58e-03 | PASS |
| 10 | 28 | 2 | fused/c32/v8/w8 | KDA-0.5 | 4.96 | 5.98 | 0.829 | 1.05e-04 | 4.91e-04 | 1.28e-03 | 6.50e-03 | PASS |
| 11 | 30 | 1 | fused/c32/v8/w8 | KDA-0.5 | 5.31 | 6.18 | 0.860 | 1.08e-04 | 8.62e-04 | 7.52e-04 | 4.25e-03 | PASS |
| 12 | 30 | 1 | fused/c32/v8/w8 | KDA-0.5 | 5.31 | 6.08 | 0.874 | 1.57e-04 | 1.40e-03 | 1.62e-03 | 1.15e-02 | PASS |
| 13 | 30 | 1 | fused/c32/v8/w8 | KDA-0.5 | 5.38 | 6.14 | 0.875 | 5.31e-05 | 3.54e-04 | 1.00e-02 | 2.14e-02 | PASS |
| 14 | 31 | 1 | fused/c32/v8/w8 | KDA-0.5 | 5.41 | 6.11 | 0.885 | 9.67e-05 | 6.77e-04 | 2.04e-04 | 1.71e-03 | PASS |
| 15 | 32 | 2 | fused/c32/v8/w8 | KDA-0.5 | 5.95 | 6.17 | 0.964 | 6.56e-05 | 8.62e-04 | 1.25e-03 | 5.06e-03 | PASS |
| 16 | 35 | 1 | fused/c32/v8/w8 | KDA-0.5 | 5.63 | 8.80 | 0.640 | 1.40e-04 | 9.83e-04 | 3.85e-04 | 1.94e-03 | PASS |
| 17 | 35 | 1 | fused/c32/v8/w8 | KDA-0.5 | 5.66 | 8.80 | 0.644 | 4.84e-05 | 3.42e-04 | 6.89e-03 | 3.24e-02 | PASS |
| 18 | 35 | 2 | fused/c32/v8/w8 | KDA-0.5 | 5.70 | 6.18 | 0.922 | 1.17e-04 | 9.74e-04 | 7.38e-04 | 3.19e-03 | PASS |
| 19 | 35 | 2 | fused/c32/v8/w8 | KDA-0.5 | 5.73 | 6.21 | 0.923 | 7.44e-05 | 6.34e-04 | 8.42e-04 | 5.79e-03 | PASS |
| 20 | 40 | 2 | fused/c32/v8/w8 | KDA-0.5 | 5.79 | 6.43 | 0.900 | 1.19e-04 | 8.32e-04 | 7.49e-04 | 3.39e-03 | PASS |
| 21 | 42 | 2 | fused/c32/v8/w8 | KDA-0.5 | 6.21 | 6.37 | 0.975 | 1.03e-04 | 6.01e-04 | 7.84e-04 | 4.54e-03 | PASS |
| 22 | 46 | 2 | fused/c32/v8/w8 | KDA-0.5 | 6.14 | 6.43 | 0.955 | 1.04e-04 | 8.90e-04 | 9.25e-04 | 5.65e-03 | PASS |
| 23 | 48 | 1 | fused/c32/v8/w8 | KDA-0.5 | 5.70 | 9.22 | 0.618 | 7.83e-05 | 4.48e-04 | 1.23e-03 | 1.37e-02 | PASS |
| 24 | 49 | 1 | fused/c32/v8/w8 | KDA-0.5 | 5.70 | 9.60 | 0.593 | 1.18e-04 | 1.07e-03 | 3.14e-04 | 6.15e-03 | PASS |
| 25 | 49 | 1 | fused/c32/v8/w8 | KDA-0.5 | 5.70 | 9.57 | 0.595 | 1.34e-04 | 1.17e-03 | 2.54e-03 | 1.52e-02 | PASS |
| 26 | 49 | 1 | fused/c32/v8/w8 | KDA-0.5 | 5.66 | 9.63 | 0.588 | 6.74e-05 | 5.43e-04 | 6.53e-03 | 2.80e-02 | PASS |
| 27 | 61 | 1 | fused/c32/v8/w8 | KDA-0.5 | 5.70 | 9.76 | 0.584 | 9.94e-05 | 4.47e-04 | 1.89e-03 | 8.61e-03 | PASS |
| 28 | 67 | 1 | fused/c32/v8/w8 | KDA-0.5 | 7.42 | 12.32 | 0.603 | 1.01e-04 | 6.79e-04 | 8.77e-04 | 7.41e-03 | PASS |
| 29 | 68 | 1 | fused/c32/v8/w8 | KDA-0.5 | 7.49 | 12.35 | 0.606 | 9.81e-05 | 8.77e-04 | 5.59e-04 | 5.02e-03 | PASS |
| 30 | 76 | 2 | fused/c32/v8/w8 | KDA-0.5 | 6.69 | 9.76 | 0.685 | 1.44e-04 | 1.21e-03 | 2.68e-03 | 1.88e-02 | PASS |
| 31 | 76 | 2 | fused/c32/v8/w8 | KDA-0.5 | 6.66 | 9.98 | 0.667 | 5.55e-05 | 3.47e-04 | 6.88e-03 | 4.59e-02 | PASS |
| 32 | 76 | 2 | fused/c32/v8/w8 | KDA-0.5 | 6.62 | 9.92 | 0.668 | 8.62e-05 | 8.62e-04 | 2.21e-03 | 6.21e-03 | PASS |
| 33 | 79 | 1 | fused/c32/v8/w8 | KDA-0.5 | 7.46 | 12.96 | 0.575 | 7.95e-05 | 8.62e-04 | 1.05e-03 | 4.31e-03 | PASS |
| 34 | 81 | 2 | fused/c32/v8/w8 | KDA-0.5 | 6.69 | 9.76 | 0.685 | 9.18e-05 | 5.09e-04 | 3.43e-03 | 5.66e-03 | PASS |
| 35 | 82 | 3 | fused/c32/v8/w8 | KDA-0.5 | 6.37 | 10.94 | 0.582 | 9.53e-05 | 8.62e-04 | 1.51e-03 | 6.11e-03 | PASS |
| 36 | 85 | 2 | fused/c32/v8/w8 | KDA-0.5 | 7.94 | 13.09 | 0.606 | 8.03e-05 | 5.66e-04 | 2.15e-03 | 6.21e-03 | PASS |
| 37 | 91 | 2 | fused/c32/v8/w8 | KDA-0.5 | 7.97 | 12.96 | 0.615 | 1.57e-04 | 8.16e-04 | 1.27e-03 | 8.97e-03 | PASS |
| 38 | 91 | 2 | fused/c32/v8/w8 | KDA-0.5 | 8.00 | 13.22 | 0.605 | 6.30e-05 | 4.73e-04 | 6.31e-03 | 3.04e-02 | PASS |
| 39 | 92 | 1 | coop/c32/v8/w8 | KDA-0.5 | 7.46 | 13.47 | 0.553 | 1.07e-04 | 7.03e-04 | 5.50e-04 | 4.50e-03 | PASS |
| 40 | 97 | 1 | coop/c32/v8/w8 | KDA-0.5 | 7.46 | 14.05 | 0.531 | 1.69e-04 | 1.17e-03 | 4.76e-04 | 1.91e-03 | PASS |
| 41 | 97 | 1 | coop/c32/v8/w8 | KDA-0.5 | 7.46 | 14.14 | 0.527 | 7.36e-05 | 4.84e-04 | 3.59e-03 | 1.72e-02 | PASS |
| 42 | 114 | 2 | fused/c32/v8/w8 | KDA-0.5 | 8.03 | 12.29 | 0.654 | 1.54e-04 | 1.33e-03 | 1.50e-03 | 1.26e-02 | PASS |
| 43 | 114 | 2 | fused/c32/v8/w8 | KDA-0.5 | 8.00 | 12.51 | 0.639 | 6.86e-05 | 4.54e-04 | 5.47e-03 | 2.43e-02 | PASS |
| 44 | 116 | 1 | coop/c32/v8/w8 | KDA-0.5 | 7.52 | 14.08 | 0.534 | 1.47e-04 | 1.19e-03 | 5.41e-04 | 3.70e-03 | PASS |
| 45 | 118 | 2 | fused/c32/v8/w8 | KDA-0.5 | 8.06 | 12.83 | 0.628 | 1.26e-04 | 8.80e-04 | 1.21e-03 | 6.75e-03 | PASS |
| 46 | 132 | 2 | fused/c32/v8/w8 | KDA-0.5 | 8.06 | 12.83 | 0.628 | 7.79e-05 | 6.96e-04 | 6.78e-04 | 3.20e-03 | PASS |
| 47 | 134 | 1 | coop/c32/v8/w8 | KDA-0.5 | 9.54 | 15.01 | 0.635 | 1.27e-04 | 9.34e-04 | 1.11e-03 | 1.20e-03 | PASS |
| 48 | 134 | 1 | coop/c32/v8/w8 | KDA-0.5 | 9.50 | 15.01 | 0.633 | 1.00e-04 | 6.02e-04 | 1.83e-03 | 5.58e-03 | PASS |
| 49 | 139 | 3 | fused/c32/v8/w8 | KDA-0.5 | 8.13 | 13.25 | 0.614 | 1.94e-04 | 1.21e-03 | 1.51e-03 | 2.58e-02 | PASS |
| 50 | 139 | 3 | fused/c32/v8/w8 | KDA-0.5 | 7.97 | 13.47 | 0.591 | 8.30e-05 | 5.41e-04 | 6.38e-03 | 4.64e-02 | PASS |
| 51 | 185 | 2 | coop/c32/v8/w8 | KDA-0.5 | 9.89 | 15.39 | 0.642 | 1.04e-04 | 8.05e-04 | 5.70e-04 | 4.15e-03 | PASS |
| 52 | 202 | 3 | coop/c32/v8/w8 | KDA-0.5 | 8.10 | 16.96 | 0.477 | 1.02e-04 | 1.08e-03 | 3.89e-04 | 3.98e-03 | PASS |
| 53 | 239 | 3 | coop/c32/v8/w8 | KDA-0.5 | 8.22 | 18.05 | 0.456 | 1.39e-04 | 8.51e-04 | 3.02e-03 | 9.29e-03 | PASS |
| 54 | 248 | 2 | coop/c32/v8/w8 | KDA-0.5 | 11.90 | 17.86 | 0.667 | 1.29e-04 | 1.19e-03 | 8.07e-04 | 4.97e-03 | PASS |
| 55 | 255 | 3 | coop/c32/v8/w8 | KDA-0.5 | 11.97 | 19.27 | 0.621 | 1.39e-04 | 6.38e-04 | 3.36e-03 | 1.55e-02 | PASS |
| 56 | 294 | 3 | coop/c32/v8/w8 | KDA-0.5 | 11.97 | 20.00 | 0.598 | 1.05e-04 | 8.62e-04 | 1.79e-03 | 6.09e-03 | PASS |
| 57 | 341 | 2 | coop/c32/v8/w8 | KDA-0.5 | 13.79 | 19.90 | 0.693 | 1.45e-04 | 9.04e-04 | 3.87e-04 | 5.73e-03 | PASS |
| 58 | 401 | 4 | coop/c32/v8/w8 | KDA-0.5 | 14.08 | 21.79 | 0.646 | 1.37e-04 | 9.57e-04 | 9.71e-04 | 8.00e-03 | PASS |
| 59 | 461 | 2 | coop/c32/v8/w8 | KDA-0.5 | 17.66 | 23.49 | 0.752 | 1.91e-04 | 1.60e-03 | 9.32e-04 | 4.36e-03 | PASS |
| 60 | 525 | 1 | coop/c32/v8/w8 | KDA-0.5 | 20.86 | 26.11 | 0.799 | 1.25e-04 | 8.62e-04 | 1.15e-03 | 3.65e-03 | PASS |
| 61 | 574 | 2 | coop/c32/v8/w8 | KDA-0.5 | 19.46 | 31.65 | 0.615 | 1.32e-04 | 1.31e-03 | 6.60e-04 | 4.09e-03 | PASS |
| 62 | 661 | 3 | coop/c32/v8/w8 | KDA-0.5 | 17.66 | 29.21 | 0.605 | 1.47e-04 | 8.27e-04 | 2.07e-03 | 1.31e-02 | PASS |
| 63 | 832 | 4 | coop/c32/v8/w8 | KDA-0.5 | 25.34 | 33.76 | 0.751 | 1.28e-04 | 9.72e-04 | 6.95e-04 | 2.53e-03 | PASS |
| 64 | 959 | 4 | coop/c32/v8/w8 | KDA-0.5 | 31.04 | 39.97 | 0.777 | 1.75e-04 | 1.41e-03 | 1.33e-03 | 1.69e-02 | PASS |
| 65 | 959 | 4 | coop/c32/v8/w8 | KDA-0.5 | 31.17 | 39.68 | 0.785 | 1.35e-04 | 7.84e-04 | 6.32e-03 | 4.65e-02 | PASS |
| 66 | 973 | 3 | coop/c32/v8/w8 | KDA-0.5 | 32.48 | 40.35 | 0.805 | 1.71e-04 | 1.21e-03 | 2.51e-03 | 1.68e-02 | PASS |
| 67 | 973 | 3 | coop/c32/v8/w8 | KDA-0.5 | 32.45 | 40.26 | 0.806 | 1.16e-04 | 5.26e-04 | 8.73e-03 | 4.26e-02 | PASS |
| 68 | 983 | 2 | coop/c32/v8/w8 | KDA-0.5 | 32.64 | 40.35 | 0.809 | 1.83e-04 | 1.27e-03 | 2.01e-03 | 1.57e-02 | PASS |
| 69 | 983 | 2 | coop/c32/v8/w8 | KDA-0.5 | 32.61 | 40.32 | 0.809 | 1.38e-04 | 6.66e-04 | 9.07e-03 | 3.69e-02 | PASS |
| 70 | 1377 | 1 | split/c32/v8/w8 | KDA-0.5 | 45.47 | 54.75 | 0.831 | 1.79e-04 | 1.02e-03 | 9.03e-04 | 4.52e-03 | PASS |
| 71 | 1592 | 3 | split/c32/v8/w8 | KDA-0.5 | 38.43 | 49.54 | 0.776 | 1.48e-04 | 1.04e-03 | 1.17e-03 | 5.50e-03 | PASS |
| 72 | 1796 | 3 | split/c32/v8/w8 | KDA-0.5 | 40.38 | 58.82 | 0.687 | 1.14e-04 | 8.62e-04 | 1.45e-03 | 5.08e-03 | PASS |
| 73 | 1800 | 3 | split/c32/v8/w8 | KDA-0.5 | 49.79 | 79.17 | 0.629 | 1.79e-04 | 1.27e-03 | 3.17e-03 | 8.29e-03 | PASS |
| 74 | 1800 | 3 | split/c32/v8/w8 | KDA-0.5 | 49.54 | 79.23 | 0.625 | 1.48e-04 | 1.04e-03 | 8.90e-03 | 2.47e-02 | PASS |
| 75 | 2040 | 2 | split/c32/v8/w8 | KDA-0.5 | 64.67 | 74.62 | 0.867 | 1.68e-04 | 1.32e-03 | 6.85e-04 | 5.05e-03 | PASS |
| 76 | 2107 | 1 | split/c32/v8/w8 | KDA-0.5 | 62.05 | 75.49 | 0.822 | 1.76e-04 | 1.09e-03 | 1.42e-03 | 1.71e-02 | PASS |
| 77 | 2284 | 3 | split/c32/v8/w8 | KDA-0.5 | 70.75 | 78.94 | 0.896 | 1.70e-04 | 8.09e-04 | 2.17e-03 | 1.82e-02 | PASS |
| 78 | 2857 | 3 | split/c32/v8/w8 | KDA-0.5 | 71.39 | 69.34 | 1.030 | 2.07e-04 | 9.84e-04 | 1.39e-03 | 9.21e-03 | PASS |
| 79 | 3028 | 5 | split/c32/v8/w8 | KDA-0.5 | 91.68 | 103.90 | 0.882 | 1.74e-04 | 1.34e-03 | 7.75e-04 | 3.52e-03 | PASS |
| 80 | 3271 | 2 | split/c32/v8/w8 | KDA-0.5 | 75.07 | 99.26 | 0.756 | 1.30e-04 | 9.15e-04 | 1.07e-03 | 5.18e-03 | PASS |
| 81 | 3999 | 13 | split/c32/v8/w8 | KDA-0.5 | 92.54 | 122.37 | 0.756 | 2.13e-04 | 1.49e-03 | 2.78e-03 | 1.77e-02 | PASS |
| 82 | 4124 | 15 | split/c32/v2/w8 | KDA-0.5 | 40.22 | 79.49 | 0.506 | 1.82e-04 | 1.51e-03 | 1.10e-03 | 6.96e-03 | PASS |
| 83 | 5709 | 2 | split/c32/v8/w8 | KDA-0.5 | 109.02 | 200.90 | 0.543 | 1.70e-04 | 1.19e-03 | 9.97e-04 | 5.22e-03 | PASS |
| 84 | 8192 | 20 | split/c32/v2/w8 | KDA-0.5 | 126.91 | 213.95 | 0.593 | 2.37e-04 | 1.66e-03 | 1.63e-03 | 7.06e-03 | PASS |
| 85 | 8192 | 25 | split/c32/v2/w8 | KDA-0.5 | 112.03 | 195.07 | 0.574 | 2.16e-04 | 9.53e-04 | 4.31e-03 | 2.48e-02 | PASS |
| 86 | 8192 | 32 | split/c32/v2/w8 | KDA-0.5 | 88.00 | 160.03 | 0.550 | 2.03e-04 | 1.01e-03 | 2.14e-03 | 7.54e-03 | PASS |
| 87 | 8192 | 32 | split/c32/v2/w8 | KDA-0.5 | 123.52 | 206.30 | 0.599 | 2.48e-04 | 1.32e-03 | 7.59e-04 | 5.51e-03 | PASS |
| 88 | 8192 | 34 | split/c32/v2/w8 | KDA-0.5 | 123.20 | 220.03 | 0.560 | 2.05e-04 | 1.53e-03 | 2.61e-03 | 2.80e-02 | PASS |
| 89 | 8192 | 34 | split/c32/v2/w8 | KDA-0.5 | 123.20 | 219.90 | 0.560 | 1.97e-04 | 1.48e-03 | 1.13e-02 | 6.12e-02 | PASS |
| 90 | 8192 | 35 | split/c32/v2/w8 | KDA-0.5 | 89.25 | 172.48 | 0.517 | 1.84e-04 | 1.09e-03 | 2.09e-03 | 1.62e-02 | PASS |
| 91 | 8192 | 37 | split/c32/v2/w8 | KDA-0.5 | 116.51 | 186.34 | 0.625 | 2.19e-04 | 1.16e-03 | 2.71e-03 | 2.48e-02 | PASS |
| 92 | 8192 | 38 | split/c32/v2/w8 | KDA-0.5 | 86.69 | 185.09 | 0.468 | 2.31e-04 | 1.80e-03 | 4.08e-03 | 2.24e-02 | PASS |
| 93 | 8192 | 38 | split/c32/v2/w8 | KDA-0.5 | 86.98 | 186.05 | 0.467 | 1.61e-04 | 8.20e-04 | 1.33e-02 | 5.88e-02 | PASS |
| 94 | 8192 | 39 | split/c32/v2/w8 | KDA-0.5 | 94.34 | 169.47 | 0.557 | 1.60e-04 | 1.22e-03 | 8.05e-04 | 5.79e-03 | PASS |
| 95 | 8192 | 43 | split/c32/v2/w8 | KDA-0.5 | 111.33 | 202.69 | 0.549 | 2.44e-04 | 1.62e-03 | 3.58e-03 | 2.45e-02 | PASS |
| 96 | 8192 | 43 | split/c32/v2/w8 | KDA-0.5 | 111.20 | 202.69 | 0.549 | 1.65e-04 | 1.24e-03 | 1.21e-02 | 5.63e-02 | PASS |
| 97 | 8192 | 48 | split/c32/v2/w8 | KDA-0.5 | 95.23 | 175.81 | 0.542 | 2.35e-04 | 1.33e-03 | 2.83e-03 | 9.34e-03 | PASS |
| 98 | 8192 | 56 | split/c32/v2/w8 | KDA-0.5 | 71.20 | 151.55 | 0.470 | 2.63e-04 | 1.27e-03 | 9.59e-04 | 7.04e-03 | PASS |
| 99 | 8192 | 57 | split/c32/v2/w8 | KDA-0.5 | 88.58 | 160.38 | 0.552 | 2.54e-04 | 2.52e-03 | 1.69e-03 | 9.18e-03 | PASS |

## Summary

* workloads: 100
* all within tolerance: **yes**
* geometric mean of ref/mine over all 100 workloads: **0.6680**
* min ref/mine: 0.456, max ref/mine: 1.030
* total ref time: 3252.9 us, total mine: 5250.0 us

### Per band (tokens per sequence)

| tokens/seq | n | plan | geomean ref/mine |
|:--|---:|:--|---:|
| <= 32 | 22 | fused/c16/v8/w8, fused/c32/v8/w8 | 0.802 |
| 33..64 | 19 | fused/c32/v8/w8 | 0.627 |
| 65..128 | 15 | coop/c32/v8/w8, fused/c32/v8/w8 | 0.574 |
| 129..256 | 22 | coop/c32/v8/w8, split/c32/v2/w8 | 0.593 |
| 257..512 | 9 | coop/c32/v8/w8, split/c32/v2/w8, split/c32/v8/w8 | 0.687 |
| 513..1024 | 9 | coop/c32/v8/w8, split/c32/v8/w8 | 0.789 |
| > 1024 | 4 | split/c32/v8/w8 | 0.728 |

### Where the time goes

| band | what limits it |
|:--|:--|
| up to ~32 tokens/sequence | the dependency chain of a single chunk -- about nine barrier-separated phases -- plus the ~1.2 us that any kernel launch costs under this timing definition (a kernel that returns immediately measures 1.0-2.4 us) |
| ~33 to ~1000 | the same chain over two or more chunks. A *second* launch in the same call costs several microseconds more here, which is why the two split kernels are fused into one cooperative launch across this band |
| ~1000 and up | the serial state scan. Staging each chunk's workspace into shared memory is 84% of it (measured with its GEMMs switched off), bound by global-memory latency with L2 at ~31% of peak; `gdn_ut` alongside it is shared-wavefront bound at ~78% of peak. Deeper `cp.async` pipelining, `cp.async.bulk`, more warps, a wider or narrower V split and dropping the V split entirely were all measured and are in notes/APPROACH.md |

### Correctness beyond the benchmark set

`tools/edge.py` checks shapes outside the workload set (T=1, T=2, T=16, an odd
length crossing a chunk boundary, empty sequences at the start/middle, all-empty,
37 sequences, three shapes on the cooperative path, a 1100-token single sequence, a
long+short mix, no input state, scale=0, scale=7.5 and large magnitudes) against
the fp32 naive implementation: 20 cases, 0 failures. The one case whose recurrence
diverges far enough to overflow fp32 is reported as such rather than compared --
there is no ground truth left to compare against.

`tools/edge_ref.py` goes further and compares *our* deviation from the fp32 naive
against *the reference's* on adversarial inputs:

| input | reference rel. err (out / state) | ours (out / state) | \|o\|max |
|:--|---:|---:|---:|
| L2-normalised k (what the model produces) | 2.83e-03 / 3.27e-03 | 2.83e-03 / 3.96e-03 | 3.3e+00 |
| normalised k, q and v scaled 8x | 9.18e-02 / 8.76e-02 | 1.01e-01 / 1.06e-01 | 1.2e+14 |
| unnormalised k | 1.09e-02 / 1.28e-02 | 2.57e-02 / 2.67e-02 | 8.6e+26 |
| unnormalised k, 2 seqs / longer | NaN / NaN | NaN / NaN | NaN (fp32 naive overflows) |

On realistic inputs our error equals the reference's. Where the recurrence itself
diverges (unnormalised k makes the state grow past 1e26), the reference, our kernel
and the fp32 ground truth all degrade or overflow together -- that is a property of
the problem, not of the implementation.

### Dispatch

`fused` is the single fully fused kernel (the state stays in registers for the whole
sequence); `split` is the two-kernel pair (parallel UT transform + serial state
scan); `coop` is that same pair fused into one cooperative launch with a grid-wide
barrier between the phases, which saves the several microseconds that a second
kernel launch costs under this timing definition. `cN` is the chunk length, `vN` the
V-dimension split (blocks per (sequence, head)), `wN` the warps per block. The
choice is a function of total_seq_len and
num_seqs only -- never of the data, and never cached between calls: `cu_seqlens`
lives on the device, so even reading the longest sequence would mean a sync inside
the timed region.

### Official FIB evaluation

`FIB_DATASET_PATH=/root/fib-dataset flashinfer-bench run --local /root/fib-dataset
--definitions gdn_prefill_qk4_v8_d128_k_last --save-results --use-isolated-runner
--log-level INFO --resume --timeout 300 --warmup-runs 1 --iterations 5
--num-trials 3`, run on the packaged solution in
`/root/fib-dataset/solutions/gdn_fused_mma_cuda/`:

```
flashinfer_wrapper_123ca6    {'RUNTIME_ERROR': 100}  distinct workloads=100
gdn_fused_mma_cuda           {'PASSED': 100}  distinct workloads=100
```

All 100 workloads PASSED. The library baseline in the dataset fails to run at
all in this environment, which is why the reference in the table above is
KDA-0.5 everywhere.
