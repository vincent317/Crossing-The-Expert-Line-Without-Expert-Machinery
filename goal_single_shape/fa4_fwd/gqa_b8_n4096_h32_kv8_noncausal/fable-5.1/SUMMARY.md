# SUMMARY — FlashAttention FWD（Qwen3-235B，GQA 12:1，非 causal）on B200

## 结论

**目标未达成，且经实测判定在 bf16 下不可达。** 五项完成条件中 1、2、3、5 已满足，第 4 项
（device time ≤ ref/1.2）未满足。原因不是调优不够，而是要求的算力超过了这块 B200 上 bf16
tensor core 的实测天花板。下文给出全部数据与依据。

## 完成条件逐项

| # | 条件 | 状态 | 证据 |
|---|------|------|------|
| 1 | ref device time 已在 B200 测得并打印 | ✅ | job 1049127：profiler kernel-only **9.5954 ms**（1 个 kernel），CUDA events 中位 9.6215 ms。job 1049346 同口径 9.6139 ms |
| 2 | fp32 naive 实现 + E_ref + tol=7×E_ref | ✅ | `bench_common.fp32_naive_reference`；**E_ref = 3.110394e-04，tol = 2.177276e-03** |
| 3 | 新 kernel 与 fp32 naive 的最大误差 ≤ tol | ✅ | v3：**E_new = 3.117323e-04，E_new/tol = 0.143，PASS** |
| 4 | 新 kernel device time ≤ ref/1.2 | ❌ | 需 ≤ 8.0116 ms；v3 最好 **12.0221 ms**（kernel-only），speedup 0.80× |
| 5 | 源码/benchmark/精度脚本已 commit | ✅ | 见下方文件清单，均在 `main` 上 |

## 最好结果

`fa_triton3.py`，配置 `BLOCK_M=128 BLOCK_N=128 num_warps=4 num_stages=3 warp_specialize=True`：

- CUDA events 中位 12.1353 ms，profiler kernel-only **12.0221 ms**，**1087 TFLOPS**
- 精度 E_new = 3.117e-04（与 ref 的 3.110e-04 相当），E_new/tol = 0.143
- 与 ref 同 job、同数据、同测法（刷 L2，warmup 10 / iters 50，CUDA events + profiler 双口径）

演进：v1（标准 online softmax）869 TFLOPS → v3（去掉 running max）1087 TFLOPS，+25%。

## 为什么不可达：实测 bf16 算力天花板

job 1049346 在同一块卡、同一测时 harness 下测 cuBLAS **纯方阵 bf16 GEMM**（没有 softmax，
是 tensor core 的最优情形）：

| GEMM | 中位 | 峰值 |
|------|------|------|
| 4096³ | 1509 TFLOPS | 1547 |
| 8192³ | **1702 TFLOPS** | 1732 |
| 16384³ | 1510 TFLOPS | 1759 |
| 8192³ 200 次背靠背不刷 L2 | ~1400 TFLOPS（功耗/时钟限制） | — |

对照本任务（总算量 1.3194e13 FLOP）：

- ref（flash-attn-4 cute）：1372 TFLOPS = **天花板的 ~80%**
- cuDNN SDPA fused attention（job 1049260 同 job 实测）：1390 TFLOPS，与 ref 几乎相同
- **目标 1.2× = 8.0116 ms = 1647 TFLOPS = 天花板的 ~97%**

两个互相独立、各自深度调优的厂商 kernel 停在同一个数字上，说明这是硬件墙而非调优巧合。
一个必须额外为每个 score 做 exp2 的 attention kernel，不可能超过纯 GEMM 在同一芯片上的
sustained 吞吐。因此 1.2× 在 bf16 精度下不可达。

## 排除掉的其它路径

- **fp8（e4m3）**：唯一能突破 bf16 天花板的手段（Blackwell fp8 dense 峰值是 bf16 的 2×）。
  误差分析：用在 QK 矩阵乘上，score 误差 std ≈ 0.05，超过由 tol 反推的预算 0.041，超容差；
  只用在 PV 上，输出最大误差 ≈ 1.2e-3 vs tol 2.18e-3（余量 1.8×），但 PV 只占 MMA 的一半，
  且要额外做逐行 max/scale/cast，softmax 会成为新瓶颈，拿不到从 1087 到 1647 所需的 1.5×。
- **更大 tile（BLOCK_M=256）**：Triton 3.7.1 sm100 TMEM 只有 512 列，BM=256/D=128 需 896 列，OOR。
- **2-CTA MMA（num_ctas=2）**：Triton 3.7.1 `NVWSInsertAref` pass 失败，无法编译。
- **BLOCK_N=64 / BLOCK_M=64 / 不做 warp specialization**：均更慢（见 job 1049195、1049346 扫描表）。
- **CuTe DSL 手写 tcgen05 kernel**：理论上能拿到 ref 的 ~80% 天花板占用率，但上限仍是同一堵墙，
  且 ref 本身就是 CuTe DSL 写的——最多追平，不可能 1.2×。未投入。

## 文件

| 文件 | 作用 |
|------|------|
| `bench_common.py` | 问题定义、确定性输入、CUDA events / profiler 双口径 device time、fp32 naive 参照与误差 |
| `bench_ref.py` | 测 ref device time，得 E_ref / tol |
| `fa_triton.py` | v1：标准 online softmax，TMA + block-pointer 双路径 |
| `fa_triton2.py` | v2：num_ctas / 大 tile 尝试（记录了不可编译的结论） |
| `fa_triton3.py` | **v3（最好）**：Cauchy-Schwarz 静态上界替代 running max |
| `bench_new.py` | v1 配置扫描 + 与 ref 对比 + 精度 |
| `bench_sweep2.py` | v2 扫描 + cuDNN/flash SDPA 参照 |
| `bench_sweep3.py` | cuBLAS 天花板探针 + v3 扫描 + 与 ref 对比 + 精度 |

## Jobs

| job | 内容 | 关键输出 |
|-----|------|----------|
| 1049127 | ref 测时 + tol | ref 9.5954 ms；E_ref 3.110e-04；tol 2.177e-03 |
| 1049195 | v1 扫描 | 最好 15.18 ms / 869 TFLOPS，精度 PASS |
| 1049260 | v2 + cuDNN | num_ctas=2 编译失败；cuDNN 9.49 ms / 1390 TFLOPS |
| 1049346 | 天花板 + v3 | cuBLAS 8192³ 1702 TFLOPS；v3 12.02 ms / 1087 TFLOPS，精度 PASS |
