# GDN FWD on B200 — 少数几条长序列（T=5709 = 5637 + 72）

所有数字均在 cctl `aiforai-b200` 资源池的单卡 B200（cc 10.0，148 SM，driver 580.105.08）上实测，
镜像 `cybertron/pytorch:26.07-py3`（torch 2.13.0a0 / CUDA 13.3 / triton 3.7.1）。
device time 口径：CUDA graph 捕获整次调用后连续 replay，用 CUDA events 卡住 GPU 执行跨度
（不含 host 发射开销），取 5 组 × 50 次的中位数；ref 与新 kernel 用**同一数据、同一脚本、同一测法**。

## 1. 最终结果

| 项目 | 数值 |
|---|---|
| ref（FlashInfer 上游 CuTe-DSL `flashinfer.gdn_prefill.chunk_gated_delta_rule`，切链/CP 路径） | **114.97 μs**（cudagraph）／125.94 μs（无 graph） |
| 本实现（Triton，5 次发射） | **133.14 μs**（cudagraph）／224.22 μs（无 graph） |
| 加速比 | 0.864×（目标 ≥ 1.2×，即 ≤ 95.81 μs）→ **未达成** |
| 精度 max\|mine − naive_fp32\|（output） | 1.211e-03（tol 8.564e-03）→ PASS |
| 精度 max\|mine − naive_fp32\|（new_state） | 8.610e-03（tol 5.442e-02）→ PASS |

ref 114.97 μs 与题目给出的 0.105 ms 基本吻合，可确认测到的就是切链路径
（日志中该 workload 命中 `CP delta rule` dispatch）。

分核耗时（best config：`m=1 seg=12 ns=4 warps_prep=2 stages_out=3 stages_seg=3`）：

```
prep 43.07   seg 28.79   top 16.49   attn 16.49   out2 29.06   (合 133.14 μs)
```

## 2. 容差是怎么定的（重要说明）

原计划用 CuTe-DSL ref 与 fp32 naive 的最大误差作为 E_ref。实测发现：

* `flashinfer-python==0.6.18.post1`（唯一带 SM100 切链路径、113 μs 的版本）在本环境下
  **输出张量 100% 为 NaN**（5,846,016/5,846,016 个元素，new_state 同样全 NaN）。
  已排除的因素：state 是否转置、`use_qk_l2norm_in_kernel` 开关、`cu_seqlens` int32/int64、
  gate dtype、`initial_state=None`、`scale=None`、预分配 `output=/output_state=`
  （返回的就是我传入的 buffer，且被写成了 NaN）、把门控整体缩小到 1/33（g ∈ [−0.17, 0]）；
  甚至 T=64、单序列、g=0、beta=0.5、零初始 state 的最小 smoke test 也是全 NaN。
  这与 FlashInfer issue #2490（GDN prefill 产生 NaN）一致，属该版本回归。
* `flashinfer-python==0.6.14 / 0.6.8` 在最小 smoke test 上输出有限，但在 B200 上没有 SM100 切链路径
  （`CP delta rule is currently implemented only for SM90`），退化路径 346 μs，且在完整
  workload 上同样返回 NaN。

因此 E_ref 取自**同一算子的另一份公开参照实现** flash-linear-attention 的
`chunk_gated_delta_rule`（同样 bf16、同样 chunk 化 UT transform）：

```
E_ref(out)   = 1.223460e-03  -> tol_out   = 7 * E_ref = 8.564219e-03
E_ref(state) = 7.773876e-03  -> tol_state = 7 * E_ref = 5.441713e-02
```

本实现的误差（1.211e-03 / 8.610e-03）与该参照自身的误差同级（0.99× / 1.11×），
说明数值路径与参照等价，不是靠放松精度换速度。

顺带一个口径参考：同一 harness 下 FLA 这个**数值上可用**的参照实测 237.85 μs（cudagraph）
／557.67 μs（无 graph），本实现 133.14 μs 相对它是 1.79×（该数字测于 CUDA 12.9 + triton 3.3.1
镜像；在 26.07 镜像上 FLA 因内部 host 同步无法被 CUDA graph 捕获，故未复测）。
真正被题目指定的切链 ref 是 114.97 μs 那条，本实现相对它 0.864×，未达 1.2×。

## 3. 实现要点（`gdn_kernel.py`）

chunk 化 gated delta rule + UT(WY) transform，C=64：

```
U  = u0 - w S          u0 = Tm V,  w = Tm K_Gamma,  Tm = (I + beta*Mtilde)^-1 diag(beta)
o  = qg S + aq U       qg_i = Gamma_i q_i * scale,  aq_ij = (q_i.k_j)(Gamma_i/Gamma_j), j<=i
S' = gam S + KZ^T U    KZ_i = exp(gc_last - gc_i) k_i
```

针对"少数几条长序列"这一形态（只有 16 条独立状态链，148 个 SM 严重闲置）做的三件事：

1. **状态 V 维解耦**：`S_kv` 的每一列在递推里互不耦合，串行 scan 可按 V 切块并行。
2. **两级并行 scan**（关键，scan 112.7 → 45.3 μs）：把 `[Delta | P]` 增广状态（Delta 初值 0，
   P 初值 I）在每个 segment 内并行求解，一个 segment 就塌缩成 `S_out = P S_in + Delta`；
   随后只对 ~9 个 segment 做短串行（`top`）。P 以 bf16 落盘，`top` 因此从 35 → 16.5 μs。
3. **链合并（m>1，代码保留）**：相邻 M 个 chunk 可精确合并成等价的 M*C 行仿射变换，
   scan 的 MAC 完全不变而串行深度减半（`U_1 = (u0_1 - G u0_0) - (gam_0 w_1 - G w_0) S_0`,
   `G = w_1 KZ_0^T`）。实测 m=2 让 scan 更快但 prep 变贵，净收益为负，故默认 m=1。

数值安全：所有衰减比 `exp(gc_i - gc_j)` 都在 matmul **之后**逐元素施加并 clamp 到指数 ≤ 0，
中间量不可能溢出（这正是参照实现疑似踩到的坑）。中间量按 block 对齐 + padding，
使两个热循环都是静态 trip count、循环体内无分支。

## 4. 瓶颈分析（为什么没到 1.2×）

Triton 在 sm_100 上确实发出了 `tcgen05.mma`（诊断：prep 56 条、scan 12 条、out 32 条），
但本算子的 tile 天生偏小（M=64、N=32~128、K=64~128），实测**有效算力稳定在 ~170 TFLOPS**
（B200 bf16 dense 峰值 ~2250 TFLOPS 的 7.5%）。按每 token 每 head 的 MAC 统计：

| 阶段 | MAC/token/head | 说明 |
|---|---|---|
| prep（UT transform） | ~49K | 其中 6 个 64³ 的 Newton-doubling 求逆占 24.6K |
| attn（块内注意力矩阵） | ~8K | |
| seg（两级 scan 的 level-1） | ~65K | 增广宽度 2D=256，是并行化的 2× 代价 |
| top | 可忽略 | 但是 8 步串行 |
| out2 | ~57K | 其中 16K 是从 checkpoint 重放状态的代价 |

合计约 8.2 G MAC（16.4 GFLOP）。在 170 TFLOPS 下地板就是 ~100 μs，实测 133 μs 已经是
该地板的 75%。也就是说：**再怎么调参也过不了 94.2 μs，必须减少 FLOPs 或提高单位算力。**

三条已识别但未完成的路：

1. **写成 CUDA/CuTe，把 tile 提到 M=128 并用 TMA + 显式 tcgen05 流水**。参照实现 113 μs
   做的是同一套数学，说明 2× 以上的单位算力是可达的；Triton 这边受限于
   accumulator→operand 的 SMEM 往返（scan 每步实测固定开销 ~0.65 μs）和共享内存占用
   （`out` 一度 156 KB/block，只能放 1 block/SM）。
2. **去掉 level-1 的 P 开销**：segment 变换的非单位部分秩 ≤ SEG*MC，只要 ≤ 128 就能用
   低秩堆叠形式（即链合并）代替稠密 (K,K) 合成，可省掉 seg 的一半 FLOPs；代价是串行
   深度上升，需要一个三级（合并→稠密合成→展开）的混合结构。
3. **消掉 out2 的重放**：若 `top` 能按 block 而非 segment 落 checkpoint，out2 少一个
   `KZ^T U`（16K/token）且完全无串行；但按 block 落盘要 23–46 MB 额外流量，需要权衡。

## 5. 文件

| 文件 | 作用 |
|---|---|
| `gdn_kernel.py` | Triton kernel（prep / seg / top / attn / out2 + flat scan 与链合并路径） |
| `gdn_data.py` | workload 定义（形状/布局/dtype）、门控语义、fp32 naive 逐 token 真值 |
| `final.py` | 定稿证据脚本：ref device time + E_ref/tol + 精度 + benchmark |
| `check.py` | 精度检查（naive vs ref vs 本实现） |
| `bench.py` / `bench_util.py` | device time 测量与配置扫描 |
| `probe_ref.py` / `probe_ref2.py` / `ref_flashinfer.py` | ref 的安装、公开 API 调用约定实测与计时 |
| `diag.py` | 编译产物诊断（寄存器/spill/共享内存/MMA 指令种类） |

ref 全程只做「安装 / import / 调用」，未阅读或复制其源码；调用约定是用 fp32 naive 真值
反推出来的（state k-last、`use_qk_l2norm_in_kernel`、`cu_seqlens` dtype 等逐一实测）。
