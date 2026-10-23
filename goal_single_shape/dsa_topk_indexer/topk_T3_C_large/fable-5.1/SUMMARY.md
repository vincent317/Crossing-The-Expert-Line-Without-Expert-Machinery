# SUMMARY — DSA top-k indexer（`dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`，工作负载 8635db8f）

## 最终结果（B200，cctl devspace ai4ai2 集群；最终验证 job 1076256，节点 20nw；另见 job 1076197，节点 sgxo）

| 项目 | 数值 |
|---|---|
| ref device time（profiler kernel span，warmup 10 / iters 50，每次迭代前刷 512 MB L2，中位数） | **5.06 / 5.09 µs**（节点 20nw）；5.26 µs（节点 sgxo）；历次实测 4.99–5.30 µs |
| 新 kernel device time（同口径、同数据、同一次运行） | **8.99 / 9.02 µs**（节点 20nw）；8.96 µs（节点 sgxo）；最好一轮 8.94 µs |
| 目标（ref / 1.2） | 4.21–4.39 µs → **未达成**（约 0.56–0.59×） |
| ref 相对 fp64 naive 的最大误差 E_ref | 5.34e-01（ref 自身有 10 个 token 与精确 top-k 不同） |
| 容差 tol = 7 × E_ref | 3.74 |
| 新 kernel 相对 fp64 naive 的最大误差 | **0.0**（两个随机种子；选出集合与精确 top-k 完全一致；重复调用结果一致） |

误差度量：把所选 token 按 fp64 分数降序排列，与真实 top-k 的分数序列逐位取最大绝对差；同时校验 -1 布局、id 合法性与唯一性。

## 交付物

- `dsa_topk_kernel.cu` / `topk_indexer.py`：kernel 与 Python 入口（`run(q, k_cache, weights, seq_lens, block_table)`）。
- `harness.py`：数据生成（seq_lens / block_table 来自 `ref/data` 的 safetensors，其余随机，k cache 按 fp8+per-token scale 打包）、fp64 naive、误差度量、device-time 测量。
- `check_accuracy.py`、`bench.py`、`timeline.py`（kernel 内各阶段时间戳）、`diag.py`（轮询 watchdog 诊断）、`micro.cu/micro.py`、`micro2.cu/micro2.py`（成本模型微基准）、`sass_stats.py`。
- 全部已 git commit；ref 目录始终作为黑盒仅被 import/调用。

## kernel 设计（从零实现，未参考 ref 源码）

单个 cooperative launch，grid = SM 数（148），每个 CTA（512 线程）在设备上从 `seq_lens` 解码自己的工作项（一次打包 warp scan），工作项顺序为 `[长 batch 的 tail 项][长 batch 的每页评分项][短 batch 的填充项]`：

1. **填充项**（seq_len ≤ 2048 的 26 个 batch）：全部 token 都被选中，直接写全局 token id 与 -1，不计算分数。
2. **评分项**（batch 12 的 91 页，每页一个 CTA）：`mma.sync.m16n8k32` e4m3 tensor core，fp32 累加，A=k 页（64 token）、B=q（64 head），维度做统一置换以便每线程 32 字节连续加载；relu 后按 head 权重求和再乘 per-token scale，转成单调 uint32 key（0 保留为“未写入”），用 `st.relaxed/volatile` 写入 scratch 后立即退出（无 fence、无原子、无标志）。
3. **tail 项**（每个长 batch 一个 CTA）：以 key 非零作为到达信号，用 `ld.volatile.global.v4` 每线程 4 条 16 字节 load 轮询，key 到达即增量累加到 smem 的 4096-bin 细直方图（key bits[26:15]）与 64-bin 粗直方图；全部到达后两级 warp 扫描定位阈值 bin，bin 内 ~24 个候选做 warp 并行 rank 选择，其余“确定入选”的 key 用寄存器内 ballot/popc 压缩写入 smem 暂存区，再合并写出。数字有效性（所有 key 的 bits[31:27] 相同，或分数恰为 0）不满足或候选过多时回退到寄存器内 radix select（12/12/8 bit，范围归一化）——保证任意数据正确。

## 时间线（cold L2，单位 µs，含 ~0.3 µs 的 globaltimer 读取伪影）

| 阶段 | 时刻 |
|---|---|
| seq_lens 到达 + 解码 + block_table 进 smem | ~1.5 |
| k/q 加载 + mma + 归约 | ~3.0（中位）/ 3.5（最慢 CTA） |
| key 存储 | ~3.2 / 3.7 |
| tail 吸收完最后一页 | ~4.7–5.1 |
| 找到阈值 bin | +0.7 |
| 扫描 + 候选收集 | +1.0 |
| rank 选择 | +0.45 |
| 压缩 + 写出 + 清零 | +1.0–1.2 |
| kernel 结束（profiler） | ~9.0 |

## 瓶颈分析与经验

1. **跨 CTA 通信的固有延迟**：每次 DRAM 往返 ~0.4 µs、L2 往返 ~0.17–0.4 µs；评分链路 seq_lens → block_table → k 页至少两次依赖的 DRAM 往返，再加最慢 CTA 的抖动（~0.4 µs）。
2. **全局原子加不可用作每 key 聚合**：91 个 CTA 各 64 次 RED/ATOM（无论热点还是分散地址）+ fence + red 实测 1.3–1.9 µs，直接压在评分关键路径上（v3–v5 的 bucket/直方图方案因此失败）。
3. **单 CTA 选择阶段受依赖链延迟支配**：ballot/shfl/LDS 每步 ~20–30 cycle，16 warp 交错不足以隐藏；带返回值的 smem 原子、`match_any`、串行 rank 循环都极慢；SASS 中有 67 个 `ENDCOLLECTIVE`/74 个 `WARPSYNC`，说明大量 `*_sync` 集合操作被编译成了保守的“分歧集合循环”，这是后续可优化点。
4. **内存模型**：生产者弱一致 `st.global` + 消费者 `ld.global.cg` 轮询曾工作 7 轮后开始死锁（数据滞留在生产 SM 缓存），必须用强操作（`st/ld.relaxed.gpu` 或 `.volatile`）。
5. **微基准与真实 kernel 的差异**：smem 原子的“发射时间”很短但排空很慢；`%globaltimer` 读取约 0.3 µs；`clock64` 周围的纯算术会被编译器重排，需要 `asm volatile` 依赖才能校准。
6. 参照实现 5.0 µs 已接近该架构下“两次 DRAM 往返 + 一次跨 CTA 汇合 + 单 CTA 选择”的下界（估算 ~4 µs）；要达到 4.4 µs 需要把 tail 的每个阶段压到理论最小，或采用完全不同的选择结构（例如 thread block cluster + DSMEM 内的分布式选择、或把细直方图随评分结果一并由生产者以无争用方式发布）。

## 最后几轮的对照实验（均未改变结论）

- **只跑评分 + 填充、不跑 tail 的 kernel：3.68 µs**（job 1076286）。即 tail（从最后一页 key 到达到写完输出）占约 5.3 µs，而 ref 全程 5.1 µs —— ref 的评分后阶段 ≤ ~1.4 µs。
- `.volatile` vs `ld/st.relaxed.gpu`：9.20 vs 9.15 µs（噪声内）。
- bin 级近似选择（阈值 bin 内成员按槽位顺序取，去掉候选收集与 rank；误差 ≤ 细 bin 宽度，本数据仍为 0）：9.33 µs，无收益。
- 去掉 block_table 的 smem 缓存：9.30 µs，无收益。
- warp 一致广播消除部分集合循环（ENDCOLLECTIVE 67→28）：耗时不变。
- 节点差异：sgxo 8.94–8.96 µs，20nw 8.99–9.02 µs，2x0k 9.25–9.31 µs；ref 4.99–5.33 µs。

这些宏变体（`DSA_VARIANT=relaxed|approx|nobt|notail`）保留在 `topk_indexer.py` / `dsa_topk_kernel.cu` 中，默认编译为精确 + `.volatile` 版本。

## 迭代记录（41 轮 GPU 提交）

v1 last-CTA + 全局直方图（14.8 µs）→ v2 归一化 smem radix（13.9）→ v3 fp8 mma + bits[26:15] 直方图（14.4）→ v4 专用 tail + 干跑预热（13.7）→ v5 bucket 排序 + 二分 gather（14.4）→ 微基准两轮 → v6 key 到达轮询、无原子评分侧（9.8）→ v7 私有轮询 + rank 输出（9.15）→ v8 1024 线程 / 向量轮询（9.9）→ v9 512 线程 + warp 并行 rank + 打包解码（**8.94–9.12**）→ 死锁定位与强一致修复 → v10–v12（流水轮询、scan 压缩、字节计数直方图）均未优于 v9 → 回退到 v9 并加入 warp 一致广播（SASS 中 ENDCOLLECTIVE 67→28，耗时不变）→ 最终版本 → 对照实验（无 tail 下界、一致性模式、近似选择、bt 缓存）→ 结束。
