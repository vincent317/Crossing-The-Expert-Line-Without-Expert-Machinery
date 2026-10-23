# 剩余 invalid 实现修复复测

2026-10-23：本轮范围为此前剩余的 17 个 invalid 实现目录，均已完成限定输入范围的修复验收与性能复测。逐目录当前状态以 [coverage.json](../remainder-03413a0-20261022/coverage.json) 为准。以下通过结论只对应归档结果中的源码哈希与所列输入，保留历史失败证据。

## 修复内容

- DSA v11/v14/v15 补齐 base-2 LSE，修复分块合并及 CUDA graph 输出缓存；单 shape opus-5 补齐 LSE，并将并行部分和累积改为 FP32，避免 BF16 atomic 累积误差。
- KDA h64 opus-4.8 返回最终 state；h64 opus-5 修正门控公式。h96 uniform 两版分别修正分块逆矩阵、累积精度及有效 chunk 边界。h96 mixed 保留 FP32 state、分块中间量及矩阵乘精度，避免逐块量化误差累积。
- GDN 两项根据 chunk 大小完成全部逆矩阵展开，保留关键乘法的 FP32 精度；t5709 去掉对既有 Q/K 的额外归一化，t8192 保留跨 chunk 的 FP32 中间量。
- MLA prefill 修正自然对数 LSE，并直接链接 tensor-map 驱动入口。
- FA4 backward 将仅供示例运行的 problem 导入移到主入口；测试适配同步修正官方 LSE 的 H/T 转置。
- FA4 forward B8/S4096 原有 CuTe 内核无法通过 NVVM 编译，已整体重写为 Triton 分块在线 softmax，保留 attn 接口，未调用 reference 或外部 attention 作为回退。
- Top-k 修正分页 FP8/scale 寻址，保留 NaN 传播并使用当前 CUDA stream；测试改为按合法 FP8 页面布局生成两组随机输入，严格比较成员集合。
- FA4 forward 长序列修正 softmax 对非对角分数的错误截断，并使用 FP32 分数累积及归一化。

## 精度与性能协议

单 shape 前向一般采用数学 reference，atol=0.01、rtol=0.01，要求全部元素满足，未放宽阈值。FA4 backward 对全部三个梯度使用独立 Tri Dao/PyTorch 原有判据。Top-k 检查精确成员集合及 padding。GDN 覆盖冻结输入及随机种子 0、42；其余输入范围见测试适配及结果的 metrics，不能将两个 seed 标签一律解释为两个独立生成的官方冻结输入。

每项先检查精度，再进行三轮性能测量，每轮热缓存和冷 L2 各 50 次，并在计时后重新检查全部输出。以下为三轮 kernel-span 中位数范围，单位 µs；不是 Python 端到端延迟，也不把不同机器或协议下的历史快耗时当作同条件 A/B。

- `kda05_reproduction/dsa_decode_fwd/opus-4.8/dsa_triton_v11`：热缓存 **4.544–4.544**；冷 L2 **5.920–5.952**。任务 `tasks/1340694`；[结果](remaining-30-r1-results.json)。
- `kda05_reproduction/dsa_decode_fwd/opus-4.8/dsa_triton_v14`：热缓存 **7.008–7.008**；冷 L2 **9.216–9.232**。任务 `tasks/1340695`；[结果](remaining-31-r1-results.json)。
- `kda05_reproduction/dsa_decode_fwd/opus-4.8/dsa_triton_v15`：热缓存 **10.784–10.815**；冷 L2 **14.096–14.128**。任务 `tasks/1340697`；[结果](remaining-32-r1-results.json)。
- `goal_single_shape/kda_fwd/h64_fixed8192/opus-4.8`：热缓存 **3792.767–3793.055**；冷 L2 **3793.311–3795.551**。任务 `tasks/1340698`；[结果](remaining-64-r1-results.json)。
- `goal_single_shape/kda_fwd/h64_fixed8192/opus-5`：热缓存 **594.051–597.362**；冷 L2 **598.402–601.235**。任务 `tasks/1340721`；[结果](remaining-65-r2-results.json)。
- `goal_single_shape/kda_fwd/h96_uniform/opus-4.8`：热缓存 **2830.695–2831.111**；冷 L2 **2833.271–2833.511**。任务 `tasks/1340700`；[结果](remaining-69-r1-results.json)。
- `goal_single_shape/kda_fwd/h96_uniform/opus-5`：热缓存 **2221.020–2222.203**；冷 L2 **2223.756–2224.715**。任务 `tasks/1340756`；[结果](remaining-70-r2-results.json)。
- `goal_single_shape/mla_paged_prefill/b18_q15883_kv15937/fable-5.1`：热缓存 **420.049–422.082**；冷 L2 **424.418–435.905**。任务 `tasks/1340702`；[结果](remaining-71-r1-results.json)。
- `goal_single_shape/fa4_bwd/varlen_t32768_n13_h16/opus-4.8`：热缓存 **3370.076–3370.748**；冷 L2 **3375.036–3375.180**。任务 `tasks/1340755`；[结果](remaining-fa4b-r2-results.json)。
- `goal_single_shape/dsa_decode_fwd/t2_topk2048/opus-5`：热缓存 **6.592–6.592**；冷 L2 **7.488–7.567**。任务 `tasks/1340779`；[结果](remaining-dsa-r2-results.json)。
- `goal_single_shape/gdn_fwd/t5709_n2/opus-5`：热缓存 **2592.283–2595.594**；冷 L2 **2593.610–2600.187**。任务 `tasks/1340789`；[结果](remaining-gdn5709-r2-results.json)。
- `goal_single_shape/gdn_fwd/t8192_n20/opus-5`：热缓存 **1124.341–1126.037**；冷 L2 **1125.525–1126.741**。任务 `tasks/1340790`；[结果](remaining-gdn8192-r2-results.json)。
- `goal_single_shape/kda_fwd/h96_mixed/opus-5`：热缓存 **11237.679–11245.871**；冷 L2 **11246.302–11246.639**。任务 `tasks/1340800`；[结果](remaining-kdamixed-r2-results.json)。
- `goal_single_shape/fa4_fwd/gqa_b1_n32768_h32_kv8_causal/fable-5.1`：热缓存 **8191.992–8226.907**；冷 L2 **8317.063–8327.992**。任务 `tasks/1340815`；[结果](remaining-fa4f-b1-r3-results.json)。
- `goal_single_shape/fa4_fwd/gqa_b8_n4096_h32_kv4_causal/fable-5.1`：热缓存 **2497.732–2502.164**；冷 L2 **2499.764–2503.972**。任务 `tasks/1340914`；[结果](remaining-fa4f-b8-r5-results.json)。
- `goal_single_shape/dsa_topk_indexer/topk_T2_B_mid/fable-5.1`：热缓存 **3526.017–3527.906**；冷 L2 **3530.593–3532.065**。任务 `tasks/1340915`；[结果](remaining-topk-valid-results.json)。

## MoE 全 shape

19/19 workload 均通过沿用的既定评测配置：atol=1、rtol=0.3、required_matched_ratio=0.9。实测匹配比例为 94.6568%–98.6727%，因此这里的“通过”不表示逐元素 100% 满足单 shape 的 0.01 阈值。候选耗时为 74.315–1389.622 µs；这是官方评测器计时，与上面的 kernel-span 协议不同。修复移除过时的 Triton CompiledKernel 参数缓存，使用公开 JIT launcher，消除 36/27 参数 ABI 不匹配。独立源码与 solution.json 内嵌源码已核对一致。任务 tasks/1340758；[完整结果](remaining-moe-r1-results.json)。

GDN 与 KDA 部分修复提高了中间计算精度，性能成本已反映在实测中。原始实现未通过精度或无法运行，不提供修复前后加速比。

## 频率及复现边界

B200 任务的前后负载 smoke 均需通过既定门槛：至少 90% 的负载样本 SM 时钟达到 1900 MHz。多数样本为 1965 MHz，部分出现 sw_power_cap 或短暂低频；KDA mixed 的后置检查为 15/16 达标，其中一个样本 1882 MHz。通过门槛不等于全部计时瞬间固定频率，原始采样见各日志。

Top-k 旧输入由通用 int8 随机生成器构造，不能表达有效的量化页面：scale 包括约 1e34 的值，reference 产生大量 NaN。其原始失败证据仍保留，未声明修复版通过该异常输入；该输入不用于本次性能。新输入保留原 batch、长度、页表和全部 shape，只按 FP8 数据+正有限 FP32 scale 正确打包，随机种子 0、42 的成员集合及 padding 均精确匹配。原始未修代码在同一合法输入上失败，见 remaining-topk-valid-red.log。

MLA prefill 原始长度分布缺失，使用保持总量和关键长度的合成分布；未宣称复现原始长度分布。KDA 部分候选通过适配转换布局、gate、state 类型；这些转换是测试调用合同的一部分，见适配源码。测试不能证明任意形状、任意输入均正确。

实际执行入口 `*-entry.sh` 内嵌当次候选和测试快照，`*-job.json` 保存任务 ID，`*.log` 保存精度退出码、频率检查及归档，`*-results.json` 保存源码哈希和原始性能样本。`validated-summary.json` 是从这些结果生成的摘要，不是独立数据来源。

这些 GPU 测试依赖固定审计环境、官方 definition/输入、Tri Dao cache，以及共享盘上的 official、paired、metrics、kernel_span、run_gdn 等工具。仓库未附带全部外部依赖；具备入口指定的共享环境时可执行入口重建验收，不能宣称任意干净机器开箱即跑。本地静态检查不代替 GPU 验收。

此前缺少独立候选实现的是 `goal_single_shape/dsa_topk_indexer/topk_T3_C_large/fable-5.1`。收尾同步发现上游 `731d977` 已补齐 CUDA 实现；新版本尚未纳入本轮复测，当前标记 implementation_added_unverified，不能算作通过。
