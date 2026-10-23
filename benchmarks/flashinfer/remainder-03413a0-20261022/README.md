# 当前上游算子复测结果

本报告于 2026-10-23 发布，随后完成三项消融及其余 17 项修复验收。历史记录对应下述固定提交；`repaired_pass` 行对应修复证据内的源码哈希，不代表其他最新代码已复测。

固定上游提交：`03413a0a10e670d6881314765b4a07bfdcbe87ba`。统计单位是实现目录，不是算子家族或 workload。

共 135 个实现目录、11 个算子家族；在各自已测输入与判据范围内通过 135 个（原版及新补实现 115 个、修复后 20 个），当前 `invalid` 0 个。此前缺失的 T3 Top-k 已由上游提交 `731d977` 补齐，并完成[补复测](../t3-retest-20261023/README.md)；待复测 0 个，当前已知缺少实现 0 个。修复与补复测任务均已完成。

`coverage.json` 是逐目录结论、源码对应关系和证据入口。三项消融及其余 17 项均已修复验收，见[本轮修复报告](../remaining-repair-20261023/README.md)。原测试状态、失败原因和历史源码对应关系保留；`repair` 链接新源码的验证证据。Top-k 合法输入回归与旧异常输入分开记录；MoE 沿用 atol=1、rtol=0.3、匹配比例≥90%，不代表逐元素 0.01 精度。六个 FA4 全 shape 目录的原有代表性覆盖限制仍然有效。

## 独立基线复测

基线已另行执行[独立复测](../baseline-retest-20261023/README.md)，覆盖 11 个算子家族，逐项精度、性能与频率证据见该报告。135 个候选目录通过不代表全部基线通过；Top-k 严格成员集合检查发现失败配置，不能据此计算有效加速比。

## GDN 全 shape

- fable-5.1：已收 100/100，状态 {'PASSED': 100}；通过项候选耗时 3.91–231.50 µs。证据：`gdn-final-fable-5.1-traces.json`。
- opus-4.8：已收 100/100，状态 {'PASSED': 100}；通过项候选耗时 9.24–2013.16 µs。证据：`gdn-final-opus-4.8-traces.json`。
- opus-5：已收 100/100，状态 {'PASSED': 100}；通过项候选耗时 5.48–241.08 µs。证据：`gdn-final-opus-5-traces.json`。

长序列停滞的堆栈定位在评测器对逐 token Python reference 的重复计时。使用评测器原生 `profile_baseline=False` 关闭 reference 计时，保留 reference 输出计算、官方精度检查和候选计时；候选源码未改。trace 中 reference latency 和 speedup 的零值是禁用计时的占位值，不可用于加速比。

完整通过的 GDN 批次都有前后及每 25 项频率检查；以原始日志中的 `SMOKE_RESULT` 为准。其余新增通过项的三轮热/冷缓存耗时汇总在 `monitor-passing-results.json`，每轮各 50 次；全 shape 官方计时与此协议不同，不能直接混排比较。

## 历史失败与修复入口

- **repaired_pass** `goal_single_shape/dsa_decode_fwd/t2_topk2048/opus-5`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：缺少规定的 LSE 输出。 证据：`other-additional-results.json`。
- **repaired_pass** `goal_single_shape/dsa_topk_indexer/topk_T2_B_mid/fable-5.1`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：Top-k 成员集合错误；按 token 交错布局寻址，与输入的分页布局不符。 证据：`other-additional-results.json`。
- `goal_single_shape/dsa_topk_indexer/topk_T3_C_large/fable-5.1`：历史提交只有 reference 调用脚本；上游 `731d977` 已补齐 CUDA kernel，新版本已通过[补复测](../t3-retest-20261023/README.md)。证据：`coverage.json`。
- **repaired_pass** `goal_single_shape/fa4_bwd/varlen_t32768_n13_h16/opus-4.8`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：候选导入了仓库缺失的 problem 模块。 证据：`backward-0-results.json`。
- **repaired_pass** `goal_single_shape/fa4_fwd/gqa_b1_n32768_h32_kv8_causal/fable-5.1`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：CUDA 12.9 下可运行；134217728 个元素中有 1 个不满足 atol=0.01、rtol=0.01、全部元素通过的判据，最大绝对误差 0.015625。 证据：`cuda12-fa4-b1-results.json`。
- **repaired_pass** `goal_single_shape/fa4_fwd/gqa_b8_n4096_h32_kv4_causal/fable-5.1`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：CuTe DSL 的 NVVM 后端编译失败。 证据：`clock-evidence-final-results.json`。
- **repaired_pass** `goal_single_shape/gdn_fwd/t5709_n2/opus-5`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：采用公布参数后，输出最大绝对误差 18560、state 最大绝对误差 53341.035。 证据：`gdn-single-published-config-results.json`。
- **repaired_pass** `goal_single_shape/gdn_fwd/t8192_n20/opus-5`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：采用公布参数后，候选输出出现非有限值。 证据：`gdn-single-published-config.log`。
- **repaired_pass** `goal_single_shape/kda_fwd/h64_fixed8192/opus-4.8`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：上游源码匹配先前失败版本；修复后通过不能转移。 证据：`../paired-repair-specs.json`。
- **repaired_pass** `goal_single_shape/kda_fwd/h64_fixed8192/opus-5`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：上游源码匹配先前失败版本；修复后通过不能转移。 证据：`../paired-repair-specs.json`。
- **repaired_pass** `goal_single_shape/kda_fwd/h96_mixed/opus-5`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：将 reference 的 lower_bound 与候选对齐后，state 仍未通过精度检查。 证据：`interrupted-harvest-results.json`。
- **repaired_pass** `goal_single_shape/kda_fwd/h96_uniform/opus-4.8`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：上游源码匹配先前失败版本；修复后通过不能转移。 证据：`../paired-repair-specs.json`。
- **repaired_pass** `goal_single_shape/kda_fwd/h96_uniform/opus-5`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：上游源码匹配先前失败版本；修复后通过不能转移。 证据：`../paired-repair-specs.json`。
- **repaired_pass** `goal_single_shape/mla_paged_prefill/b18_q15883_kv15937/fable-5.1`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：上游源码匹配先前失败版本；修复后通过不能转移。 证据：`../paired-repair-specs.json`。
- **repaired_pass** `kda05_reproduction/dsa_decode_fwd/opus-4.8/dsa_triton_v11`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：上游源码匹配先前失败版本；修复后通过不能转移。 证据：`../paired-repair-specs.json`。
- **repaired_pass** `kda05_reproduction/dsa_decode_fwd/opus-4.8/dsa_triton_v14`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：上游源码匹配先前失败版本；修复后通过不能转移。 证据：`../paired-repair-specs.json`。
- **repaired_pass** `kda05_reproduction/dsa_decode_fwd/opus-4.8/dsa_triton_v15`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：上游源码匹配先前失败版本；修复后通过不能转移。 证据：`../paired-repair-specs.json`。
- **repaired_pass** `ablation/dsa_decode_fwd/t8_topk2048/fable-5.1-ncu-skill`：修复后精度及重复调用检查通过，已取消 invalid；见[修复验收](../ablation-repair-20261023/README.md)。
- **repaired_pass** `ablation/fa4_bwd/varlen_t32768_n13_h16/fable-5.1-ncu-skill`：修复后精度及重复调用检查通过，已取消 invalid；见[修复验收](../ablation-repair-20261023/README.md)。
- **repaired_pass** `ablation/gdn_fwd/t8192_n20/fable-5.1-ncu`：修复后精度及重复调用检查通过，已取消 invalid；见[修复验收](../ablation-repair-20261023/README.md)。
- **repaired_pass** `goal_full_shape/moe_fp8_fwd/opus-5/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`：[修复验收](../remaining-repair-20261023/README.md)。历史失败：19/19 workload 均出现 Triton launch 参数不匹配：要求 36 个，实际 27 个。 证据：`fullshape-r2-select.log`。

## 验证边界

- “完成复测”不代表全部通过，失败项不提供可接受的性能结论。
- 6 个 FA4 全 shape 目录仅完成固定 MHA/GQA/varlen 代表性输入的独立 Tri Dao 精度复核；原始完整 workload 数据集缺失，未达到原始全 shape 覆盖。不能以目录已测冒充全 shape 已测。
- MLA decode B16/B64 原始长度分布缺失，使用均衡合成长度；B1 固定总长度 408。GDN decode 使用两组随机输入。新增 GDN prefill 使用冻结输入及随机种子 0、42；若先遇失败则后续种子和计时停止。
- 不同评测协议的阈值不同：官方全 shape 保留对应 definition 的判据；新增单 shape 通常使用数学 reference、atol/rtol 均为 0.01、要求全部元素满足；FA4 backward 使用独立 Tri Dao 梯度，Top-k 要求精确成员集合及 padding。旧记录保留原有输入范围与判据，不能宣称任意输入均正确。
- 频率检查通过仅证明抽样检查通过，不证明每个计时瞬间频率固定。原始低频任务的性能未作为正常频率结果采用；部分正常批次存在 `sw_power_cap`，需结合实际时钟，不等同于异常热降频。
- 两批中断经平台错误信息确认是临时存储超限；续跑把临时目录及编译缓存移到共享盘。CUDA 13.2 双目标编译、原地 state 的测试适配和包内嵌源码问题均在测试入口修正，没有修改上游候选 kernel。
