# 基线复测（2026-10-23）

本轮覆盖仓库 11 个算子家族的已识别基线。逐项结论以 [coverage.json](coverage.json) 为准；统计单位是“基线实现 × 输入配置”，Top-k 的两种权重分布分别计数，重复种子不另计数。此前 135/135 是候选实现目录统计，不能据此推断基线全部通过。

本轮 250 项基线验收中，241 项通过，9 项 Top-k 严格成员集合检查失败；另有 6 项候选对照检查全部通过。17 个入选批次均已收齐原始结果，没有尚在运行的验收项。

## 覆盖范围

- 固定 expert：GDN prefill 100 个冻结 workload、DSA 23 个 workload、MoE 19 个 workload。
- FlashKDA：3 个 KDA 配置，检查输出和最终状态。
- FA4：前向 12 个公布配置，反向 3 个代表配置（MHA、GQA、varlen）。
- 分页注意力：GQA decode 3 个配置、MLA prefill 3 个配置。
- Top-k：3 个历史基线 × 3 个 shape × 2 种权重分布，共 18 项；另有 6 项当前候选对照检查。
- AKO 锚点：GDN prefill 3 个、DSA 3 个、GDN decode 54 个配置。
- MLA decode：AKO 和 FlashInfer FA2 各 3 个配置，共 6 项。

基线源码未改。AKO 使用公开仓库 [AKO4X](https://github.com/TongmingLAIC/AKO4X/tree/ce030ebd855dffb7ba6bdaef7ba7504a92cb9393) 的固定快照，分别选择 cuda_graph_v5、hybrid_pdl_v2、cuda_bv32_register_resident、triton_split_kv_2d_reduce_pdl_rw2。固定 expert 的源码哈希已与此前归档核对。Top-k 基线从历史任务 1074317、1074336 的原始 payload 恢复，包含 AKO fast_split_v6、AKO v8、Dogacel；每份结果保存实际源码 SHA256。

## Top-k 精度与性能

18 项基线配置中 9 项通过严格检查、9 项未通过。判据为与官方 FP32 结果的全局 token 成员集合及 -1 padding 一致，并重复调用 100 次。该判据严于历史的分数容差检查，不能将本结论直接表述为“历史赛事判据失败”。原始结果同时保留排序后分数的最大绝对误差。

- T1：三种基线在两种分布均通过 100/100 次。
- T2 非负均匀权重：三种基线均通过 100/100 次。
- T2 有符号权重：fast_split_v6 和 v8 均为 0/100；Dogacel 为 16/100。
- T3 非负均匀权重：三种基线均为 0/100。
- T3 有符号权重：fast_split_v6 和 v8 均为 0/100；Dogacel 为 8/100。
- 当前候选的全部 6 项配置均通过 100/100 次，共 600 次重复检查。

基线源码中存在先将分数量化为 BF16 再选取成员的路径，与边界排名或同分选择差异相符；尚未证明这是所有失败的唯一原因。失败配置不计入有效性能比较，报告不修改基线以迎合严格判据。

仅对通过精度检查的配置执行同卡 AB/BA 配对计时。T2 非负权重的候选冷缓存 kernel span 约 3506 µs（3.51 ms），AKO fast_split_v6 约 4.128 µs、v8 约 4.005 µs，候选显著更慢，不能声称加速。T1 候选也未显示加速。T3 基线未通过严格精度检查，不提供有效加速比。详细分布、每轮样本与配对结果见 [Top-k 原始结果](baseline-topk-r2-results.json)。

## 计时与频率

使用 B200。每项性能记录有 3 轮热缓存和冷 L2 测量，每轮各 50 个样本。记录 CUPTI kernel span，包含 kernel 之间的间隔；它不是 Python 端到端耗时。不同任务的候选与基线不能直接相除作为正式加速比；本轮仅 Top-k 的有效配置提供新的同卡配对比较。

每个入选批次均检查运行前后 GPU 频率：至少 10 个负载采样点，且至少 90% 的负载点 SM 时钟不低于 1900 MHz。原始日志和 coverage 中保留采样统计。部分批次出现 sw_power_cap，因此只能说通过上述频率门槛，不能说完全没有降频或所有瞬间都固定 1965 MHz。

## 精度与覆盖边界

MoE 沿用官方 atol=1、rtol=0.3、匹配比例至少 90% 的判据，不代表逐元素 0.01 精度。KDA、FA4 和分页测试使用独立参考或固定库参考，并检查计时前后结果；具体容差见随附测试入口。多种子配置使用种子 0 和 42。

FA4 原始全 shape 数据集仍不可用，反向仅覆盖上述 3 个代表配置；不能声称全部历史全 shape 已覆盖。部分 GQA/MLA 原始 ragged 分布缺失，沿用明确的重建输入。MLA decode 的 b1_kv408、b16_kv9657、b64_kv75145 使用合成均衡长度，不能替代未知的原始分布。GQA 使用 num_splits=1、pack_gqa=None，未声称恢复历史全部调优参数。

测试入口复用固定的共享运行环境、数据集与缓存；随附 entry 脚本可追溯运行配置，但不是独立的全新机器安装包。环境兼容性修正与测试适配均未改动基线源码。每个批次随附 job 元数据、entry、日志及原始结果，coverage 对这些文件记录 SHA256。候选版本、AKO 版本和逐条证据入口亦记录在 coverage 中。

AKO DSA 最终使用 TileLang 0.1.9、apache-tvm-ffi 0.1.10 和 CUDA 12.9，三个配置全部通过。先前 CUDA 13 缺少 CCCL 头文件的问题发生在编译阶段，不计作精度失败。最终证据见 [任务 1341819 结果](baseline-ako-dsa_attention-r9-results.json) 及同名日志、entry。
