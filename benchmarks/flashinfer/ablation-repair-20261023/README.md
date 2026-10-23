# 三项消融实验修复验收

2026-10-23：仅修复下列三项，均通过 B200 精度回归并取消 invalid。该批次未修改其余 17 项；后续已完成[剩余修复](../remaining-repair-20261023/README.md)。逐目录当前状态仍以[coverage.json](../remainder-03413a0-20261022/coverage.json)为准；原始提交的失败证据保留。

## 修复及精度

- `ablation/dsa_decode_fwd/t8_topk2048/fable-5.1-ncu-skill`：纠正默认数据发布与同步配置，关闭不安全的 tagged partial 路径。种子 0、42 各连续调用 100 次，每组首尾检查全部输出，通过原有 atol/rtol=0.01、全部元素满足的标准。任务 `tasks/1340628`。
- `ablation/fa4_bwd/varlen_t32768_n13_h16/fable-5.1-ncu-skill`：dV 读取 dO 前等待加载完成，且等 dP、dV 两个消费者均完成后才释放 dO 缓冲区。两组冻结输入各连续调用 100 次，每组首尾检查三个梯度，通过独立 Tri Dao/PyTorch 误差判据。任务 `tasks/1340629`。
- `ablation/gdn_fwd/t8192_n20/fable-5.1-ncu`：直接链接驱动的 tensor-map 编码入口，消除未经检查的入口查询带来的空指针；flag warp 统一使用 lane 0 的推进上限，避免并发读取 staging 游标导致各 lane 的游标分歧。同步修正 watchdog 记录的读取步长。冻结输入及随机种子 0、42 各连续调用 100 次，每组首尾检查 output 和 state，通过原有 atol/rtol=0.01、全部元素满足的标准。任务 `tasks/1340630`。

回归没有放宽阈值或删除输出检查。700 次重复调用均完成；精度比较在各组第 1 和第 100 次执行，并非逐次比较。每项另执行三轮性能测量，每轮热、冷 L2 各 50 次。源码哈希、逐输出精度指标及原始计时样本保存在对应 `repair-*-r1-results.json`，发布源码已逐文件匹配哈希。

修复前失败证据：DSA 见 [paired-repair-specs.json](../paired-repair-specs.json)；FA4 见[流水线诊断](../clock-guarded-20261022/fa4-pipeline-diagnosis.md)；GDN 本轮原版 watchdog 复现停滞见 [gdn-watchdog.log](gdn-watchdog.log)。旧版失败仍然有效，修复后通过只适用于新源码及上述输入范围。

## 性能

以下为三轮 kernel-span 中位数的范围，单位 µs；是 profiler 中首个 kernel 起点至最后一个 kernel 终点的跨度，不是 Python 调用端到端耗时。

- DSA：热缓存 **7.9035–7.936**；冷 L2 **9.376–9.440**。
- FA4 backward：热缓存 **1406.238–1470.078**；冷 L2 **1535.710–1576.830**。
- GDN：热缓存 **90.064–90.8965**；冷 L2 **92.8485–93.152**。包含 memcpy 等 GPU 活动的跨度另存于 `activity_span_us`，不应与 kernel-span 混用。

三项任务均在 B200、CUDA 13.2 环境执行。前后负载 smoke 的采样全部为 1965 MHz、无激活限频原因，正常频率门槛检查通过。这仅证明采样检查通过，并非每个计时瞬间的连续频率保证。原始版未通过精度或不能结束，因此本轮不计算修复前后加速比，也不把跨卡、跨协议结果当作同条件 A/B。

## 证据与复现

`repair-*-r1.log` 为原始集群日志，包含任务环境、精度退出码、前后频率检查和结果归档。`repair-*-r1-entry.sh` 为实际执行入口，`repair-*-r1-job.json` 保存任务 ID。入口中的 base64 载荷包含实际测试源码及修复源码；结果 JSON 从日志归档解码得到。

`test_ablation_repairs_gpu.py` 依赖已固定的审计运行环境，不是从任意干净机器直接运行的独立测试。入口使用共享盘上已有的 upstream、官方 definition、冻结输入、Tri Dao reference cache 和审计工具（`official`、`paired`、`metrics`、`kernel_span`、`run_gdn`、`tridao`）。这些外部依赖路径保留在入口中；本仓库没有附带全部数据。具备该环境时，执行对应入口即可重建此次验收；共享盘缺失时不能仅靠本报告复现。

GDN 编译需链接 `-lcuda`；三项实际 nvcc 参数在回归测试内。GPU 测试通过 `ABLATION_REPAIR_CASE` 选择目录；本地未配置 GPU 环境时显式跳过，不将跳过算作通过。
