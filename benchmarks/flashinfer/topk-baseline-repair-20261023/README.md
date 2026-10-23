# Top-k 基线严格精度修复

本轮针对 AKO fast_split_v6、AKO v8 与 Dogacel 三种历史基线。原始快照与修复版分别记录，历史测量不被覆盖。最终逐项状态、耗时和原始证据入口见 [coverage.json](coverage.json)。

本轮已完成：原版 18 项中 9 项通过、9 项失败；修复版 **18/18 项全部通过**，共 1800 次重复调用均与官方 FP32 成员集合一致，计时后检查也全部通过。运行前后频率门槛通过，所有入选负载采样均达到 1900 MHz；存在 sw_power_cap 记录，不表述为完全没有功耗限制。

冷 L2 kernel span（三轮中位数的均值，包含两种权重分布）：

- T1：三种基线约 1.44–1.67 µs。
- T2：fast_split_v6 30.51–30.92 µs；AKO v8 17.39–17.70 µs；Dogacel 15.24–24.46 µs。
- T3：fast_split_v6 52.17–52.55 µs；AKO v8 24.24–24.49 µs；Dogacel 18.02–27.43 µs。

完整 FP32 选择增加了工作量，不能沿用原版约 4 µs 的 T2 AKO 数字。热缓存、逐轮样本及精度数据见原始结果。

## 修复内容

三种基线原先在 Top-k 选择前将分数压缩成 BF16；边界附近不同的 FP32 分数因此可能落入同一个量化桶，无法保证 FP32 成员集合一致。修复保留 FP32 分数，radix 选择检查完整 32 位。

AKO 保留 fast_split_v6 的 64 线程和 v8 的 256 线程配置，扩展为四轮字节筛选，并将 CUDA kernel 放到当前流执行，正确参与图捕获。Dogacel 保留粗筛与候选缓冲区，增加三个细化字节的筛选；每轮读取前缀计数后同步，再清空直方图，避免线程之间的读取与清空竞态。

修复版作为独立版本使用，不再视为未经修改的历史基线。重建源码内保留的历史注释、加速比与调优结论不适用于修复版本。

## 验收方法

沿用历史任务 1074336 的真实分页表与长度，三个 shape 各测非负均匀、有符号两种权重分布。对每种基线与输入配置检查与官方 FP32 reference 的全局 token 成员集合和 -1 padding 一致，重复调用 100 次，并在性能测量后再次检查。

严格测试先对未修改源码执行，再对修复版执行；判据不变。GPU 使用 B200，运行前后至少 90% 的负载样本 SM 时钟达到 1900 MHz，且负载样本不少于 10 个。每项性能记录包含三轮热缓存、冷 L2 测量，每轮各 50 次 CUPTI kernel span。该耗时包含 kernel 间隔，不是 Python 端到端耗时；原版与修复版不是同卡交替配对，不能据此声称精确加速比。

结论仅限这 18 项配置。严格成员集合判据比历史分数容差更严格；原版失败不等于历史赛事判据失败。任意其他输入或分数完全相等时的成员选择不在本轮结论范围内。

## 使用修复版本

[apply_repairs.py](apply_repairs.py) 从仓库已有的历史 payload 恢复源码，依据 [repair.json](repair.json) 校验原始 SHA256，再应用三个补丁，最后校验修复后的 SHA256。目标目录必须不存在，避免覆盖现有源码。命令最后的参数为要创建的目标目录：

```bash
python benchmarks/flashinfer/topk-baseline-repair-20261023/apply_repairs.py /tmp/topk-fixed-baselines
```

补丁分别为 [fast_split_v6](ref_ako_fs6-fp32.patch)、[AKO v8](ref_ako-fp32.patch)、[Dogacel](ref_dogacel-fp32.patch)。GPU 测试入口为 [test_topk_baselines_strict_gpu.py](test_topk_baselines_strict_gpu.py)，运行依赖与固定共享数据路径见每批次 entry；它不是全新机器安装脚本。
