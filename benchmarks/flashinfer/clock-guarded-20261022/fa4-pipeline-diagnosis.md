# FA4 varlen 重复调用故障与修复

对象：`fa4_bwd/varlen_t32768_n13_h16/fable-5.1-ncu-skill`。修复已通过正确性、重复调用和正式性能验收。主报告当前为 83/83 项有效计时。

## 根因

dO 环形缓冲区由 TMA producer 写入，A2 用它计算 dP，B 用它计算 dV。旧实现有两个相关的同步缺口：

- B 只等待 P 就开始读取 dO；P 的就绪仅证明 Q/S 路径完成，不证明 dO 已加载。
- B 在自身 dV 完成后就通知 `bar_do_empty`，但 A2 仍可能读取同一份 dO。producer 因而可能提前复用该槽位，破坏加载/消费阶段的顺序。

设备异常证据来自 `tasks/1330680`：CUDA-GDB 捕获 `Warp Illegal Instruction`，触发指令为 `mbar_arrive_expect_tx`，内联于旧 `fa4bwd.cu:226` 的 dO 加载阶段；原始摘要见 [异常记录](fa4-exception-summary.log)。外层 `unspecified launch failure` 是这个设备错误在 CUDA 同步时的报告。

修复在 B 读取 dO 前等待对应 `bar_do_full`；将 `bar_do_empty` 的提交移到 `pds_ready` 之后。此时 A2 的 dP 已完成，B 的有序 commit 又保证自身 dV 完成，两个读者都退出后才允许复用槽位。改动见工作区 `experiment_kernels/ablation/fa4_bwd/varlen_t32768_n13_h16/ncu-skill/fa4bwd.cu`；正式源码哈希见 `fa4-pipeline-source-hashes.json`。

没有修改数学计算、Tri Dao 阈值或计时协议，也没有通过主机同步、保留所有输出或关闭 CUPTI 来绕过问题。诊断中试过的 TMA 退出等待未采用。

## 为什么原正确性验证能过

这是执行时序问题。单次调用的数值检查通过不意味着所有执行时序都安全。原实现既有完整计时偶尔成功，也有关闭 CUPTI、逐次同步仍失败的记录（`tasks/1330096`，见 `fa4-diagnosis.log`）。重新编译也不能消除问题（`tasks/1330478`，见 `fa4-build-ab.log`）。

精简入口下 memcheck 的 1000 次调用和短程 synccheck/racecheck 曾报告零错误；这些检查没有复现该间歇性故障，不能证明没有竞态。最初带依赖导入的 21 项 Sanitizer API 查询报错不作为 FA4 缺陷证据。

## 回归与验收

- `tasks/1330824`：原版和修复版均在 dO producer 同一位置注入 100000 ns 延迟，使用相同两项测试、相同 180 秒限时。旧版得到 `OLD_DELAY_FAIL`，未完成测试；修复版 500 次调用在 76.049 秒内完成，包含两个 seed、同步/连续两种模式的三项梯度检查，前后 smoke 通过。见 `fa4-pipeline-controlled-entry.sh` 和 `fa4-pipeline-controlled.log`。
- `tasks/1330792`：修复版在另一张 B200 上通过两项正常回归，共 14000 次调用，前后 smoke 通过。该任务较早的旧/新延迟测试调用次数不同，不用于等条件 A/B 结论；以 `tasks/1330824` 为准。
- `tasks/1330813`：在此前 CUDA-GDB 捕获故障的 GPU `ea91b590-3275-ee03-03bf-52dd3854fc5c` 上，修复版再次通过 14000 次正常调用。正常回归包含 10000 次立即丢弃输出的同步调用，及 seed 0、42 各 1000 次同步、1000 次连续调用，定期按原 Tri Dao 判据比较 dq/dk/dv。
- 同一正式任务重新执行独立正确性验收：seed 0、42 的 dq/dk/dv 六项检查全部通过，记录为 `failure-fa4-pipeline/accuracy.json`。
- 正式热缓存和冷 L2 各 50 次 CUPTI 采样，中位数分别为 **1483.412 µs / 1697.602 µs**。前后各有同卡 baseline 的三轮试验，前后 smoke 均通过；完整记录见 `failure-fa4-pipeline/`。
- 本地测试共 64 项：56 项通过、8 项 GPU 测试本地跳过。新增的两项 GPU 回归已在上述 B200 任务执行。报告来源一致性测试先失败、导入新验收证据后通过。

正式性能构建不包含延迟注入。相对旧轮冷缓存值的变化约 +11.2%，但旧轮与新轮不是同卡同次 A/B，不能将该差异全部解释为同步修复的性能成本。

## 频率与证据边界

正式任务的前后 smoke 负载样本均为 1965 MHz；运行期遥测有 31 个 1965 MHz 样本和 1 个 1822 MHz 样本，并记录一次功耗限制。遥测覆盖 baseline/候选等阶段，不能据此宣称每个计时窗口全程满频。此次 launch failure 的定位证据是设备 barrier 异常与管线同步缺口，不将它归因于降频。

历史失败、未采用的诊断尝试及所有提交入口继续保留。`fa4-pipeline-job-status.json` 记录本次任务终态，`entry-hashes.json` 记录提交脚本哈希。当前正确性证据汇入上级 `fa4_tridao.json`，性能证据汇入本目录及上级的 JSON/CSV；后续 `tasks/1331187` 已针对修复版完成同卡同进程 cold L2 交替采样：基线 1722.952 μs、候选 1513.698 μs，1.13824×，三轮倍数 1.13547 / 1.13342 / 1.14574；前后 smoke 均通过。最终配对表已替换旧记录，详见 `../paired-fa4-pipeline-provenance.json`。运行期遥测未采到利用率达到 50% 的样本，不能据此证明每个计时窗口恒定满频。
