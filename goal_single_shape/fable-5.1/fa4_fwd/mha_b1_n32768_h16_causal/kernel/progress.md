# FlashAttention Varlen 实现进展

## 目标
- 参照：flash_attn_varlen_func，device time 0.5252 ms（job 1049051）
- 目标：≤ 0.5252 / 1.2 = **0.4377 ms**

## 已完成的版本

### V1 (QTILE=128, KTILE=128, 基础两阶段)
- 结构：5-warp (4 softmax + 1 MMA/TMA)
- Pipeline：2-stage K/V，分离的 QK 和 PV 循环
- 结果：**未测试**（初版，已被 v2 替代）

### V2 (QTILE=256, KTILE=64, 3-stage + QK lookahead)
- 改进：
  - QTILE=256 减少 work-item 到 ~820
  - 3-stage K/V pipeline
  - QK lookahead（提前一个 tile 发射 QK MMA）
- Smem：~230 KB（接近 B200 上限）
- 结果：
  - 精度：maxerr=0.009 < tol=0.063 ✓
  - 性能：**1.037 ms**（device time），speedup **0.486x**（需要 1.2x）
  - 瓶颈：比目标慢 2.4 倍

### V3 (persistent kernel, 动态调度)
- 改进：用 atomic counter 做 LPT 调度，减少 tail latency
- 结果：**超时/未完成**（可能 atomic 开销抵消了收益）

### V4 (QTILE=128, KTILE=128, lazy O rescale)
- 改进：
  - 回到 QTILE=128（增加并行度到 ~1640 work-items）
  - KTILE=128（减少 K tile 数量，减少循环开销）
  - Lazy O rescale（阈值 2^8，减少 rescale 次数）
  - Fold softmax scale into exp2
- Smem：~200 KB
- 结果：**排队中**

## 理论分析

### 计算量
T=32768, H=16, D=128, 文档长度 L_max=8192
- FLOPs ≈ 2 × T × H × L_avg × D × 2 = 551 GFLOPs
- B200 @ 2.6 GHz, 128 SM, 2048 FP16 ops/clk/SM
  理论峰值 = 128 × 2048 × 2.6e9 × 2 = **1363 TFLOPs**
- 最小时间 = 551 / 1363000 = **0.404 ms**

### Ref 的 0.5252 ms 意味着什么？
- 实际效率 = 551 / (0.5252 × 1363) = **76.9% 峰值**
- 这是极高的效率，说明 ref 做到了：
  1. 几乎完美的 pipeline（TMA、MMA、softmax 重叠）
  2. 极小的循环/调度开销
  3. 可能用了 online softmax（省内存带宽）

### V2 的 1.037 ms 问题在哪？
实际效率 = 551 / (1.037 × 1363) = **39% 峰值**

可能的瓶颈：
1. **Work-item 太少**（~820）：不能充分利用 128 SM
   - 平均每个 SM 只有 6.4 个 CTA
   - Tail latency 显著
2. **KTILE=64 太小**：导致 K tile 数量多（8192/64=128 tiles）
   - 每个文档需要循环 128 次
   - 循环开销累积
3. **Softmax 串行化**：4 个 warp 各处理 64 行，但没有真正并行
4. **Pipeline 气泡**：3-stage 的同步点可能有 stall

### V4 的预期改进
- Work-items 增加到 ~1640（2 倍）
- K tile 数量减少到 64（一半）
- Lazy rescale 减少计算
- **预期**：~0.6-0.7 ms（1.5-1.7x ref，仍然不够）

## 下一步方向

### 如果 V4 < 0.5 ms ✓
直接成功！

### 如果 V4 在 0.5-0.6 ms
尝试 V5：f16x2 packed softmax
- 用 ex2.approx.f16x2 和 cvt.rn.f16x2.f32
- 理论加速 ~1.5-2x softmax
- 预期总加速 ~1.1-1.2x

### 如果 V4 > 0.6 ms
需要更根本的改变：
1. **Online softmax**（FA2 风格）
   - 省掉 S tmem 存储（512 列 → 128 列）
   - 但实现复杂度高
2. **更大的 QTILE**（如 QTILE=256, KTILE=128）
   - 但 smem 超限，需要 1-stage K/V
3. **Split-K** 或其他并行策略
   - 增加并行度，但引入额外的 reduce pass

## 当前状态（轮次 ~30）
- V4 正在测试
- Probe (f16x2 吞吐) 正在测试
- 已有 ref baseline、naive fp32、v2 完整数据
- 距离目标还差 ~1.8-2x 性能

## 风险
如果 V4 + V5 仍然不够，可能需要：
1. 重新设计 pipeline（online softmax）
2. 或者接受无法达到 1.2x 的结果，写 SUMMARY.md 说明瓶颈

时间预算：还剩 ~10 轮，需要在 V4/V5 结果出来后快速决策。
