# Softmax 优化策略

## 当前瓶颈
V4 虽然 MMA 吞吐达到 100%（KTILE=128），但 softmax 占比可能仍然过高。
每个 Q tile (128 行) 需要 4 个 warp 各处理 32 行，每行做：
- max reduction（遍历 K tiles）
- exp2 + sum（遍历 K tiles）
- rescale O（如果 max 变化）
- 写回 P

## 优化方向：packed f16x2 exp2

### 原理
- 当前：`ex2.approx.ftz.f32` 每 cycle 每 thread 1 个 f32
- 优化：`ex2.approx.f16x2` 每 cycle 每 thread 2 个 f16
- 前提：softmax 中间计算用 f16 足够（最终 P 要转回 bf16 给 V）

### 实现步骤
1. S tmem 保持 f32（QK 是 f32 累加器）
2. max/sum 用 f32（避免溢出）
3. exp2 阶段：
   - 从 S tmem 读 f32
   - 两两打包转 f16x2: `cvt.rn.f16x2.f32`
   - `ex2.approx.f16x2` 计算
   - 累加到 f32 sum
4. P 写回 smem 时转 bf16（给 V MMA）

### 预期收益
- exp2 吞吐翻倍
- 如果 softmax 占总时间 30%，且 exp2 占 softmax 的 50%，总体提升 ~7.5%
- v4 如果是 0.77 ms，v5 可能到 ~0.71 ms（仍不够，但方向对）

### 风险
- f16 精度可能不够（需要实测误差）
- cvt 和 unpack 开销可能抵消收益

## 如果还不够：更激进的方案

### A. Online softmax（FA2 风格）
不存 S，QK 算完立刻做 softmax 并与 V 相乘更新 O。
- 省掉 S tmem（256 列）
- 省掉 P smem（32768 B）
- 可以用更大的 K/V staging buffer
- 但实现复杂，需要重写整个 pipeline

### B. 混合精度 MMA
QK 用 tf32（速度同 bf16，精度更高），PV 用 bf16。
- B200 tf32 MMA 和 bf16 速度一样
- 可能减少 softmax 数值范围，降低 rescale 频率
- 改动小

### C. 合并 K tile
当前每个 K tile 都要 max/sum，如果能 2 个 K tile 合并处理，
softmax 循环减半。但需要更大的 S tmem（512 列）或 online 计算。
