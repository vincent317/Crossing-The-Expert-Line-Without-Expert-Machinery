# SM100 (B200) 架构关键特性分析

## 已知特性
1. **tcgen05.mma**：专用的 MMA 单元，bf16/f16 输入，f32 累加
   - 一次 MMA：128x128，需要分成多个硬件 MMA 操作
   - 输入从 smem（通过 descriptor），输出到 tmem
   
2. **TMA (Tensor Memory Accelerator)**：异步 gmem→smem 传输
   - 2D 张量复制，不占用 thread 带宽
   - 通过 mbarrier 同步
   
3. **Tmem (Tensor Memory)**：专用的 on-chip 存储
   - 用于 MMA 的输入/输出
   - 需要显式 alloc/dealloc
   - V2 分配了 512 列（S0/S1/O），V4 分配了 384 列

## 可能遗漏的优化点

### 1. Warp 专业化不够
当前：4 个 warp 做 softmax（各处理 32 行），1 个 warp 做 TMA+MMA

问题：softmax warp 在等待时 idle，MMA warp 在 softmax 时 idle

**可能的改进**：
- 让 softmax warp 也参与数据搬运？
- 或者：5 个 warp 都做 softmax（每个 ~25 行），MMA 用 tcgen05 的 cta_group 模式？

### 2. MMA 吞吐未满
一个 128x128 的 MMA 需要多久？
- 如果 tcgen05.mma 是 1 个 cycle/warp，那 128x128 需要多少 cycle？
- Ref 可能用了更大的 tile（256x128？）或者不同的 MMA 分块策略

**需要验证**：mma_bench.cu 的结果（job 1049863，但我没看到输出）

### 3. Softmax 真的是瓶颈吗？
V2 是 1.037 ms，假设：
- MMA (QK + PV)：理论最小时间
- Softmax：额外开销
- 同步/调度：额外开销

如果 MMA 本身就很慢（比如只有 30% 利用率），那优化 softmax 没用。

**需要验证**：cycle-level profiling（fa_fwd3.cu 的 FA_MODE=9）

### 4. Pipeline 深度
V2 用 3-stage K/V，但可能：
- 3-stage 的 prefetch distance 不够（或太多）
- Barrier 太多导致 stall
- TMA 带宽未饱和

**Ref 可能的策略**：
- 更激进的 prefetch（4-stage？）
- 或者完全不用 staging（直接 TMA→计算→下一个）
- 或者用 async copy 而不是 TMA

### 5. Causal Mask 的处理
当前：每个 K tile 都做完整的 128x128 MMA，然后 softmax 时 mask

**可能的优化**：
- 提前知道哪些 K tile 完全在 mask 外，直接跳过
- 或者用 warp-level 的 predicate 减少无效计算

对于 causal varlen：
- 第一个文档（L=4096）：前 32 个 Q tile，每个看到的 K tile 数不同
  - Q tile 0：只看 K tile 0
  - Q tile 31：看 K tile 0-31（全部）
- 如果每个都做 full loop，浪费很大

**Ref 可能的策略**：
- Work assignment 时就考虑 causal（动态调整每个 CTA 的 K tile 范围）
- 或者用 persistent kernel + dynamic work stealing

## 最可能的根本问题

我怀疑 V2/V4 的瓶颈在于：
1. **MMA 利用率低**（可能只有 30-40%）
   - 原因：warp 专业化导致只有 1 个 warp 发射 MMA
   - tcgen05 可能支持多个 warp 协作，但我没用

2. **Causal mask 浪费**
   - 做了很多无效的 MMA（特别是短文档的早期 Q tile）
   - Ref 可能用了更聪明的 work distribution

3. **Tmem 容量限制**
   - V2 用了 512 列（接近上限），但可能 ref 用了更紧凑的布局
   - 或者 ref 用了 register tiling 而不是 tmem？

## 下一步行动

1. **等 v4 结果**：如果 < 0.5 ms，说明 QTILE=128 + KTILE=128 是对的
2. **等 mma_bench 结果**：看 tcgen05.mma 的理论峰值
3. **如果 v4 > 0.6 ms**：
   - 写 SUMMARY.md 承认无法达到目标
   - 分析根本瓶颈（可能需要看 ref 的 profiling，但题目禁止）
   
或者：
4. **尝试完全不同的策略**：
   - 多个 warp 协作做 MMA（cta_group）
   - Split-K（多个 CTA 处理同一个 Q tile）
   - Persistent kernel + 更细粒度的 work stealing

但时间不够了（还剩 ~10 轮），且这些都是 high-risk 的重写。
