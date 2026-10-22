# 理论分析：为什么 0.4 ms 这么难达到？

## 计算量估算

### MMA 计算量
- 总 tokens: 32768
- H=16, D=128
- 每个 token 对：QK 需要 D 次乘加，PV 需要 D 次乘加
- 但 causal mask：平均每个 token 只看一半的 doc

估算 FLOPs（粗略）：
- QK: 对每个 doc，L×L/2 对 tokens × 2D ops = Σ(L²/2 × 256)
- PV: 同样 Σ(L²/2 × 256)
- 总 FLOPs ≈ 2 × Σ(L²) × 128 ≈ 2 × 530M × 128 ≈ 135 GFLOPs

其中 Σ(L²) = 4096² + 217² + ... ≈ 16.8M + 47K + 67.1M + ... ≈ 530M

B200 BF16 峰值：5 PFLOPS → 0.027 ms（理论下界，不可能达到）

但实际 MMA 还要算 softmax（非 MMA），memory，所以 0.4-0.5 ms 是合理的。

## V4 为什么会慢？

假设 v4 达到 0.77 ms（预期），比 ref 慢 1.5×。可能原因：

### 1. Work 不均衡
- 我们有 1640 个 work-item，B200 有 216 SM
- 平均每个 SM 7-8 个 item
- 但 doc 长度差异大（217 vs 8192），尾部一定有 SM 空闲

Ref 可能用了更聪明的调度：
- Persistent kernel 动态分配
- 或者更大的 tile size 减少 item 数
- 或者 doc-level 并行 + thread-block 内 Q tile 循环

### 2. Softmax 开销
每个 Q tile 都要完整 softmax：
- max reduction（遍历所有 K tiles）
- exp + sum（再遍历）
- rescale O
- 写 P

我们的实现：
- v2/v4: lazy rescale（只在 max 变化时），已经比较优化
- 但仍然每个 K tile 要读写 S tmem

Ref 可能：
- online softmax（不存 S，边算边 softmax）
- 或者用了特殊的 warp-shuffle reduction（比我们的快）

### 3. Memory 层级
我们的 pipeline：
- Q: TMA → smem → tmem (via MMA)
- K/V: TMA → smem → tmem (via MMA)
- S: tmem → tmem (MMA) → tmem (softmax 读) → smem (写 P)
- O: tmem → gmem

Ref 可能：
- 更少的 tmem ↔ smem 搬运
- 更好的 bank conflict 避免
- 更激进的 prefetch

### 4. Tile size 选择
我们 v4: QTILE=128, KTILE=128
- 优点：MMA N=128 满吞吐
- 缺点：work-item 多（1640），softmax 次数多

Ref 可能：QTILE=256, KTILE=128 或更大
- 前提：他们用了 online softmax，不需要存 S（省 tmem）
- 或者用了 register blocking

## 结论

如果 v4 ≥ 0.6 ms，渐进优化（f16x2 exp2）不够，需要结构性改变：

### 方案 A：Online Softmax（优先）
不存 S，QK 算完立刻 softmax 并更新 O。
- 省 S tmem（256 列）→ 可以用更大 tile 或更多 K/V stage
- 省 P smem（32KB）
- Softmax 和 PV 可以重叠
- 实现：类似 FA2 的 "rescale-O-as-we-go" 策略

预期：0.4-0.5 ms（有机会达标）

### 方案 B：更大 Tile + Persistent Kernel
QTILE=256, KTILE=128，work-item 减半。
配合 warp-level 动态调度（不是 CTA-level，避免 v3 的 barrier 噩梦）。

预期：0.5-0.6 ms（可能不够）

### 方案 C：混合
短 doc（<1024）用小 tile，长 doc（>4096）用大 tile 或 persistent。
但代码复杂度爆炸。

## 下一步
等 v4 结果：
- 如果 < 0.5 ms：尝试 v5 (f16x2)，可能够
- 如果 0.5-0.6 ms：必须做 v6 (online softmax)
- 如果 > 0.6 ms：online softmax 也可能不够，需要更激进的方案
