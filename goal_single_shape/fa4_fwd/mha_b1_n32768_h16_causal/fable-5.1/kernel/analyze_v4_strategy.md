# V4 分析与后续策略

## V4 改动（基于 v2）
- QTILE: 256 -> 128
- KTILE: 64 -> 128
- 每个 work-item 从 256x64 tile 变成 128x128 tile
- work-item 数量翻倍（从 ~820 到 ~1640）
- smem: 从 262KB 降到 193KB（安全）
- MMA: N=64->128，理论吞吐提升 33%

## 预期性能
- v2 实测: 1.03 ms (MMA 67% 吞吐 + KTILE=64)
- v4 理论: ~0.77 ms (MMA 100% 吞吐)
- 目标: 0.418 ms (ref 0.5018 / 1.2)

## 如果 v4 未达标（很可能），瓶颈在哪？
1. **Softmax 占比过高**: 每个 Q tile 都要做完整 softmax（4 warps），v4 的 tile 数翻倍意味着 softmax cycles 翻倍
2. **Pipeline 效率**: 2-stage K/V 可能导致 MMA 等待
3. **Work 分配**: tile 数太多，尾部不均衡

## 后续优化方向（按优先级）

### A. 降低 softmax 开销（最有效）
- **lazy rescale**: 累积 max/sum 不每次都 rescale O，只在需要时做（v2 已有）
- **packed f16x2 softmax**: exp2 用 f16x2 packed 指令，吞吐翻倍
- **更大 QTILE**: 如果能做到 QTILE=256 + KTILE=128，softmax 次数减半（但 smem 会爆）

### B. 优化 Pipeline
- **3-stage K/V**: 需要削减别的 buffer（如合并 S 两个 stage）
- **prefetch**: 提前发射下一个 tile 的 K/V TMA

### C. 混合策略
- **长文档用大 tile，短文档用小 tile**: 代码复杂度高
- **persistent kernel**: warp-level 动态调度（v3 尝试过，太难调）

### D. 激进方案
- **online softmax**: 不存 S，边算边做 softmax（FA2 风格）
  - 需要重写整个 pipeline
  - 可能是唯一能达到 0.4 ms 的路径
  - 但实现复杂度高，风险大
