# FlashAttention Varlen BF16 实现总结

## 任务目标
实现 FlashAttention forward varlen packed causal MHA（bf16, head_dim=128）：
- 输入：13 个文档打包成 T=32768 tokens，max_seqlen=8192，H=16
- 参照：`flash_attn_varlen_func`，device time **0.5261 ms**（profiler kernel span）
- 目标：device time ≤ **0.4384 ms**（1.2x 加速）

## 完成情况

### ✅ 已完成项
1. **参照实现测时**（完成条件 1）
   - Device time: **0.5261 ms**（job 1049787，profiler kernel-only span）
   - CUDA events 中位数: 0.526 ms（runs/1051416: 0.5036 ms）
   - 终端打印证据：task output b49x8gwme, bmkwwulqj

2. **FP32 naive 慢实现**（完成条件 2）
   - 实现：纯 Python/PyTorch，逐文档处理，fp32 精度
   - Ref vs Naive 最大误差：**E_ref = 9.036303e-03**
   - 容差：**tol = 7 × E_ref = 6.325412e-02**
   - 终端打印证据：task output b49x8gwme

3. **新 kernel 实现与精度验证**（完成条件 3）
   - V2 kernel（QTILE=256, KTILE=64, 3-stage pipeline, QK lookahead, lazy O rescale）
   - 最大误差：**9.036303e-03 ≤ 6.325412e-02** ✓
   - 终端打印证据：task output bmkwwulqj

4. **源码保存与提交**（完成条件 5）
   - Kernel: `fa_fwd.cu`(v1), `fa_fwd2.cu`(v2), `fa_fwd3.cu`(v3), `fa_fwd4.cu`(v4)
   - Python 接口: `fa_fwd.py`, `fa_fwd2.py`, `fa_fwd3.py`, `fa_fwd4.py`
   - 测试脚本: `common.py`, `bench_ref.py`, `bench_new.py`, `check_v4.py`, `bench_v4.py`
   - 辅助工具: `smoke_umma.cu`, `probe.cu`, `mma_bench.cu`
   - 已 git commit（多次提交，包含所有版本）

### ❌ 未完成项
**性能目标**（完成条件 4）：
- V2 device time: **1.0320 ms**（profiler kernel span，job 1049863 / bmkwwulqj）
- Speedup: **0.486x**（需要 ≥1.2x）
- 目标：≤ 0.5261 / 1.2 = **0.4384 ms**
- 实际：1.0320 ms
- 与目标差距：**2.35 倍慢**
- 终端打印证据：task output bmkwwulqj

## 实现方案演进

### V1: 基础两阶段（QTILE=128, KTILE=128）
- 结构：5-warp CTA（4 softmax + 1 TMA/MMA）
- Pipeline：2-stage K/V，分离的 QK 和 PV 循环
- 状态：未测试（被 v2 替代）

### V2: 大 Q tile + 深度 pipeline（QTILE=256, KTILE=64）
- 改进：
  - QTILE=256 减少 work-item 到 ~820（每个 CTA 处理更多 Q）
  - 3-stage K/V pipeline（提前 2 个 tile prefetch）
  - QK lookahead（QK MMA 和 softmax 重叠）
- Smem: ~230 KB（接近 B200 的 232 KB 上限）
- 性能：**1.032 ms**（kernel span），**0.486x ref**
- 瓶颈分析：
  - 实际 MMA 效率：~39% 理论峰值
  - Work-item 不足：平均每 SM 只有 6.4 个 CTA
  - KTILE=64 太小：长文档（L=8192）需要循环 128 次
  - Causal mask 浪费：短 Q tile 做了大量无效 MMA

### V3: Persistent kernel + 动态调度
- 改进：用 atomic counter 实现 LPT 调度，减少 tail latency
- 状态：超时/未完成（可能 atomic 开销抵消收益）

### V4: 双向优化（QTILE=128, KTILE=128, lazy rescale）
- 改进：
  - QTILE=128 增加并行度到 ~1640 work-items
  - KTILE=128 减少循环次数到 64
  - Lazy O rescale（阈值 2^8）
  - Fold softmax scale into exp2
- Smem: ~200 KB
- 状态：**测试排队中**（GPU 资源受限）

## 性能瓶颈分析

### 理论极限
- FLOPs: ~551 GFLOPs
- B200 峰值: 1363 TFLOPs（128 SM × 2048 FP16/clk × 2.6 GHz）
- 理论最小时间: **0.404 ms**
- Ref 效率: **76.9%** 峰值（0.5252 ms）

### V2 的差距
- V2 效率: **39%** 峰值（1.032 ms）
- 与理论极限差距: **2.55x**
- 与 ref 差距: **1.96x**

### 根本原因（推测）

1. **Work distribution 不平衡**
   - Causal mask 导致不同 Q tile 的工作量差异巨大
   - Q tile 0 只需处理 1 个 K tile，Q tile 63 需要 64 个
   - V2/V4 都用静态分配，无法平衡

2. **MMA 利用率不足**
   - 只有 1 个 warp 发射 MMA（其他 4 个做 softmax）
   - MMA 单元可能在等待 softmax，或者 warp 专业化限制了吞吐
   - Ref 可能用了多 warp 协作或完全不同的 warp 分工

3. **Pipeline 气泡**
   - 3-stage prefetch 可能深度不够或同步点太多
   - Softmax 和 MMA 的重叠不完美

4. **Tmem 容量限制**
   - V2 用了 512 列（S0/S1/O），接近上限
   - 可能限制了更大的 tile 或更深的 pipeline

### Ref 可能的优势技术（推测，未验证）

1. **Online softmax（FA2 风格）**
   - 每个 K tile 算完 QK 立即做 softmax 和 PV
   - 省掉 S 的完整存储，减少 tmem 压力
   - 但实现复杂度高（需要 rescale O）

2. **动态 work stealing**
   - Persistent kernel + 细粒度任务队列
   - 考虑 causal mask，提前跳过无效的 K tile
   - V3 尝试了但可能实现不够优化

3. **更大的 CTA tile**
   - 可能用 QTILE=256 或 512（需要更多 warp 或不同的 softmax 策略）
   - 减少 CTA 数量，但每个做更多工作

4. **多 warp MMA 协作**
   - 让所有 warp 参与 MMA 发射（tcgen05 的 cta_group 模式？）
   - 提高 MMA 单元利用率

## 后续改进方向

### 如果有更多时间，可以尝试：

1. **V5: f16x2 packed softmax**
   - 用 `ex2.approx.f16x2` 和 `cvt.rn.f16x2.f32`
   - 理论加速 softmax 1.3-1.5x
   - 预期总加速: ~1.13x（不够达到目标）

2. **V6: Online softmax**
   - 实现 FA2 的 online softmax 算法
   - 省掉 S tmem（512 列 → 128 列）
   - 但需要处理 O rescale 的时机和同步

3. **V7: Causal-aware work distribution**
   - 预计算每个 (doc, head, qi) 的有效 K tile 范围
   - Work queue 按工作量排序（LPT）
   - Persistent kernel 动态分配

4. **V8: 多 warp MMA**
   - 研究 tcgen05 的 cta_group 和多 warp 协作
   - 让 5 个 warp 都参与 MMA 发射
   - Softmax 用 register 而不是 warp 专业化

### 风险评估
- V5 是渐进优化，预期收益 ~10-15%，不足以达到目标
- V6-V8 都是高风险重写，需要 5-10 轮迭代，且不保证成功
- Ref 的 76.9% 效率极高，可能用了专门的硬件特性或编译器优化

## 结论

本实现达到了：
- ✅ 功能正确性（精度 < 容差）
- ✅ 合理的性能（~50% ref，~39% 理论峰值）
- ❌ **未达到 1.2x 加速目标**

根本瓶颈在于：
1. **Work distribution**：causal mask 导致的负载不平衡
2. **MMA 利用率**：warp 专业化限制了硬件利用率
3. **架构理解不足**：可能遗漏了 SM100 的关键特性

达到 ref 的 76.9% 效率需要更深入的架构知识和可能需要数周的迭代。

## 附录：测试环境

- 硬件：B200（SM100，128 SM，232 KB smem/SM）
- CUDA：13.0（镜像 `model-align/sglang:v0.5.19`）
- 编译：`-gencode arch=compute_100a,code=sm_100a -O3`
- 参照库：flash-attn-4 4.0.0b19

## 文件清单

- `fa_fwd2.cu`：V2 kernel 源码（当前最优）
- `bench_v4.py`：Benchmark 脚本（profiler + CUDA events）
- `check_v4.py`：精度检查（vs naive fp32）
- `common.py`：数据生成和 naive 实现
- `PROGRESS.md`：详细进展记录
- 其他：`fa_fwd.cu`(v1), `fa_fwd3.cu`(v3), `fa_fwd4.cu`(v4), 以及各种测试脚本
