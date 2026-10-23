# FlashAttention Forward v4 - 总结报告

## 任务目标

实现 FlashAttention FWD (B=1, S=32768, H=32, D=128, bf16, non-causal)，达到参照实现 (flash-attn-4) 1.2x 速度。

**目标 device time**: ≤ 15.63 ms / 1.2 = **13.03 ms**

## 最终结果

### 1. 参照实现 (ref) device time ✅
- **flash-attn-4 (v4.0.0b19)**: 15.63 ms (profiler kernel-span, job 1049051)
- CUDA events 中位数: 15.7 ms (warmup 10, iters 50)

### 2. Naive fp32 实现与容差 ✅
- **E_ref** (ref vs naive_fp32): 2.063e-04
- **tol**: 7 × E_ref = **1.444e-03**

### 3. v4 kernel 精度 ✅
- **max_err** (v4 vs naive_fp32): 1.807e-04
- **结果**: 1.807e-04 ≤ 1.444e-03 **PASS**

### 4. v4 kernel device time ❌
- **CUDA events 中位数**: 16.19 ms
- **profiler kernel-span**: **15.71 ms**
- **speedup**: 15.63 / 15.71 = **0.996x** (目标 1.2x)
- **结果**: 15.71 ms > 13.03 ms，**未达标** (差距 17%)

### 5. 代码提交 ✅
- kernel 源码: `fa_kernel_v4.py`
- benchmark: `bench_v4.py`, `ablate_v4.py`, `micro_softmax.py`
- 公共模块: `common.py`
- Git commit: b05a978

## 实现架构

### Warp-Specialized Pipeline (10 warps)
```
- load partition    (1 warp):  TMA q/k/v into shared-memory ring
- mma partition     (1 warp):  tcgen05 MMAs (S=QK^T in TMEM, O+=PV)
- softmax × 2       (4 warp each): online softmax per 128-row half
- correction        (4 warp, default): rescale O in TMEM, TMA store
```

### Ping-Pong Scheduling
```
QK0(i+1) -> PV1(i) -> QK1(i+1) -> PV0(i+1)
```
两个 128-row 半块交替发射，MMA in-order 执行，S/P tile 可以在 PV(h,i) 发射后立即被 QK(h,i+1) 复用。

### 资源使用
- **TMEM**: 256 KiB (S: 2×128×128 fp32, O: 2×128×128 fp32)
- **SMEM**: ~48 KiB (Q: 2×128×128 bf16, KV ring: 3×128×128 bf16)
- **Registers**: maxnreg=128 (correction/softmax: 192 each, mma/load: 24 each)

## 瓶颈分析

### Ablation Study (job 1052070)
```
MODE 0 (full, MATH=0):        15.71 ms  ← 基准
MODE 0 (full, MATH=1):        16.84 ms  (packed math 更慢)
MODE 2 (no-softmax-math):     10.18 ms  ← softmax 贡献 ~5.5 ms
MODE 3 (no-mma, MATH=1):      13.26 ms  ← MMA 贡献 ~3.6 ms
```

**结论**: softmax partition 是流水线瓶颈 (5.5 ms / 15.7 ms = 35%)

### Micro-Benchmark (job 1052000, per 128×128 tile, 4 warps, 1 SM)
```
V0 (full softmax, MATH=0):        0.777 ms / 1024 iters = 0.76 µs/tile
V6 (load+fma+exp2+sum, no store): 0.654 ms              = 0.64 µs/tile
V8 (fp16x2 ex2.approx.f16x2):     0.647 ms              = 0.63 µs/tile
V9 (raw ex2.approx.f32 asm):      0.644 ms              = 0.63 µs/tile
V7 (load+fma+sum, no exp2):       0.193 ms              = 0.19 µs/tile
```

**结论**: exp2 指令占 (0.64 - 0.19) / 0.76 = **60%**，且 f16x2/asm 变体无法优化

### 尝试的优化方向

| 优化 | 结果 | 原因 |
|------|------|------|
| MATH=1 (fp16x2 exp2) | 16.84 ms (更慢) | 精度损失 + cvt 开销 |
| NUM_KV=2 | 16.34 ms (更慢) | TMA 流水线气泡增加 |
| 去掉分支 (v4_opt) | 15.89 ms (更慢) | 编译器本来就能优化 |
| 8-warp softmax | 1.02 µs (更慢) | 寄存器压力 + 通信开销 |
| BLOCK_N=256 | OOR | TMEM 容量不足 (需 384 KiB) |
| maxnreg=160 | OOR | 线程数超限 (需 512 > 384) |

## 根本瓶颈

**exp2 指令是硬件瓶颈**：
- 每个 softmax tile (128×128) 需要 2×128×128 = 32768 次 exp2
- 总共 256 tiles (S=32768 / BLOCK_N=128) × 2 halves = 512 tiles/CTA
- 148 CTAs × 512 tiles × 0.46 µs (纯 exp2 时间) ≈ 35 ms 理论下限
- 实际通过流水线重叠降到 5.5 ms，但仍是瓶颈

要达到 13.03 ms (-17%)，需要：
1. **减少 softmax 迭代次数**：BLOCK_N 加倍（需要更大 TMEM，B200 不支持）
2. **低精度路径**：FlashAttention-3 style fp8 softmax（精度风险）
3. **算法变更**：非 online softmax（需要两遍扫描，违反 FA 设计）

这些都需要重新设计 kernel 架构，超出当前调优范围。

## 完成度评估

| 条件 | 状态 | 证据 |
|------|------|------|
| 1. ref device time | ✅ | 15.63 ms (job 1049051, runs/1052070/log.txt) |
| 2. fp32 naive + tol | ✅ | E_ref=2.063e-04, tol=1.444e-03 (runs/1052070/log.txt) |
| 3. v4 精度 ≤ tol | ✅ | 1.807e-04 ≤ 1.444e-03 (runs/1052070/log.txt) |
| 4. v4 device time ≤ 13.03 ms | ❌ | 15.71 ms > 13.03 ms (runs/1052070/log.txt) |
| 5. 代码 commit | ✅ | commit b05a978 |

**结论**: 4/5 条件满足，性能差距 17%，受硬件 exp2 指令限制。
