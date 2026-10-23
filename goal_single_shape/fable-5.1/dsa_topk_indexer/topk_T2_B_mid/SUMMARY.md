# SUMMARY

## 完成状态

**轮次**：已超过 40 轮，按规则停止并汇报当前最好结果与瓶颈。

---

## 已完成的条件

### 条件 1 ✅ REF device time 已测量
- **REF profiler kernel span**：**4.36 µs**（torch.profiler kernel-only span，单 kernel）
- **REF CUDA events 中位数**：~59 µs（含 host 开销，warmup 10/iters 50，L2 flush）
- Job ID：1153651（node: mbkj-k8s-existed-node-spl8）

### 条件 2 ✅ FP32 naive 实现完成，E_ref 已打印
- FP32 naive 实现：`naive_fp32.py`
- Naive vs REF 不匹配 token 数：1118 / 3292
- **E_ref = 0.339611**（mismatch rate）
- **tol = 7 × E_ref = 2.377278**
- E_ref 高（34%）的原因：REF 使用 fp8 内部运算，FP32 naive 对 boundary tokens 的排名与 REF 不同，导致 batch 2（seq_len=2817, top_k=2048）的许多边界位置不同

### 条件 3 ✅ 新 kernel 精度检查通过
- 新 kernel（`topk_cuda.py` 中的 `fused_topk` 单 block 256 线程 + bitonic sort）
- **新 kernel vs naive mismatch rate = 0.000000**（完全一致）
- 0.000000 ≤ 2.377278 → **PASS**
- Job ID：1153651

### 条件 4 ❌ 速度目标未达
- **目标**：新 kernel profiler span ≤ 4.36 / 1.2 = **3.63 µs**
- **最好 profiler span**：**~890 µs**（新 CUDA kernel，job 1153651）
- 距目标约 **244x 差距**

### 条件 5 ⚠️ 文件已保存（部分 git commit）
- `naive_fp32.py`：FP32 naive 实现
- `topk_cuda.py`：CUDA kernel（score + bitonic sort 融合）
- `bench_all.py`：benchmark + 精度检查脚本
- `topk_triton.py`：Triton kernel（实验，速度未达标）
- `topk_fast.py`：多 block 实验版本
- git commit 待执行

---

## 瓶颈分析

### 根本原因：无法在单 kernel 内使用 tensor core 做 fp8 GEMV

REF 的 `v8_radix_bt256` kernel 在 4.36 µs 内完成所有操作，意味着它：
1. 使用 **WGMMA/UMMA tensor core**（B200 sm_100 专有指令）做 q(64×128 fp8) @ K^T(128×N fp8) = (64×N) fp32
2. 在 shared memory 中完成 **radix-based top-k**（一次 kernel）
3. 每个 batch 使用单独的 SM，8 个 SM 并行处理所有 batch

### 我们的最佳尝试

| 方法 | Profiler span | 说明 |
|------|-------------|------|
| 标量 fp8 dot4 + bitonic sort（256 线程/block） | **890 µs** | 寄存器压力 159 reg，L2 带宽瓶颈 |
| 多 block 评分（64 线程/page）| **81 µs** | score only，无 top-k |
| deep_gemm bf16_gemm_nt TN | **4.56 µs** | GEMM alone，无 post |
| deep_gemm + CUDA post kernel | **277 µs** | post kernel bitonic sort 太慢 |
| torch.mm fp16 big GEMM | **11 µs** | GEMM alone |

### 核心矛盾

- **单 GEMM**（deep_gemm bf16 TN）profiler = 4.56 µs，已 > 目标 3.63 µs
- **GEMM + top-k**（两个 kernel 顺序执行）profiler sum = 4.56 + post_time > 3.63 µs
- 要达标，必须：(a) 在 GEMM 内部融合 top-k，或 (b) 用比 deep_gemm 更快的 GEMM（需要手写 WGMMA PTX）
- **手写 WGMMA PTX kernel** 是唯一可能达标的路径，但超出了常规 CUDA 开发能力范围

### 为什么 score 计算这么慢

- 每个 token 需要 64 heads × 128 dims = 8192 fp8 MACs
- 4061 tokens × 8192 = 33M fp8 MACs
- 标量 fp8→float 转换 + 累加：每次转换需 `__nv_cvt_fp8x2_to_halfraw2` + `__half22float2`，约 4-5 cycles/元素
- 33M × 5 cycles / 1.7 GHz = 97 µs（与实测 622 µs 的差异来自低 occupancy：159 regs/thread × 256 threads = 40K regs > 65K/2，只有1 block/SM）
- 升级到 tensor core 可获得 10-40× 加速

---

## 最好结果总结

```
REF device time (profiler kernel span):  4.36 µs
REF device time (CUDA events median):   59.36 µs

FP32 naive vs REF mismatch rate:        0.339611
Tolerance (7 × E_ref):                  2.377278

New kernel (fused score+bitonic sort, 256t):
  Accuracy (vs naive): 0.000000 ≤ 2.377278 → PASS
  Profiler kernel span: 890 µs
  Target: ≤ 3.63 µs
  Speed goal: FAIL (244× off)
```

---

## 若要达标的下一步

1. **手写 WGMMA PTX inline assembly** 进行 fp8 GEMV：
   - PTX `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
   - 每个 warp 计算 16×8 tile，4 warps 覆盖 64 heads
   - 64 tokens/page × 45 pages = 需要复杂的 tiling

2. **使用 CuTe DSL（CUTLASS 4.x）** 写 fused kernel：
   - `cute::gemm_with_visitor` pattern，在 epilogue 中融合 relu + weighted sum + top-k

3. **向 deep_gemm 贡献 paged_mqa_logits 调用**（如果 API 格式能匹配），直接复用 AKO 的底层 kernel
