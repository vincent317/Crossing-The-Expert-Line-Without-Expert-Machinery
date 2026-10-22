# SUMMARY.md

## 完成状态

**已满足的条件：**

### 条件 1 ✓ — REF device time 已测量
- 节点：mbkj-k8s-existed-node-kojb / vgwc（ai4ai2 集群 B200）
- 口径：torch.profiler kernel-only span（`_dsa_score_kernel`）
- **REF device time = 0.42 µs/call**（profiler，10 次迭代）
- CUDA events 中位数：~52 µs（包含 host 侧开销）
- 注：任务描述中提到历史测量值 5.0 µs（jobs 1074317/1074336），本次 B200 实测为 0.42 µs（约 10× 更快）

### 条件 2 ✓ — fp32 naive 已实现，E_ref 已打印
- 实现：`naive_fp32()` 函数（PyTorch 向量化）
- E_ref = **204.558270**
- tol = 7 × E_ref = **1431.907887**
- REF vs naive 集合匹配率：0.9760

### 条件 3 ✓ — 新 kernel 精度满足 ≤ tol
- v3 kernel（score_kernel + topk_kernel 两阶段）
- E_mine = **0.000000**（与 naive 完全一致）
- accuracy_pass = True（0.000000 ≤ 1431.907887）

### 条件 4 ✗ — 速度目标未达到
- 目标：≤ 0.42 / 1.2 = **0.35 µs**
- 最佳实现（v3）：**13.41 µs**（score_kernel 9.59 µs + topk_kernel 3.82 µs）
- Speedup = 0.031x（远低于 1.2x 目标）

### 条件 5 — 代码已保存，等待 git commit

---

## 当前最好结果

| 版本 | 方法 | 精度 | device time |
|------|------|------|------------|
| v1/v2 | 单 kernel，一线程/token，串行 64 head | ✓ | 217–314 µs |
| v3 | 两 kernel，page 粒度并行 + warp shuffle 4-lane dot | ✓ | 13.41 µs |
| ref | 黑盒（AKO fast_split_v6 kernel） | — | 0.42 µs |

---

## 性能瓶颈分析

### 为什么 v3 慢 ~19×？

1. **score_kernel 计算模型**：每个 CUDA block 处理一个 (batch, page) pair（64 tokens），256 threads，4 threads/token。每 thread 处理 32 维，warp shuffle 合并 4 lane→一个 head 的 dot product。64 heads 串行循环。
   - 每 block：64 tokens × 64 heads × 128 dims = 524,288 fp8 ops
   - 总 blocks：~191 个 pages → ~191 × 256 threads 同时运行

2. **根本原因**：per-head 循环串行（64 次 warp shuffle），加上 8192 字节 sh_q 重复读取（每 token 每 head 都需要读取 32 字节），shared memory 带宽成为瓶颈。

3. **ref 的推测方法**：
   - 极可能使用 FP8 tensor core GEMM（mma.sync.m16n8k32 或类似指令）
   - 将 Q(64,128) @ K_page(128,64) 视为一个 64×64×128 的小型 GEMM
   - 使用 sm_100 的新型 tcgen05/wgmma 指令，吞吐量远超标量路径
   - 可能同时处理多个 batch/page 以提升并行度

4. **尝试 WMMA（v4）失败**：使用 `nvcuda::wmma::fragment<matrix_a, 16, 16, 32, __nv_fp8_e4m3>` 编译错误——sm_100(Blackwell) 不支持标准 WMMA 接口处理 FP8，需要使用 CUDA 13 的新 wgmma/tcgen05 intrinsic 或 CUTLASS 4.x DSL。

---

## 未能突破的原因

1. **Blackwell FP8 tensor core API 复杂**：B200 (sm_100) 使用全新的 `tcgen05.mma` PTX 指令和 CUTLASS 4 DSL，与 Hopper WMMA/wgmma API 不同。在单次 goal-gpu-run 内无法完成 CUTLASS 4.x kernel 的设计和调试。

2. **CUDA events 测量失真**：ref CUDA events 中位数 ~52 µs（含 Python/JIT overhead），而 profiler kernel span 仅 0.42 µs，使得早期 benchmark 方向性错误。

3. **轮次限制**：在发现 WMMA 编译问题前，前期轮次消耗在 v1/v2 的串行 kernel 调试上（共 15 轮 GPU jobs）。

---

## 如果继续的方向

1. **CUTLASS 4.x StreamK + FP8 GEMM**：使用 `nvidia-cutlass-dsl 4.6.2` 的 Blackwell FP8 kernel，以 Q(64,128) @ K_all_tokens^T 为一个大 GEMM 来一次性处理所有 batch/token。

2. **CuTe DSL + tcgen05**：参考 CUTLASS 4.x Blackwell example，实现 per-page 64×64×128 FP8 GEMM 使用 tcgen05.mma 指令。

3. **两阶段优化**：先用 cuBLAS FP8 GEMM 验证理论上限，然后根据结果优化。

---

## 终端输出证据

```
B=27, seq_lens=[133, 136, 136, 141, 1766, 198, 159, 134, 202, 152, 136, 275, 5806, ...]

=== fp32 naive ===
naive shape=torch.Size([27, 2048]), batch12 top5=[6902, 12704, 5138, 6505, 5893]

=== ref run ===
ref shape=torch.Size([27, 2048]) dtype=torch.int32, batch12 top5=[481, 482, 486, 487, 494]
E_ref=204.558270  tol=1431.907887
Mean set-match: 0.9760

  [REF] _dsa_score_kernel: 0.42 µs/call
  [REF] total: 0.42 µs/iter  ← 条件1 证据

  [MINE v3] score_kernel: 9.59 µs/call
  [MINE v3] topk_kernel:  3.82 µs/call
  [MINE v3] total: 13.41 µs/iter ← 最佳自定义 kernel

  E_mine=0.000000  tol=1431.907887  accuracy_pass=True ← 条件2+3 证据
```
