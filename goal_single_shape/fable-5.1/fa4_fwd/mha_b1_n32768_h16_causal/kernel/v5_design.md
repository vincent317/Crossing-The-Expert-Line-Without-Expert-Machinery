# V5 设计：Packed F16x2 Softmax

## 核心思路
在 v4 基础上，将 softmax 的 exp2 计算从 f32 改为 packed f16x2，吞吐翻倍。

## 关键改动点

### 1. Softmax exp2 循环（当前 v4）
```cpp
// 每个 thread 处理 1 行 S，遍历 128 列（KTILE=128）
for (int col = lane; col < KTILE; col += 32) {
  float s = S[row * KTILE + col];
  float e = ex2((s - m_new) * scale_log2);
  sum += e;
  P_row[col] = e;
}
```

### 2. V5 改为 f16x2（每次处理 2 列）
```cpp
// 每个 thread 处理 2 列/iter，总共 64 iter
for (int col = lane * 2; col < KTILE; col += 64) {
  float s0 = S[row * KTILE + col];
  float s1 = S[row * KTILE + col + 1];
  
  // f32 -> f16x2
  unsigned sh = pack_f16x2((s0 - m_new) * scale_log2, (s1 - m_new) * scale_log2);
  
  // ex2.approx.f16x2
  unsigned eh;
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(eh) : "r"(sh));
  
  // unpack 回 f32 累加 sum
  float e0, e1;
  asm volatile("cvt.f32.f16 %0, %1;" : "=f"(e0) : "h"((unsigned short)(eh & 0xFFFF)));
  asm volatile("cvt.f32.f16 %0, %1;" : "=f"(e1) : "h"((unsigned short)(eh >> 16)));
  sum += e0 + e1;
  
  // 写回 P (bf16)
  P_row[col] = __float2bfloat16(e0 * inv_sum);
  P_row[col + 1] = __float2bfloat16(e1 * inv_sum);
}
```

### 3. Helper 函数
```cpp
__device__ __forceinline__ unsigned pack_f16x2(float lo, float hi) {
  unsigned d;
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
```

## 精度风险评估
- S - m_new 范围：[-20, 0] (log space)
- 乘以 scale_log2 (1.44)：[-29, 0]
- f16 表示范围：[-65504, 65504]，精度 ~3 位小数
- exp2 输出：[2^-29, 1] = [1.9e-9, 1]
- f16 可以表示，但精度可能在 2^-29 附近下降

实际影响：
- softmax 主要权重在大值，小值（<2^-10）对结果影响小
- 最终 P 会转 bf16，f16 中间精度应该够
- 需要实测确认误差 ≤ tol

## 预期性能
- v4 预期 ~0.77 ms
- softmax exp2 如果占 15%，翻倍后省 7.5%
- v5 预期 ~0.71 ms（仍不够，但进步）

## 如果 v5 仍不够
需要 v6: online softmax，这是更根本的改变。
