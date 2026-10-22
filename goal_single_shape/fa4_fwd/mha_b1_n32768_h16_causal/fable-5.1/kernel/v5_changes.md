# V5 改动：基于 V4 + f16x2 Packed Softmax

## 核心思想
Softmax 是 v2/v4 的主要瓶颈之一。使用 f16x2 packed operations 可以：
1. ex2.approx.f16x2：一次处理 2 个 exp，理论 2x 吞吐
2. cvt.rn.f16x2.f32：一次转换 2 个 f32→f16

## 实现策略

### 当前 V4 softmax（每个 warp 处理 32 行）
```cpp
for (int r = warp_id * 32; r < (warp_id + 1) * 32; ++r) {
  float m = m_row[r];
  // 读 S 的一行（128 个 f32）
  // 逐元素：exp2(s * LOG2E * scale - m * LOG2E)
  // 累加 sum
  // 写 P（128 个 bf16）
}
```

### V5 改进：f16x2 batch processing
```cpp
// Step 1: 找 max（还是 f32，保持精度）
float m = -INFINITY;
for (int c = 0; c < 128; c += 4) {
  uint32_t s[4];  // 从 S tmem 读 4 个 f32
  tmem_ld4(...);
  m = fmaxf(m, fmaxf(...));  // 4-way max
}

// Step 2: packed exp + sum
// 把 scale 提前乘好
float pre = LOG2E * softmax_scale;
float m_scaled = m * pre;
float sum = 0;
uint32_t p_buf[64];  // 128 个 f16 = 64 个 uint32

for (int c = 0; c < 128; c += 2) {  // 步长 2
  float s0, s1;  // 从 S tmem 读 2 个 f32
  tmem_ld2(...);
  
  // 转成 f16x2（用于 exp2）
  // 但 exp2.f16x2 的输入需要是 f16...
  // 问题：s0, s1 是 f32，需要先算 s*pre - m_scaled（f32精度）
  //       再转 f16，再 exp2.f16x2
  
  float e0_in = s0 * pre - m_scaled;
  float e1_in = s1 * pre - m_scaled;
  
  // 转 f16x2
  uint32_t in_f16x2 = pack_f16x2(e0_in, e1_in);  // cvt.rn.f16x2.f32
  
  // exp2.approx.f16x2
  uint32_t out_f16x2;
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(out_f16x2) : "r"(in_f16x2));
  
  // 转回 f32 做 sum（为了精度）
  float e0 = __half2float(__low2half(out_f16x2));
  float e1 = __half2float(__high2half(out_f16x2));
  sum += e0 + e1;
  
  // 存 bf16（用于 P）
  p_buf[c/2] = pack_bf16x2(e0, e1);
}

// Step 3: 归一化 + 写 P
float inv_sum = 1.0f / sum;
for (int c = 0; c < 64; c++) {
  // 从 p_buf 读 bf16x2，乘 inv_sum，写回
  // 或者直接写，然后在 MMA 前做 scale？
}
```

等等，有个问题：**pack_bf16x2 不是硬件指令**！

需要改为：
```cpp
// 存 P 时直接用 __float2bfloat16_rn
__nv_bfloat16 p0 = __float2bfloat16_rn(e0 * inv_sum);
__nv_bfloat16 p1 = __float2bfloat16_rn(e1 * inv_sum);
p_smem[r * 128 + c] = p0;
p_smem[r * 128 + c + 1] = p1;
```

## 关键瓶颈
1. f16↔f32 转换开销（但 probe 会告诉我们 cvt.rn.f16x2.f32 的吞吐）
2. sum 还是要在 f32 做（精度）
3. 最终写 bf16 还是逐个的

## 预期收益
假设 exp2 是主要开销（~40% softmax 时间）：
- exp2.f32：每 clk 处理 1 个
- exp2.f16x2：每 clk 处理 2 个
- 加速：~1.3-1.5x（考虑其他开销）

如果 softmax 占总时间 30%：
- 总加速：1 / (0.7 + 0.3/1.5) ≈ **1.13x**

V4 预期 ~0.65 ms → V5 预期 ~0.58 ms（还是不够 0.44 ms）

## 结论
f16x2 优化是渐进改进，不是量变。如果 V4 > 0.6 ms，V5 也救不了。

真正的突破需要：
1. Online softmax（省掉 S 存储，减少同步）
2. 或者接受达不到目标，分析根本瓶颈
