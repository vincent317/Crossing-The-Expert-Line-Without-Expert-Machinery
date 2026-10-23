# V6 设计：Online Softmax (FA2 风格)

## 核心思想
不存储完整的 S 矩阵，而是每算完一个 K tile 的 QK，立刻做 softmax 并更新 O。
这样 softmax 和 PV 可以重叠，且省掉 S tmem 和 P smem。

## 算法（FA2 的 online softmax）

对每个 Q tile：
```
m_old = -inf, l_old = 0, O = 0  # 初始状态

for each K tile kb:
    # 1. 计算 QK -> S_kb (只保存当前 tile)
    S_kb = Q @ K_kb^T  # 128x128, 存在 tmem 临时位置
    
    # 2. 更新 max
    m_new = max(m_old, rowmax(S_kb))
    
    # 3. 计算 exp 和新的 sum
    P_kb = exp(S_kb - m_new)
    l_new = exp(m_old - m_new) * l_old + rowsum(P_kb)
    
    # 4. Rescale 旧的 O
    O = O * exp(m_old - m_new)
    
    # 5. 累加新的贡献
    O += P_kb @ V_kb
    
    # 6. 更新状态
    m_old = m_new
    l_old = l_new

# 最终归一化
O = O / l_new
```

## 关键改动（相比 v4）

### 1. 去掉存储
- 去掉 S 的两个 128x128 tmem buffer（省 512 列）
- 去掉 P smem（省 32KB）
- 只需要一个临时 S_kb (128x128) 用于当前 K tile

### 2. Softmax 循环改写
v4 的两阶段（先所有 K tile 算 max/sum，再所有 K tile 算 PV）
改为
v6 的单阶段（每个 K tile：QK → softmax → PV，流水线）

### 3. 寄存器压力
每个 softmax warp 需要维护：
- m_old, l_old (每行 1 个 f32)
- O 的累加器（已经有，128 列 x 32 行/warp）

### 4. Pipeline 重组
```
Main loop (warp 4):
  for kb in K_tiles:
    - 发射 K/V TMA (stage kb)
    - 等待 K ready
    - 发射 QK MMA
    - 等待 QK MMA 完成
    - 通知 softmax warps: S_kb ready
    - 等待 V ready
    - 等待 softmax warps: P_kb ready
    - 发射 PV MMA
    - 通知 softmax warps: PV done (可以复用 S_kb buffer)

Softmax warps (0-3):
  for kb in K_tiles:
    - 等待 S_kb ready
    - 从 S_kb tmem 读一行
    - m_new = max(m_old, rowmax(S_kb_row))
    - P_kb_row = exp(S_kb_row - m_new)
    - l_new = exp(m_old - m_new) * l_old + rowsum(P_kb_row)
    - (optional) 写 P_kb_row 到 smem
    - 等待 PV done
    - rescale O_row *= exp(m_old - m_new) / l_old
    - O_row 会被 MMA 自动累加
    - m_old = m_new, l_old = l_new
```

等等，有个问题：**PV MMA 怎么知道要乘以哪个系数？**

MMA 是 O += P @ V，但我们需要：
- 旧的 O 先 *= correction_factor
- 新的 P @ V 也要 *= 1/l_new（最后）

解决方案：
1. Softmax warp 把 correction = exp(m_old - m_new) 写到 shared
2. 主 warp 在发射 PV MMA 前，把 O rescale（需要从 tmem 读出，rescale，写回）
3. 或者：在 PV MMA 后，用 softmax warp 统一 rescale O

看起来复杂度暴增...

## 简化方案：两遍 K tile 循环

保持 v4 的结构，但：
1. 第一遍：只算 max（不算 sum，不存 S）
   - QK MMA → 读出每行的 max → 更新 m
2. 第二遍：算 exp, sum, PV
   - QK MMA → exp(S - m) → sum → PV MMA

这样还是两遍，但第一遍不需要存 S（只存 m），第二遍 online。

不对，还是要存 S，因为第二遍要 PV。

## 真正的 Online：Split-K 风格

或许需要改变 tile 策略：
- 不是 (Q_tile, 所有 K) 而是 (所有 Q, K_tile)
- 每个 K tile 算完后，部分 O 写回 gmem（带 m, l 元数据）
- 最后一个 reduce pass 合并

但这需要额外的 gmem 读写，可能更慢。

## 结论

真正的 online softmax 实现很复杂，需要仔细处理：
- Rescale O 的时机
- tmem 和 register 的数据流
- Warp 之间的同步

可能需要：
1. 先实现一个简单版本（两遍循环，第一遍不存 S）
2. 或者回到 QTILE=256, KTILE=128（v4 的变种）
3. 或者接受 v5 + 其他小优化，争取到 0.45-0.5 ms

让我先等 v4 结果再决定。
