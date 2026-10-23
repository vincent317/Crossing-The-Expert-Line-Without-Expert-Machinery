# V4b 方案：QTILE=256, KTILE=128

## Smem 布局
- Q: 256×128 bf16 = 65536 B (1 stage)
- K: 128×128 bf16 × 2 stage = 65536 B
- V: 128×128 bf16 × 2 stage = 65536 B
- P: 256×128 bf16 = 65536 B (1 buffer，两次写入)
- Barrier: 1024 B

Total = 65536 + 65536 + 65536 + 65536 + 1024 = 263168 B = **257 KB**

**超出 B200 的 232 KB 限制！**

## 尝试优化
降到 1-stage K/V？
- Q: 65536
- K: 32768 × 1
- V: 32768 × 1
- P: 65536
- Bar: 1024

Total = 65536 + 32768 + 32768 + 65536 + 1024 = 197632 B = **193 KB** ✓

但 1-stage 会导致 pipeline stall，性能可能更差。

## 或者：不存 P
如果 softmax warp 直接把 P 写成 bf16 到 smem 的临时位置，
然后 MMA 立刻消费，不需要完整的 256×128 buffer。

改为 ping-pong：
- P_buf0: 128×128 (前半)
- P_buf1: 128×128 (后半)

但 softmax 要分两批处理...复杂。

## 结论
V4b (QTILE=256, KTILE=128) 的 smem 预算太紧。
除非：
1. 用 1-stage K/V（可能更慢）
2. 复杂的 P buffer 管理
3. 或者接受 V4 (QTILE=128) 的 work-item 数量

让我先看 V4 结果。如果 V4 < 0.45 ms，那 V5 (f16x2) 可能够。
如果 V4 > 0.6 ms，那可能需要重新考虑整体策略。
