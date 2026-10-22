# DSA FWD（重负载 564007ac）B200 kernel 优化总结

## 目标与结论

| 项目 | 结果 |
|---|---|
| 参照（KDA-0.5 CuTe-DSL，`mit-han-lab/mlsys2026-flashinfer-contest-solution@7aad0429`）device time | **15.4–16.4 µs**（CUDA events、冷 L2、同一 job 内直测；各 job 间波动 ±0.5 µs） |
| 空 kernel 固定开销（同测法） | 5.1–7.0 µs（即参照 kernel 自身 span ≈ 10.3 µs，与其公布值一致） |
| fp32 naive vs fp64 naive | 输出 6.0e-6，lse 4.5e-6 |
| E_ref（参照 vs fp32 naive） | 输出 **7.73e-3**，lse 5.72e-6 → tol = 7×E_ref = **5.41e-2 / 4.01e-5** |
| 新 kernel vs fp32 naive | 输出 **1.21e-2 ≤ 5.41e-2 ✅**，lse **3.81e-6 ≤ 4.01e-5 ✅**，两次调用逐位一致 |
| 新 kernel device time（最终 v9） | **17.44 µs**（直测）/ 16.82 µs（CUDA graph），kernel span 10.75 µs |
| 目标 ≤ REF/1.2 | 需 ≤ 12.96 µs，**未达成**（0.89×） |

条件 1–3、5 已满足；条件 4（1.2× 加速）未满足。本文件按目标条款给出最好结果与瓶颈说明。

## 最终实现（`dsa_kernel.cu` / `dsa_kernel.py`）

- grid = 2×SM 数（296）个 CTA，每 CTA 8 warp，smem 95 KB 使每 SM 恰好常驻 2 个 CTA；CTA 启动时用 `%smid` + 每 SM 单调原子计数得到 (smid, slot)，据此以 chunk-major 顺序确定性领取 64 行的 chunk（slot 0 领取全部 token 的低 chunk，一 SM 一个），两个候选 chunk 的索引预先并行加载以掩盖原子延迟。
- 64 行 KV 与 Q 用 1D TMA bulk copy + mbarrier 一次性搬入 smem（-1 行零填充并在 logits 处 mask）。
- S = Q·Kᵀ 用 `mma.sync m16n8k16 bf16`，8 warp 按 4 个 k-slice × 2 个行半分工，片段加载手动提前一步（volatile ldmatrix，避免编译器把 smem 读提到 mbarrier 等待之前——这曾导致 v5 读到未落地数据）；4 个 k-slice 由 256 线程协作求和一次；softmax 每 CTA 只算一次（P 以 bf16 存 smem，各 warp 用 ldmatrix 取 A 片段）；P·V 每 warp 64 维。
- 每 chunk 发布 (m, l) 与 bf16 归一化 partial（16 KB）；`__syncthreads` 后单线程 `fence.acq_rel.gpu` + `red.add` 到 128 B 填充的 per-token 计数器；计数器单调（epoch = 计数/32，在本 CTA 到达前读取），无需复位、无带返回值原子。
- CTA (t, c) 同时归约 token t 的 slice c（头 c/2、半个 512 维）：等待 32 次到达后按 live mask 预测加载 32 个 partial（单次往返），加权归一并写 bf16 输出与 lse。

## 各版本演进（B200 实测 span / 直测时间）

| 版本 | 关键改动 | span | 实测 |
|---|---|---|---|
| v1 | 256 CTA、cp.async gather、条件分支归约 | 15 µs | 19.5 |
| v3 | TMA gather、live-only 归约 | 10 µs | 18.2 |
| v4 | 单线程 release 到达 | 10 µs | 18.2 |
| v5 | 持久化 + 位图扫描 + R=32 | 13.75（且有 hoisting bug） | 20.5 |
| v6 | 512 CTA/4 per SM、TMA Q、volatile ldmatrix | 11.25 | 18.4 |
| v7 | 256 CTA × 8 warp、每 warp fence | 12.25 | 18.6–20.2 |
| v8 | chunk-major、协作 S 归约、单线程 release | 10.75 | 16.5–17.4 |
| v9 | SM-slot 分配、单次 softmax | 10.75 | 16.8–17.4 |

## 瓶颈分析（v9 profile，`%globaltimer`）

关键路径（busy CTA 中位数）：索引加载 1.5 → KV gather 2.25 → S 0.5 → softmax+PV 1.5 → 发布 1.25 → 到达检测 0.3 → 归约 1.0，尾部另有 2 µs 来自 2 个双 busy SM。

1. **KV gather ≈ 2.25 µs 是访存突发的延迟/排队界**：5.8 MB 冷 L2 数据在 ~2 µs 内只达到 ~2.6 TB/s；R=32/64、TMA/cp.async、TLB 预热都不改变它。只有让 gather 更早发出（不可能，依赖索引）或减少字节数才能缩短。
2. **`mma.sync` 在 B200 上约 16–20 cycle/条**（旧路径约为峰值的 1/8）：每 64 行 chunk 约 1.15 µs 的张量核时间；两 CTA 共享一 SM 时翻倍。真正的解决方案是 `tcgen05` UMMA（需 UMMA 规范 smem 布局、TMEM 分配与描述符），预计可把计算降到 ~0.3 µs 并使 SM 冲突几乎免费，但实现风险高，本轮预算内未能完成。
3. **跨 CTA 同步链 ≈ 2.5–2.75 µs**：store→ack→原子（~1.0–1.25）+ 轮询检测（~0.3）+ partial 读取往返（~1.0）。这是 split-KV 归约的固有代价；两 kernel + PDL 或 cluster/DSMEM 方案估算收益有限。
4. **CTA→SM 放置不可控**：硬件按 GPC 轮转放置，静态块序总会产生若干双 busy SM；v9 的 per-SM slot 方案把冲突降到 2 个，但仍可能落在关键 token 上。进一步方案是在冲突 SM 上把两个 chunk 各自对半分（每 chunk 两个 partial 槽位），估算再省 ~1.5 µs。
5. **测量固定开销 ~6 µs**（空 kernel）：REF 的 10.3 µs span 与本 kernel 的 10.75 µs 差距很小，但 1.2× 的目标要求 span ≤ ~6.5 µs，即需同时解决 2–4。

## 文件说明

- `dsa_kernel.cu`、`dsa_kernel.py`：kernel 源码与入口 `run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale) -> (output, lse)`。
- `dsa_naive.py`：fp32/fp64 naive 参考；`dsa_data.py`、`spec/`：workload 564007ac 输入（官方 sparse_indices blob）。
- `check.py`：精度检查（E_ref、tol=7×E_ref、新 kernel 误差、主机端 partial 重组定位）；`bench.py`：同数据、同测法的 device time 基准（含空 kernel 校准与分阶段 profile）。
- 参照 kernel 文件仅在 B200 job 内原样导入用于测时，未阅读、未纳入本仓库。

---

## 第二轮：tcgen05 UMMA + TMEM 路线（结论：在此形状上无法达标）

新增源码：`dsa_kernel_tcgen05.cu` / `dsa_kernel_tcgen05.py`（数值正确：check.py RESULT: PASS，最大误差 1.208e-2 ≤ tol 5.41e-2，与 mma.sync 版一致）、`umma_bench.py`（UMMA 发射开销微基准）。`dsa_kernel.cu` 仍为最快的 mma.sync 版本（16.8–17.4 µs，参照 15.4–16.4 µs）。

| 版本 | 关键改动 | span | 直测 |
|---|---|---|---|
| v10（tcgen05） | TMA gather4 SW128 布局 + tcgen05.mma（M=128）+ TMEM，单发射线程 | 20.5 | 28.4 |
| v11–v12 | TMEM 列地址推导 slot、描述符预计算、4 个交错累加器、合并访存归约、M=64 | 19.3 | 26.5 |
| v13 | 4 个 warp 并发发射 MMA | 结果错误 + 崩溃 | — |

实测得到的硬事实（B200，CUDA 12.9，sm_100a）：

1. **`tcgen05.mma` 的发射开销约 160 cycle/条（单线程，微基准），与 M（64/128）、N=16、累加器个数均无关；commit→mbarrier 完成仅 ~115 cycle，`tcgen05.ld` 往返 ~47 cycle。** 本算子每个 64 行 chunk 需要 36（S）+ 16（P·V）= 52 条 MMA（N = 16 个头太小，K 固定 16），发射即 ≥ 2.5 µs，而 mma.sync 版本 8 个 warp 并行只需 ~1.25 µs。UMMA 的高吞吐建立在每条指令承载 128×256×16 的大计算量上，对 N=16 的解码形状是负优化。
2. **多线程并发发射 `tcgen05.mma` 在完整 kernel 中产生错误结果并触发 illegal memory access**（微基准只验证了不崩溃）；必须由单线程发射，因此发射开销无法用并行摊薄。
3. **使用 tcgen05.alloc 的 kernel 会被 CUDA 运行时报告为每 SM 仅可驻留 1 个 CTA**（探针：仅含 alloc/dealloc 的 kernel occupancy=1，其它=16），但硬件实际允许 2 个 CTA 共驻留（强制 grid=2×SM 后全部到达计数完成、结果 PASS）；第二个 CTA 要等第一个执行 `relinquish_alloc_permit` 后才被调度，故 alloc 应放在 kernel 最前。TMEM 列自下而上分配（0、128），可直接用列地址区分同一 SM 上的两个 CTA，省去原子计数。
4. **TMA `tile::gather4` 只接受 box 外维为 1 的 tensor map**（box {64,4} → illegal instruction）；`.shared::cluster` 与 `.shared::cta` 目标都可用；坐标顺序 {x, y0, y1, y2, y3}。144 次 gather4 的 gather 耗时 2.5–3.0 µs（1D bulk 128 次为 2.25 µs），逐行 tiled 拷贝 576 次需 5.75 µs。
5. **M=64 的 TMEM 累加器布局**：行 r → lane 32·(r/16) + r%16（每个 lane 四分之一区的前 16 个 lane），已由 check 验证。
6. UMMA 描述符：K-major SW128 时 LBO 字段固定为 1（16 B）、SBO=1024 B；MN-major SW128 时 LBO=沿 MN 的 64 元素块步长（8 KB）、SBO=1024 B；同一份 TMA SW128 搬入的 K/V 块可同时作为 K-major A（S=KQᵀ）与 MN-major A（Oᵀ=VᵀPᵀ）。

综合：本形状的张量核计算量极小（每 chunk 约 0.6 MFLOP），性能由访存延迟、跨 CTA 同步与指令发射开销决定；tcgen05 路线把发射开销集中到单线程反而更慢。达到 1.2× 需要 span ≤ 6.5 µs，剩余可行方向是在 mma.sync 版本上继续压缩 gather 延迟（2.25 µs）与跨 CTA 归约链（~2.5 µs），本轮预算内未能完成。
