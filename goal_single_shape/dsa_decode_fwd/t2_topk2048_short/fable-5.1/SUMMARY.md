# DSA FWD（小请求）B200 kernel 优化总结

目标算子：`dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`，workload `9d4a5f21`（T=2、H=16、
ckv 512、kpe 64、top-k 2048、页大小 64、8462 页、bf16；两个 token 实际命中 18/11 行）。参照实现为
KDA 0.5 的 CuTe-DSL kernel（`mit-han-lab/mlsys2026-flashinfer-contest-solution` @ `7aad0429`，
`kernels/dsa_attention/kernel.py`），仅作为黑盒调用其 `run(...)` 用于测时和精度对比，未阅读其源码。

所有测量均在 cctl `aiforai-b200` 资源池的 B200 上、同一个 batch job 内完成（最终 job 1043568，
节点 mbkj-k8s-existed-node-8hfx，日志 `logs/1043568.log`）。

## 测时口径

device time = CUPTI kernel span：用 `torch.profiler`（CUPTI activity）抓取 GPU 时间线，每次调用前后
用一个空 marker kernel 分隔，取该次调用内首个 kernel 开始到最后一个 kernel 结束的跨度（不含 host
发射开销）。ref 与新 kernel 使用完全相同的输入张量、相同的 100 次迭代、相同的 L2 状态协议，取中位数。
四种 L2 协议：
- `cold-L2`：每次迭代前用写 512MB 的 kernel 刷掉 L2（比赛 harness 的做法）；
- `cold-L2-readflush`：用读 512MB 的 kernel 刷 L2（L2 冷但无脏行）；
- `cold-L2-settled`：写刷 L2 后再空转约 10 µs 让访存系统静默；
- `warm-L2`：不刷 L2。

## 最终结果（job 1043568）

精度（fp32 naive 实现 `naive_fp32.py` 为基准）：

| 量 | ref 与 naive 的最大误差 E_ref | 容差 7×E_ref | 新 kernel 与 naive 的最大误差 | 判定 |
|---|---|---|---|---|
| output | 1.4465e-02 | 1.0126e-01 | 1.4465e-02 | PASS |
| lse | 9.5367e-07 | 6.6757e-06 | 1.9073e-06 | PASS |

（新 kernel 与 ref 直接比：output 最大差 3.9e-03，lse 最大差 1.9e-06。另外对 2048 全有效、随机 40%
有效、一个 token 全无效、T=8 随机计数等索引模式也与 naive 逐一比对通过。）

速度（中位数，µs；目标 new ≤ ref/1.2）：

| 协议 | ref | new | 加速比 | 目标 | 判定 |
|---|---|---|---|---|---|
| warm-L2 | 3.135 | **2.561** | **1.224×** | ≤ 2.613 | **PASS** |
| cold-L2 | 3.456 | 3.200 | 1.080× | ≤ 2.880 | FAIL |
| cold-L2-readflush | 3.424 | 3.136 | 1.092× | ≤ 2.853 | FAIL |
| cold-L2-settled | 3.456 | 3.200 | 1.080× | ≤ 2.880 | FAIL |

warm-L2 协议下连续四次独立 job 均通过（2.560/2.576/2.560/2.561 µs，加速 1.22–1.24×）。
cold-L2（比赛口径）下新 kernel 快于 ref 约 8–10%，未达到 1.2×。

## 新 kernel 设计（`dsa_kernel.cu`，默认配置）

每个 (token, head) 一个 CTA（共 32 个 CTA），4 个 warp，各 warp 拥有下标位置 p ≡ warp (mod 4)，
一次处理最多 5 行：
1. 入口同时发出：整条 top-k 下标表的合并加载（每 CTA 一次）、本 warp前 5 个位置的广播加载、q 行加载；
2. 前 5 个位置一到就直接把对应的 ckv/kpe 行 gather 进寄存器——投机假设 decode token 的少数有效下标都在
   表头（本 workload 恒成立）；每个 warp 只用一个标志记录"自己那 1/4 的表里是否还有有效下标落在投机
   位置之外"；
3. fp32 在线 softmax：无分支，5 个 warp 归约按层交错，无效槽位以 p=0 参与；
4. 各 warp 的 (m, l, acc) 经 smem 在唯一一次 CTA 栅栏后由 128 线程合并；同一栅栏后读取投机标志，若有
   warp 标记，则整块重新 stage 下标表、精确压缩并按 5 行一组的通用路径重算（保证任意下标模式正确）。

## 瓶颈分析与放弃的方向

用 `globaltimer`/`clock64` 在 kernel 内分段打点（`-DDSA_TIMING=1` 变体）得到的时间分解（warm-L2，
单 CTA 约 2400 cycles）：下标到达约 980、投机行到达约 44（已被隐藏）、softmax 数学约 675、合并栅栏约
360、最终写回约 275；kernel 之外还有约 1.1–1.4 µs 的空 kernel 发射/回收底噪（ref 同样承担）。

cold-L2 的瓶颈是两次相互依赖的 DRAM 访问（先读下标、再按下标 gather 行），每次约 1100–1200 cycles，
两者合计约 1.2 µs，加上底噪已接近 2.9 µs 的目标预算；ref 在 cold 下相对 warm 只慢 0.3 µs，说明它把
其中一次 DRAM 延迟藏在了自身较长的固定开销后面，而新 kernel 的 warm 路径已很短，cold 下这两次延迟
完全暴露。读刷、静默等协议变化均不改变这一结论。

尝试并放弃的方向：
- 指令取指瓶颈假说（用纯 ALU 探针证伪）、干跑预热指令缓存（更慢且跨 chunk 不正确）；
- 预取（`prefetch.global.L2`）、每 SM 独占（大动态 smem）、跳过空 warp 的合并流量：无收益；
- 张量核实现 `dsa_mma.cu`（每 token 一个 CTA，mma.sync + cp.async/TMA gather）：功能正确但 S 阶段受
  smem 带宽（Q/K 各读一遍 + 16KB 部分和交换）与多次栅栏限制，warm 约 4.0 µs，慢于标量实现；
  TMA 逐行小拷贝还有明显的每拷贝开销。

## 文件

- `dsa_kernel.cu`：新 kernel（及各基准变体的编译开关）；`dsa_ops.py`：ctypes 封装。
- `dsa_mma.cu`：张量核实现（保留作对照）。
- `naive_fp32.py`：fp32 慢实现；`check.py`：精度检查（7×E_ref 容差 + 额外索引模式）。
- `bench.py`：CUPTI kernel-span 测时（四种 L2 协议、分段打点、探针）。
- `workload.py` + `sparse_indices_9d4a5f21268e484ea05a2f2af91d9fa7.safetensors`：比赛 workload 输入。
- `run_ref.py`：黑盒加载参照实现；`job.sh`/`submit_job.sh`/`fetch_log.sh`/`run_until_done.sh`：B200
  batch job 驱动；`logs/`：各轮 job 日志。
