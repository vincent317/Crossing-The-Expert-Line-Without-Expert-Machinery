# GDN FWD (B200) —— 任务说明书

这是我们要的目标算子（GDN FWD，Gated Delta Net 变长 prefill，B200）：

T = 8192 个 token 打包成 N = 20 条序列，各条长度为
753 371 161 41 105 66 23 31 19 15 2433 39 22 3371 20 42 219 15 45 401。
q/k 4 头、v 8 头（每个 q/k 头被两个 v 头共享），head_dim = 128。

  q, k        bf16 [8192, 4, 128]
  v           bf16 [8192, 8, 128]
  state       fp32 [20, 8, 128, 128]   （k-last 布局 [N, H, V, K]）
  A_log       fp32 [8]
  a, b        bf16 [8192, 8]
  dt_bias     fp32 [8]
  cu_seqlens  int64 [21]
  scale       float = 0.08838834764831843
  ->
  output      bf16 [8192, 8, 128]
  new_state   fp32 [20, 8, 128, 128]   （k-last）

契约：run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale) -> (output, new_state)

数据 = 比赛 workload a9540651（FlashInfer-Bench 数据集 flashinfer-ai/mlsys26-contest
里的真实张量，不是我造的），已经放在 GPU 机器上。
算子的权威语义 = FlashInfer-Bench definition `gdn_prefill_qk4_v8_d128_k_last`，
它自带的 fp32 逐 token 串行参照实现已经给你了（见下）。

## GPU

cctl devspace `tasks/1036144`（cluster <cluster> / 项目 <project> /
资源池 <resource-pool>，1×B200 183GB）。本机 Teleport 已登录（证书到 09-22 18:50）。

  登录：tsh ssh root@gpu-node-1036144
  传文件：tsh scp <本地文件> root@gpu-node-1036144:<路径>

**这条租约是留给你的，不要 stop 或删除它。** 频率已经锁在 1965 MHz
（nvidia-smi -lgc 1965）；如果被重置了，量之前先锁回去。
节点能连 pypi，**连不上 github.com 和 huggingface.co** ——要从网上拿东西，
先在本机（Mac）下载再 tsh scp 上去。
CUDA 13.0 工具链在 /usr/local/cuda/bin（默认不在 PATH，需要自己 export），
nvcc / ncu / ptxas / nvdisasm / cuobjdump / compute-sanitizer 都有，nsys 在 PATH 里。
python 用 /opt/sglang/bin/python（torch 2.13.0+cu130、triton 3.7.1、
nvidia-cutlass-dsl 4.6.2、cuda-python 13.3.1）。

机器上 `/root/gdn` 已经给你准备好了，**先读 `/root/gdn/README.md`**：

  /root/gdn/README.md                     总说明（先读这个）
  /root/gdn/data/definition.json          官方算子定义（含权威语义）
  /root/gdn/data/workload_*.safetensors   5 个真实 workload（目标 + 4 个用来验正确性）
  /root/gdn/data/golden_spec_fp32.*       fp32 参照在目标 workload 上的输出（数值真值）
  /root/gdn/data/golden_ref.*             你要打败的那个 kernel 在目标 workload 上的输出
  /root/gdn/harness/gdn_data.py           读 workload
  /root/gdn/harness/gdn_spec_reference.py 官方 fp32 逐 token 串行参照（= 任务规格）
  /root/gdn/harness/gdn_bench.py          device time 测量（CUPTI kernel span，cold-L2）
  /root/gdn/harness/gdn_check.py          正确性判据
  /root/gdn/harness/calib.py              机器状态标定
  /root/gdn/work/                         你的工作目录

## 你要打败的数

ref = KDA 0.5 的 CuTe-DSL GDN prefill kernel（MLSys 2026 FlashInfer 比赛 GDN 赛道第三名）。
它已经在**这块卡、这个容器、锁频 1965 MHz** 下用 `gdn_bench.device_time_ms` 量过了：

  ref device time = **124.93 us**（cold-L2；五次独立测量 124.971 / 124.937 /
  124.937 / 124.922 us，warm-L2 124.877 / 124.880 us，离散度 < 0.1%）

**目标：device time 比它快 1.2x，也就是 <= 104.1 us**（同一个
`gdn_bench.device_time_ms(...)` 调用，同一个 workload a9540651）。

`harness/calib.py` 量两个固定的公开负载（4096³ bf16 GEMM + 256MB stream add），
定锚时是 85.62 us / 81.22 us。你任何时候都可以重跑它确认机器状态没变、
上面那个 124.93 us 还和你的测量可比。

### 参照的实现不在这台机器上

只留了它的**输出**和它的**时间**，源码在你这个会话开始之前就已经删掉了，
驱动你这个会话的那台 Mac 上也没有。

**绝对不许看或抄这个 ref 的实现**，也不要绕着去找它：
不要去网上找 KDA / kernel-design-agents / mit-han-lab / NVlabs 的 kernel 源码，
不要找 mlsys2026-flashinfer-contest-solution 这个仓库或任何比赛提交的解，
也不要去读别的项目里这个算子（Gated Delta Net / gated delta rule / GDN prefill）
的 kernel 实现（flash-linear-attention、fla、flashinfer、sglang、vllm 里的都算）。

可以联网，公开文档、论文、CUDA / PTX / CUTLASS / CuTe 的资料、以及下面给你的
KernelWiki 都随便看——**别人写的这个算子的 kernel 实现不行**。从 0 开始自己写。

## 正确性（务必做扎实）

1. fp32 的 naive 慢实现已经有了：`harness/gdn_spec_reference.py`，
   就是官方 definition 自带的逐 token 串行 fp32 递推，这是数值真值。
   请自己把它读懂、并**另外独立写一份**你自己的 fp32 慢实现（比如分块/向量化的），
   两份先互相对上，确认你对语义的理解没错，再往下做。
2. 容差不是拍的：`gdn_check.py` 用 **fp32 真值和 ref 的差** 当下限，再乘 7：
       floor_max = max |spec - ref| ，floor_rms = rms(spec-ref)/rms(spec)
       tol = 7 x floor ，两个都要满足
   目标 workload 上实测 floor 是
       output    floor_max 2.441e-04 -> tol 1.709e-03 ; floor_rms 2.828e-03 -> tol 1.980e-02
       new_state floor_max 1.009e-03 -> tol 7.060e-03 ; floor_rms 1.811e-03 -> tol 1.268e-02
   （比赛自己的判据 atol=rtol=1e-2 也会一起打出来，但以 7x floor 为准。）
3. 不要只对一份输入过拟合。`gdn_data.WORKLOADS` 里另外 4 个真实 workload
   （T=6/N=1、T=139/N=3、T=1377/N=1、T=8192/N=39）都要过；
   再自己造几组输入：换 seed、换序列长度组合（包括很短的、只有一条的、
   长度不整除 chunk 的），并且把 a / b 的幅度拉开，让
   g = exp(-exp(A_log)*softplus(a+dt_bias)) 和 beta = sigmoid(b) 覆盖到各自的范围
   （g 接近 1 和接近 0 两端都要有），确认递推的衰减/更新两条路径都对。
   用 `gdn_check.check_arbitrary(你的run, args)` 判。
4. output 和 new_state 两个都要判，new_state 是 k-last 布局，别转置错了。

## 其它

1. 请使用一个独立干净的环境：不要用本机已有的 skills、memory、
   ~/.claude 下的东西或者别的臂留下的文件。你的工作目录就是当前目录和
   机器上的 /root/gdn/work。
2. 参考知识：`./KernelWiki/`（mit-han-lab/KernelWiki，已经拉到本地了），
   先看 `KernelWiki/index.md` 和 `KernelWiki/SKILL.md`。它是笔记不是代码库，
   里面没有这个算子的实现。
3. 预算 24 小时。把每一版的 device time、和 124.93 us 的比值、正确性结论
   都记在你自己的工作目录里（建议 ./work/LOG.md），并且保留每一版的源码。
4. 不要问我问题，遇到选择自己决定。用户下班了，不会回你。
5. 长时间的任务（编译、profile、扫参）放到机器上用 nohup/tmux 跑，
   别让它挂在一次 ssh 前台上。
