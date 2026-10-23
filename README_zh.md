<div align="center">

# JustGoal: Surpassing Expert Kernels and Specialized Kernel Agents with Only a Goal

**只给通用 agent 一个优化目标，在 NVIDIA B200 上写出的 GPU 算子**

[English](README.md) | **中文**

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](paper.pdf)
[![Kernels](https://img.shields.io/badge/Kernels-B200%20sm__100a-76b900)](#仓库结构)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/vincent317/Crossing-The-Expert-Line-Without-Expert-Machinery?style=flat)](https://github.com/vincent317/Crossing-The-Expert-Line-Without-Expert-Machinery/stargazers)
[![Issues](https://img.shields.io/github/issues/vincent317/Crossing-The-Expert-Line-Without-Expert-Machinery)](https://github.com/vincent317/Crossing-The-Expert-Line-Without-Expert-Machinery/issues)

<img src="assets/figure3.png" width="80%" alt="JustGoal 与 KDA-0.5 复现在四个算子上跨三代模型相对参照算子的加速比">

<sub>JustGoal 与 KDA-0.5 复现在四个算子上跨三代模型相对参照算子的加速比；每根柱子是该算子各
shape 的几何平均，虚线为参照算子。</sub>

</div>

本仓库发布论文
[*JustGoal: Surpassing Expert Kernels and Specialized Kernel Agents with Only a Goal*](paper.pdf)
中生成的算子（kernel）代码。JustGoal 只给通用 agent 一个优化目标和约束，不预先配置任何算子专用的
评测设施、知识库或搜索工作流；agent 自己搭建评测循环并迭代优化实现。在 NVIDIA B200 上，JustGoal
在 **11 个算子中的 9 个**上超过所评测的开源库实现，逐算子几何平均加速比为 **1.14–2.34×**；在有
专用 kernel agent 基线的 10 个算子中，有 **5 个**超过所评测的最快 agent 生成实现。使用
**Fable 5.1** 时，单 shape JustGoal 在 12 个评测 shape 上相对我们的 KDA-0.5 复现取得 **1.24×**
的几何平均加速比。

## 仓库结构

| 目录 | 内容 |
|-|-|
| [`goal_single_shape/`](goal_single_shape/) | 指定单个 shape 的 `/goal` 实验 |
| [`goal_full_shape/`](goal_full_shape/) | 覆盖算子定义全部 shape 的 `/goal` 实验 |
| [`kda05_reproduction/`](kda05_reproduction/) | KDA-0.5 设定下的实验 |
| [`kda05_single_shape/`](kda05_single_shape/) | KDA-0.5 设定下指定单个 shape 的实验 |
| [`ablation/`](ablation/) | 消融实验 |
| [`benchmarks/`](benchmarks/) | 复测结果与证据 |

算子按 `<算子>/<shape>/<模型>/` 组织。`goal_full_shape/`、`kda05_reproduction/` 和
`kda05_single_shape/` 中的算子附带 [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) 的
`solution.json`。

## 环境

- GPU：NVIDIA B200（`sm_100a`）
- CUDA：13.0（`nvcc` 13.0.88，驱动 580.105.08）。有两个算子指向 CUDA 13.1 工具链：
  `goal_single_shape/mla_paged_prefill/b1_q33_kv34/opus-5` 和
  `goal_single_shape/gqa_paged_decode/b16_kv2841/opus-5`。
- CuTe DSL（`nvidia-cutlass-dsl`）：4.6.2。`kda05_reproduction/` 和 `kda05_single_shape/`
  中的 DSA、GDN、MoE 算子使用 KDA-0.5 设定钉住的 4.6.0。

## 参与贡献

欢迎提 Issue 和 Pull Request，特别是结果错误或编译失败的报告，以及在其他 B200
环境上的复测结果。

## 引用

<details>
<summary>BibTeX</summary>

```bibtex
@misc{chen2026justgoal,
  title  = {JustGoal: Surpassing Expert Kernels and Specialized Kernel Agents with Only a Goal},
  author = {Chen, Yaojian and Peng, Dazhi and Zhu, Zhui and Li, Leshan and Sun, Haojun
            and Zheng, Yiyun and He, Qingfeng and Li, Shangzhan and Deng, Xianglong
            and Shen, Yifei and Xiang, Shengye and Li, Zhen and Gan, Lin and Li, Yuxuan
            and Han, Xu and Liu, Zhiyuan},
  year   = {2026},
  howpublished = {\url{https://github.com/vincent317/Crossing-The-Expert-Line-Without-Expert-Machinery}}
}
```

</details>

## 许可证

本仓库采用 [Apache License 2.0](LICENSE)。改编自 NVIDIA CUTLASS 示例的文件保留其
BSD-3-Clause 头部和许可，清单见 [`NOTICE`](NOTICE)。

## 致谢

本工作基于 [FlashInfer](https://github.com/flashinfer-ai/flashinfer)、
[flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) 与 MLSys 2026
竞赛 workload、[FlashAttention](https://github.com/Dao-AILab/flash-attention) 以及
[CUTLASS / CuTe DSL](https://github.com/NVIDIA/cutlass)。

本工作来自[清华大学](https://www.tsinghua.edu.cn/)、[面壁智能](https://modelbest.cn/)、
[哈尔滨工业大学](https://www.hit.edu.cn/)和[中国科学院大学](https://www.ucas.ac.cn/)。
