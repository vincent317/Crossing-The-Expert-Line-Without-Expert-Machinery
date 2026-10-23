<div align="center">

# Crossing the Expert Line Without Expert Machinery

**前沿编程 agent 在朴素循环中写出的 GPU 算子，在 NVIDIA B200 上超过专家手写算子**

[English](README.md) | **中文**

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](paper.pdf)
[![Kernels](https://img.shields.io/badge/Kernels-B200%20sm__100a-76b900)](#仓库结构)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/vincent317/Crossing-The-Expert-Line-Without-Expert-Machinery?style=flat)](https://github.com/vincent317/Crossing-The-Expert-Line-Without-Expert-Machinery/stargazers)
[![Issues](https://img.shields.io/github/issues/vincent317/Crossing-The-Expert-Line-Without-Expert-Machinery)](https://github.com/vincent317/Crossing-The-Expert-Line-Without-Expert-Machinery/issues)

<img src="assets/figure1.png" width="60%" alt="三代模型相对专家参照的加速比">

</div>

本仓库发布论文 [*Crossing the Expert Line Without Expert Machinery*](paper.pdf) 中生成的
算子（kernel）代码。通用编程 agent 只拿到一个 workload shape 和一段简短目标，在**朴素循环**
（naive loop）中迭代：不提供任何算子专用的评测设施、领域知识或工作流，正确性检查和计时
都由 agent 自己编写。使用 **Fable 5.1** 时，生成的算子在 FlashAttention-4 反向、
DeepSeek Sparse Attention 和 Gated DeltaNet 上达到最快专家参照的 **1.08×–1.58×**。

## 动态

- 📌 **置顶**：论文草稿见 [`paper.pdf`](paper.pdf)。
- **2026-10-23**：已发布算子的复测结果（含判定为 invalid 的目录）见 [`benchmarks/flashinfer/`](benchmarks/flashinfer/remainder-03413a0-20261022/README.md)。
- **2026-10-22**：算子代码以 Apache License 2.0 发布。
- **2026-09-25**：加入论文草稿。

## 目录

- [仓库结构](#仓库结构)
- [环境](#环境)
- [局限](#局限)
- [参与贡献](#参与贡献)
- [引用](#引用)
- [许可证](#许可证)
- [致谢](#致谢)

## 仓库结构

| 目录 | 内容 |
|-|-|
| [`goal_single_shape/`](goal_single_shape/) | 指定单个 shape 的 `/goal` 实验 |
| [`goal_full_shape/`](goal_full_shape/) | 覆盖算子定义全部 shape 的 `/goal` 实验 |
| [`kda05_reproduction/`](kda05_reproduction/) | KDA-0.5 设定下的实验 |
| [`ablation/`](ablation/) | 消融实验 |
| [`benchmarks/`](benchmarks/) | 复测结果与证据 |

算子按 `<算子>/<shape>/<模型>/` 组织。`goal_full_shape/` 和 `kda05_reproduction/`
中的算子附带 [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) 的
`solution.json`。

## 环境

- GPU：NVIDIA B200（`sm_100a`）
- CUDA：13.0（`nvcc` 13.0.88，驱动 580.105.08）。有两个算子指向 CUDA 13.1 工具链：
  `goal_single_shape/mla_paged_prefill/b1_q33_kv34/opus-5` 和
  `goal_single_shape/gqa_paged_decode/b16_kv2841/opus-5`。
- CuTe DSL（`nvidia-cutlass-dsl`）：4.6.2。`kda05_reproduction/` 中的 DSA、GDN、MoE
  算子使用 KDA-0.5 设定钉住的 4.6.0。

## 局限

- 算子只面向 B200，也只在 B200 上测过。
- 大多数算子针对优化时的 shape 专门化，不是通用库算子。
- 论文是初步草稿。[`benchmarks/`](benchmarks/) 中的复测把部分已发布目录判为 invalid，
  使用具体算子前请先查看
  [`coverage.json`](benchmarks/flashinfer/remainder-03413a0-20261022/coverage.json)。

## 参与贡献

欢迎提 Issue 和 Pull Request，特别是结果错误或编译失败的报告，以及在其他 B200
环境上的复测结果。

## 引用

<details>
<summary>BibTeX</summary>

```bibtex
@misc{chen2026crossing,
  title  = {Crossing the Expert Line Without Expert Machinery},
  author = {Chen, Yaojian and Peng, Dazhi and Li, Leshan and Sun, Haojun
            and Zheng, Yiyun and He, Qingfeng and Zhu, Zhui and Xiang, Shengye
            and Li, Zhen and Gan, Lin and Li, Yuxuan and Han, Xu and Liu, Zhiyuan},
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

本工作来自[清华大学](https://www.tsinghua.edu.cn/)和[面壁智能](https://modelbest.cn/)。
