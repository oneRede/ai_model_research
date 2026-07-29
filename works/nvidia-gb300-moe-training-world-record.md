---
sourceTitle: "Setting a World Record for MoE Pre-Training on NVIDIA GB300 NVL72"
title: "在 NVIDIA GB300 NVL72 上创造 MoE 预训练世界记录"
sourceUrl: "https://developer.nvidia.com/blog/setting-a-world-record-for-moe-pre-training-on-nvidia-gb300-nvl72/"
sourceAuthor: "Kirthi Devleker"
author: "Kirthi Devleker"
sourcePublishedAt: "2026-07-21T18:30:00+00:00"
publishedAt: "2026-07-21"
sourceSiteName: "NVIDIA Technical Blog"
siteName: "NVIDIA 技术博客"
coverImage: "imgs/img-008-World-Record.webp"
translator: "Claude (Opus 5)"
translatedAt: "2026-07-29"
language: "zh-CN"
---

# 在 NVIDIA GB300 NVL72 上创造 MoE 预训练世界记录

前沿模型预训练已普遍采用混合专家模型（**Mixture of Experts, MoE**），这从根本上改变了大规模 AI 训练的瓶颈。当每 token 计算量降低时，通信效率成为决定模型能否高效扩展到数千 GPU 的关键。NVIDIA GB300 NVL72 在 DeepSeek-V3 671B 模型的预训练中创造了每 GPU 1,648 万亿次浮点运算（**TFLOPs**）的世界记录，展示了从芯片到网络再到软件的整个 AI 平台的全方位进步如何持续推动训练性能向前发展。预训练效率的每一次提升，都意味着研究人员可以在相同的 NVIDIA 基础设施上训练更大的模型、运行更多实验、更快地达到前沿能力。

行业快速转向 MoE 架构的驱动力在于其巨大的计算效率优势。在稠密模型中，每个 token 会激活所有参数，每 token 的计算量随参数总数线性增长。而 MoE 模型只激活每 token 的一部分参数。例如，DeepSeek-V3 拥有 6710 亿参数，但每 token 只激活约 370 亿参数，以远小于同等规模稠密模型的每 token 成本达到前沿级性能。

但这种效率的代价是通信开销。那些专家分布在不同的 GPU 上，因此每个 MoE 层都必须在前向和反向传播过程中，通过全对全通信（**all-to-all communication**）模式将每个 token 分发到其对应的专家，并收集计算结果。这种集体通信处在训练的关键路径上，使得吞吐量对通信的依赖程度不亚于对计算的依赖。由于这种通信发生在每一层的每个训练步骤中，微小的延迟不断累积放大，直到计算无法再掩盖全对全通信的延迟，此时增加 GPU 也无法提升吞吐量。

![MoE 训练步骤示意图，展示前向传播、反向传播、梯度同步和优化器步骤。在每一层内，注意力机制通过张量并行全归约和全对全通信连接到专家。逐层通信位于关键路径上；梯度同步每步执行一次，可以与计算重叠。](imgs/img-001-image-59.webp)

**图 1.** MoE 训练步骤解剖，展示通信发生的位置及其对性能的影响

这就是为什么预训练需要一个紧密耦合的纵向扩展域（**scale-up domain**）。在这个域内，每个 GPU 都能通过无阻塞网络结构（**non-blocking fabric**）与其他所有 GPU 通信，该网络结构提供高带宽、低延迟和完整的对分带宽（**bisection bandwidth**）。训练这种规模的模型需要的 GPU 数量超过单个域的容量，因此必须将多个域连接起来。这种横向扩展（**scale-out**）流量负载更小、频率更低，但仍必须在计算窗口内完成并保持可预测性，确保没有单个慢速链路成为瓶颈。挑战是双层的，成功的衡量标准是实际算力（**delivered FLOPs**），而非峰值算力（**peak FLOPs**）。

### NVIDIA GB300 NVL72：为紧密耦合的 AI 预训练而生

双层通信挑战需要一个围绕它设计的系统，而不仅仅是一块更快的芯片。计算、纵向扩展互联、横向扩展网络、基础设施处理和软件各自承担部分负载，任何一个环节的不足都会限制整体性能。GB300 NVL72 通过极致的协同设计（**co-design**）整体解决这些挑战——芯片、互联、网络和软件统一设计为一个整体平台，而非简单的部件组装。这是一个机架级系统。

其核心是 NVIDIA NVLink，这个纵向扩展网络结构让 72 个 NVIDIA Blackwell Ultra GPU 作为一个整体工作。第五代 NVLink 为每个 GPU 提供 1.8 TB/s 的带宽，整个机架的无阻塞全对全带宽达到 130 TB/s，因此每个 GPU 只需一跳就能到达任何其他 GPU。

带宽只是故事的一半，路径才是另一半。NVLink 采用内存语义（**memory-semantic**）：GPU 通过无损流控网络结构（**lossless, flow-controlled fabric**），以原生加载和存储操作的方式直接读写对等 GPU 的高带宽内存（**HBM**）。传输是硬件内存操作而非软件发送，因此数据路径中没有任何延迟，归约操作可以在数据流经交换机时就在交换机内部完成。

这满足了逐层流量的要求：张量并行全归约（**tensor-parallel all-reduce**）和 MoE 全对全通信留在机架内部，以全带宽和低延迟运行。在机架之外，平台通过每 GPU 800 Gbps 的 [NVIDIA ConnectX-8 SuperNIC](https://www.nvidia.com/en-us/networking/products/ethernet/supernic/)，配合 [NVIDIA Quantum-X800 InfiniBand](https://www.nvidia.com/en-us/networking/products/infiniband/quantum-x800/) 或 [NVIDIA Spectrum-X 以太网](https://www.nvidia.com/en-us/networking/spectrumx/)，进行横向扩展，将梯度流量隐藏在计算之后。

![GPU 到 GPU 内存访问路径对比。NVLink 纵向扩展通过 NVLink 交换机以两跳低延迟连接 GPU 到对等 HBM。传统纵向扩展通过 PCIe、网卡和分组交换机路由，经过六个阶段，增加了更高的延迟。](imgs/img-002-image-55.webp)

**图 2.** NVLink 网络结构与传统网络结构的对比

协同设计不仅限于训练网络结构，还延伸到基础设施服务和软件。在生产 AI 工厂中，[NVIDIA BlueField](https://www.nvidia.com/en-us/networking/products/data-processing-unit/) 数据处理单元（**DPU**）提供隔离的基础设施处理域，用于虚拟网络、存储访问、安全、遥测和生命周期管理，减少大规模训练作业的宿主 CPU 开销。

训练软件覆盖生态系统的两端。NVIDIA Megatron Core 专为这些 GPU 构建和调优。NVIDIA 还积极为开源框架贡献代码，确保 TorchTitan 和 JAX 在 GB300 NVL72 系统上全速运行。

## Megatron Core 的卓越预训练性能

在 DeepSeek-V3 671B 模型上，使用 256 个 GPU，[Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) 在 GB300 NVL72 上达到每 GPU 1,648 TFLOPs，而早期 GB200 NVL72 的结果为每 GPU 606 TFLOPs，单代提升约 3 倍实际吞吐量。

![柱状图对比 DeepSeek-V3 671B 在 256 个 GPU 上的实际训练性能（TFLOPs/GPU）。GB200 NVL72 达到 606 TFLOPs/GPU，GB300 NVL72 达到 1,648 TFLOPs/GPU —— 相同 GPU 数量下性能提升 3 倍。](imgs/img-003-image-61.webp)

**图 3.** GB200 NVL72（早期软件版本）vs GB300 NVL72（最新软件版本）性能提升 3 倍

NVIDIA 持续优化软件以从整体平台获得更好的性能。在 DeepSeek-V3 671B 规模的预训练工作负载上，这些优化的收益会不断累积。在相同的 GB300 NVL72 机架级系统上，完全由软件改进驱动，性能在六个月内提升了 1.5 倍。这表明，即使在芯片出货后很久，原始性能和训练吞吐量仍在持续改进。

![柱状图展示 DeepSeek-V3 671B 在 256 个 GPU 上的实际训练性能（TFLOPs/GPU），在不变的 GB300 NVL72 硬件上测量。性能从 6 个月前的 1,088 上升到今天的 1,648，提升约 1.5 倍，完全由软件优化驱动。](imgs/img-004-image-53.webp)

**图 4.** 六个月内仅通过软件在 GB300 NVL72 上实现 1.5 倍性能提升

## NVIDIA 加速的领先预训练框架

NVIDIA 工程师直接为 AI 社区依赖的开源框架贡献代码，添加使其在 NVIDIA GPU 上运行更快的优化。这些与 PyTorch 和 JAX 社区协作开发的贡献持续落地，性能随时间不断改进。

TorchTitan 是 PyTorch 的原生训练栈（**PyTorch's native training stack**），NVIDIA 的贡献持续改进其在 GB300 NVL72 上的性能。在 DeepSeek-V3 671B 上，这些优化累积起来，在相同基础设施上实现了约 6 倍的实际性能提升。

![TorchTitan 在 GB300 NVL72 上实际性能演进的柱状图（TFLOPs/GPU，DeepSeek-V3 671B）。经过连续的优化阶段，性能总体提升约 6 倍，由 NVIDIA 对 TorchTitan 框架的软件贡献驱动。](imgs/img-005-image-63.webp)

**图 5.** GB300 NVL72 随 TorchTitan 优化的性能演进

JAX 遵循同样的轨迹。在六个月内，NVIDIA 的 [JAX 优化](https://github.com/NVIDIA/JAX-Toolbox)在 DeepSeek-V3 671B 的 256 GPU 规模上实现了近 10 倍的性能提升——全部来自软件优化。最新软件版本达到了 1,025 TFLOPs/GPU 的卓越性能吞吐量，软件优化仍在持续演进。

![JAX 在 GB300 NVL72 上实际性能演进的柱状图（TFLOPs/GPU，DeepSeek-V3 671B，256 个 GPU），从 2026 年 1 月到今天。性能在六个月内提升近 10 倍，由 NVIDIA 软件优化驱动。](imgs/img-006-image-73.webp)

**图 6.** GB300 NVL72 随 JAX 优化的性能演进

将 DeepSeek-V3 671B 预训练从 256 个 GPU 扩展到 1,024 个 GPU，Megatron Core 保持了 98.5% 的每 GPU 性能，而 TorchTitan 和 JAX 各保持 97%，因此几乎所有新增的基础设施都转化为系统级每秒 token 吞吐量的增加。这种效率是横向扩展网络结构高效工作的体现，随着机架以每 GPU 800 Gb/s 的网络扩展，梯度流量保持隐藏在计算之后，确保增加更多 GPU 严格增加总系统吞吐量，而不会因通信开销拖累网络。

![Megatron Core、TorchTitan 和 JAX 在 GB300 NVL72 上预训练 DeepSeek-V3 671B 的扩展效率图表。三者都从 256 个 GPU 时的 100% 开始，随着数量增长到 512 和 1,024，保持在高位，其中 Megatron Core 保持 98.5%，TorchTitan 和 JAX 各保持约 97%。](imgs/img-007-image-57.webp)

**图 7.** 在 GB300 NVL72 上扩展 DeepSeek-V3 671B 预训练。Megatron Core、TorchTitan 和 JAX 在从 256 扩展到 1,024 个 GPU 时都保持了每 GPU 吞吐量

## 创造世界记录

使用 256 个 GPU 预训练 DeepSeek-V3 671B，在 GB300 NVL72 上实现了每 GPU 1,648 TFLOPs 的世界记录，使相同的训练作业能够用上一代硬件的一小部分达到相同性能。

开源框架讲述着同样的故事。随着软件的演进，相同平台上的性能持续改进。这些结果不是天花板——它们来自一个硬件、互联和软件共同设计并持续优化的平台。今天创纪录的性能只是明天更高性能的基础。

## 在 NVIDIA AI 基础设施上开始训练前沿模型

- 探索机架级架构：了解 [GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)。
- 扩展您的工作负载：开始使用 [Megatron-Core 进行前沿模型训练](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/overview.html)。
- 优化性能：阅读 NVIDIA 的 [CuteDSL 融合内核博客文章](https://developer.nvidia.com/blog/boosting-moe-training-throughput-with-advanced-fusion-kernels/)，了解如何使用 CUDA 图加速性能。

***致谢***

在 NVIDIA GB300 NVL72 平台上通过预训练 DeepSeek-V3 模型创造世界记录，反映了公司众多杰出工程师的工作。我们要感谢以下个人的贡献（按姓氏排序）：

*Aidyn Aitzhan, Michael Andersch, Jan Bernloehr, Santosh Bhavani, Ben Cashman, Carlo del Mundo, Ashraf Eassa, Fabio Paes Leme Ferriani, Matt Frank, Abhinav Goel, Vivek Goel, Elfie Guo, Eric Harper, Munira Hussain, Tomasz Jakubek, Masaki Kozuki, George Kurian, Himangshu Lahkar, Guihong Li, Kibibi Moseley, Nitin Nitin, Devin O'Kelly, Christian M. Sarofeen, Priya Sethuraman, Tejash Shah, Franciszek Szarwacki, John Tran, Qiyu Wan, 和 Cliff Woolley。*
