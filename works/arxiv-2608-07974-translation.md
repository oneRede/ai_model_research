---
title: "ZeroLock：基于模块化更新解耦的并发内存高效 LLM 训练"
sourceTitle: "ZeroLock: Concurrent Memory-Efficient LLM Training via Modular Update Decoupling"
sourceUrl: "https://arxiv.org/html/2608.07974v1"
arxivId: "2608.07974"
authors: "Wentao Dai, Xuanran Li, Yuxiang Zhang, Ming Tang, Chao Huang"
publishDate: "2026-08-08"
subjects: ["Machine Learning (cs.LG)", "Distributed, Parallel, and Cluster Computing (cs.DC)"]
pipelineRunId: "20260816-214052"
pipelineSource: "translate/20260816-214052/works-ready/arxiv-2608-07974-translation.md"
translationMode: "refined"
language: "zh-CN"
sourceLanguage: "en"
sourceFigureCount: 2
---## ZeroLock：基于模块化更新解耦的并发内存高效 LLM 训练

Wentao Dai、Yuxiang Zhang 和 Ming Tang 来自中国南方科技大学计算机科学与工程系。Xuanran Li 来自中国南方科技大学数学系。Chao Huang 来自美国新泽西州蒙特克莱尔州立大学计算学院。邮箱：{12311217,12312110,12410823}@mail.sustech.edu.cn, tangm3@sustech.edu.cn, huangch@montclair.edu。（通讯作者：Ming Tang）\* 同等贡献

Wentao Dai <sup>∗</sup>, Xuanran Li <sup>∗</sup>, Yuxiang Zhang, Ming Tang, Chao Huang

###### 摘要

在边缘设备上微调大语言模型（Large Language Model, LLM）可将模型适配到特定场景数据，同时保护隐私。尽管现有研究提出了流水线并行（pipeline parallelism）来应对边缘设备有限的内存和计算资源，但它们通常依赖反向传播（backpropagation, BP）训练。BP 训练存在更新锁定（update locking）的根本限制，可能导致严重的吞吐量和内存瓶颈。本文提出了一种无反向传播（BP-free）算法 ZeroLock，通过局部目标构建（local objective construction）将模型更新解耦为独立的分块更新。它打破了 BP 的更新锁定，从而在算法层面提高吞吐量，并通过减少激活值（activation）存储降低内存使用。就我们所知，本文首次为基于局部目标构建的方法提供了通用模型分块划分下的理论框架，通过将局部目标映射到全局目标来实现。我们证明 ZeroLock 的收敛速率（convergence rate）为 $\tilde{\mathcal{O}}(1/\sqrt{T})$，与 BP 仅相差多重对数因子（polylogarithmic factors）。我们为 ZeroLock 设计了系统并构建了真实原型，融合了早转发（early forwarding）和故障恢复（failure recovery）等技术，实现高效且鲁棒的实现。在原型上的实验表明，与基于 BP 的基线相比，ZeroLock 将内存减少了 26.5%，吞吐量提高了 4.9%。

## I 引言

边缘智能（edge intelligence）将大语言模型部署于网络边缘，实现隐私保护并提供实时推理服务。在实践中，在边缘设备上微调 LLM 非常重要，原因有二。首先，数据分布通常具有场景和个体特异性，因此必须适配模型来维持推理准确性。其次，数据隐私问题以及可能频繁的模型适配需求，使得将数据上传到云端微调变得困难。有许多此类示例同时需要模型适配和隐私保护。例如，测试时训练（test-time training）[^27] 使 LLM 适应并保持用户特定记忆，以提供个性化服务。基于脑电图信号的任务 [^2]（例如情绪识别、睡眠分期）需要为个体适配模型，因为这些信号表现出强烈的个体依赖模式。

然而，网络边缘的设备通常具有有限的计算和内存资源，因此单个设备或 GPU 难以微调整个 LLM。为了解决这一问题，现有研究提出了**流水线并行**方法 [^13] [^20] [^6]。其主要思想是将模型垂直划分为由连续层组成的分块（chunk）。每个模型分块的更新对应一个**阶段（stage）**，并分配给不同的设备，这些设备以流水线方式协作更新分块。GPipe [^13] 是一种典型方法，按顺序处理微批次（micro-batch）。1F1B [^20] 交错执行前向传播和反向更新以提高流水线利用率。在 1F1B 的基础上，PipeDream [^20] [^21] 通过权重暂存（weight stashing）实现异步流水线训练。最近的研究 [^7] [^22] [^16] [^24] 通过放置搜索、虚拟或双向阶段以及更细粒度的反向分解来减少流水线中的气泡（bubble）。其他研究 [^17] [^8] [^29] [^5] [^25] 关注考虑抢占、落后者、设备异构性或通信效率的流水线规划。Confidant [^6] 在智能手机上实现了 1F1B。

(a)

(b)

(c)           (d)

图 1：(a) BP 基线和 (b) ZeroLock 的吞吐量；(c) BP 基线和 (d) ZeroLock 的内存。在此图中，我们使用典型的 1F1B 框架作为 BP 基线，因为许多现有研究（例如 [^6]）都基于它构建。S0、S1 和 S2 是阶段，每个阶段对应一个模型分块的更新并分配给一个设备执行。F $n$、B $n$ 和 A $n$ 分别表示第 $n$ 个微批次的前向传播、反向传播和激活值。(c) 和 (d) 中的条纹表示相应激活值保存在内存中的持续时间。例如，在 S0 的时间槽 3，(c) 中 A1-A3 保存在内存中，而 (d) 中仅 A2 保存在内存中。

这些流水线并行工作（例如 [^13] [^20] [^21] [^7] [^22] [^16] [^24] [^17] [^8] [^29] [^6] [^5] [^4] [^25]）大多依赖于反向传播训练，并专注于系统级调度优化。BP 训练存在更新锁定的根本限制 [^14] [^30] [^31]。也就是说，模型更新包含前向传播后跟反向传播，因此上游层的更新需要等待下游层的前向和反向计算。因此，基于 BP 的方法可能存在关键限制：

BP 训练的普遍使用以及由此产生的吞吐量和内存瓶颈引出了主要问题：

###### 主要问题

我们如何设计一个系统，通过打破 BP 中的更新锁定，从根本上在算法层面克服吞吐量和内存瓶颈？

无反向传播训练被提出来打破 BP 中的更新锁定。主要有两类。(I) **反向梯度估计**。例如，直接反馈对齐 [^23] [^1] 通过将目标误差直接传播到所有层来绕过链式法则。零阶优化 [^18] [^28] 在前向传播期间计算基于扰动的损失差异来估计梯度。然而，这一类方法通常会导致复杂任务的模型准确性显著下降或显著的计算开销。(II) **目标重构**。例如，NoProp [^15] 将神经模块重新定义为独立的去噪单元，将噪声目标嵌入映射回干净目标。预测编码 [^19] 交替进行分块级预测误差最小化和使用局部误差更新权重。局部目标构建（例如，深度渐进单调学习 [^30]）为每一层配备独立的局部目标，允许纯粹的局部梯度计算。在类别 (II) 中，局部目标构建解耦了跨阶段的模块化更新，并具有与 BP 方法相当的模型准确性，因此是解决主要问题的有希望的候选方案。具体而言，由于每个分块使用其局部损失独立更新，其更新不需要等待其下游分块的前向和反向传播，因此可以消除由于等待而产生的气泡；同时，每个分块只需保留其激活值直到其自身更新完成，从而减少内存使用。

在本文中，基于局部目标构建的无反向传播方法，我们旨在提出一个流水线并行 LLM 训练系统，从根本上克服吞吐量和内存瓶颈。然而，这一设计并不简单，需要解决以下问题：

- 如何使用局部目标构建微调 LLM？
- 局部目标构建的模块解耦是否在理论上损害模型收敛？
- 如何设计和构建真实原型系统（针对多 GPU 服务器和 Android 手机场景）以实现高效且鲁棒的实现？

回答 Q1 需要具体设计，将 LLM 特定特征（例如，token 序列、用于高效微调的低秩适应）融入局部目标构建框架。回答 Q2 非常具有挑战性。这是因为模块化更新解耦将全局目标分解为模型分块的局部目标，而最近的研究缺乏量化优化局部目标如何改进全局目标的框架。由于轻量级、高吞吐量和鲁棒实现的真实工程需求，回答 Q3 并非易事。

现有工作尝试解决 Q1–Q3 中的一个或两个问题。PPLL [^9] 将梯度隔离的视觉块放置在不同的 GPU 上，并传输其特征进行块级更新。FluidPipe [^3] 关注 LLM，并在两阶段模型的第一部分添加辅助任务头，避免第一部分的局部更新等待第二部分的梯度。SCPL [^11] 通过每段监督对比损失解耦 BP，用于同步多 GPU 模型并行。然而，这些工作 [^9] [^3] [^11] 在 Q1 中考虑了隐藏状态对齐，未能在局部目标构建中刻画任务特定信息，并且没有解决 Q2。对于 Q3，虽然他们提出了高级流水线逻辑，但他们的设计仍停留在概念层面，缺乏关键的实现细节，如连续跨批次流水线、阶段 I/O 队列和显式 RPC 原语。同时，他们没有提供可部署的原型用于真实执行。尽管最近的工作 LoPT [^26] 解决了 Q2，但其分析仅限于两个分块，未能为连接局部目标与全局目标提供通用分析框架。

我们解决了这些挑战并回答了 Q1–Q3。我们的主要贡献总结如下：

- **ZeroLock 算法**：为了回答 Q1，我们提出了一种基于局部目标构建的无反向传播算法 ZeroLock，用于 LLM 微调，该算法融合了低秩适应（LoRA）[^12] 以及与 LLM 兼容的读出头（readout head）和损失函数。它打破了跨模型分块的更新耦合，从而在算法层面减少流水线气泡和激活值存储。
- **理论分析**：为了回答 Q2，我们建立理论等价性，将模型分块的局部最优用全局目标的形式表示。然后，模型分块的局部更新可等价表示为全局更新，由此可推导出全局收敛性。就我们所知，这是通用模型分块划分下局部目标构建算法的首个分析框架，为在给定解耦局部更新的情况下分析全局收敛提供了系统方法。我们证明 ZeroLock 的收敛速率为 $\tilde{\mathcal{O}}(1/\sqrt{T})$，与 BP 仅相差多重对数因子。
- **真实系统设计**：为了回答 Q3，基于 ZeroLock，我们为多 GPU 服务器和 Android 手机场景设计了系统并构建了原型。该系统可实现并发吞吐量（见图 1 (b)）并通过减少激活值存储降低内存使用（见图 1 (d)）。我们提出了早转发和故障恢复等技术，确保轻量级、高吞吐量和鲁棒的实现。就我们所知，我们构建了首个基于局部目标构建的无反向传播算法在 Android 系统上的原型。
- **真实原型评估**：在多 GPU 服务器上的实验表明，与基于 BP 的基线相比，ZeroLock 系统将内存使用减少了 26.5%，吞吐量提高了 4.9%。在 Android 手机上微调 TinyLlama 的峰值 PSS 小于 4000 MiB，电池温度约为 37 <sup>∘</sup> C，挂钟时间为 1644.1 秒，表明该实现在实践中是可行的。我们的代码可在 [https://anonymous.4open.science/r/unlock/_trainer-105B](https://anonymous.4open.science/r/unlock_trainer-105B) 获取。

本文其余部分组织如下。第 II 节和第 III 节分别介绍 ZeroLock 算法及其系统设计。实验在第 IV 节。第 V 节总结本文。

## II ZeroLock 算法与分析

### II-A ZeroLock 算法

考虑一个具有嵌入操作 $E(x)$ 和一组 transformer 层 $\mathcal{L}$ 的 LLM。我们引入 LoRA [^12] 以减少微调中的可训练参数。令 $w_{l}\in\mathbb{R}^{d_{1}\times d_{2}}$ 表示第 $l$ 层 transformer 的基础权重，在微调期间被冻结。令 $A_{l}\in\mathbb{R}^{d_{3}\times d_{2}}$ 和 $B_{l}\in\mathbb{R}^{d_{1}\times d_{3}}$ 表示 LoRA 中可训练的低秩分解矩阵。那么，transformer 层 $l\in\mathcal{L}$ 的参数更新表示为 $\Delta w_{l}=B_{l}A_{l}$，因此微调后的权重为 $w_{l}+\Delta w_{l}$。为了在 LLM 微调中实现模块化更新解耦（modular update decoupling），我们将整个模型划分为模型分块，并为这些分块引入局部目标以进行独立更新。<sup>1</sup>

**分块划分**：考虑将 LLM 划分为一个嵌入操作 $E(\cdot)$ 和 $K$ 个分块，其中分块 $k$ 包含一个或多个连续的 transformer 层，用集合 $\mathcal{L}_{k}$ 表示。令 $W_{k}=(w_{l}~|~l\in\mathcal{L}_{k})$ 和 $\Delta W_{k}=(\Delta w_{l}~|~l\in\mathcal{L}_{k})$ 分别表示分块 $k$ 中冻结的基础权重和可训练的 LoRA 参数，从而得到从分块输入到输出的映射 $f_{k}(\cdot;W_{k}+\Delta W_{k})$。令 $x$ 为 LLM 输入。定义 $\boldsymbol{h}_{0}\in\mathbb{R}^{S\times d}$ 和 $\boldsymbol{h}_{k}\in\mathbb{R}^{S\times d}$ 分别为 token 嵌入和分块 $k$ 的输出，其中 $S$ 为序列长度：

$$
\boldsymbol{h}_{0}=E(x),~\boldsymbol{h}_{k}=f_{k}(\boldsymbol{h}_{k-1};W_{k}+\Delta W_{k}),k=1,\cdots,K.
$$

**读出头**：为了实现局部更新，我们在分块 $k=1,\cdots,K$ 之后引入一个冻结的局部读出头：

$$
\boldsymbol{z}_{k}=\mathrm{Norm}(\boldsymbol{h}_{k})W_{\mathrm{lm}}^{\top},~\boldsymbol{p}_{k,i}=\mathrm{softmax}(\boldsymbol{z}_{k,i}).
$$

在 (2) 中，$W_{\mathrm{lm}}$ 是将归一化隐藏表示 $\mathrm{Norm}(\boldsymbol{h}_{k})$ 映射到词汇表 logits $\boldsymbol{z}_{k}\in\mathbb{R}^{S\times V}$ 的头权重，其中 $V$ 为词汇表大小。向量 $\boldsymbol{z}_{k,i}$ 是 $\boldsymbol{z}_{k}\in\mathbb{R}^{S\times V}$ 的第 $i$ 行。通过 $\mathrm{softmax}(\cdot)$，$\boldsymbol{p}_{k,i}\in\mathbb{R}^{1\times V}$ 是词汇表空间上下一个 token 的预测分布。

**局部目标**：利用读出头的输出，每个分块 $k$ 的局部损失由任务相关项（task-dependent term）$L_{\mathrm{Task}}(\cdot)$ 和一致性项（consistency term）$L_{\mathrm{Consis}}^{k\rightarrow k-1}(\cdot)$ 组成：

$$
L_{k}(\boldsymbol{p}_{k})=\alpha L_{\mathrm{Task}}(\boldsymbol{p}_{k})+(1-\alpha)L_{\mathrm{Consis}}^{k\rightarrow k-1}(\boldsymbol{p}_{k}),
$$

其中 $\alpha\in(0,1]$ 是平衡参数，$\boldsymbol{p}_{k}=(\boldsymbol{p}_{k,i},i\in\Omega)$，$\Omega$ 为移除提示或填充位置的 token 序列。

(i) **任务相关项** $L_{\mathrm{Task}}$ 是全局目标的局部版本。它将每个分块 $k$ 的读出头输出与全局真实目标对齐：

$$
L_{\mathrm{Task}}(\boldsymbol{p}_{k})=-\frac{1}{|\Omega|}\sum_{i\in\Omega}D_{\psi}\!\left(\boldsymbol{p}_{k,i},\boldsymbol{p}_{y}\right),
$$

其中 $\boldsymbol{p}_{y}\in\mathcal{R}^{1\times V}$ 是 one-hot 目标。令 $\mathcal{P}$ 表示 $\boldsymbol{p}$ 的空间。$D_{\psi}(\cdot)$ 是由严格凸且可微的势函数 $\psi:\mathcal{P}\rightarrow\mathbb{R}$ 诱导的 Bregman 散度（Bregman divergence），即 $D_{\psi}(\boldsymbol{u},\boldsymbol{v})=\psi(\boldsymbol{u})-\psi(\boldsymbol{v})-\langle\nabla\psi(\boldsymbol{v}),\boldsymbol{u}-\boldsymbol{v}\rangle$。KL 散度是 Bregman 散度的一个特例，通过选择负熵作为势函数得到。重要的是，由于最小化 KL 散度等价于最小化交叉熵，因此在实践中 (4) 可以用交叉熵损失替代。

(ii) **一致性项**将分块 $k$ 的输出与分块 $k-1$ 的输出对齐，确保跨分块的输出一致性。

$$
L_{\mathrm{Consis}}^{k\rightarrow k-1}(\boldsymbol{p}_{k})=\frac{1}{|\Omega|}\sum_{i\in\Omega}D_{\psi}\!\left(\boldsymbol{p}_{k,i},\mathrm{sg}(\boldsymbol{p}_{k-1,i})\right),
$$

其中 $\mathrm{sg}(\cdot)$ 是停止梯度算子（stop-gradient operator），满足 $\mathrm{sg}(\boldsymbol{p})=\boldsymbol{p}$ 和 $\nabla\mathrm{sg}(\boldsymbol{p})=\boldsymbol{0}$。类似地，在实践中可以使用 KL 散度。

**微调过程**：此过程包含 $T$ 次迭代。在每次迭代中，层 $l\in\mathcal{L}_{k}$ 的低秩矩阵 $A_{l}$ 和 $B_{l}$ 使用随机梯度下降进行微调：

$$
A_{l}\leftarrow A_{l}-\eta\frac{\partial f_{k}}{\partial A_{l}},~B_{l}\leftarrow B_{l}-\eta\frac{\partial f_{k}}{\partial B_{l}},
$$

其中 $\eta$ 是学习率。

###### 备注 1

根据 (6)，同一分块内的层使用 BP 更新，而来自不同分块的层的更新是解耦的，可以并行执行。这在算法层面消除了并行气泡（见图 1 (b)），并避免了在下游更新期间保留激活值的需要（见图 1 (d)）。

### II-B 理论分析：分块级性能

通过分块级性能分析，我们旨在提供关于全局目标值如何跨模型分块变化的见解。我们关注序列中的一个 token 并省略 token 下标。回顾 $\boldsymbol{p}$ 是分块的读出头输出，是词汇表空间上的分布。令 $L:\mathcal{P}\rightarrow\mathbb{R}$ 表示给定 $\boldsymbol{p}\in\mathcal{P}$ 的全局目标，即 $L(\boldsymbol{p})\triangleq L_{\mathrm{Task}}(\boldsymbol{p})$，这是表征 $\boldsymbol{p}$ 偏离全局真实目标程度的损失函数。令 $\boldsymbol{p}_{k}^{*}$ 表示最小化局部损失 $L_{k}(\cdot)$ 的分块 $k$ 的最优 $\boldsymbol{p}_{k}$，即 $\boldsymbol{p}_{k}^{*}\triangleq\arg\min_{\boldsymbol{p}\in\mathcal{P}}L_{k}(\boldsymbol{p})$：

###### 引理 1（局部最优）

局部最优 $\boldsymbol{p}_{k}^{*}$ 可以等价表示为

$$
\boldsymbol{p}_{k}^{*}\!=\!\arg\min_{\boldsymbol{p}\in\mathcal{P}}\left\{\langle\nabla L(\boldsymbol{p}_{k\!-\!1}),\boldsymbol{p}\!-\!\boldsymbol{p}_{k\!-\!1}\rangle\!+\!\frac{1}{\alpha}D_{\psi}(\boldsymbol{p},\boldsymbol{p}_{k\!-\!1})\right\}.
$$

###### 证明

回顾 Bregman 散度的定义，

$$
D_{\psi}(\boldsymbol{u},\boldsymbol{v})=\psi(\boldsymbol{u})-\psi(\boldsymbol{v})-\langle\nabla\psi(\boldsymbol{v}),\boldsymbol{u}-\boldsymbol{v}\rangle.
$$

将 $(\boldsymbol{u}=\boldsymbol{p},\boldsymbol{v}=\boldsymbol{p}_{k-1})$ 和 $(\boldsymbol{u}=\boldsymbol{p},\boldsymbol{v}=\boldsymbol{p}_{y})$ 分别代入 (8) 并重新整理前者，我们得到 $\psi(\boldsymbol{p})\!=\!\psi(\boldsymbol{p}_{k\!-\!1})\!+\!\langle\nabla\psi(\boldsymbol{p}_{k\!-\!1})\!,\!\boldsymbol{p}-\boldsymbol{p}_{k\!-\!1}\rangle\!+\!D_{\psi}(\boldsymbol{p},\boldsymbol{p}_{k\!-\!1})$，以及 $D_{\psi}(\boldsymbol{p},\boldsymbol{p}_{y})=\psi(\boldsymbol{p})-\psi(\boldsymbol{p}_{y})-\langle\nabla\psi(\boldsymbol{p}_{y}),\boldsymbol{p}-\boldsymbol{p}_{y}\rangle$。因此，

$$
\displaystyle D_{\psi}(\boldsymbol{p},\boldsymbol{p}_{y}){=}
$$

$$
\displaystyle\psi(\boldsymbol{p}_{k-1})-\psi(\boldsymbol{p}_{y})-\langle\nabla\psi(\boldsymbol{p}_{y}),\boldsymbol{p}_{k-1}-\boldsymbol{p}_{y}\rangle
$$

$$
\displaystyle-\langle\nabla\psi(\boldsymbol{p}_{y}),\boldsymbol{p}-\boldsymbol{p}_{y}\rangle+\langle\nabla\psi(\boldsymbol{p}_{y}),\boldsymbol{p}_{k-1}-\boldsymbol{p}_{y}\rangle
$$

$$
\displaystyle+\langle\nabla\psi(\boldsymbol{p}_{k-1}),\boldsymbol{p}-\boldsymbol{p}_{k-1}\rangle+D_{\psi}(\boldsymbol{p},\boldsymbol{p}_{k-1})
$$

$$
\displaystyle\overset{(a)}{=}
$$

$$
\displaystyle D_{\psi}(\boldsymbol{p}_{k-1},\boldsymbol{p}_{y})+\langle\nabla L(\boldsymbol{p}_{k-1}),\boldsymbol{p}-\boldsymbol{p}_{k-1}\rangle
$$

$$
\displaystyle+D_{\psi}(\boldsymbol{p},\boldsymbol{p}_{k-1}),
$$

其中 (a) 根据 $D_{\psi}(\boldsymbol{p}_{k-1},\boldsymbol{p}_{y})$ 和 $L(\boldsymbol{p}_{i-1})$ 的定义成立。

将 (II-B) 代入局部损失函数 $L_{k}(\boldsymbol{p})$ 得到 $L_{k}(\boldsymbol{p})=\alpha D_{\psi}(\boldsymbol{p}_{k-1},\boldsymbol{p}_{y})+\alpha\langle\nabla L(\boldsymbol{p}_{k-1}),\boldsymbol{p}-\boldsymbol{p}_{k-1}\rangle+D_{\psi}(\boldsymbol{p},\boldsymbol{p}_{k-1})$。给定 $\boldsymbol{p}_{k-1}$，项 $\alpha D_{\psi}(\boldsymbol{p}_{k-1},\boldsymbol{p}_{y})$ 是常数，因此在优化中可以省略。证毕。∎

引理 1 将局部最优 $\boldsymbol{p}^{\star}$ 表示为全局目标 $L(\cdot)$ 的形式，连接了局部目标和全局目标。同时，它展示了在朝向全局目标 $L(\boldsymbol{p})$ 的最陡下降方向移动与惩罚偏离先前分布 $\boldsymbol{p}_{k-1}$ 之间的平衡。

如许多现有的 BP 或无反向传播收敛分析（例如 [^10] [^26]）中那样，我们假设全局目标的平滑性。

###### 假设 1（全局目标平滑性）

全局目标 $L(\cdot)$ 在 Bregman 几何中相对于 $\psi$ 是 $\beta$-平滑的：

$$
L(\boldsymbol{u})\leq L(\boldsymbol{v})+\langle\nabla L(\boldsymbol{v}),\boldsymbol{u}-\boldsymbol{v}\rangle+\beta D_{\psi}(\boldsymbol{u},\boldsymbol{v}),~\boldsymbol{u},\boldsymbol{v}\in\mathcal{P}.
$$

然后，我们推导 ZeroLock 的分块级性能。

###### 命题 1（分块级性能）

假设 $\boldsymbol{p}_{k-1}$、$\boldsymbol{p}_{k}^{\star}$ 和 $\boldsymbol{p}_{k}$ 位于单纯形相对内部的紧集中，且 $\psi(\cdot)$ 是局部强凸的。定义 $\delta_{k}\triangleq D_{\psi}(\boldsymbol{p}_{k},\boldsymbol{p}_{k}^{*})$。在假设 1 和 $\alpha<\frac{1}{\beta}$ 下，

$$
L(\boldsymbol{p}_{K})\!\leq\!L(\boldsymbol{p}_{0})\!-\!\sum_{k=1}^{K}\left(\frac{1}{\alpha}\!-\!\beta\right)D_{\psi}(\boldsymbol{p}_{k},\boldsymbol{p}_{k\!-\!1})\!+\!\frac{1}{\alpha}\sum_{k=1}^{K}\delta_{k}.
$$

###### 证明

在假设 1 中代入 $\boldsymbol{u}=\boldsymbol{p}_{k}$ 和 $\boldsymbol{v}=\boldsymbol{p}_{k-1}$，

$$
L(\boldsymbol{p}_{k})\leq\underbrace{\langle\nabla L(\boldsymbol{p}_{k-1}),\boldsymbol{p}_{k}-\boldsymbol{p}_{k}^{*}\rangle}_{(i)}+\underbrace{\langle\nabla L(\boldsymbol{p}_{k-1}),\boldsymbol{p}_{k}^{*}-\boldsymbol{p}_{k-1}\rangle}_{(ii)}\\
+L(\boldsymbol{p}_{k-1})+\beta D_{\psi}(\boldsymbol{p}_{k},\boldsymbol{p}_{k-1}).
$$

由于 $\boldsymbol{p}_{k}^{\star}$ 位于概率单纯形的相对内部，(7) 的 KKT 条件给出 $\nabla L(\boldsymbol{p}_{k-1})+\frac{1}{\alpha}\bigl(\nabla\psi(\boldsymbol{p}_{k}^{\star})-\nabla\psi(\boldsymbol{p}_{k-1})\bigr)+\lambda_{k}\mathbf{1}=0$。因此，项 (i) 满足

$$
\displaystyle\alpha\langle\nabla L(\boldsymbol{p}_{k-1}),\boldsymbol{p}_{k}-\boldsymbol{p}_{k}^{\star}\rangle
$$

$$
\displaystyle\overset{(a)}{=}
$$

$$
\displaystyle\langle\nabla\psi(\boldsymbol{p}_{k-1})-\nabla\psi(\boldsymbol{p}_{k}^{\star}),\boldsymbol{p}_{k}-\boldsymbol{p}_{k}^{\star}\rangle
$$

$$
\displaystyle\overset{(b)}{=}
$$

$$
\displaystyle\delta_{k}+D_{\psi}(\boldsymbol{p}_{k}^{\star},\boldsymbol{p}_{k-1})-D_{\psi}(\boldsymbol{p}_{k},\boldsymbol{p}_{k-1}).
$$

这里，(a) 根据 KKT 条件和 $\mathbf{1}^{\top}(\boldsymbol{p}_{k}-\boldsymbol{p}_{k}^{\star})=0$ 成立，后者成立是因为 $\boldsymbol{p}_{k}-\boldsymbol{p}_{k}^{\star}$ 位于单纯形切空间中。(b) 基于 Bregman 三点恒等式成立。根据 $\boldsymbol{p}_{k}^{*}$ 的定义，项 (ii) 满足 $\langle\boldsymbol{g}_{k},\boldsymbol{p}_{k}^{*}-\boldsymbol{p}_{k-1}\rangle+\frac{1}{\alpha}D_{\psi}(\boldsymbol{p}_{k}^{*},\boldsymbol{p}_{k-1})\leq 0$。将 (i) 和 (ii) 代入 (12) 并对 $k=1,\dots,K$ 求和完成证明。∎

命题 1 表明，随着分块数量的增加，最终分块的全局损失 $L(\boldsymbol{p}_{K})$ 趋于减少，但存在由于有限参数导致的模型分块微调次优性产生的有界误差 $\sum_{k=1}^{K}\delta_{k}/\alpha$。<sup>2</sup>

### II-C 理论分析：算法收敛

我们为通用分块划分下基于局部目标构建的无反向传播方法提供了首个分析框架。如前所述，主要挑战来自于基于每个模型分块的局部更新来刻画全局收敛。为了克服这一点，我们将局部更新等价映射到全局更新（引理 3）；基于此，我们证明 ZeroLock 的收敛速率为 $\tilde{\mathcal{O}}(1/\sqrt{T})$（定理 1），与 BP 的 $\mathcal{O}(1/\sqrt{T})$ 仅相差一个多重对数因子。<sup>3</sup>

对于迭代 $t$，令 $\boldsymbol{\omega}^{t}=(\boldsymbol{\omega}_{k}^{t},k=0,1,\cdots,K)$ 表示要更新的分块参数。我们引入此符号以推广各种场景（例如 LLM、CNN）的分析。遵循第 II-A 节的符号，$\boldsymbol{\omega}_{0}^{t}$ 是算子 $E(\cdot)$ 的固定参数，对于分块 $k$ 有 $\boldsymbol{\omega}_{k}^{t}\triangleq\Delta W_{k}^{t}+W_{k}$。令 $\boldsymbol{p}^{t}_{0},\boldsymbol{p}^{t}_{1},\dots,\boldsymbol{p}^{t}_{K}$ 表示相关模型分块的读出头输出。为了分析简化，我们将 (4) 和 (5) 中的 Bregman 散度设置为 KL 散度。那么，局部损失等价于

$$
L_{k}^{t}(\boldsymbol{p};\alpha)=\alpha D_{\mathrm{KL}}(\boldsymbol{p}~\|~\boldsymbol{p}_{y})+(1-\alpha)D_{\mathrm{KL}}(\boldsymbol{p}~\|~\boldsymbol{p}_{k-1}^{t}).
$$

**从局部更新映射到全局更新**：首先，我们展示局部目标的等价性如下。

###### 引理 2（目标等价性）

最小化 $L_{k}^{t}(\boldsymbol{p};\alpha)$ 等价于最小化 $D_{\mathrm{KL}}\left(\boldsymbol{p}~\|~\boldsymbol{p}_{k}^{*,t}(\alpha)\right)$，其中

$$
\boldsymbol{p}_{k}^{*,t}(l;\alpha)=\frac{\boldsymbol{p}_{y}(l)^{\alpha}\left(\boldsymbol{p}_{k-1}^{t}(l)\right)^{1-\alpha}}{\sum_{l^{\prime}=1}^{m}\boldsymbol{p}_{y}(l^{\prime})^{\alpha}\left(\boldsymbol{p}_{k-1}^{t}(l^{\prime})\right)^{1-\alpha}},
$$

其中 $l$ 表示相关向量的第 $l$ 个元素，$m$ 表示向量的长度。

###### 证明

局部损失 $L_{k}^{t}(\boldsymbol{p};\alpha)$ 可以等价表示为

$$
L_{k}^{t}(\boldsymbol{p};\alpha)=\sum_{l=1}^{m}\boldsymbol{p}(l)\log\frac{\boldsymbol{p}(l)}{\boldsymbol{p}_{y}(l)^{\alpha}\left(\boldsymbol{p}_{k}^{t}(l)\right)^{1-\alpha}}.
$$

定义 $Z_{k}^{t}(\alpha)=\sum_{l=1}^{m}\boldsymbol{p}_{y}(l)^{\alpha}\left(\boldsymbol{p}_{k-1}^{t}(l)\right)^{1-\alpha}$。那么，

$$
L_{k}^{t}(\boldsymbol{p};\alpha)=D_{\mathrm{KL}}(\boldsymbol{p}~\|~\boldsymbol{p}_{k}^{*,t}(\alpha))-\log Z_{k}^{t}(\alpha).
$$

注意对于任何分块 $k$，向量 $\boldsymbol{p}_{y}$ 和 $\boldsymbol{p}_{k-1}^{t}$ 是给定的，因此 $Z_{k}^{t}(\alpha)$ 是常数。因此，此引理得证。∎

注意 $\boldsymbol{p}_{k}^{\ast,t}(\alpha)\triangleq(\boldsymbol{p}_{k}^{\ast,t}(l;\alpha),l=1,\cdots,m)$ 本质上是最小化 $L_{k}^{t}(\boldsymbol{p};\alpha)$ 的最优解，证明在此省略。因此，最小化 $D_{\mathrm{KL}}\left(\boldsymbol{p}~\|~\boldsymbol{p}_{k}^{*,t}(\alpha)\right)$ 等价于最小化 $\boldsymbol{p}$ 与最优解之间的差异。

然后，我们可以将 ZeroLock 中的全局更新表示为局部更新的形式。我们定义 $\boldsymbol{p}_{k}^{t}=\hat{f}_{k}(\boldsymbol{p}_{k-1}^{t};\boldsymbol{\omega}_{k}^{t})$ 为从分块 $k$ 的读出头输出 $\boldsymbol{p}_{k-1}^{t}$ 到 $\boldsymbol{p}_{k}^{t}$ 的映射，给定分块 $k$ 的参数 $\boldsymbol{\omega}_{k}^{t}$。尽管这个映射不能通过读出头操作直接获得（因为它不是一一对应的），但可以根据 $\boldsymbol{p}_{k}^{t}$ 跨分块的统计数据来近似。此外，引入此映射用于理论分析，在实践中不需要获得。定义雅可比矩阵 $\boldsymbol{J}_{\theta_{k},k}^{t}\triangleq\partial\hat{f}_{k}(\boldsymbol{p}_{k-1}^{t};\boldsymbol{\omega}_{k}^{t})/\partial\boldsymbol{\omega}_{k}^{t}$。停止梯度操作诱导所有参数的以下雅可比矩阵：

$$
\boldsymbol{J}^{t}=\operatorname{blkdiag}(\boldsymbol{J}_{\boldsymbol{\omega}_{1},1}^{t},\boldsymbol{J}_{\boldsymbol{\omega}_{2},2}^{t},\cdots,\boldsymbol{J}_{\boldsymbol{\omega}_{K},K}^{t}),
$$

其中 $\operatorname{blkdiag}(\cdot)$ 表示块对角矩阵，对角块之外的所有元素均为零。定义 $\boldsymbol{e}_{k}^{t}(\alpha)=\log\boldsymbol{p}_{k}^{t}-\log\boldsymbol{p}_{k}^{\ast,t}(\alpha)$，这是最近的 $\boldsymbol{p}_{k}^{t}$ 与最优 $\boldsymbol{p}_{k}^{\ast,t}(\alpha)$ 之间在对数形式下的差距。

基于引理 2 和 $\boldsymbol{p}_{k}^{t}=\hat{f}_{k}(\boldsymbol{p}_{k-1}^{t};\boldsymbol{\omega}_{k}^{t})$，在数据集 $\mathcal{D}$ 上最小化所有分块的 $L_{k}^{t}(\boldsymbol{p};\alpha)$ 等价于找到最小化以下式子的参数 $\boldsymbol{\omega}=(\boldsymbol{\omega}_{k},k=0,1,\dots,K)$

$$
\!\!\!\mathcal{R}_{t}(\boldsymbol{\omega};\alpha)\!=\!\mathbb{E}_{(x,y)\sim\mathcal{D}}\!\!\left[\sum_{k=1}^{K}\!D_{\mathrm{KL}}\!\!\left(\hat{f}_{k}(\boldsymbol{p}_{k\!-\!1}^{t};\boldsymbol{\omega}_{k})\|\boldsymbol{p}_{k}^{\ast,t}(\alpha)\!\right)\!\right]\!.
$$

###### 引理 3（全局更新等价性）

基于引理 2，ZeroLock 的更新规则可以表示如下：<sup>4</sup>

$$
\boldsymbol{\omega}^{t+1}\leftarrow\boldsymbol{\omega}^{t}-\eta_{t}\nabla_{\boldsymbol{\omega}}\mathcal{R}_{t}(\boldsymbol{\omega}^{t};\alpha),
$$

其中 $\nabla_{\boldsymbol{\omega}}\mathcal{R}_{t}(\boldsymbol{\omega};\alpha)=(\boldsymbol{J}^{t})^{\top}\boldsymbol{e}^{t}$，$\boldsymbol{e}^{t}\triangleq(\boldsymbol{e}^{t}_{k},k=0,1,\cdots,K)$。
###### 证明。

根据引理 2 中的式 (17),给定输出 $\boldsymbol{p}_{k}^{t}$,

$$
\nabla_{\boldsymbol{p}_{k}^{t}}L_{k}^{t}(\boldsymbol{p}_{k}^{t};\alpha)=\log\boldsymbol{p}_{k}^{t}-\log\boldsymbol{p}_{k}^{\ast,t}(\alpha)+\mathbf{1}.
$$

我们应用链式法则。令 $\ell_{k}^{t}(\boldsymbol{\omega}_{k}^{t})\triangleq L_{k}^{t}(\boldsymbol{p}_{k}^{t})$,为简化表述我们省略 $\alpha$。对于 $\boldsymbol{\omega}_{k}$ 的第 $u$ 个坐标,

$$
\frac{\partial\ell_{k}^{t}(\boldsymbol{\omega}_{k}^{t})}{\partial\boldsymbol{\omega}_{k,u}^{t}}=\sum_{l=1}^{m}\frac{\partial L_{k}^{t}(\boldsymbol{p}_{k}^{t})}{\partial\boldsymbol{p}_{k}^{t}(l)}\frac{\partial\boldsymbol{p}_{k}^{t}(l)}{\partial\boldsymbol{\omega}_{k,u}^{t}}\overset{(a)}{=}\sum_{l=1}^{m}\boldsymbol{e}_{k}^{t}(l)\frac{\partial\boldsymbol{p}_{k}^{t}(l)}{\partial\boldsymbol{\omega}_{k,u}}.
$$

等式 (a) 成立是因为 $\sum_{l=1}^{m}\frac{\partial\boldsymbol{p}_{k}^{t}(l)}{\partial\boldsymbol{\omega}_{u}^{t}}=\frac{\partial}{\partial\boldsymbol{\omega}_{u}^{t}}\sum_{l=1}^{m}\boldsymbol{p}_{k}^{t}(l)=0$,这是由于 $\sum_{l=1}^{m}\boldsymbol{p}_{k}^{t}(l)=1$,以及式 (21)。最后,将所有坐标和所有分块的梯度叠加即可得到本引理。∎

**收敛性**:定义 $\|\boldsymbol{a}\|_{\mathcal{D}}^{2}\triangleq\mathbb{E}_{x\sim\mathcal{D}_{X}}[\sum_{i=1}^{I}\|a_{i}(x)\|_{2}^{2}]$,$\langle\boldsymbol{a},\boldsymbol{b}\rangle_{\mathcal{D}}\triangleq\mathbb{E}_{x\sim\mathcal{D}_{X}}[\sum_{i=1}^{I}\langle a_{i}(x),b_{i}(x)\rangle]$。其中,$I$ 是 $\boldsymbol{a}$ 和 $\boldsymbol{b}$ 的大小,$x$ 是模型的输入,$\mathcal{D}_{X}$ 是数据集 $\mathcal{D}$ 中 $x$ 的分布。令 $\boldsymbol{p}(\boldsymbol{\omega})=(\boldsymbol{p}_{k}(\boldsymbol{\omega})\triangleq\hat{f}_{k}(\boldsymbol{p}_{k-1};\boldsymbol{\omega}_{k}),k=0,\cdots,K)$ 表示给定 $\boldsymbol{\omega}$ 时读出头的输出。为简化表述,我们将 $\mathcal{R}_{t}(\boldsymbol{\omega};\alpha)$ 重写为 $\mathcal{R}(\boldsymbol{\omega};\alpha,\tilde{\boldsymbol{p}}^{t})$,其中 $\boldsymbol{p}^{t}=\boldsymbol{p}(\boldsymbol{\omega}^{t})$。以下假设 2-4 在现有分析中被广泛采用(例如 [^26] [^10])。假设 5 在梯度有界(假设 2)以及 $\tilde{\boldsymbol{p}}^{t}$ 位于概率单纯形空间的前提下是合理的。假设 6 在简单随机采样和梯度有界的条件下是合理的。

###### 假设 2(梯度有界)。

梯度存在上界,即 $\mathbb{E}[\|\nabla_{\boldsymbol{\omega}}\mathcal{R}_{t}(\boldsymbol{\omega};\alpha)\|^{2}]\leq M(\alpha)$。

###### 假设 3(Lipschitz 连续性)。

读出输出 $\boldsymbol{p}(\boldsymbol{\omega})$ 关于 $\boldsymbol{\omega}$ 满足 Lipschitz 连续性,即对于参数空间中的所有 $\boldsymbol{\omega}$ 和 $\boldsymbol{\omega}^{\prime}$,$\|\boldsymbol{p}(\boldsymbol{\omega})-\boldsymbol{p}(\boldsymbol{\omega}^{\prime})\|_{\mathcal{D}}\leq\rho\|\boldsymbol{\omega}-\boldsymbol{\omega}^{\prime}\|$。

###### 假设 4(平滑性)。

对于参数空间中的任意 $\boldsymbol{\omega}$,$\mathcal{R}_{t}(\boldsymbol{\omega};\alpha)$ 是 $\beta_{R}(\alpha)$-平滑的,即 $\mathcal{R}_{t}(\boldsymbol{\omega}^{\prime};\alpha)\leq\mathcal{R}_{t}(\boldsymbol{\omega};\alpha)+\langle\nabla\mathcal{R}_{t}(\boldsymbol{\omega};\alpha),\boldsymbol{\omega}^{\prime}-\boldsymbol{\omega}\rangle+\frac{\beta_{R}(\alpha)}{2}\|\boldsymbol{\omega}^{\prime}-\boldsymbol{\omega}\|^{2}$。

###### 假设 5(稳定性)。

一阶目标变化沿分布更新被二次控制,即 $\left[\left\langle\nabla_{\tilde{\boldsymbol{p}}^{t}}\mathcal{R}(\boldsymbol{\omega}^{t};\alpha,\tilde{\boldsymbol{p}}^{t}),\tilde{\boldsymbol{p}}^{t+1}-\tilde{\boldsymbol{p}}^{t}\right\rangle_{\mathcal{D}}\right]_{+}\leq C_{\rm fo}(\alpha)\|\tilde{\boldsymbol{p}}^{t+1}-\tilde{\boldsymbol{p}}^{t}\|_{\mathcal{D}}^{2}$。

###### 假设 6(采样)。

小批量采样是无偏的且具有有界方差,即 $||\nabla_{\boldsymbol{\omega}}\hat{\mathcal{R}}_{t}(\boldsymbol{\omega};\alpha)-\nabla_{\boldsymbol{\omega}}\mathcal{R}_{t}(\boldsymbol{\omega};\alpha)||^{2}\leq\sigma^{2}(\alpha)/B$,其中 $\nabla_{\boldsymbol{\omega}}\hat{\mathcal{R}}_{t}(\boldsymbol{\omega};\alpha)$ 是采样小批量下的梯度,$B$ 是小批量大小。

定义漂移 $\delta_{t}^{\mathrm{ref}}(\alpha)\triangleq\left[\mathcal{R}_{t+1}(\boldsymbol{\omega}^{t+1};\alpha)-\mathcal{R}_{t}(\boldsymbol{\omega}^{t+1};\alpha)\right]_{+}$。该漂移的存在是因为分块 $k-1$ 的分布 $\boldsymbol{p}^{t}_{k-1}$ 发生了变化,它刻画了 $\mathcal{R}_{t}(\cdot)$ 在迭代间的变化。

###### 引理 4(漂移有界)。

$\mathcal{R}_{t}(\cdot)$ 的累积漂移存在上界,即 $\mathfrak{D}_{T}(\alpha)\triangleq\sum_{t=0}^{T-1}\mathbb{E}[\delta_{t}^{\mathrm{ref}}(\alpha)]\leq(C_{\rm fo}(\alpha)+\beta_{R}(\alpha)/2)\rho^{2}M(\alpha)\sum_{t=0}^{T-1}\eta_{t}^{2}$。

###### 证明。

根据 $\delta_{t}^{\mathrm{ref}}(\alpha)$ 的定义,在假设 4 中令 $\boldsymbol{\omega}^{\prime}=\boldsymbol{\omega}^{t+1}$ 和 $\boldsymbol{\omega}=\boldsymbol{\omega}_{t}$,可得 $\delta_{t}^{\rm ref}(\alpha)\leq(C_{\rm fo}(\alpha)+\frac{\beta_{R}(\alpha)}{2})\|\tilde{\boldsymbol{p}}^{t+1}-\tilde{\boldsymbol{p}}^{t}\|_{\mathcal{D}}^{2}$。基于假设 3 中 $\tilde{\boldsymbol{p}}^{t}\triangleq\tilde{\boldsymbol{p}}(\boldsymbol{\omega}^{t})$ 的 Lipschitz 连续性,$\|\tilde{\boldsymbol{p}}^{t+1}-\tilde{\boldsymbol{p}}^{t}\|_{\mathcal{D}}^{2}\leq\rho^{2}\|\boldsymbol{\omega}^{t+1}-\boldsymbol{\omega}^{t}\|^{2}$。令 $C_{\rm ref}\triangleq(C_{\rm fo}(\alpha)+\beta_{R}(\alpha)/2)\rho^{2}$。基于引理 3 和假设 2,$\mathbb{E}[\delta_{t}^{\rm ref}]\leq C_{\rm ref}(\alpha)\mathbb{E}[\|\boldsymbol{\omega}^{t+1}-\boldsymbol{\omega}^{t}\|^{2}]=C_{\rm ref}(\alpha)\eta_{t}^{2}\mathbb{E}[\|\nabla_{\boldsymbol{\omega}}\mathcal{R}_{t}(\boldsymbol{\omega};\alpha)\|^{2}]\leq C_{\rm ref}(\alpha)M(\alpha)\eta_{t}^{2}$。对 $t=0,\ldots,T-1$ 求和即可完成证明。∎

###### 定理 1(收敛性)。

令 $\eta_{t}=\eta_{0}/\sqrt{t+\gamma}\leq 1/\beta_{R}(\alpha)$,其中 $\gamma>0$。基于引理 4 和假设 4-6,

$$
\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}\|\nabla_{\boldsymbol{\omega}}\mathcal{R}_{t}(\boldsymbol{\omega};\alpha)\|^{2}\leq\frac{2(\mathcal{R}_{0}-\mathcal{R}_{opt}+\mathfrak{D}_{T}(\alpha))}{T\eta_{0}/\sqrt{T+\gamma}}\\
+\frac{\beta_{R}(\alpha)\sigma(\alpha)^{2}\sum_{t=0}^{T-1}\eta_{t}^{2}/B}{T\eta_{0}/\sqrt{T+\gamma}}=\tilde{\mathcal{O}}\left(\frac{1}{\sqrt{T}}\right),
$$

其中 $\mathcal{R}_{0}$ 和 $\mathcal{R}_{opt}$ 分别是 $\mathcal{R}(\cdot)$ 的初始值和最优值。

###### 证明。

为简化表述,我们在 $\mathcal{R}_{t}(\cdot)$ 中省略符号 $\alpha$。根据假设 4,并考虑使用采样小批量的更新,即 $\boldsymbol{\omega}^{t+1}\leftarrow\boldsymbol{\omega}^{t}-\eta^{t}\nabla\hat{\mathcal{R}}_{t}(\boldsymbol{\omega}^{t})$,我们有

$$
\mathcal{R}_{t}(\boldsymbol{\omega}^{t+1})\leq\mathcal{R}_{t}(\boldsymbol{\omega}^{t})-\eta^{t}\langle\nabla\mathcal{R}_{t}(\boldsymbol{\omega}^{t}),\nabla\hat{\mathcal{R}}_{t}(\boldsymbol{\omega})\rangle\\
+\frac{\beta_{R}(\alpha)\eta_{t}^{2}}{2}\|\nabla\hat{\mathcal{R}}_{t}(\boldsymbol{\omega}^{t})\|^{2}.
$$

取条件期望并使用假设 6 以及 $\eta_{t}\leq 1/\beta_{R}(\alpha)$,我们有

$$
\mathbb{E}[\mathcal{R}_{t}(\boldsymbol{\omega}^{t+1})]\leq\mathbb{E}[\mathcal{R}_{t}(\boldsymbol{\omega}^{t})]-\frac{\eta_{t}}{2}\|\nabla\mathcal{R}_{t}(\boldsymbol{\omega}^{t})\|^{2}+\frac{\beta_{R}(\alpha)\eta_{t}^{2}\sigma(\alpha)^{2}}{2B}.
$$

根据 $\delta_{t}^{\mathrm{ref}}(\alpha)$ 的定义,$\mathcal{R}_{t}(\boldsymbol{\omega}^{t+1})\geq\mathcal{R}_{t+1}(\boldsymbol{\omega}^{t+1})-\delta_{t}^{\mathrm{ref}}(\alpha)$。将 $\mathcal{R}_{t}(\boldsymbol{\omega}^{t+1})$ 代入式 (25) 并重新整理,

$$
\frac{\eta_{t}}{2}\mathbb{E}\|\nabla\mathcal{R}_{t}(\boldsymbol{\omega}^{t})\|^{2}\leq\mathbb{E}[\mathcal{R}_{t}(\boldsymbol{\omega}^{t})]-\mathbb{E}[\mathcal{R}_{t+1}(\boldsymbol{\omega}^{t+1})]\\
+\frac{\beta_{R}(\alpha)\eta_{t}^{2}\sigma(\alpha)^{2}}{2B}+\mathbb{E}[\delta_{t}^{\mathrm{ref}}(\alpha)].
$$

对 $t=0$ 到 $T-1$ 求和,并基于引理 4 和 $\mathcal{R}_{opt}$ 的定义,证明完成。∎

定理 1 表明,ZeroLock 算法的收敛速率为 $\tilde{\mathcal{O}}(1/\sqrt{T})$,与传统反向传播的收敛速率 ${\mathcal{O}}(1/\sqrt{T})$ 仅相差多重对数因子。同时,本节研究为一般模型分块划分下基于局部目标构建的方法提供了首个分析框架。

## III 系统设计

我们为 ZeroLock 提出了一个流水线训练系统。该系统适用于多 GPU 服务器和多设备场景。我们首先介绍系统概述,然后详细阐述系统设计。最后,我们讨论针对移动设备并行化的具体设计。

图 2:系统概述,包含一个协调器和多个执行器,每个执行器对应一个阶段(即一个模型分块的更新)。

### III-A 系统概述

如图 2 所示,该系统包含一个协调器和多个执行器(即 GPU 或设备),分为控制平面和数据平面。协调器位于控制平面,具有三个组件:**成员管理**(membership)用于跟踪活跃的执行器;**调度器**(scheduler)用于管理跨阶段的微批次更新调度;**故障恢复**(failure recovery)用于跟踪已提交请求的状态,并在检测到故障时分配重放请求。在**调度器**中,更新窗口定义了执行器被允许且必须处理的连续微批次序列,之后才能推进其进程,并作为检查点的单位;微批次调度器指示更新窗口中的微批次序列,并在检测到故障后协助微批次重放。

每个执行器同时包含控制平面和数据平面。控制平面包含三个组件:**成员管理**用于报告其注册和心跳;**执行监控**(execution monitor)用于监控其执行状态;**故障恢复**用于执行重放。执行器的数据平面负责处理一个阶段,即一个模型分块的更新。它从缓冲区中检索模型输入或上游阶段的隐藏状态,执行前向传递以确定其隐藏状态,将隐藏状态传输到下游阶段的缓冲区,并基于其局部目标执行反向更新。

我们提出以下技术来实现轻量级、高吞吐量和鲁棒的实现。(I)**早转发**(Early Forwarding)。隐藏状态的转发被设计为在局部更新之前执行,避免不必要的阶段间等待。(II)**独立执行与检查点**(Independent Execution and Checkpoint)。每个执行器仅维护其自己的可训练参数、优化器状态和检查点,支持阶段粒度的恢复,并独立构建其局部信号和执行局部更新。(III)**仅状态的阶段间交换**(State-Only Inter-Stage Exchange)。执行器仅与其他阶段交换隐藏状态(而不交换梯度或优化器状态)。(IV)**缓冲区辅助的状态交换**(Buffer-Assisted State Exchange)。每个执行器维护一个有界的隐藏状态缓冲区,存储其上游阶段多次迭代的隐藏状态,支持故障后重放。总体而言,(I)-(III) 有助于实现轻量级和高吞吐量;(II) 和 (IV) 有助于故障恢复能力。

### III-B 系统设计细节

我们首先介绍执行接口。然后介绍阶段间运行时、阶段内执行和故障恢复。

#### III-B1 阶段执行相关接口

以下定义了执行器上与阶段执行相关的统一接口。

**输入接口**。对于阶段 0 执行器,它输入 token ID 或预计算的初始隐藏表示(通过本地输入嵌入层构建)。对于后续每个阶段,它从隐藏状态缓冲区输入上游阶段的隐藏状态,该缓冲区由其上游执行器填充。同时,所有阶段都输入与特定微批次关联的注意力掩码、位置 ID 和标签,以便进行局部训练。

**前向与输出接口**。执行器使用其分块执行前向传递以生成其隐藏状态,然后将该隐藏状态从计算图中分离,并写入出站缓冲区以发送到下游阶段。

**局部更新接口**。执行器利用输入的隐藏状态和标签构建局部损失,并执行反向传递和优化器步骤以更新其可训练参数。该接口支持两种构建损失的方式:

(a) **静态模型头**(Static Model Head)。它重用预训练模型的最终归一化层和冻结模型头作为读出头,使用 token 级交叉熵和任务特定标签构建局部损失,如第 II-A 节所述。该设计不引入新的可训练参数,但假设中间隐藏状态可以被最终头解释,这对于复杂的生成任务可能不成立。

(b) **读出适配器**(Readout Adapter)。为了弥合中间隐藏状态与冻结模型头读出空间之间的差距,执行器可以插入一个阶段级读出适配器(即一个小的残差 MLP,初始化为近似恒等映射)。在训练期间,该适配器将中间隐藏空间对齐,使其更适合冻结模型头。

除了 (a) 和 (b),该接口还支持其他轻量级头,例如瓶颈投影头或受限词汇表头,以减少计算和内存开销。

#### III-B2 阶段间运行时与阶段内执行

阶段间运行时(作为调度引擎)在协调器处将多个微批次组织成一个更新窗口。这些微批次依次进入阶段流水线。对于每个微批次,阶段工作器执行**阶段内执行**:

- **输入**:从隐藏状态缓冲区检索上游阶段的隐藏状态;加载与该微批次相关的注意力掩码、位置 ID 和标签。
- **前向与输出**:执行前向传递以生成其隐藏状态;将其输出到出站缓冲区。
- **局部训练**:反向传播局部损失并为当前更新窗口累积梯度。在该窗口的最后一个微批次之后应用优化器步骤。

注意,下游执行可以在 S2 之后立即开始,无需等待上游在 S3 中的局部反向传播,从而实现更高程度的并行性。

跨阶段,**阶段间运行时**负责管理隐藏状态流。管理以**条目**(entry)的形式进行,条目包含微批次的隐藏状态张量以及用于发送、接收和状态的相关元数据。运行时跟踪每个条目从上游阶段的出站缓冲区到下游阶段的隐藏状态缓冲区的转换,使协调器能够准确控制每个阶段的缓冲区占用,而无需跟踪每个张量。同时,我们引入了飞行深度(in-flight depth)来限制每对阶段之间的条目数量,防止上游阶段压垮下游阶段。此外,在 GPU 上允许预发布接收(preposted receive),使得下游阶段可以提前提交 $\operatorname{receive\_entry}$ 请求。每个条目绑定到一个就绪事件。计算流仅等待此特定事件,避免不相关的通信操作阻塞计算流。

#### III-B3 故障恢复

故障恢复能力源于两个事实:每个阶段独立检查点其状态;隐藏状态缓冲区引入了对故障的弹性。因此,在发生故障后,只有失败的阶段需要回滚,它从缓冲区检索隐藏状态并重放请求。简要的故障恢复机制如下:

- **主动检查点**(Proactive Checkpointing):每个执行器定期捕获本地检查点,包括可训练参数、优化器状态和进度元数据。
- **响应式恢复**(Reactive Recovery):在检测到故障时,运行时从最新检查点恢复失败的阶段,并重放必要的更新窗口以赶上流水线前沿。成功的阶段保留其执行。

### III-C 移动执行后端

对于在移动设备上的部署,我们使用 ExecuTorch 框架,这是 Meta 官方的 PyTorch 原生边缘运行时。它执行提前(ahead-of-time, AOT)编译以生成静态计算表示,序列化为 .pte 程序,这使得在资源受限的终端设备上无需 Python 解释器即可进行确定性执行。然而,ExecuTorch 将前向和反向计算捆绑到单个 Python ExecuTorch (PTE) 方法中,其中前向隐藏状态仅在局部反向梯度计算之后暴露。这阻止了早转发的机会。

为了解决这个问题,我们在 PTE 方法中插入了一个轻量级流水线标记算子——具体位于分离的隐藏状态输出之后但在参数梯度子图之前。该算子不执行张量计算;相反,它向运行时发出信号以捕获隐藏输出并挂起 PTE 方法,同时保留其执行状态(栈、张量、优化器缓冲区)。我们使用两阶段 Android 接口扩展了 ExecuTorch 训练运行时。第一阶段执行到标记处并返回隐藏状态,该状态立即传输到下游设备。第二阶段恢复同一方法以完成局部反向传递,之后设备上的 AdamW 优化器更新该阶段的 LoRA 参数。在这种情况下,向下游设备的传输与局部反向计算重叠,使得生成的阶段程序支持局部训练和早转发,而不会产生跨阶段梯度远程过程调用(Remote Procedure Call, RPC)开销。

## IV 实验

我们分别使用多 GPU/CPU 服务器和 Android 设备构建了原型系统。在多 GPU/CPU 服务器上的评估展示了 (E1) 内存和吞吐量的方法比较以及 (E2) 故障恢复。在 Android 设备上的评估展示了 (E3) 设备上性能。

### IV-A 实验设置

**模型与数据**。我们使用 TinyLlama 作为预训练模型。其 Transformer 层被划分为三个连续的分块,每个分块分配给一个 NVIDIA L40 GPU 或一部 Android 手机。我们应用秩为 $4$、缩放因子为 $16$ 的 LoRA。默认情况下,实验在 AG News 的固定 10,000 个样本子集上进行;所有序列被填充或截断为 128 个 token。令 $b$ 表示一次微批次调用处理的物理批次数;$m$ 表示在一次优化器更新之前的此类微批次调用次数;因此,批次大小 $B=bm$。我们使用三个随机种子重复实验。

**方法**。我们将 ZeroLock 与三种方法进行比较。(i) GPipe [^13] 在反向传递之前完成逻辑批次中的所有前向传递。(ii) 1F1B [^20] 在逻辑批次内交错前向和反向传递,并在共享优化器更新之前刷新流水线。(iii) PipeDream [^20] 通过权重暂存(weight stashing,即存储多个副本)在更新窗口之间维护 1F1B 执行,确保前向和反向传递使用一致的参数版本。<sup>5</sup>

### IV-B E1:内存与吞吐量

尽管采用了模块化更新解耦,ZeroLock 的准确率和负对数似然(negative log-likelihood, NLL)与基线方法相当(见图 3)。基于此,我们展示其内存减少和吞吐量改进如下。

图 3:(a) 准确率和 (b) 负似然损失。

![Refer to caption](imgs/img-001-memory.png)

图 4:峰值 GPU 内存使用量以及峰值下各组件的内存使用量($b=8$ 且 $m=4$)。

#### IV-B1 内存

如图 4 所示,与 GPipe、1F1B 和 PipeDream 相比,ZeroLock 将每阶段平均峰值内存分别减少了 47.8%、14.7% 和 14.6%,将最大阶段使用量分别减少了 55.3%、26.6% 和 26.5%。大部分减少来自激活值的消除,在每阶段平均水平上分别减少了 75.4%、40.8% 和 48.8%。这种减少源于 ZeroLock 的模块化更新解耦;相比之下,其他方法实现端到端反向传播,需要保留激活值用于反向梯度计算。同时,在 $m=4$ 时,GPipe、1F1B、PipeDream 和 ZeroLock 分别在 $b=10,~20,~20,~28$ 时出现内存溢出,表明使用 ZeroLock 的系统可以承受更大的批次大小。

图 5 (a) 和 (c) 显示,当物理批次与微批次的比率(即 $b/m$)更大时,ZeroLock 更有利。具体而言,更多的物理批次(在一次微批次调用中)意味着基线方法保留更大的前向图,从而使模块化解耦带来更明显的优势。

(a)           (b)

(c)           (d)

图 5:(a) 和 (c),分别在 $B=32$ 和 $B=128$ 下的最大阶段内存使用量;(b) 和 (d),分别在 $B=32$ 和 $B=128$ 下的吞吐量。当 $b/m=8/4$ 时,与 PipeDream 相比,ZeroLock 将内存减少 26.5%,吞吐量提高 4.9%。

(a)           (b)

图 6:不同场景下的吞吐量:(a) 阶段数量,每个阶段由一个 GPU 执行;(b) 不同的发送端链路。

![Refer to caption](imgs/img-002-e4_gpu_timeline_grid.png)

图 7:CUDA 活动:(a) GPipe;(b) 1F1B;(c) PipeDream;(d) ZeroLock。

#### IV-B2 吞吐量

图 5 (b) 和 (d) 以及图 6 显示了吞吐量比较。默认设置包含三个阶段,$b=8$ 且 $m=4$。首先,在图 5 (b) 中,在默认设置下,与 GPipe、1F1B 和 PipeDream 相比,ZeroLock 将吞吐量分别提高了 55.8%、62.8% 和 4.9%。其次,在图 5 (b) 和 (d) 以及图 6 (a) 中,随着微批次调用中物理批次的增加和阶段的增加,ZeroLock 和 PipeDream 表现出更显著的吞吐量提升,显示出分别在减少每次调用中的气泡和利用额外 GPU 方面具有更好的潜力。最后,图 6 (b) 显示了在模拟发送端通信链路下的吞吐量:Wi-Fi,2 ms/1000 Mbps(延迟/带宽);移动网络,10 ms/200 Mbps;受限网络,30 ms/50 Mbps。当链路较慢时,ZeroLock 更有利。这是因为基线需要同时发送前向隐藏状态和反向隐藏梯度,而 ZeroLock 仅传输前向隐藏状态。

图 7 可视化了在 $b=8$、$m=4$、512 个数据样本和 16 次优化器更新下的 CUDA 活动跟踪。图表显示了一秒的稳态区间,在此期间所有三个 GPU 都表现出计算活动;为了避免偏差,在检查特定方法行为之前固定了此区间选择。内核和设备复制时间戳是外部测量的,没有注入设备端同步。如图所示,ZeroLock 和 PipeDream 表现出更多的并发阶段活动和更少的 GPU 空闲时间。此外,与 PipeDream 相比,ZeroLock 在各阶段之间显示出更均衡的活动负载和更高的活跃分数,改进了 13.7%。需要注意的是,此活跃分数作为流水线调度的诊断指标,而不是吞吐量的代理,因为空闲时段还包括主机调度和 Gloo 通信开销。

表 I:故障恢复。

| 方法 | 恢复时间 (ms) | 传输量 (MiB) |
| --- | --- | --- |
| 同步 1F1B | $2381.2\pm 66.6$ | 192 |
| ZeroLock | $2013.1\pm 6.4$ | 96 |

表 II:使用 Android 手机的设备上评估。

| 阶段/设备 | PTE | 队列 | 总计 | PSS | 温度 |
| --- | --- | --- | --- | --- | --- |
| S0/NX809J | 11.43/13.56 | 12.44/17.56 | 38.28/46.90 | 3423.5 | 37.0 |
| S1/Lenovo L71091 | 7.56/8.32 | 0/0.02 | 15.61/27.51 | 3408.2 | 36.0 |
| S2/Pixel 10 Pro XL | 9.53/11.40 | 0/5.79 | 10.30/18.99 | 3824.3 | 36.7 |

(a)           (b)

图 8:(a) 延迟分解;(b) 局部损失收敛。

### IV-C E2:故障恢复

我们评估了 ZeroLock 的中断恢复能力。它与 1F1B 进行比较。具体而言,1F1B 通过全局回滚和完全重放进行恢复。相比之下,ZeroLock 通过重用缓冲的隐藏状态并保留已提交的先前更新来实现局部恢复,消除了冗余的重新计算。

我们考虑三个阶段,设置 $b=1$ 和 $m=8$。故障设置如下:W0-W3,四个前导窗口;W4-W7,阶段 1 的四个故障窗口;W8-W11,四个恢复窗口。工作进程和 GPU 上下文在故障期间保持活跃。恢复延迟定义为从阶段 1 恢复到阶段 2 提交 W7 之间的时间。在表 I 中,ZeroLock 将恢复延迟减少了 368 ms,将传输流量从 192 MiB 减少到 96 MiB。

### IV-D E3:使用 Android 手机的设备上评估

对于 Android 设备上的实现,我们导出了三个 TinyLlama 训练 PTE,分别包含 Transformer 层 \[0,6\]、\[7,13\] 和 \[14,21\],并将它们分别部署在 NX809J、Lenovo L71091 和 Pixel 10 Pro XL 上。我们使用 128 个数据样本进行训练,序列长度为 128,$b=1$,LoRA 秩为 8,缩放因子为 16,学习率为 $10^{-4}$。每个阶段提交 128 次优化器步骤并写入九个检查点。

表 II 显示了单个批次请求的中位数/第 95 百分位 PTE、队列和总持续时间(以秒为单位),延迟分解如图 8 (a) 所示,应用峰值 PSS(以 MiB 为单位)以及电池温度(以 ${}^{\circ}C$ 为单位)。整个微调的总挂钟时间为 1644.1 秒,吞吐量为 0.0779 条记录/秒。图 8 (b) 显示了损失收敛。这些结果表明该实现在实践中是可行的。

## V 结论

在本工作中,我们提出了 ZeroLock 算法,该算法通过局部目标构建实现模块化更新解耦,用于大语言模型微调中的流水线并行。它通过缓解阶段间等待有效消除了流水线气泡,并通过在算法层面减少不必要的激活值存储降低了内存使用。我们为一般分块划分下基于局部目标构建的方法提供了首个分析框架,并证明 ZeroLock 的收敛速率为 $\tilde{\mathcal{O}}(1/\sqrt{T})$,与反向传播训练相当。同时,我们设计了一个部署 ZeroLock 的系统,结合了早转发和故障恢复等技术以提高系统吞吐量和鲁棒性。我们构建了真实世界原型,结果显示与基于反向传播的基线相比,ZeroLock 将内存减少了 26.5%,吞吐量提高了 4.9%。对于未来方向,进一步结合算子级优化以进一步提高吞吐量和减少内存使用是有意义的。
## VI AI 使用披露

ChatGPT 被用于辅助代码开发和实验环境搭建。具体而言，它被用于配置和排查基准测试复现所需的环境，在不同环境间运行和迁移基准测试实现，生成用于在移动设备上部署和执行 PTE 模型的 Kotlin 代码，以及使用 torchrun 生成批量实验脚本。AI 生成的代码和配置指令经过作者审查、必要时进行调整、测试和验证。作者对最终实现、实验结果及其解释承担全部责任。

[^1]: M. Akrout, C. Wilson, P. Humphreys, T. Lillicrap, and D. B. Tweed (2019) Deep learning without weight transport. In Proc. NeurIPS, Vol. 32. Cited by: §I.

[^2]: A. M. Alghamdi, M. U. Ashraf, A. A. Bahaddad, K. A. Almarhabi, W. A. Al Shehri, and A. Daraz (2025) Cross-subject eeg signals-based emotion recognition using contrastive learning. Scientific Reports 15 (1), pp. 28295. Cited by: §I.

[^3]: M. Aljahdali and M. Canini (2025) Idle no more: boosting distributed pipeline training via FluidPipe. In Proc. IEEE ICDCSW, Glasgow, U.K.. Cited by: §I, footnote 5.

[^4]: A. Borzunov, D. Baranchuk, T. Dettmers, M. Ryabinin, Y. Belkada, A. Chumachenko, P. Samygin, and C. Raffel (2023) Petals: collaborative inference and fine-tuning of large models. In Proc. ACL Syst. Demonstrations, Toronto, ON, Canada. Cited by: §I.

[^5]: J. Chen, X. Deng, Z. Xiong, S. Guo, X. Qiu, P. Wang, and D. Niyato (2025) CollaPipe: adaptive segment-optimized pipeline parallelism for collaborative LLM training in heterogeneous edge networks. arXiv:2509.19855. Cited by: §I, §I.

[^6]: Y. Chen, Y. Yan, S. Ge, Y. Qin, Y. Zheng, Q. Yang, S. He, Z. Shi, J. Chen, and Y. Shu (2025) Confidant: customizing transformer-based llms via collaborative training on mobile devices. In Proc. ACM MobiCom, Cited by: Fig. 1, §I, §I.

[^7]: S. Fan, Y. Rong, C. Meng, Z. Cao, S. Wang, Z. Zheng, C. Wu, G. Long, J. Yang, L. Xia, L. Diao, X. Liu, and W. Lin (2021) DAPPLE: a pipelined data parallel approach for training large models. In Proc. ACM PPoPP, Virtual Conf.. Cited by: §I, §I.

[^8]: S. Gandhi and C. Kozyrakis (2026) Sparse checkpointing for fast and reliable MoE training. In Proc. USENIX NSDI, Renton, WA, USA. Cited by: §I, §I.

[^9]: X. Guo, C. Xu, G. Guo, F. Zhu, C. Cai, P. Wang, X. Wei, J. Su, and J. Gao (2024) Faster multi-GPU training with PPLL: a pipeline parallelism framework leveraging local learning. arXiv:2411.12780. Cited by: §I, footnote 5.

[^10]: P. Han, C. Huang, G. Tian, M. Tang, and X. Liu (2024) Convergence analysis of split federated learning on heterogeneous data. Advances in Neural Information Processing Systems 37, pp. 103476–103544. Cited by: §II-B, §II-C.

[^11]: M. Ho, C. Wang, Y. Lin, and H. Chen (2026) SCPL: enhancing neural network training throughput with decoupled local losses and model parallelism. ACM Trans. Manage. Inf. Syst. 17 (2). Cited by: §I, footnote 5.

[^12]: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen (2022) LoRA: low-rank adaptation of large language models. In Proc. ICLR, Virtual Conf.. Cited by: 1st item, §II-A.

[^13]: Y. Huang, Y. Cheng, A. Bapna, O. Firat, M. X. Chen, D. Chen, H. Lee, J. Ngiam, Q. V. Le, Y. Wu, and Z. Chen (2019) GPipe: efficient training of giant neural networks using pipeline parallelism. In Proc. NeurIPS, Vancouver, BC, Canada. Cited by: §I, §I, §IV-A.

[^14]: Z. Huo, B. Gu, qian Yang, and H. Huang (2018) Decoupled parallel backpropagation with convergence guarantee. In Proc. ICML, Cited by: §I.

[^15]: Q. Li, Y. W. Teh, and R. Pascanu (2026) Noprop: training neural networks without back-propagation or forward-propagation. In Conference on Lifelong Learning Agents, pp. 525–544. Cited by: §I.

[^16]: S. Li and T. Hoefler (2021) Chimera: efficiently training large-scale neural networks with bidirectional pipelines. In Proc. ACM/IEEE SC, St. Louis, MO, USA. Cited by: §I, §I.

[^17]: X. Lian, S. A. Jacobs, L. Kurilenko, M. Tanaka, S. Bekman, O. Ruwase, and M. Zhang (2025) Universal checkpointing: a flexible and efficient distributed checkpointing system for large-scale DNN training with reconfigurable parallelism. In Proc. USENIX ATC, Boston, MA, USA. Cited by: §I, §I.

[^18]: S. Malladi, T. Gao, E. Nichani, A. Damian, J. D. Lee, D. Chen, and S. Arora (2023) Fine-tuning language models with just forward passes. In Proc. NeurIPS, External Links: [Document](https://dx.doi.org/10.52202/075280-2308) Cited by: §I.

[^19]: B. Millidge, A. Tschantz, and C. L. Buckley (2022) Predictive coding approximates backprop along arbitrary computation graphs. Neural Computation 34 (6), pp. 1329–1368. Cited by: §I.

[^20]: D. Narayanan, A. Harlap, A. Phanishayee, V. Seshadri, N. R. Devanur, G. R. Ganger, P. B. Gibbons, and M. Zaharia (2019) PipeDream: generalized pipeline parallelism for DNN training. In Proc. ACM SOSP, Huntsville, ON, Canada. Cited by: 1st item, §I, §I, §IV-A.

[^21]: D. Narayanan, A. Phanishayee, K. Shi, X. Chen, and M. Zaharia (2021) Memory-efficient pipeline-parallel DNN training. In Proc. ICML, Virtual Conf.. Cited by: 1st item, §I, §I.

[^22]: D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. A. Korthikanti, D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro, A. Phanishayee, and M. Zaharia (2021) Efficient large-scale language model training on GPU clusters using Megatron-LM. In Proc. ACM/IEEE SC, St. Louis, MO, USA. Cited by: §I, §I.

[^23]: A. Nøkland (2016) Direct feedback alignment provides learning in deep neural networks. Proc. NeurIPS. Cited by: §I.

[^24]: P. Qi, X. Wan, G. Huang, and M. Lin (2024) Zero bubble (almost) pipeline parallelism. In Proc. ICLR, Vienna, Austria. Cited by: §I, §I.

[^25]: M. Ryabinin, T. Dettmers, M. Diskin, and A. Borzunov (2023) SWARM parallelism: training large models can be surprisingly communication-efficient. In Proc. ICML, Honolulu, HI, USA. Cited by: §I, §I.

[^26]: H. Shi, T. Han, P. Wang, Z. Wang, X. Yang, and J. Su (2026) Rethinking local learning: a cheaper and faster recipe for LLM post-training. arXiv:2605.04913. Cited by: §I, §II-B, §II-C, footnote 5.

[^27]: A. Tandon, K. Dalal, X. Li, D. Koceja, M. Rød, S. Buchanan, X. Wang, J. Leskovec, S. Koyejo, T. Hashimoto, et al. (2025) End-to-end test-time training for long context. arXiv preprint arXiv:2512.23675. Cited by: §I.

[^28]: S. Wang, Z. Chen, and M. Tang (2026) CurvZO: adaptive curvature-guided sparse zeroth-order optimization for efficient llm fine-tuning. In Proc. ICML, Cited by: §I.

[^29]: T. Wu, L. Cao, H. Lu, X. Jiang, Y. Yu, S. Yang, G. Yang, J. Wang, L. Qu, L. Zhang, and W. Wang (2026) Attack of the bubbles: straggler-resilient pipeline parallelism for large model training. In Proc. USENIX NSDI, Renton, WA, USA. Cited by: §I, §I.

[^30]: C. Ye, R. Ye, Y. Zhang, and M. Tang (2026) Depth-progressive monotonic learning without global backpropagation. In Proc. ICML, Cited by: §I, §I, footnote 1, footnote 2, footnote 4.

[^31]: R. Ye, C. Ye, C. Huang, M. Tang, and Y. Liu (2026) Beyond-backpropagation training: methods, applications, and perspectives. TechRxiv 2026 (0103), pp.. External Links: [Document](https://dx.doi.org/10.36227/techrxiv.176740426.63642005/v1), [Link](https://www.techrxiv.org/doi/abs/10.36227/techrxiv.176740426.63642005/v1), https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176740426.63642005/v1 Cited by: §I.
