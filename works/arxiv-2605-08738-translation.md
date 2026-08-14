---
sourceTitle: "SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training"
title: "SlimQwen：大规模 MoE 模型预训练中的剪枝与蒸馏探索"
sourceUrl: "https://arxiv.org/html/2605.08738v2"
url: "https://arxiv.org/html/2605.08738v2"
requestedUrl: "https://arxiv.org/html/2605.08738v2"
adapter: "generic"
capturedAt: "2026-08-14T02:54:20.240Z"
conversionMethod: "defuddle"
kind: "generic/article"
language: "zh-CN"
sourceLanguage: "en"
pipelineRunId: "20260814-105125"
pipelineSource: "translate/20260814-105125/works-ready/arxiv-2605-08738-translation.md"
---

# SlimQwen：大规模 MoE 模型预训练中的剪枝与蒸馏探索

**作者**：Shengkun Tang<sup>†‡</sup>、Zekun Wang、Bo Zheng、Liangyu Wang、Rui Men、Siqi Zhang、Xiulong Yuan、Zihan Qiu、Zhiqiang Shen、Dayiheng Liu

**单位**：Qwen Team, Alibaba Inc.、MBZUAI、KAUST

<sup>†</sup>共同第一作者  
<sup>‡</sup>工作完成于 Qwen 团队实习期间  
<sup>†</sup>共同通讯作者

###### 摘要

结构化剪枝（Structured Pruning）和知识蒸馏（Knowledge Distillation, KD）是压缩大语言模型的典型技术，但它们如何在预训练规模上应用，特别是应用于近期的混合专家模型（Mixture-of-Experts, MoE），目前尚不清楚。在本研究中，我们系统性地研究了大规模预训练中的 MoE 压缩，重点关注三个核心问题：剪枝是否能提供比从零训练更好的初始化、专家压缩选择如何影响持续训练后的最终模型，以及哪种训练策略最有效。我们得到以下发现：首先，在深度、宽度和专家压缩三个维度上，对预训练 MoE 进行剪枝在相同训练预算下始终优于从零训练目标架构。其次，不同的一次性专家压缩方法在大规模持续预训练（Continual Pretraining）后会收敛到相似的最终性能。基于这一观察，我们引入了一种简单的部分保留策略（Partial-preservation Strategy）专家合并方法，该方法在大多数基准测试上提升了下游性能。第三，将知识蒸馏与语言建模损失（Language Modeling Loss）相结合优于单独使用 KD，特别是在知识密集型任务上。我们进一步提出了多 token 预测（Multi-token Prediction, MTP）蒸馏，实现了一致的性能提升。最后，在相同训练 token 下，渐进式剪枝（Progressive Pruning）调度优于一次性压缩，表明渐进式架构转换能产生更好的优化轨迹。综合所有方法，我们将 Qwen3-Next-80A3B 压缩为 23A2B 模型，保持了竞争力的性能。这些结果为大规模 MoE 高效压缩提供了实用指导。

## 1 引言

混合专家模型（Mixture-of-Experts, MoE）[^31] 已成为扩展大语言模型的主流架构 [^12] [^38] [^45] [^36] [^40]，但现代 MoE 大语言模型的预训练和部署仍然成本高昂。因此，在预训练规模上将预训练 MoE 压缩为保留大部分能力的较小模型是一个重要的实际问题。

结构化剪枝通过移除整个架构组件（例如层、注意力头或专家）来压缩模型，无需专用稀疏内核即可实现实际加速。由于单纯剪枝可能导致性能下降，知识蒸馏（KD）通常用于通过从教师模型向剪枝后的学生模型传递知识来恢复损失，并被广泛认为优于使用标准语言建模（LM）目标的持续预训练。尽管在密集模型上取得了广泛进展 [^26]，将这些压缩范式扩展到 MoE 模型仍面临独特挑战。具体而言，MoE 模型引入了额外的压缩维度：专家，它们可以被剪枝或合并。虽然近期研究 [^11] 全面评估了各种专家压缩方法的一次性性能，但它们在大规模持续预训练后的有效性仍未被探索。

为了弥补这一空白，我们通过系统性地研究几个实际问题，重新审视了 MoE 大语言模型的结构化剪枝和压缩后训练：(1) **初始化**：对预训练 MoE 模型进行剪枝是否比从零训练相同的目标架构提供更强的初始化？(2) **压缩策略**：不同的专家压缩策略如何在大规模持续预训练后影响最终性能？(3) **训练配方**：什么是促进性能恢复的最优压缩后训练配方？

通过在深度、宽度和专家三个维度进行大规模持续预训练，探索基于 MoE 的大语言模型压缩，我们得出以下核心发现：

首先，在匹配的训练 token 下，将预训练 MoE 模型剪枝到目标架构提供了明显优于从零训练的初始化，在推理和生成性能上都有一致的提升。

其次，我们对专家压缩进行了全面的实证分析，并提出了部分保留策略。通过在 400B token 持续预训练设置下比较各种剪枝和合并标准（例如路由频率或分数、专家激活），我们发现一次性专家剪枝或合并方法之间的最终性能差异微小，没有单一方法占据主导地位。基于这一观察以及在预训练专家特化与舍弃专家整合之间平衡的关键需求，我们提出了一种策略，明确保留目标专家的前半部分不变，同时将不太关键的剩余部分合并到其中。这防止了表示同质化，并持续增强了下游评估性能。

第三，我们证明，将下一 token 知识蒸馏（Next-token Prediction KD, NTP KD）与标准语言建模（LM）损失混合，通过线性衰减调度进行调节，在知识密集型基准测试上的恢复效果优于纯 KD。为了进一步提升压缩模型，我们提出了多 token 预测 [^8] 蒸馏（MTP KD）。这一范式将蒸馏目标扩展到单个 token 之外，从根本上增强了主干模型的训练动态和表示质量，并提高了多 token 推测解码（Speculative Decoding）中的接受率。

最后，我们研究了在从基础架构过渡到目标架构时如何渐进式地调度剪枝和蒸馏。给定目标配置，我们系统性地比较了直接一阶段压缩与三种渐进式剪枝调度：深度优先、宽度优先和联合调度。在所有配置中，渐进式策略在相同 token 预算下始终超越一次性剪枝。这证实了阶段性容量削减为知识迁移提供了明显更平滑的优化轨迹。

在实验上，我们证明了我们的剪枝和蒸馏配方可以将 Qwen3-Next-80A3B [^39] 压缩为 23A2B 模型（约 $4\times$ 压缩），在持续预训练后在广泛的评测套件上保持竞争力的下游性能，包括 MMLU 变体、BBH、GSM8K、代码和中文基准测试。总体而言，我们的结果为预训练规模上的计算高效 MoE 压缩提供了实用指导 [^40]，阐明了 (i) 如何应用跨深度/宽度/专家的结构化剪枝，(ii) 渐进式调度如何影响恢复，以及 (iii) 在长期压缩后训练期间哪种训练目标最有效。

**我们的主要贡献包括**：

- 我们对预训练规模上的大规模 MoE 压缩进行了系统性研究，涵盖结构化剪枝初始化、专家压缩、压缩后持续预训练目标和渐进式剪枝调度。我们表明，结构化剪枝提供了强有力的初始化，并且在大规模持续预训练后，不同的一次性专家剪枝/合并方法产生相似的最终性能。我们进一步提出了一种简单的部分保留专家合并策略，在各基准测试上显示出一致的改进。

- 我们引入了多 token 知识蒸馏，改进了主干模型训练和推测解码，并研究了不同的预训练损失选择。我们的实验表明，纳入 LM 损失可以改善知识密集型基准测试上的性能，而 MTP KD 在主要基准测试上实现了一致的提升。

- 我们比较了渐进式剪枝调度，发现在相同最终稀疏度和总训练 token 下，所有渐进式剪枝策略都始终优于一次性压缩。在实验上，我们将 Qwen3-Next-80A3B 压缩为 23A2B 模型，在包括一般推理、数学和编码在内的广泛基准测试上实现了竞争力的性能。

## 2 相关工作

**大语言模型中的结构化剪枝。** 结构化剪枝已被证明是一种无需特定硬件支持即可提高模型效率的有效技术。对于 MoE 大语言模型，有三个维度可以进行剪枝：1) 宽度剪枝（Width Pruning），如隐藏维度（Hidden Dimension）和前馈网络中间维度；2) 深度剪枝（Depth Pruning），通过某些度量移除整个 Transformer 块；3) 专家剪枝/合并（Expert Pruning/Merging），包括移除或合并 MoE 模块中的多个专家。一些先前工作如 ShearedLLaMA [^44] 和 SliceGPT [^1] 专注于密集大语言模型中的宽度剪枝 [^26]。对于深度剪枝，ShortGPT [^25]、Laco [^47] 和 ShortenedLLaMA [^13] 都提供了简单但有效的方法来剪枝大语言模型的深度。[^3] 提出了一种将大型 MoE 层合并为较小密集层的方法。此外，M-SMoE [^18] 和 REAP [^16] 提出合并 MoE 模块中的专家以减少内存消耗，而 [^22] 则简单地剪枝冗余专家。在本工作中，我们旨在实现高压缩比，并结合深度/宽度剪枝和专家剪枝/合并。此外，我们提出了一种简单但有效的专家合并技术，该技术在压缩后训练后提高了性能。

**压缩后恢复训练。** 由于结构化剪枝后的模型显示出不可忽视的性能下降，压缩后训练（Post-Compression Training）通常需要恢复剪枝模型的性能 [^24] [^42]。Minitron [^26] 和 Slim 应用蒸馏来改善剪枝后密集模型的性能，而 DarwinLM [^35] 和 SlimMoE [^19] 分别利用传统的语言建模损失（LM loss）和 KD。然而，Minitron 仅适用于非 MoE 模型，而 DarwinLM 和 SlimMoE 只剪枝 MoE 模块内专家的中间层维度。[^27] 系统性地研究了大语言模型的预训练蒸馏，关注 logit 处理、损失选择、缩放定律以及离线与在线教师 logit 等因素。相比之下，我们的工作研究了结构化剪枝后大型 MoE 模型的压缩后持续预训练，重点关注剪枝初始化、专家剪枝/合并以及压缩后的训练策略。

![SlimQwen 方法概览](https://arxiv.org/html/2605.08738v2/EfficientQwen-Figures.png)

**图 1**：SlimQwen 概览。我们首先对教师 MoE 模型执行结构化剪枝，包括基于重要性估计和相似性的宽度剪枝、深度剪枝和专家剪枝/合并，并采用提出的部分保留策略。然后我们采用渐进式剪枝和蒸馏，通过阶段性剪枝调度（深度优先、宽度优先或联合）逐步将教师模型转换为目标架构。最后，我们引入多 token 预测（MTP）蒸馏，通过监督多个未来 token 扩展标准的下一 token 蒸馏，提高训练有效性。

## 3 方法

### 3.1 背景与符号表示

Qwen3-Next [^39] 是一个基于混合注意力 MoE 的模型，具有 $L$ 层，每个块包括比例为 $(L_{linear}:L_{full})$ 的 Gated DeltaNet [^46] 或门控注意力（Gated Attention）[^29] 模块、具有 $N_{e}$ 个常规专家和 $N_{s}$ 个共享专家的 MoE 模块，以及 RMSNorm（均方根归一化）[^49] 模块。

对于 MoE 模块，给定输入 token $x\in\mathbb{R}^{1\times d}$，我们定义总共 $n$ 个专家，包括 $n_{\mathrm{routed}}$ 个路由专家和 $n_{\mathrm{shared}}$ 个共享专家（$n=n_{\mathrm{routed}}+n_{\mathrm{shared}}$）。每个专家是一个 SwiGLU 多层感知机（MLP），其形式为：

$$
\mathrm{Expert}(x)=(\mathrm{SiLU}(xW_{1e})\odot(xW_{2e}))W_{3e},
$$

其中 $W_{1e},W_{2e}\in\mathbb{R}^{d\times d_{\mathrm{ff}}}$ 且 $W_{3e}\in\mathbb{R}^{d_{\mathrm{ff}}\times d}$。路由器在路由专家上产生 top-$k$ 门控分数：

$$z(x)=\mathrm{softmax}\,\!\big(\mathrm{TopK}(xW^{G},k)\big),\,W^{G}\in\mathbb{R}^{d\times n_{\mathrm{routed}}}.$$

此外，我们为共享专家应用单独的共享门控：

$$z_{\mathrm{s}}(x)=\sigma(xw_{\mathrm{sh}})\in\mathbb{R}^{n_{shared}},\,w_{\mathrm{s}}\in\mathbb{R}^{d\times n_{shared}}.$$

MoE 输出为：

$$
\mathrm{MoE}(x)=\sum_{e=1}^{n_{\mathrm{routed}}}z_{e}(x)\mathrm{Expert}_{e}(x)+\sum_{s=1}^{n_{shared}}z_{\mathrm{s}}(x)\mathrm{Expert}_{s}(x).
$$

Qwen3-Next 使用 RMSNorm [^49] 归一化函数：

$$
\displaystyle\text{RMSNorm}(X)=\frac{X}{\text{RMS}(X)}\odot\gamma,\quad\text{RMS}(X)_{i}=\sqrt{\frac{1}{d}\sum_{j=1}^{d}X_{ij}^{2}+\epsilon}
$$

其中 $\text{RMS}(X)\in\mathbb{R}^{n\times 1}$ 是在隐藏维度上为每个 token 计算的均方根，$\gamma\in\mathbb{R}^{1\times d}$ 是可学习的缩放参数，常数 $\epsilon$ 用于数值稳定性。Gated DeltaNet 和 Gated Attention 的详细信息可在附录 A.1 节中找到。

### 3.2 基于 MoE 的模型压缩

在本工作中，我们专注于探索跨三个维度的基于 MoE 的模型压缩：深度、宽度和专家。我们在下面介绍每个维度的策略细节。

**深度剪枝。** 考虑一个具有 $L$ 个顺序层 $\{f_{\ell}\}_{\ell=1}^{L}$ 的模型，我们直接丢弃 $L$ 层模型的最后 $N$ 层 [^33]<sup>1</sup>：

$$
\mathcal{L}_{\mathrm{keep}}=\{1,\dots,L-N\},\qquad\tilde{L}=L-N.
$$

在我们的实验中，我们剪枝最后 25% 的层。

**宽度剪枝。** 对于宽度剪枝，我们在整个架构中减少隐藏维度，包括混合注意力、MoE 和归一化模块。我们使用在从训练数据集采样的校准数据集（Calibration Dataset）$\mathcal{D}$ 上计算的激活统计量（Activation Statistics）来估计每个隐藏维度的重要性。设 $Z\in\mathbb{R}^{B\times n\times m}$ 表示批次大小 $B$、序列长度 $n$ 和隐藏维度 $m$ 的模块输出激活。我们使用平均绝对激活值沿批次和序列维度进行聚合：

$$\mathrm{Mean}(Z)\;:=\;\frac{1}{Bn}\sum_{b=1}^{B}\sum_{t=1}^{n}\big|Z_{b,t,:}\big|\;\in\;\mathbb{R}^{m}.$$

设 $Y=\mathrm{RMSNorm}(X)\in\mathbb{R}^{B\times n\times d}$ 为 RMSNorm 输出。隐藏维度重要性（Importance Metric）计算公式为：

$$
\displaystyle I_{\mathrm{norm}}^{(k)}=\Big[\frac{\sum_{i=0}^{L}\mathrm{Mean}\big(\mathrm{RMSNorm}(X)\big)}{L}\Big]_{k},\,k=1,\dots,d.
$$

给定目标隐藏大小 $d_{t}$，我们保留具有最高重要性分数的 $d_{t}$ 个隐藏维度。

**专家压缩。** 关于专家压缩，我们比较了各种压缩策略，包括剪枝和合并。初始步骤涉及使用各种标准量化专家重要性。给定一组校准数据，基于频率的标准记录激活频率，而软 logit 方法进一步使用路由输出的 logit 对每个专家的频率进行加权。我们还考虑路由加权专家输出激活（Router-weighted Expert Output Activation, REAP）[^15]。形式上，对于每个 MoE 层，设有 $N$ 个路由专家 $\mathcal{E}=\{E_{1},\dots,E_{N}\}$ 和一个路由器 $R:\mathbb{R}^{d}\rightarrow\mathbb{R}^{N}$，其输出路由 logit $z(x)=R(x)\in\mathbb{R}^{N},x\in\mathbb{R}^{d}$。对于每个 token 表示 $x$，我们选择 top-$k$ 专家 $\mathcal{A}(x)=\mathrm{TopK}(z(x),k)\subseteq\{1,\dots,N\}$。设 $E_{j}(x)$ 为专家输出。我们可以通过以下方式计算基于频率、软 logit 和 REAP 的专家重要性：

$$
\displaystyle I_{i}^{\mathrm{Freq}}=\mathbb{E}_{x\sim\mathcal{C}}\Big[\mathbb{I}\big[i\in\mathcal{A}(x)\big]\Big],\qquad I_{i}^{\mathrm{Soft}}=\mathbb{E}_{x\sim\mathcal{C}}\Big[\frac{\mathbb{I}[i\in\mathcal{A}(x)]\cdot z_{i}(x)}{\sum_{j\in\mathcal{A}(x)}z_{j}(x)}\Big],
$$

$$
\displaystyle I^{\mathrm{REAP}}_{i}=\frac{1}{|\mathcal{X}_{i}|}\sum_{x\in\mathcal{X}_{i}}z_{i}(x)\,\big\|E_{i}(x)\big\|_{2},\qquad i=1,\dots,N,
$$

其中 $\mathbb{I}[\cdot]$ 是指示函数。在实践中，期望通过对校准集中所有 token 的平均值来计算。
对于专家合并，我们需要确定目标聚类和插值权重。我们首先使用路由 logit $z(x)$、路由权重和输出激活 $E_{j}(x)$ 量化专家间相似性。给定上述专家重要性分数，我们保留排名最高的专家。然后将每个被丢弃的专家合并到其最近的保留邻居中，使用其重要性分数作为缩放因子。专家压缩中的一个核心挑战是在知识保留和专家整合之间取得最优平衡。仅保留排名靠前的专家可以保留高度显著的知识，但可能会丢弃单独不太突出但功能互补的专家。相反，通过激进合并构建所有目标专家可能会使预训练的专家特化同质化，阻碍持续预训练期间的性能恢复。为了应对这种权衡，我们提出了一种简单的部分保留合并策略：我们保留一半目标专家不变，并通过将被丢弃的专家合并到选定的合并基中来构建其余部分。形式上，给定目标保留专家数量 $\tilde{N}<N$，我们保留一半目标具有最大重要性分数的专家：$\mathcal{S}_{\mathrm{keep}}=\operatorname*{arg\,topk}_{i\in\{1,\dots,N\}}I_{i}$，其中 $|\mathcal{S}_{\mathrm{keep}}|=\lfloor\tilde{N}//2\rfloor$，被剪枝的专家索引为 $\mathcal{S}_{\mathrm{prune}}=\{1,\dots,N\}\setminus\mathcal{S}_{\mathrm{keep}}$。最后，我们从剩余专家中选择另外 $\tilde{N}/2$ 个专家作为合并基，记为 $\mathcal{S}_{\mathrm{base}}$。对于每个 $i\in\mathcal{S}_{\mathrm{base}}$，我们找到其最相似的伙伴 $m(i)=\arg\max_{j\in\mathcal{S}_{\mathrm{merge}}}\mathrm{CosineSim}(i,j)$，并将两个专家合并为

$$
\tilde{E}_{i}=\frac{I_{i}}{I_{i}+I_{m(i)}}E_{i}+\frac{I_{m(i)}}{I_{i}+I_{m(i)}}E_{m(i)}.
$$

最终的压缩专家集由保留的专家和合并的专家组成。对于专家剪枝和专家合并，我们剪枝相应的路由权重以进行持续预训练。详细的算法描述可在算法 1 中找到。我们选择一半目标专家作为简单且对称的设计选择。直观上，保留太少的专家会削弱参数继承，而保留太多则留给整合的空间有限。保留大约一半在我们评估的设置中提供了稳健的折衷。我们在局限性部分对此进行了更多讨论。

### 3.3 蒸馏预训练

MTP 蒸馏损失。我们使用多 token 预测（MTP）模块 [^8] 来预测额外的未来 token。MTP 模块由嵌入层 $\mathrm{Emb}(\cdot)$ 和输出头 $\mathrm{OutHead}(\cdot)$ 组成，它们与主干模型共享。此外，MTP 模块包括一个 Transformer 块 $\mathrm{TRM}_{k}(\cdot)$ 和一个投影矩阵 $M_{k}\in\mathbb{R}^{d\times 2d}$。对于第 $i$ 个输入 token $t_{i}$，在预测深度 $k\in\{1,\dots,D\}$ 处，我们首先将深度 $k-1$ 处第 $i$ 个 token 的表示（记为 $h_{i}^{k-1}\in\mathbb{R}^{d}$）与第 $(i+k)$ 个 token 的嵌入 $\mathrm{Emb}(t_{i+k})\in\mathbb{R}^{d}$ 通过线性投影组合：

$$
h_{i}^{\prime k}=M_{k}\Big[\mathrm{RMSNorm}(h_{i}^{k-1});\,\mathrm{RMSNorm}\big(\mathrm{Emb}(t_{i+k})\big)\Big],
$$

其中 $[\cdot;\cdot]$ 表示拼接。特别地，当 $k=1$ 时，$h_{i}^{0}$ 指由主模型产生的 token 表示。然后将组合表示输入第 $k$ 个 Transformer 块以产生当前深度表示：$h^{k}_{1:T-k}=\mathrm{TRM}_{k}\!\left(h^{\prime k}_{1:T-k}\right)$，其中 $T$ 是序列长度，$1\!:\!T\!-\!k$ 表示切片。最后，给定 $h_{i}^{k}$ 作为输入，共享输出头计算第 $k$ 个额外预测 token 的概率分布：$p^{k}_{i+k}=\mathrm{OutHead}(h_{i}^{k})\in\mathbb{R}^{V}$，其中 $V$ 是词汇表大小。输出头 $\mathrm{OutHead}(\cdot)$ 将 $h_{i}^{k}$ 线性映射到 logit 并应用 $\mathrm{Softmax}(\cdot)$ 以获得概率。

对于每个预测深度 $k\in\{1,\dots,D\}$，第 $k$ 个 MTP 模块为位置 $i+k$ 产生学生分布 $p^{k}_{i+k}\in\mathbb{R}^{V}$。MTP LM 损失可写为：

$$
\mathcal{L}_{\mathrm{MTP\text{-}LM}}=\frac{1}{D}\sum_{k=1}^{D}(-\frac{1}{T-k}\sum_{i=1}^{T-k}\log p^{k}_{i+k}\!\left[t_{i+k}\right]).
$$

除了使用真实的 one-hot 标签外，我们还从教师模型蒸馏，该模型在相同位置提供软目标分布 $q_{i+k}\in\mathbb{R}^{V}$。我们最小化教师和学生之间的 KL 散度：

$$
\mathcal{L}_{\mathrm{MTP\text{-}KD}}=-\frac{1}{D}\sum_{k=1}^{D}(\frac{1}{T-k}\sum_{i=1}^{T-k}\sum_{v=1}^{V}q_{i+k}[v]\log p^{k}_{i+k}[v]).
$$

其中 $T$ 是输入序列长度，$V$ 是词汇表大小。因此，我们使用四个项训练模型：(i) 主干输出上的标准语言建模损失 $\mathcal{L}_{\mathrm{LM}}$ 和知识蒸馏损失 $\mathcal{L}_{\mathrm{KD}}$，MTP LM 损失 $\mathcal{L}_{\mathrm{MTP\text{-}LM}}$ 和 MTP 蒸馏损失 $\mathcal{L}_{\mathrm{MTP\text{-}KD}}$。总目标为

$$
\mathcal{L}=(1-\lambda)\,\mathcal{L}_{\mathrm{LM}}+\lambda\,\mathcal{L}_{\mathrm{KD}}+\beta\,((1-\lambda)\mathcal{L}_{\mathrm{MTP\text{-}LM}}+\lambda\mathcal{L}_{\mathrm{MTP\text{-}KD}}).
$$

其中 $\lambda$ 和 $\beta$ 是超参数，分别平衡 KD 和 LM 损失，以及主干损失和 MTP 损失。

渐进式剪枝与蒸馏。直接将教师模型压缩到紧凑的目标架构通常会导致大量知识损失。为了确保更平滑地迁移预训练能力，我们探索了三种渐进式两阶段蒸馏调度。每个调度都将结构剪枝与固定 token 蒸馏阶段交错，主要在深度和宽度的削减优先级上有所不同。深度优先在第一阶段分配一半的层削减，同时保持原始宽度，将剩余的深度和整个宽度削减留给第二阶段。相反，宽度优先在第一阶段执行一半的宽度削减，同时保持深度不变，在最后阶段完成剩余的宽度和全部深度削减。最后，联合策略在第一阶段同时削减深度和宽度各自目标的一半，在第二阶段剪枝剩余的一半以达到最终配置。通过这项探索，我们旨在确定在持续预训练期间最大化性能恢复的最优结构削减轨迹。

## 4 实验

### 4.1 实验设置

基础模型与剪枝设置。除非另有说明，我们的实验基于 80A3B 混合 MoE 模型进行，该模型包括 48 个 Transformer 块，其中 12 个全注意力层和 36 个线性注意力层。每个全注意力有 16 个查询头和 2 个键/值头，头维度为 256。整合了门控注意力 [^29]。对于 MoE 层，每个模块包含总共 512 个专家，每个 token 激活 10 个路由专家和 1 个共享专家。中间维度为 512，隐藏维度为 2048。该模型使用多 token 预测（MTP）模块训练。更多架构细节可在附录表 6 中找到。对于深度剪枝，我们移除 12 个 Transformer 块（3 个全注意力，9 个线性注意力）。在剩余层中，我们将隐藏维度从 2048 减少到 1536。此外，我们将 512 个专家合并为每个 MoE 模块 256 个，压缩模型每个 token 仅激活 8 个路由专家和 1 个共享专家。我们随机使用 1024 个样本作为校准集来计算重要性度量。

训练设置。我们在两种训练预算下评估模型：120B 和 400B 高质量、多样化的 token，全局批次大小分别为 512 和 1024。峰值学习率设置为 4e-4，通过余弦调度衰减到 3e-5，预热步数为 2000。蒸馏损失权重 $\lambda$ 从 1 线性衰减到 0.75，而 MTP 蒸馏权重 $\beta$ 遵循从 0.3 到 0.1 的余弦衰减。我们使用 Maestro 训练框架 [^48] 训练所有模型。我们在每节中解释详细的实验设置，详情可在附录表 7 中找到。

评测。我们在广泛的基准测试上评估模型的少样本性能。这些包括 MMLU [^9]、MMLU-Redux [^7] 和 MMLU-Pro [^41] 用于一般知识；BBH [^34] 用于推理；GSM-8K [^6] 用于数学；EvalPlus [^20] 用于编码，C-Eval [^10] 和 CMMLU [^17] 用于中文能力。我们在附录 A.6 节中提供更多评测。### 4.2 结果

表 1：从零开始训练的模型与从剪枝权重初始化的模型的结果对比。结果表明，在相同训练预算下，从剪枝模型开始训练对最终模型带来了收益。<sup>†</sup> 这里的 KD 损失指的是公式 12 中的组合损失。

| 方法 | MMLU | MMLU-Pro | MMLU-Redux | BBH | GSM-8K | EvalPlus | C-Eval | CMMLU | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Next-80A3B | 85.22 | 62.86 | 84.45 | 85.12 | 90.07 | 74.12 | 90.33 | 89.27 | 82.68 |
| 随机初始化 + KD 损失 <sup>†</sup> | 65.06 | 34.54 | 65.66 | 56.01 | 73.35 | 58.67 | 70.11 | 69.85 | 61.66 |
| 剪枝 + LM 损失 | 72.76 | 48.24 | 71.89 | 64.94 | 81.84 | 67.05 | 76.51 | 76.51 | 69.96 |
| 剪枝 + KD 损失 <sup>†</sup> | 75.67 | 51.19 | 74.37 | 72.29 | 83.17 | 69.30 | 80.67 | 80.95 | 73.45 |

图 2：不同初始化和训练目标下的训练损失曲线。从剪枝检查点初始化的模型比随机初始化收敛更快，并达到更低的 LM 损失。引入 KD 进一步改善了优化效果，剪枝 + KD 始终达到最低损失，其次是剪枝 + LM 损失，这证明了基于剪枝的初始化和蒸馏在高效训练中的优势。

**Q1：剪枝是否为大规模预训练中的 MoE 提供了更好的初始化？** 我们首先验证了从剪枝的 MoE 模型开始训练在预训练中的有效性。如表 1 所示，两种设置都使用来自 Qwen3-Next 教师模型的知识蒸馏在 120B token 上进行训练。与随机初始化相比，剪枝模型展现出显著优势，平均得分达到 73.45，而随机初始化为 61.66（提升 11.79 分）。这种一致的改进跨越了多个不同领域，包括知识（MMLU）、数学（GSM-8K）和编程（EvalPlus）。值得注意的是，尽管参数量减少了 3.4 倍，剪枝架构恢复了教师模型 86.5% 的性能（73.45 vs. 82.68），这表明结构化剪枝成功保留了任务关键权重，从而形成了一个信息丰富的起点。此外，训练轨迹（图 2）证实了这些发现：与随机初始化相比，剪枝初始化带来了显著更快的收敛速度和更低的语言建模损失，而组合的"剪枝 + KD"方案达到了最低损失。

**Q2：不同的专家压缩策略如何影响最终性能？** 为了评估各种专家压缩策略，我们将 24A2B MoE 模型压缩到 6A1B 架构，并持续预训练 400B token。如表 2 所示，没有任何单一的一次性剪枝或合并方法在所有下游任务上建立一致的优越性，即使某些模型在特定基准上表现出更高的性能（例如基于频率的路由 logit 分组方法在 BBH 上达到 60.17）。一个可能的解释是，一次性专家压缩（粗粒度剪枝或合并）方法无法在所有基准上一致地保持性能。此外，在合并专家期间部分保留专家在主要基准（包括 MMLU、MMLU-Pro 和 GSM8K）上带来了一致的改进。

**Q3：什么构成了压缩 MoE 的有效训练方案？**

表 2：在持续预训练期间，采用和不采用部分保留专家合并策略的模型性能对比，以及不同剪枝和合并方法的比较。结果表明：(1) 部分保留专家合并策略在主要基准上带来了性能提升，(2) 没有单一模型在所有评估任务上表现出一致的优越性能。

| 剪枝/合并 | 重要性度量 | 分组方法 | 保留 | MMLU | MMLU-Pro | MMLU-Redux | BBH | GSM-8K | EvalPlus | C-Eval | CMMLU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 专家合并 | 软 Logit | 路由权重 | 否 | 69.05 | 42.62 | 68.47 | 59.12 | 71.08 | 50.35 | 71.08 | 72.20 |
| 专家合并 | 软 Logit | 路由权重 | 是 | 69.28 | 44.05 | 67.64 | 59.81 | 74.18 | 48.00 | 71.15 | 72.73 |
| 专家剪枝 | 软 Logit | \- | \- | 68.74 | 43.23 | 69.11 | 58.97 | 74.30 | 51.69 | 71.67 | 72.26 |
| 专家剪枝 | REAP | \- | \- | 69.11 | 42.76 | 67.57 | 59.00 | 73.69 | 53.59 | 71.67 | 71.92 |
| 专家合并 | 频率 | 路由 Logit | 是 | 68.92 | 42.14 | 68.29 | 60.17 | 72.82 | 48.91 | 70.26 | 72.76 |
| 专家合并 | 软 Logit | 路由 Logit | 是 | 68.73 | 42.35 | 68.03 | 59.88 | 72.86 | 52.11 | 70.04 | 71.85 |
| 专家合并 | 软 Logit | 专家向量 | 是 | 68.88 | 42.47 | 68.32 | 59.00 | 70.74 | 49.69 | 71.38 | 72.23 |
| 专家合并 | REAP | 路由 Logit | 是 | 69.74 | 42.75 | 67.46 | 57.77 | 73.69 | 50.95 | 72.57 | 72.22 |
| 专家合并 | REAP | 专家向量 | 是 | 69.26 | 42.93 | 67.78 | 59.45 | 73.73 | 55.29 | 71.45 | 72.89 |

为了在预训练设置下建立有效的压缩后持续训练方案，我们在从 Qwen3-Next-80A3B 剪枝得到的 23A2B 模型上评估了各种损失配置，训练 120B token（表 3）。我们的分析揭示了几个发现：将下一 token 预测知识蒸馏（NTP KD）与标准语言建模损失相结合，优于纯蒸馏，特别是在知识密集型基准上，如 MMLU（从 74.16 提升到 74.93）和 MMLU-Pro（从 50.97 提升到 51.44）。此外，消融实验表明，将多 token 预测知识蒸馏（MTP KD）整合到纯 NTP KD 或综合联合目标（NTP KD + LM + MTP 损失）中，可以提升几个知识密集型基准的性能。除了主干质量外，MTP KD 在预训练和监督微调（SFT）阶段都为推测解码带来了显著的效率提升，如表 4 所示。我们报告了预训练阶段的 HumanEval、GSM8K、WMT22 [^14] 基准，以及 SFT 阶段的 RepoQA [^21]、MTBench [^4] 和 SpecBench [^43] 基准的结果。结果表明，MTP KD 在所有基准上一致地提升了从 acc\_1 到 acc\_4 的多 token 接受率。一个显著的模式是，MTP KD 带来的提升通常对更长的接受 token 序列更大。这表明 MTP KD 对于提高多 token 生成的效率特别有帮助，使得生成的 token 在推测解码期间更有可能被验证模型接受。总体而言，这些结果表明 MTP KD 不仅提高了主干训练质量，还为推测解码带来了实际效益。

表 3：不同训练损失的基准性能对比。所有模型都从 Qwen3-Next-80A3B 剪枝到 23A2B，并在 120B token 上训练。添加 LM 损失改善了知识基准（如 MMLU、MMLU-Pro），而整合 MTP KD 带来了一致的提升，完整目标在几个主要基准上达到了强劲的性能。NTP KD：下一 token 预测知识蒸馏。

| 方法 | MMLU | MMLU-Pro | MMLU-Redux | BBH | GSM-8K | EvalPlus | C-Eval | CMMLU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NTP KD | 74.16 | 50.97 | 75.85 | 71.63 | 84.27 | 67.32 | 80.00 | 80.24 |
| NTP KD + LM 损失 | 74.93 | 51.44 | 74.69 | 73.00 | 82.98 | 66.07 | 79.93 | 80.31 |
| NTP KD + MTP KD | 75.13 | 51.94 | 74.33 | 71.93 | 82.34 | 69.32 | 80.82 | 80.64 |
| NTP KD + LM 损失 + MTP 损失 | 75.29 | 51.16 | 75.09 | 72.07 | 83.02 | 68.43 | 79.78 | 80.67 |
| NTP KD + LM 损失 + MTP 损失 + MTP KD | 75.67 | 51.19 | 74.37 | 72.29 | 83.17 | 69.30 | 80.67 | 80.95 |

表 4：预训练和监督微调（SFT）阶段通过推测解码的 MTP 生成接受率（%）。结果表明，在预训练和 SFT 阶段，与 MTP 损失相比，MTP KD 在大多数基准上一致地提升了多 token 生成接受率。

<table><thead><tr><th rowspan="2">阶段</th><th rowspan="2">损失</th><th colspan="5">HumanEval</th><th colspan="5">GSM8K</th><th colspan="5">WMT22</th></tr><tr><th>acc_0</th><th>acc_1</th><th>acc_2</th><th>acc_3</th><th>acc_4</th><th>acc_0</th><th>acc_1</th><th>acc_2</th><th>acc_3</th><th>acc_4</th><th>acc_0</th><th>acc_1</th><th>acc_2</th><th>acc_3</th><th>acc_4</th></tr></thead><tbody><tr><th rowspan="2">预训练</th><th>MTP 损失</th><td>95.37</td><td>56.31</td><td>24.35</td><td>9.79</td><td>4.09</td><td>95.90</td><td>57.62</td><td>23.64</td><td>8.02</td><td>2.37</td><td>81.44</td><td>43.97</td><td>18.86</td><td>5.99</td><td>1.66</td></tr><tr><th>MTP KD</th><td>94.77</td><td>68.60</td><td>37.06</td><td>17.36</td><td>8.24</td><td>95.50</td><td>75.18</td><td>45.67</td><td>22.43</td><td>10.37</td><td>81.29</td><td>49.04</td><td>24.56</td><td>10.03</td><td>3.97</td></tr></tbody></table>

<table><thead><tr><th rowspan="2">阶段</th><th rowspan="2">损失</th><th colspan="5">RepoQA</th><th colspan="5">MTBench</th><th colspan="5">SpecBench</th></tr><tr><th>acc_0</th><th>acc_1</th><th>acc_2</th><th>acc_3</th><th>acc_4</th><th>acc_0</th><th>acc_1</th><th>acc_2</th><th>acc_3</th><th>acc_4</th><th>acc_0</th><th>acc_1</th><th>acc_2</th><th>acc_3</th><th>acc_4</th></tr></thead><tbody><tr><th rowspan="2">SFT</th><th>MTP 损失</th><td>96.02</td><td>64.68</td><td>29.36</td><td>11.23</td><td>3.91</td><td>87.85</td><td>57.40</td><td>28.72</td><td>12.46</td><td>4.93</td><td>87.61</td><td>55.58</td><td>27.73</td><td>12.02</td><td>4.60</td></tr><tr><th>MTP KD</th><td>96.17</td><td>69.49</td><td>35.94</td><td>15.67</td><td>6.59</td><td>88.55</td><td>61.30</td><td>33.10</td><td>16.03</td><td>7.04</td><td>87.97</td><td>59.85</td><td>32.21</td><td>15.22</td><td>6.56</td></tr></tbody></table>

表 5：一次性和渐进式剪枝及蒸馏的结果对比。所有模型都被剪枝到 23A2B。一次性剪枝直接在 400B token 上训练，而渐进式剪枝使用两阶段策略（40B + 360B）。渐进式方法在大多数基准上一致优于一次性剪枝，突显了在预训练期间逐步剪枝的优势。

| 方法 | Token | MMLU | MMLU-Pro | MMLU-Redux | BBH | GSM-8K | EvalPlus | C-Eval | CMMLU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 一阶段 | 400B | 75.86 | 52.97 | 75.41 | 73.97 | 85.22 | 70.07 | 83.87 | 82.26 |
| 联合 | 40B + 360B | 76.30 | 53.12 | 76.93 | 71.40 | 86.05 | 70.58 | 83.57 | 82.62 |
| 宽度优先 | 40B + 360B | 77.14 | 52.80 | 77.07 | 75.22 | 84.00 | 71.40 | 82.01 | 82.76 |
| 深度优先（SlimQwen） | 40B + 360B | 77.39 | 53.22 | 78.01 | 70.70 | 85.82 | 69.08 | 82.97 | 83.01 |

**渐进式剪枝和蒸馏。** 在一次性策略的基础上，我们进一步探索了渐进式剪枝和蒸馏的有效性。在给定最终目标架构配置的情况下，我们使用三种策略逐步剪枝基础模型：深度优先、宽度优先和联合剪枝，每种都在 3.3 节描述的两个阶段中进行。在第一阶段，中间剪枝模型使用 40B token 进行训练。然后我们进一步将其剪枝到最终目标配置，并在剩余的 360B token 上继续训练。结果如表 5 所示。总体而言，渐进式剪枝和蒸馏在直接在 400B token 上训练的一阶段剪枝上一致优于，证明了在持续预训练期间逐步模型压缩的优势。特别是，MMLU 从 75.86（一阶段）提升到 77.39（深度优先）和 77.14（宽度优先），而 MMLU-Redux 显示出显著的改进，从 75.41 提升到 78.01 和 77.07。这些发现证实了渐进式轨迹缓解了信息损失，并更好地迁移了预训练知识。我们在附录 A.5 中进一步提供了更细粒度阶段调度的结果。然而，更细粒度的阶段划分并未提供额外的基准性能提升。鉴于卓越的整体性能，我们正式将深度优先渐进式模型命名为 SlimQwen。

## 5 结论

在本文中，我们探索了 MoE 模型预训练中的剪枝和蒸馏。我们表明，即使在高压缩比下，结构化剪枝也为持续预训练提供了强大的初始化，而不同的专家剪枝和合并度量在大规模预训练后仅表现出微小差异。我们进一步提出了一种简单的部分保留专家合并策略，并在主要基准上展示了一致的性能改进。对于蒸馏，我们研究了渐进式剪枝和蒸馏的有效性，以及 LM 损失作为互补训练目标的作用。我们提出了一种新颖的多 token 预测（MTP）蒸馏目标用于预训练，在主要基准上展示了一致的性能提升。

## 参考文献

## 附录 A 附录

### A.1 架构细节

我们在表 6 中提供了原始教师模型和剪枝学生模型的架构细节。具体来说，对于门控注意力，给定输入隐藏状态 $X\in\mathbb{R}^{n\times d}$，其中 $d$ 是模型隐藏维度，$h_{q}$ 是查询头数量，门控注意力可以表示为：

$$
\mathrm{GatedAttn}(X)=\mathrm{Concat}\,\!\big({\mathrm{head}}_{1}\odot g_{i}(X),\ldots,{\mathrm{head}}_{h_{q}}\odot g_{{h_{q}}}(X)\big)W_{O},\,g_{i}(X)=\sigma(Xw_{g}^{(i)})\in\mathbb{R}^{n\times 1},
$$

其中 $W_{O}\in\mathbb{R}^{(h_{q}d_{\mathrm{head}})\times d}$ 是输出矩阵，$\sigma(\cdot)$ 表示 sigmoid 函数 $\sigma(z)=\frac{1}{1+e^{-z}}$，$w_{g}^{(i)}\in\mathbb{R}^{d\times 1}$ 是可学习的门控权重。注意力头通过缩放点积注意力计算：$\mathrm{head}_{i}=\mathrm{Attn}\!\big(Q^{(i)},K^{(m(i))},V^{(m(i))}\big),\mathrm{Attn}(Q,K,V)=\mathrm{softmax}(\frac{QK^{\top}}{\sqrt{d_{\mathrm{head}}}})V.$ 每个头的查询、键和值投影为 $Q^{(i)}=XW_{Q}^{(i)},K^{(j)}=XW_{K}^{(j)},V^{(j)}=XW_{V}^{(j)}$，其中可学习参数为 $W_{Q}^{(i)},W_{K}^{(j)},W_{V}^{(j)}\in\mathbb{R}^{d\times d_{\mathrm{head}}}.$ 我们使用分组查询注意力（GQA），具有 $h_{q}$ 个查询头和 $h_{kv}$ 个键/值头。对于 Gated DeltaNet，我们维护一个线性状态矩阵 $S_{t}\in\mathbb{R}^{d_{v}\times d_{k}},q_{t}\in\mathbb{R}^{d_{k}},k_{t}\in\mathbb{R}^{d_{k}},v_{t}\in\mathbb{R}^{d_{v}}.$ 门控 delta 规则更新状态为

$$
S_{t}\;=\;S_{t-1}\Big(\alpha_{t}\big(I-\beta_{t}k_{t}k_{t}^{\top}\big)\Big)\;+\;\beta_{t}\,v_{t}k_{t}^{\top},\qquad\alpha_{t}\in(0,1),\ \beta_{t}\in(0,1).
$$

token 混合输出由 $y_{t}\;=\;S_{t}q_{t}\in\mathbb{R}^{d_{v}}$ 读出。我们可以将其映射回模型维度 $d$：$Y_{t}=y_{t}W_{\mathrm{out}}\in\mathbb{R}^{d},\,W_{\mathrm{out}}\in\mathbb{R}^{d_{v}\times d}.$ 在我们的实现中，$d_{k}$ 对应于 $Q/K$ 隐藏维度，$d_{v}$ 对应于 $V$ 隐藏维度。

表 6：不同 MoE 变体的模型配置和参数数量。

<table><tbody><tr><th rowspan="2">模型</th><th rowspan="2"><math><semantics><msub><mi>d</mi> <mi>model</mi></msub> <annotation>d_{\mathrm{model}}</annotation></semantics></math></th><th colspan="5">自注意力</th><th colspan="4">MoE</th><th rowspan="2"><math><semantics><msub><mi>n</mi> <mi>MTP</mi></msub> <annotation>n_{\mathrm{MTP}}</annotation></semantics></math></th><td></td></tr><tr><th><math><semantics><msub><mi>n</mi> <mi>qhead</mi></msub> <annotation>n_{\mathrm{qhead}}</annotation></semantics></math></th><th><math><semantics><msub><mi>n</mi> <mi>kvhead</mi></msub> <annotation>n_{\mathrm{kvhead}}</annotation></semantics></math></th><th><math><semantics><msub><mi>d</mi> <mi>head</mi></msub> <annotation>d_{\mathrm{head}}</annotation></semantics></math></th><th><math><semantics><msub><mi>n</mi> <mi>layer</mi></msub> <annotation>n_{\mathrm{layer}}</annotation></semantics></math></th><th>注意力门控</th><th><math><semantics><msub><mi>d</mi> <mi>expert</mi></msub> <annotation>d_{\mathrm{expert}}</annotation></semantics></math></th><th><math><semantics><msub><mi>n</mi> <mi>expert</mi></msub> <annotation>n_{\mathrm{expert}}</annotation></semantics></math></th><th><math><semantics><msub><mi>n</mi> <mrow><mi>shared</mi> <mo></mo><mtext>-</mtext> <mo></mo><mi>expert</mi></mrow></msub> <annotation>n_{\mathrm{shared\mbox{-}expert}}</annotation></semantics></math></th><th>top-k</th><td></td></tr><tr><td>80B-A3B</td><td>2048</td><td>16</td><td>2</td><td>256</td><td>12</td><td>是</td><td>512</td><td>512</td><td>1</td><td>10</td><td>1</td><td></td></tr><tr><td>SlimQwen-23A2B</td><td>1536</td><td>16</td><td>2</td><td>256</td><td>8</td><td>是</td><td>512</td><td>256</td><td>1</td><td>8</td><td>1</td><td></td></tr><tr><td>23B-A2B</td><td>2048</td><td>16</td><td>2</td><td>256</td><td>7</td><td>是</td><td>512</td><td>256</td><td>1</td><td>8</td><td>1</td><td></td></tr><tr><td>SlimQwen-6A1B</td><td>1280</td><td>16</td><td>2</td><td>256</td><td>5</td><td>是</td><td>512</td><td>128</td><td>1</td><td>8</td><td>1</td><td></td></tr></tbody></table>

<table><thead><tr><th rowspan="2">模型</th><th colspan="8">线性注意力</th><th rowspan="2">总参数量</th><th rowspan="2">激活参数量</th></tr><tr><th><math><semantics><msub><mi>n</mi> <mi>vhead</mi></msub> <annotation>n_{\mathrm{vhead}}</annotation></semantics></math></th><th><math><semantics><msub><mi>n</mi> <mi>qkhead</mi></msub> <annotation>n_{\mathrm{qkhead}}</annotation></semantics></math></th><th><math><semantics><msub><mi>d</mi> <mi>vhead</mi></msub> <annotation>d_{\mathrm{vhead}}</annotation></semantics></math></th><th><math><semantics><msub><mi>d</mi> <mi>qkhead</mi></msub> <annotation>d_{\mathrm{qkhead}}</annotation></semantics></math></th><th><math><semantics><msub><mi>d</mi> <mi>conv</mi></msub> <annotation>d_{\mathrm{conv}}</annotation></semantics></math></th><th><math><semantics><msub><mi>d</mi> <mi>inner</mi></msub> <annotation>d_{\mathrm{inner}}</annotation></semantics></math></th><th><math><semantics><msub><mi>n</mi> <mi>layer</mi></msub> <annotation>n_{\mathrm{layer}}</annotation></semantics></math></th><th>注意力门控</th></tr></thead><tbody><tr><td>80B-A3B</td><td>32</td><td>16</td><td>128</td><td>128</td><td>4</td><td>4096</td><td>36</td><td>否</td><td>80B</td><td>3.8B</td></tr><tr><td>SlimQwen-23A2B</td><td>32</td><td>16</td><td>128</td><td>128</td><td>4</td><td>4096</td><td>24</td><td>否</td><td>23B</td><td>2.0B</td></tr><tr><td>23B-A2B</td><td>32</td><td>16</td><td>128</td><td>128</td><td>4</td><td>4096</td><td>21</td><td>否</td><td>23B</td><td>2B</td></tr><tr><td>SlimQwen-6A1B</td><td>32</td><td>16</td><td>128</td><td>128</td><td>4</td><td>2560</td><td>15</td><td>否</td><td>6B</td><td>1B</td></tr></tbody></table>

### A.2 训练超参数

我们在表 7 中提供了详细的预训练超参数。

表 7：120B token 和 400B token 设置的训练超参数。两种设置仅在全局批次大小上有所不同。

<table><tbody><tr><th>超参数</th><td>120B 设置</td><td>400B 设置</td></tr><tr><th>训练 token</th><td>120B</td><td>400B</td></tr><tr><th>全局批次大小</th><td>512</td><td>1024</td></tr><tr><th>学习率</th><td colspan="2"><math><semantics><mrow><mn>4</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>4\times 10^{-4}</annotation></semantics></math></td></tr><tr><th>学习率调度</th><td colspan="2">余弦衰减</td></tr><tr><th>最小学习率</th><td colspan="2"><math><semantics><mrow><mn>3</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>3\times 10^{-5}</annotation></semantics></math></td></tr><tr><th>预热步数</th><td colspan="2">2000</td></tr><tr><th>KD 损失权重 <math><semantics><mi>λ</mi> <annotation>\lambda</annotation></semantics></math></th><td colspan="2">从 1.0 到 0.75 线性衰减</td></tr><tr><th>MTP 蒸馏权重 <math><semantics><mi>β</mi> <annotation>\beta</annotation></semantics></math></th><td colspan="2">从 0.3 到 0.1 余弦衰减</td></tr><tr><th>校准样本</th><td colspan="2">1024</td></tr><tr><th>训练平台</th><td colspan="2">阿里云</td></tr></tbody></table>

### A.3 实现细节

我们的代码库基于 Megatron-LM 构建。遵循 Qwen3 MoE 模型 [^45]，我们对 MoE 应用全局批次负载均衡损失 [^28]。校准数据从预训练数据中采样。对于渐进式剪枝蒸馏，我们使用单阶段学习率衰减调度训练所有模型，使得第二阶段从第一阶段最终步骤达到的学习率开始。我们使用 AdamW 优化器，并对优化器应用默认超参数设置。对于推测解码，我们使用 MTP 模块作为草稿模型，主干模型作为验证模型，并将 acc\_0 报告为使用 MTP 模块生成一个 token 的接受率，acc\_1 报告为生成两个 token 的接受率，依此类推。我们还提供了部分保留专家合并策略的伪代码，如算法 1 所示。

算法 1 部分保留专家合并策略

专家 $\{E_{i}\}_{i=1}^{N}$，目标专家数量 $\tilde{N}$，重要性分数 $\{S_{i}\}_{i=1}^{N}$

压缩后的专家 $\tilde{\mathcal{E}}$

$S_{\mathrm{keep}}\leftarrow\operatorname{arg\,topk}_{i\in\{1,\dots,N\}}S_{i}$，其中 $|S_{\mathrm{keep}}|=\lfloor\tilde{N}/2\rfloor$

选择 $S_{\mathrm{base}}\subset\{1,\dots,N\}\setminus S_{\mathrm{keep}}$ 使得 $|S_{\mathrm{base}}|=\tilde{N}-|S_{\mathrm{keep}}|$

for all $i\in\{1,\dots,N\}\setminus(S_{\mathrm{keep}}\cup S_{\mathrm{base}})$ do

   $m(i)\leftarrow\arg\max_{j\in S_{\mathrm{base}}}\mathrm{CosineSim}(i,j)$

  将 $i$ 分配到 $m(i)$ 的合并组

end for

for all $j\in S_{\mathrm{base}}$ do

  将所有分配到 $j$ 的专家合并为 $\tilde{E}_{j}$

end for

return $\tilde{\mathcal{E}}=\{E_{i}:i\in S_{\mathrm{keep}}\}\cup\{\tilde{E}_{j}:j\in S_{\mathrm{base}}\}$

### A.4 不同深度剪枝方法的比较

我们比较了不同的深度剪枝方法，包括基于激活相似度的方法和直接剪枝最后几层的方法，如表 8 所示。形式化地，设 $h_{\ell}\in\mathbb{R}^{n\times d}$ 为层 $\ell$ 的激活，$a_{\ell}=\frac{1}{n}\sum_{t=1}^{n}h_{\ell,t}\in\mathbb{R}^{d}$ 为其 token 平均池化向量。我们计算相邻层余弦相似度：

$$
c_{\ell}=\frac{\langle a_{\ell-1},a_{\ell}\rangle}{\|a_{\ell-1}\|_{2}\|a_{\ell}\|_{2}},\qquad\ell=2,\dots,L.
$$

设 $\ell^{\star}=\arg\max_{\ell\in\{2,\dots,L\}}c_{\ell}$ 为起始索引，并剪枝 $N$ 层的连续块：$\mathcal{S}_{\mathrm{prune}}=\{\ell^{\star},\ell^{\star}+1,\dots,\ell^{\star}+N-1\}$ 和 $\mathcal{S}_{\mathrm{keep}}=\{1,\dots,L\}\setminus\mathcal{S}_{\mathrm{prune}}.$

我们在具有 24 层的预训练 15A3B 教师模型上进行实验。我们使用上面讨论的相同的 1024 个校准数据集来计算层激活，并在一次性设置中剪枝 4 层。基于激活的剪枝方法倾向于剪枝中间层。表中的结果表明，直接剪枝最后 4 层仅导致轻微的性能下降（例如在 MMLU 上从 75.62 降至 73.86），而基于激活的方法显示出显著更大的性能下降（例如在 MMLU 上从 75.62 降至 41.95）。结果也与 [^33] 的观察一致。在 120B token 的压缩后 KD 之后，最后层剪枝仍然比基于激活的方法恢复出更好的性能。表中一个有趣的现象是，使用 120B token 训练的模型在 MMLU 和 CMMLU 等基准上的性能比一次性对应模型更差。一个可能的解释是，一次性性能已经接近教师模型的性能，留下了相对较小的知识差距需要恢复。

表 8：一次性和持续预训练设置下不同深度剪枝方法的结果比较。在一次性设置中，剪枝最后一层在 MMLU 等基准上仅导致轻微的性能下降，而基于激活的方法导致显著更大的性能下降。在使用 120B token 进行压缩后 KD 之后，最后层剪枝仍然比基于激活的方法恢复出更好的性能。

| 方法 | MMLU | CMMLU | CEval | GSM8K |
| --- | --- | --- | --- | --- |
| 15A2B 教师模型 | 75.62 | 81.35 | 82.08 | 82.41 |
| 激活相似度 | 41.95 | 43.41 | 42.28 | 11.22 |
| 最后层剪枝 | 73.86 | 80.3 | 79.96 | 2.05 |
| 激活相似度 + 120B token | 69.57 | 74.32 | 75.69 | 73.84 |
| 最后层剪枝 + 120B token | 73.02 | 78.08 | 78.07 | 77.86 |

### A.5 更多阶段的渐进式剪枝和蒸馏结果

我们提供了具有更细粒度阶段的渐进式剪枝和蒸馏的结果，如表 9 所示。有两种类型的三阶段设置：深度优先和宽度优先。在深度优先设置中，我们首先剪枝一半要移除的层并训练模型 20B token，然后剪枝剩余的一半并再训练 20B token。最后，在第三阶段，我们剪枝宽度并继续训练 360B token。宽度优先设置遵循相反顺序的相同过程。结果表明，三阶段设置达到的性能与两阶段设置相当。虽然一些三阶段变体在个别基准上表现更好，但整体结果保持相似。这表明在我们的设置中，两阶段渐进式剪枝策略已经足够。

表 9：一次性和三阶段渐进式剪枝和蒸馏的结果对比。与两阶段设置相比，更细粒度的阶段划分不会产生额外的性能提升。

| 方法 | Token | MMLU | MMLU-Pro | MMLU-Redux | BBH | GSM-8K | EvalPlus | C-Eval | CMMLU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 一阶段 | 400B | 75.86 | 52.97 | 75.41 | 73.97 | 85.22 | 70.07 | 83.87 | 82.26 |
| 宽度优先 3 阶段 | 20B + 20B + 360B | 76.46 | 52.18 | 77.18 | 73.44 | 84.08 | 68.32 | 83.72 | 82.70 |
| 深度优先 3 阶段 | 20B + 20B + 360B | 77.29 | 52.63 | 77.29 | 73.37 | 84.15 | 71.12 | 83.75 | 82.65 |
| 深度优先（SlimQwen） | 40B + 360B | 77.39 | 53.22 | 78.01 | 70.70 | 85.82 | 69.08 | 82.97 | 83.01 |

### A.6 更多基准的评估

由于页面限制，我们在本节中提供了实验在更多基准上的评估结果。我们进一步添加了用于中文知识的 CEval [^10]，用于通用知识的 SuperGPQA [^37]，用于推理和上下文学习能力的 KOR-Bench [^23] 和 ICLEval [^5]，用于编程任务的 MBPP [^2]，用于多语言知识的 MMMLU [^9] 和 IncludeBase [^30]，以及用于多语言数学能力的 Mgsm [^32]。我们在表 10 中提供了从零开始训练的模型与从剪枝权重初始化的模型在这些基准上的结果对比。

表 10：从零开始训练的模型与从剪枝权重初始化的模型的更多基准结果对比。

| 方法 | CEval | SuperGPQA | KOR-Bench | ICLEval | MBPP | MMMLU | IncludeBase | Mgsm | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 随机初始化 + KD 损失 | 70.11 | 21.16 | 33.36 | 52.45 | 57.00 | 50.90 | 42.83 | 39.98 | 45.97 |
| 剪枝 + LM 损失 | 76.51 | 27.16 | 39.52 | 62.79 | 63.2 | 62.39 | 55.96 | 61.85 | 56.17 |
| 剪枝 + KD 损失 | 80.67 | 29.22 | 40.80 | 65.88 | 67.40 | 66.40 | 58.88 | 64.47 | 59.21 |

表 11：SlimQwen 和原始模型的加速和内存分析。

<table><thead><tr><th>模型</th><th>峰值内存 (GB)</th><th colspan="2">HF 后端</th><th colspan="2">vLLM 后端</th></tr><tr><th></th><th></th><th>预填充延迟 (s)</th><th>解码吞吐量 (Tok/s)</th><th>预填充延迟 (s)</th><th>解码吞吐量 (Tok/s)</th></tr></thead><tbody><tr><th>Qwen3-Next-80A3B</th><th>156.56</th><td>0.99</td><td>4.05</td><td>0.08</td><td>142.58</td></tr><tr><th>SlimQwen-23A2B</th><th>43.30</th><td>0.44</td><td>6.55</td><td>0.06</td><td>210.87</td></tr></tbody></table>
### A.7 效率分析

我们提供了 SlimQwen 与原始教师模型的效率分析，如表 11 所示。提示词长度为 128，生成长度限制为 128。该过程执行 10 次，包含 3 次预热运行，结果计算为平均值。我们分别提供了以 HuggingFace 和 vLLM 作为推理后端的结果。模型在相同的两个 GPU 上运行，张量并行大小为 2。峰值内存以 bfloat16 数据类型进行监控。我们可以观察到，SlimQwen 在预填充和解码阶段都获得了更好的加速效果。更重要的是，作为一个小规模模型，SlimQwen 可以部署在单个 80GB 显存的 GPU 上，这可以进一步提升效率，因为不需要张量并行（Tensor-Parallel, TP）或流水线并行（Pipeline-Parallel, PP）等并行策略。

[^1]: Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. Slicegpt: Compress large language models by deleting rows and columns, 2024. URL [https://arxiv.org/abs/2401.15024](https://arxiv.org/abs/2401.15024).

[^2]: Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021. URL [https://arxiv.org/abs/2108.07732](https://arxiv.org/abs/2108.07732).

[^3]: Mingyu Cao, Gen Li, Jie Ji, Jiaqi Zhang, Xiaolong Ma, Shiwei Liu, and Lu Yin. Condense, don't just prune: Enhancing efficiency and performance in moe layer pruning, 2025. URL [https://arxiv.org/abs/2412.00069](https://arxiv.org/abs/2412.00069).

[^4]: Jialin Chen, Aosong Feng, Ziyu Zhao, Juan Garza, Gaukhar Nurbek, Cheng Qin, Ali Maatouk, Leandros Tassiulas, Yifeng Gao, and Rex Ying. Mtbench: A multimodal time series benchmark for temporal reasoning and question answering, 2026. URL [https://arxiv.org/abs/2503.16858](https://arxiv.org/abs/2503.16858).

[^5]: Wentong Chen, Yankai Lin, ZhenHao Zhou, HongYun Huang, Yantao Jia, Zhao Cao, and Ji-Rong Wen. Icleval: Evaluating in-context learning ability of large language models, 2024. URL [https://arxiv.org/abs/2406.14955](https://arxiv.org/abs/2406.14955).

[^6]: Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL [https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168).

[^7]: Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, Claire Barale, Robert McHardy, Joshua Harris, Jean Kaddour, Emile van Krieken, and Pasquale Minervini. Are we done with mmlu?, 2025. URL [https://arxiv.org/abs/2406.04127](https://arxiv.org/abs/2406.04127).

[^8]: Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. In *Forty-first International Conference on Machine Learning*, 2024.

[^9]: Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021. URL [https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300).

[^10]: Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models, 2023. URL [https://arxiv.org/abs/2305.08322](https://arxiv.org/abs/2305.08322).

[^11]: Ajay Jaiswal, Jianyu Wang, Yixiao Li, Pingzhi Li, Tianlong Chen, Zhangyang Wang, Chong Wang, Ruoming Pang, and Xianzhi Du. Finding fantastic experts in moes: A unified study for expert dropping strategies and observations, 2025. URL [https://arxiv.org/abs/2504.05586](https://arxiv.org/abs/2504.05586).

[^12]: Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts. *CoRR*, abs/2401.04088, 2024.

[^13]: Bo-Kyeong Kim, Geonmin Kim, Tae-Ho Kim, Thibault Castells, Shinkook Choi, Junho Shin, and Hyoung-Kyu Song. Shortened llama: Depth pruning for large language models with comparison of retraining methods, 2024. URL [https://arxiv.org/abs/2402.02834](https://arxiv.org/abs/2402.02834).

[^14]: Tom Kocmi, Rachel Bawden, Ondřej Bojar, Anton Dvorkovich, Christian Federmann, Mark Fishel, Thamme Gowda, Yvette Graham, Roman Grundkiewicz, Barry Haddow, Rebecca Knowles, Philipp Koehn, Christof Monz, Makoto Morishita, Masaaki Nagata, Toshiaki Nakazawa, Michal Novák, Martin Popel, and Maja Popović. Findings of the 2022 conference on machine translation (WMT22). In *Proceedings of the Seventh Conference on Machine Translation (WMT)*, 2022. URL [https://aclanthology.org/2022.wmt-1.1/](https://aclanthology.org/2022.wmt-1.1/).

[^15]: Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie, Yani Ioannou, and Vithursan Thangarasa. REAP the Experts: Why Pruning Prevails for One-Shot MoE compression, 2025a. URL [https://arxiv.org/abs/2510.13999v1](https://arxiv.org/abs/2510.13999v1). arXiv:2510.13999v1 \[cs\].

[^16]: Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie, Yani Ioannou, and Vithursan Thangarasa. Reap the experts: Why pruning prevails for one-shot moe compression, 2025b. URL [https://arxiv.org/abs/2510.13999](https://arxiv.org/abs/2510.13999).

[^17]: Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese, 2024a. URL [https://arxiv.org/abs/2306.09212](https://arxiv.org/abs/2306.09212).

[^18]: Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, and Tianlong Chen. Merge, then compress: Demystify efficient smoe with hints from its routing policy, 2024b. URL [https://arxiv.org/abs/2310.01334](https://arxiv.org/abs/2310.01334).

[^19]: Zichong Li, Chen Liang, Zixuan Zhang, Ilgee Hong, Young Jin Kim, Weizhu Chen, and Tuo Zhao. Slimmoe: Structured compression of large moe models via expert slimming and distillation, 2025. URL [https://arxiv.org/abs/2506.18349](https://arxiv.org/abs/2506.18349).

[^20]: Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation, 2023. URL [https://arxiv.org/abs/2305.01210](https://arxiv.org/abs/2305.01210).

[^21]: Jiawei Liu, Jia Le Tian, Vijay Daita, Yuxiang Wei, Yifeng Ding, Yuhan Katherine Wang, Jun Yang, and Lingming Zhang. Repoqa: Evaluating long context code understanding, 2024. URL [https://arxiv.org/abs/2406.06025](https://arxiv.org/abs/2406.06025).

[^22]: Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models, 2024. URL [https://arxiv.org/abs/2402.14800](https://arxiv.org/abs/2402.14800).

[^23]: Kaijing Ma, Xinrun Du, Yunran Wang, Haoran Zhang, Zhoufutu Wen, Xingwei Qu, Jian Yang, Jiaheng Liu, Minghao Liu, Xiang Yue, Wenhao Huang, and Ge Zhang. Kor-bench: Benchmarking language models on knowledge-orthogonal reasoning tasks, 2025. URL [https://arxiv.org/abs/2410.06526](https://arxiv.org/abs/2410.06526).

[^24]: Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. In *Advances in Neural Information Processing Systems*, 2023.

[^25]: Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. Shortgpt: Layers in large language models are more redundant than you expect, 2024. URL [https://arxiv.org/abs/2403.03853](https://arxiv.org/abs/2403.03853).

[^26]: Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. Compact language models via pruning and knowledge distillation, 2024. URL [https://arxiv.org/abs/2407.14679](https://arxiv.org/abs/2407.14679).

[^27]: Hao Peng, Xin Lv, Yushi Bai, Zijun Yao, Jiajie Zhang, Lei Hou, and Juanzi Li. Pre-training distillation for large language models: A design space exploration, 2024. URL [https://arxiv.org/abs/2410.16215](https://arxiv.org/abs/2410.16215).

[^28]: Zihan Qiu, Zeyu Huang, Bo Zheng, Kaiyue Wen, Zekun Wang, Rui Men, Ivan Titov, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models, 2025a. URL [https://arxiv.org/abs/2501.11873](https://arxiv.org/abs/2501.11873).

[^29]: Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free, 2025b. URL [https://arxiv.org/abs/2505.06708](https://arxiv.org/abs/2505.06708).

[^30]: Angelika Romanou, Negar Foroutan, Anna Sotnikova, Zeming Chen, Sree Harsha Nelaturu, Shivalika Singh, Rishabh Maheshwary, Micol Altomare, Mohamed A Haggag, Alfonso Amayuelas, et al. Include: Evaluating multilingual language understanding with regional knowledge. *arXiv preprint arXiv:2411.19799*, 2024.

[^31]: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In *5th International Conference on Learning Representations*, 2017.

[^32]: Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. Language models are multilingual chain-of-thought reasoners, 2022.

[^33]: Wenfang Sun, Xinyuan Song, Pengxiang Li, Lu Yin, Yefeng Zheng, and Shiwei Liu. The curse of depth in large language models, 2026. URL [https://arxiv.org/abs/2502.05795](https://arxiv.org/abs/2502.05795).

[^34]: Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them, 2022. URL [https://arxiv.org/abs/2210.09261](https://arxiv.org/abs/2210.09261).

[^35]: Shengkun Tang, Oliver Sieberling, Eldar Kurtic, Zhiqiang Shen, and Dan Alistarh. Darwinlm: Evolutionary structured pruning of large language models, 2025. URL [https://arxiv.org/abs/2502.07780](https://arxiv.org/abs/2502.07780).

[^36]: Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. *CoRR*, abs/2507.06261, 2025a. doi: 10.48550/ARXIV.2507.06261.

[^37]: P Team, Xinrun Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, King Zhu, Minghao Liu, Yiming Liang, Xiaolong Jin, Zhenlin Wei, Chujie Zheng, Kaixin Deng, Shawn Gavin, Shian Jia, Sichao Jiang, Yiyan Liao, Rui Li, Qinrui Li, Sirun Li, Yizhi Li, Yunwen Li, David Ma, Yuansheng Ni, Haoran Que, Qiyao Wang, Zhoufutu Wen, Siwei Wu, Tyshawn Hsing, Ming Xu, Zhenzhu Yang, Zekun Moore Wang, Junting Zhou, Yuelin Bai, Xingyuan Bu, Chenglin Cai, Liang Chen, Yifan Chen, Chengtuo Cheng, Tianhao Cheng, Keyi Ding, Siming Huang, Yun Huang, Yaoru Li, Yizhe Li, Zhaoqun Li, Tianhao Liang, Chengdong Lin, Hongquan Lin, Yinghao Ma, Tianyang Pang, Zhongyuan Peng, Zifan Peng, Qige Qi, Shi Qiu, Xingwei Qu, Shanghaoran Quan, Yizhou Tan, Zili Wang, Chenqing Wang, Hao Wang, Yiya Wang, Yubo Wang, Jiajun Xu, Kexin Yang, Ruibin Yuan, Yuanhao Yue, Tianyang Zhan, Chun Zhang, Jinyang Zhang, Xiyue Zhang, Xingjian Zhang, Yue Zhang, Yongchi Zhao, Xiangyu Zheng, Chenghua Zhong, Yang Gao, Zhoujun Li, Dayiheng Liu, Qian Liu, Tianyu Liu, Shiwen Ni, Junran Peng, Yujia Qin, Wenbo Su, Guoyin Wang, Shi Wang, Jian Yang, Min Yang, Meng Cao, Xiang Yue, Zhaoxiang Zhang, Wangchunshu Zhou, Jiaheng Liu, Qunshu Lin, Wenhao Huang, and Ge Zhang. Supergpqa: Scaling llm evaluation across 285 graduate disciplines, 2025. URL [https://arxiv.org/abs/2502.14739](https://arxiv.org/abs/2502.14739).

[^38]: Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL [https://qwenlm.github.io/blog/qwen2.5/](https://qwenlm.github.io/blog/qwen2.5/).

[^39]: Qwen Team. Qwen3-next: Towards ultimate training & inference efficiency, 2025b.

[^40]: Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL [https://qwen.ai/blog?id=qwen3.5](https://qwen.ai/blog?id=qwen3.5).

[^41]: Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark, 2024. URL [https://arxiv.org/abs/2406.01574](https://arxiv.org/abs/2406.01574).

[^42]: Yuxin Wang, Minghua Ma, Zekun Wang, Jingchang Chen, Liping Shan, Qing Yang, Dongliang Xu, Ming Liu, and Bing Qin. CFSP: an efficient structured pruning framework for llms with coarse-to-fine activation information. In *Proceedings of the 31st International Conference on Computational Linguistics*, 2025.

[^43]: Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. In *Findings of the Association for Computational Linguistics*, 2024a. URL [https://aclanthology.org/2024.findings-acl.456](https://aclanthology.org/2024.findings-acl.456).

[^44]: Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. Sheared llama: Accelerating language model pre-training via structured pruning, 2024b. URL [https://arxiv.org/abs/2310.06694](https://arxiv.org/abs/2310.06694).

[^45]: An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025a. URL [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388).

[^46]: Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule, 2025b. URL [https://arxiv.org/abs/2412.06464](https://arxiv.org/abs/2412.06464).

[^47]: Yifei Yang, Zouying Cao, and Hai Zhao. Laco: Large language model pruning via layer collapse, 2024. URL [https://arxiv.org/abs/2402.11187](https://arxiv.org/abs/2402.11187).

[^48]: Xiulong Yuan, Hongqing Chen, Jiaxuan Peng, Fan Zhou, Zhixiang Ruan, Zekun Wang, Bo Zheng, Rui Men, Haiquan Wang, Zhipeng Zhang, et al. Accelerating compound llm training workloads with maestro. *arXiv preprint arXiv:2605.10501*, 2026.

[^49]: Biao Zhang and Rico Sennrich. Root mean square layer normalization. In *Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems*, 2019.
