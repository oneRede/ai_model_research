---
title: "LLaDA MoE v2：扩展混合专家扩散语言模型"
sourceTitle: "LLaDA MoE v2: Scaling Mixture-of-Experts Diffusion Language Models"
sourceUrl: "https://arxiv.org/abs/2608.03457"
sourceAuthors: "Fengqi Zhu, Shaoxuan Xu, Jingyang Ou, Zebin You, Yipeng Xing, Huabin Liu, Xiaolu Zhang, Jun Zhou, Zhenzhong Lan, Yankai Lin, Wayne Xin Zhao, Jianguo Li, Chongxuan Li, Ji-Rong Wen"
sourcePublicationDate: "2026-08-04"
sourceType: "学术论文"
sourceVenue: "arXiv:2608.03457"
sourceFigureCount: 5
translationDate: "2026-08-13"
pipelineRunId: "batch-2026-08-13"
pipelineSource: "translate/batch-2026-08-13/works-ready/llada-moe-v2-scaling-diffusion-language-models-translation.md"
---

# LLaDA MoE v2：扩展混合专家扩散语言模型

Fengqi Zhu¹, Shaoxuan Xu¹, Jingyang Ou¹, Zebin You¹, Yipeng Xing², Huabin Liu², Xiaolu Zhang², Jun Zhou², Zhenzhong Lan², Yankai Lin², Wayne Xin Zhao¹, Jianguo Li², Chongxuan Li¹, Ji-Rong Wen¹

¹ 中国人民大学高瓴人工智能学院、北京市大模型与智能治理重点实验室、教育部下一代智能搜索与推荐工程研究中心  
² 蚂蚁集团

###### 摘要

**扩散语言模型**（Diffusion Language Models，dLLMs）**（通过迭代去噪过程建模文本分布的概率模型）**为自回归（AR）语言建模提供了一种替代方案，然而**混合专家**（Mixture-of-Experts，MoE）**（通过稀疏激活扩展模型容量的架构）**dLLMs 的缩放行为仍然缺乏深入理解。我们系统性地刻画了 MoE dLLMs 的优化超参数、计算分配和架构设计如何随规模变化，识别出与先前 AR 模型缩放趋势的量化差异。具体而言，在优化方面，最优名义批次大小增长更快，而最优学习率衰减更迅速。在模型-数据配置方面，**等浮点运算分析**（IsoFLOP Analysis）**（固定计算预算下优化模型-数据配置以找到最优点）**揭示出轻微的数据侧倾向：最优 token 预算的增长快于激活的模型侧计算。在 MoE 架构方面，更大规模越来越倾向于在固定激活容量下采用更大的专家池，而适度的专家粒度在各个规模上始终有效，分配给共享专家的激活容量占比在不同规模下保持稳定。基于这些发现，我们从头开始在 23.5T tokens 上训练了 LLaDA MoE v2，一个 30B-A3B 的 dLLM。使用大约 65% 的预训练 tokens（相比 Qwen3），LLaDA MoE v2 在多个知识、推理和编码基准上接近 Qwen3 的表现。仅经过监督微调后，它在八个推理和编码基准中的七个上超越了 SDAR Chat，并在多个任务上与 Qwen3 保持接近。这些结果为 MoE dLLMs 建立了实用的缩放定律和设计原则。

<sup>5</sup> <sup>3</sup>

## 1 引言

大语言模型 [^81] 的进步主要通过规模实现：随着模型、数据和计算的增长，性能以系统性和可预测的方式提升 [^38] [^33]。这些进展大多遵循自回归（AR）范式 [^60] [^61] [^9] [^59]，该范式将文本序列的分布分解为沿固定从左到右顺序的下一 token 条件概率。扩散语言模型（dLLMs）[^50] [^62] [^58] [^57] 提供了一种替代的概率表述：它们通过迭代去噪过程定义分布，其中双向模型重建损坏序列的被掩码 tokens，允许在每一步并行解码多个 tokens [^74] [^44] [^12]。最近的 dLLMs 已能匹配强大 AR 模型的能力，无论是从头训练 [^56] [^57] [^55] 还是从预训练 AR 检查点适配 [^8] [^78]。这使其成为未来语言建模的有前景候选方案。

然而，实现这一潜力需要理解 dLLMs 在训练预算增长时的行为。现有的 dLLMs 缩放研究迄今主要集中在密集架构 [^56] [^54] [^71] [^63]。相比之下，AR 文献已广泛采用混合专家（MoE）Transformer 架构 [^70] [^64] [^41] [^21]，这些架构在相同每 token 计算量下将模型容量扩展到远超密集模型的水平 [^20] [^19] [^46] [^47] [^69] [^77]。因此，MoE dLLMs 已开始出现 [^84] [^22] [^80]，但其设计在很大程度上继承了 AR 实践。AR 经验提供了有用的先验，但不能假定其可以直接迁移：dLLMs 优化的是**掩码去噪**（Masked Denoising）**（在部分遮盖的序列上预测被遮盖位置的训练目标）**而非下一 token 预测，监督仅落在被掩码位置上，并且每次预测基于损坏序列而非因果前缀。因此，MoE dLLMs 的缩放行为仍然缺乏充分刻画。

在本工作中，我们系统性地刻画了 MoE dLLMs 在不同计算规模下的缩放行为和架构设计。我们探讨了最优批次大小和学习率如何随计算变化，如何在固定计算预算下在激活的模型侧计算和训练 tokens 之间分配，以及如何将得到的激活预算分解为路由稀疏性、专家粒度和共享容量。表 1 总结了得到的经验发现和相应证据。在这些维度上，AR 经验提供了有用的先验，但需要针对 dLLM 进行专门校准。

表 1：MoE dLLMs 的经验发现。受控扫描涵盖优化、计算分配和 MoE 架构，并通过 LLaDA MoE v2 的扩展验证。

| 类别 | 发现 | 证据 |
| --- | --- | --- |
| 优化 | 最优批次大小随计算增长比 AR 模型更陡峭。 | 图 1, 2 |
|  | 最优学习率随计算衰减比 AR 模型更快。 | 图 1, 2 |
| 计算分配 | 最优计算分配接近平衡，略微倾向数据侧。 | 图 3 |
|  | MoE dLLM 前沿比密集 dLLM 前沿更倾向数据侧。 | 表 2 |
| MoE 架构 | 更低的激活比例在更大规模上越来越有利。 | 图 4 |
|  | 适度的专家粒度，$G=8$ – $16$，在各规模下鲁棒。 | 图 4 |
|  | 最优共享专家比例在各规模下保持 $S=33.3\%$。 | 图 4 |
| 扩展验证 | 使用少 35% 的预训练 tokens 在某些任务上接近 Qwen3。 | 表 3 |
|  | 在显著更低的计算量下匹配或超越 7B-A1B。 | 图 5 |
|  | 在 SFT 后在八个任务中的七个上超越 SDAR Chat。 | 表 4 |

这些发现为扩展 MoE dLLMs 提供了实用指导，我们通过从头开始在 23.5T tokens 上训练 LLaDA MoE v2（一个 30B-A3B 模型）来验证。LLaDA MoE v2 在使用大约 65% 的预训练 tokens 的情况下，在多个知识、推理和编码基准上接近 Qwen3 30B-A3B [^77] 的表现，并且在不依赖此类指导而开发的先前 MoE dLLM [^84] 的一小部分训练计算量下达到匹配性能。标准的监督微调（SFT）将其转变为强大的指令模型：LLaDA MoE v2 在八个推理和编码基准中的七个上超越了 SDAR Chat 30B-A3B [^15]，尽管 Qwen3 额外经过了强化学习（RL）阶段，LLaDA MoE v2 在多个基准上仍与 Qwen3 保持竞争力。

## 2 预备知识

### 2.1 扩散语言模型

掩码扩散语言模型 [^5] [^10] [^50] [^66] 在 token 序列上定义了一个离散扩散过程。令 $x_{0}=(x_{0}^{1},\ldots,x_{0}^{L})$ 表示长度为 $L$ 的干净序列，令 \[MASK\] 为掩码 token。前向过程独立地损坏每个位置：在噪声水平 $t\in[0,1]$ 下，token $x_{0}^{i}$ 以概率 $t$ 被替换为 \[MASK\]，否则保持不变，产生损坏状态 $x_{t}$。

我们训练一个双向 Transformer [^70] $p_{\theta}$ 来反转这一过程，在给定损坏状态的条件下预测每个被掩码位置的干净 token。在噪声水平均匀采样的情况下，训练最小化去噪目标

$$
\mathcal{L}(\theta)=-\mathbb{E}_{x_{0},\,t,\,x_{t}}\left[\frac{1}{t}\sum_{i=1}^{L}\mathbf{1}\left[x_{t}^{i}=\text{[MASK]}\right]\log p_{\theta}\left(x_{0}^{i}\mid x_{t}\right)\right],
$$

该目标是数据负对数似然的上界。在推理时，生成从完全掩码的序列开始，通过迭代去噪步骤逐步揭示 tokens。

### 2.2 混合专家 Transformers 与缩放定律

混合专家（MoE）Transformer 通过将前馈网络替换为 $n_{e}$ 个路由专家和一个轻量级**路由器**（Router）**（MoE 架构中为每个 token 选择专家的机制）** [^35] [^64] [^41] [^21] [^86] [^19]，将模型容量与每 token 计算解耦：总参数量随 $n_{e}$ 缩放，而每个 token 仅由其中的一小部分处理。具体来说，路由器为每个 token 激活 top-$n_{a}$ 个路由专家并组合它们的输出。此外，每个 token 都由一个中间宽度为 $d_{\mathrm{share}}=n_{s}d_{\mathrm{expert}}$ 的共享专家处理，其中 $n_{s}$ 表示以一个路由专家为单位衡量的共享容量。我们用三个变量参数化这样的架构：**激活比例**（Activation Ratio）$A=(n_{a}+n_{s})/(n_{e}+n_{s})$**（每个 token 激活的专家容量占总专家池的比例）**；**专家粒度**（Expert Granularity）$G=2d_{\mathrm{model}}/d_{\mathrm{expert}}$**（衡量路由容量划分为多少个专家的参数，G 越大表示专家越细分）**；以及**共享专家比例**（Shared-Expert Ratio）$S=n_{s}/(n_{a}+n_{s})$**（分配给共享通路的激活容量占比）** [^39] [^1] [^69]。

**缩放定律**（Scaling Laws）**（描述性能随规模变化的经验规律）**使用小规模测量来估计依赖于计算的选择（包括优化超参数、模型-数据配置和架构）应如何在更大预算下变化 [^32] [^38] [^33] [^39] [^1]。在实践中，幂律拟合将最优批次大小和学习率建模为计算预算 $C$ 的函数，而 IsoFLOP 分析固定 $C$ 并扫描模型侧计算与训练 tokens 的配置 [^33] [^7] [^43]。密集模型研究通常将训练计算近似为 $C\approx 6ND$，其中 $N$ 是非嵌入参数数量，$D$ 是训练 tokens 数量，假设所有参数都参与处理每个 token [^38] [^33]。由于这一假设对 MoE 模型不成立，我们遵循先前的 MoE 缩放研究，使用 $C=MD$，其中 $M$ 表示每 token 激活的非嵌入训练浮点运算量 [^7] [^51] [^69]；完整的计算表达式在附录 A.1 中提供。

## 3 MoE dLLMs 的缩放定律

在本节中，我们分三个阶段为 MoE dLLMs 开发缩放框架。我们首先校准依赖于计算的批次大小和学习率，然后估计激活的模型侧计算与训练数据之间的最优分配，最后将得到的模型侧预算分解为激活比例、专家粒度和共享专家比例。

### 3.1 超参数的缩放定律

图 1：名义 token 批次大小和学习率随训练计算的缩放曲线。左：名义 token 批次大小。右：学习率。洋红色虚线表示我们拟合的缩放定律，蓝色虚线显示 DeepSeek LLM [^7] 的参考缩放定律，阴影区域表示我们拟合曲线周围的经验范围。

尽管超参数缩放已针对 AR 模型进行了广泛研究 [^7] [^43]，但相同的依赖于计算的定律是否能描述 dLLMs 仍然是一个开放问题。与优化因果前缀上的下一 token 预测的 AR 训练不同，dLLMs 使用部分观测序列上的掩码去噪目标进行训练。由于每次更新仅监督采样的被掩码位置，名义 token 批次大小不直接对应于有效预测目标数量；例如，在常见的均匀时间步采样下，预期只有一半的 tokens 被预测。这种减少的有效监督可能改变梯度噪声水平、优化稳定性和学习率敏感性。因此，我们研究最优批次大小和学习率如何随计算变化，既为了将它们的缩放行为与 AR 预期进行比较，也为了为后续分析建立稳定、计算高效的设置。

具体来说，我们在一组代表性模型规模（从 158M 到 3.6B）上进行超参数搜索，计算预算从 $10^{18}$ 到 $3\times 10^{20}$；详细的训练和模型设置在附录 A.3 中提供。

如图 1 所示，我们拟合批次大小和学习率相对于训练计算的缩放曲线。得到的缩放定律可以总结为

$$
B^{*}=0.374\cdot C^{0.3481},\quad\eta^{*}=64.8\cdot C^{-0.2447},
$$

其中 $C$ 表示计算预算，$B^{*}$ 和 $\eta^{*}$ 分别表示 dLLM 训练目标下的最优名义 token 批次大小和学习率。

![Refer to caption](imgs/img-001-batchsize_lr_grid.png)

图 2：在 $6 \times 10^{20}$ FLOPs 下批次大小和学习率的联合搜索。每个单元格对应一次训练运行，颜色和叠加值表示训练损失；红色星号标记拟合缩放定律预测和观测到的最佳配置。

为了测试拟合的 dLLM 专用缩放定律是否能外推到其拟合范围之外，我们将批次大小和学习率的联合运行从 $3\times 10^{20}$ 继续到 $6\times 10^{20}$ FLOPs，并检查图 2 中的损失网格。拟合的推荐在二维超参数平面中接近观测到的最佳配置，周围的损失不表明在搜索设置内存在明显更好的替代方案。这一结果支持将拟合的缩放定律作为该规模下依赖于计算的超参数的实用估计。

**优化：高计算量有利于更大批次和更快的学习率衰减。** dLLM 和 AR 训练之间的优化方向是一致的，最优批次大小随训练计算次线性增长，最优学习率在更大计算量下降低。然而，校准以系统性方式存在差异。与 AR 训练相比，我们的 dLLM 拟合显示出更陡峭的批次大小缩放曲线和更快的学习率衰减，将高计算量估计转向更大的名义批次和更小的学习率。例如，在 $10^{20}$ FLOPs 的计算预算下，DeepSeek LLM 的 AR 定律预测相似的学习率，$9.86\times 10^{-4}$ 而我们的拟合为 $8.27\times 10^{-4}$，但最优批次大小要小得多，1.02M tokens 而我们的为 3.43M tokens。这一批次大小差距与上述每个名义 token 的有效监督减少是一致的。因此，AR 缩放可以提供有用的先验，但当拟合趋势出现分歧时，不应取代 dLLM 专用校准。因此，所有后续实验都使用我们拟合的 dLLM 超参数定律。

### 3.2 计算分配的缩放定律

一旦优化超参数校准完成，下一个问题是如何在固定计算预算下在激活的模型侧计算和训练数据之间进行分配。对于 MoE dLLMs，这种权衡与其 AR 对应物不同：每个名义 token 仅在被掩码时贡献预测目标，而路由器作用于随噪声水平和掩码模式变化的损坏状态，而非因果前缀。相对于 AR 训练，固定的名义 token 预算因此在更多变的条件和路由状态集合上提供更少的监督目标。额外的 tokens 可以改善去噪目标和路由器输入的覆盖范围，赋予数据侧投资 dLLM 特有的边际价值；因此不能假定 AR 分配定律可以直接迁移。先前的 IsoFLOP 研究刻画了密集和 MoE AR 模型的这种权衡 [^33] [^7] [^51]，而 dLLMs 的类似研究主要考虑密集架构，使得 MoE dLLMs 的分配行为未得到充分刻画。

为了解决这一差距，我们通过每 token 激活的非嵌入 FLOPs $M$ 来衡量模型侧，这直接捕获了激活的模型计算；数据侧是训练 tokens 数量 $D$，给出 $C=MD$。我们在从 $10^{17}$ 到 $10^{20}$ 训练 FLOPs 的固定计算预算下扫描它们的分配，详细设置在附录 A.4 中提供。对于每个预算，我们识别扫描内的最低损失分配并拟合得到的前沿曲线 $M^{*}(C)$ 和 $D^{*}(C)$。

表 2：语言模型的代表性计算分配缩放定律。我们将 MoE dLLM 前沿与跨建模目标和架构的代表性模型-数据分配定律进行对比。前沿列报告拟合的增长指数。对于 DLMs，我们报告与我们表述匹配的掩码设置；SMDM 系数取自 DLMs 分析。

| 缩放定律 | 建模 | 架构 | 模型前沿 | 数据前沿 |
| --- | --- | --- | --- | --- |
| Kaplan [^38] | AR | 密集 | $M^{*}\propto C^{0.73}$ | $D^{*}\propto C^{0.27}$ |
| Chinchilla [^33] | AR | 密集 | $M^{*}\propto C^{0.49}$ | $D^{*}\propto C^{0.51}$ |
| DeepSeek LLM [^7] | AR | 密集 | $M^{*}\propto C^{0.5243}$ | $D^{*}\propto C^{0.4757}$ |
| Llama 3 [^25] | AR | 密集 | $M^{*}\propto C^{0.463}$ | $D^{*}\propto C^{0.537}$ |
| SMDM [^56] | AR | 密集 | $M^{*}\propto C^{0.644}$ | $D^{*}\propto C^{0.356}$ |
| Ling [^69] | AR | 密集 | $M^{*}\propto C^{0.5422}$ | $D^{*}\propto C^{0.4578}$ |
| Ling [^69] | AR | MoE | $M^{*}\propto C^{0.5095}$ | $D^{*}\propto C^{0.4905}$ |
| SMDM [^56] | 扩散 | 密集 | $M^{*}\propto C^{0.634}$ | $D^{*}\propto C^{0.366}$ |
| Quokka [^54] | 扩散 | 密集 | $M^{*}\propto C^{0.514}$ | $D^{*}\propto C^{0.486}$ |
| DLMs [^71] | 扩散 | 密集 | $M^{*}\propto C^{0.566}$ | $D^{*}\propto C^{0.434}$ |
| 本文 | 扩散 | MoE | $M^{*}\propto C^{0.475}$ | $D^{*}\propto C^{0.525}$ |

图 3：计算分配的 IsoFLOP 分析。左：固定计算预算下不同模型-数据配置的训练损失，星号标记最低损失。中和右：最优非嵌入每 token FLOPs 和训练 tokens 的拟合幂律。

图 3 的左侧面板显示了上述计算预算下的 U 型 IsoFLOP 曲线：较小的模型尽管看到更多 tokens 但受容量限制，而较大的模型因在相同预算下可训练更少 tokens 而受数据限制。星号标记的点给出每个预算的经验计算最优分配 $(M^{*}(C),D^{*}(C))$。

随着计算预算的增加，这些最优点向更大的激活非嵌入每 token FLOPs 和更多训练 tokens 移动，形成模型侧和数据侧分配前沿。图 3 的中间和右侧面板分别绘制了 $M^{*}(C)$ 和 $D^{*}(C)$；两者在对数-对数空间中都遵循近似线性趋势。在固定计算约束下拟合两条前沿给出

$$
M^{*}(C)=0.5152\cdot C^{0.475},\quad D^{*}(C)=1.9411\cdot C^{0.525}.
$$

**计算分配：接近平衡的缩放，略微倾向数据侧。** 两条前沿都随计算增长，它们的指数接近 $0.5$；数据侧指数 $0.525$ 略大于模型侧指数 $0.475$。因此，最优 token 预算的增长快于激活的模型侧计算。我们在表 2 中将这种倾向与 AR 模型和 dLLMs 的代表性计算分配定律进行对比。

在这一比较中，两个匹配的结果有助于分离架构和建模目标的影响。首先，Ling 在 AR 建模下比较密集和 MoE 架构：从密集 AR 转向 MoE AR 将模型/数据指数从 $0.5422/0.4578$ 转移到 $0.5095/0.4905$，表明稀疏激活使 AR 前沿向更多数据移动。其次，SMDM 在密集架构下比较 AR 和 dLLM 目标：从 AR 转向 dLLM 将指数从 $0.644/0.356$ 转移到 $0.634/0.366$，显示相同的数据偏好方向。我们的设置将稀疏激活与 dLLM 目标结合，为解释我们前沿的数据侧倾向提供了视角。虽然现有的密集 dLLM 前沿仍然以模型侧为主导，但我们的 MoE dLLM 前沿达到了更大的数据侧指数 $0.525$。这一结果与稀疏激活和扩散建模分别关联的数据偏好趋势一致。

拟合的前沿产生了一个简单的分配规则：MoE dLLM 的边际计算最好相对更多地花在额外的训练 tokens 上，而不是增加激活的非嵌入每 token FLOPs。遵循这一前沿在每个计算规模上固定了聚合激活模型侧预算 $M^{*}(C)$，同时留下专家数量、专家大小和激活模式未确定。

### 3.3 MoE 架构的缩放定律

将这一固定预算转化为 MoE 架构构成了一个 dLLM 特有的设计问题，因为路由器作用于随噪声水平和掩码模式变化的掩码去噪状态。我们考虑第 2 节中定义的三个架构维度：激活比例 $A$，控制路由稀疏性从而控制路由专家池的大小；专家粒度 $G$，在路由多样性与每个专家容量之间权衡；以及共享专家比例 $S$，确定分配给共享通路的激活容量比例。我们检查这些维度应如何在不同计算规模下配置。

对于每个参考计算规模 $C$，我们将激活的模型侧预算固定为 $M^{*}(C)$，并一次扫描一个架构维度，同时保持其他两个维度固定。详细设置在附录 A.5 中提供。图 4 总结了这些受控架构扫描。

![Refer to caption](imgs/img-002-architecture.png)

图 4：MoE 架构缩放。我们评估激活模型侧预算的不同分解如何影响跨计算规模的训练损失。(a)：激活比例 A，(b)：专家粒度 G，(c)：共享专家比例 S。颜色表示激活模型侧预算 $M^{*}(C)$，红色星号标记每个计算规模下的最低损失配置。

**激活比例：更大规模偏好更稀疏的激活。** 我们在图 4(4) 中检查激活比例 $A$。在固定激活模型侧预算 $M^{*}(C)$ 下，降低 $A$ 通常会降低训练损失，随着计算的增加，这种好处变得更加明显。然而，在最小预算下，第二低的激活比例略微优于最稀疏的设置。这一例外与以下直觉一致：极其稀疏的路由暴露了更大的路由专家池，这可能需要足够的训练计算才能被有效优化。这一趋势表明 MoE dLLMs 可以在更大规模上越来越多地利用这种专家容量。

**专家粒度：$G=8$ – $16$ 鲁棒。** 我们接下来在图 4(4) 中检查专家粒度 $G$。$G$ 在计算规模上没有显示单调趋势，表明它不是主要的缩放方向。相反，它主要反映路由多样性和每个专家容量之间的权衡：更粗粒度的专家提供更强的单独变换但更少的路由选择，而更细粒度的专家以更窄的专家为代价增加路由选择。对于 MoE dLLMs，这种权衡是相关的，因为掩码去噪需要对损坏上下文的多样化专业化和足够的专家表达能力。经验上，在我们的扫描中 $G=8$ 到 $G=16$ 提供了一个鲁棒范围，尽管确切的最优值不随计算系统性变化。

**共享专家比例：$S=33.3\%$ 保持最优。** 我们在图 4(4) 中检查共享专家比例 $S$。在所有计算规模上，损失曲线呈 U 型并在 $S=33.3\%$ 处达到最小值。这一最优值与 AR MoE 设计形成对比：DeepSeekMoE 采用 $S=25\%$，Qwen3 不使用共享专家 [^19] [^77]，而 [^69] 报告最优比例从 $16.7\%$ 下降到 $8.3\%$，促使采用固定的"一个共享专家"规则。我们的 dLLM 扫描反而偏好一个容量随激活模型预算成比例增长的共享通路，而不是一个相对贡献随规模而减小的固定共享组件。因此，我们推导出一个区别于 AR 启发式的 dLLM 特定规则：为每两个单位的路由激活容量维持大约一个单位的共享激活容量。

我们的架构扫描将聚合预算 $M^{*}(C)$ 转化为针对随噪声水平和掩码模式变化的损坏状态上路由的 dLLM 特定架构选择，而非因果前缀。在固定 $M^{*}(C)$ 下，更大规模偏好更稀疏的激活，而适度的专家粒度和稳定的共享容量比例在研究的规模上保持鲁棒。

## 4 训练大规模 MoE dLLMs

在上述推导的缩放定律指导下，我们设计了 LLaDA MoE v2，一个 30B-A3B 的 MoE dLLM，并在 23.5T tokens 上训练。我们在下面简要介绍其架构和训练策略，附录 B 中提供了更多细节。

**模型。** LLaDA MoE v2 有 30B 总参数，每个 token 激活 3B 参数。基于我们的缩放分析，我们采用 $(A,G,S)=(9.09\%,8,33.3\%)$。每层包含 128 个细粒度路由专家，采用 top-8 路由和一个宽度为 $4d_{\mathrm{expert}}$ 的共享专家，对应于 $n_{s}=4$，产生 $A=(8+4)/(128+4)=9.09\%$ 和 $S=4/(8+4)=33.3\%$。

**训练策略。** 我们分五个阶段训练 LLaDA MoE v2。阶段 1 和 2 各使用 10T tokens，随后在阶段 3 中进行 2T tokens 的退火。在阶段 4 中，我们将 RoPE 基数从 10,000 增加到 500,000，将上下文长度从 4K 扩展到 32K，并继续训练 500B tokens。阶段 5 以 1T tokens 的长上下文退火结束，产生最终的预训练基础模型。
### 4.1 基准测试结果

表 3：基准测试结果。我们报告了从头训练的 LLaDA MoE v2 30B-A3B 扩散语言模型的结果，以及代表性的扩散语言模型和自回归模型 Qwen3。CPT 表示从自回归模型进行持续预训练。符号 <sup>∗</sup> 表示 Qwen3 [^77] 中报告的结果，<sup>†</sup> 表示 LLaDA MoE 7B-A1B [^84] 中报告的结果。

<table><tbody><tr><th></th><td>LLaDA MoE v2</td><td>SDAR Sci</td><td>LLaDA MoE</td><td>Dream 7B</td><td>LLaDA 8B</td><td>Qwen3</td></tr><tr><th>架构</th><td>MoE</td><td>MoE</td><td>MoE</td><td>Dense</td><td>Dense</td><td>MoE</td></tr><tr><th>建模方式</th><td>Diffusion</td><td>Diffusion</td><td>Diffusion</td><td>Diffusion</td><td>Diffusion</td><td>AR</td></tr><tr><th>方法</th><td>Pretrain</td><td>CPT</td><td>Pretrain</td><td>CPT</td><td>Pretrain</td><td>Pretrain</td></tr><tr><th>总参数量</th><td>30B</td><td>30B</td><td>7B</td><td>7B</td><td>8B</td><td>30B</td></tr><tr><th>激活参数量</th><td>3B</td><td>3B</td><td>1B</td><td>7B</td><td>8B</td><td>3B</td></tr><tr><th>训练 token 数</th><td>23.5T</td><td>36 + 1.05T</td><td>21T</td><td>18 + 0.58T</td><td>2.3T</td><td>36T</td></tr><tr><th colspan="7">通用任务</th></tr><tr><th>MMLU</th><td>78.01</td><td>82.72</td><td>64.59 <sup>†</sup></td><td>69.50 <sup>†</sup></td><td>65.90 <sup>†</sup></td><td>81.38 <sup>∗</sup></td></tr><tr><th>MMLU-Pro</th><td>57.28</td><td>56.96</td><td>39.16 <sup>†</sup></td><td>48.15 <sup>†</sup></td><td>41.80 <sup>†</sup></td><td>61.49 <sup>∗</sup></td></tr><tr><th>CEval</th><td>76.11</td><td>86.95</td><td>65.56 <sup>†</sup></td><td>59.18 <sup>†</sup></td><td>70.50 <sup>†</sup></td><td>87.50</td></tr><tr><th>CMMLU</th><td>77.99</td><td>85.82</td><td>65.65 <sup>†</sup></td><td>60.87 <sup>†</sup></td><td>69.90 <sup>†</sup></td><td>86.35</td></tr><tr><th>HellaSwag</th><td>77.19</td><td>56.38</td><td>65.46</td><td>74.37</td><td>70.82</td><td>77.92</td></tr><tr><th>KorBench</th><td>45.92</td><td>40.08</td><td>31.20 <sup>†</sup></td><td>37.44 <sup>†</sup></td><td>33.68 <sup>†</sup></td><td>44.96</td></tr><tr><th colspan="7">推理任务</th></tr><tr><th>GSM8K</th><td>83.93</td><td>86.13</td><td>66.41 <sup>†</sup></td><td>77.79 <sup>†</sup></td><td>70.70 <sup>†</sup></td><td>91.81 <sup>∗</sup></td></tr><tr><th>MATH</th><td>54.72</td><td>48.52</td><td>36.10 <sup>†</sup></td><td>39.60 <sup>†</sup></td><td>27.30 <sup>†</sup></td><td>59.04 <sup>∗</sup></td></tr><tr><th>OlympiadBench</th><td>28.74</td><td>24.44</td><td>10.07 <sup>†</sup></td><td>10.22 <sup>†</sup></td><td>6.85 <sup>†</sup></td><td>30.96</td></tr><tr><th colspan="7">代码任务</th></tr><tr><th>CRUXEval</th><td>50.62</td><td>53.00</td><td>38.94</td><td>40.31</td><td>36.38</td><td>56.88</td></tr><tr><th>MBPP</th><td>71.00</td><td>60.40</td><td>52.40 <sup>†</sup></td><td>56.20 <sup>†</sup></td><td>38.20 <sup>†</sup></td><td>74.40 <sup>∗</sup></td></tr><tr><th>MultiPL-E</th><td>53.78</td><td>33.66</td><td>41.13 <sup>†</sup></td><td>27.60 <sup>†</sup></td><td>23.61 <sup>†</sup></td><td>66.53 <sup>∗</sup></td></tr><tr><th>HumanEval</th><td>50.00</td><td>33.54</td><td>45.73 <sup>†</sup></td><td>57.90 <sup>†</sup></td><td>33.50 <sup>†</sup></td><td>52.44</td></tr><tr><th>LiveCodeBench v6</th><td>31.86</td><td>39.87</td><td>16.18 <sup>†</sup></td><td>14.87 <sup>†</sup></td><td>2.53 <sup>†</sup></td><td>49.18</td></tr><tr><th>BigCodeBench</th><td>41.84</td><td>33.86</td><td>21.23 <sup>†</sup></td><td>18.33 <sup>†</sup></td><td>13.42 <sup>†</sup></td><td>45.70</td></tr></tbody></table>

如表 3 所示，我们将 LLaDA MoE v2 30B-A3B 与五个代表性的扩散语言模型和自回归基准进行了比较。规模最接近的基准是 SDAR Sci 和 Qwen3 30B-A3B：前者通过从 Qwen3 [^15] 持续预训练获得，后者是一个强大的自回归混合专家模型 [^77]。LLaDA MoE v2 在 23.5T token 上从头训练，相当于 SDAR Sci 37.05T 的 63% 和 Qwen3 36T 的 65%。

在全部 15 个基准测试中，LLaDA MoE v2 在评估的扩散语言模型中取得了最高的平均分（58.60），比 SDAR Sci 高出 3.78 分，比更小规模的扩散语言模型基准至少高出 12.44 分。尽管是从头预训练而非从自回归检查点初始化，它相比 SDAR Sci 的优势在代码基准测试上尤为明显，如 HumanEval（$+16.46$）和 BigCodeBench（$+7.98$）。与 Qwen3 相比，LLaDA MoE v2 在中文知识基准 CEval 和 CMMLU 以及某些代码任务上显示出较大差距，但在包括 OlympiadBench（$-2.22$）和 HumanEval（$-2.44$）在内的多个推理和代码基准上保持接近。在预训练 token 数仅为 Qwen3 65% 的情况下取得这样的性能，支持了我们缩放定律指导设计的有效性。

为进一步评估缩放定律指导设计的实用价值，图 5 比较了 LLaDA MoE v2 30B-A3B 在不同训练浮点运算量下的基准性能与未经缩放定律指导开发的混合专家扩散语言模型 LLaDA MoE 7B-A1B 的性能。在涵盖知识、数学和代码的多样化基准测试中，30B-A3B 模型以显著更低的计算预算达到了与 7B-A1B 模型相当或更好的性能。具体而言，我们的模型在大约 50% 的训练浮点运算量下就在 MMLU、GSM8K 和 KorBench 上超过了 LLaDA MoE 7B-A1B，甚至在不到 10% 的训练浮点运算量下就在 HellaSwag 上超过了它。基准测试结果和计算控制比较提供了实际证据，表明我们的缩放定律为大规模混合专家扩散语言模型设计提供了可行的指导，提高了预训练计算转化为下游性能的效率。

图 5：基准性能与训练计算量的关系。红色曲线显示缩放定律指导的 LLaDA MoE v2 30B-A3B 模型在不同训练 token 预算下的评估结果，黑色菱形表示未经缩放定律指导训练的 LLaDA MoE 7B-A1B 模型。

### 4.2 监督微调

表 4：监督微调结果。LLaDA MoE v2 30B-A3B 指令模型与 Qwen3 30B-A3B 和 SDAR Chat 30B-A3B 在推理和代码基准上的比较。标记 <sup>∗</sup> 的值来自原始 Qwen3 论文，标记 <sup>†</sup> 的值来自原始 SDAR 论文。

<table><thead><tr><th></th><th colspan="4">推理</th><th colspan="4">代码</th></tr><tr><th>模型</th><th>Math</th><th>Olympiad</th><th>AIME24</th><th>AIME25</th><th>MBPP</th><th>LCB v6</th><th>BigCode</th><th>MultiPL-E</th></tr></thead><tbody><tr><th>Qwen3</th><td>89.80 <sup>∗</sup></td><td>57.26</td><td>32.80 <sup>∗</sup></td><td>21.60 <sup>∗</sup></td><td>85.48</td><td>31.50</td><td>41.14</td><td>66.60</td></tr><tr><th>SDAR Chat</th><td>77.80 <sup>†</sup></td><td>34.93</td><td>16.70 <sup>†</sup></td><td>10.80 <sup>†</sup></td><td>71.60 <sup>†</sup></td><td>21.70 <sup>†</sup></td><td>39.39</td><td>45.00</td></tr><tr><th>LLaDA MoE v2</th><td>80.02</td><td>46.44</td><td>30.00</td><td>20.00</td><td>81.03</td><td>27.75</td><td>35.53</td><td>67.52</td></tr></tbody></table>

前述实验评估了预训练的 LLaDA MoE v2 基础模型的能力。我们进一步检验通过标准监督微调（SFT），这个缩放定律指导的预训练模型是否能有效适应指令遵循和复杂推理。从 LLaDA MoE v2 30B-A3B 基础检查点出发，我们构建了一个指令模型，并在数学推理和代码生成基准上进行评估。

我们在 700 万个指令-响应样本上对 LLaDA MoE v2 进行三轮微调。对于每个样本，我们保持提示未损坏，仅对响应应用公式 1 中的掩码过程，并在掩码位置上计算去噪损失。我们使用 512 个序列的批次大小和 $5.0\times 10^{-6}$ 的峰值学习率，在前 8% 的步骤中线性预热。我们在监督微调后未应用强化学习（RL），将其整合留待未来工作。

我们将所得模型与 Qwen3 30B-A3B no think 和 SDAR Chat 30B-A3B 进行比较。如表 4 所示，LLaDA MoE v2 在全部四个推理基准和四个代码基准中的三个上优于 SDAR Chat。尽管使用了更少的预训练 token，且未使用 Qwen3 [^77] 采用的额外强化学习，LLaDA MoE v2 在 AIME 24/25 [^2]、MBPP 和 LiveCodeBench 上仍与 Qwen3 保持接近，并在 MultiPL-E 上超过了它。这些结果表明 LLaDA MoE v2 仅通过监督微调就获得了强大的推理和编码能力。

## 5 相关工作

扩散语言模型（dLLMs）最近作为语言建模的新范式出现 [^5] [^10] [^14] [^27] [^29] [^50] [^66] [^76] [^82] [^45] [^57] [^67] [^40]。特别是，掩码离散扩散模型被视为自回归（AR）模型的潜在替代方案，因为它们可以在每个去噪步骤中并行生成多个 token [^4] [^73] [^12] [^16]。近期工作通过从头预训练 [^57] [^84] [^83] 和适配预训练的自回归模型 [^24] [^78] [^8] 两种方式扩展了扩散语言模型。与此同时，一些研究已开始刻画扩散语言模型的缩放行为 [^56] [^54] [^71] [^53] [^63]，但大多数局限于密集架构。

混合专家（MoE）与缩放定律。通过用多个细粒度专家替代单一前馈网络，并为每个 token 选择性激活参数子集，混合专家架构使模型能够增加容量而不会按比例增加计算量 [^64] [^41] [^21] [^20] [^37] [^19]。在语言模型缩放定律研究 [^38] [^33] 的基础上，近期研究利用缩放洞察指导计算高效的混合专家架构设计 [^17] [^1] [^51]，一些工作也将混合专家应用于扩散语言模型 [^84] [^22]。然而，现有的混合专家扩散语言模型主要采用为自回归模型开发的架构选择，其优化、计算分配和专家架构的缩放行为在很大程度上未被刻画。

## 6 结论

在本工作中，我们刻画了混合专家扩散语言模型的优化超参数、模型-数据分配和专家架构如何缩放，发现自回归趋势提供了有用的先验，但需要针对扩散语言模型的特定校准。这些结果产生了实用的设计原则，指导了 LLaDA MoE v2 30B-A3B 的从头训练。该模型在使用更少预训练 token 的情况下，在多个基准测试上接近 Qwen3，而仅通过监督微调（无强化学习）就产生了在八个推理和代码任务中的七个上优于 SDAR Chat 的指令模型。我们的实验分别改变各缩放维度，因此未能捕捉它们之间的交互作用。尽管如此，大规模结果证明了缩放定律指导的混合专家扩散语言模型设计的实用价值。

## 参考文献

## 附录 A 混合专家扩散语言模型的缩放定律

### A.1 计算核算

我们区分总参数数量、激活参数数量和缩放分析中使用的激活计算量。设 $n_{\mathrm{layer}}$ 为 Transformer 层数，$d_{\mathrm{model}}$ 为隐藏层大小，$s$ 为序列长度，$r_{\mathrm{kv}}=n_{\mathrm{kvheads}}/n_{\mathrm{heads}}$ 为键值头与查询头的比例。每个混合专家层包含 $n_{e}$ 个路由专家，每个专家的中间宽度为 $d_{\mathrm{expert}}$，其中每个 token 选择 $n_{a}$ 个专家。在实现的架构中，每个 token 还额外由一个中间宽度为 $d_{\mathrm{share}}$ 的共享专家处理。为了用统一单位表达路由容量和共享容量，我们定义 $n_{s}\equiv d_{\mathrm{share}}/d_{\mathrm{expert}}$，或等价地 $d_{\mathrm{share}}=n_{s}d_{\mathrm{expert}}$。因此，$n_{s}$ 表示以一个路由专家为单位度量的共享容量，而不是共享专家的物理数量。使用 SwiGLU [^65]，三个投影矩阵对每个路由专家包含 $3d_{\mathrm{model}}d_{\mathrm{expert}}$ 个参数，对单个共享专家包含 $3d_{\mathrm{model}}d_{\mathrm{share}}=3n_{s}d_{\mathrm{model}}d_{\mathrm{expert}}$ 个参数。

忽略偏置和归一化参数，总非嵌入参数数量和激活非嵌入参数数量为

$$
\displaystyle P_{\mathrm{nonemb}}
$$

$$
\displaystyle=n_{\mathrm{layer}}\left[2d_{\mathrm{model}}^{2}(1+r_{\mathrm{kv}})+d_{\mathrm{model}}n_{e}+3d_{\mathrm{model}}(n_{e}d_{\mathrm{expert}}+d_{\mathrm{share}})\right],
$$
$$
\displaystyle P_{\mathrm{act,nonemb}}
$$

$$
\displaystyle=n_{\mathrm{layer}}\left[2d_{\mathrm{model}}^{2}(1+r_{\mathrm{kv}})+d_{\mathrm{model}}n_{e}+3d_{\mathrm{model}}(n_{a}d_{\mathrm{expert}}+d_{\mathrm{share}})\right].
$$

第一项对应查询、键、值和输出投影；第二项对应路由器，它为每个 token 对所有 $n_{e}$ 个路由专家评分；最后一项对应路由专家和单个共享专家。代入 $d_{\mathrm{share}}=n_{s}d_{\mathrm{expert}}$ 可分别恢复等价单位形式 $3d_{\mathrm{model}}d_{\mathrm{expert}}(n_{e}+n_{s})$ 和 $3d_{\mathrm{model}}d_{\mathrm{expert}}(n_{a}+n_{s})$。我们使用一个输入嵌入矩阵和一个单独的语言模型头矩阵，对于词汇表大小 $V$，每个包含 $Vd_{\mathrm{model}}$ 个参数。它们合并的 $2Vd_{\mathrm{model}}$ 个参数在报告总参数数量或激活参数数量时包含在内，但在下面使用的非嵌入量中排除。

我们将一次乘加运算计为两次浮点运算。一层每个 token 的前向浮点运算数近似为

$$
\displaystyle F_{\mathrm{attn}}
$$

$$
\displaystyle=4d_{\mathrm{model}}^{2}(1+r_{\mathrm{kv}})+4sd_{\mathrm{model}},
$$
$$
\displaystyle F_{\mathrm{MoE}}
$$

$$
\displaystyle=2d_{\mathrm{model}}n_{e}+6d_{\mathrm{model}}(n_{a}d_{\mathrm{expert}}+d_{\mathrm{share}}),
$$

其中 $F_{\mathrm{attn}}$ 中的两项分别对应注意力投影和两个序列级注意力矩阵乘法。$F_{\mathrm{MoE}}$ 中的项对应路由、选中的路由专家和单个共享专家。将反向传播近似为前向传播的两倍，我们将每个 token 的激活非嵌入浮点运算量定义为

$$
M=3n_{\mathrm{layer}}\left[4d_{\mathrm{model}}^{2}(1+r_{\mathrm{kv}})+4sd_{\mathrm{model}}+2d_{\mathrm{model}}n_{e}+6d_{\mathrm{model}}(n_{a}d_{\mathrm{expert}}+d_{\mathrm{share}})\right].
$$

此浮点运算核算省略了输入嵌入、语言模型头、归一化操作、非线性和注意力 softmax。我们模型中的所有 Transformer 层共享相同的架构；因此，公式 8 将每层浮点运算量乘以 $n_{\mathrm{layer}}$。

最后，设 $D$ 表示训练期间处理的名义 token 总数，包括掩码和可见位置。我们缩放定律实验中使用的计算预算为

$$
C=MD.
$$

掩码和可见 token 产生相同的 Transformer 计算量，因此采样的损坏水平改变监督预测目标的数量，但不改变核算的浮点运算量。

### A.2 混合专家实现

在整个缩放实验中，我们使用混合专家 Transformer 架构，其中每一层将分组查询注意力（GQA）[^3] 与混合专家前馈块配对，后者包含附录 A.1 中参数化的路由专家和单个共享专家，全部实现为 SwiGLU 网络；唯一的例外出现在架构实验中，某些配置省略了共享专家。

路由在每一层对每个 token 独立执行。给定 token 表示 $h$，线性路由器产生 logits $r(h)\in\mathbb{R}^{n_{e}}$ 和路由分数 $p(h)=\operatorname{softmax}(r(h))$。设 $\mathcal{T}(h)$ 包含 $n_{a}$ 个最大路由分数的索引。我们将选中的分数重新归一化为

$$
w_{i}(h)=\frac{p_{i}(h)}{\sum_{j\in\mathcal{T}(h)}p_{j}(h)},\qquad i\in\mathcal{T}(h).
$$

路由专家输出为

$$
E_{\mathrm{route}}(h)=\sum_{i\in\mathcal{T}(h)}w_{i}(h)E_{i}(h).
$$

路由和共享通路组合为 $E_{\mathrm{share}}(h)+\lambda E_{\mathrm{route}}(h)$，其中 $E_{i}$ 是路由专家，$E_{\mathrm{share}}$ 是共享专家，$\lambda$ 平衡两条通路的输出尺度。共享专家比例实验中的纯路由配置省略共享项并且不使用缩放因子。

对于 $n_{s}>0$ 的配置，我们通过在初始化时匹配共享和路由通路的期望输出范数来估计 $\lambda$，遵循 [^48] 的门缩放启发式。将宽度为 $n_{s}d_{\mathrm{expert}}$ 的共享专家视为 $n_{s}$ 个专家宽度单位，并假设所有专家输出在初始化时具有相等的范数且两两正交，共享通路的范数与 $\sqrt{n_{s}}$ 成正比，而未缩放的路由通路的范数与 $(\sum_{i\in\mathcal{T}(h)}w_{i}(h)^{2})^{1/2}$ 成正比。将路由器 logits 的初始化分布近似为 $r(h)\sim\mathcal{N}(0,I_{n_{e}})$，我们估计

$$
\lambda=\mathbb{E}_{r(h)\sim\mathcal{N}(0,I_{n_{e}})}\left[\frac{\sqrt{n_{s}}}{\left(\sum_{i\in\mathcal{T}(h)}w_{i}(h)^{2}\right)^{1/2}}\right],
$$

其中期望通过蒙特卡洛采样近似，对每个具有共享专家的架构配置独立进行。

我们与模型的其余部分联合训练路由器。训练目标为

$$
\mathcal{L}=\mathcal{L}_{\mathrm{diff}}+\alpha_{\mathrm{aux}}\mathcal{L}_{\mathrm{bal}}+\alpha_{z}\mathcal{L}_{z},
$$

其中 $\mathcal{L}_{\mathrm{diff}}$ 是公式 1 中的去噪目标，$\mathcal{L}_{\mathrm{bal}}$ 是负载平衡辅助损失，$\mathcal{L}_{z}$ 是路由器 $z$ 损失；遵循常用的混合专家训练设置，我们设置 $\alpha_{\mathrm{aux}}=0.01$ 和 $\alpha_{z}=0.001$ [^21] [^64] [^86]。两个辅助项都直接添加到去噪损失中，它们的系数在所有缩放和架构扫描中保持固定。

### A.3 超参数缩放

我们在三个模型规模——158M、1B 和 3.6B——下进行超参数缩放实验，计算预算范围从 $10^{18}$ 到 $3\times 10^{20}$ 浮点运算。所有运行使用相同的预训练数据和 4096 的序列长度，并优化公式 1 中的去噪目标。我们使用 AdamW，参数为 $(\beta_{1},\beta_{2})=(0.9,0.95)$ 和 0.1 的权重衰减。学习率在 2,000 个优化器步骤内线性预热到峰值 $\eta$，使优化在峰值率保持恒定直到训练计算的最后 10% 之前进入稳定状态，然后使用余弦调度衰减到 $0.1\eta$。

对于每个模型规模，我们联合搜索全局名义 token 批次大小 $B$ 和峰值学习率 $\eta$，同时在每个计算设置内保持架构和训练 token 预算固定。表 5 总结了模型架构、计算预算和相应的搜索网格。

表 5：超参数缩放扫描的配置。上部分列出模型架构；下部分报告计算预算 $C$（浮点运算）以及全局名义 token 批次大小 $B$ 和峰值学习率 $\eta$ 的联合搜索网格。

<table><thead><tr><th colspan="9">模型架构</th></tr></thead><tbody><tr><th>模型规模</th><td><math><semantics><msub><mi>n</mi> <mi>layer</mi></msub> <annotation>n_{\mathrm{layer}}</annotation></semantics></math></td><td><math><semantics><msub><mi>d</mi> <mi>model</mi></msub> <annotation>d_{\mathrm{model}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>heads</mi></msub> <annotation>n_{\mathrm{heads}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>kvheads</mi></msub> <annotation>n_{\mathrm{kvheads}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>e</mi></msub> <annotation>n_{e}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>a</mi></msub> <annotation>n_{a}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>s</mi></msub> <annotation>n_{s}</annotation></semantics></math></td><td><math><semantics><msub><mi>d</mi> <mi>expert</mi></msub> <annotation>d_{\mathrm{expert}}</annotation></semantics></math></td></tr><tr><th>158M</th><td>6</td><td>256</td><td>8</td><td>2</td><td>64</td><td>4</td><td>1</td><td>256</td></tr><tr><th>1B</th><td>10</td><td>640</td><td>10</td><td>2</td><td>64</td><td>4</td><td>1</td><td>640</td></tr><tr><th>3.6B</th><td>16</td><td>1024</td><td>16</td><td>4</td><td>64</td><td>4</td><td>1</td><td>1024</td></tr><tr><th colspan="9">搜索配置</th></tr><tr><th>模型规模</th><td colspan="3"><math><semantics><mi>C</mi> <annotation>C</annotation></semantics></math></td><td colspan="2"><math><semantics><mi>B</mi> <annotation>B</annotation></semantics></math></td><td colspan="3"><math><semantics><mi>η</mi> <annotation>\eta</annotation></semantics></math></td></tr><tr><th>158M</th><td colspan="3"><math><semantics><mrow><mrow><mo>{</mo> <mrow><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>6</mn><mo>,</mo><mn>8</mn></mrow> <mo>}</mo></mrow> <mo>×</mo> <msup><mn>10</mn> <mn>18</mn></msup></mrow> <annotation>\{1,2,3,6,8\}\times 10^{18}</annotation></semantics></math></td><td colspan="2"><math><semantics><mrow><mo>{</mo> <mrow><msup><mn>2</mn> <mn>17</mn></msup><mo>,</mo><msup><mn>2</mn> <mn>18</mn></msup><mo>,</mo><msup><mn>2</mn> <mn>19</mn></msup><mo>,</mo><msup><mn>2</mn> <mn>20</mn></msup></mrow> <mo>}</mo></mrow> <annotation>\{2^{17},2^{18},2^{19},2^{20}\}</annotation></semantics></math></td><td colspan="3"><math><semantics><mrow><mrow><mo>{</mo> <mrow><mn>1</mn><mo>,</mo><mn>1.4</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>2.8</mn></mrow> <mo>}</mo></mrow> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>3</mn></mrow></msup></mrow> <annotation>\{1,1.4,2,2.8\}\times 10^{-3}</annotation></semantics></math></td></tr><tr><th>1B</th><td colspan="3"><math><semantics><mrow><mrow><mo>{</mo> <mrow><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>6</mn><mo>,</mo><mn>8</mn><mo>,</mo><mn>10</mn></mrow> <mo>}</mo></mrow> <mo>×</mo> <msup><mn>10</mn> <mn>19</mn></msup></mrow> <annotation>\{1,2,3,6,8,10\}\times 10^{19}</annotation></semantics></math></td><td colspan="2"><math><semantics><mrow><mo>{</mo> <mrow><msup><mn>2</mn> <mn>18</mn></msup><mo>,</mo><msup><mn>2</mn> <mn>19</mn></msup><mo>,</mo><msup><mn>2</mn> <mn>20</mn></msup><mo>,</mo><msup><mn>2</mn> <mn>21</mn></msup></mrow> <mo>}</mo></mrow> <annotation>\{2^{18},2^{19},2^{20},2^{21}\}</annotation></semantics></math></td><td colspan="3"><math><semantics><mrow><mrow><mo>{</mo> <mrow><mn>0.7</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>1.4</mn><mo>,</mo><mn>2</mn></mrow> <mo>}</mo></mrow> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>3</mn></mrow></msup></mrow> <annotation>\{0.7,1,1.4,2\}\times 10^{-3}</annotation></semantics></math></td></tr><tr><th>3.6B</th><td colspan="3"><math><semantics><mrow><mrow><mo>{</mo> <mrow><mn>0.8</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn></mrow> <mo>}</mo></mrow> <mo>×</mo> <msup><mn>10</mn> <mn>20</mn></msup></mrow> <annotation>\{0.8,1,2,3\}\times 10^{20}</annotation></semantics></math></td><td colspan="2"><math><semantics><mrow><mo>{</mo> <msup><mn>2</mn> <mn>21</mn></msup><mo>,</mo><msup><mn>2</mn> <mn>22</mn></msup><mo>,</mo><msup><mn>2</mn> <mn>23</mn></msup> <mo>}</mo></mrow> <annotation>\{2^{21},2^{22},2^{23}\}</annotation></semantics></math></td><td colspan="3"><math><semantics><mrow><mrow><mo>{</mo> <mrow><mn>1</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>7</mn></mrow> <mo>}</mo></mrow> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>4</mn></mrow></msup></mrow> <annotation>\{1,4,7\}\times 10^{-4}</annotation></semantics></math></td></tr></tbody></table>

对于每次运行，我们将其损失定义为在分配的训练浮点运算的最后 $0.5\%$ 上的平均训练损失。在每个计算预算下，我们识别出最小平均损失，并将损失不超过该最小值 $0.25\%$ 的配置视为近似最优 [^7]。我们将所有近似最优配置纳入批次大小和学习率对 $C$ 的对数-对数线性回归，得出最终的缩放定律。

### A.4 计算分配缩放

我们在 $10^{17}$ 到 $10^{20}$ 浮点运算的计算预算范围内进行等浮点运算扫描。所有运行使用与附录 A.3 中超参数缩放实验相同的预训练数据、去噪目标和优化器。在每个计算预算 $C$ 下，我们根据正文报告的拟合超参数缩放定律设置名义 token 批次大小和峰值学习率。表 8 和表 9 分别报告候选模型架构及其对应的分配配置。由于优化器步数在不同模型-数据分配间变化，我们为每次运行调整预热长度。设 $T$ 表示分配隐含的优化器步数总数。我们在 $T_{\mathrm{warm}}=\max(0.01T,100)$ 步内线性预热学习率到峰值，在训练步骤的最后 $10\%$ 之前维持峰值率，然后使用余弦调度将其衰减到峰值的 $10\%$。100 步的下限为优化稳定性提供了最小预热期 [^56]。

对于每个计算预算，我们评估一组混合专家模型，它们跨越不同的激活模型侧计算量。我们通过每个 token 的激活非嵌入浮点运算量 $M$ 度量模型侧，并为每个模型分配训练 token 预算 $D=C/M$。在每个计算预算下，我们选择损失最低的评估分配作为经验最优分配点。我们在对数-对数空间使用线性回归拟合选中的模型侧最优点，并从 $D^{*}(C)=C/M^{*}(C)$ 导出相应的数据侧边界，得出正文报告的最终计算分配缩放定律。### A.5 MoE 架构缩放

我们在五个参考计算预算下进行 MoE 架构扫描，$C\in\{6\times 10^{17},2\times 10^{18},6\times 10^{18},2\times 10^{19},6\times 10^{19}\}$ FLOPs。在每个参考预算下，我们保持 Transformer 骨干固定，每次只改变一个架构维度，同时尽可能将其他两个维度保持在离散配置允许的固定值。扫描网格使用第 2 节中的定义进行参数化：激活比例**（每个 token 激活的专家容量占总专家池的比例）** $A=(n_{a}+n_{s})/(n_{e}+n_{s})$，专家粒度**（衡量路由容量划分为多少个专家的参数）** $G=2d_{\mathrm{model}}/d_{\mathrm{expert}}$，以及共享专家比例**（分配给共享通路的激活容量占比）** $S=n_{s}/(n_{a}+n_{s})$。我们在参考预算 $C$ 处评估拟合的计算分配缩放定律，以获得目标激活非嵌入 FLOPs 每 token $M^{*}(C)$ 和计算最优 token 数 $D^{*}(C)$。为了使架构选择能够代表大规模预训练中常用的过训练机制 [^25] [^23] [^69]，我们将每个候选模型训练 $3D^{*}(C)$ 个名义 token，对应约 $3C$ 训练 FLOPs。因此，我们在实际训练预算 $3C$ 处评估拟合的超参数缩放定律，并相应地设置名义 token 批次大小 $B$ 和峰值学习率 $\eta$。通用分配目标和训练超参数在表 10 中报告；三个扫描中使用的精确模型架构在表 11、12 和 13 中报告。所有其余训练设置与附录 A.4 中的计算分配扫描相同。

在每个扫描中，候选架构的构造使得，忽略可忽略不计的路由器贡献，激活模型侧预算 $M^{*}(C)$ 得以保持。在激活比例扫描中，我们固定 Transformer 骨干、$n_{a}$、$n_{s}$ 和 $d_{\mathrm{expert}}$，只改变路由专家数量 $n_{e}$：每 token 的激活专家计算保持不变，而总参数量随着 $A$ 的降低而增长。在专家粒度扫描中，我们改变 $d_{\mathrm{expert}}$ 并按反比例缩放 $n_{e}$、$n_{a}$ 和 $n_{s}$，保持 $A$、$S$ 以及路由和共享激活宽度 $n_{a}d_{\mathrm{expert}}$ 和 $n_{s}d_{\mathrm{expert}}$。在共享专家比例扫描中，我们固定 $n_{e}$ 和 $d_{\mathrm{expert}}$，通过改变 $n_{s}$ 和 $n_{a}$ 在共享通路和路由通路之间重新分配固定的激活专家宽度 $(n_{a}+n_{s})d_{\mathrm{expert}}$；由于 $n_{s}$ 在 $A$ 的分母中，这种重新分配会导致候选模型之间激活比例的轻微漂移，这是可忽略的，不会影响受控比较。

对于每次运行，我们将其损失定义为在其分配的训练 FLOPs 的最后 $0.5\%$ 上的平均训练损失，如附录 A.3 所述。在每个参考预算下，我们沿每个架构维度选择损失最低的候选模型。

## 附录 B 训练大规模 MoE dLLMs

### B.1 模型架构

我们训练 LLaDA MoE v2 30B-A3B，这是一个大型 MoE dLLM，其详细架构在表 6 中报告。路由规则、路由和共享专家输出的组合以及辅助目标遵循附录 A.2。

Transformer 骨干包含 32 层，隐藏大小为 3072，并使用**分组查询注意力**（Grouped-Query Attention, GQA）**（多个查询头共享一组键值头，降低 KV cache 内存占用）**，具有 32 个查询头和 4 个键值头。模型使用 157,184 个 token 的词汇表。所有 Transformer 层都使用 MoE 前馈块。

为了实现架构缩放建议，我们使用 $n_{e}=128$ 个路由专家，每个 token 激活其中 $n_{a}=8$ 个，并为单个共享专家分配 $n_{s}=4$ 个专家宽度单位。设置 $G=8$ 得到 $d_{\mathrm{expert}}=d_{\mathrm{model}}/4$，因此 $d_{\mathrm{share}}=4d_{\mathrm{expert}}=d_{\mathrm{model}}$。这种离散配置产生 $A=(8+4)/(128+4)=9.09\%$ 和 $S=4/(8+4)=33.3\%$。

表 6：LLaDA MoE v2 30B-A3B 的架构。

<table><thead><tr><th colspan="2">参数规模</th><th colspan="4">Transformer 骨干</th><th colspan="3">MoE 配置</th></tr></thead><tbody><tr><th>总计</th><th>激活</th><td><math><semantics><msub><mi>n</mi> <mi>layer</mi></msub> <annotation>n_{\mathrm{layer}}</annotation></semantics></math></td><td><math><semantics><msub><mi>d</mi> <mi>model</mi></msub> <annotation>d_{\mathrm{model}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>heads</mi></msub> <annotation>n_{\mathrm{heads}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>kvheads</mi></msub> <annotation>n_{\mathrm{kvheads}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>e</mi></msub> <annotation>n_{e}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>a</mi></msub> <annotation>n_{a}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>s</mi></msub> <annotation>n_{s}</annotation></semantics></math></td></tr><tr><th>30.6B</th><th>3.4B</th><td>32</td><td>3072</td><td>32</td><td>4</td><td>128</td><td>8</td><td>4</td></tr></tbody></table>

### B.2 预训练

我们从头开始预训练 LLaDA MoE v2 30B-A3B，总共训练 23.5T 名义 token。预训练语料库由从网络收集的大量高质量文本构建而成。我们应用标准数据处理流程，收集原始文本，删除样板文件和格式错误或低质量的文档，对重复内容进行去重，并过滤有害材料。在整个预训练过程中，我们优化方程 1 中的去噪目标；路由过程和辅助目标遵循附录 A.2。

五阶段数据计划总结在表 7 中。阶段 1 和 2 从同一源语料库中抽取单独的 10T token 样本，阶段 2 混合为数学推理和代码数据分配略多的权重。对于阶段 3，我们在清洗、去重和有害内容过滤后构建了一个精选的 1T token 退火语料库，并对其训练两个 epoch，产生 2T 训练 token。阶段 4 和 5 主要使用序列长度达 32K token 的长文本数据。

我们使用 AdamW 以 BF16 精度进行训练，$(\beta_{1},\beta_{2})=(0.9,0.95)$ 和权重衰减为 0.1 [^49]。全局名义 token 批次大小为 33,554,432 个 token。每个阶段使用单独的学习率计划，在前 2,000 个优化器步骤中从零线性预热到其特定阶段的峰值。在阶段 1-4 分配训练计算的最后 10% 期间，学习率使用余弦计划衰减到下一阶段的峰值速率；在阶段 5 中，它使用相同的计划衰减到 $5.0\times 10^{-6}$。阶段 1-5 的峰值学习率分别为 $1.5\times 10^{-4}$、$1.0\times 10^{-4}$、$5.0\times 10^{-5}$、$1.0\times 10^{-5}$ 和 $7.0\times 10^{-6}$。在从阶段 3 到阶段 4 的过渡时，我们将 **RoPE base**（旋转位置编码的基础频率参数）从 10,000 增加到 500,000，以将上下文长度从 4K 扩展到 32K [^68] [^75]。完整的预训练运行消耗了约 460,000 NVIDIA B200 GPU 小时。

表 7：LLaDA MoE v2 30B-A3B 的五阶段预训练计划。

| 阶段 | 训练阶段 | Token 数 | 上下文长度 | RoPE base |
| --- | --- | --- | --- | --- |
| 1 | 基础预训练 1 | 10T | 4K | 10,000 |
| 2 | 基础预训练 2 | 10T | 4K | 10,000 |
| 3 | 预训练退火 | 2T | 4K | 10,000 |
| 4 | 上下文扩展 | 500B | 32K | 500,000 |
| 5 | 长上下文退火 | 1T | 32K | 500,000 |

### B.3 监督微调

从最终的预训练检查点开始，我们在 7M 指令-响应样例上对 LLaDA MoE v2 30B-A3B 进行三个 epoch 的微调，主要包括单轮数学推理和代码生成任务。我们遵循与预训练相同的一般过程处理数据，删除格式错误或低质量的样例，对重复内容进行去重，并过滤有害材料，然后使用统一的对话模板格式化每个样例。格式化的样例被打包成不重叠的 8K token 训练序列。

对于每个指令-响应对，我们连接提示和响应，但仅对响应 token 应用方程 1 中的前向掩码过程，将提示保留为未损坏的条件上下文，并仅在掩码响应位置计算去噪损失。附录 A.2 中描述的 MoE 负载平衡损失和路由器 $z$-损失在 SFT 期间保持活跃，其系数保持不变，分别为 $\alpha_{\mathrm{aux}}=0.01$ 和 $\alpha_{z}=0.001$。

我们使用 AdamW 更新所有模型参数，$(\beta_{1},\beta_{2})=(0.9,0.999)$，权重衰减为 0.1，梯度裁剪的最大范数为 1.0。全局批次大小为 512 个序列。学习率在前 8% 的训练步骤中线性预热到 $5.0\times 10^{-6}$，然后遵循余弦计划将其衰减到最小值 $1.0\times 10^{-6}$。我们使用最终检查点作为指令模型，在 SFT 后不应用强化学习阶段。

### B.4 评测

我们在涵盖通用任务（MMLU [^30]、MMLU-Pro [^72]、CEval [^34]、CMMLU [^42]、HellaSwag [^79]、KorBench [^52]）、数学推理（GSM8K [^18]、MATH [^31]、OlympiadBench [^28]）和代码生成（CRUXEval [^26]、MBPP [^6]、MultiPL-E [^11]、HumanEval [^13]、LiveCodeBench [^36]、BigCodeBench [^85]）的多样化基准套件上评测 LLaDA MoE v2。

对于表 3 中的基础模型结果，我们对多选任务使用条件似然，对其余任务使用条件生成，而表 4 中 SFT 评测的所有结果都是通过条件生成获得的。对于每个模型，我们优先报告其官方出版物的结果 [^77] [^15] [^84] [^78] [^57]；当基准结果不可用时，我们报告在统一评测配置下获得的分数。

我们在 MMLU、MMLU-Pro、CEval、CMMLU 和 HellaSwag 上使用条件似然。对于每个样例，我们计算给定提示的每个候选答案的条件似然，选择具有最高似然的候选，并报告基准上的准确率。对于 AR 模型，我们在其从左到右的因式分解下计算条件对数似然。对于 LLaDA MoE v2，我们遵循 SMDM、LLaDA 和先前 LLaDA MoE 中使用的似然评测协议 [^56] [^57] [^84]。对于 SDAR，我们使用 Block Diffusion 中引入的方法估计条件似然 [^4]。

对于条件生成任务，每个模型使用其原生生成过程从基准提示生成补全。对于代码任务，我们从响应中提取代码并针对基准测试用例执行它。对于数学推理任务，我们提取最终答案并使用等价检查器确定正确性。

所有指令模型都通过条件生成进行评测。我们在每个基准上最多允许 1,024 个生成 token。由于这个限制可能不足以让模型在 MATH、OlympiadBench、AIME 2024 和 AIME 2025 上产生最终答案，因此我们将这些基准的限制增加到 4,096 个 token。LLaDA MoE v2 使用**半自回归采样**（Semi-Autoregressive Sampling）**（每步生成固定数量的 token，介于完全自回归与完全并行之间的生成方式）** [^57]，块大小为 64，去噪步骤总数等于生成长度。对于 SDAR，我们遵循其推荐的块扩散解码过程，块大小为 4，同样将采样步骤总数设置为生成长度 [^15]。

表 8：计算分配扫描的模型架构。每行指定一个候选模型架构。

| 模型规模 | $n_{\mathrm{layer}}$ | $d_{\mathrm{model}}$ | $n_{\mathrm{heads}}$ | $n_{\mathrm{kvheads}}$ | $n_{e}$ | $n_{a}$ | $n_{s}$ | $d_{\mathrm{expert}}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60M | 6 | 128 | 8 | 2 | 64 | 4 | 1 | 128 |
| 63M | 7 | 128 | 8 | 2 | 64 | 4 | 1 | 128 |
| 66M | 8 | 128 | 8 | 2 | 64 | 4 | 1 | 128 |
| 69M | 9 | 128 | 8 | 2 | 64 | 4 | 1 | 128 |
| 103M | 6 | 192 | 8 | 2 | 64 | 4 | 1 | 192 |
| 111M | 7 | 192 | 8 | 2 | 64 | 4 | 1 | 192 |
| 119M | 8 | 192 | 8 | 2 | 64 | 4 | 1 | 192 |
| 184M | 8 | 256 | 8 | 2 | 64 | 4 | 1 | 256 |
| 242M | 7 | 320 | 8 | 2 | 64 | 4 | 1 | 320 |
| 296M | 6 | 384 | 8 | 2 | 64 | 4 | 1 | 384 |
| 325M | 7 | 384 | 8 | 2 | 64 | 4 | 1 | 384 |
| 354M | 8 | 384 | 8 | 2 | 64 | 4 | 1 | 384 |
| 472M | 6 | 512 | 8 | 2 | 64 | 4 | 1 | 512 |
| 574M | 6 | 576 | 8 | 2 | 64 | 4 | 1 | 576 |
| 575M | 8 | 512 | 8 | 2 | 64 | 4 | 1 | 512 |
| 627M | 9 | 512 | 8 | 2 | 64 | 4 | 1 | 512 |
| 731M | 11 | 512 | 8 | 2 | 64 | 4 | 1 | 512 |
| 1B | 10 | 640 | 10 | 2 | 64 | 4 | 1 | 640 |
| 1.1B | 11 | 640 | 10 | 2 | 64 | 4 | 1 | 640 |
| 1.4B | 10 | 768 | 12 | 4 | 64 | 4 | 1 | 768 |
| 1.6B | 12 | 768 | 12 | 4 | 64 | 4 | 1 | 768 |
| 1.8B | 14 | 768 | 12 | 4 | 64 | 4 | 1 | 768 |
| 2.8B | 12 | 1024 | 16 | 4 | 64 | 4 | 1 | 1024 |
| 3.2B | 14 | 1024 | 16 | 4 | 64 | 4 | 1 | 1024 |
| 7.5B | 22 | 1280 | 20 | 4 | 64 | 4 | 1 | 1280 |

表 9：计算分配扫描的分配配置。每行报告计算预算 $C$（以 FLOPs 为单位）、模型规模、训练 token 预算 $D$、全局名义 token 批次大小 $B$ 和峰值学习率 $\eta$。

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| $C$ | 模型规模 | $D\;(\times 10^{9})$ | $B$ | $\eta$ |
| $10^{17}$ | 60M | 2.07 | $2^{18}$ | $4.4\times 10^{-3}$ |
|  | 63M | 1.77 |  |  |
|  | 66M | 1.55 |  |  |
|  | 69M | 1.38 |  |  |
|  | 103M | 1.25 |  |  |
|  | 111M | 1.07 |  |  |
|  | 119M | 0.934 |  |  |
|  | 184M | 0.639 |  |  |
| $3\times 10^{17}$ | 63M | 5.32 | $2^{19}$ | $3.4\times 10^{-3}$ |
|  | 66M | 4.65 |  |  |
|  | 69M | 4.14 |  |  |
|  | 103M | 3.74 |  |  |
|  | 111M | 3.20 |  |  |
|  | 119M | 2.80 |  |  |
|  | 184M | 1.92 |  |  |
|  | 296M | 1.45 |  |  |
|  | 472M | 0.945 |  |  |
| $10^{18}$ | 119M | 9.34 | $2^{20}$ | $2.5\times 10^{-3}$ |
|  | 184M | 6.39 |  |  |
|  | 242M | 5.37 |  |  |
|  | 296M | 4.83 |  |  |
|  | 325M | 4.14 |  |  |
|  | 354M | 3.62 |  |  |
|  | 472M | 3.15 |  |  |
|  | 574M | 2.63 |  |  |
|  | 575M | 2.36 |  |  |
|  | 627M | 2.10 |  |  |
|  | 731M | 1.72 |  |  |
| $3\times 10^{18}$ | 184M | 19.2 | $2^{20}$ | $2.0\times 10^{-3}$ |
|  | 242M | 16.1 |  |  |
|  | 296M | 14.5 |  |  |
|  | 325M | 12.4 |  |  |
|  | 354M | 10.9 |  |  |
|  | 472M | 9.45 |  |  |
|  | 574M | 7.89 |  |  |
|  | 575M | 7.09 |  |  |
|  | 627M | 6.30 |  |  |
|  | 731M | 5.16 |  |  |
|  | 1B | 4.03 |  |  |
| $10^{19}$ | 296M | 48.3 | $2^{21}$ | $1.4\times 10^{-3}$ |
|  | 325M | 41.4 |  |  |
|  | 354M | 36.2 |  |  |
|  | 472M | 31.5 |  |  |
|  | 574M | 26.3 |  |  |
|  | 575M | 23.6 |  |  |
|  | 627M | 21.0 |  |  |
|  | 731M | 17.2 |  |  |
|  | 1B | 13.4 |  |  |
|  | 1.1B | 12.2 |  |  |
|  | 1.4B | 9.94 |  |  |
|  | 1.6B | 8.29 |  |  |
| $3\times 10^{19}$ | 472M | 94.5 | $2^{21}$ | $1.1\times 10^{-3}$ |
|  | 574M | 78.9 |  |  |
|  | 575M | 70.9 |  |  |
|  | 627M | 63.0 |  |  |
|  | 731M | 51.6 |  |  |
|  | 1B | 40.3 |  |  |
|  | 1.1B | 36.6 |  |  |
|  | 1.4B | 29.8 |  |  |
|  | 1.6B | 24.9 |  |  |
|  | 1.8B | 21.3 |  |  |
|  | 2.8B | 15.5 |  |  |
|  | 3.2B | 13.3 |  |  |
| $10^{20}$ | 731M | 172 | $2^{22}$ | $8.3\times 10^{-4}$ |
|  | 1B | 134 |  |  |
|  | 1.1B | 122 |  |  |
|  | 1.4B | 99.4 |  |  |
|  | 1.6B | 82.9 |  |  |
|  | 1.8B | 71.0 |  |  |
|  | 2.8B | 51.8 |  |  |
|  | 3.2B | 44.4 |  |  |
|  | 7.5B | 19.4 |  |  |

表 10：MoE 架构扫描的通用训练配置。在每个参考预算 $C$ 下，分配目标 $M^{*}(C)$ 和 $D^{*}(C)$ 从拟合的计算分配定律中获得。所有候选模型都训练 $3D^{*}(C)$ 个名义 token，对应约 $3C$ 训练 FLOPs，而 $B$ 和 $\eta$ 通过在 $3C$ 处评估拟合的超参数定律获得。

| $C$ | $M^{*}(C)$ | $3D^{*}(C)$ | $B$ | $\eta$ |
| --- | --- | --- | --- | --- |
| $6\times 10^{17}$ | $1.43\times 10^{8}$ | $1.26\times 10^{10}$ | $1048576$ | $2.2\times 10^{-3}$ |
| $2\times 10^{18}$ | $2.54\times 10^{8}$ | $2.36\times 10^{10}$ | $1310720$ | $1.7\times 10^{-3}$ |
| $6\times 10^{18}$ | $4.28\times 10^{8}$ | $4.20\times 10^{10}$ | $2097152$ | $1.3\times 10^{-3}$ |
| $2\times 10^{19}$ | $7.59\times 10^{8}$ | $7.91\times 10^{10}$ | $3145728$ | $9.4\times 10^{-4}$ |
| $6\times 10^{19}$ | $1.28\times 10^{9}$ | $1.41\times 10^{11}$ | $4194304$ | $7.0\times 10^{-4}$ |

表 11：激活比例扫描的配置。上半部分报告每个参考计算预算下的骨干和固定专家设置。下半部分列出跨预算共享的激活比例候选，每列给出一个对应的 $(n_{e},A)$ 对。

<table><thead><tr><th colspan="8">骨干和固定专家设置</th></tr></thead><tbody><tr><th>计算</th><td><math><semantics><msub><mi>n</mi> <mi>layer</mi></msub> <annotation>n_{\mathrm{layer}}</annotation></semantics></math></td><td><math><semantics><msub><mi>d</mi> <mi>model</mi></msub> <annotation>d_{\mathrm{model}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>heads</mi></msub> <annotation>n_{\mathrm{heads}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>kvheads</mi></msub> <annotation>n_{\mathrm{kvheads}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>a</mi></msub> <annotation>n_{a}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>s</mi></msub> <annotation>n_{s}</annotation></semantics></math></td><td><math><semantics><msub><mi>d</mi> <mi>expert</mi></msub> <annotation>d_{\mathrm{expert}}</annotation></semantics></math></td></tr><tr><th><math><semantics><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>17</mn></msup></mrow> <annotation>6\times 10^{17}</annotation></semantics></math></th><td>8</td><td>256</td><td>8</td><td>2</td><td>2</td><td>1</td><td>256</td></tr><tr><th><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mn>18</mn></msup></mrow> <annotation>2\times 10^{18}</annotation></semantics></math></th><td>8</td><td>448</td><td>8</td><td>2</td><td>2</td><td>1</td><td>448</td></tr><tr><th><math><semantics><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>18</mn></msup></mrow> <annotation>6\times 10^{18}</annotation></semantics></math></th><td>10</td><td>512</td><td>16</td><td>4</td><td>2</td><td>1</td><td>512</td></tr><tr><th><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mn>19</mn></msup></mrow> <annotation>2\times 10^{19}</annotation></semantics></math></th><td>12</td><td>640</td><td>16</td><td>4</td><td>2</td><td>1</td><td>640</td></tr><tr><th><math><semantics><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>19</mn></msup></mrow> <annotation>6\times 10^{19}</annotation></semantics></math></th><td>15</td><td>768</td><td>16</td><td>4</td><td>2</td><td>1</td><td>768</td></tr></tbody></table>

<table><thead><tr><th colspan="9">激活比例候选</th></tr></thead><tbody><tr><th>候选</th><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><th><math><semantics><msub><mi>n</mi> <mi>e</mi></msub> <annotation>n_{e}</annotation></semantics></math></th><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td><td>256</td></tr><tr><th><math><semantics><mi>A</mi> <annotation>A</annotation></semantics></math> (%)</th><td>100</td><td>60</td><td>33.3</td><td>17.6</td><td>9.1</td><td>4.6</td><td>2.3</td><td>1.2</td></tr></tbody></table>

表 12：专家粒度扫描的配置。上半部分报告每个参考计算预算下的 Transformer 骨干。在下半部分，每列给出一个候选的 $G$、$n_{e}$、$n_{a}$ 和 $n_{s}$，以及每个预算下相应的 $d_{\mathrm{expert}}$。

<table><thead><tr><th colspan="5">Transformer 骨干</th></tr></thead><tbody><tr><th>计算</th><td><math><semantics><msub><mi>n</mi> <mi>layer</mi></msub> <annotation>n_{\mathrm{layer}}</annotation></semantics></math></td><td><math><semantics><msub><mi>d</mi> <mi>model</mi></msub> <annotation>d_{\mathrm{model}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>heads</mi></msub> <annotation>n_{\mathrm{heads}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>kvheads</mi></msub> <annotation>n_{\mathrm{kvheads}}</annotation></semantics></math></td></tr><tr><th><math><semantics><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>17</mn></msup></mrow> <annotation>6\times 10^{17}</annotation></semantics></math></th><td>8</td><td>256</td><td>8</td><td>2</td></tr><tr><th><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mn>18</mn></msup></mrow> <annotation>2\times 10^{18}</annotation></semantics></math></th><td>8</td><td>448</td><td>8</td><td>2</td></tr><tr><th><math><semantics><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>18</mn></msup></mrow> <annotation>6\times 10^{18}</annotation></semantics></math></th><td>10</td><td>512</td><td>16</td><td>4</td></tr><tr><th><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mn>19</mn></msup></mrow> <annotation>2\times 10^{19}</annotation></semantics></math></th><td>12</td><td>640</td><td>16</td><td>4</td></tr><tr><th><math><semantics><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>19</mn></msup></mrow> <annotation>6\times 10^{19}</annotation></semantics></math></th><td>15</td><td>768</td><td>16</td><td>4</td></tr></tbody></table>

<table><thead><tr><th colspan="8">专家粒度候选</th></tr></thead><tbody><tr><th>候选</th><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><th><math><semantics><mi>G</mi> <annotation>G</annotation></semantics></math></th><td>2</td><td>4</td><td>6</td><td>8</td><td>12</td><td>16</td><td>20</td></tr><tr><th><math><semantics><msub><mi>n</mi> <mi>e</mi></msub> <annotation>n_{e}</annotation></semantics></math></th><td>64</td><td>128</td><td>192</td><td>256</td><td>384</td><td>512</td><td>640</td></tr><tr><th><math><semantics><msub><mi>n</mi> <mi>a</mi></msub> <annotation>n_{a}</annotation></semantics></math></th><td>2</td><td>4</td><td>6</td><td>8</td><td>12</td><td>16</td><td>20</td></tr><tr><th><math><semantics><msub><mi>n</mi> <mi>s</mi></msub> <annotation>n_{s}</annotation></semantics></math></th><td>1</td><td>2</td><td>3</td><td>4</td><td>6</td><td>8</td><td>10</td></tr><tr><th><math><semantics><mrow><msub><mi>d</mi> <mi>expert</mi></msub> <mo></mo><mrow><mo>(</mo><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>17</mn></msup></mrow><mo>)</mo></mrow></mrow> <annotation>d_{\mathrm{expert}}(6\times 10^{17})</annotation></semantics></math></th><td>256</td><td>128</td><td>85</td><td>64</td><td>43</td><td>32</td><td>26</td></tr><tr><th><math><semantics><mrow><msub><mi>d</mi> <mi>expert</mi></msub> <mo></mo><mrow><mo>(</mo><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mn>18</mn></msup></mrow><mo>)</mo></mrow></mrow> <annotation>d_{\mathrm{expert}}(2\times 10^{18})</annotation></semantics></math></th><td>448</td><td>224</td><td>149</td><td>112</td><td>75</td><td>56</td><td>45</td></tr><tr><th><math><semantics><mrow><msub><mi>d</mi> <mi>expert</mi></msub> <mo></mo><mrow><mo>(</mo><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>18</mn></msup></mrow><mo>)</mo></mrow></mrow> <annotation>d_{\mathrm{expert}}(6\times 10^{18})</annotation></semantics></math></th><td>512</td><td>256</td><td>171</td><td>128</td><td>85</td><td>64</td><td>51</td></tr><tr><th><math><semantics><mrow><msub><mi>d</mi> <mi>expert</mi></msub> <mo></mo><mrow><mo>(</mo><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mn>19</mn></msup></mrow><mo>)</mo></mrow></mrow> <annotation>d_{\mathrm{expert}}(2\times 10^{19})</annotation></semantics></math></th><td>640</td><td>320</td><td>213</td><td>160</td><td>107</td><td>80</td><td>64</td></tr><tr><th><math><semantics><mrow><msub><mi>d</mi> <mi>expert</mi></msub> <mo></mo><mrow><mo>(</mo><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>19</mn></msup></mrow><mo>)</mo></mrow></mrow> <annotation>d_{\mathrm{expert}}(6\times 10^{19})</annotation></semantics></math></th><td>768</td><td>384</td><td>256</td><td>192</td><td>128</td><td>96</td><td>77</td></tr></tbody></table>

表 13：共享专家比例扫描的配置。上半部分报告每个参考计算预算下的骨干和固定专家设置。下半部分列出跨预算共享的共享专家比例候选，每列给出一个对应的 $(n_{a},n_{s},S)$ 三元组。

<table><thead><tr><th colspan="7">骨干和固定专家设置</th></tr></thead><tbody><tr><th>计算</th><td><math><semantics><msub><mi>n</mi> <mi>layer</mi></msub> <annotation>n_{\mathrm{layer}}</annotation></semantics></math></td><td><math><semantics><msub><mi>d</mi> <mi>model</mi></msub> <annotation>d_{\mathrm{model}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>heads</mi></msub> <annotation>n_{\mathrm{heads}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>kvheads</mi></msub> <annotation>n_{\mathrm{kvheads}}</annotation></semantics></math></td><td><math><semantics><msub><mi>n</mi> <mi>e</mi></msub> <annotation>n_{e}</annotation></semantics></math></td><td><math><semantics><msub><mi>d</mi> <mi>expert</mi></msub> <annotation>d_{\mathrm{expert}}</annotation></semantics></math></td></tr><tr><th><math><semantics><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>17</mn></msup></mrow> <annotation>6\times 10^{17}</annotation></semantics></math></th><td>8</td><td>256</td><td>8</td><td>2</td><td>256</td><td>64</td></tr><tr><th><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mn>18</mn></msup></mrow> <annotation>2\times 10^{18}</annotation></semantics></math></th><td>8</td><td>448</td><td>8</td><td>2</td><td>256</td><td>112</td></tr><tr><th><math><semantics><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>18</mn></msup></mrow> <annotation>6\times 10^{18}</annotation></semantics></math></th><td>10</td><td>512</td><td>16</td><td>4</td><td>256</td><td>128</td></tr><tr><th><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mn>19</mn></msup></mrow> <annotation>2\times 10^{19}</annotation></semantics></math></th><td>12</td><td>640</td><td>16</td><td>4</td><td>256</td><td>160</td></tr><tr><th><math><semantics><mrow><mn>6</mn> <mo>×</mo> <msup><mn>10</mn> <mn>19</mn></msup></mrow> <annotation>6\times 10^{19}</annotation></semantics></math></th><td>15</td><td>768</td><td>16</td><td>4</td><td>256</td><td>192</td></tr></tbody></table>

<table><thead><tr><th colspan="8">共享专家比例候选</th></tr></thead><tbody><tr><th>候选</th><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><th><math><semantics><msub><mi>n</mi> <mi>a</mi></msub> <annotation>n_{a}</annotation></semantics></math></th><td>12</td><td>11</td><td>10</td><td>8</td><td>6</td><td>4</td><td>2</td></tr><tr><th><math><semantics><msub><mi>n</mi> <mi>s</mi></msub> <annotation>n_{s}</annotation></semantics></math></th><td>0</td><td>1</td><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td></tr><tr><th><math><semantics><mi>S</mi> <annotation>S</annotation></semantics></math> (%)</th><td>0</td><td>8.3</td><td>16.7</td><td>33.3</td><td>50</td><td>66.7</td><td>83.3</td></tr></tbody></table>

---

**参考文献**（保留原文格式）

[^1]: S. Abnar, H. Shah, D. Busbridge, A. M. E. Ali, J. Susskind, and V. Thilak Parameters vs flops: scaling laws for optimal sparsity for mixture-of-experts language models. arXiv preprint arXiv:2501.12370. Cited by: §2.2, §2.2, §5.

[^2]: AIME AIME problems and solutions. External Links: [Link](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions) Cited by: §4.2.

[^3]: J. Ainslie, J. Lee-Thorp, M. De Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai Gqa: training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 conference on empirical methods in natural language processing, pp. 4895–4901. Cited by: §A.2.

[^4]: M. Arriola, A. Gokaslan, J. Chiu, Z. Yang, Z. Qi, J. Han, S. Sahoo, and V. Kuleshov Block diffusion: interpolating between autoregressive and diffusion language models. In International Conference on Learning Representations, Vol. 2025, pp. 50726–50753. Cited by: §B.4, §5.

[^5]: J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. Van Den Berg Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems 34, pp. 17981–17993. Cited by: §2.1, §5.

[^6]: J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732. Cited by: §B.4.
## References

（参考文献列表保留原文格式）

[^7]: X. Bi, D. Chen, G. Chen, S. Chen, D. Dai, C. Deng, H. Ding, K. Dong, Q. Du, Z. Fu, et al. Deepseek llm: scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954. Cited by: §A.3, §2.2, Figure 1, §3.1, §3.1, §3.2, Table 2.

[^8]: T. Bie, M. Cao, K. Chen, L. Du, M. Gong, Z. Gong, Y. Gu, J. Hu, Z. Huang, Z. Lan, et al. Llada2. 0: scaling up diffusion language models to 100b. arXiv preprint arXiv:2512.15745. Cited by: §1, §5.

[^9]: T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems 33, pp. 1877–1901. Cited by: §1.

[^10]: A. Campbell, J. Benton, V. De Bortoli, T. Rainforth, G. Deligiannidis, and A. Doucet A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems 35, pp. 28266–28279. Cited by: §2.1, §5.

[^11]: F. Cassano, J. Gouwar, D. Nguyen, S. Nguyen, L. Phipps-Costin, D. Pinckney, M. Yee, Y. Zi, C. J. Anderson, M. Q. Feldman, et al. Multipl-e: a scalable and extensible approach to benchmarking neural code generation. arXiv preprint arXiv:2208.08227. Cited by: §B.4.

[^12]: J. Chen, Y. Liang, and Z. Liu Dflash: block diffusion for flash speculative decoding. arXiv preprint arXiv:2602.06036. Cited by: §1, §5.

[^13]: M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374. Cited by: §B.4.

[^14]: T. Chen, R. Zhang, and G. Hinton Analog bits: generating discrete data using diffusion models with self-conditioning. arXiv preprint arXiv:2208.04202. Cited by: §5.

[^15]: S. Cheng, Y. Bian, D. Liu, Y. Jiang, Y. Liu, L. Zhang, Q. Yao, Z. Tian, W. Wang, Q. Guo, et al. Sdar: a synergistic diffusion-autoregression paradigm for scalable sequence generation. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 22058–22075. Cited by: §B.4, §B.4, §1, §4.1.

[^16]: X. Cheng, X. Yu, C. Shao, J. Li, Y. Xiong, Y. Qian, J. Zhu, S. Ma, X. Zhang, J. Ye, et al. DSpark: confidence-scheduled speculative decoding with semi-autoregressive generation. arXiv preprint arXiv:2607.05147. Cited by: §5.

[^17]: A. Clark, D. de Las Casas, A. Guy, A. Mensch, M. Paganini, J. Hoffmann, B. Damoc, B. Hechtman, T. Cai, S. Borgeaud, et al. Unified scaling laws for routed language models. In International conference on machine learning, pp. 4057–4086. Cited by: §5.

[^18]: K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168. Cited by: §B.4.

[^19]: D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu, et al. Deepseekmoe: towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 1280–1297. Cited by: §1, §2.2, §3.3, §5.

[^20]: N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat, et al. Glam: efficient scaling of language models with mixture-of-experts. In International conference on machine learning, pp. 5547–5569. Cited by: §1, §5.

[^21]: W. Fedus, B. Zoph, and N. Shazeer Switch transformers: scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research 23 (120), pp. 1–39. Cited by: §A.2, §1, §2.2, §5.

[^22]: S. Feng, Z. Chen, G. Fang, X. Ma, and X. Wang DMoE: dllms with learnable block experts. arXiv preprint arXiv:2605.30876. Cited by: §1, §5.

[^23]: S. Y. Gadre, G. Smyrnis, V. Shankar, S. Gururangan, M. Wortsman, R. Shao, J. Mercat, A. Fang, J. Li, S. Keh, et al. Language models scale reliably with over-training and on downstream tasks. In International Conference on Learning Representations, Vol. 2025, pp. 67661–67682. Cited by: §A.5.

[^24]: S. Gong, S. Agarwal, Y. Zhang, J. Ye, L. Zheng, M. Li, C. An, P. Zhao, W. Bi, J. Han, et al. Scaling diffusion language models via adaptation from autoregressive models. In International Conference on Learning Representations, Vol. 2025, pp. 5046–5073. Cited by: §5.

[^25]: A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783. Cited by: §A.5, Table 2.

[^26]: A. Gu, B. Rozière, H. Leather, A. Solar-Lezama, G. Synnaeve, and S. I. Wang Cruxeval: a benchmark for code reasoning, understanding and execution. arXiv preprint arXiv:2401.03065. Cited by: §B.4.

[^27]: I. Gulrajani and T. B. Hashimoto Likelihood-based diffusion language models. Advances in Neural Information Processing Systems 36, pp. 16693–16715. Cited by: §5.

[^28]: C. He, R. Luo, Y. Bai, S. Hu, Z. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, et al. Olympiadbench: a challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850. Cited by: §B.4.

[^29]: Z. He, T. Sun, Q. Tang, K. Wang, X. Huang, and X. Qiu Diffusionbert: improving generative masked language models with diffusion models. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 4521–4534. Cited by: §5.

[^30]: D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300. Cited by: §B.4.

[^31]: D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874. Cited by: §B.4.

[^32]: J. Hestness, S. Narang, N. Ardalani, G. Diamos, H. Jun, H. Kianinejad, M. M. A. Patwary, Y. Yang, and Y. Zhou Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409. Cited by: §2.2.

[^33]: J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556. Cited by: §1, §2.2, §3.2, Table 2, §5.

[^34]: Y. Huang, Y. Bai, Z. Zhu, J. Zhang, J. Zhang, T. Su, J. Liu, C. Lv, Y. Zhang, Y. Fu, et al. C-eval: a multi-level multi-discipline chinese evaluation suite for foundation models. Advances in neural information processing systems 36, pp. 62991–63010. Cited by: §B.4.

[^35]: R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton Adaptive mixtures of local experts. Neural computation 3 (1), pp. 79–87. Cited by: §2.2.

[^36]: N. Jain, A. Gu, W. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica Livecodebench: holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations, Vol. 2025, pp. 58791–58831. Cited by: §B.4.

[^37]: A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088. Cited by: §5.

[^38]: J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei Scaling laws for neural language models. arXiv preprint arXiv:2001.08361. Cited by: §1, §2.2, Table 2, §5.

[^39]: J. Krajewski, J. Ludziejewski, K. Adamczewski, M. Pióro, M. Krutul, S. Antoniak, K. Ciebiera, K. Król, T. Odrzygóźdź, P. Sankowski, et al. Scaling laws for fine-grained mixture of experts. arXiv preprint arXiv:2402.07871. Cited by: §2.2, §2.2.

[^40]: I. Labs, S. Khanna, S. Kharbanda, S. Li, H. Varma, E. Wang, S. Birnbaum, Z. Luo, Y. Miraoui, A. Palrecha, et al. Mercury: ultra-fast language models based on diffusion. arXiv preprint arXiv:2506.17298. Cited by: §5.

[^41]: D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen Gshard: scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668. Cited by: §1, §2.2, §5.

[^42]: H. Li, Y. Zhang, F. Koto, Y. Yang, H. Zhao, Y. Gong, N. Duan, and T. Baldwin Cmmlu: measuring massive multitask language understanding in chinese. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 11260–11285. Cited by: §B.4.

[^43]: H. Li, W. Zheng, Q. Wang, H. Zhang, Z. Wang, S. Xuyang, Y. Fan, Z. Ding, H. Wang, N. Ding, et al. Predictable scale: part i, step law–optimal hyperparameter scaling law in large language model pretraining. arXiv preprint arXiv:2503.04715. Cited by: §2.2, §3.1.

[^44]: J. Li, J. Guan, W. Wu, and C. Li Refusion: a diffusion large language model with parallel autoregressive decoding. In International Conference on Learning Representations, Vol. 2026, pp. 53846–53869. Cited by: §1.

[^45]: T. Li, M. Chen, B. Guo, and Z. Shen A survey on diffusion language models. arXiv preprint arXiv:2508.10875. Cited by: §5.

[^46]: A. Liu, B. Feng, B. Wang, B. Wang, B. Liu, C. Zhao, C. Dengr, C. Ruan, D. Dai, D. Guo, et al. Deepseek-v2: a strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434. Cited by: §1.

[^47]: A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437. Cited by: §1.

[^48]: J. Liu, J. Su, X. Yao, Z. Jiang, G. Lai, Y. Du, Y. Qin, W. Xu, E. Lu, J. Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982. Cited by: §A.2.

[^49]: I. Loshchilov and F. Hutter Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101. Cited by: §B.2.

[^50]: A. Lou, C. Meng, and S. Ermon Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834. Cited by: §1, §2.1, §5.

[^51]: J. Ludziejewski, M. Pióro, J. Krajewski, M. Stefaniak, M. Krutul, J. Małaśnicki, M. Cygan, P. Sankowski, K. Adamczewski, P. Miłoś, et al. Joint moe scaling laws: mixture of experts can be memory efficient. arXiv preprint arXiv:2502.05172. Cited by: §2.2, §3.2, §5.

[^52]: K. Ma, X. Du, Y. Wang, H. Zhang, Z. Wen, X. Qu, J. Yang, J. Liu, M. Liu, X. Yue, et al. Kor-bench: benchmarking language models on knowledge-orthogonal reasoning tasks. In International Conference on Learning Representations, Vol. 2025, pp. 80062–80161. Cited by: §B.4.

[^53]: J. Ni, Q. Liu, L. Dou, C. Du, Z. Wang, H. Yan, T. Pang, and M. Q. Shieh Diffusion language models are super data learners. arXiv preprint arXiv:2511.03276. Cited by: §5.

[^54]: J. Ni, Q. Liu, C. Du, L. Dou, H. Yan, Z. Wang, T. Pang, and M. Q. Shieh Training optimal large diffusion language models. arXiv preprint arXiv:2510.03280. Cited by: §1, Table 2, §5.

[^55]: S. Nie, Q. Min, S. Xu, Z. Huang, Y. Song, Y. Shan, Y. Lin, W. X. Zhao, C. Li, and J. Wen Improved large language diffusion models. arXiv preprint arXiv:2606.25331. Cited by: §1.

[^56]: S. Nie, F. Zhu, C. Du, T. Pang, Q. Liu, G. Zeng, M. Lin, and C. Li Scaling up masked diffusion models on text. In International Conference on Learning Representations, Vol. 2025, pp. 82974–82997. Cited by: §A.4, §B.4, §1, §1, Table 2, Table 2, §5.

[^57]: S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. Zhou, Y. Lin, J. Wen, and C. Li Large language diffusion models. Advances in Neural Information Processing Systems 38, pp. 50608–50646. Cited by: §B.4, §B.4, §B.4, §1, §5.

[^58]: J. Ou, S. Nie, K. Xue, F. Zhu, J. Sun, Z. Li, and C. Li Your absorbing discrete diffusion secretly models the conditional distributions of clean data. In International Conference on Learning Representations, Vol. 2025, pp. 64972–65009. Cited by: §1.

[^59]: L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems 35, pp. 27730–27744. Cited by: §1.

[^60]: A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al. Improving language understanding by generative pre-training. Cited by: §1.

[^61]: A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog 1 (8), pp. 9. Cited by: §1.

[^62]: S. S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. T. Chiu, A. Rush, and V. Kuleshov Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems 37, pp. 130136–130184. Cited by: §1.

[^63]: S. S. Sahoo, J. Lemercier, Z. Yang, J. Deschenaux, J. Liu, J. Thickstun, and A. Jukic Scaling beyond masked diffusion language models. arXiv preprint arXiv:2602.15014. Cited by: §1, §5.

[^64]: N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean Outrageously large neural networks: the sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538. Cited by: §A.2, §1, §2.2, §5.

[^65]: N. Shazeer Glu variants improve transformer. arXiv preprint arXiv:2002.05202. Cited by: §A.1.

[^66]: J. Shi, K. Han, Z. Wang, A. Doucet, and M. Titsias Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems 37, pp. 103131–103167. Cited by: §2.1, §5.

[^67]: Y. Song, Z. Zhang, C. Luo, P. Gao, F. Xia, H. Luo, Z. Li, Y. Yang, H. Yu, X. Qu, et al. Seed diffusion: a large-scale diffusion language model with high-speed inference. arXiv preprint arXiv:2508.02193. Cited by: §5.

[^68]: J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu Roformer: enhanced transformer with rotary position embedding. Neurocomputing 568, pp. 127063. Cited by: §B.2.

[^69]: C. Tian, K. Chen, J. Liu, Z. Liu, Z. Zhang, and J. Zhou Towards greater leverage: scaling laws for efficient mixture-of-experts language models. In International Conference on Learning Representations, Vol. 2026, pp. 29806–29843. Cited by: §A.5, §1, §2.2, §2.2, §3.3, Table 2, Table 2.

[^70]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin Attention is all you need. Advances in neural information processing systems 30. Cited by: §1, §2.1.

[^71]: D. von Rütte, J. Fluri, O. Pooladzandi, B. Schölkopf, T. Hofmann, and A. Orvieto Scaling behavior of discrete diffusion language models. In International Conference on Learning Representations, Vol. 2026, pp. 36638–36663. Cited by: §1, Table 2, §5.

[^72]: Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, et al. Mmlu-pro: a more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems 37, pp. 95266–95290. Cited by: §B.4.

[^73]: Q. Wei, Y. Zhang, Z. Liu, D. Liu, and L. Zhang Accelerating diffusion large language models with slowfast: the three golden principles. arXiv e-prints, pp. arXiv–2506. Cited by: §5.

[^74]: C. Wu, H. Zhang, S. Xue, Z. Liu, S. Diao, L. Zhu, P. Luo, S. Han, and E. Xie Fast-dllm: training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. In International Conference on Learning Representations, Vol. 2026, pp. 57027–57051. Cited by: §1.

[^75]: W. Xiong, J. Liu, I. Molybog, H. Zhang, P. Bhargava, R. Hou, L. Martin, R. Rungta, K. A. Sankararaman, B. Oguz, et al. Effective long-context scaling of foundation models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4643–4663. Cited by: §B.2.

[^76]: K. Xue, Y. Zhou, S. Nie, X. Min, X. Zhang, J. Zhou, and C. Li Unifying bayesian flow networks and diffusion models through stochastic differential equations. arXiv preprint arXiv:2404.15766. Cited by: §5.

[^77]: A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388. Cited by: §B.4, §1, §1, §3.3, §4.1, §4.2, Table 3.

[^78]: J. Ye, Z. Xie, L. Zheng, J. Gao, Z. Wu, X. Jiang, Z. Li, and L. Kong Dream 7b. External Links: [Link](https://hkunlp.github.io/blog/2025/dream) Cited by: §B.4, §1, §5.

[^79]: R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi Hellaswag: can a machine really finish your sentence?. In Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 4791–4800. Cited by: §B.4.

[^80]: S. Zhang, C. Zhuang, C. Cui, Z. Yang, F. Z. Peng, Y. Zhang, H. Bai, Z. Jia, Y. Zhou, G. Chen, et al. Expert-choice routing enables adaptive computation in diffusion language models. arXiv preprint arXiv:2604.01622. Cited by: §1.

[^81]: W. X. Zhao, K. Zhou, J. Li, T. Tang, Z. Dong, Y. Hou, B. Zhang, Y. Min, J. Zhang, P. Liu, et al. A survey of large language models. Frontiers of Computer Science 20 (12), pp. 2012627. Cited by: §1.

[^82]: K. Zheng, Y. Chen, H. Mao, M. Liu, J. Zhu, and Q. Zhang Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. In International Conference on Learning Representations, Vol. 2025, pp. 63186–63227. Cited by: §5.

[^83]: F. Zhu, R. Wang, S. Nie, X. Zhang, C. Wu, J. Zhou, Y. Lin, J. Wen, and C. Li Llada 1.5: variance-reduced preference optimization for large language diffusion models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11425–11460. Cited by: §5.

[^84]: F. Zhu, Z. You, Y. Xing, Z. Huang, L. Liu, Y. Zhuang, G. Lu, K. Wang, X. Wang, L. Wei, et al. LLaDA-moe: a sparse moe diffusion language model. arXiv preprint arXiv:2509.24389. Cited by: §B.4, §B.4, §1, §1, Table 3, §5, §5.

[^85]: T. Y. Zhuo, M. C. Vu, J. Chim, H. Hu, W. Yu, R. Widyasari, I. N. B. Yusuf, H. Zhan, J. He, I. Paul, et al. Bigcodebench: benchmarking code generation with diverse function calls and complex instructions. In International Conference on Learning Representations, Vol. 2025, pp. 66602–66656. Cited by: §B.4.

[^86]: B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, and W. Fedus St-moe: designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906. Cited by: §A.2, §2.2.
