---
sourceTitle: "MoNe: Modular Neural Memory for Efficient Long Context Inference"
sourceUrl: "https://arxiv.org/html/2608.17616v1"
title: "MoNe：面向高效长上下文推理的模块化神经记忆"
requestedUrl: "https://arxiv.org/html/2608.17616v1"
adapter: "generic"
capturedAt: "2026-08-19T12:53:44.771Z"
conversionMethod: "defuddle"
kind: "generic/article"
language: "zh"
pipelineRunId: "2026-08-19-mone"
pipelineSource: "translate/2026-08-19-mone/works-ready/arxiv-2608-17616-translation.md"
sourceFigureCount: 4
authors: "Wonguk Cho, Kyubyung Chae, Tribhuvanesh Orekondy, Sunghyun Park, Hyoungwoo Park, Jeongho Kim, Arash Behboodi, Kyuwoong Hwang, Sungrack Yun"
publishDate: "2026-08-18"
arxivId: "2608.17616"
categories: ["长上下文推理", "推理优化", "测试时学习", "模块化架构"]
---
---
# MoNe：面向高效长上下文推理的模块化神经记忆

Wonguk Cho Affiliation: 高通AI研究院. Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc    Kyubyung Chae Affiliation: 高通AI研究院. Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc    Tribhuvanesh Orekondy Affiliation: 高通AI研究院. Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc    Sunghyun Park Affiliation: 高通AI研究院. Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc    Hyoungwoo Park Affiliation: 高通AI研究院. Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc    Jeongho Kim Affiliation: 高通AI研究院. Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc    Arash Behboodi Affiliation: 高通AI研究院. Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc    Kyuwoong Hwang Affiliation: 高通AI研究院. Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc    Sungrack Yun Affiliation: 高通AI研究院. Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc 通信作者: [sungrack@qti.qualcomm.com](mailto:sungrack@qti.qualcomm.com)

###### 摘要

我们提出 MoNe，一种轻量级模块化神经记忆，可附加到任何冻结预训练 Transformer 上，实现无需重训练的长上下文推理。MoNe 将上下文划分为固定大小的分段并逐段读取，通过快权重神经记忆网络的测试时学习和层局部梯度更新进行处理；在推理阶段，记忆模块仅从查询 token 生成键和值，无需重新读取任何上下文 token。这种两阶段设计将推理成本与上下文长度解耦，实现 $O(N)$ 预处理和 $O(1)$ 查询成本，且峰值 GPU 显存不随 $N$ 增长。在 128K token 时，MoNe 相比上下文学习（In-Context Learning, ICL）减少约 80% 的计算和峰值 GPU 显存，仅增加 6.4% 的参数开销。MoNe 可泛化至远超骨干模型原生窗口的上下文长度，在 RULER 的大海捞针（needle-in-a-haystack）和词提取基准测试中取得优异性能，而 ICL 在超出原生窗口后性能急剧下降。

## 1 引言

长上下文处理已成为实用 AI 应用的核心需求：基于用户历史推理的个性化助手、对冗长合同或医疗记录的文档问答、以及需要综合数万 token 信息的智能体系统。主流方法是将完整上下文作为提示输入。这种方法的浮点运算次数（FLOPs）随上下文长度呈平方级增长（$O(N^{2})$）。在移动设备等资源受限硬件上，这种成本变得难以承受；对于适合在设备上运行的小型模型，问题更加复杂：即使提示符合上下文窗口，它们也难以可靠地从长提示中提取信息，尤其是任务需要跨多个关注 token 进行复杂推理时。

![Refer to caption](imgs/img-001-overview_mone_6.png)

Refer to caption

![Refer to caption](imgs/img-002-fig_cost_table_logscale.png)

Refer to caption

作为替代方案，检索增强生成（Retrieval-Augmented Generation, RAG）[^20] [^13] 通过在生成前选择相关片段部分解决了成本问题。然而，RAG 从根本上将推理限制为局部可恢复的事实。当答案需要整合多个分散的片段时，这些片段单独看并不显著，此时基于嵌入的检索会失效。另一方面，Mamba [^10]、TTT [^31] 和 Titans [^2] 等循环模型实现了线性复杂度的上下文处理，但它们需要从头设计和训练全新架构，无法与现有预训练 Transformer 无缝集成而不进行完整重训练。

我们提出 MoNe（Modular Neural Memory，模块化神经记忆），一种轻量级插件模块，可附加到任何冻结预训练 Transformer 上，实现高效长上下文推理而无需重训练骨干模型。如图 1 所示，MoNe 将上下文划分为固定大小的分段并顺序读取，通过层局部梯度更新执行**快权重**神经记忆网络的测试时学习——不修改任何骨干模型权重。在处理完所有上下文段后，推理时，已更新的神经记忆根据问题生成记忆 token；这些记忆 token 在冻结骨干模型的自注意力中充当键和值，使模型能够仅从问题本身回答，无需直接注意任何上下文 token。

具体而言，MoNe 的效率直接源于这种两阶段设计：在测试时学习期间，大小为 $T$ 的上下文段被顺序处理，总浮点运算次数为 $O(N)$；在推理阶段，查询仅注意其生成的记忆 token——而非任何上下文 token——因此推理成本不随 $N$ 增长，而 ICL 需要 $O(N^{2})$ 浮点运算次数。快权重占用的空间大小不随上下文长度变化：在 128K token 时，相比 ICL，这减少了约 80% 的计算和峰值 GPU 显存，仅增加 6.4% 的参数开销，如图 2 所示。最终的快权重状态可在多个查询间重复使用，并在新上下文到达时增量扩展，无需重新处理过去内容。MoNe 可泛化至 128K（32 倍外推，无需额外训练），在 RULER 的 S-NIAH、MK-NIAH 和高频词提取基准测试中表现优异，而 ICL 在超出模型原生上下文窗口后性能崩溃。

我们的贡献可总结如下：

- 我们提出一种模块化神经记忆架构，可插入任何预训练 Transformer 而无需修改骨干模型，仅增加 6.4% 的参数开销。
- 我们引入一种测试时学习过程，使用层局部梯度更新来适配每层快权重神经记忆网络，以 $O(N)$ 预处理和 $O(1)$ 查询成本编码长上下文——相比 ICL 的 $O(N^{2})$——并天然支持增量上下文扩展和多查询重用。
- 我们证明，在最长 4K token 上下文上训练的 MoNe 可泛化至 128K（32 倍外推），在 S-NIAH、MK-NIAH 和高频词提取任务中取得优异性能，而 ICL 在超出原生上下文窗口后性能崩溃，同时在 128K 时减少约 80% 的总浮点运算次数和峰值 GPU 显存。

## 2 相关工作

在本节中，我们讨论我们在实现高效长上下文理解方面的工作如何与先前文献相关。

#### 长上下文问题

基于长上下文推理仍是大语言模型（LLM）的核心挑战，部分原因是自注意力随序列长度呈平方级增长。在实践中，长上下文场景出现在生成依赖大型外部上下文时，例如检索增强生成（RAG）[^32] [^36] 中的文档，或依赖扩展推理轨迹时，例如数学推理中产生的推理链 [^11]。在本工作中，我们关注前一种场景，其中回答查询需要从长输入上下文中分析、理解和检索相关信息。

尽管上下文窗口越来越大，LLM 往往无法稳健地利用长上下文，遗漏文档中埋藏的事实或无法保留来自远距离位置的信息 [^21] [^15]。上下文退化（context rot）现象进一步凸显了这一局限性 [^14]，即输入长度增加会削弱模型使用上下文中已有相关信息的能力，导致系统性性能下降，即使保留了任务相关证据。先前工作采用长上下文微调 [^4] [^9]、上下文压缩 [^33] [^19] 和记忆增强架构或系统 [^5] [^38] [^2] 来改善长上下文性能。我们的工作属于记忆增强研究方向，但不同之处在于研究在推理时训练的轻量级神经记忆模型。

#### 检索增强生成（RAG）

RAG [^20] [^13] 在推理时检索相关文档，将 LLM 扩展至其上下文窗口之外 [^18] [^30]。在个性化场景中，基于 RAG 的方法选择用户行为历史中最相关的片段来增强提示 [^27] [^23]，在个性化基准测试中取得优异结果。尽管有效，RAG 在面向设备端个性化智能体 AI 时面临根本性限制。在每次查询时反复搜索大型本地用户上下文（例如冗长的聊天历史和应用使用日志）的计算开销在资源受限设备上难以承受。更关键的是，基于嵌入的检索对独立分块的片段进行操作，使其不适合需要识别和综合分布在完整用户上下文中的多个碎片化、相互依赖事实的长上下文推理任务 [^30] [^12]。

#### 基于神经记忆的测试时学习

测试时训练（Test-Time Training, TTT）[^31] 已成为长上下文建模的替代范式，其中一个小型子网络具有快速可适配的快权重，在推理时更新以压缩和存储过去上下文作为神经记忆 [^28] [^2] [^34] [^10]。这些快权重的功能类似于 RNN 中的循环状态，实现次二次序列建模以替代完整自注意力。然而，现有 TTT 方法存在硬件利用率低的问题（通常低于峰值 FLOPs 的 5%），原因是小批量更新大小较小。LaCT [^37] 通过采用极大的分块更新（2K–1M token）解决了这一问题，实现高达 70% 的 GPU 利用率，并支持显著更大的快权重状态大小。尽管取得这些进展，所有此类方法都需要从头训练新架构，使其难以与现有基于 Transformer 的预训练模型集成。

## 3 MoNe：面向预训练 Transformer 注意力的模块化神经记忆

### 3.1 问题设定

我们考虑一个预训练的 $L$ 层 Transformer，其隐藏维度为 $d$，权重完全冻结。给定包含 $N$ 个 token 的上下文 $\mathbf{C}=(t_{1},\ldots,t_{N})$ 和一个查询 $\mathbf{q}$，我们的目标是实现高效且有效的长上下文推理，而不修改任何骨干模型参数。我们将 $\mathbf{C}$ 划分为 $S$ 个不重叠的分段 $\mathbf{s}_{1},\ldots,\mathbf{s}_{S}$，每段包含 $T{=}512$ 个 token，其中 $s\in\{1,\ldots,S\}$，$s{=}0$ 表示处理任何分段前的初始状态。

在每一层 $l$，自注意力通过冻结权重矩阵 $\mathbf{W}_{q}$、$\mathbf{W}_{k}$、$\mathbf{W}_{v}$ 将分段隐藏状态 $\mathbf{X}^{(l)}\in\mathbb{R}^{T\times d}$ 投影为查询、键和值：

$$
\mathbf{Q}=\mathbf{W}_{q}\mathbf{X}^{(l)},\quad\mathbf{K}=\mathbf{W}_{k}\mathbf{X}^{(l)},\quad\mathbf{V}=\mathbf{W}_{v}\mathbf{X}^{(l)},
$$

并计算注意力输出为：

$$
\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_{h}}}\right)\mathbf{V},
$$

其中 $d_{h}$ 是每个注意力头的维度。处理包含 $N$ 个 token 的长上下文需要 $O(N^{2})$ 浮点运算次数，并需要随 $O(N)$ 增长的 KV 缓存，使得长上下文推理在资源受限设备上代价高昂。

![Refer to caption](imgs/img-003-layer_local_update_12.png)

Refer to caption

### 3.2 向骨干模型注意力注入记忆

MoNe 在每个解码器层 $l$ 附加一个快权重神经记忆（图 3）。遵循 [^37]，我们为每层 $l$ 的记忆模块使用 SwiGLU MLP，定义为 $\mathcal{M}(\cdot\,;\,\mathbf{W}^{(l)})$，快权重参数为 $\mathbf{W}^{(l)}=\{\mathbf{W}_{\mathrm{in}},\mathbf{W}_{\mathrm{gate}},\mathbf{W}_{\mathrm{out}}\}$：

$$
\mathcal{M}(\mathbf{x};\mathbf{W}^{(l)})\;=\;\mathbf{W}_{\mathrm{out}}^{\top}\!\Bigl(\mathrm{SiLU}\!\bigl(\mathbf{W}_{\mathrm{in}}\,\mathbf{x}\bigr)\odot\mathbf{W}_{\mathrm{gate}}\,\mathbf{x}\Bigr),
$$

其中三个矩阵每个快权重头均为 $\mathbb{R}^{d_{h}\times d_{h}}$，在消费上下文段时在线更新。使用非线性网络而非单一线性映射显著增加了存储记忆的表达能力，允许每层在相同参数预算内编码更丰富的关联结构。在生成时，通过 $\mathbf{W}_{q}$ 对查询 token 进行投影，并用于从缓存的快权重状态 $\mathbf{W}^{(l)}_{S}$（处理完所有 $S$ 个分段后的状态）中读取，生成每层的记忆 token：

$$
\mathbf{h}_{j}=\mathcal{M}\!\left(\mathbf{W}_{q}\,\mathbf{x}_{j};\;\mathbf{W}^{(l)}_{S}\right).
$$

原始输出在每个快权重头上使用 RMSNorm 归一化，并通过每个头学习的门控进行缩放，产生下文使用的记忆 token $\mathbf{h}_{j}$。这些记忆 token 使用相同的冻结骨干模型矩阵 $\mathbf{W}_{k}$ 和 $\mathbf{W}_{v}$ 投影到键值空间，并与从查询 token 派生的标准注意力条目拼接：

$$
\mathrm{out}=\mathrm{Attention}\!\left(\mathbf{Q},\;[\mathbf{K}\,\|\,\mathbf{W}_{k}\mathbf{h}],\;[\mathbf{V}\,\|\,\mathbf{W}_{v}\mathbf{h}]\right),
$$

其中 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 是式 (1) 中查询 token 的标准注意力投影。虽然 $\mathbf{W}_{q}$、$\mathbf{W}_{k}$ 和 $\mathbf{W}_{v}$ 从骨干模型直接共享而不复制，MoNe 仅向这些冻结权重添加小型低秩适配器（LoRA）[^16]。记忆键值对每层具有固定的 $T$ 个条目，无论 $N$ 如何，在测试时学习和推理期间都保持 KV 缓存占用恒定。

### 3.3 关联记忆损失与更新

在测试时学习期间，记忆模块 $\mathcal{M}$ 通过关联记忆损失进行优化。键值训练目标由 $\mathbf{W}_{k}$ 和 $\mathbf{W}_{v}$ 产生，它们是配备了元训练 LoRA 适配器的冻结骨干模型权重：

$$
\mathbf{k}_{s,j}^{(l)}=\mathbf{W}_{k}\,\mathbf{x}_{s,j}^{(l)},\quad\mathbf{v}_{s,j}^{(l)}=\mathbf{W}_{v}\,\mathbf{x}_{s,j}^{(l)},
$$

其中 $j\in\{1,\ldots,T\}$ 索引分段 $s$ 内的 token。在实践中，$\mathbf{k}_{s,j}^{(l)}$ 在投影后进一步通过 SiLU 激活和逐 token L2 归一化，$\mathbf{v}_{s,j}^{(l)}$ 通过 SiLU 激活；完整流程见附录 A。遵循 [^37]，我们将关联记忆损失定义为记忆输出与目标值之间的负内积：

$$
\ell_{s,j}^{(l)}\!\left(\mathbf{W}^{(l)}\right)\;=\;-\bigl(\mathbf{v}_{s,j}^{(l)}\bigr)^{\top}\,\mathcal{M}\!\left(\mathbf{k}_{s,j}^{(l)};\,\mathbf{W}^{(l)}\right),
$$

分段级平均为 $\mathcal{L}_{s}^{(l)}=\frac{1}{T}\sum_{j=1}^{T}\ell_{s,j}^{(l)}$。最小化 $\ell_{s,j}^{(l)}$ 通过最大化记忆输出与目标值之间的内积，直接将关联 $\mathbf{k}_{s,j}^{(l)}\!\to\!\mathbf{v}_{s,j}^{(l)}$ 印刻到快权重中，无需预测误差。

快权重在每个分段通过梯度步骤更新：

$$
\mathbf{W}^{(l)}_{s}\;=\;\mathbf{W}^{(l)}_{s-1}\;-\;\boldsymbol{\mu}_{s}^{(l)},
$$

其中 $\boldsymbol{\mu}_{s}^{(l)}$ 累积 $\ell_{s,j}^{(l)}$ 的梯度，使用逐 token 学习率和数据依赖的动量衰减（附录 A）。

位置信息通过段局部 RoPE 合并到 $\mathbf{k}_{s,j}^{(l)}$ 中：每个上下文 token 被分配位置 $p_{j}^{\mathrm{local}}{=}j\bmod T$，因此位置索引始终在 $[0,T)$ 范围内，无论 $N$ 如何。在推理时，查询 token 接收 $[0,Q)$ 范围内的位置，其中 $Q$ 是查询长度；由于 $Q\ll T$，这落在相同的 $[0,T)$ 范围内，无需位置插值，允许 MoNe 泛化至任意长上下文。LoRA 适配器和元参数（$\boldsymbol{\eta}^{(l)}$、动量投影、输出缩放）通过离线基于更新记忆 $\mathbf{W}^{(l)}_{S}$ 条件下答案 token 的生成损失进行训练，在测试时学习和推理期间保持冻结，使得快权重更新在部署时作为固定的预学习算法运行。

### 3.4 测试时学习与推理

在测试时学习期间，分段被顺序消费，每层 $l$ 的快权重按式 (8) 更新：

$$
\mathbf{W}^{(l)}_{0}\;\to\;\mathbf{W}^{(l)}_{1}\;\to\;\cdots\;\to\;\mathbf{W}^{(l)}_{S}.
$$

每个更新步骤在层 $l$ 局部计算 $\ell_{s,j}^{(l)}$ 的逐 token 梯度：

$$
\nabla_{\mathbf{W}^{(l)}_{s-1}}\ell_{s,j}^{(l)}\;=\;-\mathbf{v}_{s,j}^{(l)}\,\frac{\partial\,\mathcal{M}\!\left(\mathbf{k}_{s,j}^{(l)};\mathbf{W}^{(l)}_{s-1}\right)^{\top}}{\partial\mathbf{W}^{(l)}_{s-1}},
$$

不通过任何其他层传播梯度：

$$
\frac{\partial\ell_{s,j}^{(l)}}{\partial\mathbf{W}^{(l^{\prime})}_{s-1}}=\mathbf{0}\qquad\forall\,l^{\prime}\neq l.
$$

因此，每次更新完全由层 $l$ 自身的前向传播激活计算，使记忆更新既高效又真正模块化。相比之下，通过标准交叉熵损失更新 $\mathbf{W}^{(l)}_{s-1}$ 需要从输出 logits 通过所有后续层 $\mathbf{W}^{(L)}_{s-1},\ldots,\mathbf{W}^{(l+1)}_{s-1}$ 反向传播梯度才能到达层 $l$。

在推理期间，无需更新权重，直接按式 (4) 读取记忆，生成供给冻结骨干模型的记忆 token，如式 (5) 所示。

表 1：ICL、RAG 和 MoNe（我们的方法）在不同上下文长度下于 S-NIAH、MK-NIAH 和高频词提取任务上的性能。结果按输入长度是否在模型训练上下文长度（32K token）内分组，评估对超出训练分布序列长度的泛化能力。

在模型上下文长度内 超出模型上下文长度 任务 方法 4K 8K 16K 32K 48K 64K 96K 128K S-NIAH ICL 0.95 0.98 0.94 0.94 0.64 0.60 0.42 0.28 RAG 0.94 0.93 0.91 0.93 0.93 0.93 0.97 0.89 MoNe (ours) 1.00 1.00 0.99 1.00 1.00 0.99 0.99 0.96 MK-NIAH ICL 0.89 0.84 0.83 0.93 0.41 0.13 0.11 0.00 RAG 0.92 0.91 0.88 0.79 0.80 0.66 0.74 0.71 MoNe (ours) 1.00 0.99 0.99 0.99 0.99 0.98 0.97 0.94 高频词提取 ICL 0.59 0.57 0.42 0.41 0.29 0.31 0.28 0.23 RAG 0.58 0.61 0.60 0.61 0.61 0.59 0.60 0.60 MoNe (ours) 1.00 1.00 1.00 1.00 0.99 0.99 0.96 0.96

## 4 实验

在本节中，我们首先详细说明实验设置，然后呈现定量结果和详细的计算成本分析。

### 4.1 实验设置

#### 数据集

我们在 RULER 基准测试 [^15] 的三个任务上进行评估，每个任务都设计为需要在完整上下文上精确检索或聚合，而非表面模式匹配。我们使用 S-NIAH（单针大海捞针）、MK-NIAH（多键大海捞针）和 FWE（高频词提取）任务。

#### 评估指标

S-NIAH 和 MK-NIAH 使用子串精确匹配（Sub-EM）：真实值是否作为模型输出的子串出现 [^1]。对于 FWE，我们测量变量召回：模型输出中出现的三个目标词的比例。所有任务在 4K、8K、16K 和 32K token（骨干模型原生窗口内）以及 48K、64K、96K 和 128K token（超出原生窗口）的上下文长度下进行评估。进一步的训练和评估细节见附录 A。

#### 基线

我们使用 Qwen2.5-0.5B-Instruct [^25] 作为所有方法的冻结骨干模型。

上下文学习（ICL）将完整上下文直接作为提示输入，最多至骨干模型的原生 32K 上下文窗口；超出此限制后性能急剧下降，因为模型必须注意越来越多的 token。

检索增强生成（RAG）将长上下文视为外部记忆，可在查询时检索相关信息——这是一种自然基线，通过首先将上下文压缩为可检索索引来避免二次注意力成本 [^20] [^13] [^18]。我们将每个上下文分割为 128 个 token 的块，并使用 BGE-Large [^3] 计算嵌入；与查询余弦相似度最高的前 $K$ 个块被拼接作为提示。我们评估 $K\in\{1,4,8\}$，并报告每个上下文长度的最佳分数。

### 4.2 评估结果

表 1 报告了三种方法在三个任务和八个上下文长度下的结果，覆盖骨干模型原生 32K 窗口内和超出范围。结果呈现一致趋势：MoNe 在所有任务上保持近乎完美的性能，而 ICL 在上下文超出原生限制后崩溃，RAG 在需要整合分布在完整上下文中的信息的任务上达到平台期。

#### 在原生上下文窗口内（4K–32K）

MoNe 在所有任务和上下文长度下实现近乎完美的子串精确匹配（0.99–1.00），显著优于 ICL（0.41–0.98）和 RAG（0.58–0.93）。ICL 即使在原生窗口内的退化在高频词提取任务上最为明显，该任务需要注意所有 token，但骨干模型的有限容量被压垮；MoNe 的压缩表示更加稳健。

#### 超出原生上下文窗口（48K–128K）

一旦上下文超出 32K 窗口，ICL 急剧退化：在 S-NIAH 上从 48K 时的 0.64 降至 128K 时的 0.28，在 MK-NIAH 上从 0.41 降至 0.00。RAG 更稳定但在 S-NIAH 上达到低于 0.97 的平台期，在 MK-NIAH 上低于 0.80，因为基于嵌入的检索在处理分布在多个段落中的多跳事实时存在困难。MoNe 在 128K token 时在 S-NIAH、MK-NIAH 和高频词提取任务上分别保持 0.96、0.94 和 0.96 的匹配率，尽管仅在最长 4K token 的上下文上训练。这种无需额外训练或位置插值的上下文长度泛化能力由段局部 RoPE 编码（第 3.3 节）实现，无论总上下文长度如何，都将位置索引保持在 $[0,T)$ 范围内。

表 2：不同上下文长度下的计算成本比较。括号中的值表示相对 ICL 的减少率（${\downarrow}$ 越大越好）。

方法 峰值 GPU 使用 总浮点运算次数 上下文长度 = 32K ICL 2.48 GB 58.06 T MoNe (total) 1.41 GB (${\downarrow}43\%$) 38.10 T (${\downarrow}34\%$)     $\llcorner$ 测试时学习 1.41 GB 37.24 T     $\llcorner$ 推理 1.29 GB 0.86 T 上下文长度 = 128K ICL 7.07 GB 786.33 T MoNe (total) 1.41 GB (${\downarrow}80\%$) 149.61 T (${\downarrow}81\%$)     $\llcorner$ 测试时学习 1.41 GB 148.97 T     $\llcorner$ 推理 1.29 GB 0.64 T

![Refer to caption](imgs/img-004-fig_option_a.png)

Refer to caption

### 4.3 计算成本分析

表 2 比较了 ICL 和 MoNe 的端到端计算成本，包括测试时学习和推理。在 32K token 时，MoNe 将峰值 GPU 显存减少 43%（1.41 GB vs. 2.48 GB），总浮点运算次数减少 34%（38.10 T vs. 58.06 T）。优势在更长上下文下更加明显。在 128K token 时，ICL 仅推理就需要 7.07 GB 和 786.3 T 浮点运算次数，而 MoNe 总共仅需要 1.41 GB 和 149.61 T 浮点运算次数。

重要的是，MoNe 的峰值 GPU 显存与上下文长度 $N$ 无关。在测试时学习期间，快权重更新在固定大小的分段内局部计算（式 (8)），导致恒定的 1.41 GB。在推理时，查询仅注意固定大小的记忆 token，需要 1.29 GB 显存和 0.64 T 浮点运算次数。

### 4.4 性能-效率权衡

图 4 呈现了在 MK-NIAH 上的两个消融实验，检验层覆盖和分段大小各自如何贡献长上下文泛化。

#### 层选择

为此，我们消融了将 MoNe 附加到 24 个解码器层的不同子集的效果。图 4(a) 显示，将 MoNe 附加到所有 24 层在 4K / 64K / 128K 时实现 1.00 / 0.98 / 0.94 的子串精确匹配分数。将 MoNe 限制在最后 16 层（最后 $\tfrac{2}{3}$）保持短上下文性能（4K 时 1.00），但在更长上下文下退化（64K 时 0.88，128K 时 0.70）。进一步限制到最后 8 层（最后 $\tfrac{1}{3}$）导致显著崩溃，在 64K 时降至 0.15，在 128K 时降至 0.02。每增加 8 层组仅增加约 12.6M 参数（约占骨干模型大小的 2.1%），图 4(b) 显示浮点运算次数开销同样适度：在 128K 时，全 24 层产生 149.1T 浮点运算次数，而最后 16 层为 132.6T，最后 8 层为 116.1T——受限配置仅减少 11–22% 的浮点运算次数，但性能损失要大得多。

#### 分段大小

我们在相同上下文长度下比较三种分段大小 $T\in\{128,256,512\}$。增加分段大小持续改善长上下文性能：在 64K / 128K 时，$T{=}512$ 实现 0.98 / 0.94，优于 $T{=}256$（0.91 / 0.75）和 $T{=}128$（0.85 / 0.53）。尽管所有配置都在最长 4K token 的上下文上训练，较小分段在更长测试上下文下退化更严重。如图 4(d) 所示，较小分段仅提供边际的浮点运算次数节省；如图 4(c) 所示，这以长上下文性能的显著代价为代价。因此我们采用 $T{=}512$，它在测试配置中提供最强性能。

## 5 结论

MoNe 通过模块化神经记忆为冻结预训练 Transformer 增强长上下文能力，无需修改骨干模型权重。尽管在相对较短的序列上训练，MoNe 可泛化至远超原生窗口的上下文长度。在我们的实验中，MoNe 持续优于 ICL，后者在超出训练限制后遭受性能急剧退化。它也在需要综合分布信息的任务上超越 RAG。这些性能提升以高效率实现；在 128K token 时，MoNe 相比 ICL 减少约 80% 的浮点运算次数和峰值 GPU 显存。关键是，其显存占用无论上下文长度如何都保持恒定。最后，我们的消融研究表明，完整层覆盖配合 512 的分段大小优化了性能-效率权衡。因此 MoNe 对资源受限应用高度有效，例如设备端个性化 [^6] [^24] 和持续学习 [^7] [^8]。

#### 局限性与未来工作

当前实验使用 Qwen2.5-0.5B 骨干模型和 RULER 中的受控检索任务验证了 MoNe 的核心特性——长上下文泛化、恒定显存推理和模块化集成。将这些结果扩展至更大模型规模和自然长文档工作负载（例如单/多文档问答 [^35] [^26] 和真实世界对话历史 [^22] [^29]）是自然的下一步：MoNe 的插件设计无需架构更改即可适应更大骨干模型，恒定显存推理的效率优势预计在大规模下将更加明显，因为 KV 缓存成本否则极其高昂。进一步方向是为每个神经记忆应用不同的基于 LoRA 的方法 [^16] [^17] 以提高效率并支持多个记忆模块的灵活管理。

## 参考文献

## 附录 A 实验细节

### A.1 数据集

从 RULER 基准测试 [^15]，我们使用以下三个任务：

- S-NIAH（单针大海捞针）在填充文本段落中随机深度嵌入一个形容词-名词复合词的键值对；模型必须在给定键的情况下召回准确值。我们测量子串精确匹配（Sub-EM）：真实值是否作为模型输出的子串出现。
- MK-NIAH（多键大海捞针）在 Paul Graham 散文摘录中的分散位置植入四个形容词-名词复合词的键值对；模型被查询特定键的值，而其余对充当干扰项。在被查询值上报告子串精确匹配。
- 高频词提取（FWE）用从 Zipfian 分布（$\alpha{=}2.0$）采样的合成 6 字符词填充上下文，排名最高的词被噪声 token 替换；模型必须识别前 3 个最频繁的非噪声词。我们测量变量召回：模型输出中出现的三个目标词的比例。

使用 512 token 的分段大小，我们为每个 $N\in\{2,3,\ldots,8\}$ 生成 30K 个样本，覆盖从 1K 到 4K token 的上下文长度，用于 LoRA 适配器和元参数（$\boldsymbol{\eta}^{(l)}$、动量投影）的离线训练。对于评估，我们为每个 $N\in\{8,16,\ldots,256\}$ 生成 100 个样本，范围从 4K 到 128K 上下文长度。在测试时，上下文分段被顺序处理以更新神经记忆；然后模型仅使用最终记忆状态回答，无需直接访问上下文。

### A.2 实现细节

我们使用 Qwen2.5-0.5B-Instruct 作为冻结骨干模型，在每个解码器层附加一个 SwiGLU MLP（$H{=}4$ 个头，$d_{h}{=}224$）。遵循文献[^37]，键和查询经过仿射重缩放、SiLU 和逐 token L2 归一化，值仅经过 SiLU，输出在注入注意力前经 RMSNorm 归一化和 SiLU 门控。权重在每次更新后进行逐通道 L2 重归一化，梯度使用学习到的逐 token 学习率（softplus 激活）和数据依赖的动量衰减（sigmoid 激活）。数据依赖衰减 $\beta_{s}^{(l)}\in(0,1)$ 从每个分块的隐藏状态预测，允许基于内容选择性地重置动量。对于 LoRA 适配器，使用秩 128 和 $\alpha{=}128$。我们使用 AdamW 训练一个 epoch（$\beta_{1}{=}0.9$，$\beta_{2}{=}0.95$，权重衰减 $0.1$），批大小 16，学习率 $10^{-3}$ / $5{\times}10^{-5}$（非 LoRA/LoRA），200 步预热，余弦衰减至 $\eta_{\min}{=}10^{-5}$，梯度裁剪 1.0，bf16 精度。

[^1]: A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi Self-rag: learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, Vol. 2024, pp. 9112–9141. 引用于：§4.1.

[^2]: A. Behrouz, P. Zhong, and V. Mirrokni Titans: learning to memorize at test time. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 外部链接：[Link](https://openreview.net/forum?id=8GjSf9Rh7Z) 引用于：§1, §2, §2.

[^3]: J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu M3-embedding: multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024, L. Ku, A. Martins, and V. Srikumar (Eds.), Bangkok, Thailand, pp. 2318–2335. 外部链接：[Link](https://aclanthology.org/2024.findings-acl.137/), [Document](https://dx.doi.org/10.18653/v1/2024.findings-acl.137) 引用于：§4.1.

[^4]: Y. Chen, S. Qian, H. Tang, X. Lai, Z. Liu, S. Han, and J. Jia Longlora: efficient fine-tuning of long-context large language models. In International Conference on Learning Representations, Vol. 2024, pp. 8220–8238. 引用于：§2.

[^5]: P. Chhikara, D. Khant, S. Aryan, T. Singh, and D. Yadav Mem0: building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413. 引用于：§2.

[^6]: W. Cho, S. Choi, D. Das, M. Reisser, T. Kim, S. Yun, and F. Porikli Hollowed net for on-device personalization of text-to-image diffusion models. Advances in Neural Information Processing Systems 37, pp. 43058–43079. 引用于：§5.

[^7]: W. Cho, J. Park, and T. Kim Complementary domain adaptation and generalization for unsupervised continual domain shift learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11442–11452. 引用于：§5.

[^8]: V. Dorovatas, M. Schwerin, A. D. Bagdanov, L. Caccia, A. Carta, L. Charlin, B. Hammer, T. L. Hayes, T. Hess, C. Kanan, et al. Modular memory is the key to continual learning agents. arXiv preprint arXiv:2603.01761. 引用于：§5.

[^9]: T. Gao, A. Wettig, H. Yen, and D. Chen How to train long-context language models (effectively). In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7376–7399. 引用于：§2.

[^10]: A. Gu and T. Dao Mamba: linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 外部链接：[Link](https://openreview.net/forum?id=tEYskw1VY2) 引用于：§1, §2.

[^11]: D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al. Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. 引用于：§2.

[^12]: A. Gupta, A. Shirgaonkar, A. d. L. Balaguer, B. Silva, D. Holstein, D. Li, J. Marsman, L. O. Nunes, M. Rouzbahman, M. Sharp, et al. RAG vs fine-tuning: pipelines, tradeoffs, and a case study on agriculture. arXiv preprint arXiv:2401.08406. 引用于：§2.

[^13]: K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang REALM: retrieval-augmented language model pre-training. In Proceedings of the 37th International Conference on Machine Learning, ICML'20. 引用于：§1, §2, §4.1.

[^14]: K. Hong, A. Troynikov, and J. Huber Context rot: how increasing input tokens impacts llm performance. URL https://research./ trychroma. com/context-rot, 访问于 2025 年 10 月 20 日，pp. 2025. 引用于：§2.

[^15]: C. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, and B. Ginsburg RULER: what's the real context size of your long-context language models?. In First Conference on Language Modeling, 外部链接：[Link](https://openreview.net/forum?id=kIoBbc76Sy) 引用于：§A.1, §2, §4.1.

[^16]: E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen LoRA: low-rank adaptation of large language models. In International Conference on Learning Representations, 外部链接：[Link](https://openreview.net/forum?id=nZeVKeeFYf9) 引用于：§3.2, §5.

[^17]: J. Hwang, W. Cho, and T. Kim PiCa: parameter-efficient fine-tuning with column space projection. arXiv preprint arXiv:2505.20211. 引用于：§5.

[^18]: G. Izacard and E. Grave Leveraging passage retrieval with generative models for open domain question answering. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, P. Merlo, J. Tiedemann, and R. Tsarfaty (Eds.), Online, pp. 874–880. 外部链接：[Link](https://aclanthology.org/2021.eacl-main.74/), [Document](https://dx.doi.org/10.18653/v1/2021.eacl-main.74) 引用于：§2, §4.1.

[^19]: V. Kontonis, Y. Zeng, S. Garg, L. Chen, H. Tang, Z. Wang, A. Awadallah, E. Horvitz, J. Langford, and D. Papailiopoulos MEMENTO: teaching llms to manage their own context. arXiv preprint arXiv:2604.09852. 引用于：§2.

[^20]: P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela Retrieval-augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33, pp. 9459–9474. 外部链接：[Link](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) 引用于：§1, §2, §4.1.

[^21]: N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang Lost in the middle: how language models use long contexts. Transactions of the Association for Computational Linguistics 12, pp. 157–173. 外部链接：[Link](https://aclanthology.org/2024.tacl-1.9/), [Document](https://dx.doi.org/10.1162/tacl%5Fa%5F00638) 引用于：§2.

[^22]: A. Maharana, D. Lee, S. Tulyakov, M. Bansal, F. Barbieri, and Y. Fang Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13851–13870. 引用于：§5.

[^23]: S. Mysore, Z. Lu, M. Wan, L. Yang, B. Sarrafzadeh, S. Menezes, T. Baghaee, E. B. Gonzalez, J. Neville, and T. Safavi Pearl: personalizing large language model writing assistants with generation-calibrated retrievers. In Proceedings of the 1st Workshop on Customizable NLP: Progress and Challenges in Customizing NLP for a Domain, Application, Group, or Individual (CustomNLP4U), S. Kumar, V. Balachandran, C. Y. Park, W. Shi, S. A. Hayati, Y. Tsvetkov, N. Smith, H. Hajishirzi, D. Kang, and D. Jurgens (Eds.), Miami, Florida, USA, pp. 198–219. 外部链接：[Link](https://aclanthology.org/2024.customnlp4u-1.16/), [Document](https://dx.doi.org/10.18653/v1/2024.customnlp4u-1.16) 引用于：§2.

[^24]: S. Park, J. Kim, H. Park, D. Das, S. Yun, M. Hayat, J. Choo, F. Porikli, and S. Choi Memory-efficient fine-tuning diffusion transformers via dynamic patch sampling and block skipping. arXiv preprint arXiv:2603.20755. 引用于：§5.

[^25]: Qwen,:, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu Qwen2.5 technical report. 外部链接：2412.15115, [Link](https://arxiv.org/abs/2412.15115) 引用于：§4.1.

[^26]: P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang Squad: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 conference on empirical methods in natural language processing, pp. 2383–2392. 引用于：§5.

[^27]: A. Salemi, S. Mysore, M. Bendersky, and H. Zamani LaMP: when large language models meet personalization. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L. Ku, A. Martins, and V. Srikumar (Eds.), Bangkok, Thailand, pp. 7370–7392. 外部链接：[Link](https://aclanthology.org/2024.acl-long.399/), [Document](https://dx.doi.org/10.18653/v1/2024.acl-long.399) 引用于：§2.

[^28]: J. Schmidhuber Learning to control fast-weight memories: an alternative to dynamic recurrent networks. Neural Comput. 4 (1), pp. 131–139. 外部链接：ISSN 0899-7667, [Link](https://doi.org/10.1162/neco.1992.4.1.131), [Document](https://dx.doi.org/10.1162/neco.1992.4.1.131) 引用于：§2.

[^29]: U. Shaham, M. Ivgi, A. Efrat, J. Berant, and O. Levy ZeroSCROLLS: a zero-shot benchmark for long text understanding. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 7977–7989. 引用于：§5.

[^30]: F. Shi, X. Chen, K. Misra, N. Scales, D. Dohan, E. Chi, N. Schärli, and D. Zhou Large language models can be easily distracted by irrelevant context. In Proceedings of the 40th International Conference on Machine Learning, ICML'23. 引用于：§2.

[^31]: Y. Sun, X. Li, K. Dalal, J. Xu, A. Vikram, G. Zhang, Y. Dubois, X. Chen, X. Wang, S. Koyejo, T. Hashimoto, and C. Guestrin Learning to (learn at test time): RNNs with expressive hidden states. In Forty-second International Conference on Machine Learning, 外部链接：[Link](https://openreview.net/forum?id=wXfuOj9C7L) 引用于：§1, §2.

[^32]: K. Vodrahalli, S. Ontanon, N. Tripuraneni, K. Xu, S. Jain, R. Shivanna, J. Hui, N. Dikkala, M. Kazemi, B. Fatemi, et al. Michelangelo: long context evaluations beyond haystacks via latent structure queries. arXiv preprint arXiv:2409.12640. 引用于：§2.

[^33]: C. Yang, N. Srebro, D. McAllester, and Z. Li Pencil: long thoughts with short memory. arXiv preprint arXiv:2503.14337. 引用于：§2.

[^34]: S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim Gated linear attention transformers with hardware-efficient training. In Proceedings of the 41st International Conference on Machine Learning, ICML'24. 引用于：§2.

[^35]: Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhutdinov, and C. D. Manning HotpotQA: a dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pp. 2369–2380. 引用于：§5.

[^36]: H. Yen, T. Gao, M. Hou, K. Ding, D. Fleischer, P. Izsak, M. Wasserblat, and D. Chen Helmet: how to evaluate long-context language models effectively and thoroughly. ICLR. 引用于：§2.

[^37]: T. Zhang, S. Bi, Y. Hong, K. Zhang, F. Luan, S. Yang, K. Sunkavalli, W. T. Freeman, and H. Tan Test-time training done right. In The Fourteenth International Conference on Learning Representations, 外部链接：[Link](https://openreview.net/forum?id=Tb9qAxT3xv) 引用于：§A.2, §2, §3.2, §3.3.

[^38]: Z. Zhou, A. Qu, Z. Wu, S. Kim, A. Prakash, D. Rus, J. Zhao, B. K. H. Low, and P. P. Liang Mem1: learning to synergize memory and reasoning for efficient long-horizon agents. arXiv preprint arXiv:2506.15841. 引用于：§2.
