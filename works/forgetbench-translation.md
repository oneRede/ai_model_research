---
sourceUrl: "https://arxiv.org/html/2607.26455v1"
sourceTitle: "ForgetBench: Benchmarking Forgetting Dynamics of Long-Term Parametric Memory in Language Models"
title: "ForgetBench: 大语言模型长期参数记忆遗忘动态基准测试"
authors: "Ruxi Gu, Zhenliang Zhang, Wei Wang"
date: "2026-07-29"
arxivId: "2607.26455"
pipelineRunId: "20260731-090836"
pipelineSource: "translate/20260731-090836/works-ready/forgetbench-translation.md"
sourceFigureCount: 4
language: "zh-CN"
translationQuality: "refined"
capturedAt: "2026-07-31T01:09:27.216Z"
---

# ForgetBench: 大语言模型长期参数记忆遗忘动态基准测试

Ruxi Gu <sup>1,2</sup>, Zhenliang Zhang <sup>2</sup>, Wei Wang <sup>3,2</sup>

###### 摘要

大语言模型（Large Language Models, LLMs）在知识获取和推理方面展现出强大能力，但其在重复更新下保留先前获取知识的能力仍未得到充分理解。现有评估范式主要关注单步推理或静态知识编辑，这些方法无法捕捉知识保留和退化在持续模型修改中的时序动态（temporal dynamics）。在本研究中，我们提出 ForgetBench，这是一个旨在系统性刻画大语言模型在持续知识编辑下遗忘行为的基准测试。ForgetBench 提出两种互补的评估范式，即概念型问答（concept-based QA）和场景型问答（scenario-based QA），以解耦孤立事实保留与结构化关系知识保存。基于顺序编辑框架（sequential editing framework），我们构建时序有序的知识流并评估模型在多个编辑阶段的行为。为定量分析长期保留动态，我们进一步引入统一评估框架，对知识随时间的演化建模，实现对时序衰减、保留强度和跨实例稳定性的测量。跨多种模型和编辑方法的广泛实验表明，现有方法无法在长期保留和泛化质量之间取得平衡。我们的发现凸显了未来大语言模型需要更鲁棒的记忆机制，以有效获取、更新和随时间保存知识。代码将在论文接收后发布。

**原文链接**: https://arxiv.org/abs/2607.26455

---

# ForgetBench: 大语言模型长期参数记忆遗忘动态基准测试

Ruxi Gu <sup>1,2</sup>, Zhenliang Zhang <sup>2</sup>, Wei Wang <sup>3,2</sup>

###### 摘要

大语言模型（Large Language Models, LLMs）在知识获取和推理方面展现出强大能力，但其在重复更新下保留先前获取知识的能力仍未得到充分理解。现有评估范式主要关注单步推理或静态知识编辑，这些方法无法捕捉知识保留和退化在持续模型修改中的时序动态（temporal dynamics）。在本研究中，我们提出 ForgetBench，这是一个旨在系统性刻画大语言模型在持续知识编辑下遗忘行为的基准测试。ForgetBench 提出两种互补的评估范式，即概念型问答（concept-based QA）和场景型问答（scenario-based QA），以解耦孤立事实保留与结构化关系知识保存。基于顺序编辑框架（sequential editing framework），我们构建时序有序的知识流并评估模型在多个编辑阶段的行为。为定量分析长期保留动态，我们进一步引入统一评估框架，对知识随时间的演化建模，实现对时序衰减、保留强度和跨实例稳定性的测量。跨多种模型和编辑方法的广泛实验表明，现有方法无法在长期保留和泛化质量之间取得平衡。我们的发现凸显了未来大语言模型需要更鲁棒的记忆机制，以有效获取、更新和随时间保存知识。代码将在论文接收后发布。

## 1 引言

智能系统需要在动态环境中运作，知识必须随时间持续获取、保留和更新，记忆因此至关重要。受记忆认知研究的启发，记忆通常被划分为工作记忆（working memory）和长期记忆（long-term memory）。工作记忆支持单次交互内的瞬时信息操作，而长期记忆则实现跨长时段的持久知识保留和渐进遗忘。对于大语言模型而言，理解这种长期记忆动态（long-term memory dynamics）日益重要，因为模型预期在部署后持续适应。

尽管大语言模型进展迅速，但大多数现有评估范式主要关注工作记忆能力，如长输入上的上下文推理或单次推理过程内的多跳问答，而非更新后的持久知识保留。近期研究开始探究交互系统中的长期记忆，包括多会话对话记忆和演化智能体记忆评估。然而，这些基准主要评估外部记忆（externally maintained memory），保留依赖于检索机制、记忆组织或交互历史。相比之下，我们的工作聚焦于参数记忆（parametric memory），知识被编码进模型参数，遗忘源于后续模型更新引起的累积干扰。这种设定对于持续适应尤为重要，模型通过参数修改反复获取新知识。

知识编辑（knowledge editing）近期作为一种有前景的范式出现，用于在无需完全重训练的情况下更新大语言模型中的事实知识。现有编辑基准如 zsRE 和 UnKE 通常评估单个编辑是否成功，通过包括编辑准确度、泛化能力（generalization）和局部性（locality）在内的指标。然而，这些评估将编辑视为独立操作，忽略了连续更新之间的时序交互。因此，它们无法回答可靠终身适应的基本问题：今天获取的知识能否在未来修改中存活。

为填补这一空白，我们提出 ForgetBench，一个用于评估持续知识编辑下大语言模型长期记忆动态的基准。不同于测量即时可编辑性的现有编辑评估，ForgetBench 通过顺序编辑流和重复评估研究已编辑知识的时序存活性。我们构建两种互补设定：(i) 概念型问答，隔离原子事实更新以对参数干扰（parametric interference）进行受控分析，以及 (ii) 场景型问答，通过多智能体交互图（interaction graph）引入结构化关系知识。通过跟踪模型在编辑步骤间的表现，ForgetBench 将知识保留刻画为遗忘曲线（Forgetting Curve），揭示记忆如何随时间被保留、退化或覆盖。

我们的主要贡献总结如下：(1) 我们引入 ForgetBench，一个统一基准，旨在评估持续知识编辑下大语言模型的长期参数遗忘动态（forgetting dynamics）。超越静态单步编辑评估，ForgetBench 实现对整个顺序更新过程中知识获取、保留和遗忘的时序分析。(2) 我们设计两种互补的问答构建范式，捕捉孤立事实更新和结构化关系知识。基于时序有序的知识流，我们进一步开发评估框架，量化保留强度、时序衰减和跨实例稳定性。(3) 跨多个大语言模型和编辑方法的广泛实验表明，现有编辑方法在维持长期事实保留和泛化能力方面面临挑战。我们的发现提供了对当前知识编辑范式时序局限性的系统分析，并激励未来研究朝向更鲁棒的记忆机制。

## 2 相关工作

### 2.1 工作记忆与长期记忆

认知科学中的记忆通常分为短期和长期组件。根据 Atkinson-Shiffrin 记忆模型，短期记忆支持持续处理期间的临时信息操作，而长期记忆实现跨长时段的持久存储、积累和知识检索。这一区分启发了近期研究探究语言模型能否在单次推理上下文之外维持信息的努力。

关于大语言模型记忆的早期研究主要关注工作记忆能力，包括长上下文推理、上下文工程（context engineering）、键值复用和压缩（key-value reuse and compression），以及链式思考推理（chain-of-thought reasoning）。这些方法改善了对上下文窗口中临时可用信息的利用，但它们未检验新获取的知识能否持久存储于模型参数中。类似地，检索增强生成（Retrieval-Augmented Generation, RAG）和外部记忆机制通过外部检索增强知识访问，其中记忆保留依赖于存储、检索或管理策略，而非内在参数更新。

近期基准将记忆评估扩展到长期交互场景。现有数据集如 HotpotQA、LongBench 和 LoCoMo 评估复杂上下文或对话历史上的推理，而更新的基准和研究进一步探究多会话和演化智能体环境中的记忆一致性。这些研究对大语言模型系统如何随时间检索和组织信息提供了重要见解。然而，它们的记忆演化主要由外部记忆模块、检索策略或交互历史控制，因此观察到的遗忘反映记忆访问或管理的失败，而非内部模型参数间的干扰。

遗忘曲线为刻画记忆退化提供另一视角。受 Ebbinghaus 遗忘曲线启发，Liu 等人引入遗忘曲线分析长上下文语言模型中的记忆化。然而，他们的公式根据上下文距离定义记忆年龄，测量输入序列内呈现信息的衰减。相比之下，ForgetBench 聚焦持续知识编辑下的参数记忆。ForgetBench 不评估序列级衰减或外部记忆，而是通过后续编辑操作的数量定义记忆年龄。通过构建时序有序的编辑流，它实现对超越单步编辑的长期稳定性、遗忘动态和累积参数干扰的分析。

### 2.2 知识编辑

知识编辑旨在无需完全重训练地修改大语言模型中的事实知识。基于因果干预（causal intervention）和直接权重修改的早期方法表明，事实关联可在模型参数内定位并显式更改。基于此见解，可扩展方法如 MEMIT 实现大规模多事实编辑，而 AlphaEdit 进一步将更新约束到零空间（null space）以减轻对无关知识的干扰。更近期工作探索两个主要方向：超网络方法（hypernetwork-based methods），学习辅助模型根据编辑请求预测参数更新，以及定位后编辑方法（locate-then-edit approaches），首先识别知识关键组件，然后通过优化应用针对性扰动。

评估通常在 zsRE、CounterFact 和 UnKE 等基准上进行，测量编辑成功、释义泛化和非目标知识保留。然而，这些基准本质上是静态的，将每次编辑视为独立操作，限制了它们评估长期记忆动态的能力。相比之下，我们的基准引入时序有序的编辑，实现对保留、干扰和遗忘曲线的直接测量。因此，现有编辑基准评估模型能否获取新知识，而 ForgetBench 进一步探究这种知识能否在跨长编辑时域的未来更新中存活。

## 3 方法论

### 3.1 ForgetBench 概述

如图 1 所示，ForgetBench 由两种互补的问答构建范式组成，即概念型问答和场景型问答，它们在知识粒度和结构组织上有所不同。

![参见说明](https://arxiv.org/html/2607.26455v1/x1.png)

图 1: ForgetBench 概述。我们构建两类问答序列。概念型问答 (a) 聚焦孤立概念，而场景型问答 (b) 评估结构化知识理解。随后，顺序编辑 (c) 使模型接触这些问答序列，并评估跨编辑阶段的遗忘动态。

概念型设定聚焦原子事实属性，其中每个主体与单一属性关联并随时间独立扰动。这产生完全解耦的实例，用于研究局部化知识更新。相比之下，场景型设定在交互图诱导的结构化关系知识上构建问答对。知识图谱首先通过模拟智能体-物品交互生成，然后从采样子图派生问答实例并转换为自然语言上下文，需要对互连实体的推理。基于这两种互补的构建方式，ForgetBench 在顺序编辑范式下评估模型，新知识随时间持续注入。通过定期测试模型在先前和新编辑实例上的表现，我们获得细粒度遗忘轨迹，反映即时适应和长期记忆退化。

### 3.2 数据生成

**概念型问答**。概念型设定在原子主体上构建独立问答实例。为每个主体分配一个属性，其编辑前值从预定义分布中采样，然后由独立采样的编辑后值替换，后者作为编辑目标。每个问答实例遵循固定属性查询模板 $Q$，编辑后属性 $A^{right}$ 作为真值答案，编辑前值 $A^{wrong}$ 表示被覆盖的记忆（例如，$Q$："Donald Rodriguez 多大？"，$A^{right}$："22"，$A^{wrong}$："79"）。由于所有主体独立采样，该设定消除实例间依赖性，为分析局部化知识更新提供受控环境。

**场景型问答**。如算法 1 所述，场景型设定在动态交互图诱导的结构化关系知识上构建问答对。我们初始化智能体集合 $\mathcal{A}$ 并通过多轮模拟演化其交互，活跃智能体参与生成实体和关系边的协作行为。该过程产生由智能体、物品及其交互组成的异构知识图谱 $\mathcal{G}$。

算法 1 场景型问答生成

输入：模拟轮数 $K$，子图大小 $L$，子图数量 $N$，每子图问题数 $t$
输出：问答序列 $\mathcal{D}$

初始化问答序列 $\mathcal{D}\leftarrow\emptyset$ 和长度 $T=N\times t$

// 阶段 1: 知识图谱生成

初始化智能体集 $\mathcal{A}$，物品集 $\mathcal{I}$，边集 $\mathcal{E}$

for $k=1$ to $K$ do

  通过添加和删除特定智能体更新 $\mathcal{A}$

  选择活跃智能体 $\mathcal{A}_{active}\subseteq\mathcal{A}$

  for all $a_{i}\in\mathcal{A}_{active}$ do

   形成协作组 $\mathcal{A}_{col}\subseteq\mathcal{A}$ 生成物品节点和智能体-物品交互

   更新 $\mathcal{I}$ 和 $\mathcal{E}$

  end for

end for

构建知识图谱 $\mathcal{G}=\{V=\mathcal{A}\cup\mathcal{I},E=\mathcal{E}\}$

// 阶段 2: 问答生成

初始化多选问答生成大语言模型 $\mathcal{F}$

从 $\mathcal{G}$ 采样 $N$ 个大小为 $L$ 的多样子图 $\{\mathcal{G}_{i}\}_{i=1}^{N}$

for all $\mathcal{G}_{i}$ do

  将 $\mathcal{G}_{i}$ 改写为自然语言上下文 $C_{i}$

  for $j=1$ to $t$ do

    $\{Q_{ij},A_{ij}^{right},A_{1},A_{2},A_{3}\}\leftarrow\mathcal{F}(C_{i}),$ 其中 $A_{1}$, $A_{2}$, $A_{3}$ 为错误答案

   采样干扰答案 $A_{ij}^{wrong}\sim\{A_{1},A_{2},A_{3}\}$

   将 $(Q_{ij},A_{ij}^{right},A_{ij}^{wrong})$ 添加到 $\mathcal{D}$

  end for

end for

return $\mathcal{D}$

图构建后，我们采样多个具有受控重叠的子图以维持结构多样性，并通过基于规则的改写将每个子图转换为自然语言上下文。以这些上下文为条件，使用大语言模型生成多选问答对，包含一个正确答案和若干从子图派生的合理干扰项。该公式将图结构上的关系推理转换为问答格式，实现对持续编辑下结构化知识理解和干扰的评估。

![参见说明](https://arxiv.org/html/2607.26455v1/x2.png)

图 2: 用于问答生成的示例子图。

如图 2 所示，查询 $Q$ 测试多跳关系推理："哪个用户关注了 BillGates 并转发了关于太空探索的帖子？A. BookLoverHQ B. FoodieFeest C. TravelBug101 D. CulinaryCrafts"。正确答案 $A^{right}$（A. BookLoverHQ）满足两条路径，而 $A^{wrong}$（D. CulinaryCrafts）作为干扰项，缺少到 BillGates 的关注链接。

## 4 遗忘量化

为全面评估不同时间视野上的长期记忆，我们不仅检查单个顺序编辑过程内的查询，还考察问答序列总长度的变化。具体而言，给定最大序列长度 $T$ 和评估间隔 $k$，我们构建序列长度集合 $\{t_{n}\}_{n=0}^{T/k}$，其中每个 $t_{n}$ 对应 $n\cdot k$ 编辑步骤。模型学习完整问答序列后，联合评估所有先前编辑的知识项以评估随时间的记忆保留。我们用 $\mathcal{Z}[t,i]$ 表示模型在第 $t$ 轮评估后对应第 $i$ 次编辑的查询的正确性，其中 $i\leq t\cdot k$。每个条目 $\mathcal{Z}[t,i]\in\{0,1\}$ 指示知识项是否被正确回忆。

图 3 展示了两种设定下的评估矩阵 $\mathcal{Z}$，揭示不同的时空结构。在概念型设定中，正确性在编辑和评估轮次上高度分散，反映独立引入的事实更新对干扰均匀脆弱。相比之下，场景型设定展现显著的垂直连贯性，正确性一旦建立往往持续并在后续轮次传播，表明交互间上下文依赖性诱导的更稳定预测轨迹。

总体而言，我们的评估框架通过 $\mathcal{Z}$ 的统一时空视角刻画持续知识编辑，其中水平轴对应记忆年龄（即遗忘动态，通过 $\Delta=tk-i$），垂直轴跟踪连续编辑轮次上的单个知识实例。基于此公式，我们构建一组互补指标，捕捉三个关键维度：(1) 点态编辑正确性，(2) 沿记忆年龄的时序衰减，以及 (3) 跨编辑轨迹的跨实例稳定性。这些指标共同提供对全局性能趋势和细粒度记忆演化动态的统一刻画。

![参见说明](https://arxiv.org/html/2607.26455v1/x3.png)

图 3: 概念型（上）和场景型（下）测试的评估矩阵 $\mathcal{Z}$。

### 4.1 点态编辑性能

我们在单个知识更新层面评估即时编辑行为，关注新引入的事实是否被正确学习以及先前获取的知识是否保持完整。编辑成功率（Edit Success, ES）测量新编辑的知识在插入时是否被正确获取。它定义为 $\mathcal{Z}$ 对角线条目的平均性能。保留率（Retention, Ret）评估先前编辑的知识随时间保留的程度。它定义为所有过往知识的平均性能。ES 和 Ret 计算如下：

$$
\text{ES}=\frac{k}{T}\sum_{t=1}^{T/k}\mathcal{Z}_{t,tk},\quad\text{Ret}=\frac{k}{T}\sum_{t=1}^{T/k}(\frac{1}{tk-1}\sum_{i=1}^{tk-1}\mathcal{Z}_{t,i}).
$$

ES 捕捉编辑操作的即时有效性，Ret 反映跨编辑步骤存储知识的整体稳定性，测量有多少先前学习的知识被保留。

### 4.2 时序衰减动态

为刻画长期记忆随时间的动态衰减，我们引入遗忘曲线以量化知识经历连续编辑操作后的平均保留。不同于仅计算总体平均准确度的 ES 或 Ret，遗忘曲线显式建模记忆年龄与知识保留之间的关系。对于任意记忆年龄 $\Delta$，我们考虑满足 $\Delta=tk-i$ 的所有知识状态，即所有经历了 $\Delta$ 次后续编辑的知识项。基于此公式，遗忘曲线定义为

$$
\mathcal{F}(\Delta)=\mathbb{E}[\mathcal{Z}[t,i]\mid tk-i=\Delta].
$$

$\mathcal{F}(\Delta)$ 产生跨编辑过程的记忆衰减轨迹。随着 $\Delta$ 增加，$\mathcal{F}(\Delta)$ 通常展现渐进下降，从而反映持续模型编辑下长期记忆的遗忘现象。

此外，我们基于遗忘曲线定义记忆半衰期（memory half-life），刻画记忆保留衰减到初始水平一半所需的编辑步骤数。给定遗忘曲线 $\mathcal{F}(\Delta)$，我们将初始保留定义为 $\mathcal{F}(0)$，半衰期阈值为 $\theta=\frac{1}{2}\mathcal{F}(0)$。记忆半衰期 $\Delta_{1/2}$ 定义为使 $\mathcal{F}(\Delta)\leq\theta$ 的最小 $\Delta$，即曲线与阈值的首次交叉点。

为提高对 $\mathcal{F}(\Delta)$ 局部波动的鲁棒性，我们首先应用轻量平滑预处理步骤以减少小尺度噪声。由于曲线在离散步骤评估，确切交叉点可能位于两个相邻样本之间。因此我们应用线性插值：若 $\mathcal{F}(\Delta_{k-1})>\theta$ 且 $\mathcal{F}(\Delta_{k})\leq\theta$，半衰期估计为

$$
\Delta_{1/2}=\Delta_{k-1}+\frac{\mathcal{F}(\Delta_{k-1})-\theta}{\mathcal{F}(\Delta_{k-1})-\mathcal{F}(\Delta_{k})}(\Delta_{k}-\Delta_{k-1}).
$$

同时，我们从全局保留行为和局部衰减动态两方面分析 $F(\Delta)$。我们定义遗忘曲线下面积（Area Under Forgetting Curve, AUF）测量知识在整个生命周期的平均保留水平，遗忘速度（Forgetting Speed）量化曲线的局部变化率：

$$
\text{AUF}=\mathbb{E}[\mathcal{F}(\Delta)],\quad\text{Speed}=\mathbb{E}[\mathcal{F}(\Delta)-\mathcal{F}(\Delta+1)].
$$

更大的 AUF 表明模型能够维持更高的整体保留率，从而反映更优的长期记忆质量。同时，更接近零的 Speed 是优选的，因为它暗示更稳定且近似单调的保留曲线，记忆性能的时序波动最小。

### 4.3 跨实例稳定性

虽然遗忘曲线捕捉记忆年龄 $\Delta$ 上的保留动态，但它未显式刻画每个知识轨迹跨连续编辑步骤的时序一致性。为补充此视角，我们引入时序一致性（Temporal Consistency, TC）指标以量化编辑过程中单个记忆状态的稳定性。具体而言，对于固定知识索引 $i$，我们考虑其时序序列 $\mathcal{Z}_{\cdot,i}=\{\mathcal{Z}[t,i]\}_{t\geq i/k}$，定义在观察到相应记忆状态的所有评估轮次上。我们首先定义每个知识轨迹的转换率为 $d_{i}=\frac{1}{|\mathcal{Z}_{\cdot,i}|-1}\sum_{t}|\mathcal{Z}[t+1,i]-\mathcal{Z}[t,i]|$。TC 分数计算如下：

$$
\text{TC}=\frac{\sum_{i}w_{i}(1-d_{i})}{\sum_{i}w_{i}},
$$

其中 $w_{i}=|\mathcal{Z}_{\cdot,i}|$ 考虑由 $\mathcal{Z}$ 三角结构诱导的异构时间覆盖。更高的 TC 表明单个知识状态在连续编辑轮次上经历更少转换，反映更稳定的记忆演化和持续编辑下更少的波动。

重要的是，TC 补充而非替代保留指标，因为时序稳定性和记忆保存代表不同属性。轨迹可能在目标知识丢失后仍保持稳定（高 TC）。因此，必须联合解释它们：保留测量保存知识的数量，而 TC 刻画演化平滑性。这种联合视角揭示编辑方法是否遭受突然记忆中断或展现稳定但退化的状态。

![参见说明](https://arxiv.org/html/2607.26455v1/x4.png)

图 4: 概念型和场景型测试的遗忘曲线。曲线通过分箱大小为 10 的均值聚合平滑以增强视觉清晰度。参数编辑展现非生物学的遗忘动态：概念型编辑常展现极端动态，从快速记忆崩溃到无衰减，而场景型编辑展现由共享上下文驱动的人工自我恢复。

### 4.4 泛化能力与流畅度

我们引入泛化能力（Generalization, Gen）和流畅度（Fluency, Flu）以评估持续记忆更新的质量和稳定性。具体而言，Gen 测量编辑的知识是否泛化到改写查询（改写与原始查询间的准确度差异），确保模型避免单纯死记硬背。Flu 通过计算有效词汇标记的比例监控语言生成稳定性，直接检测累积参数修改引起的文本退化或模型崩溃。

## 5 实验

### 5.1 实验设置

**基准详情**。在数据集生成过程中，概念型问答的配置文件从包含 2,500 个不同配置文件的名称池中随机采样。对于场景型问答生成，我们初始化 20 个智能体的种群，全部使用 Qwen2.5-32B 实例化。这些智能体参与超过 50 轮交互，产生包含 56,095 个节点和 96,082 条边的交互图。我们从交互图随机采样 2,500 个子图，每个包含约 20 个节点，最大成对重叠为 20%。然后将这些子图提供给专门的问答生成智能体进行问题合成。遵循此过程，我们构建了总计包含 6,431 个问题实例的基准。我们在补充材料中提供更多细节。

<table><thead><tr><th>模型</th><th>编辑方法</th><th>ES (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th>Ret (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th><math><semantics><msub><mi>Δ</mi> <mrow><mn>1</mn> <mo>/</mo> <mn>2</mn></mrow></msub> <annotation>\Delta_{1/2}</annotation></semantics></math> (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th>AUF (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th>Speed (<math><semantics><mrow><mo>→</mo> <mn>0</mn></mrow> <annotation>\rightarrow 0</annotation></semantics></math>)</th><th>TC (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th>Gen (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th>Flu (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th></tr></thead><tbody><tr><td rowspan="4">Llama-3 8B</td><td>MEMIT</td><td>0</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.001</td></tr><tr><td>WISE</td><td>0.02</td><td>0.036</td><td>52</td><td>0.035</td><td>4e-5</td><td>0.927</td><td>0.010</td><td>0.869</td></tr><tr><td>AlphaEdit</td><td>0.56</td><td>0.275</td><td>39</td><td>0.076</td><td>1e-3</td><td>0.868</td><td>-0.053</td><td>0.313</td></tr><tr><td>UltraEdit</td><td>0.06</td><td>0.032</td><td>14</td><td>0.025</td><td>1e-4</td><td>0.936</td><td>-0.008</td><td>0.809</td></tr><tr><td rowspan="4">Llama-3.1 8B</td><td>MEMIT</td><td>0</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.001</td></tr><tr><td>WISE</td><td>0.02</td><td>0.031</td><td>82</td><td>0.033</td><td>4e-5</td><td>0.924</td><td>-0.001</td><td>0.910</td></tr><tr><td>AlphaEdit</td><td>0.98</td><td>0.733</td><td>236</td><td>0.510</td><td>2e-3</td><td>0.717</td><td>-0.161</td><td>0.896</td></tr><tr><td>UltraEdit</td><td>0.06</td><td>0.023</td><td>15</td><td>0.019</td><td>1e-4</td><td>0.950</td><td>-0.003</td><td>0.923</td></tr><tr><td rowspan="4">Qwen-2.5 7B</td><td>MEMIT</td><td>0.20</td><td>0.059</td><td>5</td><td>0.005</td><td>4e-4</td><td>0.986</td><td>-0.042</td><td>0.054</td></tr><tr><td>WISE</td><td>0.10</td><td>0.037</td><td>25</td><td>0.036</td><td>2e-4</td><td>0.908</td><td>-0.003</td><td>0.700</td></tr><tr><td>AlphaEdit</td><td>0.92</td><td>0.885</td><td>452</td><td>0.804</td><td>1e-3</td><td>0.792</td><td>-0.559</td><td>0.885</td></tr><tr><td>UltraEdit</td><td>0.12</td><td>0.044</td><td>12</td><td>0.036</td><td>2e-4</td><td>0.917</td><td>-0.019</td><td>0.726</td></tr><tr><td rowspan="4">Deepseek-R1 7B</td><td>MEMIT</td><td>0.12</td><td>0.034</td><td>6</td><td>0.003</td><td>2e-4</td><td>0.988</td><td>-0.022</td><td>0.059</td></tr><tr><td>WISE</td><td>0.10</td><td>0.030</td><td>11</td><td>0.027</td><td>2e-4</td><td>0.933</td><td>-0.010</td><td>0.630</td></tr><tr><td>AlphaEdit</td><td>1.00</td><td>0.979</td><td>>500</td><td>0.974</td><td>0</td><td>0.962</td><td>-0.930</td><td>0.618</td></tr><tr><td>UltraEdit</td><td>0.10</td><td>0.066</td><td>95</td><td>0.054</td><td>2e-4</td><td>0.870</td><td>-0.043</td><td>0.698</td></tr></tbody></table>

表 1: ForgetBench 上的概念型评估结果。每个模型内每个指标的最佳结果用下划线标注，总体最佳结果用粗体突出显示。现有编辑方法遭受保留与泛化之间的严峻权衡（例如 AlphaEdit），迫使模型进行死记硬背而非关联知识整合。

**基线与评估协议**。我们选择四种代表性知识编辑方法，包括 MEMIT、WISE、AlphaEdit 和 UltraEdit。我们评估四个主流大语言模型的长期记忆能力，包括 Llama-3 8B/Llama-3.1 8B、Qwen-2.5 7B 和 DeepSeek-R1 7B，涵盖不同架构、预训练规模和推理能力。实现细节和超参数设置遵循 EasyEdit 提供的默认配置。我们采用提出的评估指标，包括基础统计测量，即测量知识获取和保留的 ES 和 Ret，以及与遗忘动态相关的指标，包括半衰期 $\Delta_{1/2}$、遗忘速度和 AUF。

初步实验表明，一旦序列长度在任一评估场景中超过 500，大多数语言模型和知识编辑方法在 ES 和 Ret 上均遭受实质性退化。因此，我们在所有实验中统一设置最大序列长度 $T$ 为 500。此外，我们设置序列间隔 $k$ 为 10，在避免过度计算成本的同时提供充分细粒度的遗忘行为评估。所有实验在 4 块 NVIDIA GeForce RTX 4090 GPU 上进行，每块具有 24 GB 内存。

### 5.2 实验结果

**概念型测试的评估结果**。表 1 报告了概念型评估结果，揭示了参数知识编辑中一个深刻的、系统性的保留-泛化权衡（retention-generalization trade-off）问题。尽管 AlphaEdit 在保留相关指标上始终表现最佳，但这些记忆保存收益是以灾难性的泛化能力（Gen）代价换来的。这种权衡在高容量模型中最为严重：在 DeepSeek-R1 上，AlphaEdit 的峰值保留率（Ret）为 0.979，直接对应着毁灭性的 Gen 分数 -0.930；在 Qwen-2.5 上，其强劲的 Ret（0.885）伴随着 -0.559 的 Gen。这种负相关揭示了零空间投影（null-space projection）的一个关键局限：施加严格的正交约束来保护历史参数免受干扰，严重限制了更新方向。结果，模型被迫进入一种死记硬背的状态，无法与其更广泛的关联结构整合，从而无法将编辑后的知识泛化到改述查询。这种僵化的划分与稳健的、关联性的人类记忆形成鲜明对比，后者能自然地平衡保留和泛化。

其他基线方法无法应对这种权衡，在权衡的两端都出现了崩溃。在没有显式干扰控制的情况下，MEMIT 遭受完全的参数遗忘（在 Llama-3 变体上保留率接近零），尽管由于静态、未更新的预测状态，它保持了较高的时序一致性（TC）。相反，WISE 充当了保守的基线；它通过路由机制保留了泛化能力（Gen $\approx$ 0）和生成流畅度（Flu），但未能将更新写入参数，导致长期保留能力薄弱。UltraEdit 同样产生最小的保留率，证明单步优化能力无法扩展到终身记忆稳定性。

图 4 上半部分的遗忘曲线证实了这些动态，同时揭示了与经典 Ebbinghaus 模型的明显偏离。虽然 Ebbinghaus 曲线的特征是平滑的单调指数衰减，但概念型 LLM 编辑表现出一种异常的、非生物学的二分法：MEMIT 遭受瞬时崩溃至零保留率，而 AlphaEdit 则保持平坦的、非衰减的轨迹。在不同主干模型中，Qwen-2.5 和 DeepSeek-R1 显示出比 Llama-3 更强的内在记忆鲁棒性，表明基础模型显著影响对参数干扰的抵抗力。

<table><thead><tr><th>模型</th><th>编辑方法</th><th>ES (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th>Ret (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th><math><semantics><msub><mi>Δ</mi> <mrow><mn>1</mn> <mo>/</mo> <mn>2</mn></mrow></msub> <annotation>\Delta_{1/2}</annotation></semantics></math> (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th>AUF (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th>Speed (<math><semantics><mrow><mo>→</mo> <mn>0</mn></mrow> <annotation>\rightarrow 0</annotation></semantics></math>)</th><th>TC (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th>Gen (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th><th>Flu (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</th></tr></thead><tbody><tr><td rowspan="4">Llama-3 8B</td><td>MEMIT</td><td>0</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.021</td></tr><tr><td>WISE</td><td>0.30</td><td>0.276</td><td>271</td><td>0.290</td><td>6e-4</td><td>0.749</td><td>-0.008</td><td>0.996</td></tr><tr><td>AlphaEdit</td><td>0.78</td><td>0.369</td><td>47</td><td>0.259</td><td>1e-3</td><td>0.821</td><td>-0.047</td><td>0.703</td></tr><tr><td>UltraEdit</td><td>0.92</td><td>0.557</td><td>202</td><td>0.553</td><td>1e-3</td><td>0.752</td><td>-0.014</td><td>0.539</td></tr><tr><td rowspan="4">Llama-3.1 8B</td><td>MEMIT</td><td>0</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.001</td></tr><tr><td>WISE</td><td>0.30</td><td>0.258</td><td>311</td><td>0.252</td><td>6e-4</td><td>0.690</td><td>-0.002</td><td>0.999</td></tr><tr><td>AlphaEdit</td><td>0.96</td><td>0.657</td><td>260</td><td>0.539</td><td>1e-3</td><td>0.932</td><td>-0.155</td><td>0.801</td></tr><tr><td>UltraEdit</td><td>0.76</td><td>0.511</td><td>398</td><td>0.554</td><td>1e-3</td><td>0.727</td><td>-0.014</td><td>0.882</td></tr><tr><td rowspan="4">Qwen-2.5 7B</td><td>MEMIT</td><td>0.14</td><td>0.110</td><td>26</td><td>0.012</td><td>2e-4</td><td>0.978</td><td>-0.002</td><td>0.065</td></tr><tr><td>WISE</td><td>0.26</td><td>0.322</td><td>399</td><td>0.326</td><td>5e-4</td><td>0.685</td><td>-0.001</td><td>0.978</td></tr><tr><td>AlphaEdit</td><td>0.96</td><td>0.691</td><td>315</td><td>0.576</td><td>-8e-5</td><td>0.917</td><td>-0.137</td><td>0.944</td></tr><tr><td>UltraEdit</td><td>0.74</td><td>0.594</td><td>458</td><td>0.611</td><td>-5e-4</td><td>0.816</td><td>-0.036</td><td>0.858</td></tr><tr><td rowspan="4">Deepseek-R1 7B</td><td>MEMIT</td><td>0.12</td><td>0.073</td><td>10</td><td>0.012</td><td>2e-4</td><td>0.968</td><td>-0.004</td><td>0.079</td></tr><tr><td>WISE</td><td>0.40</td><td>0.425</td><td>448</td><td>0.468</td><td>8e-4</td><td>0.769</td><td>0.010</td><td>0.925</td></tr><tr><td>AlphaEdit</td><td>0.94</td><td>0.784</td><td>368</td><td>0.687</td><td>-1e-4</td><td>0.953</td><td>-0.327</td><td>0.874</td></tr><tr><td>UltraEdit</td><td>0.90</td><td>0.755</td><td>403</td><td>0.673</td><td>1e-3</td><td>0.857</td><td>-0.022</td><td>0.483</td></tr></tbody></table>

表 2: ForgetBench 上的场景型评估结果。结构化环境提供了语义脚手架，显著缓解了保留-泛化权衡，并改善了所有方法的可观察召回率。

**场景型测试的评估结果**。表 2 报告了场景型设置下的结果。与概念型设置最显著的区别是，所有基线方法的编辑成功率（ES）和保留率（Ret）都系统性激增。例如，Llama-3 8B 上的 UltraEdit 从概念型测试中的近乎崩溃状态，在这里攀升至具有竞争力的性能。

我们的场景型测试表明，结构化环境提供了关键的语义脚手架，使 LLM 能够检索那些在参数层面会被遗忘的记忆。这种上下文缓冲显著缓解了保留-泛化权衡。关系密度缩小了 AlphaEdit 和 UltraEdit 之间的性能差距，同时也减轻了 AlphaEdit 灾难性的泛化能力下降（例如，其在 DeepSeek-R1 上的 Gen 分数从 -0.930 改善至 -0.327）。然而，ForgetBench 揭示了一个关键警示：虽然上下文冗余改善了可观察的召回率，但它并不保证内在的参数巩固。如 AlphaEdit 卓越的时序一致性（TC）所示，稳定的终身记忆仍然取决于权重演化，而非肤浅的上下文线索。相反，WISE 实现了近乎完美的流畅度（Flu: 0.999）和泛化能力（Gen $\approx 0$），但实际保留率有限，证明流畅的生成并不等同于真正的记忆保留。

图 4 下半部分的遗忘曲线展示出比概念型测试平滑得多的衰减梯度，表面上类似 Ebbinghaus 曲线。然而，与人类的单调衰减不同，它们显示出高度不规则的波动和负的遗忘速度值，表明存在一种人工自我恢复现象，即对相邻实体的后续编辑强化了较早的关联检索路径。

最终，这两种测试提供了互补的视角：概念型测试隔离了内在的参数记忆，而场景型测试测量了关联图式驱动的鲁棒性。这种对比凸显了一个事实：虽然人类记忆整合了事实精确性和关联灵活性，但当前的 LLM 编辑范式被迫在两者之间做出取舍。

## 6 结论

在本工作中，我们引入了 ForgetBench，这是一个旨在评估持续知识编辑下 LLM 长期记忆保留动态的基准。ForgetBench 超越了静态的单步评估，将知识保留建模为一个时序过程，并提供了一个系统框架来分析顺序更新中的遗忘轨迹。我们在不同模型和编辑方法上的实验揭示了长期事实保留与查询泛化之间的一致性权衡，表明瞬时编辑成功并不必然转化为稳定的知识保留。这些发现强调了开发更鲁棒的参数更新机制的重要性，这些机制能够在适应新信息的同时更好地维护先前获得的知识。

**局限性与未来工作**。首先，ForgetBench 依赖于合成或半合成数据集，可能无法完全捕捉现实世界知识演化的复杂性。未来工作可以整合持续演化的语料库以增强真实性。其次，我们的基准专注于顺序知识编辑。将评估扩展到替代的记忆更新机制，如持续预训练或基于强化学习的适应，将为理解 LLM 中的长期记忆动态提供更全面的视角。

[^1]: Self-rag: learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations, Cited by: §2.1.

[^2]: Human memory: a proposed system and its control processes. In Psychology of Learning and Motivation, Vol. 2, pp. 89–195. Cited by: §1, §2.1.

[^3]: Longbench: a bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (volume 1: Long papers), pp. 3119–3137. Cited by: §1, §2.1.

[^4]: Adapting language models to compress contexts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 3829–3846. Cited by: §2.1.

[^5]: Everything is editable: extend knowledge editing to unstructured data in large language models. arXiv preprint arXiv:2405.15349. Cited by: §1, §2.2.

[^6]: MemGround: long-term memory evaluation kit for large language models in gamified scenarios. arXiv preprint arXiv:2604.14158. Cited by: §1, §2.1.

[^7]: Memory: a contribution to experimental psychology. Annals of Neurosciences 20 (4), pp. 155. Cited by: §2.1, §5.2.

[^8]: Alphaedit: null-space constrained knowledge editing for language models. arXiv preprint arXiv:2410.02355. Cited by: §1, §2.2, §5.1.

[^9]: The llama 3 herd of models. arXiv preprint arXiv:2407.21783. Cited by: §5.1.

[^10]: Hierarchical orthogonal residual spread for precise massive editing in large language models. arXiv preprint arXiv:2601.11441. Cited by: §2.2.

[^11]: UltraEdit: training-, subject-, and memory-free lifelong editing in language models. arXiv preprint arXiv:2505.14679. Cited by: §5.1.

[^12]: Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. Cited by: §5.1.

[^13]: EverMemBench: benchmarking long-term interactive memory in large language models. arXiv preprint arXiv:2602.01313. Cited by: §2.1.

[^14]: Zero-shot relation extraction via reading comprehension. In Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), pp. 333–342. Cited by: §1, §2.2.

[^15]: Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems 33, pp. 9459–9474. Cited by: §2.1.

[^16]: Snapkv: llm knows what you are looking for before generation. Advances in Neural Information Processing Systems 37, pp. 22947–22970. Cited by: §2.1.

[^17]: Reinforced lifelong editing for language models. arXiv preprint arXiv:2502.05759. Cited by: §2.2.

[^18]: Forgetting curve: a reliable method for evaluating memorization capability for long-context models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 4667–4682. Cited by: §2.1.

[^19]: Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13851–13870. Cited by: §2.1.

[^20]: Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory.. Psychological review 102 (3), pp. 419. Cited by: §5.2.

[^21]: Locating and editing factual associations in gpt. Advances in Neural Information Processing Systems 35, pp. 17359–17372. Cited by: §2.2.

[^22]: Mass-editing memory in a transformer. arXiv preprint arXiv:2210.07229. Cited by: §1, §2.2, §5.1.

[^23]: Fast model editing at scale. arXiv preprint arXiv:2110.11309. Cited by: §2.2.

[^24]: Precise localization of memories: a fine-grained neuron-level knowledge editing technique for llms. arXiv preprint arXiv:2503.01090. Cited by: §2.2.

[^25]: Massive editing for large language models via meta learning. arXiv preprint arXiv:2311.04661. Cited by: §2.2.

[^26]: From recall to forgetting: benchmarking long-term memory for personalized agents. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 26814–26841. Cited by: §2.1.

[^27]: Wise: rethinking the knowledge memory for lifelong model editing of large language models. Advances in Neural Information Processing Systems 37, pp. 53764–53797. Cited by: §5.1.

[^28]: Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems 35, pp. 24824–24837. Cited by: §2.1.

[^29]: Longmemeval: benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813. Cited by: §1, §2.1.

[^30]: Softcot: soft chain-of-thought for efficient reasoning with llms. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 23336–23351. Cited by: §2.1.

[^31]: Easyedit2: an easy-to-use steering framework for editing large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 522–535. Cited by: §5.1.

[^32]: Qwen2 technical report. arXiv preprint arXiv:2407.10671. Cited by: §5.1.

[^33]: HotpotQA: a dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2369–2380. Cited by: §1, §2.1.

[^34]: Beyond static dialogues: benchmarking realistic, heterogeneous, and evolving long-term memory. arXiv preprint arXiv:2605.31086. Cited by: §2.1.
