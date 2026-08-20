---
sourceTitle: "Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"
title: "基于图结构在线难度估计的高效 RLVR 调度"
sourceUrl: "https://arxiv.org/html/2608.17941v1"
url: "https://arxiv.org/abs/2608.17941"
authors: "Zhizhao Liu, Zhiliang Tian, Xi Wang, Zhihua Wen, Yihang Xiong, Zhiquan Lai, Dongsheng Li"
publishDate: "2026-08-18"
translator: "Claude (Opus 5)"
translationDate: "2026-08-20"
pipelineRunId: "batch-20260820-084313"
pipelineSource: "translate/batch-20260820-084313/works-ready/arxiv-2608-17941-translation.md"
kind: "academic-paper"
category: "训练技术"
tags: ["RLVR", "强化学习", "难度估计", "图结构", "变分推断", "调度优化"]
language: "zh-CN"
sourceLanguage: "en"
sourceFigureCount: 1
---
# 基于图结构在线难度估计的高效 RLVR 调度

Zhizhao Liu    Zhiliang Tian    Xi Wang    Zhihua Wen    Yihang Xiong    Zhiquan Lai    Dongsheng Li

###### 摘要

RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励强化学习）可提升大语言模型的推理能力，但依赖成本高昂的 rollout 探索。为所有样本分配相同的探索预算是低效的：简单样本可能收到冗余 rollout，而困难但可学习的样本可能探索不足。现有的自适应调度器通过基于课程的样本选择或基于估计难度的非均匀 rollout 分配来解决这一不匹配问题。然而，获取可靠的在线难度估计仍具挑战性：专用探测会增加大量生成开销，而基于历史的估计器面临缺乏初始观测的冷启动问题和过时反馈问题，且通常忽略样本间的关系。为解决这些局限，我们提出了一种即插即用的基于图的在线难度估计器，它在相关样本间共享 rollout 反馈并持续更新其难度估计，从而在无需专用探测的情况下缓解冷启动和过时性问题。具体而言，我们首先基于语义和推理相似性构建难度感知样本图。基于该图，我们引入潜在难度状态，并使用 Potts 先验鼓励相邻样本共享相同状态。然后，我们采用状态级 Beta-二项模型来聚合与每个状态关联的 rollout 结果。最后，我们使用在线平均场变分算法在新反馈到达时持续更新潜在状态分配和状态级难度。我们的框架可集成到样本选择和 rollout 分配调度器中，实现无需专用探测的难度自适应探索。在多个基础模型、RL 调度器和基准测试中的实验表明，我们的框架在相同 rollout 预算下实现了更好的性能。

## 引言

RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励强化学习）通过自动可验证的反馈提升大语言模型的推理能力 [^39] [^51] [^55]。基于群体的算法，如 GRPO（Group Relative Policy Optimization，群体相对策略优化），为每个训练样本生成多个响应，并比较其奖励以更新策略 [^39]。这些 rollout 探索备选解决方案，但也产生了大量生成成本 [^18] [^50]。

然而，GRPO 通常为每个选定样本分配相同数量的 rollout [^39] [^50]。这种均匀分配将所有样本视为需要相同量的探索，尽管当前策略可能轻松解决某些样本而在其他样本上困难重重。对简单样本进行额外 rollout 通常会重复可预测的结果，而困难但可学习的样本可能收到的尝试次数过少，无法发现有用的解决轨迹 [^32] [^50] [^52]。这种预算不匹配促使了难度感知调度的发展，它使用样本难度来决定如何分配可用的探索预算。

现有研究以两种方式实现难度感知调度。首先，样本选择方法调整训练分布并决定哪些样本获得探索 [^15] [^36] [^23]。其次，rollout 分配方法为选定样本分配不同数量的 rollout，并确定每个样本获得多少探索 [^32] [^50] [^52]。尽管这些方法做出不同的调度决策，但两者都需要最新的难度估计，因为策略持续改变它能够解决的样本。这一需求使得低成本的在线难度估计对自适应 RLVR 调度至关重要。

然而，获取这种最新且可靠的估计仍具挑战性。一些研究人员通过额外的 rollout 探测显式估计训练样本在当前策略下的成功概率 [^52] [^50] [^42]。尽管这种方法提供了估计，但它会产生大量计算开销，并由于随机 rollout 数量有限而引入统计不确定性 <sup>1</sup>。此外，一些研究人员从先前训练步骤收集的结果中估计这些成功概率 [^32] [^33] [^23] [^36]。尽管这种方法减少了额外的 rollout 开销，但它面临冷启动和准确性问题，因为历史结果可能引入随机结果方差带来的噪声，并且历史结果可能随着策略模型在训练期间的演化而过时。因此，这些局限性引出了以下问题：在 RLVR 训练期间，随着策略演化，我们如何以低成本持续估计所有训练样本不断变化的难度？

我们认为，语义或推理结构相关的样本可以为难度估计提供相互信息性的证据。具有相似语义的样本通常依赖相似的模型能力，并可能表现出可利用的局部统计相关性。因此，在本文中，我们提出了一种基于图的在线难度估计框架，用于基于 RLVR 的大语言模型训练，该框架可集成到不同的 RL 训练调度器中以提升其性能。具体而言，我们首先构建一个难度感知样本图，连接具有相似语义和推理特征的样本。然后，我们将样本难度建模为图结构化潜在状态，并开发一种在线变分推断算法，使用训练期间收集的 rollout 反馈更新这些状态。通过这种方式，相关样本可以共享历史证据，使我们能够在无需额外 rollout 探测的情况下持续估计所有训练样本的成功概率。最后，我们将估计器集成到样本选择或 rollout 分配方法中，以提高 RLVR 训练效率。在不同模型和数据集上的实验证明了我们方法的准确性、效率和通用性。

我们的贡献如下：（1）我们首次将 RLVR 中的动态难度估计表述为图结构化潜在变量推断问题。（2）我们提出了一种低成本的在线估计器，它在相关样本间传播 rollout 反馈，并随着策略演化持续估计其成功概率。（3）我们将估计器集成到样本选择和 rollout 分配方法中。大量实验表明，我们的方法对冷启动友好，大幅减少了难度估计开销，同时持续改善 RLVR 调度器的下游性能。

## 相关工作

![框架概览图](https://arxiv.org/html/2608.17941v1/0728.png)

**图 1：框架概览**  
（a）构建难度感知样本图；（b）图结构化潜在模型在相关样本间聚合 rollout 反馈；（c）在线平均场变分推断更新演化的难度；（d）生成的估计指导自适应样本选择和 rollout 分配。

### RLVR 中的自适应采样和 Rollout 分配

课程学习根据难度、胜任度或可学习性对样本进行排序或重新加权 [^4] [^30]。这一原则对基于群体的 RLVR 尤为相关，其中具有几乎确定性群体奖励的提示提供很少的群体相对学习信号 [^39] [^51] [^55]。因此，提示选择方法使用估计的难度、胜任度对齐、不确定性或奖励动态来优先考虑信息性样本 [^51] [^15] [^36] [^23] [^56] [^1]。互补的研究方向分配非均匀的 rollout 数量。GVM、VIP 和 CurES 使用梯度方差或稳定性目标 [^50] [^32] [^52]，而最近的方法使用序列不确定性减少、伯努利方差代理或后验命中效用 [^47] [^14]。尽管它们的调度目标不同，这些方法都需要对当前提示行为的估计。我们的工作是互补的：它提供了一个可重用的估计器，而不是另一个调度规则。

### 推理大语言模型的动态难度估计

难度不仅取决于样本，还取决于策略模型。项目反应理论显式建模潜在能力与项目难度之间的交互，并已被改编用于 NLP 评估 [^26] [^38]。训练动态方法则通过跨 epoch 的置信度和变异性等量来刻画样本，或优先考虑仍然可学习和有用的样本 [^41] [^30]。RLVR 进一步要求这些依赖模型的估计跟踪演化的策略。现有的 RLVR 估计器遵循三种广泛策略。基于模型的方法学习价值或评分模型来预测难度，而无需对每个候选进行 rollout [^15]。基于探测的方法从额外的在策略生成中估计当前成功概率，获得及时但成本高昂且噪声较大的信号 [^50] [^52] [^42]。基于历史的方法重用训练反馈：MoPPS 执行流式贝叶斯推断 [^36]；CDAS 聚合历史性能差异 [^23]；GRESO 利用奖励的时间规律性 [^56]；VIP 将轻量级预测器拟合到近期 rollout [^32]。互补的调度方法在不同粒度上运行：Re-Schedule 从离线推理树中导出静态结构可学习性评分，用于从易到难的课程调度 [^45]，而 ARRoL 从部分 rollout 预测最终正确性，并在线修剪轨迹以平衡二元奖励 [^48]。学习的预测器必须在策略漂移下保持校准，探测和树构建会产生生成开销，历史观测可能稀疏或过时。我们的方法还借鉴了基于图的半监督学习和概率图模型，使用谱结构、Potts 先验和平均场推断在相关样本间传播稀疏反馈；详见附录 H。相比之下，我们的估计器通过显式图耦合相关样本，并通过共享潜在难度状态来池化稀疏证据。

我们的设置将这些思想与策略诱导的非平稳性相结合：rollout 观测稀疏，而其生成策略在整个训练过程中不断变化。我们构建难度感知图，将提示与时间索引潜在状态关联，通过状态级 Beta-二项因子聚合二元结果，并在训练步骤间传递变分后验。这产生了对关系和演化潜在状态的序列概率推理，以及自适应 RLVR 调度器所需的成功概率估计。

## 方法

### 3.1 概览

我们提出了一个基于图的在线框架，用于估计 RLVR 中训练样本不断演化的难度。通过利用相关样本的历史 rollout 结果，我们的框架在每个训练步骤之前预测当前策略下每个样本的成功概率和难度。如图 1 所示，该框架由四个顺序模块组成。首先，难度感知样本图联合编码训练样本的语义和难度特征，并构建稀疏训练集图（第 3.3 节）。其次，图结构化潜在难度模型将图映射到可通过 rollout 反馈动态更新的潜在难度状态，同一状态中的样本共享共同的成功概率（第 3.4 节）。第三，在线平均场变分推断更新每个样本的潜在状态分布和每个潜在状态的后验成功概率（第 3.5 节）。最后，我们使用更新后的后验估计下一训练步骤候选样本的成功概率和难度（第 3.6 节）。

### 3.2 问题表述

设强化学习的训练数据集为 $\mathcal{D}=\{x_{i}\}_{i=1}^{N}$，其中 $x_{i}$ 表示第 $i$ 个样本。遵循实际训练顺序，我们用 $\mathcal{O}_{t}$ 表示在训练步骤 $t$ 生成 rollout 的样本集。在当前策略模型 $\pi_{t}$ 下，每个样本 $i\in\mathcal{O}_{t}$ 生成 $n_{i,t}$ 个 rollout，我们用 $y_{i,t,g}\in\{0,1\}$ 表示第 $g$ 个 rollout 的二元正确性结果。我们将同一样本的 rollout 结果聚合为成功 rollout 的数量 $s_{i,t}=\sum_{g=1}^{n_{i,t}}y_{i,t,g}$，并将其经验成功率定义为 $r_{i,t}=s_{i,t}/n_{i,t}$。

我们的目标是在每个 RL 训练步骤之前准确估计当前策略正确解决每个样本的概率：$\hat{p}_{i,t}=\Pr_{Y\sim\pi_{t}}(Y\text{ is correct}\mid x_{i},\mathbf{s}_{<t},\mathbf{n}_{<t})$。然后我们将样本 $i$ 在步骤 $t$ 的难度定义为 $d_{i,t}=1-p_{i,t}$。最后，我们将这些概率或难度估计输入现有的 RL 调度框架，以指导样本选择和 rollout 分配等决策。

### 3.3 难度感知样本图

在本节中，我们基于语义和难度构建训练数据集上的稀疏相似性图。这允许先前 rollout 很少或没有的样本通过利用语义和难度相似邻居的历史反馈来估计其当前预期难度。

为了鼓励编码器捕获每个样本的语义内容、推理结构和难度相关特征，我们引入一个难度感知指令 $I_{\pi}$：指令：给定一个提示，将其编码为捕获其语义内容和难度级别的密集向量，用于相似性比较。\\n 提示：\[SAMPLE\] 设 $f_{\mathrm{emb}}$ 表示预训练的嵌入模型。样本 $x_{i}$ 的表示计算为 $\widetilde{\boldsymbol{\phi}}_{i}=f_{\mathrm{emb}}(I_{\pi},x_{i})$ 并归一化为 $\boldsymbol{\phi}_{i}=\widetilde{\boldsymbol{\phi}}_{i}/\|\widetilde{\boldsymbol{\phi}}_{i}\|_{2}$。对于每对样本，我们用余弦相似度衡量其相似性：$c_{ij}=\boldsymbol{\phi}_{i}^{\top}\boldsymbol{\phi}_{j}$。

为了捕获高维样本嵌入的底层流形结构同时降低计算成本，我们通过保留每个节点的 $k_{\mathrm{nn}}$ 个最近邻来构建 $k$ 近邻图。然后，我们仅保留两个节点互为 $k$ 近邻的边，并移除具有负相似度的边，从而产生稀疏无向样本图 $W\in\mathbb{R}^{N\times N}$。具体而言，如果 $x_{i}$ 和 $x_{j}$ 互为 $k$ 近邻且 $c_{ij}>0$，则 $W_{ij}=c_{ij}$，否则为 $0$。更大的 $W_{ij}$ 反映 $x_{i}$ 和 $x_{j}$ 之间在语义内容、推理结构和难度相关特征上的更大相似性。

### 3.4 图结构化潜在难度模型

在本节中，我们旨在通过利用图结构和先前 RL 训练期间收集的历史 rollout 结果来执行样本难度的统计推断。具体而言，我们首先在样本潜在状态上构建图结构化先验，然后通过共享的状态级成功概率建模 rollout 反馈，最后导出用于后续难度估计的后验分布。

#### 图先验

尽管 $W$ 捕获了样本间的局部相似性关系，但它不足以描述训练数据集的整体难度结构。为了进一步揭示样本在语义特征和难度级别上的相似性，我们对稀疏无向样本图 $W\in\mathbb{R}^{N\times N}$ 应用谱聚类，它产生一组初始的静态聚类标签 $c_{i}\in\{1,\ldots,K\}$。然而，谱聚类仅依赖静态样本表示，其分配可能受到表示和聚类误差的影响。更重要的是，随着我们在 RL 训练期间持续更新策略，同一样本在当前策略下的难度也可能发生变化。因此，我们将每个静态聚类标签转换为潜在状态上的平滑先验分布，我们定义 $\displaystyle A_{ik}=\begin{cases}1-\epsilon,&k=c_{i},\\
\frac{\epsilon}{K-1},&k\neq c_{i},\end{cases}$。这种转换保留了谱聚类的结构，同时也允许后续的 rollout 结果细化每个样本的潜在分配。然后，受 [^6] 启发，我们引入 Potts 图先验 [^35] 来保留初始聚类结构，同时鼓励相邻样本共享相同的潜在状态。具体而言，我们定义 $p(\mathbf{z}_{t}\mid W,A)=\frac{1}{Z(W,A)}\left(\prod_{i=1}^{N}A_{i,z_{i,t}}\right)\exp\left[\frac{\beta}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}W_{ij}\,\mathbb{I}(z_{i,t}=z_{j,t})\right],$ 其中 $\mathbf{z}_{t}=(z_{1,t},\dots,z_{N,t})$ 表示步骤 $t$ 的潜在状态分配向量，$z_{i,t}\in\{1,\dots,K\}$ 表示样本 $x_{i}$ 的分配。超参数 $\beta$ 控制图传播的强度，$Z(W,A)$ 是配分函数。

#### 状态级成功模型

为了统计推断样本难度和潜在状态分配，我们进一步将它们与历史 rollout 反馈的关系整合到贝叶斯框架中。然而，为每个样本估计独立的成功概率仍需要大量样本级 rollout 才能产生稳定估计。为了解决这个问题并提高历史反馈利用的效率，我们假设分配给同一潜在状态的样本共享共同的状态级成功概率。对于每个潜在状态 $k$，遵循 [^52] 和 [^36]，我们用 Beta 分布建模其在步骤 $t$ 的成功概率：$\theta_{k,t}\sim\operatorname{Beta}\bigl(a_{k,t}^{\mathrm{hist}},b_{k,t}^{\mathrm{hist}}\bigr)$。该分布联合捕获状态级成功概率的估计，同时支持序列贝叶斯更新。我们通过继承步骤 $t-1$ 的后验参数来设置步骤 $t$ 的先验参数，即 $a_{k,t}^{\mathrm{hist}}=a_{k,t-1}$ 和 $b_{k,t}^{\mathrm{hist}}=b_{k,t-1}$。两个参数都初始化为 $1$：$a_{k,0}=1$，$b_{k,0}=1$。在潜在状态分配 $z_{i,t}=k$ 和 rollout 结果条件独立的条件下，$n_{i,t}$ 个 rollout 中的成功数量 $s_{i,t}$ 遵循二项分布：$(s_{i,t}\mid n_{i,t},z_{i,t}=k,\theta_{k,t})\sim\operatorname{Binomial}(n_{i,t},\theta_{k,t})$。设 $\boldsymbol{\theta}_{t}=(\theta_{1,t},\dots,\theta_{K,t})$ 收集所有潜在状态的成功概率。通过结合 $\mathbf{z}_{t}$ 上的图结构化先验、每个 $\theta_{k,t}$ 上的独立 Beta 先验以及样本级二项似然，我们获得步骤 $t$ 的联合分布：$p_{t}(\mathbf{s}_{t},\mathbf{z}_{t},\boldsymbol{\theta}_{t}\mid\mathbf{n}_{t},W,A)=\;p(\mathbf{z}_{t}\mid W,A)\prod_{k=1}^{K}\operatorname{Beta}\bigl(\theta_{k,t}\mid a_{k,t}^{\mathrm{hist}},b_{k,t}^{\mathrm{hist}}\bigr)\prod_{i\in\mathcal{O}_{t}}\operatorname{Binomial}\bigl(s_{i,t}\mid n_{i,t},\theta_{z_{i,t},t}\bigr)$.

#### 后验推断

从该联合分布中，我们应用贝叶斯规则在观察 rollout 结果后推断潜在状态分配和状态级成功概率的后验。该后验允许我们联合更新关于每个样本潜在状态和共享成功概率的信念，这对后续难度估计至关重要。它因子分解为 $p_{t}(\mathbf{z}_{t},\boldsymbol{\theta}_{t}\mid\mathbf{s}_{t},\mathbf{n}_{t},W,A)=\frac{p_{t}(\mathbf{s}_{t},\mathbf{z}_{t},\boldsymbol{\theta}_{t}\mid\mathbf{n}_{t},W,A)}{p_{t}(\mathbf{s}_{t}\mid\mathbf{n}_{t},W,A)}=\frac{p_{t}(\mathbf{s}_{t},\mathbf{z}_{t},\boldsymbol{\theta}_{t}\mid\mathbf{n}_{t},W,A)}{\sum_{\mathbf{z}_{t}}\int p_{t}(\mathbf{s}_{t},\mathbf{z}_{t},\boldsymbol{\theta}_{t}\mid\mathbf{n}_{t},W,A)\,\mathrm{d}\boldsymbol{\theta}_{t}}$.

### 3.5 在线平均场变分推断

在本节中，我们采用在线平均场变分推断来支持自适应大语言模型 RL 调度的样本难度估计。在每个训练步骤，只有选定样本提供 rollout 反馈。样本图允许该反馈更新相关样本的潜在状态分布。具体而言，我们近似潜在状态 $\mathbf{z}_{t}$ 和状态级成功概率 $\boldsymbol{\theta}_{t}$ 的联合后验。然而，精确后验推断是不可行的。Potts 先验耦合相邻样本的潜在状态。这种耦合需要对所有 $K^{N}$ 种可能分配求和。其计算成本随训练样本数量呈指数增长。为了获得可处理的更新，我们遵循先前工作 [^53] [^21] 并采用平均场近似 $q_{t}(\mathbf{z}_{t},\boldsymbol{\theta}_{t})\approx\prod_{i=1}^{N}q_{i}(z_{i,t})\prod_{k=1}^{K}q(\theta_{k,t}).$这里，$q_{i}(z_{i,t})$ 表示样本 $i$ 的潜在状态分布。我们用 $q_{ik,t}\equiv q_{i}(z_{i,t}=k)$ 表示其状态分配概率。我们将每个状态级因子限制为 Beta 族：$q(\theta_{k,t})=\operatorname{Beta}(a_{k,t},b_{k,t}).$

设 $\mathcal{Q}$ 表示上述定义的变分族。我们寻找 $\mathcal{Q}$ 中使 Kullback–Leibler（KL）散度与精确后验最小化的成员：$q_{t}^{\star}=\arg\min_{q_{t}\in\mathcal{Q}}\operatorname{KL}\left[q_{t}(\mathbf{z}_{t},\boldsymbol{\theta}_{t})\,\middle\|,p_{t}(\mathbf{z}_{t},\boldsymbol{\theta}_{t}\mid\mathbf{s}_{t},\mathbf{n}_{t},W,A)\right].$ 精确后验包含 Potts 先验的配分函数 $Z(W,A)$。这使得 KL 散度难以直接评估。我们转而使用证据下界（ELBO，Evidence Lower Bound）：$\mathcal{L}_{t}(q_{t})=\mathbb{E}_{q_{t}}[\log p_{t}(\mathbf{s}_{t},\mathbf{z}_{t},\boldsymbol{\theta}_{t}\mid\mathbf{n}_{t},W,A)]-\mathbb{E}_{q_{t}}[\log q_{t}(\mathbf{z}_{t},\boldsymbol{\theta}_{t})].$ 对数边缘似然满足 $\log p_{t}(\mathbf{s}_{t}\mid\mathbf{n}_{t},W,A)=\mathcal{L}_{t}(q_{t})+\operatorname{KL}\left[q_{t}(\mathbf{z}_{t},\boldsymbol{\theta}_{t})\,\|,p_{t}(\mathbf{z}_{t},\boldsymbol{\theta}_{t}\mid\mathbf{s}_{t},\mathbf{n}_{t},W,A)\right].$ 左侧不依赖于 $q_{t}$。因此，最小化 KL 散度等价于最大化 ELBO：$q_{t}^{\star}=\arg\max_{q_{t}\in\mathcal{Q}}\mathcal{L}_{t}(q_{t}).$

在平均场近似下，我们可以评估 ELBO。具体而言，期望 Potts 交互变为 $\mathbb{E}_{q_{t}}\left[\mathbb{I}(z_{i,t}=z_{j,t})\right]=\sum_{k=1}^{K}q_{ik,t}q_{jk,t}.$ 将这一结果和变分因子代入 ELBO 得到

$$
\displaystyle\mathcal{L}_{t}(q)
$$

$$
\displaystyle=(\beta/2)\textstyle\sum_{i=1}^{N}\textstyle\sum_{j=1}^{N}W_{ij}\textstyle\sum_{k=1}^{K}q_{ik,t}q_{jk,t}
$$

$$
\displaystyle+\textstyle\sum_{i=1}^{N}\textstyle\sum_{k=1}^{K}q_{ik,t}\log A_{ik}-\textstyle\sum_{i=1}^{N}\textstyle\sum_{k=1}^{K}q_{ik,t}\log q_{ik,t}
$$

$$
\displaystyle+\textstyle\sum_{i\in\mathcal{O}_{t}}\textstyle\sum_{k=1}^{K}q_{ik,t}\bigl\{s_{i,t}[\psi(a_{k,t})-\psi(a_{k,t}+b_{k,t})]
$$

$$
\displaystyle\qquad+(n_{i,t}-s_{i,t})[\psi(b_{k,t})-\psi(a_{k,t}+b_{k,t})]\bigr\}
$$

$$
\displaystyle-\textstyle\sum_{k=1}^{K}\operatorname{KL}[\operatorname{Beta}(a_{k,t},b_{k,t})\|\operatorname{Beta}(a_{k,t}^{\mathrm{hist}},b_{k,t}^{\mathrm{hist}})].
$$

我们省略了与 $q_{t}$ 无关的项。这里，$\psi(\cdot)$ 表示 digamma 函数。附录 A 提供了详细推导。ELBO 包含两组变分参数。第一组包含分配概率 $q_{ik,t}$。第二组包含 Beta 参数 $(a_{k,t},b_{k,t})$。我们优化一组参数，同时保持另一组固定。这一过程称为坐标上升。命题 1 导出分配更新，命题 2 导出 Beta 更新。我们在推断期间交替执行这些更新。命题 3 建立了这一交替过程的收敛性。附录 A 提供了证明。

#### 命题 1：坐标最优分配更新

固定 $q(\boldsymbol{\theta}_{t})$ 和所有 $j\neq i$ 的 $q_{j}(z_{j,t})$。坐标最优更新为 $q_{ik,t}^{\star}=\operatorname{Softmax}_{k}[\log A_{ik}+\beta\sum_{j=1}^{N}W_{ij}q_{jk,t}+\mathbb{I}(i\in\mathcal{O}_{t})\ell_{ik,t}^{\mathrm{VB}}],$ 其中 $\ell_{ik,t}^{\mathrm{VB}}={}s_{i,t}\left[\psi(a_{k,t})-\psi(a_{k,t}+b_{k,t})\right]+(n_{i,t}-s_{i,t})\left[\psi(b_{k,t})-\psi(a_{k,t}+b_{k,t})\right].$

#### 命题 2：坐标最优 Beta 更新

固定 $q(\mathbf{z}_{t})$。坐标最优因子为 $q^{\star}(\theta_{k,t})=\operatorname{Beta}(a_{k,t},b_{k,t}),$ 其中 $a_{k,t}=a_{k,t}^{\mathrm{hist}}+\sum_{i\in\mathcal{O}_{t}}q_{ik,t}s_{i,t},$ 和 $b_{k,t}=b_{k,t}^{\mathrm{hist}}+\sum_{i\in\mathcal{O}_{t}}q_{ik,t}(n_{i,t}-s_{i,t}).$

#### 命题 3：单调性与收敛性

在训练步骤 $t$，我们保持 $(a_{k,t}^{\mathrm{hist}},b_{k,t}^{\mathrm{hist}})$ 固定。命题 1 和 2 中的每次更新都针对一组参数最大化 ELBO。因此，两次更新都不会降低 $\mathcal{L}_{t}(q)$，ELBO 序列收敛。这种收敛性在每个大语言模型 RL 步骤提供了稳定的估计器状态（尽管可能不是全局最优）。

我们现在在在线交替过程中应用命题 1 和 2。在训练步骤 $t$ 为 $\mathcal{O}_{t}$ 产生 rollout 反馈后，我们用先前的估计器状态初始化推断：$q_{ik,t}^{(0)}=q_{ik,t-1}$，$a_{k,t}^{(0)}=a_{k,t-1}$，$b_{k,t}^{(0)}=b_{k,t-1}.$· 我们在步骤 $t$ 的推断期间保持 $(a_{k,t}^{\mathrm{hist}},b_{k,t}^{\mathrm{hist}})$ 固定。每次迭代首先应用命题 1 更新 $q_{ik,t}$。然后应用命题 2 更新 $(a_{k,t},b_{k,t})$。第一步更新潜在状态分配，作为类 E 更新。第二步更新状态级参数，作为类 M 更新。我们重复这两步，直到 ELBO 增量低于预设阈值或迭代次数达到最大值。<sup>2</sup> 命题 1 更新已观察和未观察样本的分配。命题 2 使用观察到的反馈更新状态级成功概率。它们共同将 $\mathcal{O}_{t}$ 的反馈转换为整个训练集的更新估计器状态：$\mathcal{S}_{t}=\{q_{t},a_{t},b_{t}\}.$ 下一小节将 $\mathcal{S}_{t}$ 转换为样本级难度估计，用于自适应调度。我们还使用 $\mathcal{S}_{t}$ 初始化下一训练步骤的推断。

<table><tbody><tr><td rowspan="2">方法</td><td colspan="4">Qwen-2.5-Math-1.5B (Average@8)</td><td colspan="4">Llama-3.2-1B-Instruct (Average@8)</td></tr><tr><td>MATH500</td><td>AIME24</td><td>AIME25</td><td>Olym</td><td>MATH500</td><td>AIME24</td><td>AIME25</td><td>Olym</td></tr><tr><td>Base</td><td>39.3</td><td>3.30</td><td>3.33</td><td>25.2</td><td>17.5</td><td>0.00</td><td>0.00</td><td>3.02</td></tr><tr><td>GVM</td><td>71.9</td><td>11.3</td><td>11.3</td><td>35.8</td><td>23.9</td><td>0.42</td><td>0.00</td><td>4.59</td></tr><tr><td>GVM+Ours</td><td>74.7</td><td>16.7</td><td>13.3</td><td>38.3</td><td>25.7</td><td>3.30</td><td>0.00</td><td>4.89</td></tr><tr><td>PCL</td><td>59.7</td><td>7.92</td><td>4.58</td><td>26.1</td><td>26.7</td><td>0.42</td><td>0.00</td><td>5.04</td></tr><tr><td>PCL+Ours</td><td>61.6</td><td>11.3</td><td>6.67</td><td>27.1</td><td>27.4</td><td>0.00</td><td>0.42</td><td>5.81</td></tr><tr><td>GRESO</td><td>66.8</td><td>10.0</td><td>7.50</td><td>30.3</td><td>30.0</td><td>2.50</td><td>0.00</td><td>7.59</td></tr><tr><td>GRESO+Ours</td><td>68.2</td><td>10.0</td><td>10.0</td><td>31.9</td><td>32.6</td><td>6.67</td><td>0.42</td><td>8.26</td></tr></tbody></table>

表 1：使用两个基础模型在四个数学推理基准上不同 RL 调度方法的 Average@8 准确率。第一列列出了原始调度器及其与我们难度估计器集成的变体。下划线结果表示相对于相应原始调度器的改进（$p<5\times 10^{-5}$，见附录 C）。

### 3.6 用于自适应 RL 调度器的难度估计

在本节中，我们将前一训练步骤的状态级后验聚合为样本级的成功概率、难度估计。然后这些估计指导自适应样本选择和 rollout 分配。在 RL 训练步骤 $t$ 开始时，估计器加载前一步骤的状态 $\mathcal{S}_{t-1}=\{q_{t-1},a_{t-1},b_{t-1}\}$。对于每个潜在状态 $k$，我们计算其成功概率的后验均值为 $\mu_{k,t-1}=\frac{a_{k,t-1}}{a_{k,t-1}+b_{k,t-1}}$，我们首先为样本 $x_{i}$ 形成基于模型的预测：$\widehat{p}_{i,t}^{\text{model}}=\sum_{k=1}^{K}q_{ik,t-1}\mu_{k,t-1}$。然后我们根据该样本是否之前被 rollout 过来调整预测。如果 $x_{i}$ 没有先前 rollout，我们直接使用模型预测：$\widehat{p}_{i,t}=\widehat{p}_{i,t}^{\text{model}}$。如果它在早期步骤被采样过，我们将模型预测与其历史成功率 $\bar{r}_{i,<t}$ 平均：$\widehat{p}_{i,t}=\gamma\widehat{p}_{i,t}^{\text{model}}+(1-\gamma)\bar{r}_{i,<t})$（$\gamma=0.5$，见附录 E 中的讨论）。此外，估计的难度为 $\widehat{d}_{i,t}=1-\widehat{p}_{i,t}$。

总体而言，在训练步骤 $t$ 之前，估计器预测每个样本的难度。这些输出作为即插即用组件，用于 RLVR 训练中的大多数样本级调度和资源分配框架。

## 实验### 实现细节

**设置。** 我们在 4 × A100 80G GPU 上进行所有训练和推理。为避免数据泄漏[^46]，遵循[^50]和[^52]的做法，我们采用 Qwen 2.5 1.5B Math[^49]和 Llama 3 1B-Instruct[^17]作为基础模型。在主要实验中，我们保持环境与原始基线严格一致。我们将框架作为即插即用组件集成到原始代码中，以证明其能够改善最终的强化学习性能。为确保公平性，我们在所有实验中使用相同的计算预算，相当于标准 GVM 算法的一轮（约 150 万次 rollout，包括预采样）。我们使用 Qwen3-0.6B[^54]作为句子嵌入模型，设置 $K=320$、$k_{\mathrm{nn}}=80$、$\gamma=0.5$ 和 $\beta=2$。不同设置下的超参数或嵌入模型选择指南见附录 E。

**数据集。** 由于我们集成的所有基线都基于数学问题，我们遵循先前工作[^50][^15][^56]的实验设置，在 NuminaMath[^27][^11]上进行强化学习。我们在 MATH500[^20][^28]、AIME 2024[^2]、AIME 2025[^12]和 OlympiadBench[^19]上评估生成的模型。这些数据集的详细描述见附录 B，其在其他领域的泛化性讨论见附录 D。

**基线。** 我们在三个来自不同调度范式的代表性基线上评估即插即用组件：GVM（rollout 分配方法）[^50]；PCL（训练专用难度预测器的课程学习方法）[^15]；以及 GRESO（使用历史成功率作为难度信号的课程学习方法）[^56]。对于每个基线，我们仅将其原始难度估计器替换为我们的方法。

**指标。** 我们使用基于规则的匹配来判断结果是否与真实答案一致[^18][^40]。由于在 AIME 等小规模且具有挑战性的数据集上的单次评估可能对采样随机性敏感，我们遵循先前工作[^50][^56]，对每个样本采样八次独立 rollout，报告 Average@8 准确率。

### 整体性能

我们将难度估计器集成到三个代表性的强化学习调度器中：用于 rollout 分配的 GVM、用于课程选择的 PCL 以及用于选择性 rollout 的 GRESO。我们替换它们的原始难度估计器，同时保留其调度策略，并在相同的总生成预算下比较所有方法。

如表 1 所示，我们的估计器改善了所有三个调度器的整体性能，在大多数模型-数据集设置中都有提升。对于 GVM，GVM+Ours 仅保留用于梯度相关估计的预采样，并用我们的方法替换原始难度估计。它在八个设置中的七个改善了性能，在剩余一个设置中与基线持平。我们将在下一节提供更深入的分析。当集成到 PCL 和 GRESO 中时，我们的估计器也在大多数设置中改善了相应基线。一个合理的解释是，更稳定的难度估计帮助调度器识别在同一 rollout 组内可能产生正确和错误响应的样本。此类样本产生非退化的群体相对优势，并可能提供更有信息量的优化信号。

尽管全历史聚合可能引入时间滞后，但图传播和共享的状态级统计使其影响在相关样本之间大体系统化，更多地影响绝对校准而非相对难度排序。由于自适应调度器主要依赖排序进行样本选择和 rollout 分配，这种偏差对调度的影响有限，如上述高相关性和下游收益所支持。附录 G 进一步检验了降低过时历史影响的窗口化变体。

总体而言，这些结果表明我们的估计器可以作为不同强化学习调度策略的可复用组件，并在匹配的生成预算下改善下游性能。

### 我们的方法能否在强化学习训练期间持续跟踪样本难度？

<table><thead><tr><th rowspan="2">基础模型</th><th rowspan="2">方法</th><th colspan="2">早期步骤</th><th colspan="2">中期步骤</th><th colspan="2">后期步骤</th><th colspan="2">全程步骤</th><th rowspan="2">时间 (A100*h)</th></tr><tr><th>MAE (↓)</th><th>r (↑)</th><th>MAE (↓)</th><th>r (↑)</th><th>MAE (↓)</th><th>r (↑)</th><th>MAE (↓)</th><th>r (↑)</th></tr></thead><tbody><tr><td rowspan="5">Qwen-2.5 Math-1.5B</td><td>Sample before RL</td><td>0.116</td><td>0.544</td><td>0.139</td><td>0.856</td><td>0.151</td><td>0.865</td><td>0.135</td><td>0.522</td><td>∼ 29.6</td></tr><tr><td>Sample before Step</td><td>0.063</td><td>0.959</td><td>0.054</td><td>0.971</td><td>0.049</td><td>0.974</td><td>0.055</td><td>0.969</td><td>∼ 45.3</td></tr><tr><td>VIP</td><td>0.373</td><td>-</td><td>0.094</td><td>0.794</td><td>0.084</td><td>0.885</td><td>0.184</td><td>0.423</td><td>∼ 0.05</td></tr><tr><td>PCL</td><td>0.404</td><td>0.138</td><td>0.336</td><td>0.407</td><td>0.328</td><td>0.474</td><td>0.356</td><td>0.281</td><td>∼ 7.36</td></tr><tr><td>MoPPS</td><td>0.373</td><td>-</td><td>0.149</td><td>0.910</td><td>0.118</td><td>0.929</td><td>0.214</td><td>0.754</td><td>-</td></tr><tr><td></td><td>Ours</td><td>0.290</td><td>0.482</td><td>0.133</td><td>0.912</td><td>0.123</td><td>0.943</td><td>0.183</td><td>0.836</td><td>∼ 0.12</td></tr><tr><td rowspan="5">Llama-3.2 1B-Instruct</td><td>Sample before RL</td><td>0.116</td><td>0.374</td><td>0.141</td><td>0.648</td><td>0.154</td><td>0.671</td><td>0.137</td><td>0.394</td><td>∼ 33.4</td></tr><tr><td>Sample before Step</td><td>0.044</td><td>0.986</td><td>0.046</td><td>0.967</td><td>0.046</td><td>0.967</td><td>0.045</td><td>0.986</td><td>∼ 47.9</td></tr><tr><td>VIP</td><td>0.413</td><td>-</td><td>0.107</td><td>0.895</td><td>0.084</td><td>0.880</td><td>0.201</td><td>0.717</td><td>∼ 0.05</td></tr><tr><td>PCL</td><td>0.204</td><td>0.467</td><td>0.181</td><td>0.404</td><td>0.187</td><td>0.525</td><td>0.191</td><td>-0.231</td><td>∼ 4.38</td></tr><tr><td>MoPPS</td><td>0.413</td><td>-</td><td>0.133</td><td>0.864</td><td>0.109</td><td>0.869</td><td>0.218</td><td>0.606</td><td>-</td></tr><tr><td></td><td>Ours</td><td>0.197</td><td>0.469</td><td>0.100</td><td>0.868</td><td>0.075</td><td>0.913</td><td>0.131</td><td>0.776</td><td>∼ 0.12</td></tr></tbody></table>

表 2：GRPO 训练不同阶段的难度估计准确性和开销（批大小 = 256）。MAE 在样本级别计算，Pearson 相关系数 $r$ 在批级别计算。粗体表示最佳整体结果，下划线表示低成本方法中的最佳结果，时间（包括嵌入、聚类和推理）表示在 150K 训练样本上超过 3 个 epoch 的额外估计开销。

我们评估不同方法在同一 GRPO 训练轨迹上跟踪演化样本难度的准确性。在每个评估点，我们使用由相应策略生成的 8 次独立 rollout 的经验成功率作为参考经验成功概率。我们在表 2 中报告了校准的样本级 MAE 和排序一致性的批级 Pearson 相关系数 $r$（批大小 256），因为样本级经验成功率是离散且稀疏的。较低的 MAE 和较高的 $r$ 表示更好的估计。

我们比较了两个高成本探测基线 Sample Before RL (SBR) 和 Sample Before Step (SBS)，以及三个代表性的低成本方法：PCL（基于大语言模型的难度预测）、VIP（基于历史反馈的统计估计）和 MoPPS（基于历史的动态建模）。SBS 使用当前策略重复探测样本，因此达到最高准确性，但需要约 45-48 小时的额外计算。然而，SBR 仅在训练前估计难度一次，因此随着策略演化，其预测迅速过时，凸显了静态估计的局限性。我们的方法在低成本方法中整体表现最佳，在早期稀疏反馈阶段优势尤为明显。训练早期，它在 Qwen 和 Llama 上分别达到 0.482 和 0.469 的 Pearson 相关系数，同时在低成本基线中具有最低的 MAE。这些结果表明，图结构通过在相关样本间传递反馈来缓解观测稀疏性。随着更多 rollout 观测的积累，全轨迹相关性增加到 0.836 和 0.776，而总估计开销保持在约 0.12 小时。总体而言，我们的方法缓解了早期阶段的观测稀疏性，并以低计算开销跟踪策略引起的难度变化。

### 消融研究

<table><tbody><tr><td rowspan="2">基础模型</td><td rowspan="2">方法</td><td colspan="2">全程步骤</td></tr><tr><td>MAE (↓)</td><td>r (↑)</td></tr><tr><td rowspan="4">Qwen-2.5 Math-1.5B</td><td>Ours</td><td>0.183</td><td>0.836</td></tr><tr><td>w/o Spectral Init.</td><td>0.218</td><td>0.710</td></tr><tr><td>w/o Prior Smoothing</td><td>0.197</td><td>0.783</td></tr><tr><td>w/o Graph Sparsification</td><td>0.187</td><td>0.835</td></tr><tr><td rowspan="4">Llama-3.2 1B-Instruct</td><td>Ours</td><td>0.131</td><td>0.776</td></tr><tr><td>w/o Spectral Init.</td><td>0.197</td><td>0.686</td></tr><tr><td>w/o Prior Smoothing</td><td>0.144</td><td>0.736</td></tr><tr><td>w/o Graph Sparsification</td><td>0.134</td><td>0.742</td></tr></tbody></table>

表 3：两个基础模型上难度估计的全轨迹消融结果。较低的 MAE 和较高的 Pearson 相关系数 $r$ 表示更好的估计。

我们对谱初始化、先验平滑和图稀疏化进行消融。表 3 报告了完整训练轨迹上的结果。随机初始化导致性能大幅下降。由于我们的 EM 风格优化仅保证收敛到局部最优，谱聚类提供了高质量起点，引导推断走向更好的解，而随机初始化可能导致较差的局部最优。用硬分配替换平滑的潜在状态先验 $A_{ik}$ 也会降低性能。随着策略演化，样本难度及其潜在状态分配应相应演化。平滑先验在允许 rollout 反馈修正分配的同时保留了初始聚类结构，而硬分配阻止了这种适应。使用密集图会导致适度下降，因为弱相关边会在反馈传播中引入噪声。图稀疏化保留了样本间更可靠的关系。

### 谱聚类是否捕获语义和难度结构？

我们使用 MATH[^20]进行此分析，因为它提供了主题类别和难度级别的标准化标注，使我们能够分别评估聚类捕获的语义和难度信息。遵循第 3.3 节，我们对 MATH 问题进行嵌入并执行谱聚类。生成的聚类与主题类别强相关（$\chi^{2}=10{,}995.84$，$p<10^{-300}$；Cramér's $V=0.494$），表明谱划分有效捕获了语义结构。在控制主题类别后，难度分布在不同聚类间仍存在显著差异（Kruskal-Wallis，$p_{max}<5\times 10^{-3}$）。这些结果表明，聚类不仅捕获了问题语义，还捕获了超越主题身份的难度信息，从而为后续在线难度推断提供了有信息量的初始结构。完整统计细节见附录 F。

## 结论

在本工作中，我们通过基于图的在线难度估计解决了 RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励强化学习）中的低效探索预算分配问题。我们的框架连接具有相似语义内容和推理结构的样本，并使用 rollout 反馈跟踪其演化难度，无需专用探测。生成的估计支持样本选择和 rollout 分配调度器。在多个基础模型、调度器和基准上的实验表明，在匹配的 rollout 预算下，我们的框架在大多数设置中改善了下游推理性能，同时仅增加少量在线计算开销。这些结果展示了在线难度估计支持更高效 RLVR 训练的潜力。

## A. ELBO 推导与证明

本节推导第 3.5 节中的扩展证据下界（ELBO，Evidence Lower Bound），并证明命题 1-3。为简洁起见，定义 $\delta_{ik,t}=\mathbb{I}(z_{i,t}=k)$，并令 $C_{t}$ 收集所有与变分分布无关的项。互为近邻图是无向的，因此 $W_{ij}=W_{ji}$，且无自环，因此 $W_{ii}=0$。

### A.1 扩展 ELBO 的推导

根据第 3.4 节中的联合模型，其对数密度可以写成，直到常数 $C_{t}$：

$$
\displaystyle\log p_{t}(\mathbf{s}_{t},\mathbf{z}_{t},\boldsymbol{\theta}_{t}\mid\mathbf{n}_{t},W,A)
$$

$$
\displaystyle={}
$$

$$
\displaystyle C_{t}+\sum_{i=1}^{N}\sum_{k=1}^{K}\delta_{ik,t}\log A_{ik}+\frac{\beta}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}W_{ij}\sum_{k=1}^{K}\delta_{ik,t}\delta_{jk,t}
$$

$$
\displaystyle+\sum_{k=1}^{K}\left[(a_{k,t}^{\mathrm{hist}}-1)\log\theta_{k,t}+(b_{k,t}^{\mathrm{hist}}-1)\log(1-\theta_{k,t})\right]
$$

$$
\displaystyle+\sum_{i\in\mathcal{O}_{t}}\sum_{k=1}^{K}\delta_{ik,t}s_{i,t}\log\theta_{k,t}
$$

$$
\displaystyle+\sum_{i\in\mathcal{O}_{t}}\sum_{k=1}^{K}\delta_{ik,t}(n_{i,t}-s_{i,t})
$$

$$
\displaystyle\qquad\qquad\qquad{}\times\log(1-\theta_{k,t}).
$$

这里，$C_{t}$ 包括 $-\log Z(W,A)$、Beta 归一化常数和二项式系数，所有这些相对于 $q_{t}$ 都是固定的。

在平均场分布下：

$$
\displaystyle\mathbb{E}_{q_{t}}[\delta_{ik,t}]
$$

$$
\displaystyle=q_{ik,t},
$$
$$
\displaystyle\mathbb{E}_{q_{t}}[\delta_{ik,t}\delta_{jk,t}]
$$

$$
\displaystyle=q_{ik,t}q_{jk,t}\quad(i\neq j),
$$
$$
\displaystyle\mathbb{E}_{q_{t}}[\log\theta_{k,t}]
$$

$$
\displaystyle=\psi(a_{k,t})-\psi(a_{k,t}+b_{k,t}),
$$
$$
\displaystyle\mathbb{E}_{q_{t}}[\log(1-\theta_{k,t})]
$$

$$
\displaystyle=\psi(b_{k,t})-\psi(a_{k,t}+b_{k,t}).
$$

第二个恒等式对于图项是充分的，因为 $W_{ii}=0$。负熵贡献分解为：

$$
\displaystyle-\mathbb{E}_{q_{t}}[\log q_{t}]={}
$$

$$
\displaystyle-\sum_{i=1}^{N}\sum_{k=1}^{K}q_{ik,t}\log q_{ik,t}
$$

$$
\displaystyle-\sum_{k=1}^{K}\mathbb{E}_{q_{t}}[\log q(\theta_{k,t})].
$$

将期望对数 Beta 先验与相应的 Beta 因子熵相结合得到：

$$
\displaystyle\mathbb{E}_{q_{t}}\!\left[\log\operatorname{Beta}(\theta_{k,t}\mid a_{k,t}^{\mathrm{hist}},b_{k,t}^{\mathrm{hist}})-\log q(\theta_{k,t})\right]
$$

$$
\displaystyle\qquad=-\operatorname{KL}\!\left[\operatorname{Beta}(a_{k,t},b_{k,t})\,\middle\|\,\operatorname{Beta}(a_{k,t}^{\mathrm{hist}},b_{k,t}^{\mathrm{hist}})\right].
$$

将这些恒等式代入 $\mathcal{L}_{t}(q_{t})=\mathbb{E}_{q_{t}}[\log p_{t}]-\mathbb{E}_{q_{t}}[\log q_{t}]$ 得到下面的表达式。为保持显示紧凑，令：

$$
\displaystyle D_{k,t}^{\mathrm{Beta}}=\operatorname{KL}\!\left[\operatorname{Beta}(a_{k,t},b_{k,t})\,\middle\|\,\operatorname{Beta}(a_{k,t}^{\mathrm{hist}},b_{k,t}^{\mathrm{hist}})\right].
$$

则：

$$
\displaystyle\mathcal{L}_{t}(q_{t})={}
$$

$$
\displaystyle\frac{\beta}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}W_{ij}\sum_{k=1}^{K}q_{ik,t}q_{jk,t}
$$

$$
\displaystyle+\sum_{i=1}^{N}\sum_{k=1}^{K}q_{ik,t}\log A_{ik}
$$

$$
\displaystyle-\sum_{i=1}^{N}\sum_{k=1}^{K}q_{ik,t}\log q_{ik,t}
$$

$$
\displaystyle+\sum_{i\in\mathcal{O}_{t}}\sum_{k=1}^{K}q_{ik,t}\ell_{ik,t}^{\mathrm{VB}}
$$

$$
\displaystyle-\sum_{k=1}^{K}D_{k,t}^{\mathrm{Beta}}+C_{t},
$$

其中：

$$
\displaystyle\ell_{ik,t}^{\mathrm{VB}}={}
$$

$$
\displaystyle s_{i,t}\left[\psi(a_{k,t})-\psi(a_{k,t}+b_{k,t})\right]
$$

$$
\displaystyle+(n_{i,t}-s_{i,t})\left[\psi(b_{k,t})-\psi(a_{k,t}+b_{k,t})\right].
$$

去掉与 $q_{t}$ 无关的 $C_{t}$，即得到第 3.5 节中报告的扩展 ELBO。

### A.2 命题 1 的证明

固定 $q(\boldsymbol{\theta}_{t})$ 和除 $q_{i}(z_{i,t})$ 之外的所有分配因子。由于 $W$ 是对称的，节点 $i$ 在双重图求和中的两次出现会合并并抵消因子 $1/2$。因此，方程 (9) 中依赖于 $q_{ik,t}$ 的项为：

$$
\displaystyle\mathcal{L}_{t}(q_{i})={}
$$

$$
\displaystyle\sum_{k=1}^{K}q_{ik,t}h_{ik,t}-\sum_{k=1}^{K}q_{ik,t}\log q_{ik,t}+C,
$$
$$
\displaystyle h_{ik,t}={}
$$

$$
\displaystyle\log A_{ik}+\beta\sum_{j=1}^{N}W_{ij}q_{jk,t}+\mathbb{I}(i\in\mathcal{O}_{t})\ell_{ik,t}^{\mathrm{VB}}.
$$

为 $\sum_{k=1}^{K}q_{ik,t}=1$ 引入拉格朗日乘子 $\lambda_{i}$，定义：

$$
\displaystyle\widetilde{\mathcal{L}}_{t}(q_{i})=\mathcal{L}_{t}(q_{i})+\lambda_{i}\left(\sum_{r=1}^{K}q_{ir,t}-1\right).
$$

求导得到：

$$
\displaystyle\frac{\partial\widetilde{\mathcal{L}}_{t}(q_{i})}{\partial q_{ik,t}}=h_{ik,t}-\log q_{ik,t}-1+\lambda_{i}.
$$

令该导数为零并对 $k$ 归一化得到：

$$
\displaystyle q_{ik,t}^{\star}=\frac{\exp(h_{ik,t})}{\sum_{r=1}^{K}\exp(h_{ir,t})}=\operatorname{Softmax}_{k}(h_{ik,t}),
$$

这就是命题 1 中的更新。

### A.3 命题 2 的证明

固定 $q(\mathbf{z}_{t})$ 并定义 $q_{-k,t}=q(\mathbf{z}_{t})\prod_{r\neq k}q(\theta_{r,t})$。标准平均场坐标恒等式给出：

$$
\displaystyle\log q^{\star}(\theta_{k,t})={}
$$

$$
\displaystyle\mathbb{E}_{q_{-k,t}}\bigl[\log p_{t}(\mathbf{s}_{t},\mathbf{z}_{t},\boldsymbol{\theta}_{t}
$$

$$
\displaystyle\qquad\mid\mathbf{n}_{t},W,A)\bigr]+C.
$$

仅保留方程 (1) 中涉及 $\theta_{k,t}$ 的项，得到：

$$
\displaystyle\alpha_{k,t}
$$

$$
\displaystyle=a_{k,t}^{\mathrm{hist}}+\sum_{i\in\mathcal{O}_{t}}q_{ik,t}s_{i,t},
$$
$$
\displaystyle\gamma_{k,t}
$$

$$
\displaystyle=b_{k,t}^{\mathrm{hist}}+\sum_{i\in\mathcal{O}_{t}}q_{ik,t}(n_{i,t}-s_{i,t}),
$$
$$
\displaystyle\log q^{\star}(\theta_{k,t})={}
$$

$$
\displaystyle(\alpha_{k,t}-1)\log\theta_{k,t}
$$

$$
\displaystyle+(\gamma_{k,t}-1)\log(1-\theta_{k,t})+C.
$$

这是 Beta 分布的对数密度。因此：

$$
\displaystyle q^{\star}(\theta_{k,t})
$$

$$
\displaystyle=\operatorname{Beta}(a_{k,t},b_{k,t}),
$$
$$
\displaystyle a_{k,t}
$$

$$
\displaystyle=a_{k,t}^{\mathrm{hist}}+\sum_{i\in\mathcal{O}_{t}}q_{ik,t}s_{i,t},
$$
$$
\displaystyle b_{k,t}
$$

$$
\displaystyle=b_{k,t}^{\mathrm{hist}}+\sum_{i\in\mathcal{O}_{t}}q_{ik,t}(n_{i,t}-s_{i,t}),
$$

这证明了命题 2。

### A.4 顺序与同步实现

命题 1 给出了在所有其他因子固定时，单个分配因子 $q_{i}(z_{i,t})$ 的精确坐标最优更新。我们考虑此更新的两种实现。

**顺序参考实现。**

我们的参考实现对样本执行严格的顺序 Gauss-Seidel 扫描。在更新顺序 $i=1,\ldots,N$ 下，迭代 $m+1$ 时的分配更新为：

$$
\displaystyle q_{ik,t}^{(m+1)}=\operatorname{Softmax}_{k}\Bigg[
$$
$$
\displaystyle\log A_{ik}+\beta\sum_{j<i}W_{ij}q_{jk,t}^{(m+1)}+\beta\sum_{j>i}W_{ij}q_{jk,t}^{(m)}
$$

$$
\displaystyle+\mathbb{I}(i\in\mathcal{O}_{t})\ell_{ik,t}^{\mathrm{VB},(m)}\Bigg].
$$

因此，每次更新使用先前更新因子的最新可用值。在一次完整分配扫描后，根据命题 2 更新 Beta 因子。由于每个分配因子在其余因子固定时被优化，此实现是精确的坐标上升变分推断过程。命题 3 中的单调性和收敛结果适用于此顺序实现。

**同步加速实现。**

严格的顺序更新难以在样本上并行化。因此我们还实现了同步 Jacobi 风格更新：

$$
\displaystyle\widetilde{q}_{ik,t}^{(m+1)}=\operatorname{Softmax}_{k}\left[\log A_{ik}+\beta\sum_{j=1}^{N}W_{ij}q_{jk,t}^{(m)}+\mathbb{I}(i\in\mathcal{O}_{t})\ell_{ik,t}^{\mathrm{VB},(m)}\right].
$$

方程 (24) 中的所有分配因子从同一先前迭代并行计算，之后使用命题 2 更新 Beta 因子。

与顺序实现不同，同步分配扫描不是 ELBO 的精确块坐标最大化。因此，命题 3 不直接保证每次同步扫描都不减少 ELBO。我们将同步实现视为理论基础顺序过程的并行近似。

为评估此近似，我们比较两种实现产生的收敛样本级成功概率估计。我们将训练步骤 $t$ 时的差异定义为：

$$
\displaystyle\Delta_{t}=\max_{1\leq i\leq N}\left|\widehat{p}_{i,t}^{\mathrm{sync}}-\widehat{p}_{i,t}^{\mathrm{seq}}\right|.
$$

在评估的训练检查点上，差异保持在 $10^{-2}$ 以内。同时，同步实现允许图消息聚合和分配更新高效并行执行，从而获得显著更好的实际性能。因此我们在实践中使用并推荐同步实现，同时保留顺序实现作为理论分析的参考算法。

### A.5 命题 3 的证明

我们对第 A.4 节中描述的严格顺序实现证明此结果。固定训练步骤 $t$。在此步骤的整个推断过程中，历史参数 $(a_{k,t}^{\mathrm{hist}},b_{k,t}^{\mathrm{hist}})$、图 $(W,A)$ 和所有 rollout 观测都是固定的。

我们假设：

$$
\displaystyle A_{ik}>0\quad\text{对于每个 }i,k,
$$
$$
\displaystyle a_{k,t}^{\mathrm{hist}}>0,\qquad b_{k,t}^{\mathrm{hist}}>0\quad\text{对于每个 }k,
$$
$$
\displaystyle|W_{ij}|<\infty,\qquad n_{i,t}<\infty.
$$

当 $0<\epsilon<1$ 时，平滑初始化满足第一个条件。我们进一步假设分类因子循环更新，且每个因子在每次完整顺序扫描中更新一次。

**单调性。**

固定除一个分类因子 $q_{i}(z_{i,t})$ 之外的所有变分因子。命题 1 给出了关于此因子的 ELBO 唯一最大化器。因此，更新 $q_{i}(z_{i,t})$ 不会降低 ELBO。

类似地，在固定 $q(\mathbf{z}_{t})$ 和除 $q(\theta_{k,t})$ 之外的所有 Beta 因子后，命题 2 给出坐标最优 Beta 因子。因此更新此因子不会降低 ELBO。因此，如果 $q_{t}^{(m)}$ 表示第 $m$ 次坐标更新后的变分分布，则：

$$
\displaystyle\mathcal{L}_{t}(q_{t}^{(m+1)})\geq\mathcal{L}_{t}(q_{t}^{(m)}).
$$

**参数序列的有界性。**

每个分类参数属于概率单纯形：

$$
\displaystyle(q_{i1,t},\ldots,q_{iK,t})\in\Delta_{K}.
$$

$N$ 个分类单纯形的乘积是紧的。

将训练步骤 $t$ 观测到的成功和失败总数定义为：

$$
\displaystyle S_{t}
$$

$$
\displaystyle=\sum_{i\in\mathcal{O}_{t}}s_{i,t},
$$
$$
\displaystyle F_{t}
$$

$$
\displaystyle=\sum_{i\in\mathcal{O}_{t}}(n_{i,t}-s_{i,t}).
$$

根据命题 2 和 $0\leq q_{ik,t}\leq 1$，Beta 参数满足：

$$
\displaystyle a_{k,t}^{\mathrm{hist}}\leq a_{k,t}\leq a_{k,t}^{\mathrm{hist}}+S_{t},
$$
$$
\displaystyle b_{k,t}^{\mathrm{hist}}\leq b_{k,t}\leq b_{k,t}^{\mathrm{hist}}+F_{t}.
$$

因此，所有 Beta 参数保持在紧正区间内。完整的变分参数序列因此位于紧集：

$$
\displaystyle\mathcal{X}_{t}={}
$$

$$
\displaystyle\left(\prod_{i=1}^{N}\Delta_{K}\right)
$$

$$
\displaystyle\times\prod_{k=1}^{K}\left[a_{k,t}^{\mathrm{hist}},a_{k,t}^{\mathrm{hist}}+S_{t}\right]
$$

$$
\displaystyle\times\prod_{k=1}^{K}\left[b_{k,t}^{\mathrm{hist}},b_{k,t}^{\mathrm{hist}}+F_{t}\right].
$$

特别地，参数序列至少有一个累积点。

**ELBO 值的收敛性。**

令：

$$
\displaystyle p_{t}^{\mathrm{post}}=p_{t}(\mathbf{z}_{t},\boldsymbol{\theta}_{t}\mid\mathbf{s}_{t},\mathbf{n}_{t},W,A)
$$

表示精确后验。ELBO 满足：

$$
\displaystyle\mathcal{L}_{t}(q_{t})={}
$$

$$
\displaystyle\log p_{t}(\mathbf{s}_{t}\mid\mathbf{n}_{t},W,A)
$$

$$
\displaystyle-\operatorname{KL}\left[q_{t}\,\middle\|\,p_{t}^{\mathrm{post}}\right].
$$

由于模型具有有限数量的潜在分配、适当的 Beta 先验和有限的 rollout 计数，对数证据是有限的。由于 KL 散度非负：

$$
\displaystyle\mathcal{L}_{t}(q_{t})\leq\log p_{t}(\mathbf{s}_{t}\mid\mathbf{n}_{t},W,A).
$$

将此上界与方程 (29) 结合表明，ELBO 值序列是单调的且有上界。因此存在有限的 $\mathcal{L}_{t}^{\star}$ 使得：

$$
\displaystyle\lim_{m\rightarrow\infty}\mathcal{L}_{t}(q_{t}^{(m)})=\mathcal{L}_{t}^{\star}.
$$

**累积点的平稳性。**

考虑完整顺序扫描边界处的迭代，令 $T_{t}$ 表示对应于一次完整扫描所有分类和 Beta 因子更新的映射。在方程 (26)-(28) 下，softmax 分配更新、Beta 更新及相关的 digamma 期望在 $\mathcal{X}_{t}$ 上是连续的。因此，$T_{t}$ 是连续的。

令 $\bar{q}_{t}$ 为扫描级变分序列的累积点。假设，为了矛盾，$T_{t}(\bar{q}_{t})\neq\bar{q}_{t}$。由于每次坐标更新都是其坐标子问题的唯一最大化器，扫描中至少有一次更新会产生严格的 ELBO 增加。因此：

$$
\displaystyle\mathcal{L}_{t}(T_{t}(\bar{q}_{t}))>\mathcal{L}_{t}(\bar{q}_{t}).
$$

根据连续性，同样的正改进在 $\bar{q}_{t}$ 的邻域内也成立。这与方程 (39) 中 ELBO 值的收敛性矛盾，根据该式，一次完整扫描的 ELBO 改进必须趋于零。因此：

$$
\displaystyle T_{t}(\bar{q}_{t})=\bar{q}_{t}.
$$

因此，每个累积点都是所有精确坐标更新的不动点，因而是坐标最优的。由于在 $A_{ik}>0$ 下分类更新严格为正，且 Beta 参数保持严格为正，在此不动点处 ELBO 在单纯形约束下是可微的。因此满足相应的一阶 Karush-Kuhn-Tucker 条件。每个累积点因此是 ELBO 的平稳点。

该论证建立了 ELBO 值的收敛性和每个累积点的平稳性。

### A.6 EM 迭代可视化

![参见说明](https://arxiv.org/html/2608.17941v1/Figures/qwen_two_slice_rl_step_1_10_em_elbo_combined.png)

图 2：EM 迭代

我们在图 2 中展示了 EM 迭代下 ELBO 的演化。通常，M 步仅在第一次迭代时执行显著的大更新。E 步和 M 步的每轮 ELBO 改进大致呈指数衰减，通常在 50 次迭代内收敛到局部平稳点或局部最大值。

## B. 数据集和训练细节

**NuminaMath。**

NuminaMath 是一个大规模数学推理数据集,包含约 860K 个问题-解答对。它涵盖从中国高中数学到美国和国际数学奥林匹克竞赛的问题，数据主要从在线考试试卷和数学讨论论坛收集。每个问题都附有思维链解答和可验证的最终答案。遵循 GVM 和 CurES 的数据处理协议，我们从原始语料库中提取约 150K 个问题用于强化学习训练[^11]。
#### MATH

MATH 是一个竞赛级数学推理数据集,包含 12.5K 个问题,每个问题都附有完整的分步解答和最终答案。该数据集提供了预定义的难度标注(从 Level 1 到 Level 5)以及涵盖七个学科的类别标签:初等代数(prealgebra)、代数(algebra)、数论(number theory)、计数与概率(counting and probability)、几何(geometry)、中级代数(intermediate algebra)和微积分预备课程(precalculus)。我们仅将这些标注作为外部参考,而非训练监督信号。具体而言,我们将完整方法与移除聚类或图结构组件的变体进行比较,并检验学习到的表示和结构关系与提供的类别和难度标签的对齐程度。该分析评估了所提出的组件是否能够捕捉问题间的语义相似性并区分不同难度级别的问题。

MATH-500。MATH-500 是从原始 MATH 基准中选取的 500 个问题的代表性子集。它保留了 MATH 的多样化数学学科和难度级别,同时实现了更高效和标准化的评估。每个问题都附有参考解答和可验证的最终答案。我们使用 MATH-500 评估模型在广泛的竞赛级问题上的通用数学推理能力。

#### AIME 2024

我们的 AIME 2024 评估集合并了 AIME I 的 15 个问题和 AIME II 的 15 个问题,共计 30 个问题。这些问题涵盖代数、几何、数论、计数和概率等领域,通常需要多步推理和非平凡的数学洞察力。每个问题的答案是介于 000 到 999 之间的唯一整数,允许基于规则的可靠评估。我们使用 AIME 2024 评估在具有挑战性的竞赛级数学问题上的推理性能。AIME 的通用格式由每场考试 15 个问题组成,每个问题需要一个从 000 到 999 的整数答案。

#### AIME 2025

我们的 AIME 2025 评估集同样合并了 AIME I 和 AIME II,产生 30 个问题。每个问题需要一个介于 000 到 999 之间的整数答案,涉及跨多个领域的高级高中数学。由于这些问题的发布时间比大多数常用数学推理基准更晚,AIME 2025 为模型的推理和泛化能力提供了具有挑战性的评估。

#### OlympiadBench

OlympiadBench 是一个双语和多模态科学推理基准,包含从国际奥林匹克竞赛、中国奥林匹克竞赛和中国高考收集的 8,476 个问题。最终的 ACL 2024 版本包含 6,142 个数学问题和 2,334 个物理问题,每个问题都附有专家标注的解答。该基准既包括开放式问题也包括定理证明问题,涵盖带有文本或视觉信息的英文和中文问题。遵循 GVM 和 CurES 的评估协议,我们使用 OlympiadBench 的相应数学部分来评估在具有挑战性的奥林匹克级问题上的推理性能。

## C. 主要结果的显著性分析

### 问题与比较单元

主要实验为每个调度器-基础模型-基准组合报告一个 Average@8 点估计值。我们使用双侧精确符号检验[^13]来检验观察到的改进是否在这些异构实验设置中方向一致。该分析直接评估集成我们的方法是否产生正向变化的频率高于等概率零假设下的预期。

分析包括三个完整集成:GVM + Ours、PCL + Ours 和 GRESO + Ours。

对于集成方法 $m$、基础模型 $b$ 和基准 $d$,定义

$$
\displaystyle\Delta_{m,b,d}=\operatorname{Acc}_{m+\mathrm{Ours},b,d}-\operatorname{Acc}_{m,b,d}.
$$

如果 $\Delta_{m,b,d}>0$ 我们记录为增益,如果 $\Delta_{m,b,d}<0$ 记录为损失,如果 $\Delta_{m,b,d}=0$ 记录为平局。令 $W$、$L$ 和 $T$ 分别表示增益、损失和平局的数量。由于平局不提供方向信息,它们被排除在符号检验样本量之外,产生 $n=W+L$。

### 精确检验与结果

在零假设下,非平局比较等可能地偏向任一方法,

$$
\displaystyle H_{0}:\Pr(\Delta>0\mid\Delta\neq 0)=\tfrac{1}{2},\qquad W\sim\operatorname{Binomial}(n,\tfrac{1}{2}).
$$

双侧精确 $p$ 值为

$$
\displaystyle p_{\mathrm{sign}}=\min\left\{1,\;2\sum_{r=0}^{\min(W,L)}\binom{n}{r}2^{-n}\right\}.
$$

直接从表 1 计数得到以下结果。

| 完整集成 | $W$ | $L$ | $T$ | $p_{\mathrm{sign}}$ |
| --- | --- | --- | --- | --- |
| GVM + Ours | 7 | 0 | 1 | 0.0156 |
| PCL + Ours | 7 | 1 | 0 | 0.0703 |
| GRESO + Ours | 7 | 0 | 1 | 0.0156 |
| 合并 | 21 | 1 | 2 | $1.10\times 10^{-5}$ |

表 4:完整集成的方向性符号检验结果。平局被排除在精确检验之外。

对于 PCL 集成,$W=7$,$L=1$,$n=8$。其双侧精确 $p$ 值为

$$
\displaystyle p_{\mathrm{sign}}
$$

$$
\displaystyle=2\left[\binom{8}{0}+\binom{8}{1}\right]2^{-8}
$$

$$
\displaystyle=\frac{18}{256}=0.0703125.
$$

在名义显著性水平 $\alpha=0.10$ 下,所有三个完整集成都表现出统计显著的方向性改进。具体而言,GVM + Ours 和 GRESO + Ours 在 $\alpha=0.05$ 水平上达到显著性,而 PCL + Ours 在 $\alpha=0.10$ 水平上达到显著性。这些结果表明,我们方法的正向效应不局限于特定的调度器、基础模型或基准,而是在评估的配置中一致观察到。

对于合并分析,$W=21$,$L=1$,$n=22$。因此方程 (44) 给出

$$
\displaystyle p_{\mathrm{sign}}
$$

$$
\displaystyle=2\left[\binom{22}{0}+\binom{22}{1}\right]2^{-22}
$$

$$
\displaystyle=\frac{46}{4{,}194{,}304}=1.0967\times 10^{-5}.
$$

合并结果提供了强有力的证据反对正向和负向变化等可能的零假设。在三个完整集成中,我们的方法改进了 22 个非平局实验比较中的 21 个,对应 $95.5\%$ 的增益率。因此,精确符号检验证实了在评估的调度器、基础模型和基准中高度一致的正向改进方向。这种跨设置的一致性特别有价值,因为它表明我们方法的益处在不同的集成策略和实验条件下都能泛化,而不是依赖于单一有利配置。

## D. 跨域泛化到代码

<table><thead><tr><th rowspan="2">基础模型</th><th rowspan="2">方法</th><th colspan="2">完整步骤</th></tr><tr><th>MAE (↓)</th><th>r (↑)</th></tr></thead><tbody><tr><th rowspan="4">Qwen2.5-Coder-7B Instruct</th><th>VIP</th><td>0.197</td><td>0.504</td></tr><tr><th>PCL</th><td>0.402</td><td>0.307</td></tr><tr><th>MoPPS</th><td>0.224</td><td>0.773</td></tr><tr><th>Ours</th><td>0.176</td><td>0.845</td></tr></tbody></table>

表 5:使用 Qwen2.5-Coder-7B-Instruct 在代码生成上的难度估计性能。MAE 为样本级,r 为批次级 Pearson 相关系数。

### 设置

许多现有的自适应强化学习调度方法主要针对数学推理开发和评估。将它们扩展到代码生成可能需要针对奖励定义、采样策略或超参数的特定领域修改。重新实现这些适配可能导致基线间优化程度不均,并损害比较的公平性。因此,与其在代码领域重现所有调度基线,我们检验我们的难度估计器是否可以直接迁移到代码生成,并跟踪强化学习训练期间样本难度的变化。

我们在 LiveCodeBench[^22] release_v2(包含 511 个编程问题)上使用 Qwen2.5-Coder-7B-Instruct 进行强化学习训练。LiveCodeBench 从 LeetCode、AtCoder 和 Codeforces 收集问题,并提供平台派生的难度评级和可执行测试用例[^22]。只有当生成的程序通过所有相应测试用例时,才被认为是正确的。

与主要实验一致,我们为每个选定样本生成八个 rollout,并使用正确程序的比例作为其经验成功率。然后我们在完整训练轨迹上重放估计过程。在每个训练步骤之前,估计器仅使用从先前步骤获得的 rollout 反馈预测样本成功概率,并将这些预测与随后观察到的结果进行比较。

### 结果

如表 5 所示,我们的方法在 Qwen2.5-Coder-7B-Instruct 的强化学习轨迹中持续跟踪训练样本不断演变的成功概率。该结果表明,所提出的估计器不依赖于数学推理任务中常用的精确匹配奖励。当反馈被通过测试用例执行确定的程序正确性替换时,样本图和历史 rollout 结果仍然为估计动态样本难度提供有效信息。

总体而言,该实验提供了初步证据,表明我们的估计器可以迁移到基于执行的代码生成,而无需针对代码领域的修改。

### 讨论

我们估计器的实际需求在代码生成中可能较弱。一些竞技编程基准提供平台难度评级和执行反馈,这在某些设置中可能已足够用于课程学习或 rollout 分配。这可能部分解释了相关研究更强烈关注数学的原因。因此,该实验建立了可迁移性,而非在代码生成中的明确优势。

## E. 超参数分析

### $K$、$k_{\mathrm{nn}}$ 和 $\beta$ 的指南

我们讨论控制难度估计器的三个超参数。$K$ 是潜在难度状态的数量,决定建模粒度;$k_{\mathrm{nn}}$ 是图构建的邻域大小,控制反馈传播范围;$\beta$ 是 Potts 先验强度,平衡图平滑与对策略变化的适应。

首先,我们分析超参数 $\beta$ 和 $K$ 对难度估计的影响,如表 6 所示,其中 MAE 为样本级,r 为批次级 Pearson 相关系数。当 $\beta$ 从 $2$ 增加到 $5$ 或更高时,所有指标变得几乎相同且略微恶化,表明过强的 Potts 先验使模型过度依赖邻居信息而忽略全局难度结构。相比之下,较小的 $K$ 略微提高整体相关性 $r$,表明较少的潜在状态更好地捕捉全局难度趋势,但样本级 MAE 增加,降低了单样本精度。考虑到这种权衡,我们在主要实验中选择 $\beta=2$ 和 $K=320$。

<table><thead><tr><th rowspan="2"><math><semantics><mi>β</mi> <annotation>\beta</annotation></semantics></math></th><th rowspan="2">K</th><th colspan="2">早期步骤</th><th colspan="2">完整步骤</th></tr><tr><th>r (↑)</th><th>MAE (↓)</th><th>r (↑)</th><th>MAE (↓)</th></tr></thead><tbody><tr><th>2</th><th>320</th><td>0.482</td><td>0.290</td><td>0.836</td><td>0.183</td></tr><tr><th>2</th><th>160</th><td>0.480</td><td>0.306</td><td>0.838</td><td>0.191</td></tr><tr><th>2</th><th>80</th><td>0.494</td><td>0.313</td><td>0.843</td><td>0.194</td></tr><tr><th>2</th><th>40</th><td>0.507</td><td>0.327</td><td>0.848</td><td>0.196</td></tr><tr><th>1</th><th>320</th><td>0.488</td><td>0.302</td><td>0.849</td><td>0.189</td></tr><tr><th>5</th><th>320</th><td>0.494</td><td>0.297</td><td>0.847</td><td>0.186</td></tr><tr><th>10</th><th>320</th><td>0.490</td><td>0.297</td><td>0.847</td><td>0.186</td></tr><tr><th>20</th><th>320</th><td>0.493</td><td>0.298</td><td>0.847</td><td>0.187</td></tr></tbody></table>

表 6:Qwen-2.5-Math-1.5B 上 $\beta$ 和 $K$ 的敏感性分析。MAE 为样本级,r 为批次级 Pearson 相关系数。

邻域大小 $k_{\mathrm{nn}}$ 决定难度感知样本图的初始连接性,从而控制 rollout 反馈可以在样本间传播多远。实验表明,只要 $k_{\mathrm{nn}}$ 不设置得极小(例如低于 10),估计精度几乎不受影响:经过互为邻居和正相似性过滤后,图本身已经足够稀疏,因此少量邻居足以实现有效的反馈传播。同时,消融研究表明,移除图稀疏化仅导致性能小幅下降,表明过度增加邻域大小不会带来额外收益。从计算角度来看,增加 $k_{\mathrm{nn}}$ 会提高图上消息传播的成本——在每次变分推断迭代中,更新每个样本的潜在状态分布需要聚合其邻居的信息,理论复杂度为 $O(Nk_{\mathrm{nn}})$。然而,由于稀疏图上的随机内存访问和正相似性过滤导致的边饱和,实际墙上时钟时间不会随 $k_{\mathrm{nn}}$ 线性增长<sup>3</sup>。

### 嵌入模型和难度感知指令的讨论

#### 模型选择

我们比较了几个支持任务感知或指令条件嵌入的最新文本嵌入模型,包括 Qwen3-Embedding-0.6B、Qwen3-Embedding-4B[^54] 和 BGE-M3[^9]。当使用它们的标准密集表示来构建样本图时,产生的性能大致相似(见表 7),表明我们的框架对最近嵌入模型中的确切选择不敏感。在直接的未压缩配置中,Qwen3-Embedding-0.6B 在 Qwen-2.5-Math-1.5B 上与更大的 Qwen3-Embedding-4B 和 BGE-M3 表现相当(0.183 vs. 0.188 vs. 0.188 MAE),而在 Llama-3.2-1B-Instruct 上略差于 Qwen3-Embedding-4B(0.131 vs. 0.130)。鉴于 Qwen3-Embedding-0.6B 效率显著更高,我们采用它作为默认编码器,在精度和计算成本之间进行权衡。

模型规模和原始嵌入维度与下游性能之间不呈现单调关系。特别是,使用其 Matryoshka 兼容表示[^25]将 Qwen3-Embedding-4B 输出压缩到 1,024 维,相对于使用完整维度表示提高了性能(Qwen 上 0.179 vs. 0.188 MAE;Llama 上 0.121 vs. 0.130,如表 7 所示)。值得注意的是,1,024 维的 MRL 压缩 Qwen3-Embedding-4B 在两个基础模型上都达到最佳整体结果(Qwen 为 0.179/0.844,Llama 为 0.121/0.787),表明如果额外计算成本可接受,更大的编码器结合维度缩减可以进一步提高图质量。尽管如此,对于主要实验,我们优先考虑低开销,因此使用 Qwen3-Embedding-0.6B 作为默认。

MRL 压缩带来收益的合理解释是维度灾难:在过高的维度空间中,成对距离可能变得不太具有区分性,这可能使最近邻选择更嘈杂并削弱局部图结构。Matryoshka 表示学习(MRL)在嵌套维度上保留信息,允许在不应用任意后置投影的情况下缩短表示。我们的观察与这种解释一致,尽管它本身并未将维度确立为唯一的因果因素。

相比之下,将最近的指令感知编码器替换为早期的 Sentence-BERT 系列模型[^37],或替换为 VIP[^32] 采用的编码器配置,会导致性能大幅下降。这些较旧的表示主要针对通用句子级相似性进行优化,似乎不太适合捕捉我们的图所需的任务特定语义、推理和难度线索。基于这些结果,我们建议将我们的方法与最近的任务感知嵌入模型配对。Qwen3-Embedding-0.6B 提供了强大的默认效率-性能权衡,而截断到中等维度的更大 MRL 兼容编码器是一个可行的替代方案。

#### 难度感知指令的讨论

<table><tbody><tr><th rowspan="2">基础模型</th><th rowspan="2">方法</th><td colspan="2">完整步骤</td></tr><tr><td>MAE (↓)</td><td>r (↑)</td></tr><tr><th rowspan="6">Qwen-2.5 Math-1.5B</th><th>Qwen3-Embedding-0.6B</th><td>0.183</td><td>0.836</td></tr><tr><th>-难度感知指令</th><td>0.309</td><td>0.583</td></tr><tr><th>Qwen3-Embedding-4B</th><td>0.188</td><td>0.827</td></tr><tr><th>Qwen3-Embedding-4BMRL(1024 维)</th><td>0.179</td><td>0.844</td></tr><tr><th>BGE-M3</th><td>0.188</td><td>0.823</td></tr><tr><th>MiniLM</th><td>0.326</td><td>0.407</td></tr><tr><th rowspan="6">Llama-3.2 1B-Instruct</th><th>Qwen3-Embedding-0.6B</th><td>0.131</td><td>0.776</td></tr><tr><th>-难度感知指令</th><td>0.302</td><td>0.486</td></tr><tr><th>Qwen3-Embedding-4B</th><td>0.130</td><td>0.749</td></tr><tr><th>Qwen3-Embedding-4BMRL(1024 维)</th><td>0.121</td><td>0.787</td></tr><tr><th>BGE-M3</th><td>0.139</td><td>0.783</td></tr><tr><th>MiniLM</th><td>0.304</td><td>0.407</td></tr></tbody></table>

表 7:不同嵌入配置的完整轨迹难度估计性能。"-难度感知指令"行从默认编码器中移除了指令。MAE:样本级平均绝对误差(越低越好);r:批次级 Pearson 相关系数(越高越好)。

表 7 显示,从 Qwen3-Embedding-0.6B 中移除难度感知指令会导致估计质量大幅下降。在 Qwen-2.5-Math-1.5B 上,MAE 从 0.183 增加到 0.309,相关性从 0.836 下降到 0.583;在 Llama-3.2-1B-Instruct 上,MAE 从 0.131 增加到 0.302,相关性从 0.776 下降到 0.486。这证实了该指令对图质量的关键贡献,它鼓励编码器捕捉超越纯表面语义的难度相关和推理导向特征。没有它,嵌入倾向于连接语义相似但对于难度迁移不一定提供信息的样本。

Qwen3-Embedding-4B 与其 MRL 截断的 1,024 维变体之间的比较也表明,当通过表示学习兼容方法完成时,降低维度可以提高最近邻结构的区分性。BGE-M3 达到与 Qwen3 变体相当的性能,而 MiniLM 表现不佳,这与上述观察一致,即较旧的通用编码器缺乏构建难度感知图所需的任务感知能力。

### 插值权重 $\gamma$ 的讨论

对于具有先前 rollout 的样本,我们将基于图的预测和样本级经验成功率组合为 $\widehat{p}_{i,t}=\gamma\widehat{p}_{i,t}^{\mathrm{model}}+(1-\gamma)\bar{r}_{i,<t}$。参数 $\gamma$ 控制可迁移的基于图的证据与直接样本级历史之间的平衡。较小的 $\gamma$ 使估计由 $\bar{r}_{i,<t}$ 主导,这削弱了图传播并可能损害难度排名。较大的 $\gamma$ 更多地依赖潜在状态模型,可能忽略可靠的直接观察,导致更大的估计误差。

我们将 $\gamma=0.5$ 设为通用默认值。在我们的实验中,对于 $\gamma\in[0.5,0.8]$ 性能相对稳定。当 $\gamma<0.5$ 时,排名质量往往下降;当 $\gamma>0.8$ 时,样本级估计误差通常增加。这表明中等插值权重更可取。

更有原则的选择是根据不确定性调整 $\gamma$。潜在状态 (k) 的不确定性可以从其 Beta 后验的方差估计:

$$
v_{k,t-1}=\frac{a_{k,t-1}b_{k,t-1}}{(a_{k,t-1}+b_{k,t-1})^{2}(a_{k,t-1}+b_{k,t-1}+1)}.
$$

结合分配不确定性,这给出

$$
\widehat{u}_{i,t}=\sum_{k=1}^{K}q_{ik,t-1}\left[v_{k,t-1}+\left(\mu_{k,t-1}-\widehat{p}_{i,t}^{\mathrm{model}}\right)^{2}\right].
$$

原则上,可以在基于模型的预测置信度高时增加 $\gamma$,在样本有足够直接观察时降低它。然而,这样的规则也依赖于 rollout 计数、采样分布和下游调度器。这些因素在集成框架间变化,难以用有限实验固定。因此我们使用固定的 $\gamma=0.5$。

## F. 谱聚类的讨论

### 与学科类别的关联

对于每个分析的 MATH 问题 $i$,令 $z_{i}\in\{1,\ldots,K_{+}\}$ 表示其非空谱聚类分配,令 $c_{i}\in\{1,\ldots,U\}$ 表示其学科类别。我们形成 $K_{+}\times U$ 列联表

$$
\displaystyle O_{ku}=\sum_{i=1}^{n}\mathbb{I}(z_{i}=k,c_{i}=u),
$$

其中行总计为 $O_{k\cdot}$,列总计为 $O_{\cdot u}$,总样本量为 $n$。在聚类分配与学科类别独立的零假设下,单元格 $(k,u)$ 的期望计数为

$$
\displaystyle E_{ku}=\frac{O_{k\cdot}O_{\cdot u}}{n}.
$$

我们使用 Pearson 卡方统计量[^34]衡量对独立性的偏离,

$$
\displaystyle\chi^{2}=\sum_{k=1}^{K_{+}}\sum_{u=1}^{U}\frac{(O_{ku}-E_{ku})^{2}}{E_{ku}},
$$

其渐近零分布具有 $(K_{+}-1)(U-1)$ 自由度。将此计算应用于 MATH 训练集给出 $n=7{,}500$,$K_{+}=16$ 和 $U=7$;因此,列联表包含 $16\times 7$ 个单元格,渐近参考分布具有 $90$ 自由度。产生的统计量为 $\chi^{2}=10{,}995.84$,渐近 $p<10^{-300}$,拒绝独立性。

由于卡方统计量随样本量增加,我们额外报告 Cramér's $V$,一种用于类别关联的归一化效应量度量[^10]:

$$
\displaystyle V=\sqrt{\frac{\chi^{2}}{n\min(K_{+}-1,U-1)}}.
$$

产生的 $V=0.494$ 表明谱聚类与标注的数学学科之间存在实质性关联。该效应量分析补充了假设检验:$p$ 值确立关联在独立性假设下不太可能,而 $V$ 量化了其幅度。

### 学科类别内的难度差异

学科类别与聚类分配强关联,因此单一合并比较可能将学科组成与难度混淆。我们因此进行七个独立的 Kruskal-Wallis 检验,每个 MATH 学科类别一个[^24]。该分层询问同一学科的问题中,标注的难度级别在谱聚类间是否不同。

对于学科 $u$,令 $n_{u}$ 为其问题数量,$n_{ku}$ 为分配给非空聚类 $k$ 的数量。我们对该学科内所有 $n_{u}$ 个问题的序数难度级别进行排名,对平局分配中间排名。令 $R_{ku}$ 为聚类 $k$ 中的排名总和。未校正的 Kruskal-Wallis 统计量为

$$
\displaystyle H_{u}^{(0)}=\frac{12}{n_{u}(n_{u}+1)}\sum_{k:n_{ku}>0}\frac{R_{ku}^{2}}{n_{ku}}-3(n_{u}+1).
$$

由于 MATH 难度仅采用五个序数级别,平局很常见。令 $t_{ug}$ 为学科 $u$ 内第 $g$ 个平局组的大小。我们应用标准平局校正

$$
\displaystyle C_{u}=1-\frac{\sum_{g}(t_{ug}^{3}-t_{ug})}{n_{u}^{3}-n_{u}},\qquad H_{u}=\frac{H_{u}^{(0)}}{C_{u}}.
$$

如果 $K_{u}$ 个聚类在学科 $u$ 内非空,则原始渐近 $p$ 值从具有 $K_{u}-1$ 自由度的卡方参考分布计算:

$$
\displaystyle p_{u}=\Pr\!\left(\chi^{2}_{K_{u}-1}\geq H_{u}\right).
$$

七个原始 $p$ 值形成一个假设族。因此我们应用 Benjamini-Hochberg 程序来控制错误发现率[^5]。在将原始值排序为 $p_{(1)}\leq\cdots\leq p_{(m)}$(其中 $m=7$)后,单调调整值为

$$
\displaystyle\widetilde{p}_{(r)}=\min\left\{1,\;\min_{j\geq r}\frac{m}{j}p_{(j)}\right\}.
$$

| 学科 | 原始 $p$ | BH 调整 $p$ |
| --- | --- | --- |
| Algebra | $\approx 0$ | $\approx 0$ |
| Prealgebra | $\approx 0$ | $\approx 0$ |
| Precalculus | $\approx 0$ | $1.0\times 10^{-6}$ |
| Intermediate algebra | $2.0\times 10^{-6}$ | $3.0\times 10^{-6}$ |
| Counting/probability | $8.0\times 10^{-6}$ | $1.1\times 10^{-5}$ |
| Number theory | $3.19\times 10^{-4}$ | $3.72\times 10^{-4}$ |
| Geometry | $4.31\times 10^{-3}$ | $4.31\times 10^{-3}$ |

表 8:谱聚类间难度差异的学科内 Kruskal-Wallis 检验。显示为零的值低于报告精度;BH 校正使用未舍入的原始值。

所有七个学科特定零假设在 BH 校正后仍被拒绝,$\widetilde{p}_{u}\leq 4.31\times 10^{-3}$。因此,即使在同一学科类别的问题中,聚类成员资格也与标注的难度相关联。结合聚类-学科关联,该结果支持谱初始化捕捉语义和难度相关结构的解释。

## G. 非平稳策略下的窗口记忆

### 窗口遗忘机制

主文中描述的在线推断过程通过继承状态级 Beta 后验在训练步骤间累积 rollout 反馈,即 $a_{k,t}^{\mathrm{hist}}=a_{k,t-1}$ 和 $b_{k,t}^{\mathrm{hist}}=b_{k,t-1}$。然而,rollout 分布是非平稳的,因为策略参数在强化学习优化期间持续变化。因此,许多训练步骤之前收集的 rollout 观察可能不再准确表征当前策略的成功概率。为了减少这种过时观察的影响同时保持对当前策略的响应性,我们引入窗口遗忘机制。窗口变体保留难度感知样本图 $W$、平滑分配先验 $A$、Potts 图先验以及主方法中引入的平均场变分族。它仅改变历史 rollout 证据的组织方式:状态级成功概率从最近的滑动窗口估计,而每个先前观察样本的完整 rollout 历史被保留用于样本级预测。

#### 窗口状态级统计

在预测训练步骤 $t$ 之前,我们定义长度为 $L\geq 1$ 的前置窗口为

$$
\displaystyle\mathcal{T}_{t}^{(L)}=\left\{\tau:\max(1,t-L)\leq\tau<t\right\},
$$

其中 $L$ 表示窗口长度,与图邻接矩阵 $W$ 区分。对于潜在状态 $k$,该窗口内责任加权的成功和失败 rollout 数量为

$$
\displaystyle S_{k,t}^{(L)}=\sum_{\tau\in\mathcal{T}_{t}^{(L)}}\sum_{i\in\mathcal{O}_{\tau}}q_{ik,\tau}s_{i,\tau},
$$

和

$$
\displaystyle F_{k,t}^{(L)}=\sum_{\tau\in\mathcal{T}_{t}^{(L)}}\sum_{i\in\mathcal{O}_{\tau}}q_{ik,\tau}\left(n_{i,\tau}-s_{i,\tau}\right).
$$

窗口变体不是直接从前一步继承完整的 Beta 后验,而是从固定先验和当前窗口内的 rollout 证据重构状态级 Beta 参数:

$$
\displaystyle a_{k,t}^{(L)}=a_{0}+S_{k,t}^{(L)},\qquad b_{k,t}^{(L)}=b_{0}+F_{k,t}^{(L)}.
$$

在我们的实现中,我们设置 $a_{0}=b_{0}=1$,对应于均匀先验 $\operatorname{Beta}(1,1)$。这是 Bernoulli 成功概率的标准弱先验,在观察到 rollout 证据之前不偏向任何特定成功率。它还保证 Beta 参数保持严格正值。该性质在有限窗口下特别重要,因为某些潜在状态可能在当前窗口中未收到 rollout 观察。当 $S_{k,t}^{(L)}=F_{k,t}^{(L)}=0$ 时,$\operatorname{Beta}(1,1)$ 先验仍产生有效后验,并使变分更新中使用的后验均值和 digamma 项保持良定义和数值稳定。

理想情况下,先验均值

$$
\displaystyle\mathbb{E}[\theta_{k,t}]=\frac{a_{0}}{a_{0}+b_{0}}
$$

应接近当前策略的平均成功概率。这样的先验将为观察较少或没有最近观察的潜在状态提供更具信息性的默认估计。然而,获得当前策略平均能力的可靠估计需要在每个阶段评估训练集的足够大且有代表性的子集,这会引入大量额外的 rollout 成本。仅从调度器选择的样本估计它也可能存在偏差,因为选定的样本通常不代表完整训练分布。因此,我们采用 $\operatorname{Beta}(1,1)$ 作为先验中立性、数值稳定性和计算效率之间的实用权衡。

潜在状态 $k$ 的后验平均成功概率为

$$
\displaystyle\mu_{k,t}^{(L)}=\frac{a_{k,t}^{(L)}}{a_{k,t}^{(L)}+b_{k,t}^{(L)}}.
$$

样本 $x_{i}$ 的基于模型的预测保留与主方法相同的混合形式:

$$
\displaystyle\widehat{p}_{i,t}^{\mathrm{model}}=\sum_{k=1}^{K}q_{ik,t}\mu_{k,t}^{(L)}.
$$

当窗口覆盖所有前置训练步骤时,窗口形式化简为累积历史变体。使用有限的 $L$,早于 $L$ 个训练步骤的 rollout 观察不再直接贡献于当前状态级成功概率。#### 窗口化变分更新

窗口化变体保留平均场分解形式：

$$
\displaystyle q_{t}(\mathbf{z}_{t},\boldsymbol{\theta}_{t})\approx\prod_{i=1}^{N}q_{i}(z_{i,t})\prod_{k=1}^{K}q(\theta_{k,t}),
$$

其中 $q_{ik,t}\equiv q_{i}(z_{i,t}=k)$ 且

$$
\displaystyle q(\theta_{k,t})=\operatorname{Beta}\left(a_{k,t}^{(L)},b_{k,t}^{(L)}\right).
$$

潜在状态分配的坐标更新保留与主方法相同的三个信息来源：

$$q_{ik}^{(r+1)}=\operatorname{Softmax}_{k}\left[\log A_{ik}+\beta\sum_{j=1}^{N}W_{ij}q_{jk}^{(r)}+\mathbb{I}\left(i\in\mathcal{O}_{t}^{(L)}\right)\ell_{ik,t}^{(L)}\right],$$

其中

$$
\displaystyle\mathcal{O}_{t}^{(L)}=\bigcup_{\tau\in\mathcal{T}_{t}^{(L)}}\mathcal{O}_{\tau}.
$$

对于每个样本 $i$，我们在当前窗口内聚合其 rollout 结果：

$$
\displaystyle s_{i,t}^{(L)}=\sum_{\tau\in\mathcal{T}_{t}^{(L)}}s_{i,\tau},\qquad n_{i,t}^{(L)}=\sum_{\tau\in\mathcal{T}_{t}^{(L)}}n_{i,\tau},
$$

并计算相应的期望对数似然：

$$\ell_{ik,t}^{(L)}={}s_{i,t}^{(L)}\left[\psi\left(a_{k,t}^{(L)}\right)-\psi\left(a_{k,t}^{(L)}+b_{k,t}^{(L)}\right)\right]+\left(n_{i,t}^{(L)}-s_{i,t}^{(L)}\right)\left[\psi\left(b_{k,t}^{(L)}\right)-\psi\left(a_{k,t}^{(L)}+b_{k,t}^{(L)}\right)\right].$$

更新分配概率后，仅使用当前窗口内的 rollout 观测值重新计算状态级别的 Beta 因子：

$$
\displaystyle a_{k,t}^{(L)}=1+\sum_{\tau\in\mathcal{T}_{t}^{(L)}}\sum_{i\in\mathcal{O}_{\tau}}q_{ik}s_{i,\tau},
$$

以及

$$
\displaystyle b_{k,t}^{(L)}=1+\sum_{\tau\in\mathcal{T}_{t}^{(L)}}\sum_{i\in\mathcal{O}_{\tau}}q_{ik}\left(n_{i,\tau}-s_{i,\tau}\right).
$$

我们使用与主方法相同的停止准则交替执行分配更新和 Beta 因子更新。因此，窗口化变体不改变推断过程的坐标上升结构，仅限制了似然和状态级别 Beta 更新中使用的 rollout 证据的时间支撑范围。

#### 持久分配热启动

在累积形式中，完整的估计器状态 $\mathcal{S}_{t}=\{q_{t},a_{t},b_{t}\}$ 在连续训练步之间传递。在窗口化变体中，我们仅保留前一步的分配概率作为默认初始化：

$$
\displaystyle q_{ik,t}^{(0)}=q_{ik,t-1}.
$$

Beta 参数不直接从前一步继承。相反，它们从固定的 $\operatorname{Beta}(1,1)$ 先验和当前窗口内的 rollout 观测值重建。$q$ 的持久初始化在图结构潜在分配中保持连续性，并为有限迭代变分过程提供热启动。重要的是，继承的责任度仅用作优化初始化。它们不被视为额外的成功或失败，因此不会改变应用于状态级别充分统计量的严格时间窗口。我们还考虑了一个重置变体，该变体在每一步从静态软分配先验 $A$ 初始化 $q$；除非另有说明，默认使用持久热启动变体。

#### 持久样本级历史

滑动窗口仅应用于共享的状态级别统计量。对于每个样本 $x_{i}$，我们单独保留其在步骤 $t$ 之前的完整 rollout 历史：

$$
\displaystyle S_{i,t}^{\mathrm{all}}=\sum_{\tau<t}s_{i,\tau},\qquad C_{i,t}^{\mathrm{all}}=\sum_{\tau<t}n_{i,\tau}.
$$

对于先前观测过的样本，其历史经验成功率为：

$$
\displaystyle\overline{r}_{i,<t}=\frac{S_{i,t}^{\mathrm{all}}}{C_{i,t}^{\mathrm{all}}}.
$$

然后我们保留主方法的预测规则：

$$
\displaystyle\widehat{p}_{i,t}=\begin{cases}\widehat{p}_{i,t}^{\mathrm{model}},&C_{i,t}^{\mathrm{all}}=0,\\[4.0pt]
\gamma\widehat{p}_{i,t}^{\mathrm{model}}+(1-\gamma)\overline{r}_{i,<t},&C_{i,t}^{\mathrm{all}}>0.\end{cases}
$$

因此，最终预测的两个组件在互补的时间尺度上运行。基于模型的项使用最近的 rollout 反馈来跟踪共享潜在状态不断演化的成功概率，而样本级经验项保留每个先前观测样本所有可用的直接证据。窗口化机制因此在不丢弃与单个样本相关的长期历史经验的情况下，从可迁移的状态级别统计量中移除过时证据。

### 结果与讨论

<table><tbody><tr><td rowspan="2">基础模型</td><td rowspan="2">方法</td><td colspan="2">全部步骤</td></tr><tr><td>MAE (↓)</td><td>r (↑)</td></tr><tr><td rowspan="2">Qwen-2.5 Math-1.5B</td><td>本文方法</td><td>0.183</td><td>0.836</td></tr><tr><td>窗口化</td><td>0.127</td><td>0.712</td></tr><tr><td rowspan="2">Llama-3.2 1B-Instruct</td><td>本文方法</td><td>0.131</td><td>0.776</td></tr><tr><td>窗口化</td><td>0.119</td><td>0.747</td></tr></tbody></table>

表 9：累积估计器（本文方法）和窗口化变体（窗口化）在所有训练步骤上的样本级难度估计性能。MAE 衡量绝对预测误差，批次级别的 $r$ 表示预测成功概率与参考成功概率之间的 Pearson 相关系数。较低的 MAE 和较高的 $r$ 表示更好的性能。

![Refer to caption](https://arxiv.org/html/2608.17941v1/Figures/qwen_llama_batch_level_ours_window78.png)

图 3：累积估计器和窗口化估计器在 Qwen-2.5-Math-1.5B 和 Llama-3.2-1B-Instruct 上的批次级参考准确率与预测准确率。

我们通过重放 Qwen-2.5-Math-1.5B 和 Llama-3.2-1B-Instruct 的完整训练日志来评估窗口化变体，并将其与主实验中使用的累积估计器进行比较。如表 9 所示，窗口化变体在两个模型上均实现了更低的样本级 MAE：在 Qwen-2.5-Math-1.5B 上从 $0.183$ 降至 $0.127$，在 Llama-3.2-1B-Instruct 上从 $0.131$ 降至 $0.119$。这表明窗口化遗忘机制使每个单独样本的预测成功概率在数值上更接近其参考 rollout 结果。图 3 进一步可视化了整个训练过程中的批次级参考准确率和预测准确率。

然而，表 9 也显示，引入窗口化机制后相关性指标 $r$ 下降：在 Qwen-2.5-Math-1.5B 上从 $0.836$ 降至 $0.712$，在 Llama-3.2-1B-Instruct 上从 $0.776$ 降至 $0.747$。这表明存在权衡：虽然窗口化估计器改进了逐点数值精度，但削弱了保持样本间相对难度结构的能力。一个可能的原因是有限窗口减少了可用于状态级别估计的历史证据量，使估计值对稀疏的近期观测和采样偏差更加敏感。对于我们的难度感知调度设置，稳定的相对排序比单独的逐点校准更重要。因此，我们在主实验中保留累积估计器作为默认方法。是否使用窗口化遗忘应取决于下游目标：当优先考虑当前策略校准时它是有益的，但当稳定的样本排序至关重要时则不太适合。

[^1]: S. Bae, J. Hong, M. Y. Lee, H. Kim, J. Nam, and D. Kwak Online difficulty filtering for reasoning oriented reinforcement learning. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 700–719. Cited by: [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^2]: M. Balunovic, J. Dekoninck, I. Petrov, N. Jovanović, and M. Vechev Matharena: evaluating llms on uncontaminated math competitions. Advances in Neural Information Processing Systems 38. Cited by: [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p2.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^3]: M. Belkin, P. Niyogi, and V. Sindhwani Manifold regularization: a geometric framework for learning from labeled and unlabeled examples. Journal of Machine Learning Research 7 (85), pp. 2399–2434. External Links: [Link](http://jmlr.org/papers/v7/belkin06a.html) Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^4]: Y. Bengio, J. Louradour, R. Collobert, and J. Weston Curriculum Learning. In International Conference on Machine Learning, pp. 41–48. External Links: [Document](https://dx.doi.org/10.1145/1553374.1553380), [Link](https://mlanthology.org/icml/2009/bengio2009icml-curriculum/) Cited by: [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^5]: Y. Benjamini and Y. Hochberg Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological) 57 (1), pp. 289–300. External Links: [Document](https://dx.doi.org/10.1111/j.2517-6161.1995.tb02031.x) Cited by: [Difficulty Differences Within Subject Categories](https://arxiv.org/html/2608.17941v1#Sx11.SSx2.p3.1 "Difficulty Differences Within Subject Categories ‣ F. Discussion of Spectral Clusters ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^6]: J. Besag Spatial interaction and the statistical analysis of lattice systems. Journal of the Royal Statistical Society: Series B (Methodological) 36 (2), pp. 192–225. Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Graph Prior.](https://arxiv.org/html/2608.17941v1#Sx3.SSx4.SSSx1.p1.1 "Graph Prior. ‣ 3.4 Graph-Structured Latent Difficulty Model ‣ Method ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^7]: D. M. Blei, A. Kucukelbir, and J. D. McAuliffe Variational inference: a review for statisticians. Journal of the American Statistical Association 112 (518), pp. 859–877. External Links: [Document](https://dx.doi.org/10.1080/01621459.2017.1285773), [Link](https://doi.org/10.1080/01621459.2017.1285773), https://doi.org/10.1080/01621459.2017.1285773 Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^8]: T. Broderick, N. Boyd, A. Wibisono, A. Wilson, and M. Jordan Streaming variational bayes. In Advances in Neural Information Processing Systems, C.J. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Weinberger (Eds.), Vol. 26, pp.. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2013/file/51ef186e18dc00c2d31982567235c559-Paper.pdf) Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^9]: J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu BGE M3-Embedding: multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. External Links: 2402.03216, [Link](https://arxiv.org/abs/2402.03216) Cited by: [Models Selection](https://arxiv.org/html/2608.17941v1#Sx10.SSx2.SSSx1.p1.1 "Models Selection ‣ Discussion of Embedding Models and Difficulty-Aware Instruction ‣ E. Hyper Parameter Analysis ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^10]: H. Cramér Mathematical methods of statistics. Princeton University Press, Princeton, NJ. Cited by: [Association with Subject Categories](https://arxiv.org/html/2608.17941v1#Sx11.SSx1.p2.1 "Association with Subject Categories ‣ F. Discussion of Spectral Clusters ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^11]: G. Cui, L. Yuan, Z. Wang, H. Wang, Y. Zhang, J. Chen, W. Li, B. He, Y. Fan, T. Yu, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456. Cited by: [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p2.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [NuminaMath.](https://arxiv.org/html/2608.17941v1#Sx7.SSx6.SSSx1.p1.1 "NuminaMath. ‣ B. Details of Datasets and Training ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^12]: J. Dekoninck, N. Jovanović, T. Gehrunger, K. Rögnvaldsson, I. Petrov, C. Sun, and M. Vechev Beyond benchmarks: matharena as an evaluation platform for mathematics with llms. arXiv preprint arXiv:2605.00674. Cited by: [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p2.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^13]: W. J. Dixon and A. M. Mood The statistical sign test. Journal of the American Statistical Association 41 (236), pp. 557–566. External Links: [Document](https://dx.doi.org/10.1080/01621459.1946.10501898) Cited by: [Question and Comparison Units](https://arxiv.org/html/2608.17941v1#Sx8.SSx1.p1.1 "Question and Comparison Units ‣ C. Significance Analysis of Main Results ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^14]: Y. Fang, J. Lin, X. Fu, C. Qin, H. Shi, C. Hu, L. Pan, K. Zeng, and X. Cai How to allocate, how to learn? dynamic rollout allocation and advantage modulation for policy optimization. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 14727–14744. Cited by: [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^15]: Z. Gao, J. Kim, W. Sun, T. Joachims, S. Wang, R. Y. Pang, and L. Tan Prompt curriculum learning for efficient LLM post-training. In The Fourteenth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=zqOCacBD3P) Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p3.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p2.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p3.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^16]: S. Geman and D. Geman Stochastic relaxation, gibbs distributions, and the bayesian restoration of images. IEEE Transactions on pattern analysis and machine intelligence (6), pp. 721–741. Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^17]: A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, A. Yang, A. Fan, A. Goyal, A. Hartshorn, A. Yang, et al. The llama 3 herd of models. External Links: 2407.21783, [Link](https://arxiv.org/abs/2407.21783) Cited by: [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p1.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^18]: D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al. Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p1.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p4.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^19]: C. He, R. Luo, Y. Bai, S. Hu, Z. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, et al. Olympiadbench: a challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850. Cited by: [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p2.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^20]: D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874. Cited by: [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p2.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Do Spectral Clusters Capture Semantic and Difficulty Structure?](https://arxiv.org/html/2608.17941v1#Sx4.SSx5.p1.1 "Do Spectral Clusters Capture Semantic and Difficulty Structure? ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^21]: M. D. Hoffman, D. M. Blei, C. Wang, and J. Paisley Stochastic variational inference. Journal of machine learning research. Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [3.5 Online Mean-Field Variational Inference](https://arxiv.org/html/2608.17941v1#Sx3.SSx5.p1.1 "3.5 Online Mean-Field Variational Inference ‣ Method ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^22]: N. Jain, K. Han, A. Gu, W. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica LiveCodeBench: holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974. Cited by: [Setting](https://arxiv.org/html/2608.17941v1#Sx9.SSx1.p2.1 "Setting ‣ D. Cross-Domain Generalization to Code ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^23]: D. Kong, Q. Guo, X. Xi, W. Wang, J. Wang, X. Cai, S. Zhang, and W. Ye Rethinking the sampling criteria in reinforcement learning for llm reasoning: a competence-difficulty alignment perspective. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 31438–31446. Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p3.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p4.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [State-Level Success Model.](https://arxiv.org/html/2608.17941v1#Sx3.SSx4.SSSx2.p1.1 "State-Level Success Model. ‣ 3.4 Graph-Structured Latent Difficulty Model ‣ Method ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^24]: W. H. Kruskal and W. A. Wallis Use of ranks in one-criterion variance analysis. Journal of the American Statistical Association 47 (260), pp. 583–621. External Links: [Document](https://dx.doi.org/10.1080/01621459.1952.10483441) Cited by: [Difficulty Differences Within Subject Categories](https://arxiv.org/html/2608.17941v1#Sx11.SSx2.p1.1 "Difficulty Differences Within Subject Categories ‣ F. Discussion of Spectral Clusters ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^25]: A. Kusupati, G. Bhatt, A. Rege, M. Wallingford, A. Sinha, V. Ramanujan, W. Howard-Snyder, K. Chen, S. Kakade, P. Jain, and A. Farhadi Matryoshka representation learning. In Advances in Neural Information Processing Systems, Vol. 35. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html) Cited by: [Models Selection](https://arxiv.org/html/2608.17941v1#Sx10.SSx2.SSSx1.p2.1 "Models Selection ‣ Discussion of Embedding Models and Difficulty-Aware Instruction ‣ E. Hyper Parameter Analysis ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^26]: J. P. Lalor, H. Wu, and H. Yu Building an evaluation scale using item response theory. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, J. Su, K. Duh, and X. Carreras (Eds.), Austin, Texas, pp. 648–657. External Links: [Link](https://aclanthology.org/D16-1062/), [Document](https://dx.doi.org/10.18653/v1/D16-1062) Cited by: [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^27]: J. Li, E. Beeching, L. Tunstall, B. Lipkin, R. Soletskyi, S. Huang, K. Rasul, L. Yu, A. Q. Jiang, Z. Shen, et al. Numinamath: the largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository 13 (9), pp. 9. Cited by: [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p2.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^28]: H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe Let's verify step by step. In International Conference on Learning Representations, Vol. 2024, pp. 39578–39601. Cited by: [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p2.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^29]: C. A. McGrory, D. M. Titterington, R. Reeves, and A. N. Pettitt Variational bayes for estimating the parameters of a hidden potts model. Statistics and Computing 19 (3), pp. 329–340. Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^30]: S. Mindermann, J. M. Brauner, M. T. Razzak, M. Sharma, A. Kirsch, W. Xu, B. Höltgen, A. N. Gomez, A. Morisot, S. Farquhar, and Y. Gal Prioritized training on points that are learnable, worth learning, and not yet learnt. In Proceedings of the 39th International Conference on Machine Learning, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato (Eds.), Proceedings of Machine Learning Research, Vol. 162, pp. 15630–15649. External Links: [Link](https://proceedings.mlr.press/v162/mindermann22a.html) Cited by: [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^31]: A. Ng, M. Jordan, and Y. Weiss On spectral clustering: analysis and an algorithm. Advances in neural information processing systems 14. Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^32]: H. T. Nguyen, B. Nguyen, W. Ma, Y. Zhao, R. She, and V. A. Nguyen Adaptive rollout allocation for online reinforcement learning with verifiable rewards. In The Fourteenth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=Z5sWYACAop) Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p2.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p3.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p4.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Models Selection](https://arxiv.org/html/2608.17941v1#Sx10.SSx2.SSSx1.p4.1 "Models Selection ‣ Discussion of Embedding Models and Difficulty-Aware Instruction ‣ E. Hyper Parameter Analysis ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^33]: M. Nguyen, S. Venkatesh, H. Le, et al. SPaCe: unlocking sample-efficient large language models training with self-pace curriculum learning. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 3480–3507. Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p4.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^34]: K. Pearson On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science 50 (302), pp. 157–175. External Links: [Document](https://dx.doi.org/10.1080/14786440009463897) Cited by: [Association with Subject Categories](https://arxiv.org/html/2608.17941v1#Sx11.SSx1.p1.3 "Association with Subject Categories ‣ F. Discussion of Spectral Clusters ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^35]: R. B. Potts Some generalized order-disorder transformations. In Mathematical proceedings of the cambridge philosophical society, Vol. 48, pp. 106–109. Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Graph Prior.](https://arxiv.org/html/2608.17941v1#Sx3.SSx4.SSSx1.p1.1 "Graph Prior. ‣ 3.4 Graph-Structured Latent Difficulty Model ‣ Method ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^36]: Y. Qu, Q. Wang, Y. Mao, V. T. Hu, B. Ommer, and X. Ji Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models?. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pp. 1240–1250. Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p3.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p4.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [State-Level Success Model.](https://arxiv.org/html/2608.17941v1#Sx3.SSx4.SSSx2.p1.1 "State-Level Success Model. ‣ 3.4 Graph-Structured Latent Difficulty Model ‣ Method ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^37]: N. Reimers and I. Gurevych Sentence-BERT: sentence embeddings using siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pp. 3982–3992. External Links: [Document](https://dx.doi.org/10.18653/v1/D19-1410), [Link](https://aclanthology.org/D19-1410/) Cited by: [Models Selection](https://arxiv.org/html/2608.17941v1#Sx10.SSx2.SSSx1.p4.1 "Models Selection ‣ Discussion of Embedding Models and Difficulty-Aware Instruction ‣ E. Hyper Parameter Analysis ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^38]: P. Rodriguez, J. Barrow, A. Hoyle, J. P. Lalor, R. Jia, and J. Boyd-Graber Evaluation examples are not equally informative: how should that change NLP leaderboards?. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), C. Zong, F. Xia, W. Li, and R. Navigli (Eds.), Online, pp. 4486–4503. External Links: [Link](https://aclanthology.org/2021.acl-long.346/), [Document](https://dx.doi.org/10.18653/v1/2021.acl-long.346) Cited by: [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^39]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p1.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p2.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^40]: G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu HybridFlow: a flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256. Cited by: [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p4.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").
## 参考文献

[^41]: S. Swayamdipta, R. Schwartz, N. Lourie, Y. Wang, H. Hajishirzi, N. A. Smith, and Y. Choi Dataset cartography: mapping and diagnosing datasets with training dynamics. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), B. Webber, T. Cohn, Y. He, and Y. Liu (Eds.), Online, pp. 9275–9293. External Links: [Link](https://aclanthology.org/2020.emnlp-main.746/), [Document](https://dx.doi.org/10.18653/v1/2020.emnlp-main.746) Cited by: [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^42]: K. Team, A. Du, B. Gao, B. Xing, C. Jiang, C. Chen, C. Li, C. Xiao, C. Du, C. Liao, et al. Kimi k1. 5: scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599. Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p4.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^43]: U. Von Luxburg A tutorial on spectral clustering. Statistics and computing 17 (4), pp. 395–416. Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^44]: M. J. Wainwright and M. I. Jordan Graphical models, exponential families, and variational inference. Foundations and Trends® in Machine Learning 1 (1-2), pp. 1–305. Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^45]: H. Wang, Z. Hao, J. Luo, C. Wei, Y. Shu, L. Liu, Cheaterlin, H. Dong, and J. Chen Scheduling your LLM reinforcement learning with reasoning trees. In The Fourteenth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=V4zln7XiJj) Cited by: [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^46]: M. Wu, Z. Zhang, Q. Dong, Z. Xi, J. Zhao, S. Jin, X. Fan, Y. Zhou, H. Lv, M. Zhang, et al. Reasoning or memorization? unreliable results of reinforcement learning due to data contamination. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, pp. 33944–33952. Cited by: [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p1.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^47]: W. Xiong, C. Ye, B. Liao, H. Dong, X. Xu, C. Monz, J. Bian, N. Jiang, and T. Zhang Reinforce-ada: an adaptive sampling framework for reinforce-style llm training. arXiv e-prints, pp. arXiv–2510. Cited by: [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^48]: H. Xu, S. Chen, R. Qiu, Y. Yan, C. Luo, M. Cheng, J. He, and H. Tong Prune as you generate: online rollout pruning for faster and better rlvr. External Links: 2603.24840, [Link](https://arxiv.org/abs/2603.24840) Cited by: [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^49]: A. Yang, B. Zhang, B. Hui, B. Gao, B. Yu, C. Li, D. Liu, J. Tu, J. Zhou, J. Lin, K. Lu, M. Xue, R. Lin, T. Liu, X. Ren, and Z. Zhang Qwen2.5-math technical report: toward mathematical expert model via self-improvement. External Links: 2409.12122, [Link](https://arxiv.org/abs/2409.12122) Cited by: [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p1.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^50]: J. Yao, Y. Hao, H. Zhang, H. Dong, W. Xiong, N. Jiang, and T. Zhang Optimizing chain-of-thought reasoners via gradient variance minimization in rejection sampling and rl. Advances in Neural Information Processing Systems 38, pp. 163245–163284. Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p1.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p2.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p3.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p4.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p1.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p2.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p3.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p4.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^51]: Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, et al. Dapo: an open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems 38, pp. 113222–113244. Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p1.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^52]: Y. Zeng, Z. Sun, B. Ji, E. Min, H. Cai, S. Wang, D. Yin, H. Zhang, X. Chen, and J. Wang CurES: from gradient analysis to efficient curriculum learning for reasoning LLMs. In The Fourteenth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=QXrZ0Y3yGJ) Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p2.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p3.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p4.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [State-Level Success Model.](https://arxiv.org/html/2608.17941v1#Sx3.SSx4.SSSx2.p1.1 "State-Level Success Model. ‣ 3.4 Graph-Structured Latent Difficulty Model ‣ Method ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p1.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^53]: J. Zhang The mean field theory in em procedures for markov random fields. IEEE Transactions on signal processing 40 (10), pp. 2570–2583. Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [3.5 Online Mean-Field Variational Inference](https://arxiv.org/html/2608.17941v1#Sx3.SSx5.p1.1 "3.5 Online Mean-Field Variational Inference ‣ Method ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^54]: Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, and J. Zhou Qwen3 embedding: advancing text embedding and reranking through foundation models. External Links: 2506.05176, [Link](https://arxiv.org/abs/2506.05176) Cited by: [Models Selection](https://arxiv.org/html/2608.17941v1#Sx10.SSx2.SSSx1.p1.1 "Models Selection ‣ Discussion of Embedding Models and Difficulty-Aware Instruction ‣ E. Hyper Parameter Analysis ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p1.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^55]: C. Zheng, S. Liu, M. Li, X. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071. Cited by: [Introduction](https://arxiv.org/html/2608.17941v1#Sx1.p1.1 "Introduction ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^56]: H. Zheng, Y. Zhou, B. Bartoldson, B. Kailkhura, F. Lai, J. Zhao, and B. Chen Act only when it pays: efficient reinforcement learning for llm reasoning via selective rollouts. Advances in Neural Information Processing Systems 38, pp. 124321–124346. Cited by: [Adaptive Sampling and Rollout Allocation in RLVR](https://arxiv.org/html/2608.17941v1#Sx2.SSx1.p1.1 "Adaptive Sampling and Rollout Allocation in RLVR ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Dynamic Difficulty Estimation for Reasoning LLMs](https://arxiv.org/html/2608.17941v1#Sx2.SSx2.p1.1 "Dynamic Difficulty Estimation for Reasoning LLMs ‣ Related Work ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p2.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p3.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation"), [Implementation Details](https://arxiv.org/html/2608.17941v1#Sx4.SSx1.p4.1 "Implementation Details ‣ Experiment ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^57]: D. Zhou, O. Bousquet, T. Lal, J. Weston, and B. Schölkopf Learning with local and global consistency. In Advances in Neural Information Processing Systems, S. Thrun, L. Saul, and B. Schölkopf (Eds.), Vol. 16, pp.. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf) Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").

[^58]: X. Zhu, Z. Ghahramani, and J. D. Lafferty Semi-supervised learning using gaussian fields and harmonic functions. In Proceedings of the 20th International conference on Machine learning (ICML-03), pp. 912–919. Cited by: [H. Related work of Graph-Based Learning and Probabilistic Inference](https://arxiv.org/html/2608.17941v1#Sx13.p1.1 "H. Related work of Graph-Based Learning and Probabilistic Inference ‣ Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation").
