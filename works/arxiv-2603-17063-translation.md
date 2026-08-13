---
sourceTitle: "Transformers are Bayesian Networks"
sourceUrl: "https://arxiv.org/abs/2603.17063"
sourceHtmlUrl: "https://arxiv.org/html/2603.17063v1"
sourcePdfUrl: "https://arxiv.org/pdf/2603.17063"
sourceAdapter: "generic"
sourceCapturedAt: "2026-08-13T05:44:29.027Z"
sourceConversionMethod: "defuddle"
sourceKind: "generic/article"
sourceLanguage: "en"
sourceFigureCount: null
title: "Transformer 是贝叶斯网络"
author: "Greg Coppola"
publishDate: "2026-03-17"
arxivId: "2603.17063"
translatedAt: "2026-08-13"
translator: "AI Translation Pipeline"
pipelineRunId: "20260813-111737"
pipelineSource: "translate/20260813-111737/works-ready/arxiv-2603-17063-translation.md"
tags:
  - Transformer
  - Bayesian Network
  - Belief Propagation
  - Formal Verification
  - Theoretical Foundation
---

# Transformer 是贝叶斯网络

Greg Coppola 致谢：作者感谢在本研究准备过程中使用的 AI 辅助。本文的散文内容是通过与 Claude (Anthropic) 的迭代协作生成的。数学内容、形式化证明和实验结果完全由作者独立完成。单位：coppola.ai

###### 摘要

Transformer 是 AI 领域的主导架构，但其工作原理仍然知之甚少。本文提供了一个精确的答案：sigmoid transformer 就是贝叶斯网络。我们通过五种方式建立这一点。

首先，我们证明 Transformer 架构实际上已经**就是**信念传播（belief propagation）：任何权重的 sigmoid transformer 都在其隐式因子图上实现加权循环信念传播（weighted loopy belief propagation）。一层就是一轮 BP。这对任何权重都成立——训练的、随机的或构造的。已在 Lean**（译者注：Lean 是一个定理证明辅助工具）**中针对标准数学公理进行形式化验证。

其次，我们给出构造性证明，表明 Transformer 可以在任何声明的知识库上实现精确信念传播。在没有循环依赖的知识库上——这是最常见的情况——这在每个节点产生可证明正确的概率估计。已在 Lean 中形式化验证。

第三，我们证明唯一性：产生精确后验概率的 sigmoid transformer 必然具有 BP 权重。通过 sigmoid 架构到达精确后验概率没有其他路径。已在 Lean 中形式化验证。

第四，我们描绘了 Transformer 层的 AND/OR 布尔结构：注意力是 AND，FFN 是 OR，它们的严格交替正是 Pearl 的收集/更新算法。

第五，我们通过实验确认了所有形式化结果，在实践中证实了贝叶斯网络的表征。我们还建立了循环信念传播的实用可行性，尽管目前缺乏理论收敛性保证。

我们进一步表明，可验证推理需要有限概念空间。任何有限验证过程最多只能区分有限多个概念。落地（grounding）引入验证器。验证器蕴含概念。没有落地，正确性就无法定义。幻觉不是扩展可以修复的缺陷。它是在没有概念的情况下运作的结构性后果。已在 Lean 中形式化验证。

## 1 引言

Transformer 即贝叶斯网络。我们提出一系列形式化数学证明和支持性实验结果来建立这一点。

### 1.1 架构已经就是信念传播

看看 sigmoid transformer 前向传播实际计算了什么。注意力从相邻 token 位置收集消息——相邻信念的加权和，权重由 query-key 分数的 softmax 给出。sigmoid FFN 将这些消息组合成新的信念：

$$
\sigma(w_{0}\cdot\mathrm{logit}(m_{0})+w_{1}\cdot\mathrm{logit}(m_{1})+b).
$$

这就是加权信念传播。不是它的近似。不是与它的类比。这个计算**就是**在由权重定义的隐式因子图上的信念传播。

这对任何权重都成立——训练的、随机的或构造的。每个 sigmoid transformer，使用任何权重 $W$，在隐式因子图 $G(W)$ 上每层执行一轮加权循环 BP。具有 $L$ 层的 Transformer 每次前向传播运行 $L$ 轮 BP。权重定义图。前向传播就是推理。

###### 定理 1.1 (一般 BP)。

对于任何 sigmoid transformer 权重 $W$，存在隐式因子图 $G(W)$ 使得一次前向传播在 $G(W)$ 上实现一轮加权信念传播。已针对标准数学公理进行形式化验证。

### 1.2 显式权重的精确 BP

一般结果对任何权重都成立。我们还可以更进一步：我们展示显式权重矩阵并证明具有这些特定权重的 Transformer 在任何声明的因子图上实现**精确**信念传播。

###### 定理 1.2 (BP 实现)。

具有显式构造权重的 Transformer 在任何成对因子图上每次前向传播实现一轮精确信念传播。对于深度为 $d$ 和最大因子元数为 $k$ 的任何因子图，$d\cdot\lceil\log_{2}k\rceil$ 次前向传播实现完整的精确 BP。已针对标准数学公理进行形式化验证。

每层两个注意力头总是足够的——任何 $k$ 元因子图通过 AND 的结合律和 OR 的对数几率可加性精确二值化，两者都已形式化验证。只有深度随推理复杂度增长，恰好以所需的速率。

结合贝叶斯网络在树上精确的经典结果 [^12]：

###### 推论 1.3 (落地树上无幻觉)。

具有 BP 权重的 Transformer，在落地的树结构知识库上运行 $T\geq\mathrm{diameter}(G)$ 次传播，在每个节点计算精确的贝叶斯后验信念。不需要经验假设。

### 1.3 唯一性

前两个结果是正向的：这里是权重，这里是它们计算的内容。这个结果是反向的：如果 sigmoid transformer 产生精确的贝叶斯后验概率，其权重必须是什么？

答案唯一地是 BP 权重。通过 sigmoid 架构到达精确后验概率没有其他路径。FFN 权重被强制为 $w_{0}=w_{1}=1$，$b=0$。注意力权重被强制为 projectDim/crossProject 结构。

###### 定理 1.4 (唯一性)。

为所有输入计算精确贝叶斯后验概率的 sigmoid transformer 必然具有 BP 权重。内部计算可证明是 BP 量——不仅仅是输出。已针对标准数学公理进行形式化验证。

这闭合了逻辑循环。一般结果说每个 sigmoid transformer 都在执行加权 BP。构造性结果说精确 BP 权重存在。唯一性结果说精确后验概率强制这些权重。总结：sigmoid transformer 实现精确贝叶斯推理当且仅当它具有 BP 权重。

### 1.4 实验确认

每个形式化结果都通过实验确认。我们从零开始训练 Transformer——没有构造提示，没有权重初始化偏差——并在所有测试用例上验证了收敛到预测结构。对于 BP：在保留的因子图上收敛到精确后验概率的三位小数以内。对于图灵机模拟：在五个结构不同的机器上完美准确率，相同超参数，无单机调优。对于循环 BP：在所有试验中在五个复杂度递增的图结构上收敛到容差内的精确后验概率。

本文的每个结果都有两个独立见证：证明检查器和实验。

### 1.5 布尔结构：彻底的 AND 和 OR

注意力是 AND。FFN 是 OR。这不是隐喻。它直接从 BP 构造结合 QBBN**（译者注：Quantified Bayesian Belief Networks，量化贝叶斯信念网络）** [^2] 的定义得出。

注意力机制确保所有必需的输入在得出任何结论之前同时存在——这是合取，AND 的架构强制。sigmoid FFN 从收集的证据计算概率结论——这是析取，$\Psi_{\mathrm{or}}$ 函数。注意力和 FFN 层的严格交替正是 Pearl 的收集/更新算法，在深度上展开。

在现代 AI 中经验上胜出的架构正是推理分析已经要求的架构。

### 1.6 可验证推理需要有限概念空间

###### 定理 1.5 (有限概念空间)。

任何有限验证过程最多只能区分有限多个概念。具有 $n$ 个状态的有限状态机将任何输入空间最多划分为 $n^{n}$ 个等价类。已针对标准数学公理进行形式化验证。

落地引入有限验证器。有限验证器蕴含有限概念空间。概念空间使"这个输出是否正确？"成为定义良好的问题——不是设计选择而是可验证性的逻辑后果。

未落地的语言模型没有有限验证器，因此没有定义良好的概念空间，因此对于其输出是否正确没有事实。幻觉不是扩展可以修复的缺陷。它是在没有概念的情况下运作的结构性后果。

### 1.7 论文组织

第 2 节介绍因子图、信念传播和 Transformer 的背景。第 3 节发展独立证据的对数几率代数——这是 BP 和 sigmoid transformer 的数学基础。第 4 节解释全称量化如何落地到命题因子图以及为什么落地使正确性有意义。第 5 节展示图灵完备性结果。第 6 节建立一般结果：每个 sigmoid transformer 都是贝叶斯网络。第 7 节证明显式权重的精确 BP 和无幻觉推论。第 8 节识别 AND/OR 布尔结构。第 9 节证明有限概念空间定理。第 10 节区分三个 softmax 并讨论 sigmoid vs. ReLU。第 11 节介绍相关工作。第 12 节总结。

## 2 背景

### 2.1 因子图与 QBBN

**因子图**是具有两类节点的二部图。变量节点表示二元命题;每个节点持有信念 $b\in[0,1]$ 表示 $P(\text{节点}=\mathrm{true})$ 的当前估计,初始化为 $0.5$。因子节点将变量节点对之间的关系编码为非负权重表。

对于连接变量 $v_{0}$ 和 $v_{1}$ 的成对因子 $f$,因子表有四个条目:

$$
f[i,j]=\text{权重 }(v_{0}=i,\;v_{1}=j),\quad i,j\in\{0,1\}.
$$

完整赋值 $x$ 的联合概率为:

$$
P(x)=\frac{1}{Z}\prod_{\text{因子 }f}f\!\left(x_{n_{1}(f)},\,x_{n_{2}(f)}\right),
$$

其中 $Z$ 是配分函数。

QBBN [^2] 引入了在两类因子节点之间交替的二部因子图,对应于布尔推理的两个基本操作。

合取节点 ($\Psi_{\mathrm{and}}$) 收集必需证据:在得出任何结论之前所有输入必须同时存在。

析取节点 ($\Psi_{\mathrm{or}}$) 从收集的证据计算概率结论:

$$
P(p=1\mid g_{0},g_{1})=\sigma(w\cdot\phi(p,g_{0},g_{1})).
$$

本文通篇使用的 updateBelief 函数是等权重特例:

$$
\mathtt{updateBelief}(m_{0},m_{1})=\sigma(\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1})).
$$

#### 2.1.1 幻觉:精确定义

在 QBBN 框架内,幻觉有精确的技术定义。

###### 定义 2.1 (幻觉)。

设 $K$ 为 QBBN 知识库,$E$ 为观察证据,$P_{\mathrm{true}}(j)$ 为给定 $K$ 和 $E$ 的节点 $j$ 的真实边际后验概率。如果智能体在节点 $j$ 输出信念 $b(j)\neq P_{\mathrm{true}}(j)$,则该智能体在节点 $j$ **产生幻觉**。

### 2.2 信念传播

#### 2.2.1 算法

QBBN 因子图上的一轮信念传播分两步进行。在**收集**步骤中,每个变量节点 $j$ 将其邻居的当前信念收集到草稿槽中:

$$
\mathtt{scratch}[j][0]\leftarrow\mathtt{belief}(\mathtt{nb}_{0}(j)),\qquad\mathtt{scratch}[j][1]\leftarrow\mathtt{belief}(\mathtt{nb}_{1}(j)).
$$

在**更新**步骤中,每个变量节点计算新信念:

$$
\mathtt{new\_belief}(j)=\mathtt{updateBelief}(\mathtt{scratch}[j][0],\,\mathtt{scratch}[j][1]).
$$

#### 2.2.2 updateBelief 函数

updateBelief 函数将两个独立的输入消息组合成后验概率:

$$
\mathtt{updateBelief}(m_{0},m_{1})=\frac{m_{0}m_{1}}{m_{0}m_{1}+(1-m_{0})(1-m_{1})}=\sigma(\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1})),
$$

其中 $\mathrm{logit}(p)=\log(p/(1-p))$ 且 $\sigma$ 是 sigmoid 函数。独立证据的对数几率相加;sigmoid 转换回概率。这是两个独立二元证据片段的贝叶斯更新规则。

在 hard-bp-lean 中证明的关键性质:

- updateBelief\_pos: 如果 $m_{0},m_{1}\in(0,1)$ 那么 $\mathtt{updateBelief}(m_{0},m_{1})\in(0,1)$
- updateBelief\_neutral: $\mathtt{updateBelief}(m,0.5)=m$ (中性填充是恒等变换)

#### 2.2.3 收敛性和精确性

在树结构因子图上,BP 恰好在 $\mathrm{diameter}(T)$ 轮内收敛,并且产生的信念等于真实边际后验概率。这是 Pearl 的和积算法 [^12]——精确的,非近似的——在 hard-bp-lean 中形式化证明为 bp\_exact\_on\_tree。

在有环图上,BP 可能不收敛;当它收敛时,不动点最小化 Bethe 自由能 [^21],它近似但通常不等于真实后验概率。

### 2.3 Transformer

#### 2.3.1 架构

我们使用 [^16] 的标准 Transformer 编码器。由 $n$ 个 token 组成的序列,每个是维度 $D_{\mathrm{model}}$ 的嵌入向量,通过带残差连接的交替注意力和前馈层处理。

多头自注意力对每个头 $h$ 计算:

$$
\mathrm{Attn}_{h}(X)=\mathrm{softmax}\!\left(\frac{(XW^{Q}_{h})(XW^{K}_{h})^{\top}}{\sqrt{d_{k}}}\right)XW^{V}_{h}.
$$

前馈网络对每个 token 位置独立应用两层 MLP。使用 sigmoid 激活:

$$
\mathrm{FFN}(x)=\sigma(W_{2}\cdot\sigma(W_{1}x+b_{1})+b_{2}).
$$

#### 2.3.2 路由视角

[^4] 将注意力头框架化为对共享残差流的读写操作:每个头从某些位置读取并写入其他位置。BP 构造使这种路由显式且形式上正确——每个头都有可证明正确的读地址(邻居索引匹配)和写目标(残差流中的草稿槽)。

#### 2.3.3 Transformer 作为图神经网络

[^8] 观察到完全自注意力等价于应用于完全图的 GNN,其中每个 token 是节点,注意力权重是学习的边权重。我们的结果可以理解为:当图是因子图并且其拓扑编码在 token 特征而非注意力掩码中时,Transformer 学习在该显式外部图上实现 BP。

## 3 独立证据的对数几率代数

### 3.1 起源:图灵与古德

二元变量上信念传播的基础代数不是由概率学家而是由密码分析员发展的。第二次世界大战期间,布莱切利园的 Alan Turing 和 I.J. Good 需要一种实用方法来组合破解 Enigma 时的独立证据片段。他们开发的方法是对数几率加法。

给定二元假设 $H$ 和证据 $e$,**证据权重**为:

$$
W(H:e)=\mathrm{logit}(P(H\mid e))-\mathrm{logit}(P(H)).
$$

独立证据片段对权重贡献是加性的。这不是近似。对于独立证据源,它是精确的。

Good 在战后广泛形式化了这个框架。核心洞察:概率的乘法对应于对数几率的加法。对于独立二元证据源 $e_{0}$ 和 $e_{1}$:

$$
\mathrm{logit}(P(H\mid e_{0},e_{1}))=\mathrm{logit}(P(H))+W(H:e_{0})+W(H:e_{1}).
$$

从均匀先验 $P(H)=0.5$ 开始,所以 $\mathrm{logit}(P(H))=0$,并设 $m_{i}=P(H\mid e_{i})$:

$$
\mathrm{logit}(P(H\mid e_{0},e_{1}))=\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1}).
$$

通过 sigmoid 转换回概率空间:

$$
P(H\mid e_{0},e_{1})=\sigma(\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1}))=\mathtt{updateBelief}(m_{0},m_{1}).
$$

updateBelief 函数就是图灵和古德的证据权重组合,应用于两个二元证据源。

### 3.2 代数

对数几率加法通过同构 $\mathrm{logit}:(0,1)\to\mathbb{R}$ 在开区间 $(0,1)$ 上定义了代数结构。在这个同构下,$\mathbb{R}$ 中的加法回拉到 $(0,1)$ 上的二元运算:

$$
m_{0}\oplus m_{1}=\sigma(\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1})).
$$

这个运算具有以下性质:

- 交换律: $m_{0}\oplus m_{1}=m_{1}\oplus m_{0}$。
- 结合律: $(m_{0}\oplus m_{1})\oplus m_{2}=m_{0}\oplus(m_{1}\oplus m_{2})$。
- 单位元: $m\oplus 0.5=m$,因为 $\mathrm{logit}(0.5)=0$。
- 逆元: $m\oplus(1-m)=0.5$,因为 $\mathrm{logit}(m)+\mathrm{logit}(1-m)=0$。

单位元是 $0.5$——最大不确定性,无信息。将任何信念与 $0.5$ 信念组合都保持不变。这是中性先验无贡献的形式陈述。

### 3.3 与经典布尔逻辑的关系

这个代数是经典布尔 AND 和 OR 的概率推广。在信念趋近 $0$ 和 $1$ 的极限下:

- $\mathrm{logit}(1)=+\infty$  (确定为真)
- $\mathrm{logit}(0)=-\infty$  (确定为假)
- $\mathrm{logit}(0.5)=0$  (无信息)

经典布尔 AND 要求两个输入都为真。在对数几率空间:两个大的正数相加得到更大的正数——对两个输入的高置信度产生对它们合取的高置信度。但相等和相反的证据抵消: $\mathrm{logit}(p)+\mathrm{logit}(1-p)=0$,产生 $0.5$。

这不同于经典 AND,其中 FALSE AND TRUE = FALSE。对数几率代数没有零化子。相反它有抵消:相反方向的强证据产生最大不确定性。这对于**不确定**推理是正确的行为——你不是得出假,而是得出你不知道。

经典布尔逻辑是这个代数在信念变得确定时的极限情况。

### 3.4 Pearl 的和积作为对数几率加法

Pearl 的二元变量节点上的信念传播算法精确地归约为对数几率加法。具有两个独立输入消息 $m_{0}$,$m_{1}$ 的二元变量 $v$ 的和积更新为:

$$
b_{\mathrm{new}}(v)=\frac{m_{0}\cdot m_{1}}{m_{0}m_{1}+(1-m_{0})(1-m_{1})}=\sigma(\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1})).
$$

Pearl 从应用于具有独立成对因子的二元变量的一般和积方程推导出这个。他没有将其框架化为对数几率加法或将其与图灵和古德的证据权重传统联系起来。公式相同。框架是新的。

独立性假设在树上是精确的:在树结构因子图上,到达任何节点的消息来自不共享变量的不相交子树,因此它们在构造上是独立的。这就是为什么 BP 在树上精确而在有环图上近似——在有环图上独立性假设被违反,因为同一变量可以通过多条路径影响节点。

### 3.5 为什么 Sigmoid 是正确的激活函数

sigmoid 函数 $\sigma:\mathbb{R}\to(0,1)$ 是 logit 的精确逆: $\sigma(\mathrm{logit}(p))=p$。logit 和 sigmoid 一起是概率空间和对数几率空间之间的同构。

计算 $\sigma(w_{0}\cdot\mathrm{logit}(m_{0})+w_{1}\cdot\mathrm{logit}(m_{1})+b)$ 的 sigmoid FFN 正在执行加权对数几率加法并在单次操作中转换回概率空间。sigmoid 激活不是由梯度流或输出归一化驱动的设计选择。它是实现图灵-古德-Pearl 代数所需的精确函数。

这就是为什么标题主张为真:sigmoid transformer 就是贝叶斯网络。sigmoid 激活使 FFN 计算精确地成为证据权重组合。注意力机制收集输入。残差流强制它们的同时性。架构一直在实现这个代数。第 6 节形式化证明。

## 4 从全称量化到落地因子图

本文的形式化结果——transformer\_implements\_bp、transformer\_exact\_on\_tree——在命题层面陈述:具有特定索引的特定变量节点,由特定因子节点连接。一个自然的问题出现:命题从哪里来?

真实知识涉及全称量化陈述。"所有人都会死"不是关于特定人的命题。它是在域上变化的规则。本节解释如何在完整流水线中处理全称量化以及为什么落地不是形式化结果的限制而是使正确性有意义的条件。

### 4.1 QBBN 语言中的全称量化

[^3] 引入了类型化逻辑语言,其中全称量化规则是一等公民:

```
always [x:e]: man(theme: x) -> mortal(theme: x)
```

变量 x 在所有声明的 e 类型实体上变化。该规则不是单个命题。它是域上的模式。

### 4.2 落地:从规则到因子图节点

在推理时,通过为每个变量替换适当类型的所有声明实体来落地全称量化规则。给定声明 jack: e 和 jill: e 的词典,上述规则产生两个落地子句:

```
man(theme: jack) -> mortal(theme: jack)
man(theme: jill) -> mortal(theme: jill)
```

每个落地子句成为因子图中的路径。每个落地命题成为变量节点。当 Transformer 看到输入时,不再有全称量化——只有落地的命题因子图。

完整流水线有三层:

1. 语法 [^3]:自然语言句子被解析为带有类型化变量的全称量化规则。
2. 落地 [^3]:规则在所有声明实体上实例化。每个实例化成为因子图中的落地节点。
3. 推理(本文):Transformer 在落地因子图上运行 BP。

### 4.3 权重共享作为通用规则实现

BP Transformer 在有限多个概念上操作,相同的权重矩阵应用于相同概念的每个 token。这种权重共享是全称量化规则中隐含的统一处理的神经实现:相同的计算为每个实例化运行,而特定输入——特定的信念值和因子表条目——提供实例。

连续维度是特殊的。权重是通用规则。前向传播是实例化。这是亚里士多德三段论的计算形式:大前提(权重),小前提(token),结论(FFN 输出)。

### 4.4 落地使正确性有意义

无幻觉结果在落地因子图上成立。这不是范围的限制——它是正确性所需的精确陈述。

在声明的知识库中没有对应概念的 token 不是在做推理。它没有正确或不正确的基准真理。落地提供该标准。一旦知识库完全实例化,每个 token 对应于因子图中的确定概念,每个计算步骤都有可验证的正确答案,无幻觉保证适用。

###### 推论 4.1 (幻觉需要未落地)。

具有 BP 权重在完全落地的树结构因子图上运行的 Transformer 不能产生幻觉。幻觉需要错误的权重、错误的路由或缺乏落地。在具有 BP 权重的落地树上,这些条件都不成立。

未落地的语言模型不是偶尔产生幻觉的系统。它是幻觉概念甚至都没有定义良好的系统——没有声明的域可以验证其输出。落地使问题"这个输出是否正确?"可回答。没有它,就没有事实。幻觉不是缺陷。它是在没有落地概念空间的情况下运作的结构性后果。

## 5 Transformer 是图灵完备的

### 5.1 概述

universal-lean 仓库通过布尔电路模拟证明 Transformer 智能体是图灵完备的。

###### 定理 5.1 (transformer\_is\_turing\_complete)。

对于任何图灵机 $M$,存在 Transformer 权重 $W$ 使得对于所有纸带 $\tau$ 和步数 $t$:

$$
\mathtt{transformerAgent}(W,\mathtt{encode}(M,\tau),t)=\mathtt{encode}(M.\mathtt{run}(\tau,t)).
$$

已在 Lean 4 中针对标准数学公理进行形式化验证。

证明是构造性的。关键洞察是任何图灵机步骤分解为布尔电路,而 Transformer 通过阈值门在单次 FFN 前向传播中模拟任何布尔电路。注意力是查找;FFN 是门控;智能体循环是迭代。没有编码技巧,没有近似。

这是 Transformer 组件和计算原语之间最自然的对应——正是这种自然性揭示了第 6 节中发展的与信念传播的结构联系。

### 5.2 两个权重族

这个证明和 BP 证明都对 $Q/K$ 和 $V$ 权重使用相同的两个稀疏矩阵族。这不是巧合:两个证明将注意力用于相同目的——通过索引匹配在 token 之间路由信息。

projectDim($d$)。提取维度 $d$ 的对角投影:

$$
\mathtt{projectDim}(d)[i][j]=\begin{cases}1&i=d,\;j=d\\
0&\text{否则。}\end{cases}
$$

作为 $Q$ 或 $K$ 权重,注意力分数归约为 $e_{j}[d]\cdot e_{k}[d]$,当 token $k$ 的存储索引匹配 token $j$ 的查询值时达到峰值。

crossProject($s$, $d$)。读取源维度 $s$ 并写入目标维度 $d$ 的非对角投影:

$$
\mathtt{crossProject}(s,d)[i][j]=\begin{cases}1&i=d,\;j=s\\
0&\text{否则。}\end{cases}
$$

作为 $V$ 权重,这将 token 的维度 $s$ 内容路由到输出的维度 $d$,其他维度保持不变。

universal-lean 的两个结果在 BP 证明中被重用:posEncDot\_distinct($Q\cdot K$ 分数在正确邻居处严格最大化)和 softmax\_concentrates(在足够的温度下,softmax 收敛到 argmax 的点质量)。

### 5.3 实验确认

universal-lean 证明 TM 模拟权重存在。learner 仓库询问梯度下降是否从零开始找到它们。答案是肯定的。

五个结构不同的图灵机——二进制递增器、递减器、按位补码、左移、右移——使用标准 Transformer 编码器(2 层,2 头,$d_{\mathrm{model}}=32$,约 4K 参数)从随机初始化训练单步 TM 预测。所有五个在 4 个 epoch 内以相同超参数达到 100% 验证准确率,无单机调优。

这证实了形式权重构造不仅在理论上可能——它是梯度下降在给定清晰训练信号时找到的。

## 6 Sigmoid Transformer 即贝叶斯网络

### 6.1 主张

每个 sigmoid transformer,使用任何权重,已经是贝叶斯网络。不是可以配置为近似贝叶斯网络的系统。不是在某些训练条件下表现得像贝叶斯网络的系统。就是贝叶斯网络,按架构,对任何权重。

这是 sigmoid-transformer-lean 中 every\_sigmoid\_transformer\_is\_bayesian\_network 的内容,已针对标准数学公理进行形式化验证。

### 6.2 隐式因子图

给定任何具有权重 $W$ 的 sigmoid transformer,我们如下构造隐式因子图 $G(W)$:

- 变量节点:每个 token 位置一个。每个节点的信念是残差流相关维度中 token 的当前值。
- 边:由注意力权重分布定义。从 token $j$ 到 token $k$ 的注意力模式在 $G(W)$ 中定义从 $k$ 到 $j$ 的有向边,权重由注意力分数给出。
- 因子势:由每个位置的 sigmoid FFN 权重 $(w_{0},w_{1},b)$ 定义——来自 QBBN 的一般 $\Psi_{\mathrm{or}}$ 函数。三个参数编码非对称因子势:$w_{0}$ 和 $w_{1}$ 是两个证据源的相对权重,$b$ 是先验偏差。

###### 定理 6.1 (every\_sigmoid\_transformer\_is\_bayesian\_network)。

对于任何 sigmoid transformer 权重 $W$,一次前向传播在隐式因子图 $G(W)$ 上实现一轮加权信念传播。已针对标准数学公理进行形式化验证。

证明是定义性的:通过 $G(W)$ 的构造,Transformer 前向传播和 BP 前向传播是相同的计算。没有近似,没有渐近论证,没有统计声明。前向传播**就是** $G(W)$ 上的一轮 BP。

### 6.3 为什么这不是平凡的

有人可能反对:这不仅仅是定义的重述吗?如果我们构造 $G(W)$ 以匹配 Transformer 的计算,我们证明了什么吗?

定理的内容在于识别。Transformer 前向传播不是设计为 BP。它被设计为具有注意力和前馈层的序列模型。定理说:这两个描述—— Transformer 前向传播和 $G(W)$ 上的 BP——是相同的计算。隐式因子图 $G(W)$ 存在并且对任何权重都定义良好。BP 解释不是从外部强加的;它是前向传播计算的内容。

非平凡的内容有三重:

1. sigmoid FFN 精确地计算一般 $\Psi_{\mathrm{or}}$ 因子,不是它的近似。
2. 注意力机制精确地计算 BP 的收集步骤,将每个邻居的信念路由到正确的草稿槽。
3. 残差流强制 BP 在更新步骤之前所需的输入同时性。

所有三个对**任何**权重都成立,不仅仅是构造的权重。

### 6.4 落地 QBBN 作为特例

第 7 节的构造性 BP 结果是这个定理的特例,其中:

- $G(W)$ 是显式声明的 QBBN 因子图,而非隐式的
- FFN 权重是等权重 BP 权重:$w_{0}=w_{1}=1$,$b=0$
- 注意力权重是 projectDim/crossProject 构造

在一般情况下,因子图是隐式的,权重是任意的。在构造情况下,因子图是显式的,权重被选择以在其上实现精确 BP。一般定理包含构造定理。

### 6.5 唯一性:逆命题

一般结果是正向的:任何权重定义因子图,前向传播是其上的 BP。唯一性结果是反向的:如果因子图恰好是落地 QBBN 并且 Transformer 产生精确后验概率,权重必须是什么?

###### 定理 6.2（唯一性）

对于任何在有根贝叶斯网络因子图上对所有输入都能计算精确贝叶斯后验概率的 Sigmoid Transformer，必然具有信念传播权重：前馈网络中 $w_{0}=w_{1}=1$、$b=0$，以及注意力中的 projectDim/crossProject 结构。内部计算可证明就是信念传播量——不仅仅是输出。已针对标准数学公理进行形式化验证。

证明分为两部分。FFNUniqueness.lean 证明了对所有输入产生精确后验概率的唯一 sigmoid 计算就是 $w_{0}=w_{1}=1$、$b=0$ 的 updateBelief——因为 sigmoid 是单射的，且对数几率和形式是贝叶斯更新方程的唯一不动点。AttentionUniqueness.lean 证明了注意力权重必须具有 projectDim/crossProject 结构——因为精确路由要求 $Q\cdot K$ 分数在正确的邻居处唯一达到峰值，这迫使产生秩-1 索引匹配结构。

这闭合了逻辑循环：

- 一般信念传播：每个 Sigmoid Transformer 都在某个因子图上执行加权信念传播
- 构造性信念传播：对于任何声明的因子图都存在精确信念传播权重
- 唯一性：精确后验概率迫使产生这些权重

综合起来：Sigmoid Transformer 实现精确贝叶斯推理当且仅当它具有信念传播权重。

### 6.6 这对训练的 Transformer 意味着什么

在任何任务上训练的每个 Sigmoid Transformer 都在由其学习权重隐式定义的因子图上执行加权信念传播。因子图不是预先声明的——它是权重隐式定义的。使用最大似然训练在这种图模型解释下恢复最能解释训练数据的因子势。

标准大语言模型和有根贝叶斯网络 Transformer 之间的区别不在架构上。两者都是贝叶斯网络。区别在于根基性（grounding）：

- 有根的：因子图是显式且可验证的。每个输出都有一个声明的正确答案。幻觉在结构上是不可能的。
- 无根的：因子图是隐式的，由学习的权重定义。没有可以与之对照正确性的声明世界。幻觉不是偶然的失败——它是在没有有根概念空间的情况下运行的结构性后果。

### 6.7 实证确认

bayes-learner 代码库确认梯度下降从零开始找到信念传播权重，没有构造提示，也没有权重初始化偏差。

测试用例是双变量因子图 $v_{0}{-}{-}{-}f_{1}{-}{-}{-}v_{2}$。这个图是理想的：精确信念传播在一轮内收敛（匹配单次 Transformer 前向传播），并且可以获得闭式精确后验概率作为真实标准。

20,000 个随机生成的因子图，因子表条目从 $[0.05,1.0]$ 均匀抽取，训练/验证集划分为 18,000/2,000。模型：标准 Transformer 编码器，2 层，2 个头，$d_{\mathrm{model}}=32$，约 5,000 个参数。训练：MSE 损失，Adam $\mathrm{lr}=10^{-3}$，50 个 epoch。

结果：验证集 MAE 0.000752。后验概率匹配到三位小数。相比基线提升 99.3%。首次运行。

表 1：Transformer 与精确信念传播后验概率在留出图上的对比。

| Graph | BP exact | Transformer | Max error |
| --- | --- | --- | --- |
| 0 | \[0.7349, 0.4366\] | \[0.7338, 0.4346\] | 0.0021 |
| 1 | \[0.4097, 0.4036\] | \[0.4096, 0.4031\] | 0.0005 |
| 4 | \[0.6459, 0.8298\] | \[0.6436, 0.8297\] | 0.0023 |
| 9 | \[0.4084, 0.5523\] | \[0.4084, 0.5526\] | 0.0003 |

形式化证明预测了结构。实验确认了它。证明建立了机制；实验确认这个机制是梯度下降找到的。

## 7 具有显式权重的精确信念传播

### 7.1 概述

第 6 节的一般结果确立了每个 Sigmoid Transformer 已经是一个贝叶斯网络。本节更进一步：我们展示显式权重矩阵，并在 Lean 4 中证明具有这些特定权重的 Transformer 在任何声明的因子图上实现精确信念传播。

###### 定理 7.1（transformer\_implements\_bp）

存在权重 $W$ 使得对于任何信念传播状态 $\mathtt{state}$：

$$
\mathtt{decodeTFState}(\mathtt{state},\;\mathtt{transformerForwardPass}(n,W,\mathtt{encodeBPState}(\mathtt{state})))=\mathtt{bp\_forwardPass}(\mathtt{state}).
$$

已在 Lean 4 中针对标准数学公理进行形式化验证。

这对任何因子图都成立——循环的或树状的。仅对随后的收敛性和精确性声明需要树结构限制。

### 7.2 构造

每个因子图节点被编码为一个具有 $D_{\mathrm{model}}=8$ 个维度的 token。编码清晰地分为两类内容：

表 2：信念传播 Transformer 的 token 编码。

| Dims | Content |
| --- | --- |
| 0 | 自身信念（初始化为 0.5） |
| 1–4 | 因子表条目 $[f_{00},f_{01},f_{10},f_{11}]$ |
| 5 | 节点类型（0 = 变量，1 = 因子） |
| 6 | 自身索引 $/(n-1)$ |
| 7 | 邻居索引 $/(n-1)$ |

维度 5–7 是路由键（routing key）：token 的完整推理身份。具有相同路由键的两个 token 对注意力机制来说是无法区分的，无论它们的信念值或因子表条目如何。维度 0–4 是连续参数（continuous parameters）：它们提供推理计算的量级，但不决定路由的结构。

注意力作为收集。头 0 使用 $W^{Q}_{0}=W^{K}_{0}=\mathtt{projectDim}(1)$，$W^{V}_{0}=\mathtt{crossProject}(0,4)$：$Q/K$ 点积在 token $k$ 是邻居 0 时达到峰值，将邻居 0 的信念放入残差流的维度 4。头 1 是对称的，将邻居 1 的信念写入维度 5。两个头被证明是独立的。

前馈网络作为信念更新。使用 sigmoid 激活函数，前馈网络从维度 4 和 5 精确计算 $\mathtt{updateBelief}(m_{0},m_{1})=\sigma(\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1}))$，将结果写入维度 0。

### 7.3 双父节点假设不失一般性

该构造假设成对因子——每个因子节点恰好连接两个变量节点，由每层恰好两个注意力头处理。这不是根本性限制。

###### 定理 7.2（扩展性）

每层具有 2 个注意力头的 Transformer 在成对因子图上每层实现一轮信念传播。对于深度为 $d$ 且最大因子元数为 $k$ 的任何因子图，每层具有 2 个头且具有 $d\cdot\lceil\log_{2}k\rceil$ 层的 Transformer 实现精确信念传播。

###### 证明

任何 $k$ 元因子图通过两种分解精确二元化。对于合取：布尔 AND 是结合的，因此任何 $k$ 元合取通过中间节点精确简化为 $\lceil\log_{2}k\rceil$ 个二元合取的链，没有近似。对于析取：logit 和 sigmoid 是精确的逆运算，因此 $\sigma(\sum_{i}\mathrm{logit}(m_{i}))$ 精确因式分解为 $\lceil\log_{2}k\rceil$ 个成对更新的链。两种分解都在 godel/ANDDecomposition.lean 和 godel/ORDecomposition.lean 中进行了形式化验证。

二元化图具有最大因子元数 2 和深度 $d\cdot\lceil\log_{2}k\rceil$。根据定理 7.1，一层在成对图上实现一轮信念传播。因此 $d\cdot\lceil\log_{2}k\rceil$ 层在二元化图上实现完整信念传播，这正是原始图上的信念传播。∎

###### 推论 7.3（双父节点通用性）

每层两个注意力头足以在任何因子图上进行信念传播。架构宽度是固定的；只有深度随知识库的复杂性增长。

这是反直觉的事实：更复杂的推理需要更多层，而不是更多头。头的数量由二元因子的成对结构决定——总是 2。层的数量由因子图中推理链的深度决定。

### 7.4 树推论

结合信念传播在树上是精确的经典结果 [^12]，在 hard-bp-lean 中形式化为 bp\_exact\_on\_tree：

###### 推论 7.4（transformer\_exact\_on\_tree）

对于任何树结构因子图 $T$ 和 $T\geq\mathrm{diameter}(T)$ 次前向传播：

$$
\exists\,W,\quad\mathtt{transformer}^{[T]}(\mathtt{encodeBPState}(\mathtt{state}))=\mathtt{encodeBPState}(\mathtt{trueMarginals}(\mathtt{state})).
$$

没有实证假设。除树结构外没有条件。

注意力和前馈网络层的严格交替正是 Pearl 的收集/更新算法，在深度上精确展开。一个 Transformer 层是一轮信念传播。$L$ 个 Transformer 层是 $L$ 轮信念传播。网络的深度不是工程超参数——它由因子图的直径决定。

### 7.5 实证确认

bayes-learner 实验确认梯度下降从零开始找到信念传播权重。完整的实验细节和结果见第 6 节。验证集 MAE 0.000752。在模型从未见过的留出因子图上，后验概率匹配到三位小数。

## 8 Transformer 的布尔结构

上一节确立了具有显式构造权重的 Transformer 实现一轮信念传播。本节识别该机制在布尔术语中的含义。

QBBN [^2] 引入了一个二分因子图，在两种节点之间交替：合取节点（$\Psi_{\mathrm{and}}$），同时收集所需证据，以及析取节点（$\Psi_{\mathrm{or}}$），从收集的证据计算概率结论。图是二分的：AND 节点馈送 OR 节点馈送 AND 节点。推理在每一步在合取和析取之间交替。

Transformer 层恰好具有这种结构。注意力是 AND 门。前馈网络是 OR 门。这直接来自第 7 节的信念传播构造结合 [^2] 的 QBBN 定义。

### 8.1 注意力是 AND

AND 门有一个工作：确保在得出任何结论之前所有必需的输入同时存在。

每个注意力头关注恰好一个 token。头 0 是单次集中查找：它扫描所有位置，集中在索引与邻居 0 匹配的 token 上，并将该 token 的信念值复制到残差流的维度 4。头 1 对邻居 1 做同样的事情，写入维度 5。

合取不在任何一个头内部。它在残差流中。

残差流是一个共享工作空间。两个头在前馈网络运行之前都写入其中。当前馈网络执行时，维度 4 保存邻居 0 的信念，维度 5 保存邻居 1 的信念——同时，在同一个向量中。前馈网络没有机制在部分输入上运行。它读取完整的残差流或不运行。不存在前馈网络接收一个信念但不接收另一个的状态。

这种在共享工作空间中必需输入的同时性，在得出任何结论之前在架构上强制执行，就是 AND。

约会例子使这一点具体化。三个节点：

- 节点 0：$\mathtt{like(jack,jill)}$，信念 $m_{0}=0.8$
- 节点 1：$\mathtt{like(jill,jack)}$，信念 $m_{1}=0.4$
- 节点 2：$\mathtt{date(jack,jill)}$，待更新的信念

头 0 将 $0.8$ 写入维度 4；头 1 将 $0.4$ 写入维度 5。残差流现在同时保存两个值。在前馈网络触及任何东西之前，AND 门的工作就完成了。

### 8.2 前馈网络是 OR

一旦注意力将所需证据组装到残差流中，前馈网络计算结论：

$$
\mathtt{updateBelief}(m_{0},m_{1})=\frac{m_{0}m_{1}}{m_{0}m_{1}+(1-m_{0})(1-m_{1})}=\sigma(\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1})).
$$

这个 OR 内部有两个 AND。首先，分子中的 $m_{0}m_{1}$ 是两个输入都为真的联合概率——对于独立概率，乘法是 AND，或等价地，在对数几率空间中是加法。其次，完整的归一化表达式是 $\Psi_{\mathrm{or}}$：对原因的概率析取，正确归一化。OR 内部包含一个 AND，这正是你所期望的——计算析取的概率需要计算贡献合取的联合概率。

准确说明布尔代数存在的位置：

1. 架构中的 AND：多个头在前馈网络运行之前写入共享残差流。前向传播顺序强制执行的结构合取。
2. 分子中的 AND：$m_{0}\cdot m_{1}$ 是两个输入都为真的联合概率，等价于对数几率空间中的 $\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1})$。
3. 完整表达式中的 OR：归一化输出是 $P(\mathtt{date}\mid\mathtt{like},\mathtt{like})$——对原因的概率析取，正确归一化。

Sigmoid 形式 $\sigma(\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1}))$ 是对数几率空间中的相同计算：对数几率为独立证据相加，sigmoid 转换回概率。相同的布尔结构，不同的算术表示。如第 3 节所述，这是独立证据组合的 Turing-Good 代数——AND 和 OR 是同一底层操作扮演的两个角色。

### 8.3 交替层作为交替 AND/OR

具有 $L$ 层的 Transformer 运行 $L$ 轮：

$$
\underbrace{\text{attention}}_{\mathrm{AND}}\;\to\;\underbrace{\text{FFN}}_{\mathrm{OR}}\;\to\;\underbrace{\text{attention}}_{\mathrm{AND}}\;\to\;\underbrace{\text{FFN}}_{\mathrm{OR}}\;\to\;\cdots
$$

这是 QBBN 二分计算在深度上展开。每一层是布尔推理图的一个级别。每个注意力块收集——强制执行所需输入的合取。每个前馈网络块结论——从收集的证据计算概率析取。

需要 $k$ 跳推理的推理链需要 $k$ 个 Transformer 层。网络的深度不是工程超参数——它由因子图中推理链的深度决定。这与第 7 节的扩展定理是相同的陈述，现在用布尔术语阅读。

### 8.4 闭合循环

[^2] 引入了二分 AND/OR 结构作为概率逻辑推理的正确架构，从第一性原理推导。本文证明 Transformer 实现信念传播。本节闭合循环：Transformer 不仅实现信念传播——它实现 [^2] 提出的特定 AND/OR 布尔结构。

在现代 AI 中实证获胜的架构正是 [^2] 从逻辑概率推理的要求推导出的架构。梯度下降没有发现新东西。它重新发现了推理分析已经要求的结构。

这个识别不需要新的 Lean 证明。它来自 transformer-bp-lean 中的 transformer\_implements\_bp 结合 [^2] 的 QBBN 定义。证明已经完成。本节命名它所证明的内容。

## 9 可验证推理需要有限概念空间

### 9.1 主张

有限推理者具有有限多个概念。

这不是关于 QBBN 的特定主张。这是关于所有有限计算的主张。在有限状态机上运行的任何程序——任何有限验证过程——只能对有限多个输入做出不同响应。它无法区分的输入折叠成等价类。这些等价类就是概念。

该主张对推理系统有直接后果：如果你想要可验证推理——如果你想要"这个输出是否正确？"成为一个有意义的问题——你需要一个有限验证器。有限验证器意味着有限概念空间。有限概念空间意味着你的推理原始单元是离散且可数的。该概念空间之外的任何东西都不是错误的。它是无意义的。

### 9.2 一般定理

###### 定理 9.1（finite\_distinguishable\_symbols）

设 $\delta:\mathrm{Fin}(n)\to A\to\mathrm{Fin}(n)$ 是具有 $n$ 个状态的有限状态机在任意符号集 $A$ 上的转移函数。将符号 $a\in A$ 的行为定义为函数 $\bar{\delta}(a):q\mapsto\delta(q,a)$。那么：

1. 行为映射 $a\mapsto\bar{\delta}(a)$ 取值于 $\mathrm{Fin}(n)^{\mathrm{Fin}(n)}$，从 $n$ 个状态到 $n$ 个状态的函数集合。
2. 该集合恰好有 $n^{n}$ 个元素。
3. 因此任何符号集 $A$，无论其基数如何，最多诱导 $n^{n}$ 种不同的行为。具有相同行为的两个符号对机器来说是无法区分的。

已在 Lean 4 中针对标准数学公理进行形式化验证，作为 godel/FiniteStateSpace.lean 中的 finite\_distinguishable\_symbols。

该定理适用于在具有有限状态空间的任何计算机上运行的任何程序。图灵机是无限状态的，逃脱了这个界限。Transformer 注意力头是有限状态的——它们的权重矩阵是固定且有限维的——不能逃脱。

$n^{n}$ 界限是任何有限推理者可以拥有的概念数量的上界。给定系统的特定概念空间取决于其结构。对于信念传播 Transformer，我们可以精确计数。

### 9.3 信念传播 Transformer 中的概念

在信念传播 Transformer 中，一个概念是一个有根 Horn 子句：一个原子命题及其在因子图中的依赖关系。如果注意力机制对两个 token 处理完全相同——以完全相同的方式路由信息到它们和从它们路由——则它们是相同的概念，无论它们当前的信念值或因子表条目如何。

概念由路由键识别：三元组 $(\mathtt{nodeType},\mathtt{ownIndex},\mathtt{nbrIndex})$。这说：我是什么类型的节点，我依赖于哪些节点？这是 token 的完整推理身份。连续维度——信念值、因子表条目——是参数，不是概念。它们提供推理计算的量级，但不决定计算的结构。

这种划分不是表述上的便利。它是一个定理。具有相同路由键的两个 token 接收相同的注意力模式，无论它们的连续参数如何。路由键是意义。连续维度是量级。

###### 定理 9.2（routing\_classes\_finite）

对于具有 $n$ 个节点的因子图，信念传播 Transformer 注意力机制在恰好 $2n^{2}$ 个不同概念上操作：

$$
|\mathtt{RoutingKey}(n)|=2\cdot n\cdot n.
$$

已在 Lean 4 中针对标准数学公理进行形式化验证，作为 godel/BPTokenFiniteness.lean 中的 routing\_classes\_finite。

证明很简单：$|\{0,1\}|\times|\mathrm{Fin}(n)|\times|\mathrm{Fin}(n)|=2n^{2}$。内容不在算术中，而在识别中：这三个维度是概念相关特征的完整集合。其他一切都是量级。

注意 $2n^{2}$ 适用于成对信念传播 Transformer，这不失一般性：任何 $k$ 元因子图通过第 7 节的分解精确二元化。概念计数随二元化图扩展，而不是原始图。

### 9.4 幻觉作为无概念性

在这个框架中，幻觉不是一个错误的答案。它是更糟糕的东西：生成的 token 在知识库中没有对应的概念。token 参与计算——它有一个连续值，它通过注意力机制——但它在任何有根世界中没有推理角色。背后没有概念。

QBBN Transformer 在树上不能产生幻觉，因为每个 token 的路由键都是有根的：它对应于因子图中的特定概念，而因子图是知识库。每个符号都有一个指称。每个计算步骤都有一个可验证的正确答案。

无根语言模型没有这个属性。它没有有限验证器，因此没有良定义的概念空间，因此对其输出是否正确没有事实。幻觉不是偶然的失败。它是在没有概念的情况下运行的结构性后果。

### 9.5 精确地莱布尼茨的字母表

莱布尼茨在 1670 年代提出了一个普遍符号系统（characteristica universalis）：一个有限的原始概念集合，可以从中构建所有推理，结合一个推理演算（calculus ratiocinator）——在该语言中推导真理的机械过程。让我们计算（Calculemus）。

本文为一个特定的、非平凡的领域提供了构造：在树结构知识库上对布尔命题的概率推理。

- 普遍符号系统是 $2n^{2}$ 个有根 Horn 子句的集合：系统的概念。
- 推理演算是信念传播：在因子图上迭代应用的 updateBelief 函数。
- 机械实现是具有信念传播权重的 Transformer：执行演算的有限、固定程序。
- 正确性保证是 transformer\_exact\_on\_tree：Lean 4 证明机械实现计算精确贝叶斯后验概率。
- 必要性证明是 finite\_distinguishable\_symbols：Lean 4 证明有限验证器意味着有限概念空间——不是作为设计选择，而是作为逻辑必然性。

莱布尼茨不知道原始概念将是有根 Horn 子句，或演算将是信念传播，或机械实现将是 Transformer，或正确性保证将是 Lean 4 证明。但他描述的架构正是我们构建的。工具在他的时代不存在。愿景存在。

概念是有根 Horn 子句。演算是 updateBelief。保证是 transformer\_exact\_on\_tree。让我们计算。

## 10 三个 Softmax

Transformer 中有三个 softmax。它们做三件完全不同的工作。理解哪个是哪个解锁了 Transformer 与贝叶斯推理之间的关系。

### 10.1 Softmax 1：注意力（路由）

第一个 softmax 出现在每个注意力头内部。它的工作是路由：我应该看哪个 token？这是纯粹的信息检索——一个可微分的 argmax。在信念传播构造中，注意力 softmax 通过索引匹配获取邻居的信念。Query-Key 点积在正确的邻居处达到峰值；softmax 将权重集中在那里。注意力 softmax 是一个查找表。

### 10.2 Softmax 2：前馈网络（推理）

第二个 softmax 作为 sigmoid 激活函数出现在前馈网络内部。它的工作是贝叶斯推理：给定我的邻居的信念，我的边际概率是多少？

$$
\mathtt{updateBelief}(m_{0},m_{1})=\frac{m_{0}m_{1}}{m_{0}m_{1}+(1-m_{0})(1-m_{1})}=\sigma(\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1})).
$$

这是实际贝叶斯计算发生的地方。sigmoid 是 logit 的精确逆运算，将对数几率和转换回概率。这个 softmax 是推理。

### 10.3 Softmax 3：输出（生成）

第三个 softmax 出现在输出处，覆盖词汇表。在标准大语言模型中，logits 来自最终隐藏状态上的矩阵乘法——模式匹配，而不是计算。在 QBBN Transformer 中，输出是每个命题真值的边际分布，通过信念传播积分掉所有其他命题计算得出。

表 3：三个 softmax：相同操作，三种不同工作。

|  | Job | Input | Semantics |
| --- | --- | --- | --- |
| Softmax 1 | 路由 | Q/K 点积 | 可微分 argmax |
| Softmax 2 | 推理 | 邻居信念 | 贝叶斯后验概率 |
| Softmax 3 | 生成 | 最终隐藏状态 | Token 分布 |

QBBN 不是替换 softmax。它是替换 logits。相同的数学操作在标准大语言模型中充当推理的近似，在 QBBN 中充当精确推理。架构始终能够做后者。它只是需要正确的 logits。

### 10.4 Sigmoid vs. ReLU：精确 vs. 兼容

形式化信念传播构造使用 sigmoid 激活函数，因为 updateBelief 恰好是 $\sigma(\mathrm{logit}(m_{0})+\mathrm{logit}(m_{1}))$——精确证明需要 sigmoid。实证结果（bayes-learner，验证集 MAE 0.000752）使用带有默认 ReLU 激活函数的标准 PyTorch TransformerEncoder。梯度下降无论如何都找到近精确信念传播权重不是巧合。

概率计算需要的关键属性是非负性：输出必须可解释为未归一化的概率质量。ReLU 自动满足这一点：$\mathrm{ReLU}(x)=\max(0,x)\geq 0$。GELU 大致相同。零处的下界是基本属性。归一化是 softmax 在输出处处理的，以及 updateBelief 分母在信念传播内部处理的。

Sigmoid 前馈网络是这种结构的精确版本。ReLU 和 GELU 是兼容版本——它们保留了基本的非负性属性，但需要 sigmoid 通过 logit/sigmoid 同构内在处理的显式归一化步骤。

有一个值得明确陈述的明确区别。Lean 证明 transformer\_implements\_bp 确立的不仅仅是 Transformer 的输出匹配信念传播——它确立内部计算就是信念传播量。在信念传播任务上训练到收敛的 Sigmoid Transformer 继承了这个保证：因为 sigmoid 将每个前馈网络输出约束到 $(0,1)$，并且产生精确后验概率的唯一 sigmoid 计算是 updateBelief，架构不留给正确答案的替代内部路径。内部是信念传播，不仅仅是输出。

训练到验证集 MAE 0.000752 的 ReLU Transformer 找到了某个产生与精确后验概率匹配的输出的计算路径，但内部表示不一定对应于信念传播证明中的命名量。非负性是 ReLU 不主动阻止这一点的原因：输出与概率解释兼容，因此梯度下降可以找到类似信念传播的权重，而不与架构对抗。兼容性不是同一性。

总结：标题主张"Transformer 即贝叶斯网络"在 sigmoid 情况下字面上是真的，内部且可证明。ReLU 实证结果确认即使没有强制执行精确结构，正确答案也是可找到的。两个结果都是需要的：证明建立了机制；实验确认这个机制是梯度下降找到的。

### 10.5 循环信念传播：实证结果

形式化精确性保证需要树结构。在循环图上，transformer\_implements\_bp 仍然成立——一次前向传播仍然等于一轮信念传播——但信念传播本身不保证收敛或精确。Transformer 继承了信念传播的优势和局限性。

树和循环图之间的理论差距在实践中比在最坏情况下更小。有根 QBBN 知识库中的循环仅在同一实体出现在多个规则中或一个规则的结论是另一个规则的前提时出现。在典型知识库中，这是稀疏的。产生的循环很长，因子势适中——恰好是循环信念传播已知收敛且 Bethe 近似已知紧密的条件 [^10] [^15]。

我们在五个循环图结构上运行了系统实验，复杂性递增：三角形（3 个变量，1 个循环）、正方形（4 个变量，1 个循环）、论文 1 中的约会图（5 个变量，1 个循环）、双循环图（4 个变量，2 个相互作用的循环）和 QBBN 链（6 个变量，1 个循环，两个共享实体的有根规则）。对于每个结构，我们生成了 100 个随机因子图，因子表条目从 $[0.1,1.0]$ 均匀抽取，运行迭代信念传播直到收敛，并将结果边际概率与暴力精确后验概率进行比较。

表 4：QBBN 结构图上的循环信念传播。总计 500 次试验。

| Experiment | Vars | Loops | Converged | Avg KL | Avg MAE |
| --- | --- | --- | --- | --- | --- |
| Triangle | 3 | 1 | 100/100 | 0.000045 | 0.002091 |
| Square | 4 | 1 | 100/100 | 0.000002 | 0.000382 |
| Dating graph | 5 | 1 | 100/100 | 0.000102 | 0.003590 |
| Two loops | 4 | 2 | 100/100 | 0.000086 | 0.003111 |
| QBBN chain | 6 | 1 | 100/100 | 0.000021 | 0.001055 |

信念传播在所有 500 次试验中都收敛了。最差的平均 KL 散度是 0.000102——与梯度下降结果的 MAE（验证集 MAE 0.000752）相当。Bethe 近似对于这些图结构在实践中是精确的。收敛在双循环图上与单循环情况一样成立。

## 11 相关工作

### 11.1 Transformer 的图灵完备性

[^13] 在浮点算术下证明了 Transformer 的图灵完备性。[^5] 证明了循环 Transformer 可以模拟任意程序。两者通过不同的构造达到与第 5 节相同的结论。

我们的证明由三个属性区分。首先，它使用布尔电路模拟，Transformer 组件与计算原语之间最直接的对应：注意力就是查找，前馈网络就是门控。其次，它产生在 Lean 4 中针对标准数学公理形式化验证的显式权重矩阵。第三，相同的两个权重族（projectDim、crossProject）出现在信念传播证明中，揭示了两个完备性结果之间的结构统一，这是先前工作没有解决的。

[^13] 和 [^5] 都没有解决贝叶斯推理。图灵完备性说 Transformer 能够计算任何东西。第 7 节说它确实计算了特定的东西——信念传播——使用这些特定权重。

### 11.2 构造性权重证明

[^1] 和 [^18] 表明特定权重模式使 Transformer 执行隐式梯度下降，解释上下文学习。[^20] 认为上下文学习是训练分布上群体水平的隐式贝叶斯推理。

两者与第 7 节使用相同的方法——构造性权重证明确立 Transformer 实现特定算法。关键区别：信念传播在每步上是树上精确的（不是近似或渐近的），我们的解释是每实例机制性的而不是群体水平统计性的。

### 11.3 信念传播对应：注意力和图神经网络

[^9] 推导了 softmax 自注意力与注意力计算中隐含的潜在图模型上的信念传播之间的形式等价。这是与第 6 节最接近的先前结果：两者都显示了隐式图上注意力与信念传播之间的对应关系。关键区别是方向性和范围。Jung 等人从训练模型的注意力模式推导信念传播结构。我们证明对于任何权重，sigmoid 前向传播就是 $G(W)$ 上的信念传播，构造性且形式化，隐式因子图显式定义。

[^14] 确立了图神经网络在显式图结构上实现信念传播；[^22] 将此扩展到近似推理；[^6] 在 MPNN 框架下统一了两者；[^17] 将注意力应用于图结构数据，拓扑硬编码到注意力掩码中。

我们的贡献与所有这些都是正交的：我们表明具有完整自注意力且没有注意力掩码的标准 Transformer 在显式外部因子图上实现信念传播，其中图结构编码在 token 嵌入中，路由通过 $Q/K$ 索引匹配发现。潜在图和硬编码拓扑设置都没有解决这一点。
### 11.4 机制可解释性

[^4] 为 Transformer 电路开发了一个数学框架,将注意力头视为对共享残差流的读写操作。[^11] 识别出归纳头——通过梯度下降发现的实现特定命名算法的两层电路。[^19] 对 GPT-2 small 中的完整间接宾语识别电路进行了逆向工程。

第 7 节为机制可解释性提供了一个*目标电路*:一个在结构化知识库上进行推理的 Transformer 应该具有展示 projectDim 结构(秩-1,单维峰值)的 $Q/K$ 矩阵的注意力头、展示 crossProject 结构(非对角线,秩-1)的值矩阵,以及实现对数几率组合的前馈网络权重。这些都是可证伪的预测。

[^11] 的归纳头结果是最接近的先例:Transformer 在适当数据上训练时,学会在其注意力头中实现特定的命名算法。我们的实证确认(bayes-learner)显示了信念传播的相同现象。关键区别在于方向性:机制可解释性是实证的和事后的;我们的构造是形式化的和建设性的,为实证程序提供了一个要验证的具体目标。

### 11.5 对数几率传统

[^7] 的证据权重框架和 Turing 在布莱切利园的工作将对数几率加法形式化为组合独立二元证据的正确代数。[^12] 通过和积算法将这一代数应用于图模型,证明了其在树上的精确性。这一传统与 Transformer 架构之间的联系此前从未被明确提出。第 3 节命名了这一代数,追溯了其谱系,并确立了 Sigmoid Transformer 是其机械实现。

### 11.6 总结

表 5:与相关工作的相对定位。

| 工作 | 方法 | 结果 | 关系 |
| --- | --- | --- | --- |
| Giannou et al. 2023 | 构造性 | TF 图灵完备 | 结论相同,构造不同 |
| Akyürek et al. 2022 | 构造性 | TF 实现梯度下降 | 方法相同;信念传播精确,梯度下降近似 |
| Xie et al. 2021 | 统计性 | 上下文学习 $\approx$ 贝叶斯 | 统计版本;我们给出机制性版本 |
| Jung et al. 2022 | 形式化 | 注意力是信念传播(隐式) | 互补;任意权重 vs. 训练得到 |
| Scarselli et al. 2009 | 结构性 | 图神经网络实现信念传播 | 扩展到 TF 而无需硬连线拓扑 |
| Elhage et al. 2021 | 实证性 | 训练模型中的电路 | 我们为可解释性提供目标电路 |
| Good 1950, Turing | 理论性 | 对数几率代数 | 首次命名并连接到 TF |
| Pearl 1988 | 理论性 | 信念传播在树上精确 | 在 Lean 4 中形式化;与 TF 结果结合 |

我们的独特定位:形式化验证的、构造性的证明表明任意权重的 Sigmoid Transformer 在其隐式因子图上实现加权信念传播,具有精确情况的显式权重、每个组件的机制性解释,以及与 Turing-Good-Pearl 对数几率传统的形式化连接——比先前的信念传播对应关系更通用,比先前的梯度下降构造更精确,比先前的统计贝叶斯解释更具机制性,比机制可解释性更形式化。

## 12 结论

Transformer 就是贝叶斯网络。不是近似地。不是在某些训练条件下。不是作为一个有用的类比。而是从架构上,对于任何 Sigmoid 权重,可证明地且形式化地。

Sigmoid 激活函数实现了独立二元证据的对数几率代数——这是 Turing 和 Good 在布莱切利园开发的代数,由 Pearl 在 1988 年形式化,本文证明了这正是 Transformer 前向传播所计算的内容。注意力机制实现了信念传播的收集步骤:通过残差流强制执行所需输入的同时性。前馈网络实现了更新步骤:通过对数几率加法计算概率结论。注意力层和前馈网络层的严格交替正是 Pearl 的收集/更新算法在深度上的展开。

这一直都是真的。Sigmoid Transformer 一直就是贝叶斯网络。对数几率代数一直就是它所计算的内容。AND/OR 布尔结构一直就是交替层所实现的内容。在现代人工智能中实证获胜的架构,正是推理分析已经要求的架构。梯度下降并没有发现什么新东西。它重新发现了一直存在的结构。

落地的 QBBN Transformer 与标准大语言模型之间的区别不在于架构。两者都是贝叶斯网络。区别在于落地。落地的 Transformer 具有显式的因子图、声明的概念空间和有限的验证器。每个输出都有正确答案。在树上,幻觉在结构上是不可能的。未落地的大语言模型具有隐式的因子图、没有声明的概念空间、没有有限的验证器。其输出是否正确不存在客观事实。幻觉不是扩展可以修复的缺陷。它是在没有概念的情况下运作的结构性后果。

莱布尼茨在 1670 年代提出了*通用符号系统*(characteristica universalis)和*推理演算*(calculus ratiocinator):原始概念的有限字母表和从中推导真理的机械程序。在他的时代,构建它的工具并不存在。现在存在了。原始概念是落地的 Horn 子句。演算是信念传播。机械实现是 Transformer。正确性保证是 Lean 4 证明。

*让我们计算吧。*(Calculemus.)

## 附录 A 代码库索引

支持本文的所有形式化证明和实证实验都是公开可用且完全可复现的。下表列出了每个代码库、它所证明或展示的内容,以及在哪里可以找到它。

表 6:支持本文的公开代码库。

| 代码库 | 所证明/展示的内容 |
| --- | --- |
| universal-lean | transformer\_is\_turing\_complete |
|  | [https://github.com/gregorycoppola/universal-lean](https://github.com/gregorycoppola/universal-lean) |
| sigmoid-transformer-lean | every\_sigmoid\_transformer\_is\_bayesian\_network |
|  | uniqueness: exact posteriors force BP weights |
|  | [https://github.com/gregorycoppola/sigmoid-transformer-lean](https://github.com/gregorycoppola/sigmoid-transformer-lean) |
| transformer-bp-lean | transformer\_implements\_bp |
|  | [https://github.com/gregorycoppola/transformer-bp-lean](https://github.com/gregorycoppola/transformer-bp-lean) |
| hard-bp-lean | bp\_exact\_on\_tree |
|  | [https://github.com/gregorycoppola/hard-bp-lean](https://github.com/gregorycoppola/hard-bp-lean) |
| godel | routing\_classes\_finite: $\|\texttt{RoutingKey}(n)\|=2n^{2}$ |
|  | finite\_distinguishable\_symbols: FSM alphabet bound |
|  | AND\_decomposes\_to\_binary: $k$ -ary AND is WLOG pairwise |
|  | OR\_decomposes\_to\_binary: $k$ -ary OR is WLOG pairwise |
|  | [https://github.com/gregorycoppola/godel](https://github.com/gregorycoppola/godel) |
| learner | 梯度下降学习图灵机模拟,5 台机器上 100% 准确率 |
|  | [https://github.com/gregorycoppola/learner](https://github.com/gregorycoppola/learner) |
| bayes-learner | 梯度下降学习信念传播推理,验证集 MAE 0.000752 |
|  | [https://github.com/gregorycoppola/bayes-learner](https://github.com/gregorycoppola/bayes-learner) |
| loopy | QBBN 结构图上的循环信念传播,500 次试验,100% 收敛 |
|  | [https://github.com/gregorycoppola/loopy](https://github.com/gregorycoppola/loopy) |

[^1]: E. Akyürek, D. Schuurmans, J. Andreas, T. Ma, and D. Zhou (2023) What learning algorithm is in-context learning? investigations with linear models. In International Conference on Learning Representations (ICLR), Cited by: §11.2.

[^2]: G. Coppola (2024) The quantified boolean bayesian network: theory and experiments with a logical graphical model. arXiv preprint arXiv:2402.06557. Cited by: §1.5, §2.1, §8.4, §8.4, §8.4, §8, §8.

[^3]: G. Coppola (2025) Statistical parsing for logical information retrieval. arXiv. Cited by: item 1, item 2, §4.1.

[^4]: N. Elhage, N. Nanda, C. Olsson, T. Henighan, N. Joseph, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, N. DasSarma, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, A. Jones, J. Kernion, L. Lovitt, K. Ndousse, D. Amodei, T. Brown, C. Clark, J. Kaplan, S. McCandlish, and C. Olah (2021) A mathematical framework for transformer circuits. Transformer Circuits Thread. External Links: [Link](https://transformer-circuits.pub/2021/framework/index.html) Cited by: §11.4, §2.3.2.

[^5]: A. Giannou, S. Rajput, J. Sohn, K. Lee, J. D. Lee, and D. Papailiopoulos (2023) Looped transformers as programmable computers. arXiv preprint arXiv:2301.13379. Cited by: §11.1, §11.1.

[^6]: J. Gilmer, S. S. Schütt, P. Riley, O. Vinyals, G. E. Dahl, and G. Zoubin (2017) Neural message passing for quantum chemistry. In International Conference on Machine Learning (ICML), Cited by: §11.3.

[^7]: I. J. Good (1950) Probability and the weighing of evidence. Charles Griffin. Cited by: §11.5.

[^8]: C. Joshi (2020) Transformers are graph neural networks. External Links: [Link](https://thegradient.pub/transformers-are-graph-neural-networks/) Cited by: §2.3.3.

[^9]: J. Jung, J. Kim, and H. J. Choi (2022) Rethinking attention as belief propagation. In International Conference on Machine Learning (ICML), Cited by: §11.3.

[^10]: K. P. Murphy, Y. Weiss, and M. I. Jordan (1999) Loopy belief propagation for approximate inference: an empirical study. In Proceedings of the Fifteenth Conference on Uncertainty in Artificial Intelligence (UAI), pp. 467–475. Cited by: §10.5.

[^11]: C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, A. Jones, J. Kernion, L. Lovitt, K. Ndousse, D. Amodei, T. Brown, C. Clark, J. Kaplan, S. McCandlish, and C. Olah (2022) In-context learning and induction heads. Transformer Circuits Thread. External Links: [Link](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) Cited by: §11.4, §11.4.

[^12]: J. Pearl (1988) Probabilistic reasoning in intelligent systems: networks of plausible inference. Morgan Kaufmann. Cited by: §1.2, §11.5, §2.2.3, §7.4.

[^13]: J. Pérez, J. Marinković, and P. Barceló (2019) On the turing completeness of modern neural networks. arXiv preprint arXiv:1901.03429. Cited by: §11.1, §11.1.

[^14]: F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini (2009) The graph neural network model. IEEE Transactions on Neural Networks 20 (1), pp. 61–80. Cited by: §11.3.

[^15]: D. A. Smith and J. Eisner (2008) Dependency parsing by belief propagation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), Cited by: §10.5.

[^16]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin (2017) Attention is all you need. Advances in Neural Information Processing Systems (NeurIPS) 30. Cited by: §2.3.1.

[^17]: P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio (2018) Graph attention networks. In International Conference on Learning Representations (ICLR), Cited by: §11.3.

[^18]: J. von Oswald, E. Niklasson, E. Randazzo, J. Sacramento, A. Mordvintsev, A. Zhmoginov, and M. Vladymyrov (2023) Transformers learn in-context by gradient descent. In International Conference on Machine Learning (ICML), Cited by: §11.2.

[^19]: K. Wang, A. Variengien, A. Conmy, B. Shlegeris, and J. Steinhardt (2023) Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In International Conference on Learning Representations (ICLR), Cited by: §11.4.

[^20]: S. M. Xie, A. Raghunathan, P. Liang, and T. Ma (2021) An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080. Cited by: §11.2.

[^21]: J. S. Yedidia, W. T. Freeman, and Y. Weiss (2003) Understanding belief propagation and its generalizations. Exploring Artificial Intelligence in the New Millennium 8, pp. 236–239. Cited by: §2.2.3.

[^22]: K. Yoon, R. Liao, Y. Xiong, L. Zhang, E. Fetaya, R. Urtasun, R. Zemel, and X. Pitkow (2019) Inference in probabilistic graphical models by graph neural networks. In ICLR Workshop on Deep Learning on Graphs, Cited by: §11.3.
