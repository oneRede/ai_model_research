---
sourceTitle: "Beyond RLHF: A Unified Theoretical Framework of Alignment"
sourceUrl: "https://arxiv.org/html/2506.01523"
sourceLanguage: "en"
title: "超越 RLHF：对齐的统一理论框架"
language: "zh-CN"
requestedUrl: "https://arxiv.org/html/2506.01523"
adapter: "generic"
capturedAt: "2026-08-14T05:44:52.760Z"
conversionMethod: "defuddle"
kind: "generic/article"
pipelineRunId: "20260814-134240"
pipelineSource: "translate/20260814-134240/works-ready/arxiv-2506-01523-translation.md"
sourceFigureCount: null
---

# 超越 RLHF：对齐的统一理论框架

Jihun Yun¹ 感谢：贡献均等，所属机构：KRAFTON Juno Kim¹ 感谢：在 KRAFTON 实习期间完成的工作，所属机构：加州大学伯克利分校 Jongho Park 所属机构：KRAFTON 所属机构：加州大学伯克利分校 Junhyuck Kim 所属机构：KRAFTON Jongha Jon Ryu 所属机构：麻省理工学院 Jaewoong Cho 所属机构：KRAFTON Kwang-Sung Jun 所属机构：POSTECH
jihuny@krafton.com junokim@berkeley.edu kwangsungjun@postech.ac.kr

###### 摘要

基于人类反馈的强化学习（RLHF）已成为控制大语言模型（LLM）输出质量的主流对齐范式。然而，现有理论既未能为 RLHF 目标函数本身提供强有力的正当性，也因不同方法通常在不同框架下进行分析而难以对其理论保证进行比较。为建立统一的框架，本文探讨在何种假设下可以推导出现有或新的训练目标函数并获得理论保证。为此，我们将对齐重新定义为从配对偏好中进行**分布学习**，其基础是一个概率假设，该假设描述了偏好如何揭示目标语言模型的信息。基于此，我们提出了三种有原则的对齐目标函数：偏好最大似然估计（PMLE）、偏好蒸馏和反向 KL 最小化。我们证明这三个目标函数均享有关于目标语言模型的强非渐近 $O(1/n)$ 收敛保证，且自然避免了退化问题。特别地，反向 KL 与 RLHF 目标函数高度相似，为 RLHF 提供了有力的理论正当性。此外，我们的理论首次解释了一个经验发现：在策略目标函数（如 RLHF）通常优于似然型目标函数（如 DPO）。最后，实验结果表明，所提出的目标函数在多个任务和模型上与强基线方法具有竞争力。

### 1 引言

对齐是指通过人类偏好 [^5] [^39] 改善大语言模型（LLM）生成响应的质量（如有用性和无害性）的任务，现已成为 LLM 训练事实上的最终阶段。首个对齐方法是基于人类反馈的强化学习（RLHF）[^9] [^46]，该方法从配对偏好中训练奖励模型 $R$，然后通过强化学习优化策略 $\pi$（即语言模型）以最大化奖励：

$$
\displaystyle\max_{\pi}~~\EE_{x\sim\mathcal{D},a\sim\pi(x)}\big[\hat{R}(x,a)\big]-\beta\KL(\pi\|\pi_{0})
$$

其中 $\hat{R}$ 是从偏好数据中学习的奖励函数，$\mathcal{D}$ 是提示分布，$\pi(x)$ 是策略 $\pi$ 对提示 $x$ 的响应分布，$\KL(p\|q)=\EE_{x\sim\mathcal{D}}\left[\KL(p(x)\|q(x))\right]$ 是策略间的 KL 散度，${\pi_{0}}$ 是监督微调后的参考语言模型，$\beta>0$ 是正则化强度。

RLHF 目标函数是各种实用算法的核心，从根本上塑造了研究者对对齐问题的思考方式。例如，DPO（直接策略优化）可视为在强“全策略”假设下对 RLHF 的重新表述，使得目标函数由简单的似然比项组成，而非依赖在策略响应 [^42]；$\Psi$PO 通过将 $R(x,a)$ 泛化为偏好概率的 $\Psi$ 变换来扩展 RLHF [^4]；甚至对齐算法的理论分析也常将 RLHF 目标函数或其变体作为最终的学习理论目标，并证明收敛保证（如遗憾界）[^61] [^58] [^62] [^24] [^57]。诚然，RLHF 目标函数已被证明是有用的，人们或许会认为它是一个合理的目标函数。

表 1：所提方法及理论保证总结。每一节中，我们与现有方法（如 DPO 和 REBEL [^18]）进行类比。

| 分布学习 | 相关方法 | 奖励模型 | 需 RL 训练 | 目标函数 | 正向 KL 保证 |
| --- | --- | --- | --- | --- | --- |
| 偏好最大似然估计（第 3 节）| DPO | 不使用 | 否 | 式 (4) | $O(1/n)$（定理 4）|
| 偏好蒸馏（第 4 节）| REBEL | 需要 | 否 | 式 (11) | $O(1/n)$（定理 6）|
| 反向 KL（第 5 节）| RLHF | 需要 | 是 | 式 (15) | $O(1/n)$（定理 7）|

然而，从学习理论的角度来看，RLHF 目标函数尚无已知的正当性。首先，现有的理论保证通常显示收敛到使用真实奖励的总体 RLHF 目标函数的解（参见前述引用工作）。这实质上是一种同义反复，难以被解释为正当性。其次，虽然最大化奖励在非正式意义上听起来合理，但其正式含义并不清楚，因为对齐问题的定义根本不涉及奖励！第三，对奖励的**先验**依赖体现在以下问题中：式 (1) 是“损失 + 正则化项”形式的标准机器学习目标函数，其中损失（在策略的负奖励）从数据中学习，正则化项利用可用的起始模型。在数据充足的情况下，正则化强度 $\beta$ 应当降低，并在渐近情形下趋于零以确保充分学习。从这个意义上讲，在渐近情形下，RLHF 目标函数面临两难境地：要么趋向零正则化，使得式 (1) 的解变为确定性（作为语言模型这不可取），要么使用非零正则化而妨碍学习过程。这表明，原则上 RLHF 需要某种修正。

那么，我们应该要求什么样的正当性？一种强有力的正当性形式是可证明的保证，例如统计收敛到目标语言模型。然而，现有分析通常仅适用于其所选择的方法。例如，DPO [^1] [^27] 的理论框架与 RLHF [^61] [^58] [^62] [^24] [^57] 的理论框架不兼容（即具有不同的理论目标），使我们无法直接比较它们的理论保证。

上述讨论的局限性呼唤对齐的统一学习理论框架。因此，我们提出以下问题：

<svg id="S1.p6.pic1" height="72.67" overflow="visible" version="1.1" viewBox="0 0 477.38 72.67" width="477.38"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,72.67) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#D9D9D9;" fill="#D9D9D9" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.15 L 0 68.52 C 0 70.81 1.86 72.67 4.15 72.67 L 473.23 72.67 C 475.52 72.67 477.38 70.81 477.38 68.52 L 477.38 4.15 C 477.38 1.86 475.52 0 473.23 0 L 4.15 0 C 1.86 0 0 1.86 0 4.15 Z"></path></g><g style="--ltx-fill-color:#F2F2F2;" fill="#F2F2F2" fill-opacity="1.0"><path style="stroke:none" d="M 4.15 4.15 L 4.15 68.52 C 4.15 70.04 5.39 71.28 6.92 71.28 L 470.46 71.28 C 471.99 71.28 473.23 70.04 473.23 68.52 L 473.23 4.15 C 473.23 2.62 471.99 1.38 470.46 1.38 L 6.92 1.38 C 5.39 1.38 4.15 2.62 4.15 4.15 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 23.84 49.09)"><foreignObject style="--ltx-fo-width:31.06em;--ltx-fo-height:0.75em;--ltx-fo-depth:2.59em;font-size:10pt;" height="46.28" overflow="visible" transform="matrix(1 0 0 -1 0 10.38)" width="429.78"><span id="S1.p6.pic1.1" style="width:31.06em;"><span id="S1.p6.pic1.1.1"><span id="S1.I1.i1" style="list-style-type:none;"><span id="S1.I1.i1.p1"><span id="S1.I1.i1.p1.1"><span id="S1.I1.i1.p1.1.1" style="--ltx-fg-color:#000000;">在何种假设下（例如，目标语言模型与偏好数据如何关联）我们可以从理论上证明现有训练目标函数的合理性，甚至开发新的目标函数？</span></span></span></span></span></span></foreignObject></g></g></svg>

本文采用流行的学习理论处理方式：首先假设存在我们旨在学习的目标模型，然后对数据生成过程作为目标模型的函数进行概率假设，随后基于统计原理开发目标函数。这种处理方式广泛用于分类（如逻辑回归）、主题建模（如隐狄利克雷分配）和生成模型（如变分自编码器）。这提供了两个关键好处：(i) 明确的建模假设，通常有助于理解模型行为；(ii) 严格的学习理论保证，如一致性——随着样本量增长收敛到目标模型。或许令人惊讶的是，这种针对对齐的**完全概率**框架在很大程度上仍未被探索。相关工作见附录 A。

##### 我们的贡献

我们超越盲目地将 RLHF 目标函数作为最终目标，提出了一个新颖的统一理论框架用于对齐，该框架可被视为从配对偏好数据进行分布学习的完全概率方法，不依赖于任何**先验**的奖励最大化概念。具体而言，我们假设存在一个目标（预言）语言模型 $\pi^{*}$，并明确建模偏好反馈如何揭示关于 $\pi^{*}$ 的信息。直观地说，$\pi^{*}$ 必须对首选响应赋予更高概率，我们将其编码为假设

$$
\displaystyle\PP(a\succ b\mid x)={\frac{\pi^{*}(a\mid x)^{\gamma}}{\pi^{*}(a\mid x)^{\gamma}+\pi^{*}(b\mid x)^{\gamma}}}
$$

其中 $\gamma>0$，$a\succ b$ 表示响应 $a$ 优于 $b$。这是 Bradley-Terry (BT) 模型 [^7] 的一个实例，其偏好分数为 $\pi^{*}$ 的倾斜响应概率。与 RLHF 中 BT 模型的主要区别在于，我们的假设表明偏好模型直接依赖于目标语言模型，而非某个奖励函数。注意，我们的假设表明偏好模型**显式地**是一个语言模型。这与 DPO 形成鲜明对比，后者从 RLHF 公式出发，利用（不现实的）全策略假设来揭示奖励模型（或偏好模型）与语言模型之间存在**隐含的**关系 [^42]。

我们的简单假设导出了与现有工作相关的各种训练目标函数，并且它们的解可证明地在 KL 散度意义下收敛到 $\pi^{*}$。具体而言，我们提出以下三种算法（总结在表 1 中）：

- **PMLE（偏好最大似然估计；第 3 节）**：该目标函数最大化偏好模型 (2) 的似然，并受参考策略 ${\pi_{0}}$ 的反向 KL 正则化约束。与 DPO 类似，它相对容易优化。
- **偏好蒸馏（第 4 节）**：通过从学习的奖励模型直接估计期望偏好，最大似然估计可重写为将偏好分布蒸馏到语言模型中。与现有的奖励蒸馏 [^14] [^18] 不同，这种表述明确从 Bradley-Terry 模型 (2) 推导而来。
- **反向 KL（RKL；第 5 节）**：由于我们的目标是分布学习，优化反向 KL 散度 $\EE_{x}[\KL(\hat{\pi}(x)\|\pi^{*}(x)]$ 是自然的。虽然 $\pi^{*}$ 未知，但其非归一化形式可通过 (2) 用浅层网络估计。将此估计插入并添加 KL 正则化项，得到带有额外熵项的 RLHF 目标函数变体，可视为对 RLHF 的修正。图 1 显示，在我们的偏好模型下，对每个样本量 $n$ 调整 $\beta$ 的 RLHF 确实存在不一致性（即不收敛到真实 $\pi^{*}$），而 RKL 是一致的。
- **理论保证**：对于所有三种算法，我们证明了形如 $\EE_{x}[\KL(\pi^{*}(x)\|\hat{\pi}(x))]\leq O(1/n)$ 的正向 KL 误差上界，其中 $n$ 是偏好数据集的大小。

图 1：RLHF 与 RKL。

据我们所知，我们的保证是非渐近的，也是从配对反馈中学习分布的首个此类保证 [^12]。此外，我们的理论框架为 RLHF 提供了新颖的统计基础（除微小修正外），无需诉诸同义反复的正当性（即收敛到训练目标函数的总体版本）。有趣的是，我们的保证表明 RKL（类似于 RLHF）相比 PMLE（类似于 DPO）具有更优的理论保证，这可以视为首次在文献中对常见经验发现（RLHF 优于 DPO）的理论确认。同样，这种确认无法用现有理论做出，因为它们为每种方法提供特设的保证。

我们通过实验补充我们的理论，表明我们的方法在 TL;DR 摘要任务中通常优于基线胜率，并在通用聊天场景中生成更受偏好的响应；详见第 6 节。

### 2 预备知识

##### 对齐作为分布学习

设 $\mathcal{X}$ 和 $\mathcal{A}$ 分别为提示和响应的空间，$\mathcal{D}\in\Delta(\mathcal{X})$ 是提示上的固定分布。我们将语言模型（LM）定义为函数或策略 $\pi\mathrel{\mathop{\mathchar 58\relax}}\mathcal{X}\to\Delta(\mathcal{A})$，它确定一组条件（即上下文）分布 $\pi(\cdot\mid x)$，我们也将其简记为 $\pi(x)$。<sup>1</sup> 我们将对齐视为从配对偏好反馈中学习这些分布，该反馈从明确依赖于 $\pi^{*}$（我们希望学习的理想目标语言模型）的模型中抽取。因此，给定语言模型类 $\Pi$，我们的最终目标是找到 $\hat{\pi}\in\Pi$，使其在合适的分布距离度量下尽可能接近 $\pi^{*}$。

##### 我们的偏好模型

设 $\mu$ 为用于生成待偏好标注响应的语言模型；这可以是参考 LLM 或简单地是现有数据集。我们给定一个包含 $n$ 个独立样本的偏好数据集 $D_{n}=\{(x,a^{+},a^{-})\}$，其中 $x\sim\mathcal{D}$ 是提示，$a^{+}/a^{-}$ 是首选/非首选响应。我们假设，给定 $x$，响应对 $(a^{+},a^{-})$ 通过独立地从 $\mu(x)$ 抽取响应 $a,b$ 然后从 $\PP_{*}(a\succ b\mid x)\mathrel{\mathop{\mathchar 58\relax}}=\PP_{\pi^{*}}(a\succ b\mid x)$ 中抽样偏好来采样，其中

$$
\displaystyle\PP_{\pi}(a\succ b\mid x)\mathrel{\mathop{\mathchar 58\relax}}={\frac{\pi(a\mid x)^{\gamma}}{\pi(a\mid x)^{\gamma}+\pi(b\mid x)^{\gamma}}},
$$

然后如果 $a$ 优于 $b$ 则设置 $(a^{+},a^{-})=(a,b)$，否则设置 $(a^{+},a^{-})=(b,a)$。$\gamma$ 的值决定了策略 $\pi$ 下响应概率差异被强调或减弱的程度。在实践中，$\gamma$ 是一个超参数，通常设置为 $0<\gamma<1$。

##### 我们的偏好模型是否过强？

有人可能会疑问 (2) 是否过于限制；我们声称并非如此。我们的假设 (i) 采用 BT 模型 $\PP(a\succ b|x)=p^{*}(a|x)/(p^{*}(a|x)+p^{*}(b|x))$ 对于某个潜在偏好 $p^{*}$，然后 (ii) 将 $p^{*}$ 与目标语言模型 $\pi^{*}$ 联系起来。对于 (i)，虽然 BT 模型存在缺陷（例如不允许循环偏好），但它在许多先前工作中是常规的 [^9] [^46]。此外，我们的分布学习视角可以扩展到包含一般偏好模型。更重要的是，(ii) 是允许我们推导学习 $\pi^{*}$ 保证的关键联系。先前的理论研究为 RLHF 或其他奖励最大化目标函数提供了收敛保证 [^61] [^58] [^62] [^24] [^57]，却未能证明为何必须优化某种奖励构造。相比之下，我们工作的主要观点是识别在哪些假设下可以证明此类目标函数的合理性。事实上，我们将看到从我们框架推导的目标函数通常与基线方法（DPO、RLHF、REBEL）相似，因此我们本质上是使现有方法的隐藏假设变得明确！

##### 理论设定

我们将 $R_{\pi}(x,a)\mathrel{\mathop{\mathchar 58\relax}}=\gamma\ln\pi(a\mid x)$ 称为 $\pi\in\Pi$ **诱导的奖励**。<sup>2</sup> 中心化奖励定义为 $\bar{R}_{\pi}(x,a)\mathrel{\mathop{\mathchar 58\relax}}=R_{\pi}(x,a)-\EE_{a\sim\mu(x)}[R_{\pi}(x,a)\mid x]$。与 $\PP_{*}$ 类似，我们记 $R_{*}\mathrel{\mathop{\mathchar 58\relax}}=R_{\pi^{*}}$ 和 $\bar{R}_{*}\mathrel{\mathop{\mathchar 58\relax}}=\bar{R}_{\pi^{*}}$。最后，我们记 $\Delta\bar{R}_{\pi}\mathrel{\mathop{\mathchar 58\relax}}=\bar{R}_{\pi}-\bar{R}_{*}$。我们的主要假设在对齐文献中是标准的 [^61] [^56] [^64] [^1] [^24]，如下所示：

###### 假设 1（可实现性）

对于有限策略类 $\Pi$，$\pi^{*}\in\Pi$。

###### 假设 2（有界性）

存在 $R>0$ 使得对所有 $\pi\in\Pi$，$|\bar{R}_{\pi}(x,a)|\leq\gamma R$。

由于响应 $(a^{+},a^{-})$ 从 $\mu(x)$ 而非 $\pi^{*}(x)$ 采样，对齐问题是离线学习的一个实例，其中观察数据与我们旨在提供保证的目标分布之间存在分布偏移。因此有必要在 $\mu$ 和策略类 $\Pi$ 之间引入覆盖假设，这在离线强化学习文献中已被充分研究 [^2]。特别地，我们使用以下广义覆盖系数 [^55] [^1]。

###### 定义 3（广义覆盖系数）

对于策略类 $\Pi^{\prime}$，我们用 $C_{\Pi^{\prime}}>0$ 表示满足以下条件的最小常数：对每个 $\pi\in\Pi^{\prime}$，

$$
\displaystyle\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\big[\Delta\bar{R}_{\pi}(x,a)^{2}\big]\leq C_{\Pi^{\prime}}\EE_{x\sim\mathcal{D},a\sim\mu(x)}\big[\Delta\bar{R}_{\pi}(x,a)^{2}\big].
$$

这改进了全策略 $\ell_{\infty}$ 可集中性条件 $\sup_{\pi\in\Pi}\max_{x,a}\tfrac{\pi(a\mid x)}{\mu(a\mid x)}\leq C^{\prime}$ [^35]，因为即使后者为无穷大，前者也可以是有界的，这取决于 $\mathcal{D}$ 和奖励类 $\Delta\bar{R}$。<sup>3</sup>

### 3 偏好最大似然估计方法

我们首先介绍一个基于最大似然的目标函数，该函数可以直接从将对齐视为从配对反馈中进行分布学习推导而来。给定如第 2 节所述的偏好数据集 $D_{n}=\{(x,a^{+},a^{-})\}$，我们希望通过在 BT 偏好假设 (3) 下找到一个策略 $\hat{\pi}$ 来估计 $\pi^{*}$，该策略最大化观察到的配对偏好的似然。具体而言，候选策略 $\pi$ 下每对 $(x,a^{+},a^{-})$ 的负对数似然为：

$$
\displaystyle-\ln\PP_{\pi}(a^{+}\succ a^{-}\mid x)=-\ln\sigma\left(\gamma\ln\frac{\pi(a^{+}\mid x)}{\pi(a^{-}\mid x)}\right),
$$

其中 $\sigma(z)=1/(1+\exp(-z))$ 是逻辑 sigmoid 函数。对所有偏好对求和得到

$$
\displaystyle\!\!\!\mathcal{L}_{\textup{{PMLE}}}(\pi)=\frac{1}{n}\sum_{(x,a^{+},a^{-})\in D_{n}}\!\!\!\!\!\!-\ln\sigma\left(\gamma\ln\frac{\pi(a^{+}\mid x)}{\pi(a^{-}\mid x)}\right).
$$

通过最小化 $\mathcal{L}_{\textup{{PMLE}}}$，我们鼓励 $\pi$ 对响应 $a^{+}$ 相对于 $a^{-}$ 赋予更高概率。注意，在实践中，我们很少从头开始学习策略 $\pi$；相反，我们通常优化微调模型，称为**参考策略** ${\pi_{0}}$。因此，引入 KL 惩罚以保持 $\pi$ 接近 ${\pi_{0}}$ 用于对齐是自然的：$\beta\cdot\KL(\pi(x)\|{\pi_{0}}(x))$。将所有内容整合在一起，我们用于分布学习的 PMLE 目标函数为

$$
\displaystyle\mathcal{L}_{\textup{{PMLE}},\beta}(\pi)\mathrel{\mathop{\mathchar 58\relax}}=\mathcal{L}_{\textup{{PMLE}}}(\pi)+\beta\KL(\pi(x)\|{\pi_{0}}(x)).
$$

##### 注记

回顾 DPO [^42] 最小化目标函数

$$
\displaystyle\sum_{D_{n}}-\ln\sigma\left(\gamma\ln\frac{\pi(a^{+}\mid x)}{\pi(a^{-}\mid x)}-\gamma\ln\frac{{\pi_{0}}(a^{+}\mid x)}{{\pi_{0}}(a^{-}\mid x)}\right).
$$

与 (5) 相比，DPO 目标函数没有显式正则化项，如果策略类 $\Pi$ 足够有表达力，这可能导致不良行为。具体而言，[^14] 证明 DPO 可能收敛到退化分布。此外，[^45] 表明 DPO 依赖于强覆盖假设：如果 ${\pi_{0}}$ 未能完全覆盖相关分布，DPO 可能产生分布外响应，使其奖励估计不准确。与具有 KL 项以保持在 ${\pi_{0}}$ 支撑内的 RLHF 不同，DPO 可能对 ${\pi_{0}}$ 永远不会选择的响应赋予非零概率，从而削弱性能保证。相比之下，我们的 PMLE 目标函数 (5) 包含显式 KL 项，有效地规避了上述缺陷。

##### 收敛保证

在第 2 节的假设下，我们展示了正向 KL 的上界。在整篇论文中，仅依赖于 $R$ 的常数被隐藏。所有证明推迟到附录 C。

###### 定理 4

PMLE 估计 $\hat{\pi}=\argmin_{\pi\in\Pi}\mathcal{L}_{\textup{{PMLE}}}(\pi)$ 以至少 $1-\delta$ 的概率满足

$$
\displaystyle\EE_{x\sim\mathcal{D}}\left[\KL(\pi^{*}(x)\|\hat{\pi}(x))\right]\lesssim\frac{C_{\Pi}}{\gamma^{2}}\cdot\frac{\ln(|\Pi|/\delta)}{n}.
$$

证明在附录 C.2 中提供，部分受 [^1] 启发，但我们利用 Schulman 技巧 [^44] 后跟二次近似来获得 $1/n$ 速率，而不是直接遵循其证明所得的 $1/\sqrt{n}$。还要注意，(7) 的左侧等价于 $\mathcal{X}\times\mathcal{A}$ 上诱导的**联合**分布之间的 KL 散度：$\KL(\mathcal{D}(x)\pi^{*}(a\mid x)\|\mathcal{D}(x)\hat{\pi}(a\mid x))$。

为简单起见，我们在这里以及后续所有分析中假设 $\beta=0$，以证明纯粹从通过 (3) 考虑偏好反馈推导的目标函数已足以学习真实分布 $\pi^{*}$。尽管如此，我们认为从良好对齐的 ${\pi_{0}}$ 开始可以通过减轻常数对 $R$ 的依赖来改进保证，我们将此留待未来工作。

接下来，我们将注意力转向需要显式奖励模型的对齐方法。根据我们的理念，我们强调这些方法是从分布学习视角而非奖励最大化推导的。

### 4 偏好蒸馏方法

自 RLHF 普及以来，奖励建模的使用在研究社区中变得流行，并产生了各种扩展 [^9]。虽然奖励模型在 RLHF 目标函数 (1) 中的主要作用是将对齐视为强化学习问题，但最近的研究尝试将奖励模型用于监督学习损失，即不需要强化学习来解决的目标函数 [^21] [^14]。这些努力可以被视为从奖励模型**蒸馏信息**，正如 [^14] 所指出的。这些方法的主要好处是它们可以避免强化学习算法，后者通常收敛较慢。虽然奖励模型训练相比纯似然方法（如 DPO 或我们的 PMLE）是额外负担，但这样做的计算成本通常相当低，因为在现有 LLM 的冻结主干上训练浅层网络通常就足够了。

##### 奖励模型

由于我们的偏好模型 (3)，学习奖励模型 $R\mathrel{\mathop{\mathchar 58\relax}}\mathcal{X}\times\mathcal{A}\to\mathbb{R}$ 等价于学习语言模型 $\pi$，然后设置 $R(x,a)=\gamma\ln\pi(a\mid x)$ 加上一个加性常数。相反，给定奖励模型 $R(x,a)$，我们可以通过以下方式估计语言模型

$$
\displaystyle\pi(a\mid x)\propto\exp(\gamma^{-1}R(x,a)),\quad\forall x\in\mathcal{X}.
$$

注意，这是一个在一般情况下从中采样计算上不可行的模型。形式上，我们假设给定奖励模型类 $\mathcal{R}$，包含奖励 $R\mathrel{\mathop{\mathchar 58\relax}}\mathcal{X}\times\mathcal{A}\rightarrow\mathbb{R}$，并学习：

$$
\displaystyle\hat{R}=\argmin_{R\in\mathcal{R}}~\sum_{D_{n}}-\ln\sigma(R(x,a^{+})-R(x,a^{-})).
$$

这在 (8) 下等价于 PMLE 目标函数，但带有约束 $R\in\mathcal{R}$。

##### 偏好蒸馏

一种流行的奖励蒸馏方法是 REBEL 算法 [^18]。受全策略假设下 RLHF 解的刻画启发 [^42]，REBEL 旨在从配对响应的相对奖励值中提取信息，通过优化平方损失来强制执行 $\ln\frac{\pi(a^{+}\mid x)/{\pi_{0}}(a^{+}\mid x)}{\pi(a^{-}\mid x)/{\pi_{0}}(a^{-}\mid x)}\approx\eta(\hat{R}(x,a^{+})-\hat{R}(x,a^{-}))$

$$
\displaystyle\sum_{D_{n}}\Big(\ln\frac{\pi(a^{+}|x){\pi_{0}}(a^{-}|x)}{\pi(a^{-}|x){\pi_{0}}(a^{+}|x)}-\eta(\hat{R}(x,a^{+})-\hat{R}(x,a^{-}))\Big)^{2}
$$

其中 $\eta>0$ 控制奖励信号的强度。在我们的假设中，奖励模型可以被视为 $\gamma\ln\pi^{*}(a\mid x)$ 的偏移版本，因此我们可以优化 (10) 而不使用 ${\pi_{0}}$ 项，用 $\gamma^{-1}$ 替换 $\eta$。然而，(10) 中使用平方损失从统计角度来看并不充分合理，且不清楚平方损失是否应优于其他损失，例如绝对损失。

那么什么是合适的误差度量？我们的框架告诉我们，学习奖励模型等同于学习偏好模型。换句话说，我们训练了一个**偏好模拟器**：一个非生成语言模型估计 $\widetilde{\pi}(a\mid x)\propto\exp(\gamma^{-1}\hat{R}(x,a))$，可以从中对任何响应对采样偏好，如 $y\sim{\mathrm{Bernoulli}}(\PP_{\widetilde{\pi}}(a\succ b\mid x))$。将其插入 PMLE 将产生自然的分布学习目标函数。然而，这个过程引入了额外的随机性，可能妨碍优化。相反，观察到我们可以评估 PMLE 目标函数的期望，并用**期望**偏好替换离散标签 $y$

$$
\displaystyle\PP_{\widetilde{\pi}}(a^{+}\succ a^{-}\mid x)
$$

$$
\displaystyle={\frac{\widetilde{\pi}(a^{+}\mid x)^{\gamma}}{\widetilde{\pi}(a^{+}\mid x)^{\gamma}+\widetilde{\pi}(a^{-}\mid x)^{\gamma}}}=\sigma(\hat{R}(x,a^{+})-\hat{R}(x,a^{-})).
$$

然后，关于这个合成偏好最小化对数损失等价于最小化二元偏好分布 $\mathrm{Bern}(\PP_{\widetilde{\pi}}(a^{+}\succ a^{-}\mid x))$ 和 $\mathrm{Bern}(\PP_{\pi}(a^{+}\succ a^{-}\mid x))$ 之间的 KL 散度：

$$
\displaystyle\mathcal{L}_{\textup{{Distill}}}(\pi)\mathrel{\mathop{\mathchar 58\relax}}={\frac{1}{n}}\sum_{D_{n}}\KL\!\big(\mathrm{Bern}(\PP_{\widetilde{\pi}}(a^{+}\succ a^{-}\mid x))\,\|\,\mathrm{Bern}(\PP_{\pi}(a^{+}\succ a^{-}\mid x))\big)+\mathrm{const.}
$$

与 PMLE（第 3 节）一样，在实践中我们添加 KL 正则化项：

$$
\displaystyle\mathcal{L}_{\textup{{Distill}},\beta}(\pi)\mathrel{\mathop{\mathchar 58\relax}}=\mathcal{L}_{\textup{{Distill}}}(\pi)+\beta\KL(\pi\|{\pi_{0}}).
$$

我们指出，奖励模型训练 (9) 和偏好蒸馏 (11) 的数据可以来自不同数据集；我们的理论分析很容易适应。

##### 收敛保证

由奖励模型类 $\mathcal{R}$ 诱导的（非生成）语言模型族定义为

$$
\displaystyle\mathcal{P}_{\gamma}(\mathcal{R})\mathrel{\mathop{\mathchar 58\relax}}=\big\{\pi\mathrel{\mathop{\mathchar 58\relax}}\pi(a\mid x)\propto\exp(\gamma^{-1}R(x,a)),\forall a,x\in\mathcal{X}\text{ 对某个 }R\in\mathcal{R}\big\}.
$$

###### 假设 5

奖励诱导的语言模型类 $\mathcal{P}_{\gamma}(\mathcal{R})\subseteq\Pi$。

这个假设与生成器-验证器差距相关，后者非正式地表述为验证给定答案是否正确比生成正确答案更容易 [^30] [^53]。这种差距意味着从学习理论角度来看，$\mathcal{R}$ 比 $\Pi$ 更容易学习（$|\mathcal{R}|\ll|\Pi|$），并且被推测对实践中的 LLM 成立 [^47]。假设 5 也可以通过奖励模型通常构建在监督微调模型（冻结）主干之上的事实来证明。记 $C_{\mathcal{R}}\mathrel{\mathop{\mathchar 58\relax}}=C_{\mathcal{P}_{\gamma}(\mathcal{R})}$ 为诱导子类的广义覆盖系数，在假设 5 下有 $C_{\mathcal{R}}\leq C_{\Pi}$。

###### 定理 6

偏好蒸馏估计 $\hat{\pi}=\argmin\limits_{\pi\in\Pi}\mathcal{L}_{\textup{{Distill}}}(\pi)$ 以至少 $1-\delta$ 的概率满足

$$
\displaystyle\EE_{x\sim\mathcal{D}}\left[\KL(\pi^{*}(x)\|\hat{\pi}(x))\right]\lesssim\frac{C_{\Pi}}{\gamma^{2}}\cdot\frac{\ln(|\Pi|/\delta)}{n}.
$$

证明见附录 C.3。

##### 蒸馏的好处

上述速率 $n\gtrsim C_{\Pi}\ln|\Pi|$ 等于 PMLE (7) 的速率，因为我们假设使用从 $\mu$ 生成的响应 $D_{n}$ 学习 $\hat{\pi}\in\Pi$，与奖励模型相同。然而，如果我们能够访问“更强”的基础模型 $\pi_{0}$，自然地使用来自 $\pi_{0}$ 的响应学习 $\hat{\pi}$ 更为合理。在这种情况下，我们获得的速率为 $\gamma^{-2}(C_{0}\ln|\Pi|+C_{\mathcal{R}}\ln|\mathcal{R}|)$，其中 $C_{0}$ 是针对 $\pi_{0}$ 而非 $\mu$ 的略微修改的覆盖系数。如果 $\pi_{0}$ 相对良好对齐使得 $C_{0}<C_{\Pi}$，这比 PMLE 速率有所改进，从而严格建立了使用强基础模型进行蒸馏的好处。形式陈述和证明见附录 C.3 中的定理 14。
### 5 反向 KL 最小化方法

我们提出的两种方法都是最大化偏好似然，最终在正向 KL 散度 $\EE_{x}[\KL(\pi^{*}(x)\|\hat{\pi}(x))]$ 上享有理论保证。然而，也可以通过最小化反向 KL 散度 $\EE_{x}[\KL(\hat{\pi}(x)\|\pi^{*}(x))]$ 来学习分布 $\pi^{*}$。反向 KL 具有众所周知的“模式寻找”行为，这与正向 KL 的“模式覆盖”行为相对。这种模式寻找行为倾向于找到能生成真实内容的分布，在图像生成模型中受到青睐 [^19] [^33]。

在本节中，我们探索在假设 (3) 下对齐的反向 KL 形式，结果表明它是原始 RLHF 框架 (1) [^46] [^39] 的泛化。关于目标语言模型 $\pi^{*}$ 直接最小化反向 KL 会得到：

$$
\displaystyle\hat{\pi}=\argmin_{\pi\in\Pi}\mathbb{E}_{x\sim\mathcal{D}}\big[\mathbb{E}_{a\sim\pi(x)}[-\ln\pi^{*}(a\mid x)]-H(\pi(x))\big],
$$

其中 $H(\pi(x))$ 是 $\pi(x)$ 的香农熵。然而，这需要形如 $-\ln\pi^{*}$ 的奖励，而这正是我们试图估计的对象。为了解决这个问题，我们提出从一个代理语言模型类中找到插件估计器，这类模型更容易训练但更难采样。具体来说，我们确定 $\widetilde{\pi}=\argmin_{\pi\in\mathcal{P}_{\gamma}(\mathcal{R})}\mathcal{L}_{\textup{{PMLE}}}(\pi)$（带有适当的正则化），这等价于通过 (9) 获得 $\hat{R}$，然后像之前一样设置 $\widetilde{\pi}(a\mid x)\propto\exp(\gamma^{-1}\hat{R}(x,a))$。然后我们可以将学到的 $\widetilde{\pi}$ 插入到 (14) 中的 $\pi^{*}$ 位置，得到目标函数

$$
\displaystyle\argmin_{\pi\in\Pi}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathbb{E}_{a\sim\pi(x)}\big[-\gamma^{-1}\hat{R}(x,a)\big]-H(\pi(x))\right].
$$

归一化常数在实践中难以计算，但由于我们只需要相对奖励进行优化，它自然消失了。最后，我们再次添加关于 ${\pi_{0}}$ 的 KL 正则化项：

$$
\displaystyle\mathcal{L}_{\textup{{RKL}},\beta}(\pi)\mathrel{\mathop{\mathchar 58\relax}}
$$

$$
\displaystyle=\frac{1}{n}\sum\limits_{(x,\cdot,\cdot)\in D_{n}}-\EE_{a\sim\pi(x)}\big[\hat{R}(x,a)\big]-\gamma H(\pi(x))+\beta\KL(\pi(x)\|{\pi_{0}}(x))
$$

其中 $\beta$ 和 $\gamma$ 分别控制策略熵和 KL 正则化项的相对权重。在实践中，如同标准的 RLHF 流程 [^39] [^5]，首先从配对偏好中拟合奖励模型 $\hat{R}$ 来近似底层真实奖励 $R^{*}$，然后应用 RL 算法（例如 PPO [^43]）来最小化目标函数 (15)。

##### 与 RLHF 的关系

表面上看，RKL 可以被视为 RLHF 的泛化，因为当 $\gamma=0$ 时 RKL 目标等于 RLHF 目标。相反，在我们的偏好假设 (3) 下，RLHF 本身可以被解释为在 $\gamma\rightarrow 0$ 极限下最小化反向 KL。因此，我们关于 RKL 的结果可以被视为为 RLHF 目标 (1) **提供了理论正当性**，该目标一直被广泛视为对齐的黄金标准 [^46] [^5] [^42]，**同时也提供了一个小修正**。注意，考虑到最大熵 RL 可以被视为反向 KL 最小化 [^68]，这种与 RLHF 的联系可能并不令人意外。

##### RLHF 的困境

如引言中所讨论的，假设 $\mu$ 具有足够的覆盖性，RLHF 面临以下渐近二分法：

1. 欠拟合（对所有 $n$ 固定 $\beta>0$）：学到的语言模型是一个适当的分布，但不能离参考策略 $\pi_{0}$ 太远。
2. 退化（令 $\beta=\beta_{n}\downarrow 0$ 当 $n\rightarrow\infty$）：学到的语言模型坍缩到退化解。

两者都不理想。这一现象也可以在图 1 所示的玩具实验中看到。<sup>4</sup> 如定理 7 所证明的，在我们的方法中，正向 KL 确实随着样本量 $n$ 的增加趋于零，而对于 RLHF，它最终饱和在一个非零值。

##### 收敛保证

使用目标函数 $\mathcal{L}_{\textup{{RKL}}}\mathrel{\mathop{\mathchar 58\relax}}=\mathcal{L}_{\textup{{RKL}},0}$，我们确实能够获得以下保证（在附录 C.4 中证明）：

###### 定理 7

反向 KL 估计 $\hat{\pi}=\argmin\limits_{\pi\in\Pi}\mathcal{L}_{\textup{{RKL}}}(\pi)$ 以至少 $1-\delta$ 的概率满足

$$
\displaystyle\EE_{x\sim\mathcal{D}}\left[\KL(\pi^{*}(x)\|\hat{\pi}(x))\right]\lesssim\frac{\ln(|\Pi|/\delta)}{n}+\frac{C_{\mathcal{R}}}{\gamma^{2}}\cdot\frac{\ln(|\mathcal{R}|/\delta)}{n}.
$$

##### 为什么反向 KL 获得更好的界？

反向 KL 形式对**正向** KL 产生了改进的上界，该上界依赖于 $\mathcal{P}_{\gamma}(\mathcal{R})$ 的覆盖系数，而不是像之前的界 (7)、(13) 那样依赖于 $\Pi$。特别是，在假设 5 下，$\ln|\Pi|$ 和 $C_{\mathcal{R}}\ln|\mathcal{R}|$ 可能都远小于 $C_{\Pi}\ln|\Pi|$，甚至小于改进的偏好蒸馏保证 $C_{0}\ln|\Pi|$ (23)。

敏锐的读者可能会疑惑：反向 KL 如何能避免 $C_{\Pi}$（或 $C_{0}$），而偏好蒸馏却不能，即使它们都利用了奖励模型？原因在于偏好蒸馏的策略学习步骤仍然依赖于从 $\mu$（或 $\pi_{0}$）采样的响应对 $(a^{+},a^{-})$，而反向 KL 仅在轻量级的奖励建模步骤中使用响应。此外，界定正向 KL 而非反向 KL（即使目标函数是从最小化反向 KL 导出的）使我们能够避免比较 $\hat{\pi}$ 相对于 $\mu$ 的覆盖性，而定义 3 并不保证这一点。尽管如此，正向和反向 KL 误差仍然可以比较（但会有一个关于 $R$ 的指数常数），如我们在命题 15 中所示。

另一种方法是直接优化正向 KL：$\argmin_{\pi\in\Pi}\sum_{(x,\cdot,\cdot)\sim D_{n}}\KL(\widetilde{\pi}(x)\|\pi(x))$。然而，这在计算上是不可行的，因为它需要评估归一化常数或从未归一化分布中采样。我们将详细讨论推迟到附录 B。

表 2：Pythia 2.8B 和 6.9B 在 TL;DR 数据集上的结果。胜率由 GPT-4 评估，奖励模型（RM）分数由训练的奖励模型评估。我们报告三个随机种子的均值/标准差。

<table><tbody><tr><td></td><td colspan="3">Pythia 2.8B</td><td colspan="3">Pythia 6.9B</td></tr><tr><td>算法</td><td>胜率 <math><semantics><mrow><mo>(</mo><mo>↑</mo><mo>)</mo></mrow> <annotation>(\uparrow)</annotation></semantics></math></td><td>RM 分数 <math><semantics><mrow><mo>(</mo><mo>↑</mo><mo>)</mo></mrow> <annotation>(\uparrow)</annotation></semantics></math></td><td><math><semantics><mrow><mi>KL</mi> <mrow><mo>(</mo><mi>π</mi> <mo>∥</mo> <msub><mi>π</mi> <mn>0</mn></msub><mo>)</mo></mrow> <mrow><mo>(</mo><mo>↓</mo><mo>)</mo></mrow></mrow> <annotation>\KL(\pi\|{\pi_{0}})(\downarrow)</annotation></semantics></math></td><td>胜率 <math><semantics><mrow><mo>(</mo><mo>↑</mo><mo>)</mo></mrow> <annotation>(\uparrow)</annotation></semantics></math></td><td>RM 分数 <math><semantics><mrow><mo>(</mo><mo>↑</mo><mo>)</mo></mrow> <annotation>(\uparrow)</annotation></semantics></math></td><td><math><semantics><mrow><mi>KL</mi> <mrow><mo>(</mo><mi>π</mi> <mo>∥</mo> <msub><mi>π</mi> <mn>0</mn></msub><mo>)</mo></mrow> <mrow><mo>(</mo><mo>↓</mo><mo>)</mo></mrow></mrow> <annotation>\KL(\pi\|{\pi_{0}})(\downarrow)</annotation></semantics></math></td></tr><tr><td>DPO</td><td>47.3 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.01</td><td>2.66 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>64.35 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.68</td><td>55.6 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.53</td><td>6.09 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>54.45 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.82</td></tr><tr><td>PMLE</td><td>49.1 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.43</td><td>2.38 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>31.42 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.79</td><td>59.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.74</td><td>6.27 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>24.20 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.92</td></tr><tr><td>REBEL</td><td>71.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.26</td><td>2.86 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>25.67 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.61</td><td>82.5 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.93</td><td>7.64 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>29.80 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.71</td></tr><tr><td>偏好蒸馏</td><td>73.8 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.47</td><td>2.91 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>28.38 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.82</td><td>83.9 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.59</td><td>7.66 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.02</td><td>29.05 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.02</td></tr><tr><td>RLHF</td><td>72.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.67</td><td>2.95 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.05</td><td>24.94 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.12</td><td>82.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.92</td><td>7.85 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.03</td><td>24.38 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.77</td></tr><tr><td>反向 KL</td><td>73.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.93</td><td>3.03 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.04</td><td>25.91 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.94</td><td>84.0 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.01</td><td>7.77 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.01</td><td>24.12 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.63</td></tr></tbody></table>

表 3：UltraFeedback 上的通用对话结果。在 MT-Bench、AlpacaEval 2.0（长度控制和原始胜率，%）和 Arena Hard v0.1（%）上的 LLM 评判评估。粗体表示每对（DPO/PMLE 和 REBEL/偏好蒸馏）中的获胜者。

<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">MT-Bench</td><td colspan="2">AlpacaEval 2.0</td><td rowspan="2">Arena Hard</td></tr><tr><td>LC 胜率</td><td>胜率</td></tr><tr><td>基线（LLaMA-3-8B-Instruct）</td><td>8.10</td><td>29.62</td><td>28.71</td><td>26.8</td></tr><tr><td>DPO</td><td>8.32</td><td>44.72</td><td>43.94</td><td>45.1</td></tr><tr><td>PMLE</td><td>8.30</td><td>53.23</td><td>50.43</td><td>46.5</td></tr><tr><td>REBEL</td><td>8.17</td><td>52.04</td><td>44.99</td><td>39.8</td></tr><tr><td>偏好蒸馏</td><td>8.18</td><td>54.49</td><td>46.86</td><td>41.9</td></tr></tbody></table>

### 6 实验

在本节中，我们展示实验结果，表明我们的框架在实践中产生了具有竞争力的性能。具体来说，我们将提出的方法（PMLE、反向 KL 和偏好蒸馏）与它们成熟的基线（DPO、RLHF 和 REBEL）在一系列语言任务上进行比较。

<svg id="S6.p2.pic1" height="105.11" overflow="visible" version="1.1" viewBox="0 0 477.38 105.11" width="477.38"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,105.11) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#D9D9D9;" fill="#D9D9D9" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.15 L 0 100.96 C 0 103.25 1.86 105.11 4.15 105.11 L 473.23 105.11 C 475.52 105.11 477.38 103.25 477.38 100.96 L 477.38 4.15 C 477.38 1.86 475.52 0 473.23 0 L 4.15 0 C 1.86 0 0 1.86 0 4.15 Z"></path></g><g style="--ltx-fill-color:#F2F2F2;" fill="#F2F2F2" fill-opacity="1.0"><path style="stroke:none" d="M 4.15 4.15 L 4.15 100.96 C 4.15 102.48 5.39 103.72 6.92 103.72 L 470.46 103.72 C 471.99 103.72 473.23 102.48 473.23 100.96 L 473.23 4.15 C 473.23 2.62 471.99 1.38 470.46 1.38 L 6.92 1.38 C 5.39 1.38 4.15 2.62 4.15 4.15 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 23.84 65.7)"><foreignObject style="--ltx-fo-width:31.06em;--ltx-fo-height:1.89em;--ltx-fo-depth:3.79em;font-size:10pt;" height="78.72" overflow="visible" transform="matrix(1 0 0 -1 0 26.21)" width="429.78"><span id="S6.p2.pic1.1" style="width:31.06em;"><span id="S6.p2.pic1.1.1"><span id="S6.I1.i1" style="list-style-type:none;"><span id="S6.I1.i1.p1"><span id="S6.I1.i1.p1.1"><span id="S6.I1.i1.p1.1.1" style="--ltx-fg-color:#000000;">评估点。</span><span id="S6.I1.i1.p1.1.3" style="--ltx-fg-color:#000000;">纯粹从我们的分布学习假设 (3) 导出的理论正当化目标函数能否在实践中与强有力的经验基线（DPO、REBEL、RLHF）保持竞争力？</span></span></span></span></span></span></foreignObject></g></g></svg>

#### 6.1 TL;DR 摘要生成

我们遵循 [^18] [^45] 的标准 TL;DR 设置 [^46]，使用 Pythia-2.8B 和 Pythia-6.9B [^6]；完整的训练和评估细节推迟到附录 D.2。我们报告相对于人类参考的 GPT-4 胜率（在 600 个测试样本上）、训练的奖励模型分数以及 $\KL(\pi\|{\pi_{0}})$。

##### 结果

结果见表 2。对于 Pythia-2.8B 和 Pythia-6.9B，我们的分布学习目标在此实验中实现了比各自基线更高的胜率，其中胜率是语言模型质量的最直接衡量标准。此外，由于 PMLE 使用在线数据实现 KL 正则化项，与仅依赖离线数据集的 DPO 相比，它实现了更低的 KL 项；这一发现与 [^45] 报告的结果一致。至于 RLHF 和 REBEL，两种方法在每个实验中使用相同的 KL 惩罚，自然导致相似的 $\KL(\pi\|{\pi_{0}})$ 值。总体而言，这些实验提供了证据，表明从我们的假设 (2) 导出的算法可以与流行的基线保持竞争力。

#### 6.2 通用对话

我们在 UltraFeedback [^10] 上训练 LLaMA-3-8B-Instruct [^20]，使用 ArmoRM-Llama3-8B-v0.1 [^52] 作为奖励模型，遵循 [^18]；完整设置在附录 D.3 中。我们专注于基于似然的一对（PMLE vs. DPO）和基于奖励蒸馏的一对（偏好蒸馏 vs. REBEL），因为这些方法在算法上与它们的基线不同，而 RKL 只是在与 RLHF 相同的 PPO 更新中添加了一个熵项。对话质量通过三个 LLM 评判基准评估：MT-Bench [^67]、AlpacaEval 2.0 [^11] 和 Arena Hard [^29]。

##### 结果

我们的主要目标是检验从分布学习视角导出的目标函数能否在现实对话设置中与强有力的经验基线保持竞争力。结果令人鼓舞：PMLE 和偏好蒸馏在评估的基准测试中相对于 DPO 和 REBEL 表现良好，在 AlpacaEval 2.0 和 Arena Hard 上对齐质量提升尤其显著，同时在 MT-Bench 上保持可比性。结合 TL;DR 结果，这些发现表明我们的理论驱动目标不仅与强基线具有竞争力，而且在改进实际对齐性能方面也很有前景。

### 7 结论

我们将对齐形式化为从单一明确假设 (3)（关于偏好如何与目标语言模型相关）进行分布学习，并导出了三种有原则的方法（PMLE、偏好蒸馏、RKL），它们修正并泛化了现有方法，具有 $O(1/n)$ 的正向 KL 保证，并伴随实验结果显示在多个 TL;DR 和对话评估设置中具有竞争力的性能和提升。未来方向包括跨领域比较模式寻找与模式覆盖目标、将框架扩展到替代散度和偏好模型，以及通过对 ${\pi_{0}}$ 的更强假设来收紧对 $R$ 的指数依赖性。

### 参考文献

## 附录

### 附录 A 相关工作

##### 基于强化学习的偏好优化

偏好优化中被广泛采用的范式是基于人类反馈的强化学习（RLHF）。在这个框架中，首先在从人类标注者收集的偏好数据集上训练一个奖励模型——实际上充当分类器，随后利用学到的奖励模型运行 RL 算法，如 PPO [^9] [^69]。RLHF 及其变体在训练 ChatGPT [^38] 等著名大语言模型方面发挥了重要作用，并在文本摘要、问答、指令遵循和文本到图像生成等多样化应用中取得了显著成功 [^46] [^36] [^39] [^28] [^31]。我们建议感兴趣的读者参阅 [^26] 关于 RLHF 的最新专门综述。

##### 不使用强化学习和奖励模型

直接策略优化（DPO）通过将每个偏好对的对数比率作为训练信号，直接用单一对比交叉熵损失训练策略，从而摒弃了显式奖励模型 [^42]。这种无 RL 目标被证明能够在不需要奖励模型、价值网络或在策略采样的情况下匹配基于 PPO 的 RLHF，并产生了诸如蒸馏 DPO [^49]、Cal-DPO [^54]、扩散 DPO [^51]、$\Psi$ PO [^4]、SLiC/SLiC-HF [^66] [^65]、GPO [^48]、$\chi$ PO [^24]、R-DPO [^40]、ODPO [^3]、SimPO [^34]、RRHF [^59]、KTO [^13]、ORPO [^22] 等众多变体。

同时，这种从偏好标签直接优化的方法在某些维度上被注意到不如传统 RLHF。一个挑战源于完全依赖离线数据集，这可能导致分布外响应。这可能是由于训练期间缺乏足够的在策略交互 [^45]。已经提出了一些混合方法来克服这个问题：迭代 DPO 使用标注的在线偏好进行迭代训练 [^32]，HyPO 结合离线数据进行偏好优化和在线数据进行 KL 正则化 [^45]，在线 DPO 利用快慢追逐来模拟竞争 [^41]。

##### 不使用强化学习但使用奖励模型

另一种突出的偏好优化方法是奖励蒸馏。这一工作路线旨在将奖励模型的偏好信息直接蒸馏到策略中。如第 4 节所讨论的，REBEL 目标 [^18] 使用简单的平方损失目标将两个响应的似然对数比回归到奖励差异上，这一过程使用在策略响应的批次重复进行。来自 [^14] 的奖励蒸馏可以被视为 REBEL 的简化版本，我们只使用偏好数据集中的响应。DRDO 通过联合匹配预言奖励并学习人类偏好，在一次传递中学习奖励模型和策略 [^37]。最后，[^64] 开发了一个大语言模型蒸馏流程来同时蒸馏数据和奖励。

##### 偏好优化的理论分析

[^61] 研究了基于 MLE 的奖励模型的离策略偏好 RL，类似于我们的，但仅获得了关于最大化策略价值的保证。[^56] 提出了一种探索性的 DPO 版本，被证明能以有利的覆盖参数实现 $\widetilde{O}(\sqrt{T})$ 遗憾。[^62] 提出了一种在线直接对齐算法，也达到了 $\widetilde{O}(\sqrt{T})$ 遗憾。[^58] 在线性参数化奖励模型下导出了在线和离线版本 RLHF 的遗憾界；另见 [^16] 对具有线性-softmax 策略的 RL 的理论分析。[^8] 引入了 VPO，一种用于在线和离线 RLHF 的价值正则化 DPO 类型目标，并在线性奖励下证明了遗憾界。$\chi$ PO 算法被证明在较弱的单策略集中性下也能实现最优样本复杂度（同样以遗憾衡量）[^24]。

[^1] 的工作与我们的论文最相关，特别是 PMLE（第 3 节）：他们对最小化 DPO 类型目标的离线 RLHF 变体进行了理论分析，并展示了关于最优策略 $\pi^{*}$ 的正向 KL 界。然而，这种形式化不是源于分布学习视角，而仅仅是他们强可实现性假设（假设 3.2）的副产品。此外，他们的上界对超额风险 $\varepsilon=L(\pi)-L(\pi^{*})$ 有平方根依赖性，当应用于我们的框架时产生 $1/\sqrt{n}$ 的统计速率。相比之下，通过附录 C 中更仔细的分析，我们获得了 $1/n$ 的改进速率。

### 附录 B 关于反向 KL 的补充说明

##### 先验平滑

与标准 RLHF 目标的一个关键区别在于我们的形式化如何平衡奖励最大化与**先验平滑**。为了明确比较，我们为一个玩具对齐问题说明附加熵项的效果。考虑在单纯形 $\Delta_{K}=\{\mathrm{\mathbf{p}}\in\mathbb{R}_{\geq 0}^{d}\mathrel{\mathop{\mathchar 58\relax}}\sum_{k=1}^{K}p_{k}=1\}$ 上学习一个 $K$-分类分布，这可以被视为响应长度为 1、词汇表大小为 $K$ 的无上下文语言模型。假设我们给定一个固定向量 $\mathrm{\mathbf{p}}_{0}\in\Delta_{K}$ 作为参考模型和一个学到的奖励函数 $\hat{\mathrm{\mathbf{r}}}=(r_{1},\cdots,r_{K})$。在温度为 $\beta+\gamma$ 的标准 RLHF 方法 (1) 中，对所有 $k\in[K]$，最优策略由 $\hat{p}_{k}^{\textup{{RLHF}}}\propto p_{0,k}\exp(\frac{r_{k}}{\beta+\gamma})$ 给出。相比之下，我们的反向 KL 目标 (15) 可以重新排列为

$$
\displaystyle\mathcal{L}_{\textup{{RKL}},\beta}(\mathrm{\mathbf{p}})
$$

$$
\displaystyle=-\mathrm{\mathbf{p}}\cdot\hat{\mathrm{\mathbf{r}}}-\gamma H(\mathrm{\mathbf{p}})+\beta\KL(\mathrm{\mathbf{p}}||\mathrm{\mathbf{p}}_{0})
$$

$$
\displaystyle=-\mathrm{\mathbf{p}}\cdot\hat{\mathrm{\mathbf{r}}}+(\beta+\gamma)\KL(\mathrm{\mathbf{p}}||\mathrm{\mathbf{p}}_{0}^{\alpha})+\text{const.}
$$

其中 $\alpha\mathrel{\mathop{\mathchar 58\relax}}=\frac{\beta}{\beta+\gamma}$，产生策略 $\hat{p}_{k}\propto p_{0,k}^{\alpha}\exp(\frac{r_{k}}{\beta+\gamma})$。附加指数 $\alpha\in(0,1)$ 作用是将先验从 $\mathrm{\mathbf{p}}_{0}$ 平滑到 $\mathrm{\mathbf{p}}_{0}^{\alpha}$，为初始概率低的动作分配相对更多的质量。这促进了探索，特别是对于在基础策略下不太可能的动作，使得估计策略 $\hat{\mathrm{\mathbf{p}}}$ 即使 $\mathrm{\mathbf{p}}_{0}$ 是退化分布也不会太接近退化分布。

##### 正向 KL 的不可行性

另一种方法是直接优化正向 KL：

$$
\displaystyle\textstyle\argmin_{\pi\in\Pi}\sum_{(x,\cdot,\cdot)\sim D_{n}}\KL(\widetilde{\pi}(x)\|\pi(x)).
$$

在这里，我们不使用 $(a^{+},a^{-})$，因此对 $\mu$ 的依赖消失了，我们不会为 $C_{\Pi}$ 付出代价，类似于定理 7。然而，我们如何计算正向 KL？由于语言模型中 $\mathcal{A}$ 的庞大规模，直接计算是不可行的。相反，人们可能尝试从 $\widetilde{\pi}(x)$ 采样并执行随机优化；然而，这样的采样是不可行的，因为我们只能访问未归一化版本 $\exp(\gamma^{-1}\hat{R}(x,\cdot))$。另一个尝试是使用以下事实

$$
\displaystyle\KL(\widetilde{\pi}(x)\|\pi(x))=\EE_{a\sim\pi(x)}\bigg[\frac{\widetilde{\pi}(a\mid x)}{\pi(a\mid x)}\ln{\frac{\widetilde{\pi}(a\mid x)}{\pi(a\mid x)}}\bigg].
$$

虽然我们不必从 $\widetilde{\pi}(x)$ 采样，但我们现在必须评估 $\widetilde{\pi}(a\mid x)$ 的值，而这再次是不可行的。

### 附录 C 理论保证

#### C.1 辅助引理

我们需要以下基本结果。

###### 引理 8

对所有 $a,b\in\mathbb{R}$，成立 $|\sigma(a)-\sigma(b)|\geq\frac{1}{4}e^{-(|a|\vee|b|)}|a-b|$。

###### 证明

回顾 $\sigma(z)=1/(1+\exp(-z))$ 是逻辑 sigmoid 函数。$\sigma^{\prime}$ 是对称的，因此

$$
\sigma^{\prime}(z)=\sigma^{\prime}(|z|)={\frac{1}{1+e^{|z|}}}{\frac{1}{1-e^{-|z|}}}\geq{\frac{1}{2(1+e^{|z|})}}\geq{\frac{1}{4}}e^{-|z|}.
$$

由于对称性，假设 $b>a$ 就足够了。那么，

$$
\displaystyle\sigma(b)-\sigma(a)=\int_{a}^{b}\sigma^{\prime}(z)\dif z\geq\int_{a}^{b}{\frac{1}{2}}e^{-|z|}\dif z\geq{\frac{1}{4}}e^{-(|a|\vee|b|)}(b-a)
$$

如所需。∎

接下来两个引理将允许我们在对数比率的期望（即 KL 散度）和平方对数比率之间转换。让我们定义辅助函数

$$
\displaystyle\psi(z)\mathrel{\mathop{\mathchar 58\relax}}=\frac{z-1-\ln z}{(\ln z)^{2}}.
$$

###### 引理 9

对于 $r_{\max}>1$，对所有 $r\in\lparen 0,r_{\max}]$ 成立

$$
\displaystyle r-1-\ln r\leq({\frac{1}{2}}\vee\psi(r_{\max}))(\ln r)^{2}\leq\frac{r_{\max}}{(\ln r_{\max})^{2}}(\ln r)^{2}.
$$

###### 证明

定义辅助函数

$$
\displaystyle f(r)\mathrel{\mathop{\mathchar 58\relax}}={\frac{1}{2}}(\ln r)^{2}-(r-1-\ln r).
$$

对于 $r\in(0,1)$，成立 $f(1)=0$ 且 $f^{\prime}(r)=\frac{\ln r-r+1}{r}<0$。因此，$f(r)>0$ 意味着

$$
r-1-\ln r\leq{\frac{1}{2}}(\ln r)^{2}.
$$

对于 $r\in[1,r_{\max}]$，容易验证 $\psi$ 在 $(1,\infty)$ 上非递减，因此

$$
\displaystyle{\frac{r-1-\ln r}{(\ln r)^{2}}}=\psi(r)\leq\psi(r_{\max})\leq\frac{r_{\max}}{(\ln r_{\max})^{2}},
$$

如所示。∎

###### 引理 10

对于 $r_{\min}>0$，对所有 $r\in[r_{\min},\infty)$ 成立

$$
\displaystyle r-1-\ln r\geq\frac{1}{e(\ln r_{\min}^{-1}\vee 1)}(\ln r)^{2}.
$$

###### 证明

引理 9 中定义的函数 $\psi$ 通过设置 $\psi(1)\mathrel{\mathop{\mathchar 58\relax}}={\frac{1}{2}}$ 扩展到 $(0,\infty)$ 上的非递减连续函数。当 $r\geq e^{-1}$ 时，有 $\psi(r)\geq\psi(e^{-1})=e^{-1}$。

当 $r_{\min}\leq r<e^{-1}$ 时，我们使用 $\ln r\leq\frac{1}{1-e^{-1}}(r-1)$ 这一事实来界定

$$
\displaystyle\psi(r)\geq\frac{(1-e^{-1})\ln r-\ln r}{(\ln r)^{2}}=\frac{1}{e\ln r^{-1}}\geq\frac{1}{e\ln r_{\min}^{-1}}.
$$

∎

###### 引理 11（对称化不等式）

设 $D_{n},\tilde{D}_{n}$ 是 $n$ 个独立同分布样本的两个数据集，$C(\pi,D_{n})$ 是策略 $\pi$ 和数据集 $D_{n}$ 的任意泛函，$\hat{\pi}\mathrel{\mathop{\mathchar 58\relax}}=\hat{\pi}(D_{n})$ 是从 $D_{n}$ 计算的任意估计器。那么以 $1-\delta$ 的概率成立

$$
\displaystyle-\log\EE_{\tilde{D}_{n}}[\exp(C(\hat{\pi},\tilde{D}_{n}))]\leq-C(\hat{\pi},D_{n})+\ln(|\Pi|/\delta).
$$

###### 证明

例如，这在 [^15] 的定理 6 的证明中得到证明。∎

#### C.2 第 3 节的证明

以下关于最大似然估计器的收敛界基本是经典的 [^63] [^50]；为完整性，我们遵循 [^15] 的定理 6 提供简要证明。

###### 命题 12

设 $\hat{\pi}=\argmin_{\pi\in\Pi}\mathcal{L}_{\textup{{PMLE}}}(\pi)$ 且 $\beta=0$。那么，以至少 $1-\delta$ 的概率，

$$
\displaystyle\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\left[(\PP_{\widehat{\pi}}(a\succ b\mid x)-\PP_{*}(a\succ b\mid x))^{2}\right]\leq\frac{4\ln(|\Pi|/\delta)}{n}.
$$

###### 证明。

回顾每个偏好对 $(x,a^{+},a^{-})$ 的收集方式：首先从 $\mu(x)$ 独立采样 $a,b$，然后以概率 $\PP_{*}(a\succ b\mid x)$ 设定 $(a^{+},a^{-})=(a,b)$。换句话说，对于指示器 $y=1_{\{a^{+}=a\}}$ 满足 $\PP(y=1)=\PP_{*}(a\succ b\mid x)$，我们可以写成

$$
\displaystyle\mathcal{L}_{\textup{{PMLE}}}(\pi)=\frac{1}{n}\sum_{(x,a,b)\in D_{n}}-y\ln\PP_{\pi}(a\succ b\mid x)-(1-y)\ln\PP_{\pi}(b\succ a\mid x),
$$

其中我们滥用了符号，将对应每个 $(x,a^{+},a^{-})$ 的 $(x,a,b)$ 求和写成对 $(x,a,b)\in D_{n}$ 求和。定义量

$$
\displaystyle C(\pi,D_{n})
$$

$$
\displaystyle=\frac{1}{2}\sum_{(x,a,b)\in D_{n}}y\ln\frac{\PP_{\pi}(a\succ b\mid x)}{\PP_{*}(a\succ b\mid x)}+(1-y)\ln\frac{\PP_{\pi}(b\succ a\mid x)}{\PP_{*}(b\succ a\mid x)}
$$

$$
\displaystyle=\frac{n}{2}(\mathcal{L}_{\textup{{PMLE}}}(\pi^{*})-\mathcal{L}_{\textup{{PMLE}}}(\pi))
$$

并记 $\hat{\pi}$ 为 $\mathcal{L}_{\textup{{PMLE}}}(\pi)$ 在 $\pi\in\Pi$ 上的最小化器。由引理 11 可得

$$
\displaystyle-\log\EE_{\tilde{D}_{n}}[\exp(C(\hat{\pi},\tilde{D}_{n}))]\leq-C(\hat{\pi},D_{n})+\ln(|\Pi|/\delta)\leq\ln(|\Pi|/\delta)
$$

$$
\displaystyle-\log\EE_{\tilde{D}_{n}}[\exp(C(\hat{\pi},\tilde{D}_{n}))]
$$

$$
\displaystyle=-n\log\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\EE_{y|a,b,x}\left[\left(\frac{\PP_{\hat{\pi}}(a\succ b\mid x)}{\PP_{*}(a\succ b\mid x)}\right)^{y/2}\left(\frac{\PP_{\hat{\pi}}(b\succ a\mid x)}{\PP_{*}(b\succ a\mid x)}\right)^{(1-y)/2}\right]
$$

$$
\displaystyle=-n\log\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\left[\sqrt{\PP_{\hat{\pi}}(a\succ b\mid x)\PP_{*}(a\succ b\mid x)}+\sqrt{\PP_{\hat{\pi}}(b\succ a\mid x)\PP_{*}(b\succ a\mid x)}\right].
$$

为简化表示，令 $p_{\hat{\pi}}=\PP_{\hat{\pi}}(a\succ b\mid x)$ 和 $p_{*}=\PP_{*}(a\succ b\mid x)$，我们进一步有

$$
\displaystyle-\log\EE\left[\sqrt{p_{\hat{\pi}}p_{*}}+\sqrt{(1-p_{\hat{\pi}})(1-p_{*})}\right]
$$

$$
\displaystyle\geq 1-\EE\left[\sqrt{p_{\hat{\pi}}p_{*}}+\sqrt{(1-p_{\hat{\pi}})(1-p_{*})}\right]
$$

$$
\displaystyle=\EE\left[\frac{1}{2}(\sqrt{p_{\hat{\pi}}}-\sqrt{p_{*}})^{2}+\frac{1}{2}(\sqrt{1-p_{\hat{\pi}}}-\sqrt{1-p_{*}})^{2}\right]
$$

$$
\displaystyle=\EE\left[\frac{(p_{\hat{\pi}}-p_{*})^{2}}{2(\sqrt{p_{\hat{\pi}}}+\sqrt{p_{*}})^{2}}+\frac{(p_{\hat{\pi}}-p_{*})^{2}}{2(\sqrt{1-p_{\hat{\pi}}}+\sqrt{1-p_{*}})^{2}}\right]
$$

$$
\displaystyle\geq{\frac{1}{4}}\EE\left[(p_{\hat{\pi}}-p_{*})^{2}\right],
$$

由此得到所需的界。∎

**定理 4 的证明**。我们的证明部分受到 [^1] 的启发。关键区别在于，他们的定理依赖于一个假设，即 $\hat{\pi}$ 的总体损失与 $\pi^{*}$ 的总体损失相差不远，这是一个相当强的假设。相比之下，我们的定理提供了端到端的保证。此外，直接应用他们的定理会导致 $1/\sqrt{n}$ 的速率而非 $1/n$。我们通过应用 Schulman 技巧 [^44] 并结合引理 9 获得了改进。我们将在注记 13 中详细阐述。

使用引理 8 和以下事实

$$
\displaystyle\mathinner{\!\left\lvert\gamma\ln\frac{\pi(a\mid x)}{\pi(b\mid x)}\right\rvert}=\mathinner{\!\left\lvert\bar{R}(x,a)-\bar{R}(x,b)\right\rvert}\leq 2\gamma R~,
$$

我们可以下界为

$$
\displaystyle\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\left[(\PP_{\hat{\pi}}(a\succ b\mid x)-\PP_{*}(a\succ b\mid x))^{2}\right]
$$

$$
\displaystyle\geq\frac{e^{-4\gamma R}}{4}\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\bigg[\bigg(\gamma\ln\frac{\hat{\pi}(a\mid x)}{\hat{\pi}(b\mid x)}-\gamma\ln\frac{\pi^{*}(a\mid x)}{\pi^{*}(b\mid x)}\bigg)^{2}\bigg]
$$

$$
\displaystyle=\frac{e^{-4\gamma R}}{4}\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\left[(\Delta\bar{R}_{\hat{\pi}}(x,a)-\Delta\bar{R}_{\hat{\pi}}(x,b))^{2}\right]
$$

$$
\displaystyle=\frac{e^{-4\gamma R}}{2}\EE_{x\sim\mathcal{D},a\sim\mu(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)^{2}\right]
$$

其中最后一个不等式利用了当 $X$ 和 $Y$ 独立同分布时 $\EE[(X-Y)^{2}]=2\EE[(X-\EE[X])^{2}]$。因此，使用命题 12，中心化奖励的差满足

$$
\displaystyle\EE_{x\sim\mathcal{D},a\sim\mu(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)^{2}\right]\leq 8e^{4\gamma R}\cdot\frac{\ln(|\Pi|/\delta)}{n}.
$$

定义归一化因子

$$
Z_{\pi}(x)\mathrel{\mathop{\mathchar 58\relax}}=\sum_{a\in\mathcal{A}}\exp\left(\frac{1}{\gamma}\bar{R}_{\pi}(x,a)\right)=\exp\left(-\frac{1}{\gamma}\EE_{a\sim\mu(x)}[R_{\pi}(x,a)\mid x]\right),\quad Z_{*}\mathrel{\mathop{\mathchar 58\relax}}=Z_{\pi^{*}}
$$

使得 $\pi(a\mid x)=Z_{\pi}(x)^{-1}\exp(\gamma^{-1}\bar{R}_{\pi}(x,a))$。根据假设 2，对所有 $\pi\in\Pi,x\in\mathcal{X}$ 有 $|\mathcal{A}|e^{-R}\leq Z_{\pi}(x)\leq|\mathcal{A}|e^{R}$，因此

$$
\displaystyle 0<\frac{\hat{\pi}(a\mid x)}{\pi^{*}(a\mid x)}=\frac{Z_{*}(x)}{Z_{\hat{\pi}}(x)}\exp\left(\frac{1}{\gamma}\Delta\bar{R}_{\hat{\pi}}(x,a)\right)\leq e^{4R}.
$$

然后，我们使用 Schulman 技巧 [^44] 并结合引理 9 对 $\pi^{*},\hat{\pi}$ 之间的 KL 散度进行界定：

$$
\displaystyle\EE_{x\sim\mathcal{D}}\left[\KL(\pi^{*}(x)\|\hat{\pi}(x))\right]
$$

$$
\displaystyle=\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\bigg[\ln\frac{\pi^{*}(a\mid x)}{\hat{\pi}(a\mid x)}\bigg]
$$

$$
\displaystyle=\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\bigg[\frac{\hat{\pi}(a\mid x)}{\pi^{*}(a\mid x)}-1-\ln\frac{\hat{\pi}(a\mid x)}{\pi^{*}(a\mid x)}\bigg]
$$

$$
\displaystyle\leq({\frac{1}{2}}\vee\psi(e^{4R}))\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\bigg[\bigg(\ln\frac{\hat{\pi}(a\mid x)}{\pi^{*}(a\mid x)}\bigg)^{2}\bigg].
$$

提取归一化常数，我们进一步有

$$
\displaystyle\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\bigg[\bigg(\ln\frac{\hat{\pi}(a\mid x)}{\pi^{*}(a\mid x)}\bigg)^{2}\bigg]
$$

$$
\displaystyle\leq\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\bigg[2\left(\ln\frac{\hat{\pi}(a\mid x)Z_{\hat{\pi}}(x)}{\pi^{*}(a\mid x)Z_{*}(x)}\right)^{2}+2\left(\ln\frac{Z_{*}(x)}{Z_{\hat{\pi}}(x)}\right)^{2}\bigg]
$$

$$
\displaystyle=\frac{2}{\gamma^{2}}\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)^{2}\right]+2\EE_{x\sim\mathcal{D}}\bigg[\left(\ln\frac{Z_{*}(x)}{Z_{\hat{\pi}}(x)}\right)^{2}\bigg].
$$

使用定义 3 和式 (17)，第一项的界为

$$
\displaystyle\frac{2}{\gamma^{2}}\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)^{2}\right]\leq\frac{2C_{\Pi}}{\gamma^{2}}\EE_{x\sim\mathcal{D},a\sim\mu(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)^{2}\right]\leq\frac{16C_{\Pi}e^{4\gamma R}}{\gamma^{2}}\cdot\frac{\ln(|\Pi|/\delta)}{n}.
$$

对于第二项，我们首先刻画 $\ln\frac{Z_{\hat{\pi}}(x)}{Z_{*}(x)}$ 的上界和下界。使用

$$
\displaystyle 1=\EE_{a\sim\pi^{*}(x)}\bigg[\frac{\hat{\pi}(a\mid x)}{\pi^{*}(a\mid x)}\bigg]=\frac{Z_{*}(x)}{Z_{\hat{\pi}}(x)}\EE_{a\sim\pi^{*}(x)}\bigg[\exp\left(\frac{1}{\gamma}\Delta\bar{R}_{\hat{\pi}}(x,a)\right)\bigg],
$$

我们有

$$
\displaystyle\ln\frac{Z_{\hat{\pi}}(x)}{Z_{*}(x)}=\ln\EE_{a\sim\pi^{*}(x)}\bigg[\exp\left(\frac{1}{\gamma}\Delta\bar{R}_{\hat{\pi}}(x,a)\right)\bigg]
$$

$$
\displaystyle\geq{\frac{1}{\gamma}}\EE_{a\sim\pi^{*}(x)}[\Delta\bar{R}_{\hat{\pi}}(x,a)]
$$

其中最后的不等式由 Jensen 不等式得到。此外，使用对所有 $x\in(-\infty,A]$ 成立的不等式 $e^{x}\leq 1+x+\frac{e^{A}}{2}x^{2}$，我们有

$$
\displaystyle\ln\frac{Z_{\hat{\pi}}(x)}{Z_{*}(x)}
$$

$$
\displaystyle=\ln\EE_{a\sim\pi^{*}(x)}\bigg[\exp\left(\frac{1}{\gamma}\Delta\bar{R}_{\hat{\pi}}(x,a)\right)\bigg]
$$

$$
\displaystyle\leq\EE_{a\sim\pi^{*}(x)}\bigg[\exp\left(\frac{1}{\gamma}\Delta\bar{R}_{\hat{\pi}}(x,a)\right)\bigg]-1
$$

$$
\displaystyle\leq\frac{1}{\gamma}\EE_{a\sim\pi^{*}(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)\right]+\frac{e^{2R}}{2\gamma^{2}}\EE_{a\sim\pi^{*}(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)^{2}\right].
$$

因此，我们有

$$
\displaystyle\mathinner{\!\left\lvert\ln{\frac{Z_{\hat{\pi}}(x)}{Z_{*}(x)}}\right\rvert}\leq\mathinner{\!\left\lvert{\frac{1}{\gamma}}\EE_{a\sim\pi^{*}(x)}[\Delta\bar{R}_{\hat{\pi}}(x,a)]\right\rvert}+\frac{e^{2R}}{2\gamma^{2}}\EE_{a\sim\pi^{*}(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)^{2}\right],
$$

这意味着，使用 $\forall x,y\in\mathbb{R},(x+y)^{2}\leq 2x^{2}+2y^{2}$，

$$
\displaystyle\EE_{x\sim\mathcal{D}}\bigg[\left(\ln\frac{Z_{*}(x)}{Z_{\hat{\pi}}(x)}\right)^{2}\bigg]
$$

$$
\displaystyle\leq\frac{2}{\gamma^{2}}\EE_{x\sim\mathcal{D}}\bigg[\left(\EE_{a\sim\pi^{*}(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)\right]\right)^{2}\bigg]+\frac{e^{4R}}{2\gamma^{4}}\EE_{x\sim\mathcal{D}}\bigg[\left(\EE_{a\sim\pi^{*}(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)^{2}\right]\right)^{2}\bigg]
$$

$$
\displaystyle\leq\frac{2R^{2}e^{4R}+2}{\gamma^{2}}\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)^{2}\right]
$$

$$
\displaystyle\leq\frac{16C_{\Pi}(R^{2}e^{4R}+1)e^{4\gamma R}}{\gamma^{2}}\cdot\frac{\ln(|\Pi|/\delta)}{n}~.
$$

综合所有结果，我们得出：

$$
\displaystyle\EE_{x\sim\mathcal{D}}\left[\KL(\pi^{*}(x)\|\hat{\pi}(x))\right]
$$

$$
\displaystyle\leq({\frac{1}{2}}\vee\psi(e^{4R}))\left(\frac{16C_{\Pi}e^{4\gamma R}}{\gamma^{2}}\cdot\frac{\ln(|\Pi|/\delta)}{n}+\frac{32C_{\Pi}(R^{2}e^{4R}+1)e^{4\gamma R}}{\gamma^{2}}\cdot\frac{\ln(|\Pi|/\delta)}{n}\right)
$$

$$
\displaystyle=({\frac{1}{2}}\vee\psi(e^{4R}))\frac{16(2R^{2}e^{4R}+3)C_{\Pi}e^{4\gamma R}}{\gamma^{2}}\cdot\frac{\ln(|\Pi|/\delta)}{n}.
$$

我们注意到，根据引理 9，项 ${\frac{1}{2}}\vee\psi(e^{4R})$ 进一步上界为 $\frac{e^{4R}}{16R^{2}}$。

###### 注记 13。

我们的关键创新之一是式 (C.2)。在 [^1] 中，他们使用 Cauchy-Schwarz 不等式推导出界

$$
\displaystyle\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\mathinner{\Bigl[\ln{\frac{\pi^{*}(a\mid x)}{\hat{\pi}(a\mid x)}}\Bigr]}\leq\sqrt{\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\mathinner{\Bigl[\mathinner{\Bigl(\ln{\frac{\pi^{*}(a\mid x)}{\hat{\pi}(a\mid x)}}\Bigr)}^{2}\Bigr]}}~,
$$

这相比我们的推导引入了额外的平方根。直接采用他们的方法会导致 $1/\sqrt{n}$ 的速率而非 $1/n$。

∎

#### C.3 第 4 节的证明

**定理 6 的证明**。忽略常数因子，我们的蒸馏目标等价于最小化

$$
\displaystyle\frac{1}{n}\sum_{(x,a^{+},a^{-})\in D_{n}}\KL\left(\text{Bern}(\PP_{\widetilde{\pi}}(a^{+}\succ a^{-}\mid x))\|\text{Bern}(\PP_{\pi}(a^{+}\succ a^{-}\mid x))\right),
$$

该目标可以达到零损失，因为 $\widetilde{\pi}\in\mathcal{P}_{\gamma}(\mathcal{R})\subseteq\Pi$ 是一个有效解。因此，解 $\hat{\pi}$ 必须满足

$$
\displaystyle\PP_{\widetilde{\pi}}(a\succ b\mid x)=\PP_{\hat{\pi}}(a\succ b\mid x),\quad\forall(x,a,b)\in D_{n}
$$

（回顾我们使用 $(a,b)$ 表示独立的未标注响应）。定义集合

$$
\displaystyle\mathcal{K}\mathrel{\mathop{\mathchar 58\relax}}=\left\{(\pi_{1},\pi_{2})\in\mathcal{P}_{\gamma}(\mathcal{R})\times\Pi\mathrel{\mathop{\mathchar 58\relax}}\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\big[|\PP_{\pi_{1}}(a\succ b\mid x)-\PP_{\pi_{2}}(a\succ b\mid x)|\big]>\varepsilon\right\},
$$

可得

$$
\displaystyle\PP\left((\widetilde{\pi},\hat{\pi})\in\mathcal{K}\right)
$$

$$
\displaystyle=\sum_{(\pi_{1},\pi_{2})\in\mathcal{K}}\PP(\widetilde{\pi}=\pi_{1},\hat{\pi}=\pi_{2})
$$

$$
\displaystyle\leq\sum_{(\pi_{1},\pi_{2})\in\mathcal{K}}\PP\left(\PP_{\pi_{1}}(a\succ b\mid x)=\PP_{\pi_{2}}(a\succ b\mid x),\;\forall(x,a,b)\in D_{n}\right)
$$

$$
\displaystyle=\sum_{(\pi_{1},\pi_{2})\in\mathcal{K}}\PP\left(\PP_{\pi_{1}}(a\succ b\mid x)=\PP_{\pi_{2}}(a\succ b\mid x)\right)^{n}
$$

$$
\displaystyle\leq\sum_{(\pi_{1},\pi_{2})\in\mathcal{K}}\left(1-\EE\left[|\PP_{\pi_{1}}(a\succ b\mid x)-\PP_{\pi_{2}}(a\succ b\mid x)|\right]\right)^{n}
$$

$$
\displaystyle\leq\sum_{(\pi_{1},\pi_{2})\in\mathcal{K}}(1-\varepsilon)^{n}
$$

$$
\displaystyle\leq|\mathcal{K}|^{2}\exp(-\varepsilon n).
$$

因此 $\PP\left((\widetilde{\pi},\hat{\pi})\in\mathcal{K}\right)\leq|\Pi|^{2}\exp(-\varepsilon n)$，即

$$
\displaystyle\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\big[|\PP_{\widetilde{\pi}}(a\succ b\mid x)-\PP_{\hat{\pi}}(a\succ b\mid x)|\big]\leq\frac{2\ln(|\Pi|/\delta)}{n}
$$

以至少 $1-\delta$ 的概率成立，因此

$$
\displaystyle\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\left[(\PP_{\widetilde{\pi}}(a\succ b\mid x)-\PP_{\hat{\pi}}(a\succ b\mid x))^{2}\right]\leq\frac{2\ln(|\Pi|/\delta)}{n}
$$

也成立。另一方面，对 $\mathcal{P}_{\gamma}(\mathcal{R})$ 应用命题 12，我们有

$$
\displaystyle\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\left[(\PP_{\widetilde{\pi}}(a\succ b\mid x)-\PP_{*}(a\succ b\mid x))^{2}\right]\leq\frac{4\ln(|\mathcal{R}|/\delta)}{n}
$$

以至少 $1-\delta$ 的概率成立。因此通过联合界，以至少 $1-\delta$ 的概率有

$$
\displaystyle\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\left[(\PP_{\hat{\pi}}(a\succ b\mid x)-\PP_{*}(a\succ b\mid x))^{2}\right]\leq\frac{4\ln(2|\Pi|/\delta)+8\ln(2|\mathcal{R}|/\delta)}{n}~.
$$

此外，由引理 8 可知

$$
\displaystyle\mathinner{\!\left\lvert\PP_{\hat{\pi}}(a\succ b\mid x)-\PP_{*}(a\succ b\mid x)\right\rvert}
$$

$$
\displaystyle=\mathinner{\!\left\lvert\sigma\bigg(\gamma\ln\frac{\hat{\pi}(a\mid x)}{\hat{\pi}(b\mid x)}\bigg)-\sigma\bigg(\gamma\ln\frac{\pi^{*}(a\mid x)}{\pi^{*}(b\mid x)}\bigg)\right\rvert}
$$

$$
\displaystyle\geq\frac{\gamma e^{-2\gamma R}}{2}\mathinner{\!\left\lvert\ln\frac{\hat{\pi}(a\mid x)}{\hat{\pi}(b\mid x)}-\ln\frac{\pi^{*}(a\mid x)}{\pi^{*}(b\mid x)}\right\rvert}
$$

$$
\displaystyle=\frac{e^{-2\gamma R}}{2}\mathinner{\!\left\lvert\Delta\bar{R}_{\hat{\pi}}(x,a)-\Delta\bar{R}_{\hat{\pi}}(x,b)\right\rvert},
$$

这意味着

$$
\displaystyle\EE_{x\sim\mathcal{D},a\sim\mu(x)}\left[(\Delta\bar{R}_{\hat{\pi}}(x,a))^{2}\right]
$$

$$
\displaystyle={\frac{1}{2}}\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\left[(\Delta\bar{R}_{\hat{\pi}}(x,a)-\Delta\bar{R}_{\hat{\pi}}(x,b))^{2}\right]
$$

$$
\displaystyle\leq 2e^{4\gamma R}\cdot\frac{4\ln(2|\Pi|/\delta)+8\ln(2|\mathcal{R}|/\delta)}{n}.
$$

最后如定理 4 的证明，我们结合界

$$
\displaystyle\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\bigg[\bigg(\ln\frac{\hat{\pi}(a\mid x)}{\pi^{*}(a\mid x)}\bigg)^{2}\bigg]
$$

$$
\displaystyle\leq\frac{2}{\gamma^{2}}\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)^{2}\right]+2\EE_{x\sim\mathcal{D}}\bigg[\left(\ln\frac{Z_{*}(x)}{Z_{\hat{\pi}}(x)}\right)^{2}\bigg]
$$

$$
\displaystyle\EE_{x\sim\mathcal{D}}\bigg[\left(\ln\frac{Z_{*}(x)}{Z_{\hat{\pi}}(x)}\right)^{2}\bigg]\leq\frac{2R^{2}e^{4R}+2}{\gamma^{2}}\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\left[\Delta\bar{R}_{\hat{\pi}}(x,a)^{2}\right]
$$

以及式 (22) 得出

$$
\displaystyle\EE_{x\sim\mathcal{D}}\left[\KL(\pi^{*}(x)\|\hat{\pi}(x))\right]\leq({\frac{1}{2}}\vee\psi(e^{4R}))\frac{16(2R^{2}e^{4R}+3)C_{\Pi}e^{4\gamma R}}{\gamma^{2}}\cdot\frac{\ln(2|\Pi|/\delta)+2\ln(2|\mathcal{R}|/\delta)}{n}.
$$

∎

###### 定理 14。

给定基准策略 $\pi_{0}$，记 $C_{0}$ 为满足以下条件的最小常数：对每个 $\pi,\pi^{\prime}\in\Pi$，<sup>5</sup>

$$
\displaystyle\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\big[(\Delta\bar{R}_{\pi}(x,a)-\Delta\bar{R}_{\pi^{\prime}}(x,a))^{2}\big]\leq C_{0}\EE_{x\sim\mathcal{D},a\sim\pi_{0}(x)}\big[(\Delta\bar{R}_{\pi}(x,a)-\Delta\bar{R}_{\pi^{\prime}}(x,a))^{2}\big].
$$

偏好蒸馏估计 $\hat{\pi}=\argmin\nolimits_{\pi\in\Pi}\mathcal{L}_{\textup{{Distill}}}(\pi)$（其中 $D_{n}$ 中的响应由 $\pi_{0}$ 生成而非 $\mu$）满足

$$
\displaystyle\EE_{x\sim\mathcal{D}}\left[\KL(\pi^{*}(x)\|\hat{\pi}(x))\right]\lesssim\frac{1}{\gamma^{2}}\cdot\frac{C_{0}\ln(|\Pi|/\delta)+C_{\mathcal{R}}\ln(|\mathcal{R}|/\delta)}{n}
$$

以至少 $1-\delta$ 的概率成立。

###### 证明。

在定理 6 的证明中，式 (20) 现在将 $\mu$ 替换为 $\pi_{0}$ 后仍成立，而式 (21) 保持不变。由引理 8 可得

$$
\displaystyle\frac{2\ln(|\Pi|/\delta)}{n}
$$

$$
\displaystyle\geq\EE_{x\sim\mathcal{D},a,b\sim\pi_{0}(x)}\left[(\PP_{\widetilde{\pi}}(a\succ b\mid x)-\PP_{\hat{\pi}}(a\succ b\mid x))^{2}\right]
$$

$$
\displaystyle\geq\frac{\gamma^{2}e^{-4\gamma R}}{4}\EE_{x\sim\mathcal{D},a,b\sim\pi_{0}(x)}\left[\left(\ln\frac{\hat{\pi}(a\mid x)}{\hat{\pi}(b\mid x)}-\ln\frac{\widetilde{\pi}(a\mid x)}{\widetilde{\pi}(b\mid x)}\right)^{2}\right]
$$

$$
\displaystyle=\frac{e^{-4\gamma R}}{4}\EE_{x\sim\mathcal{D},a,b\sim\pi_{0}(x)}\left[\left(\Delta\bar{R}_{\hat{\pi}}(x,a)-\Delta\bar{R}_{\hat{\pi}}(x,b)-\Delta\bar{R}_{\widetilde{\pi}}(x,a)+\Delta\bar{R}_{\widetilde{\pi}}(x,b)\right)^{2}\right]
$$

$$
\displaystyle=\frac{e^{-4\gamma R}}{2}\EE_{x\sim\mathcal{D},a\sim\pi_{0}(x)}\left[\left(\Delta\bar{R}_{\hat{\pi}}(x,a)-\Delta\bar{R}_{\widetilde{\pi}}(x,a)\right)^{2}\right]
$$

$$
\displaystyle\geq\frac{e^{-4\gamma R}}{2C_{0}}\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\left[\left(\Delta\bar{R}_{\hat{\pi}}(x,a)-\Delta\bar{R}_{\widetilde{\pi}}(x,a)\right)^{2}\right],
$$

其中最后一行使用了覆盖系数 $C_{0}$ 的修改定义。此外，式 (21) 蕴含

$$
\displaystyle\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\left[(\Delta\bar{R}_{\widetilde{\pi}}(x,a))^{2}\right]
$$

$$
\displaystyle\leq C_{\mathcal{R}}\EE_{x\sim\mathcal{D},a\sim\mu(x)}\left[(\Delta\bar{R}_{\widetilde{\pi}}(x,a))^{2}\right]
$$

$$
\displaystyle={\frac{1}{2}}C_{\mathcal{R}}\EE_{x\sim\mathcal{D},a,b\sim\mu(x)}\left[(\Delta\bar{R}_{\widetilde{\pi}}(x,a)-\Delta\bar{R}_{\widetilde{\pi}}(x,b))^{2}\right]
$$

$$
\displaystyle\leq 2C_{\mathcal{R}}e^{4\gamma R}\cdot\frac{4\ln(|\mathcal{R}|/\delta)}{n}.
$$

通过联合界，以至少 $1-\delta$ 的概率有

$$
\displaystyle\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\left[(\Delta\bar{R}_{\hat{\pi}}(x,a))^{2}\right]
$$

$$
\displaystyle\leq 2\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\left[\left(\Delta\bar{R}_{\hat{\pi}}(x,a)-\Delta\bar{R}_{\widetilde{\pi}}(x,a)\right)^{2}\right]+2\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\left[(\Delta\bar{R}_{\widetilde{\pi}}(x,a))^{2}\right]
$$

$$
\displaystyle\leq 8e^{4\gamma R}\cdot\frac{C_{0}\ln(2|\Pi|/\delta)+2C_{\mathcal{R}}\ln(2|\mathcal{R}|/\delta)}{n}.
$$

证明的其余部分类似进行。∎

#### C.4 第 5 节的证明

**定理 7 的证明**。论证的第一步与定理 6 的证明类似，但具有更强的上确界范数界，这在可实现性假设下得到保证。实际上，忽略常数因子，反向 KL 目标等价于

$$
\displaystyle\hat{\pi}=\argmin_{\pi\in\Pi}\frac{1}{n}\sum\limits_{(x,\cdot,\cdot)\in D_{n}}\KL(\pi(x)\|\widetilde{\pi}(x)),
$$

根据假设 5，该目标达到零损失。定义集合

$$
\displaystyle K\mathrel{\mathop{\mathchar 58\relax}}=\Bigg\{(\pi_{1},\pi_{2})\in\Pi\times\mathcal{P}_{\gamma}(\mathcal{R})\mathrel{\mathop{\mathchar 58\relax}}\EE_{x\sim\mathcal{D}}\bigg[\sup_{a\in\mathcal{A}}\left(\ln\frac{\pi_{1}(a\mid x)}{\pi_{2}(a\mid x)}\right)^{2}\bigg]>\varepsilon\Bigg\},
$$

可得

$$
\displaystyle\PP\left(\EE_{x\sim\mathcal{D}}\bigg[\sup_{a\in\mathcal{A}}\left(\ln\frac{\hat{\pi}(a\mid x)}{\widetilde{\pi}(a\mid x)}\right)^{2}\bigg]>\varepsilon\right)
$$

$$
\displaystyle=\sum_{(\pi_{1},\pi_{2})\in K}\PP(\hat{\pi}=\pi_{1},\widetilde{\pi}=\pi_{2})
$$

$$
\displaystyle\leq\sum_{(\pi_{1},\pi_{2})\in K}\PP(\pi_{1}(x)=\pi_{2}(x),\;\forall x\in D_{n})
$$

$$
\displaystyle=\sum_{(\pi_{1},\pi_{2})\in K}\PP_{x\sim\mathcal{D}}(\pi_{1}(x)=\pi_{2}(x))^{n}.
$$

注意对任意 $(\pi_{1},\pi_{2})\in K$，如定理 4 的证明，有 $\pi_{1}(a\mid x)/\pi_{2}(a\mid x)\leq e^{4R}$，因此

$$
\displaystyle\sup_{a\in\mathcal{A}}\left(\ln\frac{\pi_{1}(a\mid x)}{\pi_{2}(a\mid x)}\right)^{2}\leq 16R^{2}\cdot 1_{\{\pi_{1}(x)\neq\pi_{2}(x)\}},\quad\forall x\in\mathcal{X}.
$$

这意味着

$$
\displaystyle\PP_{x\sim\mathcal{D}}(\pi_{1}(x)=\pi_{2}(x))
$$

$$
\displaystyle=1-\EE_{x\sim\mathcal{D}}[1_{\{\pi_{1}(x)\neq\pi_{2}(x)\}}]\leq 1-\frac{\varepsilon}{16R^{2}},
$$

因此

$$
\displaystyle\PP\left(\EE_{x\sim\mathcal{D}}\bigg[\sup_{a\in\mathcal{A}}\left(\ln\frac{\hat{\pi}(a\mid x)}{\widetilde{\pi}(a\mid x)}\right)^{2}\bigg]>\varepsilon\right)\leq|K|\left(1-\frac{\varepsilon}{16R^{2}}\right)^{n}\leq|\Pi|^{2}\exp\left(-\frac{\varepsilon n}{16R^{2}}\right).
$$

我们现在对正向 KL 散度进行界定。再次应用引理 9，

$$
\displaystyle\EE_{x\sim\mathcal{D}}\left[\KL(\pi^{*}(x)\|\hat{\pi}(x))\right]
$$

$$
\displaystyle\leq({\frac{1}{2}}\vee\psi(e^{4R}))\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\bigg[\bigg(\ln\frac{\hat{\pi}(a\mid x)}{\pi^{*}(a\mid x)}\bigg)^{2}\bigg]
$$

$$
\displaystyle\leq(1\vee 2\psi(e^{4R}))\left(\EE_{x\sim\mathcal{D}}\bigg[\sup_{a\in\mathcal{A}}\bigg(\ln\frac{\hat{\pi}(a\mid x)}{\widetilde{\pi}(a\mid x)}\bigg)^{2}\bigg]+\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\bigg[\bigg(\ln\frac{\widetilde{\pi}(a\mid x)}{\pi^{*}(a\mid x)}\bigg)^{2}\bigg]\right).
$$

第一项可通过式 (24) 进行界定。对于第二项，在策略类 $\mathcal{P}_{\gamma}(\mathcal{R})$ 而非 $\Pi$ 上重复定理 4 的推导，我们得到

$$
\displaystyle\EE_{x\sim\mathcal{D},a\sim\pi^{*}(x)}\bigg[\bigg(\ln\frac{\widetilde{\pi}(a\mid x)}{\pi^{*}(a\mid x)}\bigg)^{2}\bigg]\leq\frac{16C_{\mathcal{R}}(R^{2}e^{4R}+1)e^{4\gamma R}}{\gamma^{2}}\cdot\frac{\ln(|\mathcal{R}|/\delta)}{n}.
$$

综合所有结果，我们得出：

$$
\displaystyle\EE_{x\sim\mathcal{D}}\left[\KL(\pi^{*}(x)\|\hat{\pi}(x))\right]
$$

$$
\displaystyle\leq(1\vee 2\psi(e^{4R}))\left(16R^{2}\cdot\frac{\ln(2|\Pi|^{2}/\delta)}{n}+\frac{16C_{\mathcal{R}}(R^{2}e^{4R}+1)e^{4\gamma R}}{\gamma^{2}}\cdot\frac{\ln(2|\mathcal{R}|/\delta)}{n}\right)
$$

以至少 $1-\delta$ 的概率成立。∎

###### 命题 15。

成立以下关系

$$
\displaystyle\EE_{x}[\KL(\hat{\pi}(x)\|\pi^{*}(x))]\leq{\frac{(4R\vee 1)e^{8R+1}}{R^{2}}}\EE_{x}[\KL(\pi^{*}(x)\|\hat{\pi}(x))].
$$

###### 命题的证明。

为简化表示，记比率 $r={\frac{\pi^{*}(a\mid x)}{\hat{\pi}(a\mid x)}}$。使用与式 (18) 相同的论证，我们有 $r\in[e^{-4R},e^{4R}]$。然后，我们有

$$
\displaystyle r-1-\ln r\stackrel{{\scriptstyle(a)}}{{\leq}}{\frac{e^{4R}}{R^{2}}}(\ln r)^{2}={\frac{e^{4R}}{R^{2}}}\left(\ln{\frac{1}{r}}\right)^{2}\stackrel{{\scriptstyle(b)}}{{\leq}}{\frac{e^{4R}}{R^{2}}}e(4R\vee 1)\left({\frac{1}{r}}-1-\ln{\frac{1}{r}}\right),
$$

其中 $(a)$ 由引理 9 得到，$(b)$ 由引理 10 并将 $r$ 替换为 $1/r$ 得到。因此，

$$
\displaystyle\KL(\hat{\pi}(x)\|\pi^{*}(x))
$$

$$
\displaystyle=\EE_{a\sim\hat{\pi}(x)}[r-1-\ln r]
$$

$$
\displaystyle\leq{\frac{e^{4R}}{R^{2}}}e(4R\vee 1)\EE_{a\sim\hat{\pi}(x)}\left[{\frac{1}{r}}-1-\ln{\frac{1}{r}}\right]
$$

$$
\displaystyle\leq{\frac{e^{4R}}{R^{2}}}e(4R\vee 1)e^{4R}\EE_{a\sim\pi^{*}(x)}\left[{\frac{1}{r}}-1-\ln{\frac{1}{r}}\right]
$$

$$
\displaystyle={\frac{e^{4R}}{R^{2}}}e(4R\vee 1)e^{4R}\KL(\pi^{*}(x)\|\hat{\pi}(x)).
$$

∎

### 附录 D 实验细节

在第 D.1 节中，我们描述了玩具实验的详细设置。在第 D.2 节中，我们提供了在 TL;DR 数据集上训练和评估的模型卡、超参数和计算资源的实现细节。在第 D.3 节中，我们提供了第 6 节通用对话实验的详细信息，并展示了在 MT-Bench 和 AlpacaEval 2.0 上的额外结果。

#### D.1 玩具实验

我们采用表格化设置，词汇表和上下文大小均等于 $10$。在偏好模型 (2) 下，我们固定 $\gamma=0.5$。为了保持目标策略 $\pi^{*}$ 和参考策略 ${\pi_{0}}$ 接近，对于每个上下文 $x\in\mathcal{X}$，我们抽取 logits $\alpha^{*},\alpha_{0}\in\mathbb{R}^{10}$，其条目独立同分布于 $\mathcal{N}(0,0.1^{2})$，并设定 $\pi^{*}(\cdot\mid x)=\mathrm{softmax}(\alpha^{*})$ 和 ${\pi_{0}}(\cdot\mid x)=\mathrm{softmax}(\alpha^{0})$。然后我们通过优化式 (1) 和式 (15) 训练策略，并评估正向 KL $\mathbb{E}_{x}\big[\mathrm{KL}(\pi^{*}(\cdot\mid x)\|\pi(\cdot\mid x)\big]$。我们变化样本量 $n\in\{2^{5},\cdots,2^{16}\}$，对每个 $n$ 调整 RLHF 和 RKL 的 KL 系数 $\beta$，同时保持 RKL 的 $\gamma=0.5$ 固定。

#### D.2 TL;DR 摘要

##### 数据集。

我们使用在相关文献中广泛使用的 TL;DR 数据集 [^18] [^45] [^25]，可公开获取<sup>6</sup>。我们在表 4 中总结了数据集统计信息。注意 DPO 和 PMLE 在具有偏好标签的偏好数据集上训练，而其他算法基于人类参考评估策略，因为它们利用在线响应。

表 4：TL;DR 数据集统计信息。

| 数据集 | 训练 | 验证 | 测试 |
| --- | --- | --- | --- |
| 人类参考 | 117K | 64.5K | 6.55K |
| 偏好 | 92.9K | 83.8K | N/A |

##### 模型。

我们使用 Pythia-2.8B<sup>7</sup> 和 Pythia-6.9B<sup>8</sup> [^6] 作为我们的预训练模型，使用最大上下文长度 $512$ 和最大生成长度最多 $53$ 个 token。为了训练效率，在对 SFT 模型进行全参数微调后，我们使用 LoRA（低秩适配器）[^23] 进行对齐。
##### 实现细节

我们在一个公开可用的代码库 <sup>9</sup> 的基础上实现了三种方法（PMLE、反向 KL、偏好蒸馏）；其中偏好蒸馏特别基于另一个公开可用的代码基线 <sup>10</sup>。对于 PMLE（第 3 节），我们使用 [^45] 中描述的在线响应实现了公式 (5) 中的 KL 正则化项。DPO 基线使用 4 块 A100 40GB GPU 大约需要 3 小时，PMLE 大约需要 6 小时。此外，反向 KL 和偏好蒸馏，以及它们对应的基线 RLHF 和 REBEL，使用 4 块 A100 40GB GPU 大约需要 2.5 天。最后，胜率由 GPT-4 使用 gpt-4 检查点（截至 2025 年 5 月 23 日）进行评判。

算法 1 偏好蒸馏（第 4 节）

 输入：奖励模型 $\hat{R}$，策略类 $\Pi$，采样分布 $\mu$，学习率 $\eta$，提示数据集 $\{x_{i}\}_{i=1}^{n}$

 for $t=0,1,\ldots,T-1$ do

  使用偏好模拟器计算概率
$$
\displaystyle\mathbb{P}_{\tilde{\pi}}(a_{1}\succ a_{2}\mid x)
$$

$$
\displaystyle\mathrel{\mathop{\mathchar 58\relax}}=\sigma\big(\hat{R}(x,a_{1})-\hat{R}(x,a_{2})\big)
$$

$$
\displaystyle\mathbb{P}_{\tilde{\pi}}(a_{2}\succ a_{1}\mid x)
$$

$$
\displaystyle\mathrel{\mathop{\mathchar 58\relax}}=\sigma\big(\hat{R}(x,a_{2})-\hat{R}(x,a_{1})\big)
$$

  使用公式 (11) 和 (12) 计算偏好蒸馏损失 $\mathcal{L}_{\textup{{Distill}},\beta}(\pi)$

   $\pi_{t+1}\leftarrow\pi_{t}-\eta\nabla\mathcal{L}_{\textup{{Distill}},\beta}(\pi_{t})$

 end for

##### 伪代码

由于 PMLE 和反向 KL 的实现直接基于相应的 DPO 和 RLHF 基线，因此较为简单，我们为偏好蒸馏提供伪代码以便更好地理解。如 [^18] 中所述，基础分布 $\mu$ 在我们的伪代码（算法 1）中也可以是 $\pi_{t}$。遵循 REBEL 的基线代码实现，我们也从分布 $\pi_{t}$ 中采样在线响应。

##### 超参数

我们采用了几项研究 [^25] [^18] [^45] 中使用的几乎相同的超参数。为了完整性，我们在表 5 中总结了实验中使用的超参数。请注意，[^18] 仅对 RLHF 和 REBEL 训练单个 epoch，但我们无法仅用一个 epoch 重现他们的结果。相反，遵循实现细节 [^25]，我们考虑总回合数为 $10^{6}$，这大约对应 8.5 个 epoch。在这种设置下，我们能够重现基线结果或获得更好的结果。因此，反向 KL 和偏好蒸馏也在此设置下进行评估。

表 5：TL;DR 摘要任务的超参数配置。

| 设置 | 参数 |  |
| --- | --- | --- |
| SFT & RM | batch size: 64   learning rate: 3e-6 | schedule: cosine decay   train epochs: 1 |
| DPO | batch size: 64   learning rate: 3e-6   schedule: linear decay | train epochs: 1   $\beta$: 0.05 |
| PMLE | batch size: 512   learning rate: 1e-6   schedule: linear decay | train epochs: 1   $\beta$: 1e-5   $\gamma$: 1e-2 |
| REBEL | batch size: 512   learning rate: 3e-6   schedule: linear decay   total episodes: 1e6 | num epochs: 4   $\eta$: 1.0   kl coefficient: 0.05 |
| Preference Distillation | batch size: 512   learning rate: 3e-6   schedule: linear decay   total episodes: 1e6 | num epochs: 4   $\gamma$: 0.1   kl coefficient: 0.05 |
| RLHF (via PPO) | batch size: 512   learning rate: 3e-6   schedule: linear decay   total episodes: 1e6   num epochs: 4 | discount factor: 1   gae $\lambda$: 0.95   clip ratio: 0.2   value function coeff: 0.1   kl coefficient: 0.05 |
| Reverse KL (Sec. 5) | batch size: 512   learning rate: 3e-6   schedule: linear decay | total episodes: 1e6   kl coefficient: 0.05   entropy coefficient: 0.01 |
| LoRA Adapter   Config | r: 1024   $\alpha$: 2048 | dropout: 0.0   bias: False |
| Generation   Config | sampling: true   top k: 0.0   top p: 1.0 | min length: 53   max new tokens: 53   temperature: 0.1 (for DPO and PMLE) or 0.7 (others) |

#### D.3 通用对话

##### 数据集与模型

在此实验中，我们使用 UltraFeedBack 数据集 [^10]，该数据集在各种基线中使用。我们使用 LLaMA-3-8B-Instruct <sup>11</sup> 作为基础模型，使用 ArmoRM-Llama3-8B-v0.1 <sup>12</sup> 作为奖励模型。也可以使用其他公开的基础模型和奖励模型。

##### 实现

与 TL;DR 实验一样，我们的偏好蒸馏实现基于 [^18]，该代码是公开可用的。LLaMA-3-8B-Instruct 的总训练时间在 4 块 A100 GPU 上大约需要 7 天。

##### 超参数

与 TL;DR 实验类似，我们采用了 [^18] 的超参数配置；为了完整性，完整规格在表 6 中呈现。请注意，由于实验规模的限制，我们选择了在 TL;DR 实验中使用的最佳超参数 $\gamma$。因此，所报告的偏好蒸馏性能可能是保守的，因为更精细的超参数调整可能会产生进一步的性能提升。

表 6：通用对话实验的超参数配置。

| 设置 | 参数 |  |
| --- | --- | --- |
| DPO | batch size: 128   learning rate: 5e-7   schedule: cosine decay   train epochs: 1   $\beta$: 0.05 |  |
| PMLE | batch size: 128   learning rate: 3e-7   schedule: cosine decay   train epochs: 1   $\gamma$: 5e-3   KL coefficient: 0.0 |  |
| REBEL | batch size: 128   learning rate: 3e-7   schedule: cosine decay   train epochs: 1   num epochs: 1   $\eta$: $10^{6}$ |  |
| Preference distillation | batch size: 128   learning rate: 3e-7   schedule: cosine decay   train epochs: 1   $\beta$: 0.05 |  |
| Generation   Config | sampling: true   top k: 0.0   top p: 0.9 | min length: 1024   max new tokens: 1024   temperature: 0.8 |

[^1]: A. Agarwal, C. Dann, and T. V. Marinov (2025) Design considerations in offline preference-based rl. arXiv preprint arXiv:2502.06861. Cited by: Appendix A, §C.2, §1, §2, §2, §3, Remark 13.

[^2]: A. Agarwal, N. Jiang, S. M. Kakade, and W. Sun (2019) Reinforcement learning: theory and algorithms. CS Dept., UW Seattle, Seattle, WA, USA, Tech. Rep. Cited by: §2.

[^3]: A. Amini, T. Vieira, and R. Cotterell (2024) Direct preference optimization with an offset. CoRR abs/2402.10571. Cited by: Appendix A.

[^4]: M. G. Azar, Z. D. Guo, B. Piot, R. Munos, M. Rowland, M. Valko, and D. Calandriello (2024) A general theoretical paradigm to understand learning from human preferences. In International Conference on Machine Learning, Cited by: Appendix A, §1.

[^5]: Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, et al. (2022) Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862. Cited by: §1, §5, §5.

[^6]: S. Biderman, H. Schoelkopf, Q. G. Anthony, H. Bradley, K. O'Brien, E. Hallahan, M. A. Khan, S. Purohit, U. S. Prashanth, E. Raff, et al. (2023) Pythia: a suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. Cited by: §D.2, §6.1.

[^7]: R. A. Bradley and M. E. Terry (1952) Rank analysis of incomplete block designs: i. the method of paired comparisons. Biometrika 39 (3/4), pp. 324–345. Cited by: §1.

[^8]: S. Cen, J. Mei, K. Goshvadi, H. Dai, T. Yang, S. Yang, D. Schuurmans, Y. Chi, and B. Dai (2025) Value-incentivized preference optimization: a unified approach to online and offline RLHF. In The Thirteenth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=SQnitDuow6) Cited by: Appendix A.

[^9]: P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei (2017) Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems (NeurIPS). Cited by: Appendix A, §1, §2, §4.

[^10]: G. Cui, L. Yuan, N. Ding, G. Yao, W. Zhu, Y. Ni, G. Xie, Z. Liu, and M. Sun (2023) Ultrafeedback: boosting language models with high-quality feedback. CoRR. Cited by: §D.3, §6.2.

[^11]: Y. Dubois, P. Liang, and T. Hashimoto (2024) Length-controlled alpacaeval: a simple debiasing of automatic evaluators. In First Conference on Language Modeling, External Links: [Link](https://openreview.net/forum?id=CybBmzWBX0) Cited by: §6.2.

[^12]: V. Dumoulin, D. D. Johnson, P. S. Castro, H. Larochelle, and Y. Dauphin (2023) A density estimation perspective on learning from pairwise human preferences. arXiv preprint arXiv:2311.14115. Cited by: §1.

[^13]: K. Ethayarajh, W. Xu, N. Muennighoff, D. Jurafsky, and D. Kiela (2024) Model alignment as prospect theoretic optimization. In Proceedings of the International Conference on Machine Learning (ICML), Cited by: Appendix A.

[^14]: A. Fisch, J. Eisenstein, V. Zayats, A. Agarwal, A. Beirami, C. Nagpal, P. Shaw, and J. Berant (2025) Robust preference optimization through reward model distillation. Transactions on Machine Learning Research (TMLR). External Links: ISSN 2835-8856 Cited by: Appendix A, 2nd item, §3, §4.

[^15]: D. J. Foster and A. Krishnamurthy (2021) Efficient first-order contextual bandits: prediction, allocation, and triangular discrimination. Advances in Neural Information Processing Systems (NeurIPS), pp. 18907–18919. Cited by: §C.1, §C.2.

[^16]: D. J. Foster, Z. Mhammedi, and D. Rohatgi (2025) Is a good foundation necessary for efficient reinforcement learning? the computational role of the base model in exploration. External Links: 2503.07453, [Link](https://arxiv.org/abs/2503.07453) Cited by: Appendix A.

[^17]: G. Gabbianelli, G. Neu, and M. Papini (2024) Importance-weighted offline learning done right. In International Conference on Algorithmic Learning Theory (ALT), pp. 614–634. Cited by: footnote 3.

[^18]: Z. Gao, J. Chang, W. Zhan, O. Oertell, G. Swamy, K. Brantley, T. Joachims, D. Bagnell, J. D. Lee, and W. Sun (2024) Rebel: reinforcement learning via regressing relative rewards. Advances in Neural Information Processing Systems (NeurIPS). Cited by: Appendix A, §D.2, §D.2, §D.2, §D.3, §D.3, 2nd item, Table 1, §4, §6.1, §6.2.

[^19]: I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio (2014) Generative adversarial nets. Advances in Neural Information Processing Systems (NeurIPS). Cited by: §5.

[^20]: A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. (2024) The llama 3 herd of models. arXiv preprint arXiv:2407.21783. Cited by: §6.2.

[^21]: S. Guo, B. Zhang, T. Liu, T. Liu, M. Khalman, F. Llinares, A. Rame, T. Mesnard, Y. Zhao, B. Piot, et al. (2024) Direct language model alignment from online ai feedback. arXiv preprint arXiv:2402.04792. Cited by: §4.

[^22]: J. Hong, N. Lee, and J. Thorne (2024) ORPO: monolithic preference optimization without reference model. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), Cited by: Appendix A.

[^23]: E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen (2022) LoRA: low-rank adaptation of large language models. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=nZeVKeeFYf9) Cited by: §D.2.

[^24]: A. Huang, W. Zhan, T. Xie, J. D. Lee, W. Sun, A. Krishnamurthy, and D. J. Foster (2025) Correcting the mythos of KL-regularization: direct alignment without overoptimization via chi-squared preference optimization. In International Conference on Learning Representations, Cited by: Appendix A, Appendix A, §1, §1, §2, §2, footnote 3.

[^25]: S. Huang, M. Noukhovitch, A. Hosseini, K. Rasul, W. Wang, and L. Tunstall (2024) The n+ implementation details of RLHF with PPO: a case study on TL;DR summarization. In First Conference on Language Modeling, External Links: [Link](https://openreview.net/forum?id=kHO2ZTa8e3) Cited by: §D.2, §D.2.

[^26]: T. Kaufmann, P. Weng, V. Bengs, and E. Hüllermeier (2024) A survey of reinforcement learning from human feedback. External Links: 2312.14925, [Link](https://arxiv.org/abs/2312.14925) Cited by: Appendix A.

[^27]: B. Kveton, X. Li, J. McAuley, R. Rossi, J. Shang, J. Wu, and T. Yu (2025) Active learning for direct preference optimization. External Links: 2503.01076 Cited by: §1.

[^28]: K. Lee, H. Liu, M. Ryu, O. Watkins, Y. Du, C. Boutilier, P. Abbeel, M. Ghavamzadeh, and S. S. Gu (2023) Aligning text-to-image models using human feedback. External Links: 2302.12192, [Link](https://arxiv.org/abs/2302.12192) Cited by: Appendix A.

[^29]: T. Li, W. Chiang, E. Frick, L. Dunlap, T. Wu, B. Zhu, J. E. Gonzalez, and I. Stoica (2025) From crowdsourced data to high-quality benchmarks: arena-hard and benchbuilder pipeline. In International Conference on Machine Learning, pp. 34209–34231. Cited by: §6.2.

[^30]: X. L. Li, V. Shrivastava, S. Li, T. Hashimoto, and P. Liang (2024) Benchmarking and improving generator-validator consistency of language models. In The Twelfth International Conference on Learning Representations, Cited by: §4.

[^31]: Y. Liang, J. He, G. Li, P. Li, A. Klimovskiy, N. Carolan, J. Sun, J. Pont-Tuset, S. Young, F. Yang, J. Ke, K. D. Dvijotham, K. M. Collins, Y. Luo, Y. Li, K. J. Kohlhoff, D. Ramachandran, and V. Navalpakkam (2024) Rich human feedback for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19401–19411. Cited by: Appendix A.

[^32]: J. Liu, Z. Zhou, J. Liu, X. Bu, C. Yang, H. Zhong, and W. Ouyang (2024) Iterative length-regularized direct preference optimization: a case study on improving 7b language models to gpt-4 level. External Links: 2406.11817, [Link](https://arxiv.org/abs/2406.11817) Cited by: Appendix A.

[^33]: Q. Mao, H. Lee, H. Tseng, S. Ma, and M. Yang (2019) Mode seeking generative adversarial networks for diverse image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1429–1437. Cited by: §5.

[^34]: Y. Meng, M. Xia, and D. Chen (2024) Simpo: simple preference optimization with a reference-free reward. Advances in Neural Information Processing Systems (NeurIPS) 37, pp. 124198–124235. Cited by: Appendix A.

[^35]: R. Munos (2003) Error bounds for approximate policy iteration. In Proceedings of the International Conference on Machine Learning (ICML), Cited by: §2.

[^36]: R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders, X. Jiang, K. Cobbe, T. Eloundou, G. Krueger, K. Button, M. Knight, B. Chess, and J. Schulman (2022) WebGPT: browser-assisted question-answering with human feedback. External Links: 2112.09332, [Link](https://arxiv.org/abs/2112.09332) Cited by: Appendix A.

[^37]: A. Nath, C. Jung, E. Seefried, and N. Krishnaswamy (2025) Simultaneous reward distillation and preference learning: get you a language model who can do both. External Links: 2410.08458, [Link](https://arxiv.org/abs/2410.08458) Cited by: Appendix A.

[^38]: OpenAI (2022) ChatGPT: Optimizing Language Models for Dialogue. Note: [https://openai.com/blog/chatgpt](https://openai.com/blog/chatgpt) Accessed: 2025-05-19 Cited by: Appendix A.

[^39]: L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. (2022) Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems (NeurIPS). Cited by: Appendix A, §1, §5, §5.

[^40]: R. Park, R. Rafailov, S. Ermon, and C. Finn (2024) Disentangling length from quality in direct preference optimization. In Findings of the Association for Computational Linguistics (ACL), Cited by: Appendix A.

[^41]: B. Qi, P. Li, F. Li, J. Gao, K. Zhang, and B. Zhou (2024) Online dpo: online direct preference optimization with fast-slow chasing. External Links: 2406.05534, [Link](https://arxiv.org/abs/2406.05534) Cited by: Appendix A.

[^42]: R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn (2023) Direct preference optimization: your language model is secretly a reward model. Advances in Neural Information Processing Systems (NeurIPS) 36, pp. 53728–53741. Cited by: Appendix A, §1, §1, §3, §4, §5.

[^43]: J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017) Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. Cited by: §5.

[^44]: J. Schulman (2020) External Links: [Link](http://joschu.net/blog/kl-approx.html) Cited by: §C.2, §C.2, §3.

[^45]: Y. Song, G. Swamy, A. Singh, J. Bagnell, and W. Sun (2024) The importance of online data: understanding preference fine-tuning via coverage. Advances in Neural Information Processing Systems (NeurIPS) 37, pp. 12243–12270. Cited by: Appendix A, §D.2, §D.2, §D.2, §3, §6.1, §6.1.

[^46]: N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano (2020) Learning to summarize with human feedback. Advances in Neural Information Processing Systems (NeurIPS). Cited by: Appendix A, §1, §2, §5, §5, §6.1.

[^47]: G. Swamy, S. Choudhury, W. Sun, Z. S. Wu, and J. A. Bagnell (2025) All roads lead to likelihood: the value of reinforcement learning in fine-tuning. External Links: 2503.01067 Cited by: §4.

[^48]: Y. Tang, Z. D. Guo, Z. Zheng, D. Calandriello, R. Munos, M. Rowland, P. H. Richemond, M. Valko, B. Avila Pires, and B. Piot (2024) Generalized preference optimization: a unified approach to offline alignment. In Proceedings of the International Conference on Machine Learning (ICML), Cited by: Appendix A.

[^49]: L. Tunstall, E. E. Beeching, N. Lambert, N. Rajani, K. Rasul, Y. Belkada, S. Huang, L. V. Werra, C. Fourrier, N. Habib, N. Sarrazin, O. Sanseviero, A. M. Rush, and T. Wolf (2024) Zephyr: direct distillation of LM alignment. In First Conference on Language Modeling, Cited by: Appendix A.

[^50]: S.A. van de Geer (2009) Empirical processes in m-estimation. Cambridge Series in Statistical and Probabilistic Mathematics, Cambridge University Press. Cited by: §C.2.

[^51]: B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, and N. Naik (2023) Diffusion model alignment using direct preference optimization. External Links: 2311.12908, [Link](https://arxiv.org/abs/2311.12908) Cited by: Appendix A.

[^52]: H. Wang, W. Xiong, T. Xie, H. Zhao, and T. Zhang (2024) Interpretable preferences via multi-objective reward modeling and mixture-of-experts. arXiv preprint arXiv:2406.12845. Cited by: §6.2.

[^53]: P. West, X. Lu, N. Dziri, F. Brahman, L. Li, J. D. Hwang, L. Jiang, J. Fisher, A. Ravichander, K. Chandu, et al. (2024) The generative ai paradox:"what it can create, it may not understand". In The Twelfth International Conference on Learning Representations, Cited by: §4.

[^54]: T. Xiao, Y. Yuan, H. Zhu, M. Li, and V. G. Honavar (2024) Cal-dpo: calibrated direct preference optimization for language model alignment. In The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS), Cited by: Appendix A.

[^55]: T. Xie, C. Cheng, N. Jiang, P. Mineiro, and A. Agarwal (2021) Bellman-consistent pessimism for offline reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: §2.

[^56]: T. Xie, D. J. Foster, A. Krishnamurthy, C. Rosset, A. H. Awadallah, and A. Rakhlin (2025) Exploratory preference optimization: harnessing implicit q\*-approximation for sample-efficient RLHF. In The Thirteenth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=QYigQ6gXNw) Cited by: Appendix A, §2.

[^57]: T. Xie, D. J. Foster, A. Krishnamurthy, C. Rosset, A. H. Awadallah, and A. Rakhlin (2025) Exploratory preference optimization: harnessing implicit q\*-approximation for sample-efficient RLHF. In International Conference on Learning Representations, Cited by: §1, §1, §2.

[^58]: W. Xiong, H. Dong, C. Ye, Z. Wang, H. Zhong, H. Ji, N. Jiang, and T. Zhang (2024) Iterative preference learning from human feedback: bridging theory and practice for RLHF under KL-constraint. In Proceedings of the International Conference on Machine Learning (ICML), Cited by: Appendix A, §1, §1, §2.

[^59]: H. Yuan, Z. Yuan, C. Tan, W. Wang, S. Huang, and F. Huang (2023) Rrhf: rank responses to align language models with human feedback. Advances in Neural Information Processing Systems (NeurIPS). Cited by: Appendix A.

[^60]: W. Zhan, B. Huang, A. Huang, N. Jiang, and J. Lee (2022) Offline reinforcement learning with realizability and single-policy concentrability. In Proceedings of the Conference on Learning Theory (COLT), Cited by: footnote 3.

[^61]: W. Zhan, M. Uehara, N. Kallus, J. D. Lee, and W. Sun (2024) Provable offline preference-based reinforcement learning. In The Twelfth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=tVMPfEGT2w) Cited by: Appendix A, §1, §1, §2, §2.

[^62]: S. Zhang, D. Yu, H. Sharma, H. Zhong, Z. Liu, Z. Yang, S. Wang, H. Hassan, and Z. Wang (2024) Self-exploring language models: active preference elicitation for online alignment. External Links: 2405.19332, [Link](https://arxiv.org/abs/2405.19332) Cited by: Appendix A, §1, §1, §2.

[^63]: T. Zhang (2007) From $\varepsilon$ -entropy to kl-entropy: analysis of minimum information complexity density estimation. The Annals of Statistics 34. Cited by: §C.2.

[^64]: Y. Zhang, L. Wang, M. Fang, Y. Du, C. Huang, J. Wang, Q. Lin, M. Pechenizkiy, D. Zhang, S. Rajmohan, and Q. Zhang (2025) Distill not only data but also rewards: can smaller language models surpass larger ones?. External Links: 2502.19557, [Link](https://arxiv.org/abs/2502.19557) Cited by: Appendix A, §2.

[^65]: Y. Zhao, R. Joshi, T. Liu, M. Khalman, M. Saleh, and P. J. Liu (2023) SLiC-hf: sequence likelihood calibration with human feedback. CoRR abs/2305.10425. Cited by: Appendix A.

[^66]: Y. Zhao, M. Khalman, R. Joshi, S. Narayan, M. Saleh, and P. J. Liu (2023) Calibrating sequence likelihood improves conditional language generation. In Proceedings of the International Conference on Learning Representations (ICLR), Cited by: Appendix A.

[^67]: L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, et al. (2023) Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems 36, pp. 46595–46623. Cited by: §6.2.

[^68]: B. D. Ziebart (2010) Modeling purposeful adaptive behavior with the principle of maximum causal entropy. Carnegie Mellon University. Cited by: §5.

[^69]: D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving (2019) Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593. Cited by: Appendix A.