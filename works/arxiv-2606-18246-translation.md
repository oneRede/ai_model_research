---
sourceTitle: "Variable-Width Transformers"
sourceUrl: "https://arxiv.org/html/2606.18246v1"
sourceRequestedUrl: "https://arxiv.org/html/2606.18246v1"
title: "可变宽度 Transformer"
sourceAdapter: "generic"
sourceCapturedAt: "2026-08-14T02:27:26.965Z"
sourceConversionMethod: "defuddle"
sourceKind: "generic/article"
sourceLanguage: "en"
translatedLanguage: "zh-CN"
translatedAt: "2026-08-14"
arxivId: "2606.18246"
authors: "Zhaofeng Wu, Oliver Sieberling, Shawn Tan, Rameswar Panda, Yury Polyanskiy, Yoon Kim"
affiliation: "MIT & MIT-IBM Watson AI Lab"
publishedDate: "2026-06-16"
sourceFigureCount: 9
pipelineRunId: "20260814-101401-arxiv-2606-18246"
pipelineSource: "translate/20260814-101401/works-ready/arxiv-2606-18246-translation.md"
---

# 可变宽度 Transformer

**作者**：Zhaofeng Wu, Oliver Sieberling, Shawn Tan, Rameswar Panda, Yury Polyanskiy, Yoon Kim

**单位**：麻省理工学院（MIT）& MIT-IBM Watson AI Lab

## 摘要

扩大模型规模，特别是深度和宽度，推动了基于 Transformer 的语言模型取得重大进展。然而，大多数架构在所有层中保持恒定宽度，尽管不同层可能扮演不同的计算角色，但参数和计算预算仍被均匀分配。在本工作中，我们通过提出一种 ×形的 ><former 架构，对跨网络深度的非均匀容量分配（Nonuniform Capacity Allocation）进行了实证研究。该设计保持较宽的早期和晚期层，同时收窄中间层，利用无参数调整（Parameter-Free Resizing）的残差调整机制。在 200M 到 2B 参数的仅解码器语言模型（稠密模型）以及 3B 参数的混合专家模型（MoE）上，我们的 ><former 在语言建模损失上始终优于参数匹配的恒定宽度基线。通过减少平均层宽度，该架构还需要更少的总体浮点运算量（FLOPs）（在拟合的损失匹配缩放曲线下减少 22%），以及更小的 KV 缓存（KV Cache）内存和 I/O 开销（I/O Cost）（减少 15%）。在分析中，我们展示了这种瓶颈（Bottleneck）结构导致残差流（Residual Stream）中的表示在质量上有所不同。总体而言，我们的结果表明，非均匀宽度分配可以实现语言模型更资源优化的缩放。

<sup>†</sup>![Refer to caption](imgs/img-001-overview_bowtie.png)

图 1：我们提出 ><former 架构，该架构在早期和晚期层使用较宽的隐藏维度，在中间层使用较窄的维度，形成蝴蝶结形（×形）的宽度分布。

## 1 引言

缩放一直是现代人工智能进展的关键驱动力。缩放的一个主要维度是模型规模。在 Transformer 的背景下，模型规模是 Transformer 块维度（模型的"宽度"）<sup>1</sup>和 Transformer 块数量（模型的"深度"）的函数。因此，许多先前工作研究了如何通过缩放 Transformer 宽度和深度来优化模型规模。早期研究认为，模型形状（即宽度与深度的比例）对性能的影响不如总参数量重要 [^19]，但后续工作发现模型形状会导致显著差异 [^20] [^39] [^31]，在拟合缩放定律时应当考虑这一因素 [^23]。

然而，即使在关于 Transformer 最优全局形状的争论中，这些研究通常保留了一个较少被审视的假设：模型的宽度在深度上是恒定的。也就是说，一旦选择了隐藏维度大小，每个 Transformer 块都会获得大致相同的计算/参数预算。这种恒定宽度（Uniform-Width）设计很方便，但显然不是最优的。不同的层可能在计算过程中扮演不同的角色，固定的总参数或浮点运算量预算不必在深度上均匀分配。这引出了一个普遍的问题：在固定深度和参数预算下，所有层是否应该具有相同的宽度，还是应该非均匀地分配容量？

我们通过训练具有跨深度非均匀参数和计算分配的仅解码器 Transformer 语言模型，对这个问题进行实证研究。具体而言，对于给定的参数和深度约束，我们在几种设置下改变模型形状：增长型（∨形）、收窄型（∧形）、先增长后收窄（◇形）以及先收窄后增长（×形）。在这些设置中，我们发现 ×形模型（早期和晚期层较宽但中间层较窄）优于参数匹配的恒定宽度 Transformer。我们称其为 ><former。这与先前关于仅对前馈网络（FFN）中间维度进行逐层分配的工作不同，后者发现将更多计算分配给中间层有益 [^17]；相反，我们的结果表明，重新分配完整的块宽度会导致不同的最优配置。

一个关键的实现细节是可变宽度层如何与残差流交互。天真地在层之间改变残差维度会引入投影瓶颈并改变跳跃路径。相反，我们保持固定的全局残差维度，并允许每个块读取和写入残差流的特定层切片。未被给定块使用的坐标绕过该块，并通过复制在上游投影。我们发现，这种固定残差（Fixed Residual Dimension）构造对于实现非均匀宽度配置的收益至关重要。

非均匀宽度分配也具有效率优势：与恒定宽度 Transformer 相比，它需要更少的训练和推理浮点运算量，同时还减少了 KV 缓存内存和移动激活的 I/O 开销。因为层的参数量与其宽度的平方成正比，而注意力浮点运算量和 KV 缓存大小与宽度线性相关，匹配恒定基线的参数量会导致平均层宽度的减少（因此 KV 缓存也减少）。在 200M–2B 参数的模型上，><former 在困惑度上相对于参数匹配的恒定宽度基线实现了约 3% 的相对改进，同时将 KV 缓存大小减少约 10%，浮点运算量减少约 3%。这些优势也扩展到混合专家模型 Transformer。我们进一步分析了最优瓶颈宽度和瓶颈位置如何依赖于模型预算，为缩放非均匀宽度 Transformer 提供了实证指导。最后，我们还进行了分析以理解 ><former 的优势，表明它采用了与恒定宽度基线不同的表示策略，并缓解了中间层的表示坍缩（Representation Collapse）。

## 2 可变宽度 Transformer

标准 Transformer 包含一系列 $L$ 层。在每一层 $\ell\in[1,L]$ 中，一个 Transformer 块 $\mathcal{B}^{\ell}:\mathbb{R}^{d}\to\mathbb{R}^{d}$ 通过 ${\mathbf{x}}^{\ell}=\mathcal{B}^{\ell}({\mathbf{x}}^{\ell-1})+{\mathbf{x}}^{\ell-1}$ 变换来自前一层的输入 ${\mathbf{x}}^{\ell-1}$。$d$ 是模型维度。我们将 ${\mathbf{x}}^{0}$ 定义为输入嵌入。

在本工作中，我们质疑为什么 $d$ 必须保持恒定。许多过去的工作表明，Transformer 语言模型的不同层执行不同的功能，这自然可能需要不同的容量 [^41] [^26] [^34]。这促使每一层 $\ell$ 具有不同的维度 $d_{\ell}$。

然而，一个实际的挑战是，这需要在层之间调整大小：${\mathbf{x}}^{\ell}=\mathcal{B}^{\ell}(f^{\ell}({\mathbf{x}}^{\ell-1}))+f^{\ell}({\mathbf{x}}^{\ell-1})$，其中 $f^{\ell}:\mathbb{R}^{d_{\ell-1}}\to\mathbb{R}^{d_{\ell}}$ 用于调整隐藏状态的大小。我们考虑一种无参数方法。当收缩维度时，即 $d_{\ell}<d_{\ell-1}$，我们简单地截断额外的维度，即 $f^{\ell}({\mathbf{x}})={\mathbf{x}}[:d_{\ell}]$。当扩展维度时，即 $d_{\ell}>d_{\ell-1}$，我们从最近主动处理它的层恢复每个先前截断的维度。形式化地，对于每个坐标索引 $i\in\{1,\dots,d_{\ell}\}$，调整大小后的隐藏状态 $f^{\ell}({\mathbf{x}}^{\ell-1})\in\mathbb{R}^{d_{\ell}}$ 的第 $i$ 个元素构造为：

$$
[f^{\ell}({\mathbf{x}}^{\ell-1})]_{i}=[{\mathbf{x}}^{\ell^{\prime}}]_{i}\quad\text{其中}\quad\ell^{\prime}=\max\{\tilde{\ell}<\ell\mid d_{\tilde{\ell}}\geq i\}
$$

如果不存在这样的先前层（即，如果所需维度超过所有先前层的最大宽度），则该坐标用 $0$ 填充。参见 §4.4 中的消融实验，显示这些方法优于训练投影层或始终用 0 填充等替代方案。

使用这种扩展方法，我们可以从数学上将可变宽度模型概念化为恒定宽度模型，除了（1）每层只读取/写入残差流维度的一个子集，以及（2）它具有更大的残差流宽度（等于最宽层的宽度）。

我们使用两个额外的参数研究不同的形状：$\ell^{*}$，瓶颈层的层索引，以及 $d_{\ell^{*}}$，其维度。我们以几何方式参数化其余层宽度：<sup>2</sup> 对于 $\ell\leq\ell^{*}$（早期层），$d_{\ell}=\alpha^{-}d_{\ell-1}$，变化率为 $\alpha^{-}$；对于 $\ell>\ell^{*}$（晚期层），$d_{\ell}=\alpha^{+}d_{\ell-1}$，变化率为 $\alpha^{+}$。通过 $\alpha^{+}$、$\alpha^{-}$ 和 $\ell^{*}$ 的不同设置，我们可以恢复不同类型的形状：当 $\alpha^{-}<1$ 且 $\alpha^{+}>1$ 时，我们获得 ×形模型；当 $\alpha^{-}>1$ 且 $\alpha^{+}<1$ 时，我们获得 ◇形模型；当 $\ell^{*}$ 为 1 或 $L$ 时，我们获得 ∨形或 ∧形模型。我们保持输入和输出嵌入的大小不变：第一层的 QKV 投影和最后一层的多层感知机（MLP）下投影会调整这些大小不匹配。<sup>3</sup>

我们的可变宽度架构的一个引人注目的特性是，当在参数量上与恒定宽度基线匹配时，它需要严格更少的总体浮点运算量，并且具有严格更低的平均层宽度（因此，更低的 KV 缓存大小和移动激活的更低 I/O 开销）。具体而言，Transformer 层的参数量由线性投影矩阵（QKV、输出和 MLP）主导，这些矩阵与隐藏维度的平方成正比：$P_{\ell}\approx Kd_{\ell}^{2}$，其中 $K$ 是一个常数，取决于投影矩阵的数量和 MLP 扩展因子。因此，如果我们将可变宽度模型的参数量与恒定宽度 $d$ 的基线匹配，我们实际上等同于维度平方和：

$$
K\sum_{\ell=1}^{L}d_{\ell}^{2}=KLd^{2}\implies\frac{1}{L}\sum_{\ell=1}^{L}d_{\ell}^{2}=d^{2}.
$$

因为均值的平方上界为平方的均值，并且因为可变宽度确保宽度 $d_{\ell}$ 不是恒定的，所以平均层大小严格更小：

$$
\left(\frac{1}{L}\sum_{\ell=1}^{L}d_{\ell}\right)^{2}<\frac{1}{L}\sum_{\ell=1}^{L}d_{\ell}^{2}=d^{2}\implies\frac{1}{L}\sum_{\ell=1}^{L}d_{\ell}<d.
$$

对于浮点运算量，首先，线性投影中每个 token 的浮点运算数量严格与权重数量成正比，因此在参数匹配时，总稠密浮点运算量与基线保持相同。对于注意力点积，其浮点运算量与隐藏维度线性相关：$\text{FLOPs}_{\ell}\propto N^{2}d_{\ell}$，其中 $N$ 是序列长度。因此，总注意力计算 $\sum_{\ell=1}^{L}N^{2}d_{\ell}=N^{2}\sum_{\ell=1}^{L}d_{\ell}$ 因此严格低于基线 $N^{2}Ld$。<sup>4</sup>

总结而言，可变宽度 Transformer 有 4 个参数：$\ell^{*}$、$d_{\ell^{*}}$、$\alpha^{+}$ 和 $\alpha^{-}$。我们设置两个约束：$d_{1}=d_{L}$（对于 ◇形和 ×形）以及参数量与恒定宽度基线匹配。<sup>5</sup>因此，在我们的实验中，我们将 $\ell^{*}$ 和 $d_{\ell^{*}}$ 视为两个超参数，并自动求解所有层宽度。<sup>6</sup>推导过程参见附录 A。

## 3 ><former

表 1：预训练超参数。

<table><tbody><tr><td>参数量</td><td>层数 (<math><semantics><mi>L</mi> <annotation>L</annotation></semantics></math>)</td><td>隐藏维度 (<math><semantics><mi>d</mi> <annotation>d</annotation></semantics></math>)</td><td>批大小</td><td>Token 数</td><td>专家数（总/激活）</td><td>MLP 中间大小</td></tr><tr><td colspan="7">稠密模型</td></tr><tr><td>200M</td><td>16</td><td>640</td><td>512</td><td>10B</td><td>–</td><td>–</td></tr><tr><td>500M</td><td>24</td><td>960</td><td>1024</td><td>25B</td><td>–</td><td>–</td></tr><tr><td>1B</td><td>32</td><td>1280</td><td>2048</td><td>50B</td><td>–</td><td>–</td></tr><tr><td>2B</td><td>40</td><td>1600</td><td>4096</td><td>100B</td><td>–</td><td>–</td></tr><tr><td colspan="7">混合专家模型</td></tr><tr><td>3B（1B 激活）</td><td>40</td><td>1600</td><td>4096</td><td>100B</td><td>22 / 3</td><td>512</td></tr></tbody></table>

在本节中，我们讨论训练可变宽度 Transformer。在介绍训练设置后，我们首先确定 ×形模型效果最好，然后识别在不同模型规模下表现良好的瓶颈层索引 $\ell^{*}$ 和维度 $d_{\ell^{*}}$ 的参数化。使用这个配方，我们在各种规模上预训练 ><former 和恒定宽度基线，发现 ><former 始终以更小的预训练浮点运算量占用和 KV 缓存大小实现更好的损失和下游任务性能。

### 3.1 训练设置

我们预训练四种模型规模——200M、500M、1B 和 2B——具有不同的层数和隐藏大小（表 1）。对于每种规模，我们预训练恒定宽度 Transformer 和可变宽度模型。我们还考虑了一个具有 3B 总参数/1B 激活参数的混合专家模型。为了将可变宽度 MoE 模型与基线进行参数匹配，我们匹配总参数数量——这导致可变宽度模型的激活参数减少 3%，但我们在 §3.4 中展示，尽管如此，它仍然优于恒定宽度基线。

对于预训练数据，我们在 DCLM 数据集上训练 [^21]。对于每种模型规模，我们将模型训练到 Chinchilla 最优（Chinchilla-Optimal）的 2.5 倍 [^16]，即训练的 token 数量等于参数量的 50 倍，例如，2B 模型训练 100B token。我们在长度为 4096 的序列上训练，批大小取决于模型规模（表 1）。输入使用 OpenAI 的 cl100k_base 分词器进行分词。<sup>7</sup>

所有模型均使用最大更新参数化（Maximal Update Parametrization，µP）训练 [^43]。我们使用 µP 感知的初始化和优化器参数组，在各个规模上使用相同的 AdamW 超参数：学习率 $10^{-2}$，$\beta=(0.9,0.95)$，权重衰减 0.1，以及 $\epsilon=10^{-10}$。我们使用幂次学习率衰减（Power Learning-Rate Decay）计划。学习率在前约 8% 的训练步骤中线性预热，然后在剩余步骤中衰减。所有模型均以 bfloat16 精度训练。遵循常见做法，我们从所有线性投影（包括注意力、MLP 和输出投影）中省略偏置项 [^3] [^12]。我们使用 SwiGLU 激活函数（SwiGLU Activation）[^36]。

我们测量每个模型的训练损失，<sup>8</sup>在最后 1,000 步上平均（以 10 步为增量）进行平滑。我们还报告预训练浮点运算量，单位为 PFLOP/s-天，即假设每秒 1 PFLOP 所需的天数 [^40]。最后，我们比较平均层大小，它与推理期间的 KV 缓存大小成正比。

### 3.2 ×形效果最好

图 2：比较不同形状的可变宽度 Transformer，每种形状在多个超参数选择上进行扫描。×形模型表现最好。

我们首先在 500M 参数规模上探索不同的形状。除了常规的恒定宽度 Transformer 外，我们还实验了 ◇形、×形、∨形和 ∧形。由于每种形状的最优超参数（$\ell^{*}$ 和 $d_{\ell^{*}}$，§2）可能不同，我们对每种形状的每个超参数实验 3 种选择。这相当于 ×形和 ◇形各 9 次运行，以及 ∨形和 ∧形各 3 次运行，对于后者 $\ell^{*}$ 不相关。在图 2 中，我们绘制了它们的损失和预训练浮点运算量。我们看到 ×形始终表现最好。<sup>9</sup>

### 3.3 寻找特定的宽度调度

![Refer to caption](imgs/img-002-hparams.png)

图 3：瓶颈层索引和维度对语言建模损失的影响，参数化为总层数和基础维度的比率。$\ell^{*}=r_{\ell}L$ 和 $d_{l^{*}}=r_{d}d$。我们还显示了基线性能（用 $r_{d}=1$ 表示）和结果的平均层大小。这种参数化在各种模型规模上产生了相对一致的模型性能模式。

如 §2 所述，我们需要选择瓶颈层索引 $\ell^{*}$ 和瓶颈维度 $d_{l^{*}}$。理想情况下，我们希望找到一个在各种模型规模上都表现良好的配方，这样我们就不必在每个模型规模上单独搜索它们。因此，我们将这两个超参数参数化为总层数 $L$ 和隐藏大小 $d$ 的比率：$\ell^{*}=r_{\ell}L$ 和 $d_{l^{*}}=r_{d}d$。我们在小模型规模上扫描 $r_{\ell}$ 和 $r_{d}$ 的不同值：200M、500M 和 1B。图 3 显示了结果。虽然并非单一的 $(r_{\ell},r_{d})$ 对始终最优，但这种基于比率的参数化在各种模型规模上导致*大致*相似的趋势这一事实很有趣。基于这次扫描，默认情况下，我们使用 $\ell^{*}=0.75L$ 和 $d_{l^{*}}=0.3d$。

### 3.4 ><former 优于恒定宽度 Transformer

表 2：><former 与恒定宽度 Transformer 的性能、预训练浮点运算量和平均层大小。><former 始终以更低的预训练浮点运算量和平均层大小实现更低的损失。

<table><tbody><tr><td>规模</td><td>模型</td><td>损失</td><td>PFLOP/s-天</td><td>平均层大小</td></tr><tr><td rowspan="2">200M</td><td>Transformer</td><td>3.452</td><td>0.18</td><td>640</td></tr><tr><td>> <former</td><td>3.430</td><td>0.17 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 3.2%)</td><td>576 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 10.0%)</td></tr><tr><td rowspan="2">500M</td><td>Transformer</td><td>3.138</td><td>1.11</td><td>960</td></tr><tr><td>> <former</td><td>3.099</td><td>1.07 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 3.7%)</td><td>855 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 11.0%)</td></tr><tr><td rowspan="2">1B</td><td>Transformer</td><td>2.926</td><td>4.52</td><td>1280</td></tr><tr><td>> <former</td><td>2.890</td><td>4.41 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 2.6%)</td><td>1145 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 10.5%)</td></tr><tr><td rowspan="2">2B</td><td>Transformer</td><td>2.751</td><td>16.92</td><td>1600</td></tr><tr><td>> <former</td><td>2.726</td><td>16.49 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 2.5%)</td><td>1426 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 10.9%)</td></tr><tr><td rowspan="2">3B/1B MoE</td><td>Transformer</td><td>2.726</td><td>10.13</td><td>1600</td></tr><tr><td>> <former</td><td>2.710</td><td>9.66 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 4.6%)</td><td>1426 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 10.9%)</td></tr></tbody></table>

图 4：语言建模损失与预训练浮点运算量（左）和平均层大小（右）的关系。><former 以更小的浮点运算量和平均层大小成本产生更低的损失。

<table><tbody><tr><td></td><td colspan="12">准确率 (<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math>)</td><td colspan="2">困惑度 (<math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math>)</td></tr><tr><td>模型</td><td>ARC-C</td><td>ARC-E</td><td>BoolQ</td><td>COPA</td><td>HellaSwag</td><td>LAMBADA</td><td>OBQA</td><td>PIQA</td><td>RACE</td><td>SciQ</td><td>WinoGrande</td><td>平均</td><td>LAMBADA</td><td>WikiText</td></tr><tr><td>2B 恒定宽度</td><td>33.0</td><td>59.5</td><td>59.4</td><td>76.0</td><td>55.9</td><td>55.4</td><td>33.8</td><td>73.3</td><td>34.3</td><td>79.5</td><td>57.0</td><td>56.1</td><td>8.18</td><td>16.96</td></tr><tr><td>2B > <former</td><td>34.4</td><td>63.3</td><td>60.9</td><td>73.0</td><td>57.9</td><td>56.1</td><td>33.6</td><td>74.4</td><td>33.4</td><td>82.0</td><td>60.2</td><td>57.2</td><td>7.43</td><td>16.32</td></tr><tr><td>MoE 基线</td><td>33.7</td><td>62.2</td><td>63.0</td><td>77.0</td><td>57.3</td><td>56.1</td><td>37.4</td><td>74.8</td><td>34.6</td><td>83.9</td><td>55.6</td><td>57.8</td><td>7.78</td><td>16.36</td></tr><tr><td>MoE > <former</td><td>33.2</td><td>61.0</td><td>59.5</td><td>80.0</td><td>58.7</td><td>56.2</td><td>37.8</td><td>75.2</td><td>34.3</td><td>80.2</td><td>60.1</td><td>57.8</td><td>7.45</td><td>15.98</td></tr></tbody></table>

表 3：模型在标准语言模型评估数据集上的性能。对于准确率指标，粗体表示在 $p<0.05$ 的单侧检验下显著更高的值；对于困惑度，粗体表示更低的值。><former 在基于困惑度的任务上始终优于恒定宽度 Transformer，2B ><former 在大多数自然语言理解任务上获胜。

表 2 显示，在我们测试的所有模型规模上，><former 优于恒定宽度 Transformer，同时需要更少的浮点运算量和平均层大小（即 KV 缓存大小减少）。

在图 4（左）中，我们对 ><former 和恒定宽度 Transformer 拟合损失与预训练浮点运算量的缩放定律曲线 [^19]，发现拟合紧密。同样，在图 4（右）中，我们也发现损失与平均层大小之间存在紧密的幂律拟合。从这些缩放定律曲线中，我们计算出 ><former 可以用 77.8% 的浮点运算量和 85.1% 的平均层宽度实现 2B 恒定宽度 Transformer 的损失（2.751）。此外，两条缩放定律曲线都显示，><former 不仅具有更小的截距，而且缩放指数也略陡，这表明差距可能在更大规模上扩大。

我们还使用 lm-evaluation-harness [^8] 在零样本（Zero-shot）设置下在标准语言模型下游评估基准上测试这些模型。该测试套件涵盖自然语言理解（NLU）任务，如常识推理、阅读理解等，以及基于困惑度的任务。对于多项选择任务，我们在可用时报告归一化准确率，因为它通过归一化选择可能性来纠正答案长度效应。当不提供时，我们报告标准准确率。我们在附录 B 的表 5 中报告数据集统计信息和指标。我们评估了 2B 模型和 MoE 模型，并在表 3 中展示了结果。><former 在基于困惑度的任务上始终优于恒定宽度 Transformer。2B ><former 在大多数 NLU 任务上也领先。MoE ><former 在 NLU 准确率上表现参差不齐，但改进了两个困惑度指标；在这些模型规模上，我们将困惑度视为语言模型质量的更具信息性的指标。我们再次注意到，><former 以更少的浮点运算量和内存实现这一点，对于 MoE 模型，激活参数也更少（§3.1）。

图 5：2B ><former 与 2B 恒定宽度 Transformer 中 MLP 激活维度的利用频率，分别针对每一层可视化。阴影面板对应于瓶颈层。><former 更均匀地利用 MLP 激活维度。

## 4 分析

众所周知，Transformer 对深度的使用效率不高 [^13]，经常出现"压缩谷（Compression Valley）"，即中间层的表示能力坍缩并压缩计算 [^37] [^6]。通过检查 MLP 中间激活和每层之后的残差流，我们发现 ><former 采用了不同的表示策略，它缓解了中间层的坍缩，并比恒定宽度 Transformer 更有效地使用其容量。

### 4.1 ><former 改进 MLP 激活利用率

直观地说，><former 强制执行信息瓶颈，可能鼓励模型更有效地使用其表示容量。我们通过检查中间 MLP 激活的利用率来操作化这一点。先前的可解释性工作将 Transformer MLP 层视为包含键值记忆：上投影层编码不同概念的键，中间激活表示 MLP 输入与概念的亲和力，下投影编码这些概念的值，根据亲和力分数（激活）加权取它们的线性组合 [^10] [^11]。我们在 WikiText-2 验证集上测量 2B ><former 与恒定宽度 Transformer 的 MLP 激活密度，该验证集包含 252,986 个 token [^27]。因为 SwiGLU 是连续的，如果维度的激活幅度大于某个阈值，我们就认为该维度是活跃的。在图 7 中，我们显示 ><former 在跨阈值的 MLP 内强制执行更密集的激活。

密集激活不一定理想，因此我们还检查每个 MLP 激活维度的边际利用率：一个维度在跨 token 时被激活的频率（阈值设为 0.1）。在机制可解释性文献中，低边际利用率和"死维度（Dead Dimensions）"是容量利用不足的强指标 [^2] [^9]。图 5 显示了跨层的这一数量。虽然两个模型都没有完美均匀的分布，但 ><former 始终在激活维度之间实现显著更好的负载平衡。在附录 C 中，我们还显示，如果我们额外考虑激活幅度，也会出现类似的趋势，表明 ><former 更均匀地使用网络中间层的激活维度。

图 6：2B ><former 与 2B 恒定宽度 Transformer 中 MLP 激活的密度，跨阈值。><former 更密集地激活 MLP 激活维度。

图 7：2B ><former 与 2B 恒定宽度 Transformer 中层输出的归一化矩阵熵（Normalized Matrix Entropy）（§4.2）。><former 在中间到最终层具有更高的矩阵熵，这对应于在这些层中更均匀地使用残差维度。

### 4.2 ><former 缓解中间层表示坍缩

现在我们从研究 MLP 激活转向每层之后的残差流。对深度恒定宽度语言模型的最近分析揭示了"压缩谷"的出现，即语言模型的中间层在表示能力上坍缩，其特征是表示熵的严重下降 [^37] [^6]。遵循 [^6]，我们跟踪所有层的残差流的归一化矩阵熵：

$$
\frac{1}{\log r}\left(-\sum_{j=1}^{r}p_{j}\log p_{j}\right),\quad p_{j}=\sigma_{j}^{2}/\|\mathbf{X}\|_{F}^{2}
$$

其中 $\sigma_{j}$ 是输入特征表示矩阵 $\mathbf{X}$ 的排序奇异值，秩为 $r$，同样使用 WikiText-2 验证集计算。与有效维度（Effective Dimension）指标密切相关 [^15] [^33]，更高的矩阵熵表示表示空间的使用更"均匀"。

我们在此分析中考虑每层的隐藏状态。对于 ><former，回顾 §2 中我们的解释，将其视为具有宽残差流，其中每一层只读取/写入维度的一个子集。因此，我们将这个宽残差流视为其有效隐藏状态。

在图 7 中，我们看到基线模型表现出严重的压缩谷：在中间层，其归一化熵下降到接近零，表明 token 表示已坍缩为高度退化的低秩子空间，尽管宽度很大。这与先前的发现一致 [^37] [^6]。相比之下，><former 重构了这种动态。虽然它在早期层主动降低其熵以压缩表示（预期宽度减少），但它避免了中间层坍缩。在整个瓶颈和最终层中，><former 保持更高的归一化熵，这可能表明物理上约束参数空间鼓励网络维持高熵流形。

### 4.3 通过 Logit 透镜观察预测动态

图 8：2B ><former 与恒定宽度基线的 Logit 透镜（Logit Lens）分析。左：><former 在网络的大部分中间部分为目标 token 分配更高的概率。中：><former 的解码 token 分布在中间层中变化更渐进。右：><former 在早期层中熵较低，但在最终层中下降不那么快。

为了理解残差流中的这些几何差异如何影响模型预测，我们使用 Logit 透镜将中间隐藏状态投影到词汇空间 [^30]。具体而言，在每一层，我们通过应用最终 RMS 归一化然后应用反嵌入矩阵来解码隐藏状态。如 §4.2 中所述，我们将 ><former 的有效宽残差流视为其隐藏状态，限制为反嵌入可见的残差维度。

对于每一层，我们测量目标 token 的对数概率、解码 token 分布的熵以及相邻 Logit 透镜分布之间的逐层 KL 散度。我们通过平均两个方向来对称化这个 KL，将其用作解码分布随深度变化速度的代理。

图 8 显示，><former 在网络的早期到中间的大部分中为目标 token 分配更高的概率，解码分布熵更低。同时，其解码 token 分布在层之间变化更渐进，如较低的逐层 KL 所反映的。在最终层中，随着概率质量集中在目标 token 上，分布再次快速变化。

| 扩展方法 | 损失 |
| --- | --- |
| 恒定宽度 | 3.138 |
| 前向传递（Carry-forward） | 3.099 |
| 零填充 | 3.124 |
| 投影 | 3.150 |

表 4：在 500M 规模上扩展额外维度的不同方法的性能比较。简单地从较低层前向传递特征表现最好。

### 4.4 消融实验

我们分析了扩展维度的替代方法。除了我们的默认方法（通过复制坐标通过残差流前向传递特征）之外，我们还考虑（1）用 0 填充和（2）训练投影层从前一层表示预测额外维度。<sup>10</sup>对于每种方法，我们还在多个超参数配置上进行扫描，并报告最佳损失。我们在 500M 规模上进行消融，表 4 显示复制特征表现最好。## 5 局限性

一个主要的限制是，我们的方法为高效训练增加了显著的复杂性。具体而言，为了实现高效训练，需要为许多不同的形状开发和优化内核（kernel），每种形状都有不同的延迟、内存占用和计算特征。固定残差构造也可能增加额外开销，因为围绕比基线宽度 $d$ 更宽的全局残差流进行切片、复制和零填充会引入额外的内核启动，尽管其中大部分可以通过内核融合来缓解。异构的逐层宽度还与标准的张量并行和流水线并行技术存在冲突。

然而，我们强调，这些是实现层面而非算法层面的局限性：可变宽度 Transformer 仍然是矩阵乘法密集型的，我们描述的差距反映的是当前基础设施已经针对均匀宽度模式进行了大量优化，而非架构的内在属性。我们预期，专门构建的内核将大幅缩小理论效率与实际效率之间的差距。

更广泛地说，虽然我们并不呼吁立即采用 ><former，但我们希望未来的架构研究能够利用设计中这一此前未被注意到的自由度。

## 6 相关工作

#### Transformer 中的非均匀宽度分配

多个 Transformer 变体在深度上非均匀地分配参数。DeLighT 使用块级缩放，使早期块更浅/更窄，后期块更深/更宽 [^24]。OpenELM 在仅解码器的语言模型中采用逐层缩放，通过改变各层的注意力和前馈维度 [^25]。最近的逐层缩放变体探索了框架式（framed）、反向（reverse）和皇冠式（crown）分配模式 [^1]。[^17] 通过重新分配 MLP 容量研究了前馈网络的逐层重要性，发现将 MLP 集中在中间层有益。我们的工作与这些方法不同之处在于，我们改变的是完整块的隐藏维度，而不仅仅是注意力头数量、MLP 扩展倍数或轻量级块内部组件。这需要解决可变宽度块与残差流的交互方式；我们的固定残差构造让不活跃的坐标可以绕过较窄的块。

#### 序列长度上的瓶颈

也有一些工作在序列长度上进行压缩。Funnel-Transformer 逐渐缩短隐藏状态序列，然后恢复用于预测的 token 级表示 [^5]。Hourglass Transformer 对激活进行下采样和上采样，以构建显式的层次化语言模型 [^28]。Perceiver 模型使用交叉注意力将高维输入提取到紧凑的潜在瓶颈中，然后再应用 Transformer 风格的处理 [^18]。这些方法主要在 token 数量或潜在槽（latent slot）上设置瓶颈。相比之下，我们的架构保留了 token 序列长度，而是在深度上引入隐藏宽度的瓶颈。

#### Transformer 之外的瓶颈设计

瓶颈架构在 Transformer 之外有着悠久的历史。U-Net 和堆叠沙漏网络使用编码器-解码器结构，反复降低和恢复空间分辨率，通常带有保留高分辨率信息的跳跃连接 [^32] [^29]。其他架构在通道维度上引入瓶颈：ResNet 使用瓶颈残差块来降低深度卷积网络的成本 [^14]，而 MobileNetV2 使用带有线性瓶颈的倒残差块来构建高效的视觉模型 [^35]。然而，Transformer 应用主要使用非瓶颈架构的通道维度。

#### 超连接（Hyper-Connections）

通过扩展残差流容量，><former 在概念上与超连接（Hyper-Connections, HC）[^44] [^42] [^7] 相关。然而，机制不同：HC 使用多个残差流之间的学习混合，而 ><former 在单个全局残差流中使用确定性切片和前向传递。在较窄的层中，不活跃的坐标绕过块并在宽度扩展时重新引入。因此，><former 提供了一种互补的方式来改变残差容量，而无需 [^42] 识别为大规模 HC 不稳定性来源的学习残差混合矩阵。

## 7 结论

在这项工作中，我们挑战了 Transformer 深度上均匀容量分配的标准假设，引入了 ><former，一种可变宽度架构。在从 200M 到 3B 参数（稠密和 MoE）的评估中，参数匹配的 ><former 优于均匀基线，同时在数学上和实证上都减少了 FLOPs 和 KV 缓存内存。此外，我们的分析表明，这种瓶颈设计可能充当结构正则化器，迫使网络更均匀地利用其表示空间。这些发现表明，非均匀宽度分配是一种高效且有前景的策略，可用于扩展未来的语言模型。

## 致谢

本研究部分得到了 MIT-IBM Watson AI Lab 以及美国国家科学基金会 CAREER Award No. 2441872 和 NSF grant No. CCF-21-12665 的支持。

## References

## 附录 A 参数匹配宽度计算

在这里，我们推导如何从瓶颈层 $\ell^{*}$ 和瓶颈维度 $d_{\ell^{*}}$ 实例化几何宽度调度，同时匹配恒定宽度基线的参数计数。推导在连续宽度空间中进行；整数舍入仅在确定参数匹配宽度后应用。

设 $L$ 为 Transformer 层数，$d$ 为恒定宽度基线的隐藏维度，$v$ 为词汇表大小。我们假设输入和输出嵌入保持基线宽度 $d$。残差流具有层相关的宽度 $d_{\ell}$，相邻层之间的调整大小是无参数的。我们施加对称的端点宽度，$d_{1}=d_{L}=\bar{d}$。

层宽度在瓶颈两侧遵循几何级数：

$$
d_{\ell}=\begin{cases}\alpha^{-}d_{\ell-1},&1<\ell\leq\ell^{*},\\
\alpha^{+}d_{\ell-1},&\ell^{*}<\ell\leq L,\end{cases}
$$

其中 $\alpha^{-}\leq 1$ 且 $\alpha^{+}\geq 1$。对称端点意味着 $(\alpha^{-})^{\ell^{*}-1}(\alpha^{+})^{L-\ell^{*}}=1$，这对任何候选 $\alpha^{-}\in(0,1]$ 约束了 $\alpha^{+}$：

$$
\alpha^{+}=(\alpha^{-})^{-\frac{\ell^{*}-1}{L-\ell^{*}}}.
$$

因此，形状严格由 $\alpha^{-}$ 决定。我们定义无量纲因子 $c_{\ell}(\alpha^{-})$ 使得 $d_{\ell}=\bar{d}\,c_{\ell}(\alpha^{-})$：

$$
c_{\ell}(\alpha^{-})=\begin{cases}(\alpha^{-})^{\ell-1},&1\leq\ell\leq\ell^{*},\\
(\alpha^{-})^{\ell^{*}-1}(\alpha^{+})^{\ell-\ell^{*}},&\ell^{*}<\ell\leq L.\end{cases}
$$

要匹配恒定宽度基线的参数计数，我们必须考虑主要的层参数和端点修正。对于带有 SwiGLU 的稠密 Transformer 块，每层参数计数与 $Kd_{\ell}^{2}$ 成比例，其中 $K=4+N_{m}E$（$E=4$ 是 MLP 扩展因子，$N_{m}=3$ 对于 SwiGLU 是 MLP 投影的数量）。忽略层归一化和偏置项，基线参数计数为 $P_{\mathrm{base}}=2vd+LKd^{2}$。

因为嵌入固定在宽度 $d$，如果我们的调度需要 $\bar{d}>d$（如 ><former 的情况），我们用 0 填充初始嵌入并截断最终的反嵌入。这导致第一个注意力层和最后一个 MLP 层中的未使用参数。具体来说，第一个注意力 QKV 映射和最后一个 MLP 输出映射分别包含 $3\bar{d}(\bar{d}-d)$ 和 $E\bar{d}(\bar{d}-d)$ 个未使用参数。因此，总端点修正为：

$$
W_{\mathrm{end}}(\bar{d})=\mathbf{1}\{\bar{d}>d\}\,(3+E)\bar{d}(\bar{d}-d).
$$

将我们的可变宽度模型的有效参数等同于基线，得到：

$$
K\bar{d}^{2}S_{2}(\alpha^{-})-W_{\mathrm{end}}(\bar{d})=LKd^{2},
$$

其中 $S_{2}(\alpha^{-})=\sum_{\ell=1}^{L}c_{\ell}(\alpha^{-})^{2}$。

代入 $W_{\mathrm{end}}(\bar{d})$，我们可以将其简化为：

$$
\Big[KS_{2}(\alpha^{-})-\mathbf{1}\{\bar{d}>d\}\,(3+E)\Big]\bar{d}^{2}+\Big[\mathbf{1}\{\bar{d}>d\}\,d(3+E)\Big]\bar{d}-LKd^{2}=0.
$$

由于系数依赖于分段指示器 $\mathbf{1}\{\bar{d}>d\}$，我们通过假设指示器的一个状态（0 或 1）来求解，应用标准二次公式找到正根，并选择与我们假设自洽的根。这将有效端点宽度表达为 $\alpha^{-}$ 的函数，我们将其记为 $\bar{d}_{\alpha^{-}}$。给定 $\alpha^{-}$ 的瓶颈宽度为：

$$
b(\alpha^{-})=\bar{d}_{\alpha^{-}}(\alpha^{-})^{\ell^{*}-1}.
$$

我们使用 $\alpha^{-}\in(0,1]$ 上的一维数值求解器来求解 $b(\alpha^{-})=d_{\ell^{*}}$，其中 $d_{\ell^{*}}$ 是期望的瓶颈维度。最后，连续宽度被舍入到注意力头维度 $Q$ 的最近倍数以确保兼容性：

$$
\widehat{d}_{\ell}=\mathrm{round}\!\left(\frac{d_{\ell}}{Q}\right)Q.
$$

## 附录 B 数据集统计

在本节中，我们报告表 3 中使用的数据集统计信息。

表 5：任务评估配置和指标。

| Task | Domain | Split / Instances | Metric |
| --- | --- | --- | --- |
| OpenBookQA | science QA | 500 | acc\_norm |
| PIQA | physical commonsense | 1,838 | acc\_norm |
| SciQ | science QA | 1,000 | acc\_norm |
| ARC-Easy | grade-school science QA | 2,376 | acc\_norm |
| ARC-Challenge | difficult science QA | 1,172 | acc\_norm |
| BoolQ | yes/no reading comprehension | 3,270 | acc |
| COPA | causal commonsense | 100 | acc |
| HellaSwag | commonsense completion | 10,042 | acc\_norm |
| WinoGrande | pronoun/coreference reasoning | 1,267 | acc |
| RACE | reading comprehension | 1,045 | acc |
| WikiText | language modeling | 62 documents | perplexity |
| LAMBADA OpenAI | long-context word prediction | 5,153 | acc, perplexity |

## 附录 C 额外结果

图 9：2B ><former 与 2B 恒定宽度 Transformer 的 MLP 激活参与率（PR；§4.1）。我们展示了原始 PR 和按层宽度归一化的 PR。><former 在中间层具有更高的 PR，对应于这些层中激活维度的使用更均匀。

§4.1 中的分析表明 ><former 实现了更好的激活密度，但它没有考虑激活幅度。大型语言模型经常出现严重的离群维度，使得其余活跃维度在计算上变得微不足道。为了评估这一点，我们计算 MLP 激活上的能量参与率（PR）[^22] [^4]。设 $a_{t,i}$ 表示 token $t$ 的维度 $i$ 的激活，$e_{i}=\sum_{t}a_{t,i}^{2}$ 表示维度 $i$ 在所有 token $t$ 上的总能量。有效利用的维度数由 $N_{\text{eff}}=(\sum_{i}e_{i})^{2}/\sum_{i}e_{i}^{2}$ 给出。直观地说，$N_{\text{eff}}$ 充当表示平等性的连续度量：如果单个离群维度占据所有数值能量，$N_{\text{eff}}$ 崩溃为 1，无论实际宽度 $d_{\ell}$ 是多少。相反，如果计算能量均匀分布在所有坐标上，$N_{\text{eff}}$ 达到其理论最大值 $d_{\ell}$。因此，计算宽度归一化分数（$N_{\text{eff}}/d_{\ell}$）提供了有效利用维度比例的度量。我们在图 9 中报告了绝对和归一化 PR。

结果揭示了 ><former 与恒定宽度 Transformer 之间的区别。对于基线，其宽度归一化的能量利用率在第 10 层左右崩溃到接近零（<5%）。相比之下，><former 通过在中间层保持约 1,000 个有效维度的绝对 PR，更均匀地分配能量利用，产生更丰富的表示流形。通过限制参数可用性，><former 中的瓶颈可能充当结构正则化器，鼓励网络将更密集的表示打包到可用容量中。

[^1]: Baroian, A. and Notebomer, K. Crown, frame, reverse: Layer-wise scaling variants for llm pre-training. *arXiv preprint arXiv:2509.06518*, 2025.

[^2]: Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N. L., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., and Olah, C. Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*, 2023. URL [https://transformer-circuits.pub/2023/monosemantic-features/index.html](https://transformer-circuits.pub/2023/monosemantic-features/index.html).

[^3]: Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., Schuh, P., Shi, K., Tsvyashchenko, S., Maynez, J., Rao, A., Barnes, P., Tay, Y., Shazeer, N., Prabhakaran, V., Reif, E., Du, N., Hutchinson, B., Pope, R., Bradbury, J., Austin, J., Isard, M., Gur-Ari, G., Yin, P., Duke, T., Levskaya, A., Ghemawat, S., Dev, S., Michalewski, H., Garcia, X., Misra, V., Robinson, K., Fedus, L., Zhou, D., Ippolito, D., Luan, D., Lim, H., Zoph, B., Spiridonov, A., Sepassi, R., Dohan, D., Agrawal, S., Omernick, M., Dai, A. M., Pillai, T. S., Pellat, M., Lewkowycz, A., Moreira, E., Child, R., Polozov, O., Lee, K., Zhou, Z., Wang, X., Saeta, B., Diaz, M., Firat, O., Catasta, M., Wei, J., Meier-Hellstern, K., Eck, D., Dean, J., Petrov, S., and Fiedel, N. PaLM: Scaling language modeling with pathways, 2022. URL [https://arxiv.org/abs/2204.02311](https://arxiv.org/abs/2204.02311).

[^4]: Clark, D. G., Marschall, O., van Meegen, A., and Litwin-Kumar, A. Connectivity structure and dynamics of nonlinear recurrent neural networks. *Phys. Rev. X*, 15:041019, Nov 2025. doi: 10.1103/2jt7-c8cq. URL [https://link.aps.org/doi/10.1103/2jt7-c8cq](https://link.aps.org/doi/10.1103/2jt7-c8cq).

[^5]: Dai, Z., Lai, G., Yang, Y., and Le, Q. V. Funnel-transformer: Filtering out sequential redundancy for efficient language processing. In *Advances in Neural Information Processing Systems*, volume 33, pp. 4271–4282. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.cc/paper/_files/paper/2020/file/2cd2915e69546904e4e5d4a2ac9e1652-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/2cd2915e69546904e4e5d4a2ac9e1652-Paper.pdf).

[^6]: de Llano, E. Q., Arroyo, A., Barbero, F., Dong, X., Bronstein, M. M., LeCun, Y., and Shwartz-Ziv, R. Attention sinks and compression valleys in LLMs are two sides of the same coin. In *The Fourteenth International Conference on Learning Representations*, 2026. URL [https://openreview.net/forum?id=c5TFhCJ6fs](https://openreview.net/forum?id=c5TFhCJ6fs).

[^7]: DeepSeek-AI. DeepSeek-V4: Towards highly efficient million-token context intelligence, 2026.

[^8]: Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac'h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. The language model evaluation harness, 07 2024. URL [https://zenodo.org/records/12608602](https://zenodo.org/records/12608602).

[^9]: Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. Scaling and evaluating sparse autoencoders. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https://openreview.net/forum?id=tcsZt9ZNKD](https://openreview.net/forum?id=tcsZt9ZNKD).

[^10]: Geva, M., Schuster, R., Berant, J., and Levy, O. Transformer feed-forward layers are key-value memories. In Moens, M.-F., Huang, X., Specia, L., and Yih, S. W.-t. (eds.), *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pp. 5484–5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.446. URL [https://aclanthology.org/2021.emnlp-main.446/](https://aclanthology.org/2021.emnlp-main.446/).

[^11]: Geva, M., Caciularu, A., Wang, K., and Goldberg, Y. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Goldberg, Y., Kozareva, Z., and Zhang, Y. (eds.), *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pp. 30–45, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.3. URL [https://aclanthology.org/2022.emnlp-main.3/](https://aclanthology.org/2022.emnlp-main.3/).

[^12]: Groeneveld, D., Beltagy, I., Walsh, E., Bhagia, A., Kinney, R., Tafjord, O., Jha, A., Ivison, H., Magnusson, I., Wang, Y., Arora, S., Atkinson, D., Authur, R., Chandu, K., Cohan, A., Dumas, J., Elazar, Y., Gu, Y., Hessel, J., Khot, T., Merrill, W., Morrison, J., Muennighoff, N., Naik, A., Nam, C., Peters, M., Pyatkin, V., Ravichander, A., Schwenk, D., Shah, S., Smith, W., Strubell, E., Subramani, N., Wortsman, M., Dasigi, P., Lambert, N., Richardson, K., Zettlemoyer, L., Dodge, J., Lo, K., Soldaini, L., Smith, N., and Hajishirzi, H. OLMo: Accelerating the science of language models. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 15789–15809, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.841. URL [https://aclanthology.org/2024.acl-long.841/](https://aclanthology.org/2024.acl-long.841/).

[^13]: Gromov, A., Tirumala, K., Shapourian, H., Glorioso, P., and Roberts, D. The unreasonable ineffectiveness of the deeper layers. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https://openreview.net/forum?id=ngmEcEer8a](https://openreview.net/forum?id=ngmEcEer8a).

[^14]: He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

[^15]: Hill, M. D. Diversity and evenness: A unifying notation and its consequences. *Ecology*, 54(2):427–432, 1973. doi: https://doi.org/10.2307/1934352. URL [https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1934352](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1934352).

[^16]: Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Vinyals, O., Rae, J. W., and Sifre, L. Training compute-optimal large language models. In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS '22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

[^17]: Ikeda, W., Yano, K., Takahashi, R., Lee, J., Shibata, K., and Suzuki, J. Layerwise importance analysis of feed-forward networks in transformer-based language models. *arXiv preprint arXiv:2508.17734*, 2025.

[^18]: Jaegle, A., Gimeno, F., Brock, A., Zisserman, A., Vinyals, O., and Carreira, J. Perceiver: General perception with iterative attention. In *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pp. 4651–4664. PMLR, 2021. URL [https://proceedings.mlr.press/v139/jaegle21a.html](https://proceedings.mlr.press/v139/jaegle21a.html).

[^19]: Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models, 2020. URL [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361).

[^20]: Levine, Y., Wies, N., Sharir, O., Bata, H., and Shashua, A. Limits to depth efficiencies of self-attention. *Advances in Neural Information Processing Systems*, 33:22640–22651, 2020.

[^21]: Li, J., Fang, A., Smyrnis, G., Ivgi, M., Jordan, M., Gadre, S. Y., Bansal, H., Guha, E. K., Keh, S., Arora, K., Garg, S., Xin, R., Muennighoff, N., Heckel, R., Mercat, J., Chen, M. F., Gururangan, S., Wortsman, M., Albalak, A., Bitton, Y., Nezhurina, M., Abbas, A. K. M., Hsieh, C.-Y., Ghosh, D., Gardner, J. P., Kilian, M., Zhang, H., Shao, R., Pratt, S. M., Sanyal, S., Ilharco, G., Daras, G., Marathe, K., Gokaslan, A., Zhang, J., Chandu, K., Nguyen, T., Vasiljevic, I., Kakade, S. M., Song, S., Sanghavi, S., Faghri, F., Oh, S., Zettlemoyer, L., Lo, K., El-Nouby, A., Pouransari, H., Toshev, A. T., Wang, S., Groeneveld, D., Soldaini, L., Koh, P. W., Jitsev, J., Kollar, T., Dimakis, A., Carmon, Y., Dave, A., Schmidt, L., and Shankar, V. Datacomp-LM: In search of the next generation of training sets for language models. In *The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2024. URL [https://openreview.net/forum?id=CNWdWn47IE](https://openreview.net/forum?id=CNWdWn47IE).

[^22]: Litwin-Kumar, A., Harris, K. D., Axel, R., Sompolinsky, H., and Abbott, L. Optimal degrees of synaptic connectivity. *Neuron*, 93(5):1153–1164.e7, 2017. ISSN 0896-6273. doi: https://doi.org/10.1016/j.neuron.2017.01.030. URL [https://www.sciencedirect.com/science/article/pii/S0896627317300545](https://www.sciencedirect.com/science/article/pii/S0896627317300545).

[^23]: McLeish, S., Kirchenbauer, J., Miller, D. Y., Singh, S., Bhatele, A., Goldblum, M., Panda, A., and Goldstein, T. Gemstones: A model suite for multi-faceted scaling laws. *arXiv preprint arXiv:2502.06857*, 2025.

[^24]: Mehta, S., Ghazvininejad, M., Iyer, S., Zettlemoyer, L., and Hajishirzi, H. Delight: Deep and light-weight transformer. *arXiv preprint arXiv:2008.00623*, 2020.

[^25]: Mehta, S., Sekhavat, M. H., Cao, Q., Horton, M., Jin, Y., Sun, C., Mirzadeh, I., Najibi, M., Belenko, D., Zatloukal, P., et al. OpenELM: An efficient language model family with open training and inference framework. *arXiv preprint arXiv:2404.14619*, 2024.

[^26]: Meng, K., Bau, D., Andonian, A. J., and Belinkov, Y. Locating and editing factual associations in GPT. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), *Advances in Neural Information Processing Systems*, 2022. URL [https://openreview.net/forum?id=-h6WAS6eE4](https://openreview.net/forum?id=-h6WAS6eE4).

[^27]: Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. In *International Conference on Learning Representations*, 2017. URL [https://openreview.net/forum?id=Byj72udxe](https://openreview.net/forum?id=Byj72udxe).

[^28]: Nawrot, P., Tworkowski, S., Tyrolski, M., Kaiser, L., Wu, Y., Szegedy, C., and Michalewski, H. Hierarchical transformers are more efficient language models. In Carpuat, M., de Marneffe, M.-C., and Meza Ruiz, I. V. (eds.), *Findings of the Association for Computational Linguistics: NAACL 2022*, pp. 1559–1571, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-naacl.117. URL [https://aclanthology.org/2022.findings-naacl.117/](https://aclanthology.org/2022.findings-naacl.117/).

[^29]: Newell, A., Yang, K., and Deng, J. Stacked hourglass networks for human pose estimation. In *European conference on computer vision*, pp. 483–499. Springer, 2016.

[^30]: nostalgebraist. Interpreting GPT: the logit lens. LessWrong, 2020. URL [https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).

[^31]: Petty, J., Steenkiste, S., Dasgupta, I., Sha, F., Garrette, D., and Linzen, T. The impact of depth on compositional generalization in transformer language models. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pp. 7239–7252, 2024.

[^32]: Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical image computing and computer-assisted intervention*, pp. 234–241. Springer, 2015.

[^33]: Roy, O. and Vetterli, M. The effective rank: A measure of effective dimensionality. In *2007 15th European Signal Processing Conference*, pp. 606–610, 2007.

[^34]: Sajjad, H., Dalvi, F., Durrani, N., and Nakov, P. On the effect of dropping layers of pre-trained transformer models. *Comput. Speech Lang.*, 77(C), January 2023. ISSN 0885-2308. doi: 10.1016/j.csl.2022.101429. URL [https://doi.org/10.1016/j.csl.2022.101429](https://doi.org/10.1016/j.csl.2022.101429).

[^35]: Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 4510–4520, 2018.

[^36]: Shazeer, N. GLU variants improve transformer, 2020. URL [https://arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202).

[^37]: Skean, O., Arefin, M. R., Zhao, D., Patel, N. N., Naghiyev, J., LeCun, Y., and Shwartz-Ziv, R. Layer by layer: Uncovering hidden representations in language models. In *Forty-second International Conference on Machine Learning*, 2025. URL [https://openreview.net/forum?id=WGXb7UdvTX](https://openreview.net/forum?id=WGXb7UdvTX).

[^38]: Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. *Neurocomput.*, 568(C), February 2024. ISSN 0925-2312. doi: 10.1016/j.neucom.2023.127063. URL [https://doi.org/10.1016/j.neucom.2023.127063](https://doi.org/10.1016/j.neucom.2023.127063).

[^39]: Tay, Y., Dehghani, M., Rao, J., Fedus, W., Abnar, S., Chung, H. W., Narang, S., Yogatama, D., Vaswani, A., and Metzler, D. Scale efficiently: Insights from pre-training and fine-tuning transformers. *arXiv preprint arXiv:2109.10686*, 2021.

[^40]: Team, K., Chen, G., Zhang, Y., Su, J., Xu, W., Pan, S., Wang, Y., Wang, Y., Chen, G., Yin, B., Chen, Y., Yan, J., Wei, M., Zhang, Y., Meng, F., Hong, C., Xie, X., Liu, S., Lu, E., Tai, Y., Chen, Y., Men, X., Guo, H., Charles, Y., Lu, H., Sui, L., Zhu, J., Zhou, Z., He, W., Huang, W., Xu, X., Wang, Y., Lai, G., Du, Y., Wu, Y., Yang, Z., and Zhou, X. Attention residuals, 2026. URL [https://arxiv.org/abs/2603.15031](https://arxiv.org/abs/2603.15031).

[^41]: Tenney, I., Das, D., and Pavlick, E. BERT rediscovers the classical NLP pipeline. In Korhonen, A., Traum, D., and Màrquez, L. (eds.), *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pp. 4593–4601, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1452. URL [https://aclanthology.org/P19-1452/](https://aclanthology.org/P19-1452/).

[^42]: Xie, Z., Wei, Y., Cao, H., Zhao, C., Deng, C., Li, J., Dai, D., Gao, H., Chang, J., Yu, K., Zhao, L., Zhou, S., Xu, Z., Zhang, Z., Zeng, W., Hu, S., Wang, Y., Yuan, J., Wang, L., and Liang, W. mhc: Manifold-constrained hyper-connections, 2026. URL [https://arxiv.org/abs/2512.24880](https://arxiv.org/abs/2512.24880).

[^43]: Yang, G., Yu, D., Zhu, C., and Hayou, S. Tensor programs VI: Feature learning in infinite depth neural networks. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?id=17pVDnpwwl](https://openreview.net/forum?id=17pVDnpwwl).

[^44]: Zhu, D., Huang, H., Huang, Z., Zeng, Y., Mao, Y., Wu, B., Min, Q., and Zhou, X. Hyper-connections. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https://openreview.net/forum?id=9FqARW7dwB](https://openreview.net/forum?id=9FqARW7dwB).
