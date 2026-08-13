---
title: "全带宽 Transformer"
sourceUrl: "https://arxiv.org/abs/2608.08888"
sourceTitle: "Full-bandwidth transformer"
sourceAuthors: "Xi Wang, Ziyang Cai, Zheng Zhan, Harry Dong, Ying Fan, Gustavo de Rosa, Tim Pearce, John Langford"
sourceOrganizations: "Johns Hopkins University, Princeton University, Microsoft"
sourcePublishedAt: "2026-08-09"
sourceArxivId: "2608.08888"
sourceCategory: "cs.AI"
sourceFigureCount: 1
translatedAt: "2026-08-13"
translator: "Claude (Opus)"
pipelineRunId: "batch-20260813-100857"
pipelineSource: "translate/batch-20260813-100857/works-ready/arxiv-2608-08888-translation.md"
---

# 全带宽 Transformer

Xi Wang    Ziyang Cai    Zheng Zhan    Harry Dong    Ying Fan   
Gustavo de Rosa    Tim Pearce    John Langford    \[0.5em\] Johns Hopkins University  Princeton University  Microsoft

###### 摘要

自回归 Transformer 沿两个轴进行计算：水平方向跨越生成的 token，垂直方向穿过模型深度。密集注意力为每个 token 提供了对历史的广泛水平访问，但解码步骤之间的垂直反馈通道仍然很窄：只有采样的 token 返回到栈底，而顶层隐藏状态被丢弃。我们引入了**全带宽 Transformer**（full-bandwidth transformer），通过**潜在反馈**（latent feedback）拓宽这一通道：在每个解码步骤，前一步的顶层隐藏状态通过门控线性单元与采样 token 嵌入融合，并作为下一步的输入反馈回去。潜在反馈让非语言化的计算能够以更新的深度预算重新进入栈底，同时保留了标准 Transformer 架构、KV 缓存和语言建模目标。为了在不失去并行教师强制的情况下训练全带宽 Transformer，我们使用调度式多遍目标，在预训练后期引入潜在反馈，并混合少量更深的反馈遍次以确保稳定性。我们将 1B 参数的全带宽 Transformer 训练至 400B tokens，发现潜在反馈改善了验证损失、5-shot 语言模型评估、数学和编码生成以及指令微调性能。在每 token 解码开销可忽略的情况下，全带宽 Transformer 达到或接近用大约 1.5 倍更多 token 训练的标准 Transformer 的性能，并能在相同或更好的准确率下产生更短的推理轨迹。

<sup>†</sup>

## 1 引言

扩展大型语言模型主要意味着增加模型参数和在更多 token 上训练 [^22]。然而，随着预训练规模的持续扩大，高质量独特数据的可用性成为越来越大的约束。这促使我们重新审视扩展轴本身：我们能否通过为每个 token 分配更多计算来从中提取更多有用的学习信号，而不仅仅依赖更多数据？循环、迭代和基于反馈的计算为追求这个方向提供了自然的途径，但额外的 FLOP 只有在训练期间转化为更丰富的表示或推理时转化为更有效的计算时才有意义。

自回归 Transformer 暴露了一个特别未充分利用的此类计算机会。它们已经包含一个反馈循环：在步骤 $t-1$ 采样的 token 成为步骤 $t$ 的输入（图 1，左）。这个循环使得思维链解码 [^33] 能够执行深度随生成 token 数量增长的计算 [^24]。但作为通信通道来衡量，这个循环极其狭窄：解码将模型的整个顶层状态——一个 $D$ 维向量——压缩为携带最多 $\log_{2}|V|$ 比特的单个符号。非语言化的计算并未被擦除——中间激活保留在 KV 缓存中并保持可访问——但它是**深度冻结的**（depth-frozen）：在第 $\ell$ 层产生的状态只能被 $\ell$ 层以上的层读取，因此它永远无法返回栈底进行进一步处理，而最深的状态——顶层的输出——从未被缓存。因此，语言化是信息重新进入底层并接受新计算的唯一通道，代价是被挤压通过单个 token。模型必须要么花费 token 来叙述其中间状态，要么在每个位置从头重新计算该状态。

在本工作中，我们提出**全带宽 Transformer**，将这一通道拓宽到其完整宽度。具体来说，我们引入潜在反馈解码，在解码期间通过门控线性单元将前一步的顶层隐藏状态与采样 token 的嵌入融合，其中状态位于值路径上，token 充当门控，并将结果作为下一步输入反馈回去（图 1 右，第 3.1 节）。我们将能够以这种方式解码的 Transformer 称为全带宽 Transformer，因为其步间反馈现在携带完整的隐藏状态而不是单薄的 token。采样的 token 被保留，因此模型仍然产生普通文本，并可以灵活地使用标准监督语言建模损失进行训练；改变的是反馈不再局限于 token 的身份。根据设计，这提供了标准解码所缺乏的两点：(i) 非语言化的状态——不确定性、部分结果、计划——可以以更新的深度预算重新进入栈底并跨步骤进一步处理，而不是在产生它的层级上冻结在缓存中；(ii) 每一层，包括最浅的层，都看到由**完整**栈处理过的历史，而不仅仅是其下方的层；至关重要的是，这几乎不改变架构且服务成本极小：融合每个生成的 token 增加两次矩阵乘法，注意力和 KV 缓存保持不变，预填充运行一次或可选择运行两次以获得更好的性能。

障碍在于训练。预训练的模型从未在其输入中看到过隐藏状态，因此潜在反馈不能简单地在推理时开启；而且它定义的递归在位置上是顺序的，因此直接在其上训练将放弃使 Transformer 高效训练的并行教师强制。我们通过**多遍**（multi-pass）机制解决这个问题（第 3.3 节）：每一遍将前一遍的隐藏状态向右移动一个位置，与 token 嵌入融合，并在所有位置上并行重新运行栈，因此顺序性在少数几遍中支付，而不是跨序列支付。两个要素使这在实践中可行。**渐进调度**（progressive schedule）将大部分训练花在普通的单遍目标上，使运行可以从标准预训练检查点开始，仅在后期引入额外的反馈遍次；**前缀混合**（prefix mixin）随机化序列内融合输入开始的位置，匹配推理时的提示然后生成结构。经验上，我们发现调度的组成以一种意外的方式很重要：仅用两个反馈遍次训练产生一个递归，一旦展开超过其训练深度就会**发散**，而混合少至 3% 的三遍批次将学习到的映射变成向不动点的**收缩**，在超过训练深度时保持稳定（图 3）。

经验上，全带宽 Transformer 将可忽略的额外推理计算转化为相当于显著更多训练数据的收益。利用预填充的多次前向传播，递归训练的模型在验证损失和多项选择准确率上都匹配在两倍 token 上训练的无递归基线（图 4）。在自由形式生成（图 5）——GSM8K、Math500、HumanEval、MBPP——上，潜在反馈在每个任务上都优于**相同**权重的标准解码，匹配 $2\times$ token 基线，并在某些任务上接近用多达 $5\times$ token 训练的基线；这些收益通过长上下文扩展和指令微调得以保留（表 1）。在基础模型上，潜在反馈通常在相同或更好的准确率下产生明显更短的推理轨迹（图 6 和 8）——这正是拓宽通道所预测的行为，计算乘坐隐藏状态而不是逐个 token 语言化。

![Refer to caption](works/imgs/arxiv-2608-08888/img-001-fig1.png)

图 1：标准解码与潜在反馈解码。**左**：在标准 Transformer 中，当前状态只能访问较低层的过去状态（蓝色）；更深的过去状态（白色）不可达，唯一的步间反馈是采样的 token 嵌入（绿色）。**右**：全带宽 Transformer 使用潜在反馈，通过维度保持的门控（⊗，公式 (4)）将前一步的顶层隐藏状态与采样 token 嵌入融合，并将其作为下一步输入反馈回去。这将完整的隐藏状态信息返回到栈底，使由所有层处理过的历史对后续计算可访问。

## 2 背景

给定大小为 $|V|$ 的词汇表和 $D$ 维残差流，一个仅解码器的 LLM 将 $T$ 个 token 的输入序列（嵌入为 $\{\bm{e}_{t}\}_{t=1}^{T}\in\mathbb{R}^{T\times D}$）通过 $L$ 个注意力–MLP 块进行映射。最终层的隐藏状态 $\{\bm{h}_{t}^{L}\}_{t=1}^{T}$ 由语言模型头 $W_{\text{head}}\in\mathbb{R}^{|V|\times D}$ 投影到下一个 token 的分布：

$$
\bm{h}_{t}^{L}=f_{\theta}(\bm{e}_{t};\,C),\qquad\bm{e}_{t}\leftarrow\mathrm{Decode}\!\left(\bm{h}_{t-1}^{L}\right),\quad C=\bm{e}_{0},\bm{e}_{1},\ldots,\bm{e}_{t-1}.
$$

#### KV 缓存

在 Transformer 的自回归解码期间，先前计算的键和值被缓存并重用，避免对前缀的重复计算。与 RNN 和状态空间模型将历史压缩为固定大小的循环状态不同，密集注意力 Transformer 保留所有过去 token 的显式表示，因此每个新的隐藏状态可以直接注意完整的缓存历史。

#### 模型水平轴与垂直轴的带宽

将水平轴（跨位置）与垂直轴（跨深度）分开是有用的，因为这两者以不同的速率传输信息。**水平方向**，密集注意力实际上是全带宽的：在生成 token $t$ 时，第 $\ell$ 层状态 $\bm{h}_{t}^{\ell}$ 可以读取每个较早位置的缓存表示。**垂直方向**，访问受到限制：$\bm{h}_{t}^{\ell}$ 不能读取任何更深的过去状态 $\bm{h}_{t^{\prime}}^{\ell^{\prime}}$，其中 $t^{\prime}<t$ 且 $\ell^{\prime}\geq\ell$（图 1，左）。形式上，在计算位置 $t$ 的第 $\ell$ 层时可达的状态是

$$
\mathcal{R}_{\text{std}}(t,\ell)=\big\{(t^{\prime},\ell^{\prime}):t^{\prime}<t,\;\ell^{\prime}<\ell\big\},\qquad\bigl\lvert\mathcal{R}_{\mathrm{std}}\bigr\rvert=\Theta(T\ell),
$$

因此新 token 的浅层仅看到历史的**部分处理**视图，即使这些相同位置的更深、更完全处理的状态已经被计算并位于缓存中。因此，过去的计算会持续存在但被**深度冻结**，即在第 $\ell$ 层产生的表示只能被 $\ell$ 层以上的层读取，永远无法路由回底层进行进一步处理。这是第 3.1 节拓宽的窄垂直通道。

重要的是，这种深度依赖约束也使 Transformer 能够在位置上并行训练：顺序计算仅需跨层，而不需跨 token。然而，在解码时，生成已经在 token 上是顺序的，因此这个约束没有带来任何好处——这为对过去隐藏状态的更丰富依赖打开了大门，我们接下来将开发这一点。

## 3 通过潜在反馈解码拓宽带宽

### 3.1 潜在反馈解码

全带宽 Transformer 的核心创新是潜在反馈解码，它将前一步的顶层隐藏状态反馈到输入中。在步骤 $t$，

$$
\bm{h}_{t}^{L}=f_{\theta}\!\left(\bm{e}_{t}\otimes\bm{h}_{t-1}^{L};\;C\right),\qquad\textrm{其中}~\bm{e}_{t}\leftarrow\mathrm{Decode}\!\left(\bm{W}^{\textrm{head}}\bm{h}_{t-1}^{L}\right),\;C=\bm{e}_{0},\bm{e}_{1}\otimes\bm{h}_{0}^{L},\ldots,\bm{e}_{t-1}\otimes\bm{h}_{t-2}^{L}
$$

其中 $f_{\theta}$ 是 $L$ 层 Transformer 栈，$\cdot\otimes\cdot$ 融合采样 token 的嵌入与前一个潜在状态，$C$ 是过去的上下文（所有较早位置的 KV 缓存）。标准解码（公式 (1)）是其中只有采样 token 在步骤之间传递的特例。

融合 $\otimes$ 是一个门控线性单元：

$$
\bm{e}_{t}\otimes\bm{h}_{t-1}=\bm{W}^{U}\bm{h}_{t-1}\odot\sigma(\bm{W}^{G}\bm{e}_{t}),
$$

其中 $\bm{W}^{U},\bm{W}^{G}\in\mathbb{R}^{D\times D}$。这种不对称是故意的：隐藏状态占据值路径，而 token 嵌入仅作为乘法门控进入。像 $\bm{e}_{t}+\bm{W}\bm{h}_{t-1}$ 这样的对称融合会留下一条捷径：模型可以抑制状态路径，恢复普通 token 输入，并达到普通预训练损失，从而使宽通道未被使用。当训练从加法路径可以复现的低损失的标准检查点开始时，这条捷径特别诱人。公式 (4) 关闭了它，因为丢弃 $\bm{h}_{t-1}$ 就丢弃了输入本身，而 token 的身份仅在它施加在状态上的 $D$ 维门控模式中存活。因此读取状态是强制性的而不是可选的。

#### 潜在反馈服务成本为零

增加的推理成本与上下文长度和模型深度无关，每个 token 低于 1%。状态 $\bm{h}_{t-1}^{L}$ 在标准解码期间已经计算，因此唯一的额外工作是融合：两次 $D\times D$ 矩阵乘法，相对于通过 $L$ 个块的前向传播可以忽略不计。因为融合保持输入维度 $D$，架构、KV 缓存布局和服务栈保持不变，解码循环仅改变两行（图 2，右）。该方案也与 vLLM 兼容：我们将顶层状态存储在专用缓冲区中，采用多 token 预测实现使用的机制（附录 D）。

### 3.2 潜在反馈解码与标准 CoT

标准 CoT 通过单一反馈通道执行串行计算：每个 token 被附加到上下文并成为下一个输入。状态是 token 序列，

$$
s_{t+1}=s_{t}\|a_{t},\qquad a_{t}\sim\pi_{\theta}(\cdot\mid s_{t})\in\mathcal{V},\qquad s_{t}=x_{1:t},
$$

因此在步骤之间传递的唯一东西是离散动作序列。底层问题解决状态原则上可能是过去动作的确定性函数，但从 token 历史中恢复它本身就是一个状态跟踪问题，而固定深度的 Transformer 每次前向传播只有有界的串行计算。CoT 通过将中间状态外部化为语言来规避这一点：模型写出部分结果、子目标和簿记，然后在写出的轨迹上调节未来计算。

设 $\bm{u}_{i}=\bm{e}(a_{i-1})\otimes\bm{z}_{i-1}$ 为位置 $i$ 的融合输入（其中 $\bm{u}_{1}=\bm{e}_{0}$），因此被注意的上下文是 $C_{t}=\bm{u}_{1:t-1}$。状态是 $s_{t}=(a_{1:t},\,\bm{z}_{t})$：token 轨迹和最近的潜在状态。潜在反馈解码的一步是

$$
a_{t}\sim\pi_{\theta}(\cdot\mid s_{t})\in\mathcal{V},\qquad\bm{z}_{t+1}=f_{\theta}\!\left(\bm{e}(a_{t})\otimes\bm{z}_{t};\;\bm{u}_{1:t}\right),\qquad a_{1:t+1}=a_{1:t}\|a_{t},
$$

其中 $\cdot\otimes\cdot$ 是公式 (4) 的门控，$f_{\theta}$ 是完整栈。过去的潜在状态 $\bm{z}_{1:t-1}$ 不显式携带：每个已经折叠到 $\bm{u}_{1:t}$ 中，因此折叠到 KV 缓存中，所以只有 $\bm{z}_{t}$（缓存从不存储）作为递归变量传播。

#### 潜在反馈改善计算可访问性

由于 $z_{t+1}$ 是 $x_{1:t+1}$ 的确定性函数，它不携带上下文尚未确定的信息；收益是计算上的，而不是信息上的。具体来说，重新注入解除了公式 (2) 的深度限制（其可达集要求 $\ell^{\prime}<\ell$），使得每一层（包括最低层）都读取完整历史，

$$
\mathcal{R}_{\mathrm{lf}}(t,\ell)\;=\;\bigl\{\,(t^{\prime},\ell^{\prime})\;:\;t^{\prime}<t,\;0\leq\ell^{\prime}\leq L\,\bigr\},\qquad\bigl\lvert\mathcal{R}_{\mathrm{lf}}\bigr\rvert=\Theta(TL),
$$

如图 1（右）所示。在标准 CoT 中，每个新 token 仅访问上下文的部分处理视图。改进的可访问性也在第 4.4 节中得到经验验证。

#### 潜在反馈添加草稿空间

潜在反馈还提供了隐式草稿本，减轻了语言化中间状态的压力。状态维护从仅沿序列轴移动到也沿深度轴：中间结果可以通过 $z$ 沿栈更新，而不仅仅通过扩展 token 序列。这预测在推理任务上更短的展开，第 4.3 节证实了这一点。

#### 潜在反馈不提供什么

我们提供两个重要的澄清：

- **无可变寄存器**。RNN 和状态空间模型在每一步覆写压缩状态。潜在反馈在形式上是递归的，但过去的状态持续存在于 KV 缓存中而不是被覆写，因此每个较早的状态对当前 token 保持直接可读。
- **解码时无增加的渐近深度**。潜在反馈不改变解码的串行深度：无论有无它，每一步都有深度 $\mathcal{O}(L)$ 的图，因此 $T$ 个 token 花费 $\mathcal{O}(TL)$。改变的是路径的**带宽**，语言通道和连续通道现在并行演化。注意全带宽 Transformer 可以通过多遍预填充在预填充时进一步增加深度，我们将在下一节介绍。

### 3.3 潜在反馈解码的并行训练

列表 1：训练：$k$ 遍的一步。

[⬇](data:text/plain;base64,ZGVmIGdsdV9jcm9zcyhoLCBlKTogICAgICAjIFtULERdLFtULERdLT5bVCxEXQogICAgcmV0dXJuIChoIEAgV191KSAqIHNpZ21vaWQoZSBAIFdfZykKCmUgPSBlbWJlZCh0b2tlbnMpICAgICAgICAgIyBbVCwgRF0KaCA9IG1vZGVsKGUpICAgICAgICAgICAgICAjIHBhc3MgMSAoc3RhbmRhcmQpCmxvc3MgPSBudHBfbG9zcyhoKQpmb3IgXyBpbiByYW5nZShrIC0gMSk6ICAgICMgcGFyYWxsZWwgaW4gVAogICAgeCA9IGdsdV9jcm9zcyhzaGlmdF9yaWdodChoKSwgZSkKICAgIHggPSBwcmVmaXhfbWl4aW4oeCwgZSkgIyByYW5kb20gcGxhaW4gcHJlZml4CiAgICBoID0gbW9kZWwoeCkKICAgIGxvc3MgKz0gbnRwX2xvc3MoaCk=)

def glu\_cross(h, e): # \[T,D\],\[T,D\]->\[T,D\]

return (h @ W\_u) \* sigmoid(e @ W\_g)

e = embed(tokens) # \[T, D\]

h = model(e) # pass 1 (standard)

loss = ntp\_loss(h)

for \_ in range(k - 1): # parallel in T

x = glu\_cross(shift\_right(h), e)

x = prefix\_mixin(x, e) # random plain prefix

h = model(x)

loss += ntp\_loss(h)

列表 2：推理（Soft）；取消注释第 2 行得到 Fused；第 7 行显示 Standard 解码输入。

[⬇](data:text/plain;base64,aCA9IG1vZGVsKGVtYmVkKHByb21wdCkpICMgcHJlZmlsbCwgaDogW1QsIERdCiNoID0gbW9kZWwoZ2x1X2Nyb3NzKHNoaWZ0X3JpZ2h0KGgpLCBlbWJlZChwcm9tcHQpKSkKdG9rID0gc2FtcGxlKGxtX2hlYWQoaFstMV0pKQpoX3ByZXYgPSBoWy0xXQp3aGlsZSBub3QgZG9uZTogICAgICAgICAgICAgIyBkZWNvZGUKICAgIHggPSBnbHVfY3Jvc3MoaF9wcmV2LCBlbWJlZCh0b2spKQogICAgIyBzdGFuZGFyZCBkZWNvZGluZzogeCA9IGVtYmVkKHRvaykKICAgIGhfcHJldiA9IG1vZGVsX3N0ZXAoeCwga3ZfY2FjaGUpCiAgICB0b2sgPSBzYW1wbGUobG1faGVhZChoX3ByZXYpKQ==)

h = model(embed(prompt)) # prefill, h: \[T, D\]

#h = model(glu\_cross(shift\_right(h), embed(prompt)))

tok = sample(lm\_head(h\[-1\]))

h\_prev = h\[-1\]

while not done: # decode

x = glu\_cross(h\_prev, embed(tok))

\# standard decoding: x = embed(tok)

h\_prev = model\_step(x, kv\_cache)

tok = sample(lm\_head(h\_prev))

图 2：伪代码中的潜在反馈。训练（左）在 $k$ 遍中支付顺序性，每遍在位置上并行。推理（右）与标准解码的区别在于单行（第 6 行与注释的第 7 行）：输入是融合状态而不是单独的 token 嵌入，重用先前用于解码的状态。

图 3：少量三遍批次稳定长期潜在反馈。我们通过重复应用融合预填充遍次来测试学习到的反馈映射是否能外推超过训练期间看到的遍数。仅用单遍和双遍批次训练的模型在其训练视野之外失败：验证损失增加，隐藏状态更新大小振荡。添加少量三遍批次使迭代稳定：（左）验证损失在许多反馈步骤中保持平稳，（右）隐藏状态变化 $\|\bm{h}^{(k)}-\bm{h}^{(k-1)}\|$ 衰减到小平台。此诊断使用重复反馈遍次作为潜在反馈解码期间遇到的长期自组合的代理。

在解码时，潜在反馈在生成的位置上展开。设 $\bm{u}_{t}$ 为在位置 $t$ 实际馈送到 Transformer 栈的输入。第一个位置接收普通 token 嵌入，而每个后续位置接收当前 token 嵌入与前一个顶层状态的融合：

$$
\displaystyle\bm{u}_{1}
$$

$$
\displaystyle=\bm{e}_{1},
$$
$$
\displaystyle\bm{h}_{1}
$$

$$
\displaystyle=f_{\theta}(\bm{u}_{1};C_{1}),
$$
$$
\displaystyle\bm{u}_{t}
$$

$$
\displaystyle=\bm{e}_{t}\otimes\bm{h}_{t-1},
$$
$$
\displaystyle\bm{h}_{t}
$$

$$
\displaystyle=f_{\theta}(\bm{u}_{t};C_{t}),\qquad t\geq 2.
$$

这里 $\cdot\otimes\cdot$ 是公式 (4) 的门控融合，$C_{t}$ 是前面输入 $\bm{u}_{1:t-1}$ 上的 KV 缓存。因此栈看到的输入序列是

$$
\bm{e}_{1},\ \bm{e}_{2}\otimes\bm{h}_{1},\ \bm{e}_{3}\otimes\bm{h}_{2},\ \bm{e}_{4}\otimes\bm{h}_{3},\ldots
$$

而不是单独的嵌入。由于标准下一个 token 预测模型仅在此位置上的普通 token 嵌入上训练，全带宽 Transformer 也必须在这些潜在反馈输入上训练。

公式 (8) 的精确递归在 $t$ 上是顺序的：位置 $t$ 的输入依赖于位置 $t-1$ 的完整前向传播，因此直接在其上训练将牺牲使 Transformer 高效预训练的并行教师强制。我们改为采用多前向遍次近似。对于序列中的每个位置，我们多次计算顶层状态，写 $\bm{h}_{t}^{(k)}$ 表示第 $k$ 遍位置 $t$ 的状态（本节省略层上标 $L$）：

$$
\displaystyle\bm{h}_{t}^{(1)}
$$

$$
\displaystyle=f_{\theta}(\bm{e}_{t};\,C^{(1)}),
$$
$$
\displaystyle C^{(1)}
$$

$$
\displaystyle=\bm{e}_{1},\ldots,\bm{e}_{t-1},
$$
$$
\displaystyle\bm{h}_{t}^{(2)}
$$

$$
\displaystyle=f_{\theta}\!\big(\bm{e}_{t}\otimes\bm{h}_{t-1}^{(1)};\,C^{(2)}\big),
$$
$$
\displaystyle C^{(2)}
$$

$$
\displaystyle=\bm{e}_{1},\,\bm{e}_{2}\otimes\bm{h}_{1}^{(1)},\,\ldots,\,\bm{e}_{t-1}\otimes\bm{h}_{t-2}^{(1)},
$$
$$
\displaystyle\qquad\ldots
$$

$$
\displaystyle\bm{h}_{t}^{(k)}
$$

$$
\displaystyle=f_{\theta}\!\big(\bm{e}_{t}\otimes\bm{h}_{t-1}^{(k-1)};\,C^{(k)}\big),
$$
$$
\displaystyle C^{(k)}
$$

$$
\displaystyle=\bm{e}_{1},\,\bm{e}_{2}\otimes\bm{h}_{1}^{(k-1)},\,\ldots,\,\bm{e}_{t-1}\otimes\bm{h}_{t-2}^{(k-1)}.
$$

第一遍是普通的无反馈前向传播（$\bm{h}_{t}^{(1)}\equiv\bm{h}_{t}$）；每个后续遍次将前一遍的状态向右移动一个位置，与 token 嵌入融合，并在所有位置上并行重新运行完整栈，因为它需要的每个状态都在前一遍中完成。

然后我们将标准教师强制的下一个 token 预测损失 <sup>1</sup> 应用于每一遍的输出。保留第一遍损失保留了模型的无反馈操作模式，这是在推理时处理提示的方式。我们不分离梯度，因此来自后续遍次的损失反向传播到较早遍次的潜在状态，充当辅助目标；这确实增加了内存占用。整体目标是

$$
\mathcal{L}^{K}(\theta)=\underbrace{\sum_{t=1}^{T}-\log p_{\theta}\!\left(x_{t+1}\mid\bm{e}_{1:t}\right)}_{\text{标准 NTP 目标}}\;+\;\lambda\,\frac{1}{K-1}\sum_{k=2}^{K}\sum_{t=1}^{T}-\log p_{\theta}\!\left(x_{t+1}\mid\bm{e}_{1:t}^{(k)}\right),
$$

其中 $\bm{e}_{1:t}^{(k)}=\bm{e}_{1},\,\bm{e}_{2}\otimes\bm{h}_{1}^{(k-1)},\,\ldots,\,\bm{e}_{t}\otimes\bm{h}_{t-1}^{(k-1)}$ 是公式 (10)–(11) 的第 $k$ 遍融合输入。在所有实验中，我们设置 $\lambda=1$ 而不进行任何调整。

伪代码如图 2 左所示。我们将此训练方案称为**时间并行性**（temporal parallelism），遵循在训练期间并行化循环计算的常见策略 [^37] [^4] [^21]。每一遍都是潜在反馈递归的 Jacobi 式更新：来自前一遍的隐藏状态向右移动一个位置，与 token 嵌入融合，并用于并行更新所有位置。因此，每个额外的遍次将潜在反馈推进一个 token。在 $k$ 遍之后，来自位置 $t$ 的顶层状态可以影响到位置 $t+k-1$ 的输入，因此 $k$ 遍在 $k-1$ 个 token 步的视野上训练反馈转换。因此训练在遍次之间而不是位置之间支付顺序性，将长度为 $T$ 的循环展开减少到 $k$ 次并行 Transformer 评估，大约是标准教师强制计算的 $k\times$。然而学习到的局部转换与解码期间使用的相同，其中潜在反馈在每个生成的 token 上因果应用一次。

#### 反馈遍次调度

在解码时，反馈循环无限展开，因此训练的映射必须在远多于任何训练预算可以模拟的自组合下保持稳定；然而在整个训练期间运行许多遍次的成本过于昂贵，因为每一遍将运行成本乘以倍数。因此，调度前向遍次的数量——多少遍以及何时——对于使潜在反馈训练实用至关重要。

**多少遍次**。我们通过检查迭代反馈映射是否达到稳定不动点来选择遍次数：超过该深度，额外的遍次既不会大幅改变隐藏状态，也不会改善损失。这种稳定性比重复重新计算整个输入的架构（例如循环 Transformer）更容易获得，因为每个反馈遍次保持 token 嵌入固定，仅通过门控更新隐藏状态路径。实际上，这意味着目标不是在完整推理视野上训练，而是训练反馈映射直到它在进一步自组合下变得稳定。

**何时引入反馈遍次**。因为反馈遍次代价高昂，大部分训练使用标准单遍目标。我们在训练中期渐进地引入潜在反馈：首先使用双遍批次，然后使用少量更多遍次的批次。这使运行可以从普通预训练检查点开始，将大部分计算花在标准教师强制上，并仅在训练中期需要稳定反馈映射时支付额外的反馈遍次成本。

图 3 说明了调度的可行性。我们研究了在 200B tokens 上训练的 1B 模型。仅用单遍和双遍批次训练的模型（75% 单遍，25% 双遍；绿色）在训练深度表现良好但无法外推：超过该深度，验证损失急剧上升，隐藏状态变化 $\|\bm{h}^{(k)}-\bm{h}^{(k-1)}\|$ 振荡而不是衰减，表明迭代已离开训练状态分布。仅添加 3% 三遍批次（75% 单遍，22% 双遍，3% 三遍；蓝色）在质量上改变了行为：验证损失在 $30$ 个反馈步骤中保持平稳，隐藏状态变化衰减到小平台。这表明学习到的反馈映射表现得像向不动点的收缩，使远超训练中看到的反馈深度在我们的测试中保持稳定。相同的外推行为延续到推理：百 token 展开没有显示出崩溃的迹象（图 5，实心绿线），我们在 $k=1000$ 反馈遍次下观察到类似的稳定性（附录图 10）。

#### 前缀混合

多遍训练和推理之间仍存在分布不匹配。在解码时，序列是异构的：提示位置携带普通 token 嵌入（由单次预填充遍次处理），而生成的位置携带融合输入。相反，在公式 (10)–(11) 的遍次中，第一个位置之后的**每个**位置都被融合。因此仅在完全融合遍次上训练的模型在推理时遇到分布外边界，恰好在提示结束和生成开始的地方。为了缩小这一差距，我们应用**前缀混合**：在第一遍之后的每一遍中，我们采样随机前缀长度 $p$，并将位置 $t\leq p$ 恢复为普通嵌入，仅融合后缀。因此训练覆盖在任意点从普通输入切换到融合输入的序列，即单预填充推理的结构。或者，提示本身可以通过第二个融合预填充遍次运行，使所有位置匹配融合分布；混合消除了这种需要，但我们支持两者，对应于摘要中声明的"相同或加倍的预填充"开销。

#### 长反馈视野的稳定性配方

在推理时，潜在反馈可能应用数百或数千个生成的 token，远超训练期间使用的几个反馈遍次。因此，我们使用几种轻量级稳定化技术来保持反馈映射在长自组合下表现良好。

- **稳态隐藏状态规模**。我们在重复应用反馈时保持携带状态 $\bm{h}_{t}^{L}$ 的幅度有界。为了防止顶层状态范数随深度增长，我们使用深度缩放 [^34] [^30]，使得 $\left\lVert\bm{h}_{t}^{L}\right\rVert\sim\mathcal{O}(1)$ 而不是 $\mathcal{O}(L)$，这可能发生在标准前归一化模型中。我们还在将融合输入 $\bm{e}_{t}\otimes\bm{h}_{t-1}^{L}$ 馈送到模型之前对其应用 RMSNorm。
- **通过权重绑定共享输入基**。模型处理两种类型的输入：标准预填充期间的普通 token 嵌入，以及潜在反馈解码期间的融合隐藏状态/token 输入。因此，我们通过绑定嵌入层和读出层的权重来鼓励嵌入空间和顶层隐藏状态空间保持在兼容基中，减少融合权重学习两个输入分布之间大的校正旋转的负担。
- **噪声正则化**。在训练期间，我们在融合之前向携带的隐藏状态添加小抖动噪声，
	$$
	\bm{h}_{t}^{L}=f_{\theta}\!\left(\bm{e}_{t}\otimes\bm{h}_{t-1}^{L}+\epsilon;C\right),\qquad\epsilon\sim\mathrm{Uniform}[-\sigma,\sigma]^{D}.
	$$
	这使反馈映射暴露于每个训练状态周围的局部邻域，使其对可能在长反馈视野上累积的小偏差不那么敏感。

采用这些技巧的完整训练伪代码呈现在附录图 9 中。

### 3.4 潜在反馈训练改善预训练数据效率

除了在解码时启用潜在反馈之外，反馈遍次还充当隐藏状态上的辅助训练信号。在标准下一个 token 预测损失中，顶层状态 $\bm{h}_{t}^{L}$ 仅通过下一个 token 的预测进行监督。然而，在后续反馈遍次中，$\bm{h}_{t}^{L}$ 被移位、融合到后续位置的输入中，并可以通过因果注意力影响多个未来位置的损失。因此，来自后续预测的梯度反向传播到较早的隐藏状态，鼓励它们可重用作输入而不仅仅在输出层具有预测性。

经验上，即使在解码时不使用潜在反馈，这也改善了预训练数据效率。当使用标准解码进行评估时，用潜在反馈目标训练的模型相对于仅用普通下一个 token 目标训练的可比模型，在 LM Eval 和自由形式生成任务上有所改进。因此，我们可以将潜在反馈训练视为在相同 token 流上花费额外训练时计算的一种方式，在不改变服务时解码管道的情况下改善表示。

潜在反馈训练还启用了一种简单的预填充时测试时扩展形式。在评估时，我们可以使用公式 (10)–(11) 在提示上应用 $k$ 个额外的融合遍次。这些遍次在生成开始之前精炼提示状态，以 $k$ 次额外并行预填充前向传播的成本改善困惑度和下游准确率。见第 4.1 节。

## 4 实验

为了评估全带宽 Transformer，我们使用第 3.3 节的潜在反馈训练配方预训练 1B 参数模型（附录 A）。我们对矩阵参数使用 NorMuon [^25]，学习率 $1\times 10^{-2}$，权重衰减 $0.01$；对所有其他参数使用 Adam，学习率 $5\times 10^{-4}$，无权重衰减。所有运行使用 WSD 学习率调度 [^18] [^20]，200 步预热和 25% 冷却阶段衰减到零。在冷却期间，我们添加系数为 $1\times 10^{-5}$ 的 z-loss [^6]，并按照 AdamC [^10] 随学习率一起衰减权重衰减，这有助于防止权重和梯度范数变得不稳定。对于所有实验，我们在训练期间使用 $\sigma=0.02$ 的抖动噪声（公式 (13)）。

模型在与 Phi-4 [^1] 相同的数据混合上训练，上下文长度为 8192。除非另有说明，我们使用 300K tokens 的全局批量大小；1T token 无反馈基线使用 1.2M tokens 的更大全局批量大小。对于潜在反馈运行，我们报告训练 token 数和 **token 等效计算**，定义为训练 token 乘以每批次的平均前向遍次数。根据这种计算方式，双遍批次的成本是标准教师强制的 $2\times$，三遍批次的成本是 $3\times$。

| 运行 | 反馈遍次混合 | Tokens | Token 等效计算 |
| --- | --- | --- | --- |
| 10B | 100% 三遍 | 10B | 40B |
| 100B | 75% 单遍，25% 三遍 | 100B | 150B |
| 200B | 75% 单遍，22% 双遍，3% 三遍 | 200B | 256B |
| 400B | 75% 单遍，22% 双遍，3% 三遍 | 400B | 512B |

### 4.1 融合预填充改善非生成性能

图 4：预填充期间的反馈遍次改善非生成性能。使用额外融合遍次重新运行预填充（公式 (10)–(11)）在训练规模上改善了验证困惑度（左）和 10 个任务的 5-shot LM Eval 准确率（右）；大部分收益在第一个递归步骤就到达。误差棒表示平均值的一个标准误差，通过传播各个任务的标准误差获得（平方求和并除以任务数）。

图 4 绘制了验证损失和在 RTE、TruthfulQA-MC2、ARC-Easy、ARC-Challenge、BoolQ、PIQA、WinoGrande、OpenBookQA、COPA 和 MMLU 上的平均 5-shot LM Eval 准确率，作为预填充期间应用的反馈遍次数的函数。步骤 0 是没有潜在反馈的普通预填充，对应公式 (9)。每个额外步骤在来自公式 (10)–(11) 的融合输入上重新运行栈，通过门控将前一遍的顶层状态反馈回去。三个发现突出：

首先，**收益是前置的**。大部分改进在第一个融合预填充遍次之后出现，这是顶层隐藏状态在输入处可用的第一遍。进一步的遍次继续有帮助，但收益递减。这与潜在反馈充当提示的增加有效深度一致，一旦完整栈状态暴露给第 0 层，最大收益就到达。

其次，**潜在反馈训练在未使用时成本很小**。在步骤 0，模型作为没有反馈的普通 Transformer 进行评估，潜在反馈模型相对于标准基线仅放弃少量验证损失，同时已经改善了平均 LM Eval 准确率。因此，即使对于在推理时不应用融合预填充遍次的部署，训练配方也是有用的。

第三，**少量预填充时计算匹配大得多的标准基线**。使用两个反馈遍次，100B token 全带宽 Transformer 达到 200B token 标准基线，200B token 全带宽 Transformer 达到 400B token 标准基线。在这种情况下，融合预填充将适度的推理时计算转化为大约 $2\times$ 的预训练数据效率。

最后，我们在附录 B 的表 2 中将我们的模型与类似参数规模的其他模型在 0-shot LM Eval 性能上进行比较，发现我们的模型在类似或更多预算下训练的模型中表现相当或更好。这些结果表明全反馈 Transformer 改进了强基线。

### 4.2 潜在反馈解码改善解码性能

图 5：我们比较第 4.2 节开头定义的三种解码模式：Standard、Soft 和 Fused，在自由形式生成任务上；在数学任务上，Soft 通常给出最大收益，表明在生成过程中携带隐藏状态有助于推理。在编码任务上，Fused 通常最强，表明在生成之前精炼提示表示特别有用。编码结果报告每个问题 10 次展开的 Pass@3，温度从 $\{0.3,0.5,0.7\}$ 中为每种方法单独选择。

我们现在评估潜在反馈是否改善开放式生成。我们比较三种解码模式：

- **Standard**：单遍预填充；生成仅使用 token 嵌入。这将全带宽模型作为普通 Transformer 评估，并衡量当反馈通道在推理时未使用时潜在反馈训练的成本。
- **Soft**：单遍预填充；生成使用公式 (3) 中的潜在反馈。提示位置携带普通嵌入，而生成的位置携带融合输入，匹配第 3.3 节中前缀混合诱导的异构提示然后生成模式。每个 token 的唯一开销是两次 $D\times D$ 矩阵乘法。
- **Fused**：提示首先由额外的融合预填充遍次处理，如公式 (10)；然后生成像 Soft 中一样进行。这在解码开始之前给提示状态一轮潜在反馈精炼，代价是一次在提示 token 上并行的额外预填充遍次。

因此 Standard 和 Soft 具有相同的预填充成本，而 Fused 使预填充成本加倍，同时保持与 Soft 和实际上 Standard 相同的每 token 解码成本。

#### 评估设置

我们在 GSM8K [^7]、MATH-500 [^26]、HumanEval [^5] 和 MBPP [^3] 上进行评估。我们报告数学的 Pass@1 和编码的 Pass@3。对于编码，Pass@3 从每个问题 10 次展开中估计，温度在 $\{0.3,0.5,0.7\}$ 上为每种解码模式单独网格搜索。我们不使用 top-$k$ 或 top-$p$ 采样。

#### 潜在反馈解码改善基础模型

图 5 在两个递归训练规模（100B-400B tokens，实线）上评估基础模型的三种解码模式，与在 100B–1T tokens 上训练的无递归基线（虚线）进行比较。四个观察。首先，Soft 在两个规模的每个任务上都优于 Standard；收益仅来自解码，模型权重保持固定。其次，首选模式取决于任务：Soft 在数学上产生最大收益（在 Math500 上，200B 模型从 $0.27$ 改善到 $0.37$，甚至超过 1T 无递归基线），而 Fused 在编码上最强（HumanEval $0.31\to 0.34$；MBPP $0.38\to 0.40$ 在 200B），与编码奖励提示的更深表示和数学奖励通过生成携带的状态一致。第三，在潜在反馈下，200B 递归模型接近或超过用 $2$–$5\times$ token 训练的无递归基线（例如，在 GSM8K 和 HumanEval 上接近 1T 基线）。第四，Pass@3 与 Pass@1 一起改善，表明在隐藏状态上调节生成不会崩溃采样多样性或损害探索。

#### 改进通过指令微调延续

我们进一步对 200B 和 400B 模型应用从 8K 到 32K 的长上下文扩展（12B tokens）和指令微调（6B tokens）（图 4 和 5 中的绿色和紫色线），然后在没有少样本示例的情况下进行评估。因为这些阶段比预训练短得多，我们在整个过程中用**三个**前向遍次训练它们，而不是使用预训练反馈遍次调度。结果如表 1 所示。Soft 和 Fused 在所有四个任务上继续优于 Standard；例如，GSM8K 从 $64.5$ 改善到 $67.9$，HumanEval 从 $42.5$ 改善到 $45.9$。它们也在每个任务上优于匹配的 200B token 标准基线。在 MBPP 上，Fused 缩小了与 1T token 标准基线的大部分剩余差距（$41.2$ 对 $41.9$）。

<table><tbody><tr><td></td><td colspan="3">全带宽，200B</td><td colspan="3">全带宽，400B</td><td colspan="3">标准 Transformer</td></tr><tr><td>任务</td><td>Standard</td><td>Soft</td><td>Fused</td><td>Standard</td><td>Soft</td><td>Fused</td><td>200B</td><td>400B</td><td>1T</td></tr><tr><td>GSM8K (Pass@1)</td><td>64.52</td><td>67.93</td><td>67.55</td><td>67.90</td><td>71.00</td><td>71.80</td><td>62.93</td><td>68.39</td><td>70.13</td></tr><tr><td>MATH-500 (Pass@1)</td><td>43.80</td><td>45.60</td><td>45.60</td><td>46.00</td><td>45.40</td><td>48.40</td><td>42.40</td><td>46.40</td><td>47.40</td></tr><tr><td>HumanEval (Pass@3)</td><td>42.54</td><td>45.06</td><td>45.92</td><td>46.50</td><td>47.20</td><td>47.60</td><td>37.16</td><td>44.85</td><td>50.01</td></tr><tr><td>MBPP (Pass@3)</td><td>38.39</td><td>39.80</td><td>41.22</td><td>40.50</td><td>40.60</td><td>41.70</td><td>38.61</td><td>40.28</td><td>41.93</td></tr></tbody></table>

表 1：潜在反馈收益通过指令微调延续。我们在长上下文扩展和指令微调后评估全带宽 Transformer，不使用少样本示例。分数是百分比。对于数学任务，我们报告 Pass@1；对于编码任务，我们报告从每个问题 10 次展开估计的 Pass@3，为每个设置从 $\{0.3,0.5,0.7\}$ 中选择最佳温度。粗体表示每个全带宽训练规模内的最佳解码模式。

图 6：来自 200B 运行（图 5 中的绿线）在 Math500 上的推理长度和准确率。没有任何少样本示例或指令微调的基础模型生成更短的解决方案（用中位数而不是平均值衡量以防止异常值），同时给出更好的准确率，图 8 中提供了一个具体例子。

((a)) 状态检索。每个序列指定两个计数器之间的二元关系（"Completion"）或存储的绝对二元值（"Memory"），后跟不同数量（由线条颜色表示）的与标签无关的干扰 token。一个递归步骤使目标状态在第 $0$ 层跨输入长度几乎完美可解码，而标准预填充需要多层才能从前缀重建它。

((b)) 多寄存器最新写入跟踪。每个序列对八个二元寄存器中的每一个执行 2、4 或 8 次写入，然后查询一个寄存器的最新值。我们在每个残差深度探测其值。递归预填充相对于标准预填充改善了浅层标准可访问性；一个递归步骤的收益在更深层和具有更多覆写的输入中减少，其中完全递归表现最佳，显示了在整个序列中维护状态的好处。

图 7：全带宽 Transformer 将全局状态暴露给浅层。在三个合成任务上，我们在深度上线性探测最终输入 token 的残差流（0 表示输入）以预测输入的二元状态（详见附录 F）；我们比较了使用 token 嵌入作为输入的标准预填充与递归预填充，其中前一个 token 的顶层状态融合到当前 token 的输入中，类似于公式 (8) 但使用输入 token 而不是采样 token；$k$ 步递归在最后 $k$ 个 token 上应用此融合（代价为 $k+1$ 次前向传播），而完全递归在整个任务序列中应用它（代价为完全顺序预填充）。

### 4.3 潜在反馈实现更简洁的推理

在基础模型上，Soft 解码通常在相同或更好的准确率下产生明显更短的推理轨迹；图 8 显示了示例（其他示例显示在附录 G 中）。这是拓宽通道预测的行为：Standard 必须语言化的中间计算——逐个 token，每步 $\log_{2}|V|$ 比特——可以改为乘坐隐藏状态，因此需要更少的 token 来达到答案。值得注意的是，这种效果在指令微调后消失。我们将此归因于微调数据相对于潜在反馈解码是离策略的：目标轨迹由标准逐 token 推理产生（并模仿其冗长性），因此拟合它们重新施加完全语言化的风格，而不管状态能携带什么。在潜在反馈下的在策略后训练可能保留简洁性，我们将其留给未来工作。

### 4.4 全带宽 Transformer 在浅层残差中携带更丰富的信息

最后，为了直接验证增加的带宽，我们运行受控的状态跟踪实验，其中目标是固定的但中间上下文变化（完整构造在附录 F 中）。两个任务隔离了效果。**完成跟踪**询问完成的计数器在一系列无操作更新后是否达到了所需的计数器；**延迟记忆**要求模型在一系列与标签无关的草稿操作后恢复初始二元状态。两者都在共享的冒号处结束，标签完全由其之前的信息确定，因此在该冒号处的探测衡量每层已经重建了多少全局状态。

我们比较两种预填充模式。在**标准预填充**下，最终 token 作为其普通嵌入进入；在**单步递归预填充**下，该嵌入与前一个 token 的顶层状态融合（公式 (4)），正是潜在反馈在解码时提供给第 0 层输入的。然后我们在每个残差流深度拟合目标（完成/更多或零/一）的线性探测。

两种模式在栈底部差异很大。在标准预填充下，浅层残差只能读取层匹配的、部分处理的前缀（公式 (2) 的可达性约束），因此重建全局状态需要几层进一步计算；第 0 层探测接近随机。递归预填充改为在第 0 层输入处暴露完全处理的前缀摘要，第 0 层探测准确率上升到完成跟踪的 $99.6\%$ 和延迟记忆的 $100\%$。因此递归提供了一条高带宽捷径，将全局聚合信息传输到浅层计算中，这是全带宽视图预测的机制。

一个注意事项需要强调：改进的**可解码性**本身并不意味着改进的**输出**。目标在第 0 层线性可恢复表明信息存在，而不是模型使用它来决定下一个 token；使状态可用和因果利用它是不同的，只有下游任务结果（第 4 节）说明后者。

<svg id="S4.F8.pic1" height="58.08" overflow="visible" version="1.1" viewBox="0 0 650 58.08" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,58.08) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#666666;" fill="#666666" fill-opacity="1.0"><path style="stroke:none" d="M 0 5.91 L 0 52.17 C 0 55.44 2.64 58.08 5.91 58.08 L 644.09 58.08 C 647.36 58.08 650 55.44 650 52.17 L 650 5.91 C 650 2.64 647.36 0 644.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z"></path></g><g style="--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" fill-opacity="1.0"><path style="stroke:none" d="M 1.97 5.91 L 1.97 36.66 L 648.03 36.66 L 648.03 5.91 C 648.03 3.73 646.27 1.97 644.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 42.57)"><foreignObject style="--ltx-fo-width:38.13em;--ltx-fo-height:0.69em;--ltx-fo-depth:0em;font-size:10pt;" height="9.61" overflow="visible" transform="matrix(1 0 0 -1 0 9.61)" width="527.61"><span id="S4.F8.pic1.1" style="width:38.13em;"><span id="S4.F8.pic1.1.1"><span id="S4.F8.pic1.1.1.1" style="--ltx-fg-color:#FFFFFF;">问题</span></span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 16.2)"><foreignObject style="--ltx-fo-width:47.41em;--ltx-fo-height:0.63em;--ltx-fo-depth:0.18em;font-size:10pt;" height="11.07" overflow="visible" transform="matrix(1 0 0 -1 0 8.65)" width="656.02"><span id="S4.F8.pic1.2" style="width:47.41em;"><span id="S4.F8.pic1.2.1"><span id="S4.F8.pic1.2.1.1" style="font-size:90%;--ltx-fg-color:#000000;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\frac{137}{500}"><semantics><mfrac style="--ltx-fg-color:#000000;" mathcolor="#000000"><mn style="--ltx-fg-color:#000000;" mathcolor="#000000">137</mn> <mn style="--ltx-fg-color:#000000;" mathcolor="#000000">500</mn></mfrac> <annotation encoding="application/x-tex">\frac{137}{500}</annotation></semantics></math> 的十进制展开中小数点右边的最后一个非零数字是什么？</span></span></span></foreignObject></g></g></svg>

<svg id="S4.F8.pic2" height="210.98" overflow="visible" version="1.1" viewBox="0 0 650 210.98" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,210.98) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#BFBFBF;" fill="#BFBFBF" fill-opacity="1.0"><path style="stroke:none" d="M 0 5.91 L 0 205.07 C 0 208.33 2.64 210.98 5.91 210.98 L 644.09 210.98 C 647.36 210.98 650 208.33 650 205.07 L 650 5.91 C 650 2.64 647.36 0 644.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z"></path></g><g style="--ltx-fill-color:#F9F9F9;" fill="#F9F9F9" fill-opacity="1.0"><path style="stroke:none" d="M 1.97 5.91 L 1.97 169.5 L 648.03 169.5 L 648.03 5.91 C 648.03 3.73 646.27 1.97 644.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 178.86)"><foreignObject style="--ltx-fo-width:43.85em;--ltx-fo-height:1.89em;--ltx-fo-depth:0.25em;font-size:10pt;" height="29.67" overflow="visible" transform="matrix(1 0 0 -1 0 26.21)" width="606.76"><span id="S4.F8.pic2.1" style="width:43.85em;"><span id="S4.F8.pic2.1.1"><span id="S4.F8.pic2.1.1.1" style="--ltx-fg-color:#FFFFFF;">Standard 解码 <span id="S4.F8.pic2.1.1.1.1">(200B 模型)</span></span></span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 16.2)"><foreignObject style="--ltx-fo-width:47.41em;--ltx-fo-height:10.23em;--ltx-fo-depth:0.18em;font-size:10pt;" height="143.91" overflow="visible" transform="matrix(1 0 0 -1 0 141.49)" width="656.02"><span id="S4.F8.pic2.2" style="width:47.41em;"><span id="S4.F8.pic2.2.1"><span id="S4.F8.pic2.2.1.1" style="font-size:90%;--ltx-fg-color:#000000;">为了确定 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\frac{137}{500}"><semantics><mfrac style="--ltx-fg-color:#000000;" mathcolor="#000000"><mn style="--ltx-fg-color:#000000;" mathcolor="#000000">137</mn> <mn style="--ltx-fg-color:#000000;" mathcolor="#000000">500</mn></mfrac> <annotation encoding="application/x-tex">\frac{137}{500}</annotation></semantics></math> 的十进制展开中小数点右边的最后一个非零数字，我们从执行 137 除以 500 开始。</span></span> <span id="S4.F8.pic2.2.2"><span id="S4.F8.pic2.2.2.1" style="font-size:90%;--ltx-fg-color:#000000;">首先，我们设置除法：</span></span> <span id="S4.Ex3"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" data-latex="137\div 500"><semantics><mrow><mn style="--ltx-fg-color:#000000;" mathcolor="#000000" mathsize="0.900em">137</mn> <mo style="--ltx-fg-color:#000000;" mathcolor="#000000" mathsize="0.900em">÷</mo> <mn style="--ltx-fg-color:#000000;" mathcolor="#000000" mathsize="0.900em">500</mn></mrow> <annotation encoding="application/x-tex">137\div 500</annotation></semantics></math> </span><span id="S4.F8.pic2.2.3"><span id="S4.F8.pic2.2.3.1" style="font-size:90%;--ltx-fg-color:#000000;">由于 137 小于 500，十进制展开将以零开始。<span id="S4.F8.pic2.2.3.1.1" style="--ltx-fg-color:#0000FF;">[…省略 326 个 tokens…]</span></span></span> <span id="S4.F8.pic2.2.4"><span id="S4.F8.pic2.2.4.1" style="font-size:90%;--ltx-fg-color:#000000;">由于余数现在是 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="0"><semantics><mn style="--ltx-fg-color:#000000;" mathcolor="#000000">0</mn> <annotation encoding="application/x-tex">0</annotation></semantics></math>，<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\frac{137}{500}"><semantics><mfrac style="--ltx-fg-color:#000000;" mathcolor="#000000"><mn style="--ltx-fg-color:#000000;" mathcolor="#000000">137</mn> <mn style="--ltx-fg-color:#000000;" mathcolor="#000000">500</mn></mfrac> <annotation encoding="application/x-tex">\frac{137}{500}</annotation></semantics></math> 的十进制展开是：</span></span> <span id="S4.Ex4"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" data-latex="0.274"><semantics><mn style="--ltx-fg-color:#000000;" mathcolor="#000000" mathsize="0.900em">0.274</mn> <annotation encoding="application/x-tex">0.274</annotation></semantics></math> </span><span id="S4.F8.pic2.2.5"><span id="S4.F8.pic2.2.5.1" style="font-size:90%;--ltx-fg-color:#000000;">小数点右边的最后一个非零数字是 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\boxed{4}"><semantics><menclose notation="box"><mn style="--ltx-fg-color:#000000;" mathcolor="#000000">4</mn></menclose> <annotation encoding="application/x-tex">\boxed{4}</annotation></semantics></math>。</span></span></span></foreignObject></g></g></svg>

<svg id="S4.F8.pic3" height="127.96" overflow="visible" version="1.1" viewBox="0 0 650 127.96" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,127.96) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#9999FF;" fill="#9999FF" fill-opacity="1.0"><path style="stroke:none" d="M 0 5.91 L 0 122.05 C 0 125.31 2.64 127.96 5.91 127.96 L 644.09 127.96 C 647.36 127.96 650 125.31 650 122.05 L 650 5.91 C 650 2.64 647.36 0 644.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z"></path></g><g style="--ltx-fill-color:#F2F2FF;" fill="#F2F2FF" fill-opacity="1.0"><path style="stroke:none" d="M 1.97 5.91 L 1.97 86.47 L 648.03 86.47 L 648.03 5.91 C 648.03 3.73 646.27 1.97 644.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 95.84)"><foreignObject style="--ltx-fo-width:43.85em;--ltx-fo-height:1.89em;--ltx-fo-depth:0.25em;font-size:10pt;" height="29.67" overflow="visible" transform="matrix(1 0 0 -1 0 26.21)" width="606.76"><span id="S4.F8.pic3.1" style="width:43.85em;"><span id="S4.F8.pic3.1.1"><span id="S4.F8.pic3.1.1.1" style="--ltx-fg-color:#FFFFFF;">Soft 解码 <span id="S4.F8.pic3.1.1.1.1">(200B 模型)</span></span></span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 16.2)"><foreignObject style="--ltx-fo-width:47.41em;--ltx-fo-height:4.23em;--ltx-fo-depth:0.18em;font-size:10pt;" height="60.88" overflow="visible" transform="matrix(1 0 0 -1 0 58.46)" width="656.02"><span id="S4.F8.pic3.2" style="width:47.41em;"><span id="S4.F8.pic3.2.1"><span id="S4.F8.pic3.2.1.1" style="font-size:90%;--ltx-fg-color:#000000;">为了找到 <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\frac{137}{500}"><semantics><mfrac style="--ltx-fg-color:#000000;" mathcolor="#000000"><mn style="--ltx-fg-color:#000000;" mathcolor="#000000">137</mn> <mn style="--ltx-fg-color:#000000;" mathcolor="#000000">500</mn></mfrac> <annotation encoding="application/x-tex">\frac{137}{500}</annotation></semantics></math> 的十进制展开中小数点右边的最后一个非零数字，我们可以执行除法并观察十进制展开的模式。</span></span> <span id="S4.F8.pic3.2.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\frac{137}{500}=0.274"><semantics><mrow><mfrac style="--ltx-fg-color:#000000;" mathcolor="#000000"><mn style="--ltx-fg-color:#000000;" mathcolor="#000000" mathsize="0.900em">137</mn> <mn style="--ltx-fg-color:#000000;" mathcolor="#000000" mathsize="0.900em">500</mn></mfrac> <mo style="--ltx-fg-color:#000000;" mathcolor="#000000" mathsize="0.900em">=</mo> <mn style="--ltx-fg-color:#000000;" mathcolor="#000000" mathsize="0.900em">0.274</mn></mrow> <annotation encoding="application/x-tex">\frac{137}{500}=0.274</annotation></semantics></math></span> <span id="S4.F8.pic3.2.3"><span id="S4.F8.pic3.2.3.1" style="font-size:90%;--ltx-fg-color:#000000;"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\frac{137}{500}"><semantics><mfrac style="--ltx-fg-color:#000000;" mathcolor="#000000"><mn style="--ltx-fg-color:#000000;" mathcolor="#000000">137</mn> <mn style="--ltx-fg-color:#000000;" mathcolor="#000000">500</mn></mfrac> <annotation encoding="application/x-tex">\frac{137}{500}</annotation></semantics></math> 的十进制展开中小数点右边的最后一个非零数字是 4。</span></span></span></foreignObject></g></g></svg>

图 8：来自 200B 模型在 Standard 和 Soft 解码下的输出定性比较（贪婪解码）。两者都达到正确答案；Soft 解码大幅更简洁。截断的文本用 \[…\] 表示。这在指令微调版本上不再可观察到。

## 5 相关工作

#### 缓解解码时的深度瓶颈

全带宽 Transformer 背后的一个核心思想是引入与顺序解码过程重叠的额外计算。还有其他工作考虑类似的想法。**Feedback Transformer** [^12] 是沿这条线的开创性工作；在每个位置，它们生成每层表示的混合，并让未来位置的注意力注意聚合表示而不是像标准 Transformer 中的同层键值。然而，它们的训练在输入 token 上是顺序的，限制了其可扩展性，而我们的训练在所有位置上并行化。此外，我们的方法不涉及修改结构，仅修改输入。请注意，他们的消融也支持我们对反馈层的选择：仅从最顶层构建的记忆几乎匹配全层混合，而从第一层构建的记忆表现不比标准 Transformer 好。还有一些非常近期的工作探索类似方向。**$T^{2}$ MLR** [^4] 将最后位置的后期中间层表示与当前位置的早期中间层表示注入。**Latent Recurrent Transformer** [^21] 从前一位置的固定源层存储隐藏状态，并通过额外的键/值投影通过注意力和直接进入残差流将其注入到当前位置。方法论上，我们的方法在训练方法和动机上相似。我们的方法主要在重新注入点上不同；具体来说，我们的注入发生在模型"外部"，因此不引入架构更改，因为我们仅修改输入的构造。我们也引入最少量的额外参数。对于具有 $D$ 维残差的 $L$ 层 Transformer，我们仅引入两个线性投影（每个大小为 $D\times D$），相比之下 $T^{2}$ MLR 的额外 MLP（$5D^{2}$ 参数）和 LRT 的逐层投影引入 $LD^{2}$ 参数。更大和主要的区别在于经验评估的范围：我们的工作执行更大规模的预训练（高达 400B tokens），具有递归调度；因此，我们设法在不同工作负载上经验验证实际推理时改进，而 LRT 仅考虑非自由形式评估（类似于我们在图 4 右侧的设置），$T^{2}$ MLR 仅在对数学语料库微调模型后考虑合成状态跟踪任务和 gsm8k。然而，考虑到精神上的相似性，我们没有理由认为一种方法的性能会与其他方法显著不同，确切地说哪种方法（更广泛地说，哪种形式的过去隐藏状态注入）在大规模上给出最佳性能仍不清楚，因为我们没有资源进行验证。

#### 潜在和连续推理

我们的方法将顶层潜在馈送到上下文中，类似于潜在推理方法的核心思想，如 Coconut [^19] 和 Soft Thinking [^39]。最大的区别是：(a) 我们专注于预训练；(b) 我们使用隐藏状态来"增强"生成而不是替换离散 token，因此我们的方法更容易监督（但我们可能不太 token 高效）。Hybrid Latent Reasoning via Reinforcement Learning [^36] 提出在后训练时的展开期间同时使用顶层隐藏状态和生成的 token 嵌入，然而他们没有利用顶层隐藏状态，而是使用它来生成词汇嵌入的加权混合，因此不清楚它是否像全带宽 Transformer 那样改善可达性。还有在预训练时研究潜在推理的工作，特别是 PonderLM-2 [^37] 考虑交错的嵌入/隐藏状态作为输入。值得注意的是，他们的训练方法与我们相似，因为他们使用多次前向传播来替换顺序展开，然而他们的方法将输入长度（以及 KV 缓存大小）加倍，因此他们引入的训练和推理开销比全带宽 Transformer 更多。

#### 循环网络的并行训练

另一个相关方向是循环网络的并行训练。这的大多数应用考虑线性特殊情况，如 Mamba [^17] 或 Gated Deltanet [^35]。这些显然是强大的技术，在各种架构中使用，但在所有这些用途中，它们与标准 Transformer 层混合，可以补偿从线性约束继承的缺失表示能力。ParaRNN [^9] 通过解耦过程中每个点的优化并使用牛顿迭代以达到收敛来进一步并行化非线性循环神经网络的训练，结果与用于语言建模的 Transformer 相当。这里的方法走另一条路，在 Transformer 上构造递归，结果改进了基线 Transformer，而且这里的方法似乎显著更高效。

#### 数据高效的预训练

最后，我们的工作属于改善 LLM 预训练数据效率的广泛类别，即给定相同的模型大小和固定数据，我们如何使用更多 FLOP 在固定或更多推理开销下构建更强大的模型。现有方法考虑表示上的额外目标（超出 NTP）[^27] [^38] [^8] [^32]，鼓励隐藏状态包含更丰富的信息。最近也有 NanoGPT 慢速运行竞赛 <sup>2</sup> 研究这种设置，其中官方解决方案 [^28] 训练 LLM 的深度集成并将它们蒸馏到单个模型中以获得更好的性能。与这些方法相比，我们的框架使用额外的训练 FLOP 来解锁一种新型解码模式，在推理时提供免费的性能提升。此外，我们相信技术可以在文献之间流动，例如，我们使用的深度缩放也被证明对训练循环 Transformer 的稳定性很重要 [^29]。我们对递归调度的经验验证也表明仅在训练后期引入计算密集型辅助目标的可行性。

**循环 Transformer** [^13] [^11] [^15] [^14] 也属于这一类别，其中额外的训练 FLOP 通过推理时的重复计算实现测试时扩展。我们的方法在训练期间与循环 Transformer 相似，因为模型的输出在多次前向传播中反复作为输入反馈。然而，在推理时，两种方法在支付额外计算的位置上有所不同。循环 Transformer 通过显式重新应用 Transformer 栈来获得额外的有效深度，从而随递归步数增加推理计算。相反，潜在反馈集成到自回归解码循环中：它重用在前一个 token 处已经产生的顶层状态，仅需要轻量级融合操作，而不需要每个生成的 token 的额外 Transformer 块评估。因此，全带宽 Transformer 保留了递归计算的大部分好处，同时招致每 token 解码开销可忽略不计，仅在使用可选的多遍预填充时才需要额外计算。

更广泛地说，这些方法指向预训练相关扩展轴的转变。传统扩展主要改变模型参数和训练 token。然而，在大规模训练中，可行的设计空间也受到 GPU pod 大小、时钟预算和高质量独特 token 可用性的约束。一旦每参数 token 比率和可访问的高质量数据池成为约束，简单地增加独特训练 token 的数量不再是唯一，甚至不是最直接的改进路径。一个有前途的轴是通过递归、迭代或基于反馈的机制在每个独特 token 上花费更多计算。

## 6 局限性

当前工作有两个主要局限性。首先，我们的实验规模限于 1B 参数模型，我们没有在更大规模的模型上验证该方法。然而，我们相信潜在反馈解码对于更深的模型可能引入更多好处，其中顶层隐藏状态包含更丰富的信息。其次，反馈遍次调度基于启发式；未来工作可以考虑对递归训练阶段的长度进行更严格的消融，以及更原则性的方法来确定递归步数，例如通过来自 [^37] 的 Jacobi 迭代收敛诊断。

## 参考文献

## 附录 A 模型架构

该模型是一个仅解码器的因果语言模型，具有绑定的 100,352 token 嵌入和输出头、24 个 Transformer 层、1,536 维隐藏状态和 6,656 维 SiLU GLU 前馈块。其门控分组查询注意力使用 16 个查询头、8 个共享键/值头、逐头门控、QK RMS 归一化和在 8,192 token 上下文上的旋转位置；大多数层使用 2,048 token 滑动窗口，而每六层使用完全注意力。RMS 归一化应用于每个残差块周围和最终输出处。

## 附录 B 与类似规模的其他模型的 LM eval 性能比较

<table><tbody><tr><td>模型名称</td><td>Tokens</td><td>W/G</td><td>PIQA</td><td>OBQA</td><td>ARC-E</td><td>ARC-C</td><td>平均</td></tr><tr><td>OPT 1.3B</td><td>300B</td><td>59.59</td><td>72.36</td><td>33.40</td><td>50.80</td><td>29.44</td><td>49.87</td></tr><tr><td>Pythia 1B</td><td>300B</td><td>53.43</td><td>69.21</td><td>31.40</td><td>48.99</td><td>27.05</td><td>46.21</td></tr><tr><td>Pythia 1.4B</td><td>300B</td><td>57.38</td><td>70.95</td><td>33.20</td><td>54.00</td><td>28.50</td><td>49.34</td></tr><tr><td>TinyLlama 1B</td><td>2T</td><td>59.43</td><td>73.56</td><td>36.80</td><td>55.47</td><td>32.68</td><td>53.23</td></tr><tr><td>Llama3.2 1B</td><td>9T</td><td>60.46</td><td>74.54</td><td>37.00</td><td>60.48</td><td>35.75</td><td>55.31</td></tr><tr><td>Qwen3 1.7B</td><td>36T</td><td>61.01</td><td>72.36</td><td>36.80</td><td>69.91</td><td>43.26</td><td>57.30</td></tr><tr><td rowspan="5">EvoLM 1B <sup><a href="#fn:31">31</a></sup></td><td>20B</td><td>51.30</td><td>67.85</td><td>32.80</td><td>54.80</td><td>29.61</td><td>46.44</td></tr><tr><td>40B</td><td>54.62</td><td>69.59</td><td>36.20</td><td>58.08</td><td>30.29</td><td>49.38</td></tr><tr><td>80B</td><td>53.59</td><td>70.78</td><td>37.20</td><td>62.71</td><td>35.92</td><td>51.88</td></tr><tr><td>160B</td><td>53.99</td><td>71.71</td><td>36.60</td><td>63.09</td><td>36.09</td><td>52.30</td></tr><tr><td>320B</td><td>53.51</td><td>71.93</td><td>37.20</td><td>62.29</td><td>36.18</td><td>52.49</td></tr><tr><td rowspan="2">全带宽 Transformer 1B</td><td>200B (0 反馈遍次)</td><td>60.46</td><td>71.11</td><td>34.60</td><td>62.42</td><td>34.73</td><td>52.66</td></tr><tr><td>200B (1 反馈遍次)</td><td>62.59</td><td>71.49</td><td>35.00</td><td>63.43</td><td>35.41</td><td>53.58</td></tr></tbody></table>

表 2：0-shot LM Eval 性能比较，EvoLM 和其他开源模型的数字采用自 [^31] 附录表 4。

## 附录 C 训练的完整伪代码

列表 3：训练：$k$ 遍的一步。

[⬇](data:text/plain;base64,ZGVmIGdsdV9jcm9zcyhoLCBlKTogICAgICAjIFtULERdLFtULERdLT5bVCxEXQogICAgcmV0dXJuIChoIEAgV191KSAqIHNpZ21vaWQoZSBAIFdfZykKCmUgPSBlbWJlZCh0b2tlbnMpICAgICAgICAgIyBbVCwgRF0KaCA9IG1vZGVsKGUpICAgICAgICAgICAgICAjIHBhc3MgMSAoc3RhbmRhcmQpCmxvc3MgPSBudHBfbG9zcyhoKQpmb3IgXyBpbiByYW5nZShrIC0gMSk6ICAgICMgcGFyYWxsZWwgaW4gVAogICAgaCA9IGggKyB1bmlmb3JtKC1kZWx0YSwgZGVsdGEpICMgaml0dGVyIG5vaXNlCiAgICB4ID0gZ2x1X2Nyb3NzKHNoaWZ0X3JpZ2h0KGgpLCBpbnB1dF9ybXNub3JtXzEoZSkpCiAgICB4ID0gcHJlZml4X21peGluKHgsIGUpICMgcmFuZG9tIHBsYWluIHByZWZpeAogICAgaCA9IG1vZGVsKGlucHV0X3Jtc25vcm1fMSh4KSkKICAgIGxvc3MgKz0gbnRwX2xvc3MoaCk=)

def glu\_cross(h, e): # \[T,D\],\[T,D\]->\[T,D\]

return (h @ W\_u) \* sigmoid(e @ W\_g)

e = embed(tokens) # \[T, D\]

h = model(e) # pass 1 (standard)

loss = ntp\_loss(h)

for \_ in range(k - 1): # parallel in T

h = h + uniform(-delta, delta) # jitter noise

x = glu\_cross(shift\_right(h), input\_rmsnorm\_1(e))

x = prefix\_mixin(x, e) # random plain prefix

h = model(input\_rmsnorm\_1(x))

loss += ntp\_loss(h)

图 9：全带宽 Transformer 的完整训练代码，包含归一化层和正则化噪声。

## 附录 D vLLM 兼容性

vLLM 上的实现遵循与 EAGLE [^23] / MTP [^16] 相同的设计模式：它保留每个请求的最新主干隐藏状态，并在下一个解码步骤之前将其就地复制到持久的、固定地址的模型缓冲区中，允许 CUDA 图捕获 forward 内部的 glu cross 门控（公式 (4)）。一个修补的 GPUModelRunner.\_model\_forward 将分离的隐藏状态存储在以请求 ID 为键的字典中，使用 query\_start\_loc 将打包的行映射到请求，并删除已完成的请求。然后我们的 forward 函数通过学习的 glu cross 门控将保存的状态与下一个 token 嵌入融合，然后回收结果隐藏状态。与将目标隐藏状态发送到单独的推测草稿模型的 EAGLE/MTP 不同，我们的模型将其自己的状态反馈到同一模型中以定义实际的下一个 token 分布。

## 附录 E 扩展外推结果

图 10：类似于图 3，但将反馈遍次数扩展到 1,000。外推在远超训练时使用的 3 遍之后仍然保持稳定。

## 附录 F 状态跟踪任务的解释

我们构造成对的合成示例，其标签由出现在共享最终冒号之前的信息确定。目标 token 本身永远不包含在输入中。我们附加 $0,8,32,128,$ 或 $256$ 个语义上为空的草稿更新，允许我们在不改变目标的情况下改变序列长度。在最终冒号处，我们记录第 0 层输入和每个 Transformer 块的输出。

#### 完成跟踪

每个输入指定所需计数 $a$ 和已完成计数 $b$。如果 $a=b$ 则目标是 done，否则是 more。对于每个无序数字对 $\{a,b\}$，我们包含所有四种分配 $(a,a),(a,b),(b,a),(b,b)$，在字段和标签上平衡每个数字。一个代表性的匹配对，缩写显示八个重复的干扰项，是

```
required = 4                 required = 4
completed = 9                completed = 4
scratch = 7                  scratch = 7
scratch += 0                 scratch += 0
  ... (8 updates)              ... (8 updates)
Status:                      Status:
```

左侧目标是 more，而右侧目标是 done。这两个示例共享所需计数、草稿上下文、干扰序列和最终 token；只有两个计数器之间的关系改变。

#### 延迟记忆

每个输入首先分配一个二元状态，然后呈现与标签无关的草稿操作。目标根据初始状态是 zero 或 one。例如，

```
state = 0                    state = 1
scratch = 0                  scratch = 0
scratch ^= 0                 scratch ^= 0
scratch ^= 1                 scratch ^= 1
scratch ^= 1                 scratch ^= 1
scratch ^= 0                 scratch ^= 0
scratch ^= 1                 scratch ^= 1
scratch += 0                 scratch += 0
  ... (8 updates)              ... (8 updates)
# final state:               # final state:
```

相应的目标是 zero 和 one。因此模型必须在处理相同的中间上下文时保留初始位。完成跟踪测试从多个字段计算的关系状态，而延迟记忆测试已指定状态的持久传输。

#### 多寄存器最新写入跟踪

我们还测试递归预填充是否可以暴露几个独立更新的变量。输入为寄存器 $r_{0},\ldots,r_{m-1}$ 分配二元值，执行八次与标签无关的草稿更新，然后查询一个寄存器。目标根据该寄存器的最新分配是 zero 或 one。例如，以下匹配的输入共享完整的更新历史，仅在查询的寄存器上不同：

```
r4 = 0                      r4 = 0
r4 = 1                      r4 = 1
r0 = 1                      r0 = 1
r7 = 0                      r7 = 0
  ... (10 assignments)        ... (10 assignments)
r7 = 1                      r7 = 1
r1 = 0                      r1 = 0
scratch = 7                 scratch = 7
scratch += 0                scratch += 0
  ... (7 updates)             ... (7 updates)
query = r0                   query = r1
Value:                       Value:
```

这里最新值是 $r_{0}=1$ 和 $r_{1}=0$，所以左侧目标是 one，右侧目标是 zero。因此模型必须保留每个寄存器的最新值，并将最终查询绑定到该状态的适当组件。

#### 探测构造

我们在每个残差流深度使用四折分组交叉验证训练 $L_{2}$ 正则化线性分类器。完成分组保留整个无序数字对组，记忆分组保留完整的草稿上下文组。扩大的实验包含来自 80 组的 $1{,}600$ 个完成示例和来自 128 组的 $1{,}280$ 个记忆示例。因为每个示例都在同一冒号 token 处结束，标准第 0 层表示除了共享 token 嵌入外不包含标签信息；任何高于随机的可访问性必须通过处理前缀或通过递归融合引入。

#### 寄存器计数和覆写扫描

在寄存器计数扫描中，每个输入包含 16 次分配和八次空更新，并填充到恰好 180 个 token；只有寄存器数量在 $m\in\{1,2,4,8\}$ 上变化。这将维护更多变量的影响与输入长度和总更新计数分开。我们每个寄存器计数使用 128 个结构组。每个组包含一个随机寄存器更新调度、其按位值补码和每个寄存器的查询，分组交叉验证保留整个调度和所有关联查询。结果扫描每个预填充条件包含 $3{,}840$ 个示例。为了直接改变覆写干扰，我们然后固定 $m=8$ 并使用每个寄存器 $2,4,$ 或 $8$ 次写入。每个设置包含来自 128 组的 $2{,}048$ 个示例，分别产生 180、276 和 468 个 token 的输入。

#### 递归后缀控制

除了标准和完全递归预填充之外，我们仅递归预填充最后 $k\in\{1,2,4\}$ 个输入 token，同时标准预填充前面的前缀。一步仅在共享的最终冒号处融合状态，两步递归处理 Value:，四步另外包括查询的寄存器数字和换行符。我们在最终冒号处在第 $0,1,2,$ 和 $4$ 层以及每个剩余深度探测残差流，使用相同的分组 $L_{2}$ 正则化分类器。此扫描区分在整个更新序列中累积的信息与在处理最终查询时局部可访问的信息。

## 附录 G 模型输出

[包含附录 G 的图表，显示模型输出示例的比较]

---

**注释**：

<sup>1</sup> 我们将标准下一个 token 预测（NTP）损失应用于所有遍次。替代方案包括针对多个后续 token 的联合预测 [^2] [^16] [^32]，我们将其留给未来工作。

<sup>2</sup> https://github.com/KellerJordan/modded-nanogpt

[^1]: M. Abdin, J. Aneja, H. Behl, S. Bubeck, R. Eldan, S. Gunasekar, M. Harrison, R. J. Hewett, M. Javaheripi, P. Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905.

[^2]: K. Ahn, A. Lamb, and J. Langford. Efficient joint prediction of multiple future tokens. arXiv preprint arXiv:2503.21801.

[^3]: J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

[^4]: Z. Cai, X. Zhu, Y. Dong, Y. He, and S. Arora. T^2MLR: transformer with temporal middle-layer recurrence. arXiv preprint arXiv:2607.15178.

[^5]: M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba. Evaluating large language models trained on code. 2107.03374.

[^6]: A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, et al. PaLM: scaling language modeling with pathways. Journal of Machine Learning Research 24 (240), pp. 1–113.

[^7]: K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

[^8]: B. Dai, Y. Liu, D. Xue, Y. Song, Q. Guo, K. Chen, X. Wang, B. Zhou, and Z. Lin. Context-level language modeling by learning predictive context embeddings. arXiv preprint arXiv:2510.20280.

[^9]: F. Danieli, P. Rodriguez, M. Sarabia, X. Suau, and L. Zappella. ParaRNN: unlocking parallel training of nonlinear RNNs for large language models. arXiv preprint arXiv:2510.21450.

[^10]: A. Defazio. Why gradients rapidly increase near the end of training. arXiv preprint arXiv:2506.02285.

[^11]: M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and Ł. Kaiser. Universal transformers. arXiv preprint arXiv:1807.03819.

[^12]: A. Fan, T. Lavril, E. Grave, A. Joulin, and S. Sukhbaatar. Addressing some limitations of transformers with feedback memory. arXiv preprint arXiv:2002.09402.

[^13]: Y. Fan, A. Svete, and K. Lee. Bridging the gap between latent and explicit reasoning with looped transformers. arXiv preprint arXiv:2606.31779.

[^14]: J. Geiping, S. M. McLeish, N. Jain, J. Kirchenbauer, S. Singh, B. R. Bartoldson, B. Kailkhura, A. Bhatele, and T. Goldstein. Scaling up test-time compute with latent reasoning: a recurrent depth approach. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[^15]: A. Giannou, S. Rajput, J. Sohn, K. Lee, J. D. Lee, and D. Papailiopoulos. Looped transformers as programmable computers. In International Conference on Machine Learning, pp. 11398–11442.

[^16]: F. Gloeckle, B. Y. Idrissi, B. Rozière, D. Lopez-Paz, and G. Synnaeve. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737.

[^17]: A. Gu and T. Dao. Mamba: linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling.

[^18]: A. Hägele, E. Bakouch, A. Kosson, L. B. Allal, L. Von Werra, and M. Jaggi. Scaling laws and compute-optimal training beyond fixed training durations. Advances in Neural Information Processing Systems 37, pp. 76232–76264.

[^19]: S. Hao, S. Sukhbaatar, D. Su, X. Li, Z. Hu, J. Weston, and Y. Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769.

[^20]: S. Hu, Y. Tu, X. Han, C. He, G. Cui, X. Long, Z. Zheng, Y. Fang, Y. Huang, W. Zhao, et al. MiniCPM: unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395.

[^21]: Z. Huang, X. He, L. Ren, Y. Wang, B. Peng, H. Cheng, S. Wang, P. He, J. Gao, Y. J. Lee, et al. Latent recurrent transformer: architecture exploration, training strategies, and scaling behavior. arXiv preprint arXiv:2605.26797.

[^22]: J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

[^23]: Y. Li, F. Wei, C. Zhang, and H. Zhang. EAGLE: speculative sampling requires rethinking feature uncertainty. arXiv preprint arXiv:2401.15077.

[^24]: Z. Li, H. Liu, D. Zhou, and T. Ma. Chain of thought empowers transformers to solve inherently serial problems. In International Conference on Learning Representations, Vol. 2024, pp. 11911–11943.

[^25]: Z. Li, L. Liu, C. Liang, W. Chen, and T. Zhao. NorMuon: making Muon more efficient and scalable. In Forty-third International Conference on Machine Learning.

[^26]: H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe. Let's verify step by step. arXiv preprint arXiv:2305.20050.

[^27]: Y. Liu, Y. Song, Y. Wang, K. Ge, A. Lamb, Q. Guo, K. Chen, B. Zhou, and Z. Lin. Next concept prediction in discrete latent space leads to stronger language models. arXiv preprint arXiv:2602.08984.

[^28]: B. Mandal, S. Berman, A. Vegesna, and S. Dahal. Q0: primitives for hyper-epoch pretraining. arXiv preprint arXiv:2606.03938.

[^29]: S. Movahedi, V. Milovanović, S. L. Feigin, A. Theus, T. Hofmann, V. Boeva, T. K. Rusch, and A. Orvieto. Fixed-point reasoners: stable and adaptive deep looped transformers. arXiv preprint arXiv:2606.18206.

[^30]: L. Noci, S. Anagnostidis, L. Biggio, A. Orvieto, S. P. Singh, and A. Lucchi. Signal propagation in transformers: theoretical perspectives and the role of rank collapse. Advances in Neural Information Processing Systems 35, pp. 27198–27211.

[^31]: Z. Qi, F. Nie, A. Alahi, J. Zou, H. Lakkaraju, Y. Du, E. Xing, S. Kakade, and H. Zhang. EvoLM: in search of lost language model training dynamics. arXiv preprint arXiv:2506.16029.

[^32]: J. Teoh, M. Tomar, K. Ahn, E. S. Hu, T. Pearce, P. Sharma, A. Krishnamurthy, R. Islam, A. Lamb, and J. Langford. Next-latent prediction transformers learn compact world models. arXiv preprint arXiv:2511.05963.

[^33]: J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems 35, pp. 24824–24837.

[^34]: G. Yang, D. Yu, C. Zhu, and S. Hayou. Tensor programs VI: feature learning in infinite depth neural networks. In The Twelfth International Conference on Learning Representations.

[^35]: S. Yang, J. Kautz, and A. Hatamizadeh. Gated delta networks: improving Mamba2 with delta rule. In International Conference on Learning Representations, Vol. 2025, pp. 29687–29707.

[^36]: Z. Yue, B. Jin, H. Zeng, H. Zhuang, Z. Qin, J. Yoon, L. Shang, J. Han, and D. Wang. Hybrid latent reasoning via reinforcement learning. Advances in Neural Information Processing Systems 38, pp. 5501–5530.

[^37]: B. Zeng, H. Li, S. Song, Y. Wang, Z. Wang, Z. He, X. Wang, and Z. Lin. PonderLM-2: pretraining LLM with latent thoughts in continuous space. arXiv preprint arXiv:2509.23184.

[^38]: X. Zhang, D. Zhang, S. Zhang, X. Qin, Y. Cheng, and J. Yan. NITP: next implicit token prediction for LLM pre-training. In Forty-third International Conference on Machine Learning.

[^39]: Z. Zhang, X. He, W. Yan, A. Shen, C. Zhao, and X. Wang. Soft thinking: unlocking the reasoning potential of LLMs in continuous concept space. Advances in Neural Information Processing Systems 38, pp. 168990–169012.

