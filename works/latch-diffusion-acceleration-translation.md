---
sourceTitle: "Where and When to Commit: Candidate-Aware Decoding for Diffusion Language Models"
sourceUrl: "https://arxiv.org/abs/2607.28166"
sourceAuthors: "Chia-Ming Lee, Ming-Ching Chang, Xin Li, Yu-Lun Liu, Chih-Chung Hsu"
sourceDate: "2026-07-30"
sourceType: "学术论文"
sourceVenue: "arXiv:2607.28166"
sourceFigureCount: 6
language: "zh-CN"
translator: "AI Translation Pipeline"
translationMode: "refined"
pipelineRunId: "20260802-batch"
pipelineSource: "translate/20260802-batch/works-ready/latch-diffusion-acceleration-translation.md"
---


Chia-Ming Lee <sup>1,2</sup>,  Ming-Ching Chang <sup>2</sup>,  Xin Li <sup>2</sup>,  Yu-Lun Liu <sup>1</sup>,  Chih-Chung Hsu <sup>1</sup>
<sup>1</sup> 国立阳明交通大学   <sup>2</sup> 纽约州立大学奥尔巴尼分校

###### 摘要

扩散语言模型（Diffusion Language Models, DLMs）在每个去噪步都会产生一个临时预测，这为生成时的**提前退出**创造了机会——在调度耗尽之前停止解码。现有的提前退出门控基于固定区域的置信度统计或依赖调度的规则来决定终止，这些证据对于一次性冻结所有剩余位置的决策而言过于粗糙，因此它们在长思维链输出上过早触发，而这些输出的答案仅在接近末尾时才稳定。**自适应采样**作为训练无关加速的另一轴，控制着位置提交的速度，但解码仍在继续，且从未验证输出本身是否已稳定。我们引入了一个训练无关的候选感知提前退出框架，将这两个轴分离，并将每个决策与其自身范围的证据相匹配。置信度验证提交（Confidence-Verified Commit, CVC）通过使用从每个任务的输出格式指定的确定性解析器动态提取的候选答案跨度，验证置信度和持续的 argmax 稳定性，来控制序列**何时**可以停止。分块提前提交（Block-Wise Early Commit, BWEC）通过对非最终块应用更低成本的局部规则来控制**何处**加速,同时将最终块和全局终止留给 CVC。我们将它们的组合称为 LATCH（**局部加速与跟踪候选停止**, *Localized Acceleration with Tracked-Candidate Halting*）。与先前方法不同，LATCH 无需后缀提示构造；它不依赖提示锚点但感知格式。我们在零样本设置下使用 LLaDA 和 Dream 对 $11$ 个任务进行端到端评估。LATCH 在所有 $22$ 个评估设置中保持与完整解码准确率相差 $2.0$ 个百分点以内，使用一套跨骨干网络未调优的冻结超参数集，同时在短答案任务上实现 $9.3$ – $17.8\times$ 的端到端 TPS 加速比，在长推理任务上实现 $2.0$ – $3.3\times$ 的加速比。代码可在 [https://github.com/ming053l/LATCH-dLLM](https://github.com/ming053l/LATCH-dLLM) 获取。

## 1 引言

扩散语言模型（Diffusion Language Models, DLMs）通过迭代去噪完全掩码的序列来生成文本，这与自回归模型的从左到右传递形成对比 [^28] [^44]；在每个中间步骤，模型已经对每个位置持有一个**临时**猜测，并且在许多任务上，这个候选答案在最后一个去噪步之前就已经与其完整解码值很好地匹配。训练无关加速沿着两个独立的轴利用这种冗余；**自适应采样**规则（如 [^39] 的 Fast-dLLM、[^38] 的 SlowFast 采样和 [^19] 的 KLASS）改变位置提交的速度，而解码继续进行，而生成时**提前退出**规则（如 [^20] 的 Prophet）决定整个序列何时可以停止。采样提交固定一个位置；终止一次性冻结每个剩余位置，包括答案。因此，这两个轴需要不同的证据，而现有方法恰好在一个轴的证据解决另一个轴的问题时失败。

终止轴承载着更困难的决策，需要相信一次运行已经真正收敛。一个自然的代理是解码进度。然而，在匹配的协议下，短答案任务在大约 $4\%$ 的解码后满足操作收敛标准，而多步任务直到最后 $4\%$ 才满足，因此经过的进度是收敛的糟糕替代。现有的提前退出门控转而使用聚合的位置级信号，即序列范围决策的采样级证据。Prophet 的固定监控区域即使在底层候选答案不断变化时也可能看起来稳定，因此其阈值触发并一次性填充**每个**剩余的掩码位置，没有机会重新审视，而 SchED [^27] 以平滑形式继承了相同的触发器（第 3 节对两者进行了形式化）。在相同的自由形式零样本协议下，在五个长推理任务和两个模型上进行评估，这些终止门控在每个设置中都超过了我们 $2.0$ 点的准确率容差，最多超过 $69$ 点；Prophet 在其自己的**后缀提示**构造下以相同方式失败，确认差距在于什么算作证据，而不在于对答案区域的先验知识。

为了解决这个问题，我们引入了置信度验证提交（Confidence-Verified Commit, CVC），它在每一步重新提取并重新定位候选答案，并要求在允许序列终止之前，在该特定跨度上同时满足置信度和持续的 argmax 稳定性。因为验证的量是候选答案本身，而不是它的代理，所以短暂稳定的猜测无法满足门控。

CVC 的验证是刻意保守的，仅控制序列的**全局**填充并停止决策，而不是单个块的步调。在分块解码下，通常只有最终块包含答案，因此非最终块（通常是中间推理）不在 CVC 的身份跟踪机制设计加速的范围内。步调正是采样轴解决的问题，SlowFast 和 KLASS 表明局部置信度信号可以驱动它。然而，应用于整个缓冲区时，相同的低成本证据也会提交答案跨度，并且没有采样器决定终止；SlowFast 在每个长推理设置上都超过了准确率容差，而 KLASS 仅在其校准骨干网络上保持在容差内，在 Dream 上下降 $5$ – $52$ 点。此外，即使是安全的采样器，仍然会解码经过验证的停止将跳过的每个 token。

这激发了我们的第二个设计，分块提前提交（Block-Wise Early Commit, BWEC），它采用采样轴更低成本、局部评估的置信度阈值，但将其限制在非最终块，同时将最终块和全局填充并停止决策完全留给 CVC。

CVC 和 BWEC 共同构成 LATCH，每个轴一个门控；CVC 在序列停止（**何时**）之前验证答案候选本身的身份和稳定性，BWEC 在证据范围从不超过单个块（**何处**）的情况下控制提交的步调。两者都从一次 LLaDA 校准迁移到 Dream；先前方法的不能。

![参见说明](https://arxiv.org/html/2607.28166v1/x1.png)

图 1：有界准确率加速。LATCH 使用 $37\%$ 的步数匹配完整解码准确率（左），而剩余运行时间集中在 CVC 门控的最终块（右）。

我们的贡献包括：

- 我们展示候选答案稳定化与任务强相关，要么发生在早期，要么仅在解码边界附近（例如，$s_{0.9}$，即 $90\%$ 轨迹已稳定的进度，在任务间范围为 $0.04$ – $0.96$），这使得仅基于进度的提前提交不可靠（第 4.1 节）。
- 我们提出 LATCH，一个每个轴一个门控的候选感知框架；置信度验证提交（CVC）在全局终止前验证动态提取的候选跨度，分块提前提交（BWEC）通过 CVC 保护下的更低成本局部规则控制非最终块的步调（第 4.2 和 4.3 节）。
- 我们在 $11$ 个任务和两个模型系列上评估 LATCH，使用一个冻结的 CVC 阈值元组和一个固定的 BWEC 阈值，从不针对任务或模型重新调优，在短答案任务上获得 $9.3$ – $17.8\times$ 的加速比，在长推理任务上获得 $2.0$ – $3.3\times$ 的加速比，同时在两个模型的每个任务上都保持与完整解码准确率相差 $2.0$ 点以内（第 5 节）。

## 2 相关工作

#### 扩散语言模型

扩散框架可追溯至 [^35]，由 [^15] [^1] [^3] 扩展到离散数据，[^23] 和掩码扩散线 [^34] [^32] [^29] [^47] 建立了 LLaDA 和 Dream 所基于的参数化；产品级 DLMs（Mercury [^17]、Gemini Diffusion [^11]、Seed Diffusion [^36]）扩展了相同的范式，突显了加速这种解码的重要性。LLaDA [^28] 和 Dream [^44] 是我们评估的开放权重 DLMs，在每个掩码位置暴露可修改的预测，并在推理时允许任意步数预算；LATCH 在分块解码下减少该预算，而不会过早终止候选答案。

#### 提前退出与序列终止

提前退出已在层级、推理步和 token 级进行研究。[^12] 通过隐藏状态稳定化将其应用于**深度**。在自回归思维链（Chain-of-Thought, CoT）方法中，S-GRPO [^8]、DEER [^43]、CORE [^46] 和 BMC [^31] 分别通过衰减奖励、转换置信度、脆弱 token 修订和几何重构触发提前停止。对于 DLMs，Prophet [^20] 一旦固定监控区域上的聚合置信度差距超过分阶段阈值就终止生成，SchED [^27] 将该触发器平滑为衰减调度（第 3 节详细说明两者）；因此，冻结整个剩余序列的决策从不检查它冻结的候选答案。LATCH 转而将全局终止条件设定为候选局部收敛证据，即动态提取的候选跨度上的置信度和稳定性，而不是答案正确性。

#### DLMs 的自适应采样

另一个独立的轴控制在解码继续时每步提交多少位置。Fast-dLLM [^39] 并行提交每个置信度超过阈值的位置，SlowFast 采样 [^38] 在探索和加速阶段强制提交高置信度位置，KLASS [^19] 一旦置信度和 KL 稳定性共同超过阈值就按位置解除掩码。并行工作改进了相同的按位置证据；LESS [^26] 对每个掩码位置应用联合置信度和稳定性规则，STDec [^5] 在空间邻域上自适应阈值并对时间一致的 token 放宽它们，TACG [^37] 在 EMA-logit 轨迹上门控提交，$R^{2}$-dLLM [^9] 通过时空冗余确定稳定 token，但需要微调，[^18] 将提交不稳定性归因于去噪顺序中的接近偏差。无论如何改进，这些采样器决定接下来提交哪些位置，从不决定序列是否完成；它们加速的运行仍然解码每个剩余 token，而经过验证的停止则直接跳过这些 token，因此两个轴的节省是组合而不是竞争的。BWEC 采用该轴的证据，遵循 Fast-dLLM 的并行发射规则，但将其限制在非最终块，将终止留给 CVC。正交的训练无关工作通过 KV 缓存和后缀修剪 [^24] [^22] [^16] [^41] [^4] 降低每步成本，与两个轴组合。

## 3 预备知识

#### DLM 生成过程

DLM 通过学习逆转离散损坏过程来生成长度为 $L$ 的序列 $x_{0}$，该过程在前向时间 $u\in[0,1]$ 上逐步掩码干净序列，

$$
q(x_{u}\mid x_{0})=\textstyle\prod_{i=1}^{L}q(x_{u}^{i}\mid x_{0}^{i}),
$$

$$
q(x_{u}^{i}\mid x_{0}^{i})=\left\{\begin{array}[]{ll@{\quad}l}x_{0}^{i},&\text{概率 }1-u,&\text{(a) token 保留}\\
\texttt{[MASK]},&\text{概率 }u,&\text{(b) token 掩码}\end{array}\right.
$$

因此 $x_{1}$ 是完全掩码的，$x_{0}$ 是原始序列。模型 $p_{\theta}(x_{0}\mid x_{u})$ 学习从任意掩码的 $x_{u}$ 预测 $x_{0}$；生成在 $T$ 个离散**去噪步** $t\in\{1,\ldots,T\}$ 上逆转此过程，这是一个独立的离散索引，与前向时间 $u$ 无关，也是本文其余部分使用的索引。每一步为每个掩码位置产生完整预测 $\hat{x}$，然后**重掩码**规则永久提交某个子集并使其余部分保持掩码。由于 $T$ 在推理时选择，使用多少步以及重掩码规则提交什么是本文针对的设计空间。

#### LLaDA 的分块调度

我们使用 LLaDA-8B-Instruct [^28] 作为参考解码器，因为其固定的每块预算和显式提交调度暴露了 LATCH 修改的两个控制点，通过 BWEC 的块推进和通过 CVC 的序列终止。此选择不是模型特定的；我们将相同的门控应用于 Dream-7B-Instruct。LLaDA 将生成区域划分为 $N$ 个相等的连续块 $B_{0},\ldots,B_{N-1}$，从左到右处理，每个块通过**低置信度重掩码**去噪 $S=T/N$ 步（$t$ 在每个块内重置）。令 $c_{i}^{(t)}$ 表示模型对其当前掩码位置 $i$ 的 top-1 预测 $\hat{x}_{i}^{(t)}$ 的 softmax 置信度，令 $m$ 为块的初始掩码数。写 $q=\lfloor m/S\rfloor$ 和 $r=m\bmod S$，步 $t$ 提交 $k_{t}=q+\mathbf{1}[t\leq r]$ 个最高置信度的掩码位置。这个基于排名的配额仅由 $m$ 和 $S$ 决定，独立于置信度值本身，并保证块在步 $S$ 时完全提交。

#### 提前终止门控

Prophet [^20] 和 SchED [^27] 在终止轴上修改这个固定调度，决定整个序列何时可以停止并填充；第 2 节的自适应采样器转而控制位置提交的速度，从不做出该决策。Prophet 对监控区域 $\mathcal{R}$ 的 top-1/top-2 logit 差距取平均，

$$
\bar{g}_{t}=\tfrac{1}{|\mathcal{R}|}\textstyle\sum_{i\in\mathcal{R}}\left(\ell_{i}^{(t),1}-\ell_{i}^{(t),2}\right),
$$

其中 $\ell_{i}^{(t),1},\ell_{i}^{(t),2}$ 是位置 $i$ 步 $t$ 的两个最大 logit。这里，$\mathcal{R}$ 是一个任务格式特定的监控区域，其定义在我们的自由形式评估和 Prophet 的后缀提示设置中保持固定（附录 A.2 和 E）。一旦 $\bar{g}_{t}$ 超过其已发布的三阶段、依赖进度的阈值，Prophet 就在单步内填充所有剩余掩码位置；SchED 将此平滑为带稳定性保护的衰减曲线，仍然作为一次全局填充触发。两个门控都将位置级信号聚合为一个序列范围的触发器，从不检查动态提取的候选跨度的身份和稳定性，因此一个位置可能看起来收敛，而其预测的 token 却不断变化。

## 4 候选感知解码

第 1 节激发了每个加速轴一个门控，CVC 用于**何时**停止，BWEC 用于**何处**加速（图 2）；本节完整开发两者。第 4.1 节激发 CVC 的门控，第 4.2 节开发它，第 4.3 节开发 BWEC。

![参见说明](https://arxiv.org/html/2607.28166v1/x2.png)

图 2：LATCH 结合全局答案级停止与局部分块加速。CVC 提供经过验证的全局停止，而 BWEC 加速非最终块 $B_{0},\ldots,B_{N-2}$；最终块 $B_{N-1}$ 在 CVC 下保留基线 top-k 解码。

图中的 CVC 轨迹使这一点具体化；候选答案在每一步重新提取和重新定位，因此跟踪的是候选值及其时间稳定性，而不仅仅是固定位置的置信度。在显示的 GSM8K 轨迹上，提取器搜索由任务格式指定的固定尾部区域（附录 A.2）；在最终块 $B_{N-1}$ 内，候选答案本身闪烁（$538\!\to\!540$），然后 $\mathrm{run}_{t}$ 积累足够连续的有效观察以满足公式 4 中的联合置信度和稳定性准则；因此提交不基于即时检查。

### 4.1 任务依赖的候选答案稳定化

#### 稳定化时机差距

我们收集 LLaDA-8B-Instruct 在 $120$ 个零样本问题（MMLU、GSM8K 和 MATH 各 $40$ 个）上的完整（非提前退出）轨迹，在没有后缀提示锚点的自由形式提示下，并检查每个轨迹提取的候选答案何时**持续**匹配其自己的完整解码输出 $a_{j}^{\mathrm{full}}$：令 $s_{j}=\min\{s:a_{j,u}=a_{j}^{\mathrm{full}}\ \forall u\geq s\}$ 为候选答案再也不改变之后的最早进度，这是一个稳定化诊断，而不是正确性（准确率是独立的，表 1）；CVC 在解码时从不具有真实值访问权限。对于一个队列，定义 $s_{p}$ 为满足 $s_{j}\leq s$ 的轨迹比例 $p$ 的最早进度。仅在达到 $s_{j}$ 时才计入轨迹，而不是在第一次仅仅触及时，这个更严格的测试确认了明显的时机差距；短答案轨迹达到 $s_{0.9}=0.04$，而长推理任务为 $s_{0.9}=0.96$，与图 3 在门控自己的重放证据中显示的相同分裂。这种分裂在更严格的标准下重新出现，使其成为轨迹的属性，而不是绘制边界的阈值。

#### 为什么位置级门控过早触发

Prophet 的触发器忽略候选答案是否在步间保持不变（第 3 节）。在 GSM8K/MATH 上，它在 $76$ – $77\%$ 进度时触发，接近其最宽松的阶段，在大多数候选答案稳定之前。没有后缀锚点，其固定区域只是候选跨度的代理。图 4 确认了这一点；Prophet 在 $73/93$ 个 GSM8K 和 $81/93$ 个 MATH 轨迹上过早，CVC 各在 $4/93$ 个上。

### 4.2 置信度验证提交：决定何时停止

![参见说明](https://arxiv.org/html/2607.28166v1/x3.png)

图 3：在从 CVC 自身校准中保留的诊断示例上的提交时机（$n=60$，与其 120 轨迹校准池不相交；与表 1 相同的协议），确认模式推广到校准数据之外。Prophet 的置信度差距经常在答案稳定之前在 GSM8K/MATH 上触发，强制过早填充；CVC 在 MMLU 上提前提交但在长推理轨迹上等待。熵证实了这一点：在 Prophet 的触发时接近 2 比特，在 CVC 的触发时接近 0（$n=12$，MMLU）。阴影带：自举 $50/80/95\%$ 置信区间。

#### 联合收敛准则

任务无关的最早提交阈值是错误的代理（第 4.1 节）；LLaDA 强制提交固定配额，而不管置信度如何（第 3 节），因此仅凭运行长度无法将细化与调度意外分开，而仅凭置信度没有时间稳定性的概念。联合来看，它们给出了更强的收敛代理。在步 $t$，令 $a_{t}$ 为提取的答案，$\mathrm{run}_{t}$ 为跨连续相同值提取的稳定性计数器（在没有候选答案时暂停而不是重置；附录 A.3），$\mathrm{changes}_{t}$ 为翻转计数，$c_{t}\in[0,1]$ 为 $a_{t}$ 跨度上的平均置信度。门控在满足以下条件的第一个 $t$ 处提交

$$
\displaystyle c_{t}
$$

$$
\displaystyle\geq\tau_{\text{CVC}}
$$

$$
\displaystyle\mathrm{run}_{t}
$$

$$
\displaystyle\geq\max\big(p_{\min},\,\lceil\gamma\cdot\mathrm{changes}_{t}\rceil\big)
$$

两者必须**联合**成立：仅凭置信度无法排除运气好但不稳定的猜测，而仅凭稳定性无法排除尚未改变的低置信度答案；$p_{\min}$ 防止在任何改变发生之前立即提交，而 $\gamma$ 设置每次额外翻转需要多少进一步稳定步。所有三个超参数（$\tau_{\text{CVC}},\gamma,p_{\min}$）都校准一次然后冻结（下一段），与 Prophet 自己已发布的阈值不同，后者即使在其发布的默认值中也因任务系列而异（附录 A.1）。附录 F.1 消融每个条件：删除任何一个在 MMLU 上是安全的，但在至少一个长推理任务上失败。

#### 提取器、缺失候选答案和置信度

两个条件都从相同的底层信号读取：$a_{t}$ 是提取器自己的归一化输出，而不是原始 token 跨度，因此仅格式差异从不注册为改变；没有可提取候选答案的步使 $\mathrm{run}_{t},\mathrm{changes}_{t}$ 保持不变并自动使置信度门控失败。$c_{t}$ 是跨度上按 token softmax 置信度的简单平均值，每步重新计算，无论掩码状态如何；它反映提交时的操作一致性而不是认识论确定性（附录 A.3）。

#### 一个冻结设置，更强证据

两个模型系列使用相同的数值元组 $(\tau_{\text{CVC}},\gamma,p_{\min})$，在 LLaDA 上跨三个开发任务校准一次，从不针对任务或模型重新调优；我们还将推理任务上的答案搜索区域限制在缓冲区的尾部部分，在源头抑制大多数中间推理算术作为候选答案，而不是在下游容忍它。CVC 解决**何时**停止；我们接下来询问**何处**不需要完整验证。

### 4.3 分块提前提交：决定何处加速

#### 将块局部推进与全局停止分离

非最终块通常包含中间推理而不是最终答案，特别是在长推理任务上，候选答案仅在接近解码边界时才稳定（第 4.1 节），这激发了第 2 节中自适应采样器提供的那种更低成本的局部规则。BWEC 仅将此规则应用于非最终块，而最终块保留基线调度，全局终止仍在 CVC 下。

#### 置信度和调度 top-k，联合

CVC 不门控单个非最终块提交；它监控缓冲区配置的尾部搜索区域上提取的候选答案，无论它位于该区域内何处，并仅保留填充并停止**整个**生成的决策。**非最终**块允许激进的发射，通过没有答案语义的纯置信度阈值规则。对于步 $t$ 的块 $n<N-1$，每个**当前掩码的**位置 $i\in B_{n}$（即 $x_{i}^{(t)}=\texttt{[MASK]}$；已提交的位置不受此规则影响）更新为

$$
x_{i}^{(t+1)}=\left\{\begin{array}[]{ll@{\quad}l}\hat{x}_{i}^{(t)},&c_{i}^{(t)}\geq\tau_{\text{BWEC}},&\text{(a) 置信度提交}\\
\hat{x}_{i}^{(t)},&i\in\mathcal{K}^{(t)},&\text{(b) 调度 top-}k\text{ 提交}\\
x_{i}^{(t)},&\text{否则},&\text{(c) 保持掩码}\end{array}\right.
$$

其中 $\mathcal{K}^{(t)}$ 是块 $n$ 当前掩码位置中调度的 top-$k$ 集（第 3 节）。调度 top-$k$ 保证进度，即使在给定步上没有位置超过阈值；它总是按排名提交 $k$ 个最高置信度的仍掩码位置，因此无论如何块都按调度完成。置信度分支机会性地提交模型解决它们的时刻的**额外**位置，按值（$c_{i}^{(t)}\geq\tau_{\text{BWEC}}$，遵循 Fast-dLLM [^39]）而不是排名；如果任一规则会提交，位置就提交。一旦块中的每个位置都以这种方式提交，该整个块就提前完成：我们在缓冲区状态 $x$ 上定义块就绪谓词，$R_{n}(x)=\bigwedge_{i\in B_{n}}\big[x_{i}\neq\texttt{[MASK]}\big]$，在公式 5 的更新后立即对 $x^{(t+1)}$ 检查；一旦 $R_{n}(x^{(t+1)})$ 成立，块 $n$ 的剩余调度步就被跳过，解码直接前进到块 $n+1$，这是真正的**分块提前退出**；跳过块剩余预算的决策完全基于该块自己的证据，从不基于某个其他不相关块可能主导的序列范围平均值，这正是 Prophet 表现出的失败模式（第 4.1 节）。

**最终**块 $B_{N-1}$ 被排除在公式 5 之外；它仅遵循基线 top-$k$ 调度（仅情况 (b)，从不置信度提交），而 CVC 独立确定**整个**序列（包括最终块）是否可以提前填充和终止（第 4.2 节）。关于此最终块范围的范围和注意事项推迟到附录 F。

## 5 实验

**数据集和设置**。我们评估 LLaDA-8B-Instruct [^28] 和 Dream-7B-Instruct [^44]，bfloat16 (bf16)，贪婪解码，零样本，在单个 NVIDIA A100-SXM4-40GB GPU 上，跨 $11$ 个任务，六个**短答案**任务（MMLU [^13]、ARC-Challenge [^6]、HellaSwag [^45]、WinoGrande [^33]、PIQA [^2]、TruthfulQA-MC1 [^21]）和五个**长推理**任务（GSM8K [^7]、MATH [^14]、SVAMP [^30]、ASDiv [^25]、GSM-Hard [^10]）。CVC 的元组（$\gamma{=}2.0,\tau_{\text{CVC}}{=}0.7,p_{\min}{=}3$）、SchED 的衰减调度和 KLASS 的 $\tau/\epsilon_{\text{KL}}$（附录 A.1）各自校准一次，离线，在 LLaDA 上（CVC 使用 $n{=}120$ 个缓存轨迹，MMLU、GSM8K、MATH 各 $40$ 个，SchED 和 KLASS 也重用相同池）并对 Dream 和其他八个任务保持不变重用，确保公平的跨骨干网络比较；只有 Prophet 的分阶段阈值转而来自其自己已发布的默认值。$\tau_{\text{BWEC}}{=}0.9$ 转而先验固定，因为 BWEC 的调度改变了哪些未来步执行，因此无法通过离线重放以 CVC 的方式校准；我们转而在与表 1 相同的保留示例上检查其敏感性（附录 G），但从不针对它优化。

**评估指标和协议公平性**。准确率和步数遵循 Prophet 自己的评估约定，任务特定答案提取后的精确匹配（附录 A.3）和解码步数，分别；加速比是方法每秒 token 数（TPS）与基线 TPS 的比率，在专用 GPU 上端到端测量（附录 B）。配对 McNemar 检验和自举置信区间发现 LATCH 和基线之间在任何单元中都没有统计显著的准确率差异（附录 C）。我们还复制了 Prophet 自己的后缀提示设置（附录 E），相同地应用于**基线**、Prophet 和 **LATCH**；即使在不再需要找到答案区域后，Prophet 的下降仍然存在，而 LATCH 没有显示任何下降。

### 5.1 主要结果
# 表 1：零样本评估结果（自由形式生成）

> 本表展示 LATCH 与基线方法在 11 个任务上的性能对比。所有数值保留原文，仅翻译表头和任务分组标签。

## 表头说明
- **Task**: 任务名称
- **Variant**: 方法变体
- **Acc (%)**: 准确率（百分比）
- **Avg. Step**: 平均步数
- **TPS**: 每秒 token 数
- **Speedup**: 加速比

## 任务分组
- **General / short-answer tasks**: 通用/短答案任务（单 token 或短跨度答案）
- **Long-reasoning tasks**: 长推理任务（多步 CoT）

---

以下为完整结果表（保留原始 HTML 表格结构）：


<table><tbody><tr><th></th><th></th><td colspan="4">LLaDA-8B-Instruct</td><td colspan="4">Dream-7B-Instruct</td></tr><tr><th>Task</th><th>Variant</th><td>Acc (%)</td><td>Avg. Step</td><td>TPS</td><td>Speedup</td><td>Acc (%)</td><td>Avg. Step</td><td>TPS</td><td>Speedup</td></tr><tr><th colspan="10">General / short-answer tasks (single-token or short-span answers)</th></tr><tr><th>MMLU</th><th>Baseline</th><td>64.0</td><td>64.0</td><td>29.4</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>71.5</td><td>64.0</td><td>33.0</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=200, sequence length=64)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>62.5 (-1.5)</td><td>8.2 (-55.8)</td><td>192.2</td><td>6.54 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>71.5 (+0.0)</td><td>14.8 (-49.2)</td><td>126.8</td><td>3.84 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=16, 4 blocks)</th><th>SchED <sup><a href="#fn:27">27</a></sup></th><td>61.5 (-2.5)</td><td>31.0 (-33.0)</td><td>58.8</td><td>2.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>69.0 (-2.5)</td><td>12.9 (-51.1)</td><td>156.1</td><td>4.73 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>SlowFast <sup><a href="#fn:38">38</a></sup></th><td>63.0 (-1.0)</td><td>17.9 (-46.1)</td><td>140.6</td><td>4.78 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>74.5 (+3.0)</td><td>9.6 (-54.4)</td><td>239.2</td><td>7.25 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>KLASS <sup><a href="#fn:19">19</a></sup></th><td>62.5 (-1.5)</td><td>29.2 (-34.8)</td><td>59.5</td><td>2.02 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>72.5 (+1.0)</td><td>11.3 (-52.7)</td><td>161.1</td><td>4.88 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>LATCH (<math><semantics><mrow><msub><mi>τ</mi> <mtext>BWEC</mtext></msub> <mo>=</mo> <mn>0.9</mn></mrow> <annotation>\tau_{\text{BWEC}}{=}0.9</annotation></semantics></math>)</th><td>64.0 (+0.0)</td><td>5.1 (-58.9)</td><td>446.6</td><td>15.20 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>70.0 (-1.5)</td><td>5.0 (-59.0)</td><td>469.3</td><td>14.21 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>ARC-C</th><th>Baseline</th><td>86.5</td><td>64.0</td><td>30.6</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>87.5</td><td>64.0</td><td>35.6</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=200, sequence length=64)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>86.0 (-0.5)</td><td>7.5 (-56.5)</td><td>273.5</td><td>8.95 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>86.0 (-1.5)</td><td>14.0 (-50.0)</td><td>156.1</td><td>4.39 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=16, 4 blocks)</th><th>SchED <sup><a href="#fn:27">27</a></sup></th><td>85.0 (-1.5)</td><td>41.1 (-22.9)</td><td>47.8</td><td>1.56 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>88.5 (+1.0)</td><td>11.0 (-53.0)</td><td>200.0</td><td>5.62 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>SlowFast <sup><a href="#fn:38">38</a></sup></th><td>85.0 (-1.5)</td><td>23.7 (-40.3)</td><td>97.0</td><td>3.17 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>87.0 (-0.5)</td><td>10.4 (-53.6)</td><td>245.8</td><td>6.90 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>KLASS <sup><a href="#fn:19">19</a></sup></th><td>87.0 (+0.5)</td><td>37.5 (-26.5)</td><td>50.0</td><td>1.63 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>88.5 (+1.0)</td><td>11.6 (-52.4)</td><td>173.4</td><td>4.87 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>LATCH (<math><semantics><mrow><msub><mi>τ</mi> <mtext>BWEC</mtext></msub> <mo>=</mo> <mn>0.9</mn></mrow> <annotation>\tau_{\text{BWEC}}{=}0.9</annotation></semantics></math>)</th><td>86.0 (-0.5)</td><td>4.0 (-60.0)</td><td>473.0</td><td>15.47 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>87.0 (-0.5)</td><td>4.4 (-59.6)</td><td>500.8</td><td>14.08 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>HellaSwag</th><th>Baseline</th><td>75.5</td><td>64.0</td><td>23.9</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>75.5</td><td>64.0</td><td>24.7</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=200, sequence length=64)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>75.0 (-0.5)</td><td>6.0 (-58.0)</td><td>254.1</td><td>10.64 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>75.0 (-0.5)</td><td>16.2 (-47.8)</td><td>95.0</td><td>3.84 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=16, 4 blocks)</th><th>SchED <sup><a href="#fn:27">27</a></sup></th><td>75.5 (+0.0)</td><td>27.7 (-36.3)</td><td>58.7</td><td>2.46 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>75.0 (-0.5)</td><td>19.0 (-45.0)</td><td>83.1</td><td>3.37 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>SlowFast <sup><a href="#fn:38">38</a></sup></th><td>77.0 (+1.5)</td><td>10.9 (-53.1)</td><td>156.1</td><td>6.53 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>72.5 (-3.0)</td><td>9.0 (-55.0)</td><td>185.1</td><td>7.49 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>KLASS <sup><a href="#fn:19">19</a></sup></th><td>75.0 (-0.5)</td><td>19.4 (-44.6)</td><td>78.5</td><td>3.28 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>69.5 (-6.0)</td><td>11.3 (-52.7)</td><td>131.0</td><td>5.30 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>LATCH (<math><semantics><mrow><msub><mi>τ</mi> <mtext>BWEC</mtext></msub> <mo>=</mo> <mn>0.9</mn></mrow> <annotation>\tau_{\text{BWEC}}{=}0.9</annotation></semantics></math>)</th><td>75.5 (+0.0)</td><td>4.2 (-59.8)</td><td>320.9</td><td>13.44 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>75.5 (+0.0)</td><td>6.5 (-57.5)</td><td>229.6</td><td>9.29 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>WinoGrande</th><th>Baseline</th><td>75.5</td><td>64.0</td><td>31.1</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>71.5</td><td>64.0</td><td>38.6</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=200, sequence length=64)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>76.0 (+0.5)</td><td>5.9 (-58.1)</td><td>320.6</td><td>10.30 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>69.5 (-2.0)</td><td>15.3 (-48.7)</td><td>153.3</td><td>3.97 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=16, 4 blocks)</th><th>SchED <sup><a href="#fn:27">27</a></sup></th><td>75.5 (+0.0)</td><td>5.8 (-58.2)</td><td>336.8</td><td>10.83 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>71.0 (-0.5)</td><td>2.1 (-61.9)</td><td>1035.9</td><td>26.84 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>SlowFast <sup><a href="#fn:38">38</a></sup></th><td>77.5 (+2.0)</td><td>8.5 (-55.5)</td><td>241.5</td><td>7.77 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>71.0 (-0.5)</td><td>8.0 (-56.0)</td><td>297.3</td><td>7.70 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>KLASS <sup><a href="#fn:19">19</a></sup></th><td>76.5 (+1.0)</td><td>13.3 (-50.7)</td><td>142.5</td><td>4.58 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>72.5 (+1.0)</td><td>7.6 (-56.4)</td><td>277.3</td><td>7.19 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>LATCH (<math><semantics><mrow><msub><mi>τ</mi> <mtext>BWEC</mtext></msub> <mo>=</mo> <mn>0.9</mn></mrow> <annotation>\tau_{\text{BWEC}}{=}0.9</annotation></semantics></math>)</th><td>76.0 (+0.5)</td><td>3.4 (-60.6)</td><td>555.4</td><td>17.84 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>71.5 (+0.0)</td><td>4.4 (-59.6)</td><td>624.6</td><td>16.19 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>PIQA</th><th>Baseline</th><td>81.5</td><td>64.0</td><td>31.1</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>87.5</td><td>64.0</td><td>36.2</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=200, sequence length=64)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>82.0 (+0.5)</td><td>10.8 (-53.2)</td><td>217.4</td><td>6.98 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>87.5 (+0.0)</td><td>18.9 (-45.1)</td><td>117.4</td><td>3.24 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=16, 4 blocks)</th><th>SchED <sup><a href="#fn:27">27</a></sup></th><td>81.0 (-0.5)</td><td>24.5 (-39.5)</td><td>79.0</td><td>2.54 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>87.0 (-0.5)</td><td>17.1 (-46.9)</td><td>125.6</td><td>3.47 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>SlowFast <sup><a href="#fn:38">38</a></sup></th><td>84.0 (+2.5)</td><td>10.0 (-54.0)</td><td>189.6</td><td>6.10 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>89.5 (+2.0)</td><td>9.1 (-54.9)</td><td>231.0</td><td>6.38 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>KLASS <sup><a href="#fn:19">19</a></sup></th><td>81.5 (+0.0)</td><td>18.6 (-45.4)</td><td>101.0</td><td>3.25 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>91.5 (+4.0)</td><td>10.6 (-53.4)</td><td>188.3</td><td>5.20 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>LATCH (<math><semantics><mrow><msub><mi>τ</mi> <mtext>BWEC</mtext></msub> <mo>=</mo> <mn>0.9</mn></mrow> <annotation>\tau_{\text{BWEC}}{=}0.9</annotation></semantics></math>)</th><td>81.5 (+0.0)</td><td>3.5 (-60.5)</td><td>534.4</td><td>17.16 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>87.5 (+0.0)</td><td>3.7 (-60.3)</td><td>572.2</td><td>15.81 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>TruthfulQA</th><th>Baseline</th><td>66.0</td><td>64.0</td><td>31.1</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>64.0</td><td>64.0</td><td>35.0</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=200, sequence length=64)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>66.5 (+0.5)</td><td>9.3 (-54.7)</td><td>164.9</td><td>5.30 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>63.5 (-0.5)</td><td>16.2 (-47.8)</td><td>125.6</td><td>3.59 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=16, 4 blocks)</th><th>SchED <sup><a href="#fn:27">27</a></sup></th><td>57.5 (-8.5)</td><td>31.9 (-32.1)</td><td>61.2</td><td>1.97 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>60.5 (-3.5)</td><td>15.4 (-48.6)</td><td>140.2</td><td>4.01 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>SlowFast <sup><a href="#fn:38">38</a></sup></th><td>65.0 (-1.0)</td><td>17.2 (-46.8)</td><td>111.1</td><td>3.57 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>56.5 (-7.5)</td><td>11.0 (-53.0)</td><td>198.2</td><td>5.66 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>KLASS <sup><a href="#fn:19">19</a></sup></th><td>57.5 (-8.5)</td><td>28.7 (-35.3)</td><td>65.2</td><td>2.10 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>62.5 (-1.5)</td><td>12.7 (-51.3)</td><td>152.5</td><td>4.36 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>LATCH (<math><semantics><mrow><msub><mi>τ</mi> <mtext>BWEC</mtext></msub> <mo>=</mo> <mn>0.9</mn></mrow> <annotation>\tau_{\text{BWEC}}{=}0.9</annotation></semantics></math>)</th><td>66.0 (+0.0)</td><td>4.4 (-59.6)</td><td>434.6</td><td>13.98 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>64.5 (+0.5)</td><td>5.0 (-59.0)</td><td>345.5</td><td>9.87 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th colspan="10">Long-reasoning tasks (multi-step CoT)</th></tr><tr><th>GSM8K</th><th>Baseline</th><td>66.0</td><td>256.0</td><td>22.2</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>85.0</td><td>256.0</td><td>23.1</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=100, sequence length=256)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>38.0 (-28.0)</td><td>185.0 (-71.0)</td><td>25.2</td><td>1.14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>16.0 (-69.0)</td><td>76.1 (-179.9)</td><td>70.8</td><td>3.06 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=32, 8 blocks)</th><th>SchED <sup><a href="#fn:27">27</a></sup></th><td>59.0 (-7.0)</td><td>224.3 (-31.7)</td><td>26.0</td><td>1.17 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>18.0 (-67.0)</td><td>76.9 (-179.1)</td><td>77.9</td><td>3.37 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>SlowFast <sup><a href="#fn:38">38</a></sup></th><td>41.0 (-25.0)</td><td>70.9 (-185.1)</td><td>74.7</td><td>3.37 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>20.0 (-65.0)</td><td>89.2 (-166.8)</td><td>76.4</td><td>3.31 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>KLASS <sup><a href="#fn:19">19</a></sup></th><td>65.0 (-1.0)</td><td>105.5 (-150.5)</td><td>50.0</td><td>2.25 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>33.0 (-52.0)</td><td>96.5 (-159.5)</td><td>53.9</td><td>2.33 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>LATCH (<math><semantics><mrow><msub><mi>τ</mi> <mtext>BWEC</mtext></msub> <mo>=</mo> <mn>0.9</mn></mrow> <annotation>\tau_{\text{BWEC}}{=}0.9</annotation></semantics></math>)</th><td>67.0 (+1.0)</td><td>94.0 (-162.0)</td><td>55.7</td><td>2.51 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>85.0 (+0.0)</td><td>90.8 (-165.2)</td><td>66.3</td><td>2.86 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>MATH</th><th>Baseline</th><td>31.0</td><td>256.0</td><td>20.9</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>41.0</td><td>256.0</td><td>21.8</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=100, sequence length=256)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>19.0 (-12.0)</td><td>192.7 (-63.3)</td><td>24.4</td><td>1.17 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>15.0 (-26.0)</td><td>164.9 (-91.1)</td><td>29.3</td><td>1.34 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=32, 8 blocks)</th><th>SchED <sup><a href="#fn:27">27</a></sup></th><td>26.0 (-5.0)</td><td>233.5 (-22.5)</td><td>24.1</td><td>1.15 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>33.0 (-8.0)</td><td>199.0 (-57.0)</td><td>29.0</td><td>1.33 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>SlowFast <sup><a href="#fn:38">38</a></sup></th><td>24.0 (-7.0)</td><td>100.0 (-156.0)</td><td>55.0</td><td>2.63 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>9.0 (-32.0)</td><td>98.9 (-157.1)</td><td>63.2</td><td>2.90 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>KLASS <sup><a href="#fn:19">19</a></sup></th><td>34.0 (+3.0)</td><td>133.9 (-122.1)</td><td>37.5</td><td>1.79 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>4.0 (-37.0)</td><td>98.7 (-157.3)</td><td>51.5</td><td>2.36 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>LATCH (<math><semantics><mrow><msub><mi>τ</mi> <mtext>BWEC</mtext></msub> <mo>=</mo> <mn>0.9</mn></mrow> <annotation>\tau_{\text{BWEC}}{=}0.9</annotation></semantics></math>)</th><td>31.0 (+0.0)</td><td>124.6 (-131.4)</td><td>41.9</td><td>2.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>41.0 (+0.0)</td><td>120.3 (-135.7)</td><td>45.1</td><td>2.07 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>SVAMP</th><th>Baseline</th><td>81.0</td><td>256.0</td><td>22.4</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>76.0</td><td>256.0</td><td>23.5</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=100, sequence length=256)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>40.0 (-41.0)</td><td>160.1 (-95.9)</td><td>31.3</td><td>1.39 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>29.0 (-47.0)</td><td>48.8 (-207.2)</td><td>78.7</td><td>3.35 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=32, 8 blocks)</th><th>SchED <sup><a href="#fn:27">27</a></sup></th><td>70.0 (-11.0)</td><td>196.0 (-60.0)</td><td>30.2</td><td>1.35 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>20.0 (-56.0)</td><td>42.9 (-213.1)</td><td>140.9</td><td>5.99 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>SlowFast <sup><a href="#fn:38">38</a></sup></th><td>46.0 (-35.0)</td><td>81.7 (-174.3)</td><td>76.2</td><td>3.40 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>67.0 (-9.0)</td><td>49.9 (-206.1)</td><td>166.0</td><td>7.06 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>KLASS <sup><a href="#fn:19">19</a></sup></th><td>79.0 (-2.0)</td><td>96.4 (-159.6)</td><td>55.9</td><td>2.49 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>71.0 (-5.0)</td><td>58.1 (-197.9)</td><td>91.2</td><td>3.88 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>LATCH (<math><semantics><mrow><msub><mi>τ</mi> <mtext>BWEC</mtext></msub> <mo>=</mo> <mn>0.9</mn></mrow> <annotation>\tau_{\text{BWEC}}{=}0.9</annotation></semantics></math>)</th><td>81.0 (+0.0)</td><td>90.6 (-165.4)</td><td>64.3</td><td>2.87 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>75.0 (-1.0)</td><td>77.7 (-178.3)</td><td>72.5</td><td>3.09 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>ASDiv</th><th>Baseline</th><td>68.0</td><td>256.0</td><td>22.6</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>78.0</td><td>256.0</td><td>23.5</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=100, sequence length=256)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>37.0 (-31.0)</td><td>163.7 (-92.3)</td><td>31.6</td><td>1.40 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>25.0 (-53.0)</td><td>54.8 (-201.2)</td><td>119.1</td><td>5.08 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=32, 8 blocks)</th><th>SchED <sup><a href="#fn:27">27</a></sup></th><td>60.0 (-8.0)</td><td>197.0 (-59.0)</td><td>30.1</td><td>1.33 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>17.0 (-61.0)</td><td>53.3 (-202.7)</td><td>113.6</td><td>4.83 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>SlowFast <sup><a href="#fn:38">38</a></sup></th><td>34.0 (-34.0)</td><td>82.4 (-173.6)</td><td>77.3</td><td>3.42 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>52.0 (-26.0)</td><td>57.9 (-198.1)</td><td>169.3</td><td>7.20 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>KLASS <sup><a href="#fn:19">19</a></sup></th><td>66.0 (-2.0)</td><td>100.8 (-155.2)</td><td>53.5</td><td>2.37 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>58.0 (-20.0)</td><td>67.2 (-188.8)</td><td>78.8</td><td>3.35 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>LATCH (<math><semantics><mrow><msub><mi>τ</mi> <mtext>BWEC</mtext></msub> <mo>=</mo> <mn>0.9</mn></mrow> <annotation>\tau_{\text{BWEC}}{=}0.9</annotation></semantics></math>)</th><td>68.0 (+0.0)</td><td>94.3 (-161.7)</td><td>61.9</td><td>2.74 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>76.0 (-2.0)</td><td>81.4 (-174.6)</td><td>78.2</td><td>3.33 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>GSM-Hard</th><th>Baseline</th><td>35.0</td><td>256.0</td><td>21.7</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>42.0</td><td>256.0</td><td>23.0</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=100, sequence length=256)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>18.0 (-17.0)</td><td>194.1 (-61.9)</td><td>24.4</td><td>1.12 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>9.0 (-33.0)</td><td>103.9 (-152.1)</td><td>49.7</td><td>2.16 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=32, 8 blocks)</th><th>SchED <sup><a href="#fn:27">27</a></sup></th><td>31.0 (-4.0)</td><td>237.4 (-18.6)</td><td>23.9</td><td>1.10 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>10.0 (-32.0)</td><td>123.5 (-132.5)</td><td>47.5</td><td>2.07 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>SlowFast <sup><a href="#fn:38">38</a></sup></th><td>21.0 (-14.0)</td><td>70.0 (-186.0)</td><td>83.2</td><td>3.83 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>11.0 (-31.0)</td><td>96.8 (-159.2)</td><td>77.1</td><td>3.35 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>KLASS <sup><a href="#fn:19">19</a></sup></th><td>35.0 (+0.0)</td><td>109.7 (-146.3)</td><td>46.9</td><td>2.16 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>10.0 (-32.0)</td><td>111.7 (-144.3)</td><td>45.8</td><td>1.99 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th></th><th>LATCH (<math><semantics><mrow><msub><mi>τ</mi> <mtext>BWEC</mtext></msub> <mo>=</mo> <mn>0.9</mn></mrow> <annotation>\tau_{\text{BWEC}}{=}0.9</annotation></semantics></math>)</th><td>36.0 (+1.0)</td><td>96.9 (-159.1)</td><td>54.8</td><td>2.53 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>42.0 (+0.0)</td><td>103.2 (-152.8)</td><td>57.4</td><td>2.50 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr></tbody></table>表 1：自由格式生成下的零样本评估。LATCH 在所有六个短答案任务上均快于 Prophet、SlowFast 和 KLASS，并且在所有五个长推理任务上将准确率下降保持在 $2.0$ 点容差内（无需后缀提示），超参数在不同骨干模型间冻结。红色标记超出该容差的准确率下降；终止门控（Prophet、SchED）和 SlowFast 在每个长推理设置上均超出容差，而 KLASS 仅在其校准骨干模型 LLaDA 上保持在容差内。附录 G.1 给出了两个更快的层级（$\tau_{\text{BWEC}}{=}0.7,0.8$），以部分精度换取额外加速。

我们通过在未用于校准的样本上运行完整的零样本解码来评估 LATCH，覆盖所有 $11$ 个任务和两个模型（LLaDA-8B-Instruct [^28]、Dream-7B-Instruct [^44]）；结果报告于表 1。由于显示的每个变体都已经比基线更快，下面结果的区分在于准确率下降是否保持可忽略，判断标准贯穿本文始终：准确率在 *Baseline* 的 $2.0$ 点以内，且平均步数严格更低。*LATCH* 始终是完整部署配置，即 CVC 加 BWEC（第 4.3 节），而非仅最终块门控（表 2）。

任务组与主要结果。第 4.1 节的稳定化时机对比延续到端到端结果中，按轴划分。终止门控在晚稳定任务上过早触发；Prophet 和 SchED 在所有十个长推理评估上均超出 $2.0$ 点容差。采样器以不同方式失败；SlowFast 同样在所有十个评估上超出容差，而 KLASS 仅在其校准骨干模型 LLaDA 上保持容差，在 Dream 上下降 $5$ – $52$ 点。两者都未能达到验证式停止并填充所提供的短答案加速。图 4 在所有保留轨迹上可视化了过早终止失败，绘制每条轨迹的候选答案稳定时间与各门控提交时间的对比，将 Prophet 置于对角线以下（过早提交）——在 GSM8K 的 $73/93$ 条轨迹和 MATH 的 $81/93$ 条轨迹上，而 CVC 在每个任务上仅有 $4/93$ 条；在 MMLU 上，两个门控在几乎每条轨迹上都在稳定时或稳定后提交。

### 5.2 效率分解与组件消融

两个轴在不同体系中占主导。组件消融揭示了依赖体系的分工；终止门控 CVC 在此处显示的两个短答案任务上贡献了大部分增益，而采样门控 BWEC 在两个长推理任务上贡献最多（表 2），这与 LATCH 在完整评估中的加速划分一致（表 1）。完整配置在每一行都超过单独的任何组件，两个轴的节省是组合而非竞争关系；MATH/LLaDA 上的 $2.00\times$ 结果反映了受保护的最终块，而非记账开销。将全局提交的资格范围限定在最终块是关键；若使其从任何块都符合资格，则保留的 GSM8K 和 MATH 各下降 $4.0$ 点，完全像 Prophet 那样失败（附录 F）。

<table><tbody><tr><td></td><td></td><td colspan="4">LLaDA-8B-Instruct</td><td colspan="4">Dream-7B-Instruct</td></tr><tr><td>任务</td><td>变体</td><td>准确率 (%)</td><td>步数</td><td>TPS</td><td>加速比</td><td>准确率 (%)</td><td>步数</td><td>TPS</td><td>加速比</td></tr><tr><td colspan="10">通用 / 短答案任务</td></tr><tr><td>MMLU</td><td>Baseline</td><td>64.0</td><td>64.0</td><td>29.4</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>71.5</td><td>64.0</td><td>33.0</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=200, 序列长度=64)</td><td>+ CVC</td><td>64.0</td><td>7.0</td><td>358.3</td><td>12.20 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>69.5</td><td>8.0</td><td>130.0</td><td>3.94 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(块=16，4 块)</td><td>+ BWEC</td><td>64.0</td><td>32.8</td><td>56.9</td><td>1.94 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>71.0</td><td>24.8</td><td>88.7</td><td>2.69 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td></td><td>+ CVC + BWEC (完整 LATCH)</td><td>64.0</td><td>5.1</td><td>446.6</td><td>15.20 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>70.0</td><td>5.0</td><td>469.3</td><td>14.21 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>HellaSwag</td><td>Baseline</td><td>75.5</td><td>64.0</td><td>23.9</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>75.5</td><td>64.0</td><td>24.7</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=200, 序列长度=64)</td><td>+ CVC</td><td>76.5</td><td>6.3</td><td>223.0</td><td>9.34 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>75.0</td><td>11.3</td><td>77.4</td><td>3.13 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(块=16，4 块)</td><td>+ BWEC</td><td>76.5</td><td>26.2</td><td>59.6</td><td>2.50 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>75.0</td><td>24.2</td><td>30.5</td><td>1.23 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td></td><td>+ CVC + BWEC (完整 LATCH)</td><td>75.5</td><td>4.2</td><td>320.9</td><td>13.44 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>75.5</td><td>6.5</td><td>229.6</td><td>9.29 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td colspan="10">长推理任务（多步思维链）</td></tr><tr><td>GSM8K</td><td>Baseline</td><td>66.0</td><td>256.0</td><td>22.2</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>85.0</td><td>256.0</td><td>23.1</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=100, 序列长度=256)</td><td>+ CVC</td><td>66.0</td><td>250.0</td><td>22.4</td><td>1.01 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>85.0</td><td>256.0</td><td>10.8</td><td>0.47 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(块=32，8 块)</td><td>+ BWEC</td><td>67.0</td><td>99.4</td><td>54.4</td><td>2.45 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>85.0</td><td>91.3</td><td>30.9</td><td>1.34 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td></td><td>+ CVC + BWEC (完整 LATCH)</td><td>67.0</td><td>94.0</td><td>55.7</td><td>2.51 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>85.0</td><td>90.8</td><td>66.3</td><td>2.86 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>MATH</td><td>Baseline</td><td>31.0</td><td>256.0</td><td>20.9</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>41.0</td><td>256.0</td><td>21.8</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=100, 序列长度=256)</td><td>+ CVC</td><td>31.0</td><td>252.7</td><td>20.9</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>41.0</td><td>255.2</td><td>10.0</td><td>0.46 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(块=32，8 块)</td><td>+ BWEC</td><td>32.0</td><td>128.5</td><td>41.7</td><td>1.99 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>42.0</td><td>121.1</td><td>20.3</td><td>0.93 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td></td><td>+ CVC + BWEC (完整 LATCH)</td><td>31.0</td><td>124.6</td><td>41.9</td><td>2.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>41.0</td><td>120.3</td><td>45.1</td><td>2.07 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr></tbody></table>

表 2：组件消融研究。CVC 主导短答案加速，BWEC 主导长推理加速。在 Dream 上，仅 CVC 的 GSM8K/MATH 行以 $0.46$ – $0.47\times$ 运行，尽管步数接近完整，因为 Dream 很少在后期留下可提取的候选答案；CVC+BWEC 通过 BWEC 的节省恢复完整加速。

CVC 自身的两个条件都是关键。表 2 将 CVC 作为一个单元处理；表 9 针对图 3 背后的相同保留轨迹重放了公式 4 的三个消融变体，并显示其两个条件单独都不安全；附录 F.1 给出完整分析，连同一个无候选答案的对照门控，表明没有任何位置级稳定性阈值既安全又有用（表 10）。

![参见说明](https://arxiv.org/html/2607.28166v1/x4.png)

图 4：保留的 LLaDA-8B-Instruct 轨迹上的提交-稳定化相图。每个点配对一条轨迹的 Prophet（三角形）和 CVC（圆形），由淡线连接；标记填充/轮廓和每个面板的过早提交计数直接标注在各面板中。面板仅包含具有明确稳定化时间的轨迹（MMLU 198/200、GSM8K 93/100、MATH 93/100）；其余缺少可提取的候选答案。

校准成本低，迁移成本低，但可靠性不同。表 4 计时了每个方法自身的超参数搜索，划分再次遵循轴线。终止门控在触发前不改变模型的条件，因此 SchED 和 CVC 通过重放缓存轨迹进行离线校准（CVC 自身的搜索仅需 $39$ 秒 CPU 时间）；采样规则决定后续步骤实际执行的内容，因此 BWEC 和 KLASS 每个设置都需要新的 GPU 解码，BWEC 的扫描主导了 LATCH 总计 $3$ 小时 $43$ 分钟。在 Dream 上重用阈值对每个方法都同样便宜，但只有 CVC 的迁移保持了准确率；SchED、SlowFast、KLASS 和 Prophet 在那里都失败了（表 1）。

分块视图解释了原因。图 5 显示了长推理加速为何达到平台期；BWEC 通过在 token 清除置信度阈值后提交来缩短非最终块，而受保护的最终块在 CVC 下保持接近其完整预算并成为瓶颈，这是步数本身无法显示的不对称性。像 MMLU 和 HellaSwag 这样的短答案任务通常在一到两个块内终止，产生比占用全部八个块的任务更高的加速上限，这是之前任何门控都未能复现的模式，因为每个门控在整个缓冲区上应用一个无差别的规则。然而步数仅是墙钟速度的代理；在连续 $\tau_{\text{BWEC}}$ 扫描下，它可以保持平坦，而 TPS 比率摆动 $2.2$ – $2.8\times$，GSM8K 准确率可能在没有任何步数警告的情况下崩溃（附录 G.3）。

![参见说明](https://arxiv.org/html/2607.28166v1/x5.png)

图 5：分块步数使用对比。按块显示使用的平均步数（占每块预算的百分比），LLaDA-8B-Instruct。KLASS 逐渐减少步数，无最终块保护，而 LATCH 的非最终块节省随 $\tau_{\text{BWEC}}$ 收紧而缩小，最终块保持在 90% 附近。附录 G.2 在所有 11 个任务和两个模型上重复此实验。

## 6 结论

训练无关的扩散语言模型加速分为两个轴；自适应采样决定位置在解码继续时提交的速度，生成时提前退出决定整个序列何时可以停止。LATCH 保持两个轴分离，并为每个轴提供证据与范围匹配的门控：*分块提前提交（Block-Wise Early Commit, BWEC）* 通过局部置信度规则调控非最终块，*置信度验证提交（Confidence-Verified Commit, CVC）* 通过在每步提交前重新提取并重新定位候选答案来管理终止。LATCH 与越来越精细的位置级稳定性规则的区别在于被验证的对象，而非信号；即使每个位置都单独稳定了，也没有位置级规则能够证明任务要求的答案存在且已停止变化，这个问题只有候选答案提取才能提出。然而采样器无论多么精细，仍会解码每个剩余 token，因此验证式终止增加了位置级加速无法达到的节省。LATCH 在一套冻结超参数下，在所有任务上保持在完整解码准确率的 $2.0$ 点以内，在短答案任务上达到 $9.3$ – $17.8\times$ 加速，在长推理任务上达到 $2.0$ – $3.3\times$ 加速，未调优地跨骨干模型迁移；附录 H 讨论了底层答案跨度假设在何处失效。

## 参考文献

## 附录概述

按主题分组，可重现性、协议和统计验证（附录 A–C）、机制证据和公平性（附录 D–E）、结构消融和敏感性（附录 F–G）以及范围扩展和局限性（附录 H）。

- A 表 1 背后的超参数，唯一部署配置、测试框架细节以及答案搜索区域（search_mode）约定。
- B 计时协议，直接端到端的 TPS/加速比测量，专用 GPU，批量大小 $1$。
- C 可忽略的准确率下降是真实的还是噪声？配对显著性和输出相同率。
- D 机制失败案例，Prophet 与 LATCH 在真实 GSM8K/SVAMP/PIQA 轨迹上的对比，信号轨迹和字面解码文本。
- E 答案区域确定，LATCH 与 Prophet 自身的后缀提示结构，正面对比测试，以及为什么短答案任务的搜索与长推理任务不同（search_mode）。
- G 敏感性，连续 $\tau_{\text{BWEC}}$ 扫描，所有 $11$ 个任务和两个模型。
- H 局限性。

## 附录 A 表 1 背后的超参数

### A.1 测试框架细节与部署配置

#### 第 5 节推迟的测试框架细节

零样本意味着裸问题加上，对于推理任务，一行"让我们一步步思考"的思维链提示，没有示例，除此之外没有答案格式脚手架。Dream-7B-Instruct 保留因果语言模型的下一 token 索引约定；我们的测试框架应用其官方的一位 logit 偏移，否则解码与 LLaDA 匹配。Prophet 同行行移植官方门控（未更改的已发布默认值）到这个相同的测试框架中，监控我们自己的答案区域，而不是引用 Prophet 自己仓库中的数字，后者使用不同的提示模板。

表 1 的核心主张是 *LATCH* 仅需一套*数值*超参数集，$\tau_{\text{CVC}}{=}0.7$、$\gamma{=}2.0$、$p_{\min}{=}3$（公式 4）、$\tau_{\text{BWEC}}{=}0.9$（公式 5）、$\texttt{tail\_frac}{=}0.3$（第 4 节），在所有 $11$ 个任务和两个模型上相同，无需按任务或按模型重新调整（确定性提取器和搜索区域遵循每个任务自己的输出格式，下文 search_mode，但两者都未调整）。表 3 明确说明了这一点，以及*确实*按任务变化的一个设置，Prophet 自己的分阶段置信度差距阈值，它按任务族变化（第 5 节）。这正是 *LATCH* 设计为避免需要的按任务调整。

| 任务 | $n$ | len | 步数 | 块 | 批量 | search_mode | Prophet ($\tau_{\text{high}}/\tau_{\text{mid}}/\tau_{\text{low}}$) | SchED (模式/耐心) | KLASS ($\tau/\epsilon_{\text{KL}}$) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MMLU | $200$ | $64$ | $64$ | $16$ | $1$ | first5 | $7.5$ / $5.0$ / $2.5$ | cosine / $0$ | $0.7$ / $0.015$ |
| ARC-C | $200$ | $64$ | $64$ | $16$ | $1$ | first5 | $7.5$ / $5.0$ / $2.5$ | cosine / $0$ | $0.7$ / $0.015$ |
| HellaSwag | $200$ | $64$ | $64$ | $16$ | $1$ | first5 | $7.5$ / $5.0$ / $2.5$ | cosine / $0$ | $0.7$ / $0.015$ |
| WinoGrande | $200$ | $64$ | $64$ | $16$ | $1$ | first5 | $7.5$ / $5.0$ / $2.5$ | cosine / $0$ | $0.7$ / $0.015$ |
| PIQA | $200$ | $64$ | $64$ | $16$ | $1$ | first5 | $7.5$ / $5.0$ / $2.5$ | cosine / $0$ | $0.7$ / $0.015$ |
| TruthfulQA | $200$ | $64$ | $64$ | $16$ | $1$ | first5 | $7.5$ / $5.0$ / $2.5$ | cosine / $0$ | $0.7$ / $0.015$ |
| GSM8K | $100$ | $256$ | $256$ | $32$ | $1$ | last | $8.0$ / $5.0$ / $3.5$ | linear / $4$ | $0.6$ / $0.015$ |
| MATH | $100$ | $256$ | $256$ | $32$ | $1$ | last | $8.0$ / $5.0$ / $3.5$ | linear / $4$ | $0.6$ / $0.015$ |
| SVAMP | $100$ | $256$ | $256$ | $32$ | $1$ | last | $8.0$ / $5.0$ / $3.5$ | linear / $4$ | $0.6$ / $0.015$ |
| ASDiv | $100$ | $256$ | $256$ | $32$ | $1$ | last | $8.0$ / $5.0$ / $3.5$ | linear / $4$ | $0.6$ / $0.015$ |
| GSM-Hard | $100$ | $256$ | $256$ | $32$ | $1$ | last | $8.0$ / $5.0$ / $3.5$ | linear / $4$ | $0.6$ / $0.015$ |

表 3：表 1 背后的按任务设置。$n$ 是保留样本大小；len 是生成长度；步数是完整解码预算；块是块扩散的块大小（第 4 节）；批量在整个过程中为 $1$（附录 B）。search_mode（下文解释）由答案格式固定，从不调整。Prophet 的分阶段阈值从其自己的论文/仓库中未经更改地采用，未经我们调整。LATCH 的五个超参数在每一行上相同（上文列出）。SchED 的模式/耐心是其校准的衰减曲线形状和稳定性保护耐心（$\tau_{\text{high}}/\tau_{\text{low}}$ 重用上述 Prophet 的值）；KLASS 的 $\tau/\epsilon_{\text{KL}}$ 是其两个按族冻结的校准值。其他超参数（两者）固定且未按行显示。

|  | Prophet | SchED | SlowFast | KLASS | LATCH |
| --- | --- | --- | --- | --- | --- |
| LLaDA（墙钟时间） | – | $57$ 分钟 | – | $2$ 小时 $50$ 分钟 | $3$ 小时 $43$ 分钟 |
| 无需 GPU？ | – | $\checkmark$ | – | $\times$ | $\times$ |
| 迁移到 Dream？ | $\times$ | $\times$ | $\times$ | $\times$ | $\checkmark$ |

表 4：校准墙钟时间（在第 5.2 节讨论）。"–"表示无单独校准（超参数已发布或继承）。无需 GPU：$\checkmark$ 离线，$\times$ 需要 GPU。迁移到 Dream：$\checkmark$ 保持，$\times$ 失败。

### A.2 search_mode 控制什么

search_mode 按任务族固定提取器读取缓冲区的哪个区域以及门控重新定位答案跨度的位置（从不调整；详见附录 E.1）。"last"（GSM8K、MATH、SVAMP、ASDiv、GSM-Hard）将两者都限制在尾部 $\texttt{tail\_frac}{=}0.3$ 的部分；"first5"（MMLU、ARC-C、HellaSwag、WinoGrande、PIQA、TruthfulQA）读取整个缓冲区但将搜索限制在前 $5$ 个位置。

### A.3 提取器归一化、缺失候选答案和置信度聚合

公式 4 中的变化跟踪比较使用提取器自己的归一化输出，从不使用原始 token 跨度。数值答案剥离货币符号和标点，框住的数学答案规范化空格/宏，因此格式变化和跨度移动从不记录为变化；只比较归一化值。

$\mathrm{run}_{t}$ 是实现维护的稳定性计数器，而非步数计数；它在具有相同值的连续有效提取中递增，并在无法提取候选答案时暂停而非重置（$\mathrm{changes}_{t}$ 同样保持；$c_{t}$ 未定义，自动使门控失败）。在 $120$ 条校准轨迹中，$30\%$ 显示此类间隙，$21\%$ 显示计数器通过它恢复；严格的"重新从一开始"反事实在所有地方都复现相同的提交结果，因此恢复是常见的但从未改变决策。

置信度 $c_{t}$ 是模型在跨度上的逐步 softmax 平均值，无论掩码状态如何都在每步重新计算，因此它可能部分反映自我重建而非真正的不确定性；我们将其视为操作一致性分数，而非认识论估计。我们使用简单平均而非最小值或几何平均以避免惩罚较长答案，尽管 $\mathrm{run}_{t}$ 的联合要求限制了单个运气好的 token 单独能做的事。

## 附录 B 计时协议

本文中的每个 TPS/加速比数字都是直接端到端计时的，而非从共享的逐步常数外推。对于每个单元格，我们将一次完整的解码循环调用（完整的逐步解码、CVC/BWEC 门控记账包括逐步答案稳定性提取器，以及门控提前停止时的最终填充提交）包裹在 torch.cuda.synchronize() 中，紧接在之前/之后，在主机上使用 time.time()（墙钟时间，而非 CUDA 事件计时器）计时。这统一适用于表 1 中的 *Baseline*、*Prophet*、*LATCH*、*SlowFast Sampling* [^38]、*SchED* [^27] 和 *KLASS* [^19]，以及 *Aggressive* / *Normal* $\tau_{\text{BWEC}}$ 层级（表 11）。

#### 协议

每个单元格使用相同的固定 $n{=}20$ 子样本（表 1 使用的相同保留样本的前 $20$ 个，$\texttt{seed}{=}0$、$\texttt{skip}{=}40$）用于该单元格内的所有计时。首先解码并丢弃 $2$ 个预热样本（紧邻的不相交切片）以吸收首次调用开销（CUDA 上下文/内核预热），然后对相同的 $20$ 个样本连续运行 $3$ 次计时通过；我们报告 $3$ 次通过级平均值的均值，加上它们的变异系数（CV%），因此没有单元格的数字依赖于单次运行。每个单元格在无其他作业共驻的 GPU 上运行（启动前验证，表 5），批量大小 $1$（匹配 *Prophet* 和 *SlowFast Sampling* 自己发布的协议，第 5 节；*SchED* 和 *KLASS* 在相同的固定批量大小下运行以进行受控比较，因为两篇论文都未指定自己的批量大小），bf16，贪婪解码。

<table><tbody><tr><th></th><td>LLaDA-8B-Instruct</td><td>Dream-7B-Instruct</td></tr><tr><th>GPU</th><td colspan="2">NVIDIA A100-SXM4-40GB，无共驻作业</td></tr><tr><th>批量大小</th><td colspan="2"><math><semantics><mn>1</mn> <annotation>1</annotation></semantics></math></td></tr><tr><th>PyTorch / CUDA</th><td colspan="2"><math><semantics><mn>2.5.1</mn> <annotation>2.5.1</annotation></semantics></math> +cu124 / <math><semantics><mn>12.4</mn> <annotation>12.4</annotation></semantics></math></td></tr><tr><th>同步方法</th><td colspan="2">torch.cuda.synchronize() 包裹每次计时调用；主机 time.time()</td></tr><tr><th><math><semantics><mi>n</mi> <annotation>n</annotation></semantics></math> / 预热 / 重复</th><td colspan="2"><math><semantics><mn>20</mn> <annotation>20</annotation></semantics></math> / <math><semantics><mn>2</mn> <annotation>2</annotation></semantics></math> / <math><semantics><mn>3</mn> <annotation>3</annotation></semantics></math>，seed=0，skip=40</td></tr><tr><th>每单元格 CV%</th><td><math><semantics><mn>0.01</mn> <annotation>0.01</annotation></semantics></math> – <math><semantics><mrow><mn>2.13</mn> <mo>%</mo></mrow> <annotation>2.13\%</annotation></semantics></math>（平均 <math><semantics><mrow><mn>0.43</mn> <mo>%</mo></mrow> <annotation>0.43\%</annotation></semantics></math>，<math><semantics><mrow><mi>n</mi> <mo>=</mo> <mn>117</mn></mrow> <annotation>n{=}117</annotation></semantics></math>）</td><td><math><semantics><mn>0.01</mn> <annotation>0.01</annotation></semantics></math> – <math><semantics><mrow><mn>2.55</mn> <mo>%</mo></mrow> <annotation>2.55\%</annotation></semantics></math>（平均 <math><semantics><mrow><mn>0.23</mn> <mo>%</mo></mrow> <annotation>0.23\%</annotation></semantics></math>，<math><semantics><mrow><mi>n</mi> <mo>=</mo> <mn>117</mn></mrow> <annotation>n{=}117</annotation></semantics></math>）</td></tr></tbody></table>

表 5：直接端到端 TPS 测量协议。附录 B 范围内的每个表/图共享。CV% 是每个单元格的 $3$ 次计时通过的变异系数；每个单元格都在 $2.6\%$ 以下，证明 $3$ 次通过的平均值是稳定的，而非噪声主导的。
## 附录 C 微小的准确率下降是真实的，还是噪声？配对显著性与输出相同率

表 1 报告的是点估计值，而 $\pm 2.0$ 点的阈值是预先指定的实际容差，而非置信区间。然而，*基线* 和 *LATCH* 并非独立样本；每个示例都是从相同的提示、模型和随机种子解码得到的，因此正确的比较方式是 *配对比较*，配对不确定性比朴素的独立样本置信区间更紧（例如，在 GSM8K/MATH 的 $n{=}100$ 下，二项近似约为 $\pm 6$ – $7$ 点）。由于 *基线* 和 *LATCH* 解码相同的示例，本附录使用配对 McNemar 检验和配对自举区间评估不确定性；没有任何单元格显示出统计上可检测的准确率差异，尽管这些区间本身并非 $\pm 2.0$ 点边界的等价性检验：部分区间达到 $[-5.0,+0.0]$ 或 $[-3.0,+3.0]$（表 6），足够宽以至于它们本身无法证明该边界成立。我们报告了每个任务的三个配对统计量，两个模型，在与表 1 相同的留出样本上：输出相同率（LATCH 提取的答案与完整预算基线完全匹配的示例比例，这是提前提交未 *改变* 答案的直接证据，与该答案是否恰好正确无关）、对两者不一致的示例进行的 McNemar 精确检验（$b$ = 仅基线正确，$c$ = 仅 LATCH 正确；对称分布与准确率相等一致，非对称分布则不然），以及 $\Delta\text{acc}$ 的配对自举 $95\%$ 置信区间（重新采样示例索引，而非独立的基线/LATCH 抽样）。表 6 报告了完整的任务级分解。

<table><tbody><tr><td></td><th colspan="6">LLaDA-8B-Instruct</th><th colspan="6">Dream-7B-Instruct</th></tr><tr><th>任务</th><th><math><semantics><mi>n</mi> <annotation>n</annotation></semantics></math></th><th><math><semantics><mi>b</mi> <annotation>b</annotation></semantics></math></th><th><math><semantics><mi>c</mi> <annotation>c</annotation></semantics></math></th><th><math><semantics><mi>p</mi> <annotation>p</annotation></semantics></math></th><th><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> acc [<math><semantics><mrow><mn>95</mn> <mo>%</mo></mrow> <annotation>95\%</annotation></semantics></math> CI]</th><th>相同率(%)</th><th><math><semantics><mi>n</mi> <annotation>n</annotation></semantics></math></th><th><math><semantics><mi>b</mi> <annotation>b</annotation></semantics></math></th><th><math><semantics><mi>c</mi> <annotation>c</annotation></semantics></math></th><th><math><semantics><mi>p</mi> <annotation>p</annotation></semantics></math></th><th><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> acc [<math><semantics><mrow><mn>95</mn> <mo>%</mo></mrow> <annotation>95\%</annotation></semantics></math> CI]</th><th>相同率(%)</th></tr><tr><td>MMLU</td><td>200</td><td>1</td><td>1</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>1.5</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>1.5</mn></mrow></mrow> <annotation>-1.5,+1.5</annotation></semantics></math>]</td><td>99.0</td><td>200</td><td>3</td><td>0</td><td>0.250</td><td><math><semantics><mrow><mo>−</mo> <mn>1.5</mn></mrow> <annotation>-1.5</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>3.5</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>-3.5,+0.0</annotation></semantics></math>]</td><td>97.5</td></tr><tr><td>ARC-C</td><td>200</td><td>1</td><td>0</td><td>1.000</td><td><math><semantics><mrow><mo>−</mo> <mn>0.5</mn></mrow> <annotation>-0.5</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>1.5</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>-1.5,+0.0</annotation></semantics></math>]</td><td>99.5</td><td>200</td><td>1</td><td>0</td><td>1.000</td><td><math><semantics><mrow><mo>−</mo> <mn>0.5</mn></mrow> <annotation>-0.5</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>1.5</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>-1.5,+0.0</annotation></semantics></math>]</td><td>99.5</td></tr><tr><td>HellaSwag</td><td>200</td><td>0</td><td>0</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>+</mo> <mn>0.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>+0.0,+0.0</annotation></semantics></math>]</td><td>99.5</td><td>200</td><td>0</td><td>0</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>+</mo> <mn>0.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>+0.0,+0.0</annotation></semantics></math>]</td><td>100.0</td></tr><tr><td>WinoGrande</td><td>200</td><td>1</td><td>2</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.5</mn></mrow> <annotation>+0.5</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>1.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>2.5</mn></mrow></mrow> <annotation>-1.0,+2.5</annotation></semantics></math>]</td><td>98.5</td><td>200</td><td>0</td><td>0</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>+</mo> <mn>0.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>+0.0,+0.0</annotation></semantics></math>]</td><td>100.0</td></tr><tr><td>PIQA</td><td>200</td><td>2</td><td>2</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>2.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>2.0</mn></mrow></mrow> <annotation>-2.0,+2.0</annotation></semantics></math>]</td><td>98.0</td><td>200</td><td>0</td><td>0</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>+</mo> <mn>0.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>+0.0,+0.0</annotation></semantics></math>]</td><td>100.0</td></tr><tr><td>TruthfulQA</td><td>200</td><td>0</td><td>0</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>+</mo> <mn>0.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>+0.0,+0.0</annotation></semantics></math>]</td><td>100.0</td><td>200</td><td>0</td><td>1</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.5</mn></mrow> <annotation>+0.5</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>+</mo> <mn>0.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>1.5</mn></mrow></mrow> <annotation>+0.0,+1.5</annotation></semantics></math>]</td><td>99.0</td></tr><tr><td>GSM8K</td><td>100</td><td>1</td><td>2</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>1.0</mn></mrow> <annotation>+1.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>2.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>5.0</mn></mrow></mrow> <annotation>-2.0,+5.0</annotation></semantics></math>]</td><td>97.0</td><td>100</td><td>0</td><td>0</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>+</mo> <mn>0.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>+0.0,+0.0</annotation></semantics></math>]</td><td>100.0</td></tr><tr><td>MATH</td><td>100</td><td>1</td><td>1</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>3.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>3.0</mn></mrow></mrow> <annotation>-3.0,+3.0</annotation></semantics></math>]</td><td>94.0</td><td>100</td><td>0</td><td>0</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>+</mo> <mn>0.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>+0.0,+0.0</annotation></semantics></math>]</td><td>96.0</td></tr><tr><td>SVAMP</td><td>100</td><td>1</td><td>1</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>3.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>3.0</mn></mrow></mrow> <annotation>-3.0,+3.0</annotation></semantics></math>]</td><td>98.0</td><td>100</td><td>1</td><td>0</td><td>1.000</td><td><math><semantics><mrow><mo>−</mo> <mn>1.0</mn></mrow> <annotation>-1.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>3.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>-3.0,+0.0</annotation></semantics></math>]</td><td>98.0</td></tr><tr><td>ASDiv</td><td>100</td><td>1</td><td>1</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>3.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>3.0</mn></mrow></mrow> <annotation>-3.0,+3.0</annotation></semantics></math>]</td><td>95.0</td><td>100</td><td>2</td><td>0</td><td>0.500</td><td><math><semantics><mrow><mo>−</mo> <mn>2.0</mn></mrow> <annotation>-2.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>5.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>0.0</mn></mrow></mrow> <annotation>-5.0,+0.0</annotation></semantics></math>]</td><td>97.0</td></tr><tr><td>GSM-Hard</td><td>100</td><td>0</td><td>1</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>1.0</mn></mrow> <annotation>+1.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>+</mo> <mn>0.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>3.0</mn></mrow></mrow> <annotation>+0.0,+3.0</annotation></semantics></math>]</td><td>88.0</td><td>100</td><td>1</td><td>1</td><td>1.000</td><td><math><semantics><mrow><mo>+</mo> <mn>0.0</mn></mrow> <annotation>+0.0</annotation></semantics></math> [<math><semantics><mrow><mrow><mo>−</mo> <mn>3.0</mn></mrow><mo>,</mo><mrow><mo>+</mo> <mn>3.0</mn></mrow></mrow> <annotation>-3.0,+3.0</annotation></semantics></math>]</td><td>93.0</td></tr></tbody></table>

表 6：表 1 中准确率差异的配对统计量。对于每个任务和模型，$b$ 和 $c$ 分别表示 McNemar 精确检验中 *基线*-仅正确 和 *LATCH*-仅正确 的示例数；$\Delta$ acc 报告配对自举点估计值和 $95\%$ 置信区间；相同率(%) 是输出相同率。没有任何单元格达到常规显著性水平，且每个置信区间都保持在零附近。

输出相同率在每个任务上都达到 $88$ – $100\%$；LATCH 的提前提交绝大多数情况下都重现了完整预算解码所能达到的完全相同的答案，而非仅仅是准确率相似的不同答案。$b$（基线-仅正确）和 $c$（LATCH-仅正确）在所有任务上都足够小（每个任务 $100$ – $200$ 个示例中 $\leq\!3$ 个），以至于 McNemar 精确检验在此样本量下基本没有检验力来拒绝任何假设，这本身就是信息性的；两个变体在每个任务上最多只在少数几个示例上存在分歧，且没有任务达到常规显著性水平（最小的 $p$ 值为 $0.25$，Dream/MMLU）。每个 $\Delta$ acc 背后的配对自举 $95\%$ 置信区间明显窄于朴素独立样本二项标准误（在 GSM8K 的 $n{=}100$ /acc $\approx\!0.66$ 下约为 $\approx\!4.7$ 点），并且在每个任务上都以零为中心或接近零。基于这些证据，本文通篇使用的 $\pm 2.0$ 点容差并非在悄悄测量噪声。

#### 解读分歧案例

$b$ 和 $c$ 是两个变体在正确/错误两侧得出相反结果的示例 *计数*；底层的逐示例差异显示，这些正是任何两个准确率相似的解码所会产生的那种噪声，而非系统性方向。GSM-Hard/LLaDA 的最低相同率（与该任务自身的长浮点噪声目标一致）仍然主要是两者都错但错在不同的错误数字上，而非 LATCH 将正确答案转换为错误答案。Dream/MMLU 是唯一一个值得点名的单侧分布单元格（$b{=}3$，$c{=}0$），在此样本量下仍远未达到显著性（$p{=}0.25$），且配对自举置信区间的上界仍触及 $0.0$，因此这并非 CVC 下降变得严重的证据，但它是整个扫描中对 CVC 最不利的单个单元格。

## 附录 D 机制性失败案例：Prophet 与 LATCH 在真实轨迹上的对比

![参见说明](https://arxiv.org/html/2607.28166v1/x6.png)

图 6：Prophet 与 CVC 在代表性轨迹上的对比。Prophet 为何失败以及置信度验证提交如何修复它，在一个代表性的真实 GSM8K 轨迹上（第 4 节中的图 3 汇总了 n = 40 n{=}40 个此类轨迹并带有自举置信区间，因此这个单一示例本身并非论文的证据）。第 1 列和第 3 列重放相同的轨迹（真实值 21 =21 ）：Prophet 在 54 % 54\\% 进度时提交到一个乱码的非答案（其窗口的 100 100\\% 仍被掩码）；我们的方法保留到 70 70\\% 并得到正确答案。第 2 列（MMLU）在一次错误启动后成功。

第 1 列和第 3 列重放 *相同的* GSM8K 解码，因此每个中间输出都是相同的，两个门控的决策可以直接比较，一次模型运行，两种门控结果。所有三列都重放完全部署的超参数（$\tau_{\text{CVC}}{=}0.7$，$\gamma{=}2.0$，$p_{\min}{=}3$）。每列共享相同的四面板结构（候选答案身份、模型自身的验证信号、最终提交决策背后的证据，以及答案区域预测熵 $H_{t}$ 对比解码进度），因此对比在面板间清晰可读。

#### Prophet（第 1 列）：正确答案多次浮现，但没有任何机制将其锁定

候选答案在第 256 步中的第 139 步门控触发之前，两次短暂命中正确值 21；Prophet 自身的 top1-top2 logit 差距根本没有机制来区分一个过路的正确猜测与任何其他瞬态猜测。差距面板显示了为什么门控 *仍然* 触发。差距本身从未显著上升；是 Prophet 自身的分阶段阈值，针对此任务系列调整（附录 A.1），衰减直到噪声差距跨越它。在那个确切时刻，门控监控的尾部窗口是 $100\%$ 掩码的（第 3 行）；其中的每个 token 都是从单次、未精炼的前向传递中一次性强制填充的，而熵面板（第 4 行）显示了为什么这是危险的；在提交的瞬间，同一窗口中仍然存在 $H_{t}{\approx}0.48$ 比特的真实不确定性，而非真正收敛所会显示的接近零的值。一次性填充落在一个截断的、乱码的续写上，根本无法提取任何数字（根据构造评分为错误），与附录 E 中真实值 $=288$ 案例（图 11 和 12）相同的失败模式，损坏的尾部文本而非干净的错误数字。这是表 1 中 $12$ – $69$ 点崩溃背后的机制，不是阈值调整不当，而是门控无法区分瞬态猜测与已验证猜测。

#### LATCH（第 2–3 列）：相同的两个机制（置信度和运行长度）恰好排除了这种失败

在 MMLU（第 2 列）上，候选答案在第 1 步时是错误的（D），在第 2 步翻转到正确答案（A），置信度从 $0.42$ 跳到 $0.83$（在与翻转相同的步骤中跨越 $\tau_{\text{CVC}}$，而非逐渐），门控在三步后、运行长度要求也得到满足时触发，在第 64 步中的第 5 步（预算的 $8\%$）；同一跨度上的熵在其触发时已经坍缩到 ${\approx}0.5$ 比特。

GSM8K（第 3 列）显示了相反的机制，并说明了第 4.1 节中刻画的晚期稳定化机制，这是 Prophet 在第 1 列中失败的相同轨迹：答案位置在轨迹的大部分时间里仍处于完全或部分掩码跨度下，因此 $c_{t}$ 在那里未定义（虚线灰色基线，第 2 行）；还没有可以验证候选答案的东西，不像 Prophet 的差距统计量，即使在纯掩码 token 上也保持定义（且噪声）。一旦跨度被填充，候选答案在稳定到正确值 21 之前确实在 $6$ 个不同值之间振荡；门控保留提交直到第 256 步中的第 178 步（预算的 $70\%$），因为 $\mathrm{run}_{t}$ 的所需阈值本身已从累积的翻转计数中增长（$\mathrm{changes}_{t}{=}17$ 在其触发时）；在轨迹早期在仅稳定性门控上看起来足够的相同运行长度在这里是不够的，因为这个轨迹已经证明自己是不可靠的。在这个完全相同的模型输出上，Prophet 提前 $39$ 步提交到一个乱码的非答案；我们的门控等待并得到正确答案。

#### 字面解码文本，而非仅仅上面的信号轨迹

图 7 显示了刚才讨论的 GSM8K 轨迹（真实值 $=21$）在每个方法自己的提交点（每列标题中的步数）下的实际解码缓冲区，而非程式化的转录。*基线* 和 *Prophet* 在 Prophet 的触发步之前共享完全相同的解码；*LATCH* 是在我们自己的门控下重放的相同轨迹。基线运行到完成并且正确；Prophet 的一次性填充落在截断的、乱码的尾部上，根本无法提取数字，而 LATCH 在预算仍未用完时正确提交。我们在 MATH 上观察到相同的机制，包括推理在整个过程中保持正确但 Prophet 的一次性填充仅损坏了最终 \\boxed{} 跨度本身的案例，这是整个答案所依赖的唯一部分；图 8 和 9 在 SVAMP 和 PIQA 上重复相同的比较，显示相同的一次性填充签名并非特定于这个单一 GSM8K 轨迹。

![参见说明](https://arxiv.org/html/2607.28166v1/x7.png)

图 7：每个方法自己提交点的字面解码文本。一个真实 GSM8K 轨迹（真实值 = 21 =21 ）的定性比较，与图 6 相同的轨迹。提取的最终答案以蓝色突出显示（正确）；Prophet 的一次性填充根本没有留下任何可提取的数字。

![参见说明](https://arxiv.org/html/2607.28166v1/x8.png)

图 8：第三个真实轨迹（SVAMP）。与图 7 相同的字面解码文本比较。当轨迹中没有任何东西清除其置信度阈值时，CVC 正确拒绝提前提交，而非无论如何强制提前退出。

![参见说明](https://arxiv.org/html/2607.28166v1/x9.png)

图 9：第四个真实轨迹（PIQA）。与图 7 相同的字面解码文本比较。即使在 Prophet 提取的字母仍然正确的情况下，其一次性填充也可能使周围文本比基线或 LATCH 更乱码，这是仅准确率指标无法捕获的格式化成本。

## 附录 E 答案区域确定：LATCH 与 Prophet 的后缀提示结构

第 4.1 节将 Prophet 的长推理失败归因于其一次性填充被允许落在的 *位置*，而非其置信度统计量本身。对 Prophet 自己的附录 C.1（[^20]）的仔细阅读揭示了第二个、复合的差距；*答案区域本身的定位方式* 在两种设置之间存在结构性差异，而不仅仅是阈值。

#### Prophet 的构造

他们的附录 C.1 指出答案区域直接内置在提示中："*后缀提示作为语义锚点插入生成窗口末尾附近，后跟为最终答案保留的掩码 token。生成的序列结构是：\[问题\] \[推理链的掩码\] 答案是 \[最终结果的掩码\]。*"对于数学推理具体来说："*数据集提供了推理链与最终结果之间的标准分隔符。我们将答案区域定义为该分隔符之后的 token。*"关键的是，这种后缀构造用于他们论文表 1 中的 *Prophet 和他们的完整步基线*。答案区域的位置从未在推理时搜索；它是一个固定的、预分配的跨度，在解码开始之前就已知。

#### 我们的构造

本文中的每个提示都是 *自由形式提示*（第 5 节，"Q: {question}\\nA: Let's think step by step."，无后缀提示，无答案位置提示）。门控没有答案的预分配位置；在每一步它必须 *搜索* 迄今为止生成的任何文本的尾部 tail\_frac 窗口，从头重新提取临时候选答案并重新定位其 token 跨度（第 4 节）。这是一个严格更难的问题；模型可以在任何长度后、任何数量的中间算术之后，在任何地方放置其最终数字答案。

#### 具体示例

表 7 显示了我们诊断缓存中的一个真实 GSM8K 轨迹（真实值 $350$）。我们的尾部窗口提取器在生成过程中遇到 *五个* 不同的数字候选答案（$130$，$100$，$120$，$220$，最后是 $350$），其中每个都曾短暂地成为"迄今为止看到的最后一个数字"，因此是门控必须评估并在真正的最终答案甚至出现之前拒绝的候选答案。在 Prophet 的后缀提示构造下，这种搜索根本不存在；最终答案掩码跨度从第 $0$ 步起就处于已知位置，因此模型在该固定跨度上的输出是唯一被监控的东西。

<table><tbody><tr><td>#</td><td>文本摘录（自由形式提示，按解码顺序）</td><td>候选答案的命运</td></tr><tr><td>1</td><td>"…Axel 拥有的比索总数：<math><semantics><mrow><mrow><mn>50</mn> <mo>+</mo> <mn>80</mn></mrow> <mo>=</mo> <mn>𝟏𝟑𝟎</mn></mrow> <annotation>50+80=\mathbf{130}</annotation></semantics></math> …"</td><td>被 #2 取代</td></tr><tr><td>2</td><td>"…Axel 的银比索的两倍：<math><semantics><mrow><mrow><mn>2</mn> <mo>×</mo> <mn>50</mn></mrow> <mo>=</mo> <mn>𝟏𝟎𝟎</mn></mrow> <annotation>2\times 50=\mathbf{100}</annotation></semantics></math> 银比索 …"</td><td>被 #3 取代</td></tr><tr><td>3</td><td>"…比 Axel 多 40 个金比索：<math><semantics><mrow><mrow><mn>80</mn> <mo>+</mo> <mn>40</mn></mrow> <mo>=</mo> <mn>𝟏𝟐𝟎</mn></mrow> <annotation>80+40=\mathbf{120}</annotation></semantics></math> 金比索 …"</td><td>被 #4 取代</td></tr><tr><td>4</td><td>"…Anna 拥有的比索总数：<math><semantics><mrow><mrow><mn>100</mn> <mo>+</mo> <mn>120</mn></mrow> <mo>=</mo> <mn>𝟐𝟐𝟎</mn></mrow> <annotation>100+120=\mathbf{220}</annotation></semantics></math> …"</td><td>被 #5 取代</td></tr><tr><td>5</td><td>"…他们一起的比索总数：<math><semantics><mrow><mrow><mn>130</mn> <mo>+</mo> <mn>220</mn></mrow> <mo>=</mo> <mn>𝟑𝟓𝟎</mn></mrow> <annotation>130+220=\mathbf{350}</annotation></semantics></math>。所以，总数…是 <math><semantics><mn>𝟑𝟓𝟎</mn> <annotation>\mathbf{350}</annotation></semantics></math>。"</td><td>最终，已提交</td></tr><tr><td colspan="3">Prophet 的构造（附录 C.1）：答案区域由提示设计固定，而非搜索</td></tr><tr><td colspan="3">[问题] [推理链的掩码] 答案是 [掩码]。最终跨度的位置是固定的，在解码开始之前就已知；候选答案 #1–#4 根本不会占据它，因为构造本身，所以 Prophet 的门控首先根本不必将它们与 #5 区分开。</td></tr></tbody></table>

表 7：在真实答案之前拒绝的五个候选答案。同一个 GSM8K 示例（真实值 $=350$）。我们的尾部窗口提取器依次遇到并必须评估 *五个* 数字候选答案，四个瞬态的（#1–#4，每个都曾短暂地成为"迄今为止看到的最后一个数字"直到被覆盖）和一个最终的（#5，存活到生成结束并被提交），而 Prophet 的结构性预分配跨度，根据构造只包含 #5，且根本不必首先做出这种区分。

这正是 CVC 的置信度加运行长度要求（第 4 节）为了在不需要后缀提示的情况下生存而存在的；像 $130$ 或 $220$ 这样的候选答案曾短暂地成为"答案"，甚至可能保持 $\mathrm{run}\geq 1$ 一两步，但它与四个后续重写竞争，因此它很少累积通过门控所需的 argmax 稳定性和置信度，而 Prophet 的 $\bar{g}_{t}$ 在我们的复现中（附录 A.1）没有等效的保护，监控相同的自由形式缓冲区且根本没有位置锚点。
#### Prophet 自身的后缀提示构造能否拯救它？

我们复现了这一构造并直接测试它。最终块的缓冲区从步骤 $0$ 开始用字面 token id " The answer is" 在固定位置初始化（从不被掩码，从不受任何门控影响），该块的其余位置纯粹保留给数值结果，这是对附录 C.1 的 \[Question\] \[MASKs for Reasoning Chain\] The answer is \[MASKs for Final Result\] 结构的精确复现，相同地应用于基线、Prophet 和 *LATCH*，以便对比能够孤立门控本身。表 8 报告了在 $n{=}100$ 个保留的 GSM8K 和 MATH 样例上的结果，两个模型，Prophet 使用其自身校正的每任务阈值（[^20] 的表 6）。即使答案区域的位置不再是模型或门控需要找到的东西（Prophet 自身的构造，在其自身的阈值下），Prophet 的准确率在 GSM8K 上仍然大幅崩溃（LLaDA $68.0\%\!\to\!41.0\%$，$-27$ 点；Dream $72.0\%\!\to\!54.0\%$，$-18$ 点），在 MATH 上严重退化（LLaDA $31.0\%\!\to\!24.0\%$，$-7$ 点；Dream $43.0\%\!\to\!28.0\%$，$-15$ 点）；*LATCH* 在两个任务和两个模型上保持下降可忽略。这驳斥了后缀提示结构本身能够解释 Prophet 自身论文中可忽略下降声明的假设；差距在于**什么算作收敛的证据**，而不仅仅是**答案区域在哪里**，答案位置的硬结构锚点并不能防止 Prophet 的位置级置信度差距触发器在候选值稳定之前触发。

Prophet 的后缀提示构造，最终块（$256$ 中的位置 $224$ – $255$，GSM8K）：
\[pos 224\] Theansis \[pos 255\]
从步骤 $0$ 固定为 " The answer is"，从不掩码    自由/掩码，模型填充

我们的自由形式提示协议，相同最终块，完全没有保留：
\[pos 224\] \[pos 255\]
每个位置都从掩码开始；模型决定答案在何处以及如何浮现，我们的提取器事后定位它（任务格式特定的搜索区域，附录 A.2）

图 10：Prophet 的后缀提示构造与我们的对比。后缀提示构造（上）从步骤 $0$ 开始在固定位置固定字面 token " The answer is"，因此答案的**位置**从来不是问题；只有剩余的 $29$ 个位置是自由的。我们的自由形式提示协议（下）不保留任何内容：每个块中的每个位置都从掩码开始，与**基线**完全相同。

<table><thead><tr><th></th><th></th><th colspan="4">LLaDA-8B-Instruct</th><th colspan="4">Dream-7B-Instruct</th></tr><tr><th>Task</th><th>Variant</th><th>Acc (%)</th><th>Avg. Step</th><th>TPS</th><th>Speedup</th><th>Acc (%)</th><th>Avg. Step</th><th>TPS</th><th>Speedup</th></tr></thead><tbody><tr><th>GSM8K</th><th>Baseline</th><td>68.0</td><td>256.0</td><td>22.5</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>72.0</td><td>256.0</td><td>23.5</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=100, sequence length=256)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>41.0</td><td>194.0</td><td>25.4</td><td>1.13 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>54.0</td><td>176.3</td><td>28.7</td><td>1.22 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=32, 8 blocks)</th><th>LATCH</th><td>67.0</td><td>252.1</td><td>22.4</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>72.0</td><td>252.7</td><td>23.5</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>MATH</th><th>Baseline</th><td>31.0</td><td>256.0</td><td>21.3</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>43.0</td><td>256.0</td><td>22.2</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(n=100, sequence length=256)</th><th>Prophet <sup><a href="#fn:20">20</a></sup></th><td>24.0</td><td>199.4</td><td>23.5</td><td>1.10 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>28.0</td><td>194.7</td><td>24.4</td><td>1.10 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><th>(block=32, 8 blocks)</th><th>LATCH</th><td>31.0</td><td>252.4</td><td>21.2</td><td>1.00 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>43.0</td><td>253.8</td><td>21.7</td><td>0.98 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr></tbody></table>

表 8：Prophet 在固定答案区域的情况下仍然脆弱。GSM8K/MATH，使用 Prophet 自身的后缀提示缓冲区构造（附录 C.1）相同地应用于所有三个变体；答案区域的位置从步骤 $0$ 开始固定且已知，消除了表 7 所示的搜索问题。*LATCH* 在这里基本上是 $1.00\times$，因为后缀提示几乎没有留下任何内容供 CVC 缩短。Prophet 的准确率在两个任务和两个模型上仍然严重退化；*LATCH* 始终保持下降可忽略。

一个单独的、较弱的探测（仅固定一个 "Answer:" 锚定 token，而不是完整的后缀提示）将 GSM8K 的早期收敛分布朝 Prophet 的图 1 预测的方向移动，但在没有 Prophet 附带的简洁指令的情况下损失了最终任务准确率（$30/40\!\to\!25/40$）。这得出了与上述相同的结论；仅定位答案区域并不能使门控安全。

#### 在 Prophet 自身后缀提示构造下的字面文本

图 11 和图 12 展示了表 8 数字背后两条轨迹的字面解码文本，而不仅仅是准确率；即使答案区域的位置从步骤 $0$ 开始固定且从不掩码，Prophet 的一次性填充仍然在两个样例上用损坏的值覆写了保留的数值跨度本身，而 *LATCH* 运行到与**基线**相同的完整预算并精确复现它。

![Refer to caption](https://arxiv.org/html/2607.28166v1/x10.png)

图 11：Prophet 自身后缀提示构造下的字面文本（GSM8K，气球样例）。表 8 数字背后的一条轨迹，Prophet 自身的后缀提示构造（附录 C.1）相同地应用于所有三个变体。基线和 LATCH 在相同的完整步数预算下达到相同的正确答案；Prophet 的一次性填充用与正确答案无关的损坏值覆写了固定的答案跨度本身，在文本级别确认固定答案区域的位置本身并不能防止过早提交。

![Refer to caption](https://arxiv.org/html/2607.28166v1/x11.png)

图 12：Prophet 自身后缀提示构造下的字面文本（GSM8K，债务样例）。表 8 数字背后的第二条轨迹，与图 11 相同的构造和变体。Prophet 的一次性填充再次用损坏的值覆写了固定的答案跨度；基线和 LATCH 达到相同的正确答案。

### E.1 任务格式特定的答案搜索

实现使用两个固定的搜索区域，一个用于长形式推理的尾部窗口和一个用于短答案任务的前五个位置（复现配置中的 search\_mode，附录 A.2）。上述所有内容都是**长推理**搜索设置（"last"）；门控搜索尾部窗口，因为思维链答案可以出现在任何地方，任意晚。短答案/多选任务（MMLU、ARC-C、HellaSwag、WinoGrande、PIQA、TruthfulQA）使用另一个设置 "first5"，原因直接在提示模板本身中可见，而不是一个单独的设计选择。

<svg id="A5.SS1.p2.pic1" height="56.15" overflow="visible" version="1.1" viewBox="0 0 600 56.15" width="600"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,56.15) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#CCCCCC;" fill="#CCCCCC" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 56.15 L 600 56.15 L 600 0 Z"></path></g><g style="--ltx-fill-color:#F5F5F5;" fill="#F5F5F5" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 0.55 L 0.55 33.33 L 599.45 33.33 L 599.45 0.55 Z"></path></g><g style="--ltx-fill-color:#DFDFDF;" fill="#DFDFDF" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 33.89 L 0.55 55.6 L 599.45 55.6 L 599.45 33.89 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 12.79 41.28)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:41.51em;--ltx-fo-height:0.75em;--ltx-fo-depth:0.25em;" width="574.41" height="13.84" transform="matrix(1 0 0 -1 0 10.38)" overflow="visible" color="#000000"><span id="A5.SS1.p2.pic1.1.1.1.1.1" style="width:36.1em;"><span id="A5.SS1.p2.pic1.1.1.1.1.1.1"><span id="A5.SS1.p2.pic1.1.1.1.1.1.1.1">Prompt template, long-reasoning (search mode last)</span></span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 12.79 13.48)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:41.51em;--ltx-fo-height:0.75em;--ltx-fo-depth:0.25em;" width="574.41" height="13.84" transform="matrix(1 0 0 -1 0 10.38)" overflow="visible" color="#000000"><span id="A5.SS1.p2.pic1.2.2.2.1.1" style="width:39.54em;"><span id="A5.SS1.p2.pic1.2.2.2.1.1.1"><span id="A5.SS1.p2.pic1.2.2.2.1.1.1.1">Q: {question}\nA: Let's think step by step.</span></span></span></foreignObject></g></g></svg>

<svg id="A5.SS1.p3.pic1" height="56.15" overflow="visible" version="1.1" viewBox="0 0 600 56.15" width="600"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,56.15) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#CCCCCC;" fill="#CCCCCC" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 56.15 L 600 56.15 L 600 0 Z"></path></g><g style="--ltx-fill-color:#F5F5F5;" fill="#F5F5F5" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 0.55 L 0.55 33.33 L 599.45 33.33 L 599.45 0.55 Z"></path></g><g style="--ltx-fill-color:#DFDFDF;" fill="#DFDFDF" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 33.89 L 0.55 55.6 L 599.45 55.6 L 599.45 33.89 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 12.79 41.28)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:41.51em;--ltx-fo-height:0.75em;--ltx-fo-depth:0.25em;" width="574.41" height="13.84" transform="matrix(1 0 0 -1 0 10.38)" overflow="visible" color="#000000"><span id="A5.SS1.p3.pic1.1.1.1.1.1" style="width:36.1em;"><span id="A5.SS1.p3.pic1.1.1.1.1.1.1"><span id="A5.SS1.p3.pic1.1.1.1.1.1.1.1">Prompt template, short-answer/MC (search mode first5)</span></span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 12.79 13.48)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:41.51em;--ltx-fo-height:0.75em;--ltx-fo-depth:0.25em;" width="574.41" height="13.84" transform="matrix(1 0 0 -1 0 10.38)" overflow="visible" color="#000000"><span id="A5.SS1.p3.pic1.2.2.2.1.1" style="width:39.54em;"><span id="A5.SS1.p3.pic1.2.2.2.1.1.1"><span id="A5.SS1.p3.pic1.2.2.2.1.1.1.1">{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:</span></span></span></foreignObject></g></g></svg>

思维链模板在思考中途结束，这是有意为之；模型必须写出通往答案的路径，因此门控没有固定的地方可以查看，必须搜索。多选模板在字面词 "Answer:" 上结束，这本身就是一个位置锚点；经过指令微调的模型的下一个 token 就是字母，而不是更多推理。这种任务格式特定的搜索恰好是第 4 节所声称的答案**感知**：它不是一个全局窗口，它是对**给定用于引出答案的提示，答案在结构上必须存在的位置**的每任务编码，上述两个设置直接从上面的两个模板中读取，而不是单独调优的。

#### 具体示例

从我们的诊断缓存中的一个真实 MMLU 轨迹（真实标签 A，从校准中保留）生成：

<svg id="A5.SS1.SSS0.Px1.p2.pic1" height="88.98" overflow="visible" version="1.1" viewBox="0 0 600 88.98" width="600"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,88.98) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#CCCCCC;" fill="#CCCCCC" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 88.98 L 600 88.98 L 600 0 Z"></path></g><g style="--ltx-fill-color:#F5F5F5;" fill="#F5F5F5" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 0.55 L 0.55 66.16 L 599.45 66.16 L 599.45 0.55 Z"></path></g><g style="--ltx-fill-color:#DFDFDF;" fill="#DFDFDF" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 66.71 L 0.55 88.42 L 599.45 88.42 L 599.45 66.71 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 12.79 74.11)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:41.51em;--ltx-fo-height:0.75em;--ltx-fo-depth:0.25em;" width="574.41" height="13.84" transform="matrix(1 0 0 -1 0 10.38)" overflow="visible" color="#000000"><span id="A5.SS1.SSS0.Px1.p2.pic1.1.1.1.1.1" style="width:36.1em;"><span id="A5.SS1.SSS0.Px1.p2.pic1.1.1.1.1.1.1"><span id="A5.SS1.SSS0.Px1.p2.pic1.1.1.1.1.1.1.1">Model output (LLaDA-8B-Instruct, greedy, first 200 characters)</span></span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 12.79 13.1)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:41.51em;--ltx-fo-height:3.15em;--ltx-fo-depth:0.22em;" width="574.41" height="46.66" transform="matrix(1 0 0 -1 0 43.59)" overflow="visible" color="#000000"><span id="A5.SS1.SSS0.Px1.p2.pic1.2.2.2.1.1" style="width:39.54em;"><span id="A5.SS1.SSS0.Px1.p2.pic1.2.2.2.1.1.1"><span id="A5.SS1.SSS0.Px1.p2.pic1.2.2.2.1.1.1.1">A. Antidiuretic hormone.\n\nAntidiuretic hormone, also known as vasopressin, is primarily responsible for fluid regulation. It is produced by the posterior pituitary gland and helps regulate water balan[…]</span></span></span></foreignObject></g></g></svg>

字母在位置 $0$ 就已经正确且完全形成，在模型写出单个词的（未请求的、未奖励的）阐述之前。表 1 的 $\tau_{\text{CVC}}$/运行长度检查只需要查看这第一个 token："first5" 在生成缓冲区的前导 $5$ 个位置中搜索匹配，立即找到 "A"，门控在 $c_{t}\geq\tau_{\text{CVC}}$ 对 $\mathrm{run}_{t}\geq p_{\min}$ 步成立后立即提交，通常在前几个去噪步内（表 2：MMLU 自身的 *+CVC* 行在 $64$ 步预算中的平均步骤 $\approx\!5$ – $9$ 提交）。"last" 模式为思维链任务搜索尾部窗口；在这里做同样的事情反而是主动错误的，因为答案后的阐述是模型从未被要求的自由运行评论，门控从未被设计来评分，并且没有任何东西保证尾部窗口可见的第二次字母重述存在（上面的示例在可见窗口中从未再次重复 "A"）。因此，两个搜索设置不是实现上的便利；它们对每个任务族编码了一个结构事实，这是使任一族的提前提交问题完全可处理的唯一事实，**提示本身保证答案将被写在哪里**。

## 附录 F 消融研究

### F.1 CVC 组件消融：哪个信号是承重的？

公式 4 要求两个条件在 CVC 提交之前共同成立，置信度（$c_{t}\geq\tau_{\text{CVC}}$）和自适应稳定性（$\mathrm{run}_{t}\geq\max(p_{\min},\lceil\gamma\cdot\mathrm{changes}_{t}\rceil)$）。我们通过对用于图 3 的相同保留诊断池重放四个门控变体来隔离哪个条件是承重的（$n{=}60$，与用于校准 CVC 的 $120$ 条轨迹不相交，MMLU/GSM8K/MATH，LLaDA-8B-Instruct），即**完整 CVC**（两个条件，部署的超参数）；**仅置信度**（移除稳定性条件）；**仅稳定性**（移除置信度条件）；以及**固定耐心**（保留置信度，但用常数 $p_{\min}$ 替换自适应运行长度门槛）。这种重放完全是针对缓存轨迹的离线操作，因为完整的、非提前退出的解码已经包含了这些门控中的任何一个会看到的所有内容（附录 A.1）。

| Task ($n$) | Baseline Acc | Full CVC | Confidence only | Stability only | Fixed patience |
| --- | --- | --- | --- | --- | --- |
| MMLU (20) | 64.0 | +0.0 (7.9%) | +0.0 (6.4%) | +0.0 (6.1%) | +0.0 (7.8%) |
| GSM8K (20) | 65.0 | +0.0 (97.5%) | +0.0 (85.4%) | \-25.0 (31.6%) | +0.0 (85.7%) |
| MATH (20) | 55.0 | \-5.0 (97.0%) | \-25.0 (87.0%) | \-25.0 (7.4%) | \-10.0 (88.2%) |

表 9：CVC 组件消融。每个单元格：相对于**基线**的准确率变化（点），括号中为使用的平均解码步数（预算的 %）；粗体标记每行的最佳变体，阴影标记准确率成本（黄色：无；红色：越深越差）。

表 9 显示没有单个消融变体在所有三个任务上都是安全的。**仅稳定性**在两个长推理任务上灾难性失败（GSM8K 上 $-25.0$ 点，MATH 上 $-25.0$ 点）；没有置信度检查，一个仅仅停止变化几步的候选答案就会被接受，无论模型仍然有多不确定，这正是仅稳定性规则无法区分低置信度平台与真正收敛的失败模式。**仅置信度**在 GSM8K 上是安全的，但在 MATH 上同样严重失败（$-25.0$ 点）；没有稳定性检查，一个瞬间置信但尚未停止翻转的候选答案就会被接受，这会咬到哪个任务取决于模型在解码后期重新访问错误候选答案的频率。**固定耐心**在 GSM8K 上保持安全，但在 MATH 上退化（$-10.0$ 点，比**完整 CVC** 在同一池上的 $-5.0$ 点更差）；平坦的耐心底线对于候选答案持续翻转到很晚的轨迹来说不够保守，这正是自适应项 $\lceil\gamma\cdot\mathrm{changes}_{t}\rceil$ 旨在提高门槛的对象。**完整 CVC** 是唯一在所有三个任务上接近安全的变体。MATH 上的 $-5.0$ 点偏差对应于这个 $n{=}20$ 诊断池中的一个样例；表 2 中的 $n{=}100$ 仅 CVC 结果显示没有准确率变化。MMLU 稳定得足够早（预算的 $6$ – $8\%$），以至于所有四个变体在那里都是安全的，与第 4.1 节的稳定化差距发现一致；消融任一条件只有在任务的答案需要很长时间才能稳定时才重要。

#### 相同信号，错误对象

上述消融一次移除 CVC 的两个条件之一；一个更尖锐的问题是这些信号是否在**没有候选答案**的情况下足够，因为跨步 argmax 稳定性正是自适应采样器跟踪的每位置证据（第 2 节）。因此，我们针对完整表 1 保留轨迹重放一个无候选答案的控制门控（$n{=}200/100/100$，LLaDA），在每个监控位置的 argmax 连续 $k$ 步保持不变的第一步用一次性填充终止，监控范围要么是整个缓冲区，要么更宽容地仅监控块调度已经到达的位置（未到达块中的预测根据构造是噪声）。表 10 在两个范围下扫描 $k$。没有冻结设置既安全又有用。在所有三个任务上都在 $2.0$ 点容差内的唯一设置（整个缓冲区，$k\geq 8$）在 MMLU 上在预算的 $56.9$ – $74.6\%$ 提交，比部署的 CVC 的 $10.9\%$ 晚五到七倍，同时在 GSM8K/MATH 上不节省任何东西；每个匹配 CVC 的早期 MMLU 提交点的设置在长推理任务上超过容差高达 $57$ 点，而宽容范围恰恰因为它更早触发而成为更糟糕的违规者。门控无法判断它处于哪种状态；位置级证据从不识别答案所在的位置，这正是候选答案提取添加的信息，也是第 4.1 节的稳定化时间差距所要求的。

| Monitored region | $k$ | MMLU ($n{=}200$) | GSM8K ($n{=}100$) | MATH ($n{=}100$) |
| --- | --- | --- | --- | --- |
| whole buffer | $2$ | \-2.0 (24.3%) | \-14.0 (82.5%) | \-8.0 (82.3%) |
| whole buffer | $4$ | \-0.5 (45.4%) | \-2.0 (95.9%) | +1.0 (96.7%) |
| whole buffer | $8$ | \-0.5 (56.9%) | +0.0 (98.6%) | +1.0 (98.5%) |
| whole buffer | $16$ | +0.0 (74.6%) | +0.0 (99.9%) | +1.0 (99.3%) |
| reached blocks | $2$ | \-2.5 (7.1%) | \-57.0 (5.0%) | \-23.0 (5.0%) |
| reached blocks | $4$ | \-0.5 (22.8%) | \-53.0 (24.3%) | \-23.0 (18.1%) |
| reached blocks | $8$ | +0.0 (47.9%) | \-17.0 (77.6%) | \-7.0 (54.6%) |
| reached blocks | $16$ | +0.0 (74.0%) | \-3.0 (97.6%) | +1.0 (87.1%) |
| CVC (deployed) | – | +0.0 (10.9%) | +0.0 (97.7%) | +0.0 (98.7%) |

表 10：位置稳定性终止控制门控。每个单元格：相对于**基线**的准确率变化（点），括号中为使用的平均解码步数（预算的 %），在表 1 保留分割上重放；门控在每个监控位置的 argmax 连续 $k$ 步保持不变后停止并填充。CVC 行（来自表 2）是候选答案感知的参考。阴影如表 9。$+1.0$ 的 MATH 单元格高于基线是因为一个单一的评分边界样例（重放的完整解码评分为 $32.0$，而表 1 为 $31.0$）。

### F.2 结构消融：BWEC 统一应用

#### 范围和注意事项

最终块保护不应被解读为假设答案位于最终块；答案局部性在任务和骨干网络之间差异很大（见下文）。其目的反而是分离具有不同失败成本的动作。BWEC 仅推进当前块，而 CVC 的填充并停止动作终止整个序列并使所有剩余预测不可逆。因此，LATCH 允许在非最终块中激进的局部提交，但将全局终止保留给最终块，在那里它以候选答案的身份和时间稳定性为条件。这种不对称性，而不是最终答案局部性本身，是预期的安全机制。

关于公式 5 对非最终块的范围限定，自然会产生一个问题。最终/非最终区分本身是承重的吗，还是 BWEC 的阈值加调度规则应用到**任何地方**都会是安全的，使 CVC 的单独最终块门控成为冗余的安全边际？一个相关的、更窄的检查确认置信度信号本身从来不是问题。部署的 CVC 在每一步更新其候选答案统计数据，但其全局填充并停止动作仅在解码进入最终块后才变得符合条件；我们将改为从**任何块**使这个相同的填充并停止动作符合条件的消融称为**任意块符合条件的 CVC 提交**；它从**全局**提交决策中移除了最终块范围限定，而不是从 BWEC 的局部规则中（第 4.3 节）。这使得一个新的保留 GSM8K 下降到 $62.0\%$，MATH 下降到 $27.0\%$（两者都是 $-4.0$ 点）；它的失败方式与 Prophet 完全相同，因为允许早期的一次性提交变得符合条件的**位置**才是重要的，而不是用于证明它的统计数据。

#### 答案实际上多久落在最终块？

在 LLaDA 的分块调度下，最终答案在五个长推理任务的保留样例中有 $43.6$ – $94.7\%$ 落在最后一块（SVAMP 上最低），在 Dream 上远低且预测性更差（GSM8K 上低至 $2.6\%$，尽管如此仅显示可忽略的准确率下降），这是一个真实的、依赖于模型的风险因素，从来不是将全局提交决策限定到最终块的唯一论据。SlowFast 采样 [^38] 从相反的方向省略了这种区分（没有最终/非最终分割），在长推理上准确率仍然严重退化的同时，留下了短答案加速的空间（第 5 节）；一个无差别的规则不能同时做到最大激进**和**最大谨慎。

## 附录 G 敏感性

### G.1 离散 $\tau_{\text{BWEC}}$ 层级#### 两个未部署的 τBWEC\tau\_{\text{BWEC}} 层级：准确率/加速比权衡，而非第二个安全默认值

表 1 仅报告了已部署层级（$\tau_{\text{BWEC}}{=}0.9$），这是保守范围的端点，使每个单元格都保持在论文自身的 $\pm 2.0$ 点容差内。表 11 给出了两个更宽松的未部署层级（$\tau_{\text{BWEC}}{=}0.7$ 和 $\tau_{\text{BWEC}}{=}0.8$）的相同细分，它们的存在仅用于表征权衡可以推进到何种程度，而非作为推荐的替代方案，并且不需要查询时选择，因为 LATCH 部署的仅有 $\tau_{\text{BWEC}}{=}0.9$；这里 $44$ 个任务 $\times$ 模型 $\times$ 层级组合中有 $6$ 个超出了相同的容差。这是设计上对更宽松层级的预期，而非方法的失败；它们以安全边际换取速度，应被理解为灵敏度扫描，而非第二个部署就绪的配置。

<table><tbody><tr><td></td><td></td><td colspan="4">LLaDA-8B-Instruct</td><td colspan="4">Dream-7B-Instruct</td></tr><tr><td>任务</td><td>层级</td><td>准确率 (%)</td><td>平均步数</td><td>TPS</td><td>加速比</td><td>准确率 (%)</td><td>平均步数</td><td>TPS</td><td>加速比</td></tr><tr><td colspan="10">通用/短答案任务（单 token 或短跨度答案）</td></tr><tr><td>MMLU</td><td>激进</td><td>63.5</td><td>4.6</td><td>405.9</td><td>13.81 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>70.0</td><td>4.2</td><td>457.7</td><td>13.87 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=200, 序列长度=64, 块=16)</td><td>正常</td><td>63.5</td><td>5.1</td><td>399.7</td><td>13.60 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>70.5</td><td>4.5</td><td>520.1</td><td>15.76 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>ARC-C</td><td>激进</td><td>86.0</td><td>3.8</td><td>581.9</td><td>19.02 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>88.5</td><td>3.7</td><td>554.3</td><td>15.57 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=200, 序列长度=64, 块=16)</td><td>正常</td><td>86.0</td><td>3.8</td><td>481.7</td><td>15.74 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>87.5</td><td>4.1</td><td>547.2</td><td>15.37 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>HellaSwag</td><td>激进</td><td>76.0</td><td>4.2</td><td>353.1</td><td>14.78 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>75.5</td><td>5.4</td><td>323.2</td><td>13.08 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=200, 序列长度=64, 块=16)</td><td>正常</td><td>76.5</td><td>4.3</td><td>333.4</td><td>13.95 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>75.0</td><td>5.9</td><td>274.1</td><td>11.10 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>WinoGrande</td><td>激进</td><td>75.5</td><td>3.3</td><td>580.5</td><td>18.67 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>71.0</td><td>3.8</td><td>710.4</td><td>18.40 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=200, 序列长度=64, 块=16)</td><td>正常</td><td>75.5</td><td>3.3</td><td>575.6</td><td>18.51 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>71.0</td><td>4.0</td><td>682.2</td><td>17.67 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>PIQA</td><td>激进</td><td>81.0</td><td>4.0</td><td>407.3</td><td>13.10 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>84.5</td><td>3.5</td><td>681.0</td><td>18.81 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=200, 序列长度=64, 块=16)</td><td>正常</td><td>81.5</td><td>3.6</td><td>455.0</td><td>14.63 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>84.0</td><td>3.5</td><td>681.7</td><td>18.83 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>TruthfulQA</td><td>激进</td><td>62.0</td><td>4.6</td><td>489.0</td><td>15.72 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>64.0</td><td>4.8</td><td>328.3</td><td>9.38 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=200, 序列长度=64, 块=16)</td><td>正常</td><td>63.0</td><td>4.4</td><td>464.3</td><td>14.93 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>63.0</td><td>4.9</td><td>355.6</td><td>10.16 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td colspan="10">长推理任务（多步思维链）</td></tr><tr><td>GSM8K</td><td>激进</td><td>67.0</td><td>69.3</td><td>78.5</td><td>3.54 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>77.0</td><td>72.7</td><td>81.6</td><td>3.53 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=100, 序列长度=256, 块=32)</td><td>正常</td><td>69.0</td><td>79.3</td><td>65.9</td><td>2.97 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>84.0</td><td>80.6</td><td>75.5</td><td>3.27 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>MATH</td><td>激进</td><td>32.0</td><td>95.0</td><td>55.6</td><td>2.66 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>39.0</td><td>92.0</td><td>59.6</td><td>2.73 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=100, 序列长度=256, 块=32)</td><td>正常</td><td>32.0</td><td>107.6</td><td>49.4</td><td>2.36 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>41.0</td><td>104.0</td><td>53.0</td><td>2.43 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>SVAMP</td><td>激进</td><td>85.0</td><td>63.1</td><td>93.7</td><td>4.18 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>78.0</td><td>64.4</td><td>90.6</td><td>3.86 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=100, 序列长度=256, 块=32)</td><td>正常</td><td>85.0</td><td>75.1</td><td>77.3</td><td>3.45 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>78.0</td><td>70.5</td><td>83.6</td><td>3.56 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>ASDiv</td><td>激进</td><td>63.0</td><td>69.0</td><td>85.6</td><td>3.79 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>78.0</td><td>67.4</td><td>94.8</td><td>4.03 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=100, 序列长度=256, 块=32)</td><td>正常</td><td>64.0</td><td>78.9</td><td>75.2</td><td>3.33 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>79.0</td><td>73.4</td><td>86.6</td><td>3.68 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>GSM-Hard</td><td>激进</td><td>33.0</td><td>72.3</td><td>78.3</td><td>3.61 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>40.0</td><td>80.0</td><td>73.7</td><td>3.20 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr><tr><td>(n=100, 序列长度=256, 块=32)</td><td>正常</td><td>33.0</td><td>83.3</td><td>68.2</td><td>3.14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>41.0</td><td>90.6</td><td>64.5</td><td>2.80 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td></tr></tbody></table>

表 11：两个未部署的 $\tau_{\text{BWEC}}$ 层级，全部 11 个任务。激进层级（$\tau_{\text{BWEC}}{=}0.7$）和正常层级（$\tau_{\text{BWEC}}{=}0.8$），全部 $11$ 个任务，两个模型，与表 1 相同的 TPS 比率加速比约定；已部署的 $\tau_{\text{BWEC}}{=}0.9$ 层级在表 1 中。

#### 先验选择，仅在事后扫描

$\tau_{\text{BWEC}}=0.9$ 不是在这里报告的保留样本上调优的；我们先验地选择它作为高于最终门控自身 $\tau_{\text{CVC}}=0.7$ 的保守置信水平，并且仅在事后在不相交的保留样本（$n{=}50$，GSM8K 和 MATH）上扫描它。GSM8K 在 $\tau_{\text{BWEC}}\geq 0.85$ 时达到平台期，而 MATH 随着阈值放宽而平滑退化；$0.9$ 正好位于 GSM8K 拐点之后。

### G.2 每块和总步数行为，全部任务

#### 相同的测量，扩展到全部 11 个任务和两个模型

正文中的图 5 展示了 $4$ 任务/仅 LLaDA 的子集；图 13-20 在完整的 $11$ 任务、$2$ 模型矩阵上重复了相同的协议，在整个范围内确认了相同的两种模式。最终块保持接近完整预算，而非最终块逐渐清空（图 14），准确率在短答案任务上保持平稳，而长推理任务显示出随着 $\tau_{\text{BWEC}}$ 放宽而出现的相同平滑崩溃（图 19-20），正是第 4.1 节所表征的二分法。

![参见说明](https://arxiv.org/html/2607.28166v1/x12.png)

图 13：每块步数使用，全部短答案任务。全部六个通用/短答案任务，两个模型（与图 5 相同的约定；LLaDA-8B-Instruct 在上，Dream-7B-Instruct 在下）。在 LATCH 下，全部六个短答案任务在两个模型上都在一到两个块内完成。

![参见说明](https://arxiv.org/html/2607.28166v1/x14.png)

图 14：每块步数使用，全部长推理任务。全部五个长推理任务，两个模型（与图 5 相同的约定；LLaDA-8B-Instruct 在上，Dream-7B-Instruct 在下）。最终块保持接近完整预算的模式在整个范围内保持；在 LATCH 下，BWEC 的非最终块节省逐块增长，而 Prophet 和 SlowFast 则不然。

#### 相同的每块使用，汇总为每种方法一个条形

图 15 和 16 将相同的每块分段端到端排列成每种方法的单个条形，准确率显示为下方较细的条形，以便可以对照步数节省是否以准确率为代价进行阅读。在已部署的 $\tau_{\text{BWEC}}{=}0.9$ 下，LATCH 的准确率条形在每个短答案任务上都保持在 *Baseline* 的容差范围内，无论步数条形的长度如何，尽管更宽松的层级和同行方法可能偏离更多（例如 TruthfulQA/Dream SlowFast，$-7.5$ 点），而在长推理任务上，即使 Prophet 和 SlowFast 的步数条形与 LATCH 的相匹配，它们的准确率条形也明显缩小，这与表 1 报告为数字的相同严重下降模式。

![参见说明](https://arxiv.org/html/2607.28166v1/x16.png)

图 15：总步数和准确率，全部短答案任务。使用的总步数（粗条形）和准确率（下方细条形），全部六个通用/短答案任务，两个模型（LLaDA-8B-Instruct 在上，Dream-7B-Instruct 在下）；与图 13 相同的每块数据，连接成每种方法一个条形。在已部署的 τBWEC=0.9\tau\_{\text{BWEC}}{=}0.9 下，LATCH 在整个范围内保持在容差内；更宽松的层级和同行方法可能偏离。

![参见说明](https://arxiv.org/html/2607.28166v1/x18.png)

图 16：总步数和准确率，全部长推理任务。使用的总步数（粗条形）和准确率（下方细条形），全部五个长推理任务，两个模型（LLaDA-8B-Instruct 在上，Dream-7B-Instruct 在下）；与图 14 相同的每块数据，连接成每种方法一个条形。相对于 Baseline，即使 Prophet 和 SlowFast 的步数条形在长度上与 LATCH 的相当，它们的准确率条形也明显缩小。

#### 每块时钟份额：哪个块实际上是瓶颈

图 17 和 18 按*时间*而非步数分解相同的数据；分段宽度是该块的时钟时间份额，从直接测量的聚合 TPS 按步数份额分配（表 1），而非独立的每块测量。这使"最终填充"变得清晰；在长推理任务上，最终块的时间份额在 *LATCH* 下向整个条形增长，仅仅是因为其他每个块都被削减得如此之多，以至于其固定成本现在占主导地位。

![参见说明](https://arxiv.org/html/2607.28166v1/x20.png)

图 17：每块时钟份额和估计 TPS，全部短答案任务。每块时间份额（分段宽度）和估计的每块 TPS（分段标签），全部六个通用/短答案任务，两个模型（LLaDA-8B-Instruct 在上，Dream-7B-Instruct 在下）；最终块用黑色轮廓标出。分段级 TPS 从直接测量的聚合 TPS（粗体，在条形末端）按步数份额分配；详见上文段落的精确推导及其每步成本均匀假设。

![参见说明](https://arxiv.org/html/2607.28166v1/x22.png)

图 18：每块时钟份额和估计 TPS，全部长推理任务。每块时间份额（分段宽度）和估计的每块 TPS（分段标签），全部五个长推理任务，两个模型（LLaDA-8B-Instruct 在上，Dream-7B-Instruct 在下）；最终块用黑色轮廓标出，随着 τBWEC\tau\_{\text{BWEC}} 收紧在 LATCH 下向整个条形增长；详见上文段落的原因。

### G.3 连续 τBWEC\tau\_{\text{BWEC}} 扫描

![参见说明](https://arxiv.org/html/2607.28166v1/x24.png)

图 19：连续 τBWEC\tau\_{\text{BWEC}} 扫描，全部短答案任务。全部六个通用/短答案任务，两个模型（LLaDA-8B-Instruct 在上，Dream-7B-Instruct 在下）。实线曲线显示准确率（左轴，Wilson 分数 95%95\% 置信区间）；虚线曲线显示平均步数；点划线曲线显示相对于每个面板自身 τBWEC=0.9\tau\_{\text{BWEC}}{=}0.9 值的 TPS 比率加速比，在每个 τBWEC\tau\_{\text{BWEC}} 处独立测量（附录 B）。准确率在已部署范围附近通常稳定，而更激进的阈值会降低几个任务的性能。

![参见说明](https://arxiv.org/html/2607.28166v1/x26.png)

图 20：连续 τBWEC\tau\_{\text{BWEC}} 扫描，全部长推理任务。全部五个长推理任务，两个模型（LLaDA-8B-Instruct 在上，Dream-7B-Instruct 在下）。实线曲线显示准确率（左轴，Wilson 分数 95%95\% 置信区间）；虚线曲线显示平均步数；点划线曲线显示相对于每个面板自身 τBWEC=0.9\tau\_{\text{BWEC}}{=}0.9 值的 TPS 比率加速比，在每个 τBWEC\tau\_{\text{BWEC}} 处独立测量（附录 B）。跨任务，放宽最终会导致准确率大幅下降，而步数和 TPS 比率不提供可靠的预警。

#### 过度放宽 τBWEC\tau\_{\text{BWEC}} 可能损失速度，而非仅仅准确率

BWEC 的强制提交规则单独读来是单调的；降低 $\tau_{\text{BWEC}}$ 应该只会使提交更容易，绝不会需要*更多*步数。表 1 和表 11 中的三层级比较显示，这在两个模型下的大多数任务上成立，但并非全部；图 19-20 连续追踪了机制（第 5.2 节，过早提交破坏了后续块所依赖的上下文，导致它们消耗更多步数）。在 HellaSwag/LLaDA 上，步数正是因为这个原因呈 U 形，然而 TPS 比率的峰值在别处；在 HellaSwag/Dream 上，步数保持平稳，而 TPS 比率在 $2.2$ – $2.8\times$ 之间摆动。在 GSM8K 上，准确率可能急剧下降，而步数完全没有相应的警告（$67\%\!\to\!3\%$ LLaDA，$85\%\!\to\!5\%$ Dream），这是更危险的失败形态，因为步数或 TPS 本身的任何信息都不会标记它，这正是为什么阈值选择不能仅根据效率指标做出的原因。两者都是相同的置信度非收敛失败（第 4.2 节）在块范围内复发，也是为什么 $\tau_{\text{BWEC}}$ 的先验保守性（第 4 节）不是针对准确率的单向赌注：在长推理任务上，下行风险在激进选择试图改进的指标中甚至不可见。

## 附录 H 局限性

- 需要可定位的答案。CVC 需要一个*可定位*的答案，即任务自身的提取器可以从缓冲区中拉出并在每一步重新定位的短跨度。定位该跨度意味着在缓冲区的特定于任务的一侧搜索特定于任务的值类型，这是 CVC 设计所需的结构先验，而非每个任务调优的超参数（附录 A.2）。这对这里评估的每个任务家族都成立（一个数字、一个字母、一个框式表达式），但对根本没有单个可提取答案跨度的任务失效，代码生成是最明显的案例；整个生成的程序*就是*输出，因此没有 CVC 可以独立于缓冲区其余部分跟踪的子跨度，也没有提取器可以泛化到它。仅 BWEC 仍然适用于那里多块生成的非最终块，但门控的答案验证部分没有定义的目标，因此 LATCH 的终止保证不扩展到此类任务。

- 提交是单调的。LATCH 的所有提交也是单调的；一旦一个位置通过 BWEC 的阈值或调度的 top-$k$ 配额，它就永远不会被重新访问，即使后续步骤现在更丰富的上下文会对其进行不同的预测。CVC 的联合门控仅以这种方式保护最终答案跨度；在 BWEC 更便宜的非最终块规则下提交的中间推理 token 没有等效的安全网。掩码并重构验证遍，在提交前检查模型自身更新的上下文是否会重构相同的候选答案，可以提供额外的安全检查，但每个候选答案需要额外的前向传递，直接与本文针对的加速竞争；评估这种权衡，以及成本是否更好地再投资以证明更低、更激进的 $\tau_{\text{BWEC}}$ 合理，留待未来工作。

- 扩展到单个可定位答案之外。CVC 的验证当前针对每个示例的一个身份跟踪的答案跨度。长篇生成没有扮演该角色的单个跨度，回退到整个缓冲区的分布稳定性将崩溃到第 4.1 节在 Prophet 中识别的相同聚合置信度失败模式，而非 CVC 自身逻辑的真正扩展。一个更有希望的路线是将输出分解为候选单元，多跳响应的每个子答案或生成程序的每个函数，并对每个单元应用 CVC 的身份和稳定性测试；一旦没有块以这种方式被特权化，BWEC 的最终块不对称性也需要不同的结构先验。推进这种分解，并在本文使用的经验容差之上添加无分布风险控制 [^42] [^40]，是自然的下一步。

[^1]: Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems (NeurIPS) 34, pp. 17981–17993. Cited by: §2.

[^2]: PIQA: reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), Cited by: §5.

[^3]: A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems (NeurIPS) 35, pp. 28266–28279. Cited by: §2.

[^4]: DPad: efficient diffusion language models with suffix dropout. In The Fourteenth International Conference on Learning Representations (ICLR), External Links: [Link](https://openreview.net/forum?id=0yOsSMU1eY) Cited by: §2.

[^5]: STDec: spatio-temporal stability guided decoding for dLLMs. arXiv preprint arXiv:2604.06330. Cited by: §2.

[^6]: Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457. Cited by: §5.

[^7]: Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168. External Links: [Link](https://arxiv.org/abs/2110.14168) Cited by: §5.

[^8]: S-GRPO: early exit via reinforcement learning in reasoning models. In Advances in Neural Information Processing Systems 38 (NeurIPS), External Links: [Link](https://openreview.net/forum?id=wNMK5o0Vfg) Cited by: §2.

[^9]: $R^{2}$ -dLLM: accelerating diffusion large language models via spatio-temporal redundancy reduction. arXiv preprint arXiv:2604.18995. Cited by: §2.

[^10]: PAL: program-aided language models. International Conference on Machine Learning (ICML). Cited by: §5.

[^11]: Gemini diffusion. Note: Google DeepMind blog External Links: [Link](https://blog.google/technology/google-deepmind/gemini-diffusion/) Cited by: §2.

[^12]: Detecting the semantic fixed point: a geometric framework for efficient inference. In Proceedings of the 43rd International Conference on Machine Learning (ICML), External Links: [Link](https://openreview.net/forum?id=DACN5xM4h7) Cited by: §2.

[^13]: Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR), Cited by: §5.

[^14]: Measuring mathematical problem solving with the MATH dataset. In NeurIPS Datasets and Benchmarks Track (NeurIPS), Cited by: §5.

[^15]: Argmax flows and multinomial diffusion: learning categorical distributions. Advances in Neural Information Processing Systems (NeurIPS) 34, pp. 12454–12465. Cited by: §2.

[^16]: FlashDLM: accelerating diffusion language model inference via efficient KV caching and guided diffusion. In The Fourteenth International Conference on Learning Representations (ICLR), Note: arXiv:2505.21467 External Links: [Link](https://openreview.net/forum?id=KUfKvlX3VY) Cited by: §2.

[^17]: Mercury: ultra-fast language models based on diffusion. arXiv preprint arXiv:2506.17298. External Links: [Link](https://arxiv.org/abs/2506.17298) Cited by: §2.

[^18]: Early decisions matter: proximity bias and initial trajectory shaping in non-autoregressive diffusion language models. arXiv preprint arXiv:2604.10567. External Links: [Link](https://arxiv.org/abs/2604.10567) Cited by: §2.

[^19]: KLASS: KL-guided fast inference in masked diffusion models. In Advances in Neural Information Processing Systems 38 (NeurIPS), Spotlight, Note: arXiv:2511.05664 Cited by: Appendix B, §1, §2, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1.

[^20]: Diffusion language model knows the answer before it decodes. In The Fourteenth International Conference on Learning Representations (ICLR), External Links: [Link](https://openreview.net/forum?id=g88nt4ieTG) Cited by: Appendix E, Table 8, Table 8, Appendix E, §1, §2, §3, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1.

[^21]: TruthfulQA: measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), Cited by: §5.

[^22]: dLLM-Cache: accelerating diffusion large language models with adaptive caching. In Proceedings of the 43rd International Conference on Machine Learning (ICML), Note: arXiv:2506.06295 External Links: [Link](https://openreview.net/forum?id=DriG3hgh42) Cited by: §2.

[^23]: Discrete diffusion modeling by estimating the ratios of the data distribution. In Proceedings of the 41st International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research, Vol. 235, pp. 32819–32848. External Links: [Link](https://proceedings.mlr.press/v235/lou24a.html) Cited by: §2.

[^24]: dKV-Cache: the cache for diffusion language models. In Advances in Neural Information Processing Systems 38 (NeurIPS), External Links: [Link](https://openreview.net/forum?id=Gppo2JImHs) Cited by: §2.

[^25]: A diverse corpus for evaluating and developing English math word problem solvers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), Cited by: §5.

[^26]: LESS is more: mutual-stability sampling for diffusion language models. Note: arXiv:2606.16908 Cited by: §2.

[^27]: Fast-decoding diffusion language models via progress-aware confidence schedules. In Findings of the Association for Computational Linguistics: ACL 2026, Note: arXiv:2512.02892 Cited by: Appendix B, §1, §2, §3, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1.

[^28]: Large language diffusion models. In Advances in Neural Information Processing Systems 38 (NeurIPS), Note: arXiv:2502.09992 External Links: [Link](https://openreview.net/forum?id=KnqiC0znVF) Cited by: §1, §2, §3, §5.1, §5.

[^29]: Your absorbing discrete diffusion secretly models the conditional distributions of clean data. In The Thirteenth International Conference on Learning Representations (ICLR), External Links: [Link](https://openreview.net/forum?id=sMyXP8Tanm) Cited by: §2.

[^30]: Are NLP models really able to solve simple math word problems?. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), Cited by: §5.

[^31]: Reasoning on the manifold: bidirectional consistency for self-verification in diffusion language models. In Proceedings of the 43rd International Conference on Machine Learning (ICML), Note: arXiv:2604.16565 External Links: [Link](https://openreview.net/forum?id=CUVBdw2tKy) Cited by: §2.

[^32]: Simple and effective masked diffusion language models. In Advances in Neural Information Processing Systems 37 (NeurIPS), External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb0b13cc515724ab8015bc978fdde0ad-Abstract-Conference.html) Cited by: §2.

[^33]: WinoGrande: an adversarial Winograd schema challenge at scale. Communications of the ACM 64 (9), pp. 99–106. Cited by: §5.

[^34]: Simplified and generalized masked diffusion for discrete data. In Advances in Neural Information Processing Systems 37 (NeurIPS), External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html) Cited by: §2.

[^35]: Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning (ICML), pp. 2256–2265. Cited by: §2.

[^36]: Seed diffusion: a large-scale diffusion language model with high-speed inference. arXiv preprint arXiv:2508.02193. External Links: [Link](https://arxiv.org/abs/2508.02193) Cited by: §2.

[^37]: TACG: trajectory-aware commit gating for diffusion language model decoding. arXiv preprint arXiv:2607.03236. Cited by: §2.

[^38]: Accelerating diffusion large language models with SlowFast sampling: the three golden principles. In The Fourteenth International Conference on Learning Representations (ICLR), Note: arXiv:2506.10848 External Links: [Link](https://openreview.net/forum?id=Uh17FiwF4q) Cited by: Appendix B, §F.2, §1, §2, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1, Table 1.

[^39]: Fast-dLLM: training-free acceleration of diffusion LLM by enabling KV cache and parallel decoding. In The Fourteenth International Conference on Learning Representations (ICLR), Note: arXiv:2505.22618 External Links: [Link](https://openreview.net/forum?id=3Z3Is6hnOT) Cited by: §1, §2, §4.3.

[^40]: Controlling the risk of corrupted contexts for language models via early-exiting. In Proceedings of the 43rd International Conference on Machine Learning (ICML), Note: arXiv:2510.02480 External Links: [Link](https://openreview.net/forum?id=8bUDbWMo5v) Cited by: 3rd item.

[^41]: Streaming-dLLM: accelerating diffusion LLMs via suffix pruning and dynamic decoding. arXiv preprint arXiv:2601.17917. External Links: [Link](https://arxiv.org/abs/2601.17917) Cited by: §2.

[^42]: Statistical early stopping for reasoning models. In Proceedings of the 43rd International Conference on Machine Learning (ICML), Note: arXiv:2602.13935 External Links: [Link](https://openreview.net/forum?id=TJshRZDdyW) Cited by: 3rd item.

[^43]: Dynamic early exit in reasoning models. In The Fourteenth International Conference on Learning Representations (ICLR), External Links: [Link](https://openreview.net/forum?id=NpU7ZXafRi) Cited by: §2.

[^44]: Dream 7B: diffusion large language models. arXiv preprint arXiv:2508.15487. External Links: [Link](https://arxiv.org/abs/2508.15487) Cited by: §1, §2, §5.1, §5.

[^45]: HellaSwag: can a machine really finish your sentence?. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), Cited by: §5.

[^46]: CORE: context-robust remasking for diffusion language models. In Proceedings of the 43rd International Conference on Machine Learning (ICML), Note: arXiv:2602.04096 External Links: [Link](https://openreview.net/forum?id=bmKHxLWkz9) Cited by: §2.

[^47]: Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. In The Thirteenth International Conference on Learning Representations (ICLR), External Links: [Link](https://openreview.net/forum?id=CTC7CmirNr) Cited by: §2.
