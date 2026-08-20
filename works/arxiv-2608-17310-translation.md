---
title: "Agentic ESOpt：以最小GPU内存需求微调长视距LLM智能体"
sourceTitle: "Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Memory Requirements"
sourceUrl: "https://arxiv.org/html/2608.17310v1"
sourceRequestedUrl: "https://arxiv.org/html/2608.17310v1"
authors:
  - Zhi Zheng
  - Rongsheng Chen
  - Yunpeng Ba
  - Zhenkun Wang
  - Yee Whye Teh
  - Wee Sun Lee
institution: "新加坡国立大学（NUS）"
arxivId: "2608.17310"
publishDate: "2026-08-18"
translatedAt: "2026-08-20"
pipelineRunId: "20260820-091049"
pipelineSource: "translate/20260820-091049/works-ready/arxiv-2608-17310-translation.md"
adapter: "generic"
capturedAt: "2026-08-20T01:11:57.164Z"
conversionMethod: "defuddle-api"
fallbackReason: "defuddle.md produced higher-quality markdown than local extraction"
kind: "generic/article"
language: "zh-CN"
sourceLanguage: "en"
sourceFigureCount: 4
---

# Agentic ESOpt：以最小GPU内存需求微调长视距LLM智能体

新加坡国立大学（NUS）

Zhi Zheng    Rongsheng Chen    Yunpeng Ba    Zhenkun Wang    Yee Whye Teh    Wee Sun Lee

###### 摘要

强化学习（Reinforcement Learning, RL）在单轮LLM微调中表现出色。然而，长视距（Long-Horizon）智能体化推理引入了日益分支化的交互和稀疏奖励（Sparse Rewards），暴露了RL的若干局限：其重量级的反向传播（Backpropagation）使得微调更大规模的LLM变得不切实际，而更长的视距轨迹（Trajectory）使信用分配（Credit Assignment）问题显著加剧。本文论证了进化策略（Evolution Strategies, ES）可能是微调长视距LLM智能体的更优选择。与智能体化RL相比，ES提供三个关键优势：1) 模型可扩展性：ES实现全参数优化（Full-Parameter Optimization）仅需最小的推理级GPU内存，使微调大型LLM成为可能。2) 灵活性：其轻量级的黑盒反馈（Black-Box Feedback）接口使ES微调易于与提示空间进化（如技能优化与测试时计算，Test-Time Compute）组合；3) 长视距可扩展性：ES执行轨迹级参数归因（Parameter Attribution）而无需跨视距分解奖励，随着视距长度增长，可扩展性优于智能体化RL。

基于此洞察，我们提出Agentic ESOpt，一个为灵活的参数-上下文协同进化量身定制的全参数智能体化微调框架。在每个步骤，Agentic ESOpt在当前LLM参数周围采样扰动（Perturbation），用奖励评估生成的智能体，并应用在线奖励加权更新。为改善探索-适应权衡（Exploration-Adaptation Trade-off），Agentic ESOpt进一步引入扰动尺度 $\sigma$ 的余弦衰减（Cosine Decay）调度。我们在训练时微调和智能体化测试时计算场景下评估Agentic ESOpt。在长视距数独（Sudoku）任务上，Agentic ESOpt使用Qwen3.5-4B模型超越RL方法12.50%。在WebArena-Lite上，对Qwen3.5-27B的全参数优化将无技能基线提升6.69%，结合Agentic ESOpt与Trace2Skill进一步将Trace2Skill基线提升2.42%。在测试时自动启发式设计中，Agentic ESOpt执行在线提示-参数协同进化，在36个设置中的28个超越匹配基线。

<sup>†</sup> <sup>†</sup> <sup>†</sup> <sup>†</sup> <sup>†</sup>![Refer to caption](imgs/img-001-Agent.png)

(a) 智能体化推理的挑战

## 1 引言

先进的大型语言模型（Large Language Models, LLMs）如Qwen3、DeepSeek-R1和Gemini 2.5已展示出作为通用智能体（Agent）的强大能力[^44] [^8] [^6]。凭借工具使用、长上下文处理和多模态交互能力，这些模型可以导航网站[^55]、编辑代码仓库[^45]以及协调多步骤软件工作流[^40]。然而，通用智能体在不常见的工具API[^22]和专业的科学或算法任务[^16]上仍可能表现不佳。因此，高效微调先进LLM智能体以适应特定任务的专业知识仍然至关重要[^2]。

强化学习在单轮LLM微调中展现了显著成效[^31] [^20] [^53] [^37]。然而，在长视距智能体化推理中，随着交互分支日益增多且仅提供稀疏反馈，智能体化RL的若干局限被暴露出来。如图1所示，首先，智能体化RL需要存储重量级激活值、优化器状态并通过轨迹执行反向传播，使得对更大LLM进行全参数微调变得越来越不切实际。此外，随着轨迹变得更长、更分支化，将稀疏的轨迹级奖励分配回单个决策变得更加困难[^13]。

本文论证进化策略[^28]可能是微调长视距LLM智能体的更优选择。ES不执行反向传播，而是在当前LLM参数周围采样扰动，用环境奖励评估扰动后的智能体，并应用奖励加权的参数更新。与智能体化RL相比，ES提供三个关键优势：

- 1) 模型可扩展性：ES实现全参数优化仅需推理级GPU内存（最小量级），大幅降低微调更大LLM智能体的内存门槛。
- 2) 灵活性：其轻量级的黑盒反馈接口使ES微调易于与技能空间进化[^25]和测试时计算[^16]组合。
- 3) 长视距可扩展性：与通常导致长视距信用分配不佳的RL估计器不同，ES执行轨迹级参数归因而无需跨轮次分解奖励，随着视距长度增长，可扩展性优于智能体化RL。

最近的工作已探索将ES用于单轮LLM推理[^27] [^36] [^29]，其中ES实现了更高的GPU内存效率但性能略低于RL方法。然而，我们论证ES的结构性优势在微调长视距智能体时尤为突出，ES可以显著优于RL，而不仅仅是更廉价的替代方案。为实现这一优势，我们提出Agentic ESOpt，一个用于训练时智能体微调和智能体化测试时计算的全参数ES框架。在每一代，Agentic ESOpt采样全参数扰动，用环境奖励评估生成的智能体，并应用在线奖励加权更新。其轻量级黑盒更新实现在提示空间优化循环内的即时参数适应，允许参数更新补充技能空间优化和测试时搜索。为改善探索-适应权衡，Agentic ESOpt进一步为扰动尺度 $\sigma$ 引入余弦衰减机制，在训练时优化中保留非零终止值 $\sigma_{T}$ 以实现温和的平滑正则化，在测试时优化中将 $\sigma_{T}$ 衰减至0以实现渐进式的更精细适应。

我们在训练时和测试时计算场景下评估Agentic ESOpt对4B至27B规模LLM的智能体化微调。在训练时微调中，我们研究长视距推理、ReAct风格的工具使用和网页智能体。在数独任务上，匹配FLOP的基于RL和ES的方法在较小的最小成功视距（5视距和10视距）下表现相当，但随着视距增长，它们的相对排序发生变化：在15轮时，Agentic ESOpt相比最强的群相对策略优化（GRPO）基线达到+12.5%。在ReAct风格的数学和文档问答任务中，Agentic ESOpt相对Qwen3.5-4B基础模型平均提升13.7%，相对智能体化GRPO提升8.3%。在WebArena-Lite上，对Qwen3.5-27B的全参数优化将无技能基线从29.47%提升至36.16%，而将Agentic ESOpt与Trace2Skill结合则将其从33.94%提升至36.36%。在测试时智能体化启发式设计中，Agentic ESOpt在36个比较中的28个改善了匹配基线。此外，初步的种群敏感性研究进一步表明，更强的LLM骨干可以用更小的种群获得有效的ES更新。我们的贡献总结如下：

- 我们识别出在长视距智能体化推理中，ES变得优于智能体化RL。我们将这一转变归因于ES的三个关键属性：通过推理级GPU内存实现模型可扩展性、通过黑盒轨迹反馈实现灵活性、通过轨迹级参数归因实现长视距可扩展性。
- 我们引入Agentic ESOpt，一个无反向传播的ES框架，仅需最小GPU内存。它支持训练时适应和测试时计算中的灵活智能体化微调。它允许参数优化与提示空间进化组合，同时扰动的余弦调度改善探索-适应权衡。
- 我们在长视距数独、ReAct风格工具使用、网页智能体和自动启发式设计上验证Agentic ESOpt，使用4B至27B规模模型。Agentic ESOpt在长视距数独和ReAct风格数学/文档问答上超越RL，实现27B WebArena智能体的全参数适应，改善Trace2Skill，并在36个设置中的28个增强现有测试时进化搜索。

## 2 预备知识：智能体化LLM推理

我们定义一个多轮LLM智能体，重复观察环境并产生动作如下：

$$
a_{t}\sim\pi_{\theta}(a_{t}\mid\bm{o}_{\leq t},c_{t}),
$$

其中 $\theta$ 表示模型参数，$\bm{o}_{\leq t}$ 是交互历史，$c_{t}$ 是外部提示、记忆、技能或工具指令。一个回合产生一条多轮轨迹 $\bm{\tau}=(o_{1},a_{1},\ldots,o_{H},a_{H})$，其中视距（Horizon） $H$ 是终止前智能体-环境交互的轮次数。其回报为 $R(\bm{\tau})=\sum_{t=1}^{H}\gamma^{t-1}r_{t}$。在许多智能体化推理任务中，奖励是稀疏的：中间奖励通常为零，对于 $t<H$ 有 $r_{t}=0$，只有完成的轨迹接收任务得分。在某些问题（如MATH与DocVQA的ReAct风格工具使用）中，智能体接收完整的多轮轨迹 $\bm{\tau}$ 作为输入，但在某些具有（部分）马尔可夫性质的问题（数独、WebArena）中，智能体只需接收部分局部输入（如仅 $o_{t}$）。

特定任务智能体的优化可作用于 $c_{t}$ 或 $\theta$。提示空间优化方法在冻结LLM的同时优化 $c_{t}$。它们轻量级，但只能激发冻结策略已经可访问的行为[^46] [^25] [^16] [^54]。策略微调方法通过监督微调（Supervised Fine-Tuning, SFT）、在线策略蒸馏（On-Policy Distillation, OPD）、群相对策略优化（Group Relative Policy Optimization, GRPO）或近端策略优化（Proximal Policy Optimization, PPO）优化 $\theta$ 以获得更好的智能体能力。

##### 智能体化SFT与OPD

智能体化SFT从专家动作学习[^1]，而智能体化OPD从更高能力LLM的token分布学习[^35]。两种方法都需要超出标量环境奖励的标签，这需要额外成本，因此我们将其排除在比较范围之外。

##### 智能体化PPO与GRPO

智能体化PPO和智能体化GRPO需要标量环境奖励，使其在许多应用场景中有用[^26]。智能体化GRPO[^31]为同一任务采样 $G$ 条轨迹，并为轨迹 $i$ 分配群相对优势如下：

$$
\widehat{A}_{i}=\frac{R_{i}-\operatorname{mean}_{j}(R_{j})}{\operatorname{std}_{j}(R_{j})+\varepsilon}.
$$

令 $q_{i,t}(\theta)=\pi_{\theta}(a_{i,t}\mid h_{i,t})/\pi_{\theta_{\mathrm{old}}}(a_{i,t}\mid h_{i,t})$，其中 $h_{i,t}=(\bm{o}_{\leq t},c_{t})$。智能体化GRPO最小化损失如下：

$$
\mathcal{L}_{\mathrm{GRPO}}=-\frac{1}{G}\sum_{i=1}^{G}\sum_{t=1}^{H_{i}}\min\!\left(q_{i,t}\widehat{A}_{i},\operatorname{clip}(q_{i,t},1-\epsilon,1+\epsilon)\widehat{A}_{i}\right)+\beta D_{\mathrm{KL}}(\pi_{\theta}\|\pi_{\mathrm{ref}}),
$$

然而，最近的研究[^3] [^9] [^51]指出，尽管GRPO的群相对优势在单轮场景中具有代表性，但无法覆盖多轮场景。为实现更好的信用分配，PPO最近被应用于多轮智能体化RL[^15] [^10]，它使用相同的截断替代但用由评论家LLM估计的轮次级优势[^30]替换 $\widehat{A}_{i}$：

$$
\mathcal{L}_{\mathrm{PPO}}=-\sum_{t=1}^{H}\min\!\left(q_{t}\widehat{A}_{t},\operatorname{clip}(q_{t},1-\epsilon,1+\epsilon)\widehat{A}_{t}\right),\quad\widehat{A}_{t}=\sum_{l=0}^{H-t}(\gamma\lambda)^{l}\bigl(r_{t+l}+\gamma V_{\phi}(h_{t+l+1})-V_{\phi}(h_{t+l})\bigr).
$$

我们认为PPO无法完全消除长视距困难。它需要评论家预热阶段来学习有意义的价值，而稀疏终止奖励使其早期优势不可靠。即使有训练良好的评论家，策略梯度仍需求和 $H$ 个动作级得分项，因此其方差仍依赖于视距长度。这些局限激发了下文引入并在附录C.3中分析的轨迹级估计器。信用分配中的这些缺陷激发了Agentic ESOpt的参数空间ES公式。

## 3 方法：Agentic ESOpt

如图2所示，Agentic ESOpt通过在当前LLM周围采样参数扰动、用标量环境奖励评估扰动后的智能体并应用奖励加权参数更新来执行全参数ES优化。该更新是仅前向的，仅需存储噪声种子并使用原地加减[^27]，因此Agentic ESOpt的GPU内存需求保持最小，与推理需求相同。此外，相同的黑盒轨迹反馈可被技能空间优化器或测试时计算复用。

![Refer to caption](imgs/img-002-main-process.png)

图2：Agentic ESOpt的详细工作流。从当前LLM开始，Agentic ESOpt采样参数扰动，在环境中评估扰动后的智能体，归一化其标量奖励，并应用奖励加权的ES更新。与智能体化RL相比，Agentic ESOpt提供模型可扩展性、优化灵活性和长视距可扩展性。其轻量级黑盒接口还允许与提示空间优化方法（如Trace2Skill（基于LLM）和EoH（基于启发式））轻松组合，实现在现有测试时计算过程内的即时参数适应。

形式化地，令 $\bm{\tau}=(o_{0},a_{0},\ldots,o_{H},a_{H})$ 表示由策略 $\pi_{\theta}$ 诱导的交互轨迹，令 $R(\bm{\tau})$ 表示其标量轨迹回报。对于固定的外部智能体状态 $c$，目标如下：

$$
J(\theta;c)=\mathbb{E}_{\bm{\tau}\sim\pi_{\theta}(\cdot\mid c)}\left[R(\bm{\tau})\right].
$$

Agentic ESOpt通过在 $\theta$ 周围的参数空间搜索来优化目标，其中 $\bm{\epsilon}\in\mathbb{R}^{d}$ 是 $d$ 维参数上的全参数扰动，高斯平滑目标为：

$$
J_{\sigma}(\theta;c)=\mathbb{E}_{\bm{\epsilon}\sim\mathcal{N}(0,I)}\left[J(\theta+\sigma\bm{\epsilon};c)\right].
$$

然后我们可以推导ES伪梯度如下[^28] [^27]：

$$
\nabla_{\theta}J_{\sigma}(\theta;c)=\frac{1}{\sigma}\mathbb{E}_{\bm{\epsilon}}\left[J(\theta+\sigma\bm{\epsilon};c)\bm{\epsilon}\right].
$$

推导详见附录C.1。ES梯度从标量得分估计，无需微分智能体-环境交互。

在实现中，为估计式(4)，Agentic ESOpt采样 $G$ 个扰动 $\bm{\epsilon}_{1},\ldots,\bm{\epsilon}_{G}$，评估相应的扰动智能体，并获得奖励 $R_{i}=R(\bm{\tau}_{i})$。为减少方差，我们使用z得分在种群内归一化奖励如下：

$$
\hat{R}_{i}=\frac{R_{i}-\mu_{R}}{s_{R}+\varepsilon},\quad\text{其中}\quad\mu_{R}=\frac{1}{G}\sum_{j=1}^{G}R_{j},\quad s_{R}^{2}=\frac{1}{G}\sum_{j=1}^{G}(R_{j}-\mu_{R})^{2}.
$$

在实践中，与式(4)中的标准ES估计器不同，我们的归一化更新省略显式的 $1/\sigma$ 因子，用 $\alpha$ 作为有效更新尺度。因此实现的更新为：

$$
\theta_{t+1}=\theta_{t}+\frac{\alpha}{G}\sum_{i=1}^{G}\hat{R}_{i}\bm{\epsilon}_{i}.
$$

我们遵循[^27]仅存储每个扰动的噪声种子并对每个扰动智能体使用原地加减，因此Agentic ESOpt仅需与推理相同量级的GPU内存。

##### 提示空间组合与提示-参数协同进化

测试时计算和提示空间优化方法通常在整个搜索过程中保持LLM参数固定。因此，搜索只能重新加权或重组冻结策略已经可访问的行为，当解决任务需要改变底层策略本身时，这可能限制优化。相比之下，Agentic ESOpt的轻量级黑盒更新允许在提示空间搜索的同时即时执行参数适应，实现提示-参数协同进化。

令 $\mathcal{D}_{t}$ 表示在迭代 $t$ 收集的轨迹和得分，$\mathcal{U}_{\mathrm{ES}}$ 为Agentic ESOpt参数更新，$\mathcal{U}_{c}$ 为提示 $c_{t}$ 的外部更新规则。一般的交替外循环可将两个空间更新为

$$
\theta_{t+1}=\mathcal{U}_{\mathrm{ES}}(\theta_{t};c_{t},\mathcal{D}_{t}),\quad c_{t+1}=\mathcal{U}_{c}(c_{t};\mathcal{D}_{t}).
$$

### 3.1 扰动半径 $\sigma$ 的余弦衰减

使用扰动 $\bm{\epsilon}\sim\mathcal{N}(0,I)$，Agentic ESOpt有效地优化式(3)中的高斯平滑目标。然而，非零扰动半径引入平滑偏差，其主导项表征为：

###### 引理1（高斯平滑偏差）

假设 $J(\theta;c)$ 在 $\theta$ 的邻域内充分光滑。则

$$
J_{\sigma}(\theta;c)=J(\theta;c)+\frac{\sigma^{2}}{2}\mathrm{Tr}\!\left(\nabla_{\theta}^{2}J(\theta;c)\right)+O(\sigma^{4}).
$$

证明见附录C.2。二阶项 $\mathrm{Tr}(\nabla_{\theta}^{2}J)$ 可视为正则化项。对于最大化目标，它惩罚尖锐的局部最优解同时偏好更平坦的参数邻域。因此，更大的 $\sigma$ 引入更强的正则化但也带来更大的原始目标偏差。

现有用于单轮LLM微调的ES方法通常在整个优化过程中使用固定扰动半径[^27]，未明确适应这一权衡。Agentic ESOpt转而在 $T$ 个更新步骤中逐渐减小 $\sigma$，在早期使用更大半径以实现更广泛探索和更强正则化，在后期使用更小半径以增加开发并减少目标偏差如下：

$$
\sigma_{t}=\sigma_{T}+(\sigma_{0}-\sigma_{T})\frac{1+\cos(\pi t/T)}{2},\qquad t=0,\ldots,T.
$$

对于训练时Agentic ESOpt，我们保留非零 $\sigma_{T}$ 以平衡开发与探索和正则化。相反，测试时计算关注当前任务的无偏结果而非智能体的泛化，因此我们将 $\sigma_{T}$ 衰减至零以最小化优化末期的目标偏差。

## 4 智能体化数独：长视距可扩展性的受控实验

为展示Agentic ESOpt相对智能体化RL在长视距可扩展性上的优势，本节首先从理论上分析它们的特性。然后我们在受控的多轮数独环境中评估提出的Agentic ESOpt，其中最小可能视距长度 $H$ 由数独板上的掩码数量严格定义。
# Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Memory Requirements

NUS — National University of Singapore

Zhi Zheng    Rongsheng Chen    Yunpeng Ba    Zhenkun Wang    Yee Whye Teh    Wee Sun Lee

###### 摘要

强化学习（Reinforcement Learning, RL）在单轮大语言模型微调中展现了良好的前景。然而，长视距智能体化推理引入了日益分支化的交互和稀疏奖励，暴露了强化学习的几个局限性：其重量级反向传播使得微调更大的大语言模型变得不切实际，而更长视距的轨迹使信用分配变得更加困难。本文认为，进化策略（Evolution Strategies, ES）可能是微调长视距大语言模型智能体的更好选择。与智能体化强化学习相比，进化策略提供了三个关键优势：1）模型可扩展性：进化策略实现了仅需最少推理级GPU内存的全参数优化，使微调大型大语言模型成为可能。2）灵活性：其轻量级黑盒反馈接口使进化策略微调易于与提示空间演化（如技能优化和测试时计算）组合；以及 3）长视距可扩展性：进化策略执行轨迹级参数归因而无需跨视距分解奖励，随着视距长度增长，其可扩展性优于智能体化强化学习。

基于这一洞察，我们提出了Agentic ESOpt，一个专为灵活参数-上下文协同演化而设计的全参数智能体化微调框架。在每一步，Agentic ESOpt在当前大语言模型参数周围采样扰动，用奖励评估生成的智能体，并应用在线奖励加权更新。为了改善探索-适应权衡，Agentic ESOpt进一步引入了扰动尺度 $\sigma$ 的余弦衰减调度。我们在训练时微调和智能体化测试时计算两种设置下评估Agentic ESOpt。在长视距数独任务上，Agentic ESOpt在Qwen3.5-4B上的表现比强化学习方法高出12.50%。在WebArena-Lite上，Qwen3.5-27B的全参数优化将无技能基线提升了6.69%，将Agentic ESOpt与Trace2Skill结合进一步将Trace2Skill基线提升了2.42%。在测试时自动启发式设计中，Agentic ESOpt执行在线提示-参数协同演化，在36个设置中的28个改进了其匹配基线。

<sup>†</sup> <sup>†</sup> <sup>†</sup> <sup>†</sup> <sup>†</sup>![Refer to caption](imgs/img-001-Agent.png)

(a) 智能体化推理的挑战

## 1 引言

先进的大语言模型（LLMs），如Qwen3、DeepSeek-R1和Gemini 2.5，已展现出作为通用智能体的强大能力[^44] [^8] [^6]。凭借其在工具使用、长上下文处理和多模态交互方面的能力，这些模型可以导航网站[^55]、编辑代码库[^45]，以及协调多步骤软件工作流[^40]。然而，通用智能体在不常见的工具API[^22]和专业化科学或算法任务[^16]上仍可能表现不佳。因此，高效微调先进的大语言模型智能体以适应特定任务的专业知识仍然很重要[^2]。

强化学习在单轮大语言模型微调中已展现出显著的有效性[^31] [^20] [^53] [^37]。然而，在长视距智能体化推理中，随着引入日益分支化的交互和仅提供稀疏反馈，智能体化强化学习的几个局限性被暴露出来。如图1所示，首先，智能体化强化学习需要存储重量级激活、优化器状态，并通过轨迹执行反向传播，使得对更大的大语言模型进行全参数微调变得越来越不切实际。此外，随着轨迹变得更长且更加分支化，将稀疏的轨迹级奖励分配回各个决策变得更加困难[^13]。

本文认为，进化策略（ES）[^28]可能是微调长视距大语言模型智能体的更好选择。进化策略不执行反向传播，而是在当前大语言模型参数周围采样扰动，用环境奖励评估扰动后的智能体，并应用奖励加权的参数更新。与智能体化强化学习相比，进化策略提供三个关键优势：

- 1）模型可扩展性：进化策略实现了仅需推理级GPU内存的全参数优化，这是最小量级，大幅降低了微调更大大语言模型智能体的内存障碍。
- 2）灵活性：其轻量级黑盒反馈接口使进化策略微调易于与技能空间演化[^25]和测试时计算[^16]组合。
- 3）长视距可扩展性：与通常导致糟糕的长视距信用分配的强化学习估计器不同，进化策略执行轨迹级参数归因而无需跨回合分解奖励，随着视距长度增长，其可扩展性优于智能体化强化学习。

最近的工作已探索将进化策略用于单轮大语言模型推理[^27] [^36] [^29]，其中进化策略实现了更高的GPU内存效率，但性能略低于强化学习方法。然而，我们认为进化策略的结构优势在微调长视距智能体时尤为明显，进化策略可能显著优于强化学习，而不仅仅是一个更廉价的替代品。为了实现这一优势，我们提出了Agentic ESOpt，一个用于训练时智能体微调和智能体化测试时计算的全参数进化策略框架。在每一代，Agentic ESOpt采样全参数扰动，用环境奖励评估生成的智能体，并应用在线奖励加权更新。其轻量级黑盒更新使得在提示空间优化循环中能够即时进行参数适应，允许参数更新补充技能空间优化和测试时搜索。为了改善探索-适应权衡，Agentic ESOpt进一步引入了扰动尺度 $\sigma$ 的余弦衰减机制，在训练时优化中保留非零终端 $\sigma_{T}$ 以实现温和的平滑正则化，并在测试时优化中将 $\sigma_{T}$ 衰减到0以实现渐进式更精细的适应。

我们在4B到27B的大语言模型上评估Agentic ESOpt用于训练时和测试时计算的智能体化微调。在训练时微调中，我们研究了长视距推理、ReAct风格的工具使用和网络智能体。在数独任务上，在计算量匹配的情况下，基于强化学习和进化策略的方法在较小的最小成功视距（5视距和10视距）时具有竞争力，但随着视距增长，它们的相对排序发生变化：在15回合时，Agentic ESOpt相比最强的群相对策略优化（GRPO）基线达到了+12.5%的提升。在ReAct风格的数学和文档问答任务中，Agentic ESOpt相比Qwen3.5-4B基础模型平均提升了13.7%，相比智能体化GRPO提升了8.3%。在WebArena-Lite上，Qwen3.5-27B的全参数优化将无技能基线从29.47%提升到36.16%，而将Agentic ESOpt与Trace2Skill结合则将其从33.94%提升到36.36%。在测试时智能体化启发式设计中，Agentic ESOpt在36个比较中的28个改进了匹配基线。此外，初步的种群敏感性研究进一步表明，更强的大语言模型骨干可以用更小的种群获得有用的进化策略更新。我们的贡献总结如下：

- 我们识别出，在长视距智能体化推理中，进化策略变得优于智能体化强化学习。我们将这一转变归因于进化策略的三个关键特性：通过推理级GPU内存实现的模型可扩展性、通过黑盒轨迹反馈实现的灵活性，以及通过轨迹级参数归因实现的长视距可扩展性。
- 我们引入了Agentic ESOpt，一个无需反向传播、仅需最少GPU内存的进化策略框架。它支持训练时适应和测试时计算中的灵活智能体化微调。它允许参数优化与提示空间演化组合，同时扰动的余弦调度改善了探索-适应权衡。
- 我们在长视距数独、ReAct风格工具使用、网络智能体和自动启发式设计上，使用4B到27B的模型验证了Agentic ESOpt。Agentic ESOpt在长视距数独和ReAct风格数学/文档问答上优于强化学习，能够对27B的WebArena智能体进行全参数适应，改进了Trace2Skill，并在36个设置中的28个增强了现有的测试时演化搜索。

## 2 预备知识：智能体化大语言模型推理

我们定义一个多轮大语言模型智能体，它重复观察环境并产生动作如下：

$$
a_{t}\sim\pi_{\theta}(a_{t}\mid\bm{o}_{\leq t},c_{t}),
$$

其中 $\theta$ 表示模型参数，$\bm{o}_{\leq t}$ 是交互历史，$c_{t}$ 是外部提示、记忆、技能或工具指令。一个回合产生一个多轮轨迹 $\bm{\tau}=(o_{1},a_{1},\ldots,o_{H},a_{H})$，其中视距 $H$ 是终止前智能体-环境交互的回合数。其回报为 $R(\bm{\tau})=\sum_{t=1}^{H}\gamma^{t-1}r_{t}$。在许多智能体化推理任务中，奖励是稀疏的：中间奖励通常为零，对于 $t<H$ 有 $r_{t}=0$，只有完成的轨迹才会收到任务分数。在某些问题（例如，数学和文档问答的ReAct风格工具使用）中，智能体接收完整的多轮轨迹 $\bm{\tau}$ 作为输入，但在某些具有（部分）马尔可夫性质的问题（数独、WebArena）中，智能体只需要接收部分局部输入（例如，仅 $o_{t}$）。

特定任务智能体的优化可以作用于 $c_{t}$ 或 $\theta$。提示空间优化方法优化 $c_{t}$ 同时冻结大语言模型。它们是轻量级的，但只能激发冻结策略已经可访问的行为[^46] [^25] [^16] [^54]。策略微调方法通过监督微调（Supervised Fine-Tuning, SFT）、在线策略蒸馏（On-Policy Distillation, OPD）、群相对策略优化（Group Relative Policy Optimization, GRPO）或近端策略优化（Proximal Policy Optimization, PPO）优化 $\theta$ 以获得更好的智能体化能力。

##### 智能体化SFT与OPD

智能体化SFT从专家动作中学习[^1]，而智能体化OPD从更高能力大语言模型的词元分布中学习[^35]。这两种方法都需要超越标量环境奖励的标签，这需要额外成本，因此我们将它们排除在比较范围之外。

##### 智能体化PPO与GRPO

智能体化PPO和智能体化GRPO需要标量环境奖励，使它们在许多应用场景中有用[^26]。智能体化GRPO[^31]为同一任务采样 $G$ 条轨迹，并为轨迹 $i$ 分配群相对优势如下：

$$
\widehat{A}_{i}=\frac{R_{i}-\operatorname{mean}_{j}(R_{j})}{\operatorname{std}_{j}(R_{j})+\varepsilon}.
$$

令 $q_{i,t}(\theta)=\pi_{\theta}(a_{i,t}\mid h_{i,t})/\pi_{\theta_{\mathrm{old}}}(a_{i,t}\mid h_{i,t})$，其中 $h_{i,t}=(\bm{o}_{\leq t},c_{t})$。智能体化GRPO最小化损失如下：

$$
\mathcal{L}_{\mathrm{GRPO}}=-\frac{1}{G}\sum_{i=1}^{G}\sum_{t=1}^{H_{i}}\min\!\left(q_{i,t}\widehat{A}_{i},\operatorname{clip}(q_{i,t},1-\epsilon,1+\epsilon)\widehat{A}_{i}\right)+\beta D_{\mathrm{KL}}(\pi_{\theta}\|\pi_{\mathrm{ref}}),
$$

然而，最近的研究[^3] [^9] [^51]指出，尽管GRPO的群相对优势在单轮场景中具有代表性，但它无法覆盖多轮场景。为了实现更好的信用分配，PPO最近被应用于多轮智能体化强化学习[^15] [^10]，它使用相同的裁剪替代但用由评价器大语言模型估计的回合级优势替换 $\widehat{A}_{i}$[^30]：

$$
\mathcal{L}_{\mathrm{PPO}}=-\sum_{t=1}^{H}\min\!\left(q_{t}\widehat{A}_{t},\operatorname{clip}(q_{t},1-\epsilon,1+\epsilon)\widehat{A}_{t}\right),\quad\widehat{A}_{t}=\sum_{l=0}^{H-t}(\gamma\lambda)^{l}\bigl(r_{t+l}+\gamma V_{\phi}(h_{t+l+1})-V_{\phi}(h_{t+l})\bigr).
$$

我们认为PPO无法完全消除长视距困难。它需要一个评价器预热阶段来学习有意义的价值，而稀疏的终端奖励使其早期优势不可靠。即使有训练良好的评价器，策略梯度仍然对 $H$ 个动作级分数项求和，因此其方差仍然依赖于视距长度。这些局限性激发了下面介绍的轨迹级估计器，并在附录C.3中进行分析。这些信用分配中的缺陷激发了Agentic ESOpt的参数空间进化策略公式。

## 3 方法：Agentic ESOpt

如图2所示，Agentic ESOpt通过在当前大语言模型周围采样参数扰动、用标量环境奖励评估扰动后的智能体，并应用奖励加权的参数更新，来执行全参数进化策略优化。这种更新是仅前向的，只需要存储噪声种子并使用就地加法和减法[^27]，因此Agentic ESOpt的GPU内存需求保持最小，与推理需求相同。此外，相同的黑盒轨迹反馈可以被技能空间优化器或测试时计算重用。

![Refer to caption](imgs/img-002-main-process.png)

图2：Agentic ESOpt的详细工作流程。从当前大语言模型开始，Agentic ESOpt采样参数扰动，在环境中评估扰动后的智能体，归一化其标量奖励，并应用奖励加权的进化策略更新。与智能体化强化学习相比，Agentic ESOpt提供了模型可扩展性、优化灵活性和长视距可扩展性。其轻量级黑盒接口还允许与提示空间优化方法（如基于大语言模型的Trace2Skill和基于启发式的EoH）轻松组合，在现有测试时计算过程中实现即时参数适应。

形式上，令 $\bm{\tau}=(o_{0},a_{0},\ldots,o_{H},a_{H})$ 表示由策略 $\pi_{\theta}$ 诱导的交互轨迹，令 $R(\bm{\tau})$ 表示其标量轨迹回报。对于固定的外部智能体状态 $c$，目标如下：

$$
J(\theta;c)=\mathbb{E}_{\bm{\tau}\sim\pi_{\theta}(\cdot\mid c)}\left[R(\bm{\tau})\right].
$$

Agentic ESOpt通过在 $\theta$ 周围的参数空间中搜索来优化目标，其中 $\bm{\epsilon}\in\mathbb{R}^{d}$ 是对 $d$ 维参数的全参数扰动，高斯平滑目标为：

$$
J_{\sigma}(\theta;c)=\mathbb{E}_{\bm{\epsilon}\sim\mathcal{N}(0,I)}\left[J(\theta+\sigma\bm{\epsilon};c)\right].
$$

然后我们可以推导出进化策略伪梯度如下[^28] [^27]：

$$
\nabla_{\theta}J_{\sigma}(\theta;c)=\frac{1}{\sigma}\mathbb{E}_{\bm{\epsilon}}\left[J(\theta+\sigma\bm{\epsilon};c)\bm{\epsilon}\right].
$$

推导详见附录C.1。进化策略梯度是从标量分数估计的，无需对智能体-环境交互进行微分。

在实现中，为了估计式(4)，Agentic ESOpt采样 $G$ 个扰动 $\bm{\epsilon}_{1},\ldots,\bm{\epsilon}_{G}$，评估相应的扰动智能体，并获得奖励 $R_{i}=R(\bm{\tau}_{i})$。为了减少方差，我们用z分数在种群内归一化奖励如下：

$$
\hat{R}_{i}=\frac{R_{i}-\mu_{R}}{s_{R}+\varepsilon},\quad\text{其中}\quad\mu_{R}=\frac{1}{G}\sum_{j=1}^{G}R_{j},\quad s_{R}^{2}=\frac{1}{G}\sum_{j=1}^{G}(R_{j}-\mu_{R})^{2}.
$$

在实践中，与式(4)中的规范进化策略估计器不同，我们的归一化更新省略了显式的 $1/\sigma$ 因子，$\alpha$ 作为有效更新尺度。因此实现的更新为：

$$
\theta_{t+1}=\theta_{t}+\frac{\alpha}{G}\sum_{i=1}^{G}\hat{R}_{i}\bm{\epsilon}_{i}.
$$

我们遵循[^27]，仅存储每个扰动的噪声种子，并对每个扰动智能体使用就地加法和减法，因此Agentic ESOpt仅需要与推理相同数量的GPU内存。

##### 提示空间组合与提示-参数协同演化

测试时计算和提示空间优化方法通常在整个搜索过程中保持大语言模型参数固定。因此，搜索只能重新加权或重新组合冻结策略已经可访问的行为，这可能在解决任务需要改变底层策略本身时限制优化。相比之下，Agentic ESOpt的轻量级黑盒更新允许在提示空间搜索的同时即时执行参数适应，实现提示-参数协同演化。

令 $\mathcal{D}_{t}$ 表示在迭代 $t$ 收集的轨迹和分数，$\mathcal{U}_{\mathrm{ES}}$ 表示Agentic ESOpt参数更新，$\mathcal{U}_{c}$ 表示提示 $c_{t}$ 的外部更新规则。然后一般的交替外循环可以更新两个空间为

$$
\theta_{t+1}=\mathcal{U}_{\mathrm{ES}}(\theta_{t};c_{t},\mathcal{D}_{t}),\quad c_{t+1}=\mathcal{U}_{c}(c_{t};\mathcal{D}_{t}).
$$

### 3.1 扰动半径 $\sigma$ 的余弦衰减

对于扰动 $\bm{\epsilon}\sim\mathcal{N}(0,I)$，Agentic ESOpt有效地优化了式(3)中的高斯平滑目标。然而，非零扰动半径引入了平滑偏差，其首阶项表征为：

###### 引理1（高斯平滑偏差）

假设 $J(\theta;c)$ 在 $\theta$ 的邻域内充分光滑。那么

$$
J_{\sigma}(\theta;c)=J(\theta;c)+\frac{\sigma^{2}}{2}\mathrm{Tr}\!\left(\nabla_{\theta}^{2}J(\theta;c)\right)+O(\sigma^{4}).
$$

证明见附录C.2。二阶项 $\mathrm{Tr}(\nabla_{\theta}^{2}J)$ 可以被视为正则化项。对于最大化目标，它惩罚尖锐的局部最优值，同时偏好更平坦的参数邻域。因此，更大的 $\sigma$ 引入更强的正则化，但也引入与原始目标更大的偏差。

现有的用于单轮大语言模型微调的进化策略方法通常在整个优化过程中使用固定的扰动半径[^27]，而不显式适应这种权衡。相反，Agentic ESOpt在 $T$ 个更新步骤中逐渐减小 $\sigma$，在早期使用更大的半径进行更广泛的探索和更强的正则化，在后期使用更小的半径增加利用并减少目标偏差，如下所示：

$$
\sigma_{t}=\sigma_{T}+(\sigma_{0}-\sigma_{T})\frac{1+\cos(\pi t/T)}{2},\qquad t=0,\ldots,T.
$$

对于训练时Agentic ESOpt，我们保留非零的 $\sigma_{T}$ 以平衡利用与探索和正则化。相比之下，测试时计算关注当前任务的无偏结果而不是智能体的泛化，因此我们将 $\sigma_{T}$ 衰减到零，以在优化结束时最小化目标偏差。

## 4 智能体化数独：关于长视距可扩展性的受控实验

为了展示Agentic ESOpt相比智能体化强化学习在长视距可扩展性上的优势，本节首先从理论上分析它们的特性。然后我们在受控的多轮数独环境中评估所提出的Agentic ESOpt，其中最小可能视距长度 $H$ 由数独棋盘上的掩码数量严格定义。

##### 可扩展性的理论原因

考虑一条包含 $H$ 个动作的轨迹（Trajectory）$\bm{a}=(a_{1},\ldots,a_{H})$，其终端回报（Terminal Return）为 $R(\bm{a})$。一个简单的智能体化强化学习（Agentic RL）估计器（例如公式 (1)、公式 (2)）带有基线 $b$，其形式如下：

$$
\widehat{g}_{\mathrm{PG}}=(R(\bm{a})-b)\sum_{t=1}^{H}\nabla_{\theta}\log\pi_{\theta}(a_{t}\mid s_{t}).
$$

遵循 [^28] 的分析，假设回报与任何单个动作的相关性较弱，每步得分项近似不相关，且进化策略（ES）和策略梯度引起的回报变化相当。那么估计器方差将随视距（Horizon）近似线性增长，如下所示：

$$
\operatorname{Var}[\widehat{g}_{\mathrm{PG}}]\approx\operatorname{Var}[R(\bm{a})]\operatorname{Var}\!\left[\sum_{t=1}^{H}\nabla_{\theta}\log\pi_{\theta}(a_{t}\mid s_{t})\right]\propto H.
$$

Agentic ESOpt 为完整的推出（Rollout）采样一个参数扰动（Perturbation）$\bm{\epsilon}$。其对应的估计器为：

$$
\widehat{g}_{\mathrm{ES}}=(R(a(\theta+\sigma\bm{\epsilon}))-b)\frac{\bm{\epsilon}}{\sigma},\quad\operatorname{Var}[\widehat{g}_{\mathrm{ES}}]\approx\operatorname{Var}[R(\bm{a})]\operatorname{Var}\!\left[\frac{\bm{\epsilon}}{\sigma}\right].
$$

因此，参数-得分项 $\bm{\epsilon}/\sigma$ 不会对 $H$ 求和。Agentic ESOpt 将终端回报直接归因于一个连贯的策略变化，而无需让同一标量结果在 $H$ 个视距中进行区分。这种扩展性论证预测，随着有效视距的增长，Agentic ESOpt 具有相对优势。

重要的是，这种比较隔离了两个估计器中依赖视距的结构。智能体化强化学习在多个回合中累积动作-得分项，而 Agentic ESOpt 通过直接的参数空间搜索避免了这一点，其方差不会随 $H$ 显式增长。这预示着随着有效视距增长，参数空间归因（Parameter Attribution）将具有越来越大的相对优势。附录 C.3 中提供了完整的协方差展开和其他估计器难度来源的讨论。图 6 进一步可视化了随 $H^{*}$ 增加的局部参数空间邻域。尽管在更困难的设置中奖励对比度降低，但 ES 参数得分本身不会引入对回合的额外求和。

**可控长视距数独（Sudoku）**。为了实证检验预测的长视距扩展行为，我们设计了一个具有可控最小成功视距的多回合数独环境。该环境仅提供终端奖励（Terminal Reward），每个有效动作最多填充一个单元格。我们将任务 $x$ 的最短成功视距定义为 $H^{*}(x)=\min_{\bm{\tau}:R(\bm{\tau})=1}|\bm{\tau}|$。因此，遮蔽 5、10 或 15 个单元格分别给出 $H^{*}\in\{5,10,15\}$。实际视距 $H=|\bm{\tau}|$ 可能超过 $H^{*}$，因为存在无效或无效率的动作。我们的理论分析关注实际视距 $H$，而实验按 $H^{*}$ 分组。

**实验设置**。数独实验在 4 张 NVIDIA H100 80GB GPU 上进行。我们将 Agentic ESOpt 与两个 8-推出（Rollout）的智能体化 GRPO（群相对策略优化）运行（具有不同的采样配置）、智能体化 PPO（近端策略优化）[^15] 以及普通智能体化 ES 进行比较，在 Qwen3.5-4B 上测试。我们为每个 $H^{*}$ 创建了一个包含 32 个实例的训练数据集和一个包含 32 个实例的评估数据集。根据模型卡片中的建议，在所有评估中，温度（Temperature）为 0.7，top-p 为 0.8，top-k 为 20。我们使用任务成功率作为主要指标，其中仅当数独在交互预算内完全解决时，一个回合才算成功，并报告三次评估运行的均值和标准差。详细的训练和解码配置见附录 D.1.1。

(a) 最终评估成功率

(b) $H^{*}=5$ 评估曲线

(c) $H^{*}=15$ 评估曲线

图 3：按最小成功视距 $H^{*}$ 分组的智能体化数独性能。(a) 报告 PPO、更强的 GRPO-B 配置和 Agentic ESOpt 在 3 次运行中的最终成功率平均值及标准差误差条。Agentic ESOpt 值下方的红色/绿色注释报告其与更强智能体化 RL 结果的差异。(b) 和 (c) 显示了各方法的评估曲线。普通 ES 和 Agentic ESOpt 都使用 $G=32$。

表 1：按最小成功视距 $H^{*}$ 分组的智能体化数独最终评估成功率（$\times$ 100）和 GPU 内存需求。值报告为均值 $\pm$ 标准差。

| 方法 | GPU 内存需求 | $H^{*}=5$ | $H^{*}=10$ | $H^{*}=15$ |
| --- | --- | --- | --- | --- |
| Qwen3.5-27B | 51.75GB | $86.46\pm 3.90$ | $50.00\pm 2.55$ | $28.13\pm 2.55$ |
| Qwen3.5-4B | 8.41GB | $63.54\pm 7.80$ | $31.25\pm 4.42$ | $10.42\pm 1.47$ |
| + 智能体化 PPO <sup><math data-latex="*" display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo mathsize="0.900em">∗</mo> <annotation>*</annotation></semantics></math></sup> | 89.40GB | $\mathbf{90.63\pm 0.00}$ | $56.25\pm 0.00$ | $0.00\pm 0.00$ |
| + 智能体化 GRPO <sup><math data-latex="\dagger" display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo mathsize="0.900em">†</mo> <annotation>\dagger</annotation></semantics></math></sup> | 58.88GB | $80.21\pm 1.47$ | $44.79\pm 2.95$ | $30.21\pm 2.95$ |
| + 智能体化 GRPO <sup><math data-latex="\ddagger" display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo mathsize="0.900em">‡</mo> <annotation>\ddagger</annotation></semantics></math></sup> | 58.88GB | $85.42\pm 1.47$ | $\mathbf{67.71\pm 1.47}$ | $40.63\pm 2.55$ |
| + Agentic ESOpt (G=32) | 8.41GB | $89.58\pm 2.95$ | $62.50\pm 2.55$ | $\mathbf{53.13\pm 2.55}$ |
| 无 $\sigma$ 衰减（普通 ES） | 8.41GB | $85.42\pm 3.90$ | $55.21\pm 5.89$ | $42.71\pm 3.90$ |
| 无 $\sigma_{T}$（即 $\sigma_{t}=0$） | 8.41GB | $85.42\pm 3.90$ | $54.17\pm 3.90$ | $28.13\pm 2.55$ |

<sup><math data-latex="*" display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo mathsize="0.800em">∗</mo> <annotation>*</annotation></semantics></math></sup> PPO 训练使用温度 $1$，top-$p=1$ 和 top-$k=-1$。<sup><math data-latex="\dagger" display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup> GRPO 训练使用温度 $0.7$，top-$p=0.8$ 和 top-$k=20$。<sup><math data-latex="\ddagger" display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>‡</mo> <annotation>\ddagger</annotation></semantics></math></sup> GRPO 训练使用温度 $1$，top-$p=1$ 和 top-$k=-1$。所有 GRPO 和 PPO 评估，以及普通 ES 和 Agentic ESOpt 训练和评估推出，都使用推荐的温度-$0.7$，top-$p=0.8$，top-$k=20$。

表 1 和图 3 (a) 显示了一个随最小成功视距变化的明显趋势。在 $H^{*}=5$ 时，PPO 以 $90.63\%$ 领先，其次是 Agentic ESOpt 的 $89.58\%$ 和更强 GRPO 配置的 $85.42\%$。在 $H^{*}=10$ 时，GRPO 以 $67.71\%$ 领先，其次是 Agentic ESOpt 的 $62.50\%$ 和 PPO 的 $56.25\%$。在 $H^{*}=15$ 时，Agentic ESOpt 以 $53.13\%$ 成为最强，比 GRPO 的 $40.63\%$ 高出 12.50 个百分点，而 PPO 在仅有终端奖励的情况下变得无效，仅为 $0\%$。在稀疏终端奖励（Sparse Terminal Rewards）下，其评价器（Critic）无法学习到可靠的价值信号，导致产生的优势估计对信用分配（Credit Assignment）无信息。Agentic ESOpt 在小 $H^{*}$ 时并不占主导地位；其相对优势随着 $H^{*}$ 的增长而显现，此时智能体化 RL 的信用分配变得更困难。这种排序反转——$H^{*}=5$ 时 PPO 最强，$H^{*}=10$ 时 GRPO 最强，$H^{*}=15$ 时 Agentic ESOpt 最强——因此比统一的胜利更具信息量：它与视距依赖的优势区间一致，而非全局更强的优化器或单一有利的解码配置。

评估曲线图 3 (b, c) 阐明了这种分离如何发展。在 $H^{*}=5$ 时，各方法具有相似的学习曲线。然而，在 $H^{*}=15$ 时，GRPO 变体在训练的初始步骤中无法快速改进，而 PPO 则迅速崩溃。它们的性能与长视距信用分配不良有关，智能体化 RL 通常达到视距限制（对 $H^{*}=15$ 设置为 45）。在图 7 (b) 中，$H^{*}=15$ 设置有 45 回合的交互预算；GRPO 的回合数在步骤 60 左右有所改善，而 Agentic ESOpt 保持接近最小成功视距，最终为 15.41 回合。

与普通 ES 相比，Agentic ESOpt 可以在较长运行的后期得到改善，从步骤 60 的 $39.58\%$ 提高到步骤 100 的 $53.13\%$。这意味着由 sigma 调度提供的更早、更广泛的探索避免了算法陷入局部最优。如表 1 所示，在普通智能体化 ES 中移除余弦衰减（Cosine Decay）会导致更差的结果，而将最终 sigma 设置为 0 会导致过拟合和评估性能不佳。

<svg id="S4.SS0.SSS0.Px1.p8.pic1" height="161.83" overflow="visible" version="1.1" viewBox="0 0 650 161.83" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,161.83) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.84 L 0 139.41 C 0 142.08 2.17 144.24 4.84 144.24 L 645.16 144.24 C 647.83 144.24 650 142.08 650 139.41 L 650 4.84 C 650 2.17 647.83 0 645.16 0 L 4.84 0 C 2.17 0 0 2.17 0 4.84 Z" /></g><g style="--ltx-fill-color:#F9F7FB;" fill="#F9F7FB" fill-opacity="1.0"><path style="stroke:none" d="M 0.9 4.84 L 0.9 139.41 C 0.9 141.58 2.66 143.34 4.84 143.34 L 645.16 143.34 C 647.34 143.34 649.1 141.58 649.1 139.41 L 649.1 4.84 C 649.1 2.66 647.34 0.9 645.16 0.9 L 4.84 0.9 C 2.66 0.9 0.9 2.66 0.9 4.84 Z" /></g><g transform="matrix(1.0 0.0 0.0 1.0 6.92 141.48)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="matrix(1 0 0 1 0 0)"><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.7 L 0 16.65 C 0 18.7 1.66 20.36 3.7 20.36 L 76.61 20.36 C 78.66 20.36 80.31 18.7 80.31 16.65 L 80.31 3.7 C 80.31 1.66 78.66 0 76.61 0 L 3.7 0 C 1.66 0 0 1.66 0 3.7 Z" /></g><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 3.7 L 0.55 16.65 C 0.55 18.39 1.96 19.8 3.7 19.8 L 76.61 19.8 C 78.35 19.8 79.76 18.39 79.76 16.65 L 79.76 3.7 C 79.76 1.96 78.35 0.55 76.61 0.55 L 3.7 0.55 C 1.96 0.55 0.55 1.96 0.55 3.7 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.64 7.76)"><foreignobject style="--ltx-fo-width:5.22em;--ltx-fo-height:0.54em;--ltx-fo-depth:0.15em;font-size:9.06pt;" height="8.61" overflow="visible" transform="matrix(1 0 0 -1 0 6.73)" width="65.44"><span id="S4.SS0.SSS0.Px1.p8.pic1.1" style="font-size:70%;--ltx-fg-color:#FFFFFF;">核心要点 1</span></foreignobject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 7.82 6.43)"><foreignobject style="--ltx-fo-width:45.85em;--ltx-fo-height:9.09em;--ltx-fo-depth:0em;font-size:10pt;" height="125.84" overflow="visible" transform="matrix(1 0 0 -1 0 125.84)" width="634.43"><span id="S4.SS0.SSS0.Px1.p8.pic1.2" style="width:45.85em;"><span id="S4.SS0.SSS0.Px1.p8.pic1.2.1"><span id="S4.SS0.SSS0.Px1.p8.pic1.2.1.1" style="--ltx-fg-color:#000000;">在 OpenAI ES 分析 <sup id="fnref:28-4"><a href="#fn:28">28</a></sup> 中使用的<span id="S4.SS0.SSS0.Px1.p8.pic1.2.1.1.1">弱相关扩展假设</span>下，参数空间 ES 避免了动作空间策略梯度的显式视距级得分累积。数独实验展示了预测的<span id="S4.SS0.SSS0.Px1.p8.pic1.2.1.1.2">视距依赖交叉点</span>：Agentic ESOpt 在短视距时并非统一最强，但在最大的可控 <math data-latex="H^{*}" display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mi style="--ltx-fg-color:#000000;" mathcolor="#000000">H</mi> <mo style="--ltx-fg-color:#000000;" mathcolor="#000000">∗</mo></msup> <annotation encoding="application/x-tex">H^{*}</annotation></semantics></math> 时成为最强。</span></span></span></foreignobject></g></g></svg>

表 2：智能体化数独的训练计算量和实际时间。智能体化 GRPO 和 Agentic ESOpt 都在相同的四张 NVIDIA H100 GPU 上执行，并充分利用可用硬件。

<table><tr><td></td><td colspan="2"><math><semantics><mrow><msup><mi>H</mi> <mo>∗</mo></msup> <mo>=</mo> <mn>5</mn></mrow> <annotation>H^{*}=5</annotation></semantics></math></td><td colspan="2"><math><semantics><mrow><msup><mi>H</mi> <mo>∗</mo></msup> <mo>=</mo> <mn>10</mn></mrow> <annotation>H^{*}=10</annotation></semantics></math></td><td colspan="2"><math><semantics><mrow><msup><mi>H</mi> <mo>∗</mo></msup> <mo>=</mo> <mn>15</mn></mrow> <annotation>H^{*}=15</annotation></semantics></math></td></tr><tr><td>方法</td><td>FLOPs</td><td>时间</td><td>FLOPs</td><td>时间</td><td>FLOPs</td><td>时间</td></tr><tr><td>Qwen3.5-4B + 智能体化 GRPO</td><td>3.2 EFLOPs</td><td>5.4 小时</td><td>7.6 EFLOPs</td><td>13.1 小时</td><td>10.9 EFLOPs</td><td>19.0 小时</td></tr><tr><td>Qwen3.5-4B + Agentic ESOpt (<math><semantics><mi>G</mi> <annotation>G</annotation></semantics></math> =32)</td><td>3.1 EFLOPs</td><td>3.1 小时</td><td>6.3 EFLOPs</td><td>5.8 小时</td><td>9.4 EFLOPs</td><td>9.4 小时</td></tr></table>

##### 计算量和实际时间效率

内存和计算结果表征了 Agentic ESOpt 的两个互补效率属性。表 1 中的 GPU 内存结果衡量了其最小训练侧需求：Agentic ESOpt 仅需要 8.41GB，等于 Qwen3.5-4B 主干的推理内存，比 GRPO 的 58.88GB 需求低 85.7%。

ES 需要更大的种群（$G=32$ 对比 GRPO 中的 8-推出）；然而，它并未引入相应的模型计算劣势。如附录 C.5 所分析，$G=32$ 的 Agentic ESOpt 与 8-推出 GRPO 具有可比的模型 FLOPs 预算。如表 2 所示，在实际的四-H100 实现中，测量的实际时间结果表明，更大的 ES 种群在此环境中不会转化为额外的端到端开销。这种比较应从不同的效率轴进行解释：Agentic ESOpt 在内存效率上显著更高，在模型计算上具有竞争力，同时它有意地花费更多独立环境评估来替代参考模型评估和反向传播（Backpropagation）。

<svg id="S4.SS0.SSS0.Px2.p3.pic1" height="131.32" overflow="visible" version="1.1" viewBox="0 0 650 131.32" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,131.32) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.84 L 0 108.89 C 0 111.56 2.17 113.72 4.84 113.72 L 645.16 113.72 C 647.83 113.72 650 111.56 650 108.89 L 650 4.84 C 650 2.17 647.83 0 645.16 0 L 4.84 0 C 2.17 0 0 2.17 0 4.84 Z" /></g><g style="--ltx-fill-color:#F9F7FB;" fill="#F9F7FB" fill-opacity="1.0"><path style="stroke:none" d="M 0.9 4.84 L 0.9 108.89 C 0.9 111.06 2.66 112.83 4.84 112.83 L 645.16 112.83 C 647.34 112.83 649.1 111.06 649.1 108.89 L 649.1 4.84 C 649.1 2.66 647.34 0.9 645.16 0.9 L 4.84 0.9 C 2.66 0.9 0.9 2.66 0.9 4.84 Z" /></g><g transform="matrix(1.0 0.0 0.0 1.0 6.92 110.96)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="matrix(1 0 0 1 0 0)"><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.7 L 0 16.65 C 0 18.7 1.66 20.36 3.7 20.36 L 76.61 20.36 C 78.66 20.36 80.31 18.7 80.31 16.65 L 80.31 3.7 C 80.31 1.66 78.66 0 76.61 0 L 3.7 0 C 1.66 0 0 1.66 0 3.7 Z" /></g><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 3.7 L 0.55 16.65 C 0.55 18.39 1.96 19.8 3.7 19.8 L 76.61 19.8 C 78.35 19.8 79.76 18.39 79.76 16.65 L 79.76 3.7 C 79.76 1.96 78.35 0.55 76.61 0.55 L 3.7 0.55 C 1.96 0.55 0.55 1.96 0.55 3.7 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.64 7.76)"><foreignobject style="--ltx-fo-width:5.22em;--ltx-fo-height:0.54em;--ltx-fo-depth:0.15em;font-size:9.06pt;" height="8.61" overflow="visible" transform="matrix(1 0 0 -1 0 6.73)" width="65.44"><span id="S4.SS0.SSS0.Px2.p3.pic1.1" style="font-size:70%;--ltx-fg-color:#FFFFFF;">核心要点 2</span></foreignobject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 7.82 9.12)"><foreignobject style="--ltx-fo-width:45.85em;--ltx-fo-height:6.69em;--ltx-fo-depth:0.19em;font-size:10pt;" height="95.32" overflow="visible" transform="matrix(1 0 0 -1 0 92.63)" width="634.43"><span id="S4.SS0.SSS0.Px2.p3.pic1.2" style="width:45.85em;"><span id="S4.SS0.SSS0.Px2.p3.pic1.2.1"><span id="S4.SS0.SSS0.Px2.p3.pic1.2.1.1" style="--ltx-fg-color:#000000;">Agentic ESOpt 展示了<span id="S4.SS0.SSS0.Px2.p3.pic1.2.1.1.1">高模型侧效率</span>，仅需要推理级（Inference-Level）GPU 内存。尽管它评估的轨迹比智能体化 GRPO 更多（<math data-latex="G=32" display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi style="--ltx-fg-color:#000000;" mathcolor="#000000">G</mi> <mo style="--ltx-fg-color:#000000;" mathcolor="#000000">=</mo> <mn style="--ltx-fg-color:#000000;" mathcolor="#000000">32</mn></mrow> <annotation encoding="application/x-tex">G=32</annotation></semantics></math> 对比 8 次推出），节省的参考模型和反向传播计算使模型 FLOPs 和在数独中测量的实际时间保持可比，使表 4 中 27B 智能体的全参数适应成为可行。</span></span></span></foreignobject></g></g></svg>

## 5 用于训练时微调的 Agentic ESOpt

除了手工设计的数独，本节评估 Agentic ESOpt 在将大语言模型（LLM）微调（Fine-Tuning）为特定目的智能体方面的能力。我们针对智能体化数学推理中的 Python 计算工具使用、DocVQA 中的 OCR 和文档图像分析工具，以及 WebArena 中的网络智能体进行了微调。

### 5.1 在数学和 DocVQA 上的智能体化 ReAct 风格工具使用

我们首先对 LLM 进行微调，使其具备 ReAct 风格交互中的工具使用能力[^48]。该任务是一个相对长视距（Long-Horizon）的场景，通常视距长度 $H>10$。我们遵循 [^25] 中的设置和超参数，在 400 个 DAPO 问题上训练数学任务，并在 100 个保留的 DAPO 问题和分布外的 30 个 AIME 2026 问题上进行评估。对于 DocVQA，我们在 50 个问题的验证子集上微调智能体（Agent），并在保留的 100 个问题上进行评估。在评估所有方法时，我们将 MATH 和 DocVQA 的视距限制都设置为 50 轮，每轮的 max_token 分别限制为 4096（MATH）和 512（DocVQA）。此设置不会造成显著的截断。我们在 Qwen3.5-4B 上实现了两种微调方法：Agentic GRPO 和 Agentic ESOpt。此外，我们还实现了一种提示空间优化方法 Trace2Skill，该方法通过采样收集轨迹（Trajectory），并在最后为技能合成轨迹。我们还实现了微调方法与 Trace2Skill 的顺序组合，其中 Agentic GRPO + Trace2Skill 和 Agentic ESOpt + Trace2Skill 表示使用在 No Skill 提示下训练期间收集的轨迹进行技能蒸馏。GRPO 同样使用 8 次展开（rollout），我们为 Agentic ESOpt 设置 $G=16$，为 Trace2Skill 设置 16 个样本。如附录 C.5 所讨论的，在微调智能体时，Agentic ESOpt 仅消耗约 GRPO 一半的浮点运算量（FLOPs）。详细设置见第 D.2 和 D.3 节。

表 3：No Skill 和 Trace2Skill 上下文下的数学推理和 DocVQA 结果。Qwen3.5-27B No Skill 是仅评估基线；所有优化行均使用 Qwen3.5-4B。DAPO、AIME 2026 和 DocVQA 准确率以 % 报告。Mean@4 对四次采样的 Pass@1 准确率取平均；Pass@4 或 Max@4 表示四次采样中的最佳结果。

<table><tr><td></td><td></td><td colspan="4">数学推理</td><td colspan="4">DocVQA</td></tr><tr><td></td><td></td><td colspan="2">DAPO</td><td colspan="2">AIME 2026</td><td colspan="2">ANLS</td><td colspan="2">准确率</td></tr><tr><td>模型</td><td>方法</td><td>Mean@4</td><td>Pass@4</td><td>Mean@4</td><td>Pass@4</td><td>Mean@4</td><td>Max@4</td><td>Mean@4</td><td>Pass@4</td></tr><tr><td>Qwen3.5-27B</td><td>No Skill</td><td>65.8</td><td>87.0</td><td>76.7</td><td>93.3</td><td>0.5036</td><td>0.7843</td><td>51.8</td><td>69.0</td></tr><tr><td>Qwen3.5-4B</td><td>No Skill</td><td>63.0</td><td>86.0</td><td>55.8</td><td>86.7</td><td>0.3875</td><td>0.5981</td><td>40.3</td><td>53.0</td></tr><tr><td>Qwen3.5-4B</td><td>Agentic GRPO + No Skill</td><td>68.8</td><td>83.0</td><td>58.3</td><td>76.7</td><td>0.4627</td><td>0.5398</td><td>48.0</td><td>56.0</td></tr><tr><td>Qwen3.5-4B</td><td>Agentic ESOpt + No Skill</td><td>76.8</td><td>86.0</td><td>70.8</td><td>96.7</td><td>0.5043</td><td>0.6507</td><td>52.5</td><td>61.0</td></tr><tr><td></td><td>$\Delta$ vs No Skill</td><td>$\uparrow$ 13.8</td><td>0.0</td><td>$\uparrow$ 15.0</td><td>$\uparrow$ 10.0</td><td>$\uparrow$ 0.1168</td><td>$\uparrow$ 0.0526</td><td>$\uparrow$ 12.3</td><td>$\uparrow$ 8.0</td></tr><tr><td>Qwen3.5-4B</td><td>Trace2Skill</td><td>64.8</td><td>82.0</td><td>50.8</td><td>83.3</td><td>0.4612</td><td>0.6772</td><td>47.3</td><td>69.0</td></tr><tr><td>Qwen3.5-4B</td><td>Agentic GRPO + Trace2Skill</td><td>67.8</td><td>85.0</td><td>50.0</td><td>80.0</td><td>0.4743</td><td>0.5692</td><td>49.5</td><td>60.0</td></tr><tr><td>Qwen3.5-4B</td><td>Agentic ESOpt + Trace2Skill</td><td>77.3</td><td>86.0</td><td>71.7</td><td>96.7</td><td>0.5086</td><td>0.6654</td><td>52.8</td><td>61.0</td></tr><tr><td></td><td>$\Delta$ vs Trace2Skill</td><td>$\uparrow$ 12.5</td><td>$\uparrow$ 4.0</td><td>$\uparrow$ 20.8</td><td>$\uparrow$ 13.3</td><td>$\uparrow$ 0.0474</td><td>$\downarrow$ 0.0118</td><td>$\uparrow$ 5.5</td><td>$\downarrow$ 8.0</td></tr></table>

如表 3 所示，Agentic ESOpt 在 ReAct 风格的数学和 DocVQA 任务上始终优于匹配的 Agentic GRPO 基线。在没有演化技能的情况下，Agentic ESOpt 将 Qwen3.5-4B 基础模型在 DAPO 和 AIME 2026 Mean@4 上分别提升了 13.8 和 15.0 个百分点，并将 DocVQA Mean@4 准确率提升了 12.3 个百分点。在这三个指标上取平均，Agentic ESOpt 相对基础模型提升了 13.7 个百分点，相对 Agentic GRPO 提升了 8.3 个百分点。Agentic ESOpt 还能与 Trace2Skill 有效组合：组合方法在 DAPO、AIME 2026 和 DocVQA 上实现了 Qwen3.5-4B Mean@4 的最佳结果，展示了参数空间优化方法与外部技能优化组合的灵活性。

除了平均性能提升外，Agentic ESOpt 还保持了强大的 Pass@4 性能。我们在附录 D.3.4 中展示了完整的 Pass@K 性能，最高到 $k=32$。在各数据集上，Agentic ESOpt 变体在每个报告的 Pass@K 指标上都优于其匹配的 GRPO 基线[^50]。在 AIME 2026 和 DocVQA 上，Agentic ESOpt 相对基础 LLM 提升了 Pass@K 性能。这表明 Agentic ESOpt 执行了与 Agentic GRPO 不同且更有利的优化范式。

<svg id="S5.SS1.p4.pic1" height="98.11" overflow="visible" version="1.1" viewBox="0 0 650 98.11" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,98.11) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.84 L 0 75.68 C 0 78.35 2.17 80.52 4.84 80.52 L 645.16 80.52 C 647.83 80.52 650 78.35 650 75.68 L 650 4.84 C 650 2.17 647.83 0 645.16 0 L 4.84 0 C 2.17 0 0 2.17 0 4.84 Z" /></g><g style="--ltx-fill-color:#F9F7FB;" fill="#F9F7FB" fill-opacity="1.0"><path style="stroke:none" d="M 0.9 4.84 L 0.9 75.68 C 0.9 77.85 2.66 79.62 4.84 79.62 L 645.16 79.62 C 647.34 79.62 649.1 77.85 649.1 75.68 L 649.1 4.84 C 649.1 2.66 647.34 0.9 645.16 0.9 L 4.84 0.9 C 2.66 0.9 0.9 2.66 0.9 4.84 Z" /></g><g transform="matrix(1.0 0.0 0.0 1.0 6.92 77.75)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="matrix(1 0 0 1 0 0)"><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.7 L 0 16.65 C 0 18.7 1.66 20.36 3.7 20.36 L 76.61 20.36 C 78.66 20.36 80.31 18.7 80.31 16.65 L 80.31 3.7 C 80.31 1.66 78.66 0 76.61 0 L 3.7 0 C 1.66 0 0 1.66 0 3.7 Z" /></g><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 3.7 L 0.55 16.65 C 0.55 18.39 1.96 19.8 3.7 19.8 L 76.61 19.8 C 78.35 19.8 79.76 18.39 79.76 16.65 L 79.76 3.7 C 79.76 1.96 78.35 0.55 76.61 0.55 L 3.7 0.55 C 1.96 0.55 0.55 1.96 0.55 3.7 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.64 7.76)"><foreignobject style="--ltx-fo-width:5.22em;--ltx-fo-height:0.54em;--ltx-fo-depth:0.15em;font-size:9.06pt;" height="8.61" overflow="visible" transform="matrix(1 0 0 -1 0 6.73)" width="65.44"><span id="S5.SS1.p4.pic1.1" style="font-size:70%;--ltx-fg-color:#FFFFFF;">要点 3</span></foreignobject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 7.82 9.12)"><foreignobject style="--ltx-fo-width:45.85em;--ltx-fo-height:4.29em;--ltx-fo-depth:0.19em;font-size:10pt;" height="62.11" overflow="visible" transform="matrix(1 0 0 -1 0 59.42)" width="634.43"><span id="S5.SS1.p4.pic1.2" style="width:45.85em;"><span id="S5.SS1.p4.pic1.2.1"><span id="S5.SS1.p4.pic1.2.1.1" style="--ltx-fg-color:#000000;">Agentic ESOpt 在不牺牲 Pass@K 覆盖率的情况下提升平均性能：在数学和 DocVQA 任务中，两个 Agentic ESOpt 变体在每个报告的 Pass@4 指标上都优于其匹配的 Agentic GRPO 基线，这表明参数优化后成功轨迹的覆盖范围更广。</span></span></span></foreignobject></g></g></svg>

### 5.2 WebArena 的智能体化推理：模型可扩展性实验

接下来，我们在 WebArena-Lite 上评估 Agentic ESOpt，这是一个从 WebArena [^19] 派生的 165 任务浏览器基准。每个任务提供一个自然语言目标和一个交互式网站状态。智能体观察 WebRL 风格的文本浏览器表示，在基于 id 的动作空间中行动，并在完成轨迹后接收任务成功反馈[^55] [^26]。评估集包含 21 个 Reddit、32 个 GitLab、35 个 CMS、28 个 Map、46 个 OSS 和 3 个 Wikipedia 任务；表 4 中单独显示了五个主要类别，而 Wikipedia 仅包含在数据集平均值中。

本设置将 Agentic ESOpt 扩展到在四个 NVIDIA H100 80GB GPU 上运行的 Qwen3.5-27B。在此模型规模下，全参数优化（Full-Parameter Optimization）的 Agentic RL 在四个 H100 80GB GPU 上已不再实用。相比之下，Agentic ESOpt 保持了推理级（Inference-Level）的内存需求，使我们能够对 27B 网络智能体进行全参数适应。因此，本实验侧重于大模型可行性和真实世界智能体适应，补充了第 4 节和第 5.1 节中受控的 ES 与 RL 对比。我们比较了使用相同 Qwen3.5-27B 主干的两对设置：No Skill 对比 Agentic ESOpt + No Skill，以及 Trace2Skill 对比 Agentic ESOpt + Trace2Skill。两个 Agentic ESOpt 变体共享在没有技能的情况下学习的相同参数更新；组合变体另外从生成的轨迹中蒸馏出一个技能，而没有第二个技能条件的 ES 阶段或交替联合优化。我们还报告了闭源的 GPT-5.4、GPT-5.4-mini 和 GPT-5.4-nano 作为参考点。完整的设置细节和评估曲线见附录 D.4 和 D.4.4，任务提示和蒸馏技能见 E.4、E.4.2 和 E.4.3。

所有 Qwen3.5-27B 的 Agentic ESOpt、Trace2skill 和评估展开都使用温度 $0.7$、2048 个 token 的生成预算、WebRL 风格观察和 WebRL 基于 id 的动作。Agentic ESOpt 设置种群大小（Population Size）$G=8$，使用采样训练案例上的任务成功作为奖励。Trace2skill 也为每个问题采样 8 次运行。最终结果在官方 165 任务 WebArena-Lite 评估集上进行评估[^19]。

表 4：WebArena-Lite 成功率（%）。我们报告 3 次运行的平均值和标准差。相对于配对基线改进的 Agentic ESOpt 单元格以绿色阴影标记；粗体表示每列中的最佳结果。显示的五个网站类别包含 162 个任务；其余 Wikipedia 任务包含在数据集平均值中。

<table><tr><td>模型</td><td>方法</td><td>Reddit (21)</td><td>GitLab (32)</td><td>CMS (35)</td><td>Map (28)</td><td>OSS (46)</td><td>数据集平均</td></tr><tr><td colspan="8">强冻结基线（仅评估）</td></tr><tr><td>GPT-5.4</td><td>No Skill</td><td>47.62</td><td>46.88</td><td>46.67</td><td>19.05</td><td>21.01</td><td>34.14 $\pm$ 0.76</td></tr><tr><td>GPT-5.4-mini</td><td>No Skill</td><td>39.68</td><td>29.17</td><td>30.48</td><td>13.10</td><td>13.77</td><td>23.23 $\pm$ 1.14</td></tr><tr><td>GPT-5.4-nano</td><td>No Skill</td><td>39.68</td><td>27.08</td><td>19.05</td><td>11.90</td><td>8.70</td><td>18.79 $\pm$ 0.99</td></tr><tr><td>Qwen3.5-27B</td><td>No Skill</td><td>50.79</td><td>35.42</td><td>41.90</td><td>8.33</td><td>21.01</td><td>29.47 $\pm$ 1.14</td></tr><tr><td>Qwen3.5-27B</td><td>Agentic ESOpt + No Skill</td><td>49.21</td><td>43.75</td><td>49.52</td><td>14.29</td><td>30.43</td><td>36.16 $\pm$ 0.70</td></tr><tr><td></td><td>$\Delta$ vs No Skill</td><td>$\downarrow$ 1.58</td><td>$\uparrow$ 8.33</td><td>$\uparrow$ 7.62</td><td>$\uparrow$ 5.96</td><td>$\uparrow$ 9.42</td><td>$\uparrow$ 6.69</td></tr><tr><td>Qwen3.5-27B</td><td>Trace2Skill</td><td>49.21</td><td>39.58</td><td>46.67</td><td>13.10</td><td>28.26</td><td>33.94 $\pm$ 3.37</td></tr><tr><td>Qwen3.5-27B</td><td>Agentic ESOpt + Trace2Skill</td><td>52.80</td><td>41.67</td><td>50.48</td><td>10.71</td><td>32.61</td><td>36.36 $\pm$ 0.86</td></tr><tr><td></td><td>$\Delta$ vs Trace2Skill</td><td>$\uparrow$ 3.59</td><td>$\uparrow$ 2.09</td><td>$\uparrow$ 3.81</td><td>$\downarrow$ 2.39</td><td>$\uparrow$ 4.35</td><td>$\uparrow$ 2.42</td></tr></table>

如表 4 所示，Agentic ESOpt 将 Qwen3.5-27B No Skill 基线从 29.47% 提升到 36.16%，提升了 6.69 个百分点。当与 Trace2Skill 组合时，它进一步将 Trace2Skill 基线从 33.94% 提升到 36.36%，展示了 Agentic ESOpt 在大型 LLM 上与技能空间优化组合的灵活性。

## 6 Agentic ESOpt 在测试时计算上的应用：自动启发式设计

传统的测试时计算（Test-Time Compute）通常在整个搜索过程中固定 LLM 参数，这在解决需要改变底层策略本身的任务时可能限制优化效果。由于 Agentic ESOpt 具有高度灵活性，它可以结合测试时计算来提升性能[^34]。

在本节中，我们评估 Agentic ESOpt 提升典型测试时计算问题的能力——自动启发式设计（Automatic Heuristic Design, AHD）[^16] [^54] [^12] [^17]。其中，代表性工作 EoH[^16] 通过维护一个启发式算法种群并应用交叉和变异算子，在固定训练数据集上改进这些算法的性能。在生成每个新启发式时，它向 LLM 智能体提供问题描述和 Python 函数签名；LLM 生成启发式代码，评估器在求解器中执行代码后返回标量目标值。Agentic ESOpt 可以插入到现有的 EoH 和纯采样程序中，而无需改变它们的外部搜索框架。因此，Agentic ESOpt + Sample 或 EoH 同时支持启发式空间搜索和参数空间适应。我们考虑两种 AHD 场景，并在相同评估预算下比较 Sample 和 EoH 与其对应的 Agentic ESOpt 变体。在构造式 AHD 中，启发式为 NP 难组合优化问题（定义见附录 C.4）做出局部决策：TSP、KP 和 ASP；在 ACO 风格的 AHD 中，算法为蚁群优化器提供启发式信息，用于 TSP、CVRP 和 BPP。TSP、CVRP 和 BPP 是最小化问题，而 KP 和 ASP 是最大化问题。

对于构造式任务，Optimal 行仅作为参考报告已知最优值 $x^{\star}$。我们将归一化最优性差距定义为 $g(x)=\frac{|x-x^{\star}|}{|x^{\star}|}$，并将 Agentic ESOpt 结果 $m$ 相对于其匹配基线 $b$ 的增益报告为 $\Delta=\frac{g(b)}{g(m)}-1.$。正式的启发式空间视角、完整搜索设置、ACO 风格结果、消融实验和准确的生成提示索引见附录 C.4、D.5、D.5.2 和 E.5。

对于 Agentic ESOpt + EoH，所有目标在内部表示为最小化成本，奖励为所选父代成本减去扰动子代成本。对于 Agentic ESOpt + Sample，奖励为负的子代成本。奖励在每个参数更新批次内进行 z-score 归一化。Agentic ESOpt 保留原始 EoH 算子，仅将参数更新附加到变异算子 $m1$ 和 $m2$ 上。我们报告最终代中的最佳候选，并对重复运行取平均。表 17 提供了代表性 TSP 和 KP 设置的 20 次运行显著性分析。

表 5：在总评估次数 $T\in\{1000,2000\}$ 下设计构造式启发式。TSP 为最小化问题；KP 和 ASP 为最大化问题。Optimal 行显示最优目标值，并非实验基线。Sample 和 EoH 各自与其 Agentic ESOpt 对应版本配对；$\Delta$ 行报告文中定义的差距比率增益。基线和 Agentic ESOpt 在所有运行中均使用 LLaMA-3.1-8B-Instruct。

<table><tr><td>Method</td><td>TSP <math><semantics><mrow><mi>N</mi> <mo>=</mo> <mn>20</mn></mrow> <annotation>N=20</annotation></semantics></math></td><td>TSP <math><semantics><mrow><mi>N</mi> <mo>=</mo> <mn>50</mn></mrow> <annotation>N=50</annotation></semantics></math></td><td>KP <math><semantics><mrow><mrow><mi>N</mi> <mo>=</mo> <mn>50</mn></mrow><mo>,</mo><mrow><mi>W</mi> <mo>=</mo> <mn>12.5</mn></mrow></mrow> <annotation>N=50,W=12.5</annotation></semantics></math></td><td>KP <math><semantics><mrow><mrow><mi>N</mi> <mo>=</mo> <mn>100</mn></mrow><mo>,</mo><mrow><mi>W</mi> <mo>=</mo> <mn>25</mn></mrow></mrow> <annotation>N=100,W=25</annotation></semantics></math></td><td>ASP <math><semantics><mrow><mrow><mi>N</mi> <mo>=</mo> <mn>15</mn></mrow><mo>,</mo><mrow><mi>W</mi> <mo>=</mo> <mn>10</mn></mrow></mrow> <annotation>N=15,W=10</annotation></semantics></math></td><td>ASP <math><semantics><mrow><mrow><mi>N</mi> <mo>=</mo> <mn>21</mn></mrow><mo>,</mo><mrow><mi>W</mi> <mo>=</mo> <mn>15</mn></mrow></mrow> <annotation>N=21,W=15</annotation></semantics></math></td></tr><tr><td>Optimal</td><td>3.8199</td><td>5.6750</td><td>20.0370</td><td>40.2710</td><td>3,003</td><td>43,596</td></tr><tr><td>Greedy Construct</td><td>4.4797</td><td>6.9590</td><td>19.9850</td><td>40.2250</td><td>1,530</td><td>15,050</td></tr><tr><td colspan="7">Total Evaluations: <math><semantics><mrow><mi>T</mi> <mo>=</mo> <mn>1000</mn></mrow> <annotation>T=1000</annotation></semantics></math></td></tr><tr><td>Sample</td><td>4.3286</td><td>6.7110</td><td>19.9896</td><td>40.2297</td><td>2,753</td><td>30,336.67</td></tr><tr><td>Agentic ESOpt + Sample</td><td>4.2336</td><td>6.5488</td><td>19.9899</td><td>40.2314</td><td>2,729</td><td>31,082.67</td></tr><tr><td><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> vs Sample</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 22.96%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 18.56%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 0.79%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 4.42%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 8.76%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 5.96%</td></tr><tr><td>EoH</td><td>4.2481</td><td>6.5450</td><td>19.9958</td><td>40.2320</td><td>2,760</td><td>28,465.67</td></tr><tr><td>Agentic ESOpt + EoH</td><td>4.2170</td><td>6.4631</td><td>20.0007</td><td>40.2351</td><td>2,770</td><td>30,887.33</td></tr><tr><td><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> vs EoH</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 7.83%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 10.39%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 13.59%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 8.65%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 4.67%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 29.18%</td></tr><tr><td colspan="7">Total Evaluations: <math><semantics><mrow><mi>T</mi> <mo>=</mo> <mn>2000</mn></mrow> <annotation>T=2000</annotation></semantics></math></td></tr><tr><td>Sample</td><td>4.2585</td><td>6.6008</td><td>19.99118</td><td>40.2320</td><td>2,759</td><td>30,123.33</td></tr><tr><td>Agentic ESOpt + Sample</td><td>4.2098</td><td>6.5332</td><td>19.99120</td><td>40.2320</td><td>2,751</td><td>30,269.33</td></tr><tr><td><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> vs Sample</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 12.51%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 7.88%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 0.04%</td><td>0.00%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 3.17%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 1.10%</td></tr><tr><td>EoH</td><td>4.2165</td><td>6.4706</td><td>19.9984</td><td>40.2356</td><td>2,764</td><td>28,016.00</td></tr><tr><td>Agentic ESOpt + EoH</td><td>4.1799</td><td>6.4442</td><td>19.9987</td><td>40.2358</td><td>2,784</td><td>30,124.67</td></tr><tr><td><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> vs EoH</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 10.19%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 3.43%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 0.81%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 0.48%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 9.92%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 24.36%</td></tr></table>

如表 5 所示，Agentic ESOpt + EoH 在两种评估预算下均改进了所有六个构造式测试集。Agentic ESOpt + Sample 在 12 次比较中改进了 9 次，有 1 次持平和 2 次退化。在两种构造式基线中，Agentic ESOpt 因此在 24 次匹配比较中改进了 21 次。ACO 风格的结果和额外的消融实验在 D.5.2 节中报告。综合构造式和 ACO 风格设置，Agentic ESOpt 在 12 个测试集和 6 种场景的 36 次匹配方法-预算比较中改进了 28 次。

<svg id="S6.p6.pic1" height="164.52" overflow="visible" version="1.1" viewBox="0 0 650 164.52" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,164.52) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.84 L 0 142.1 C 0 144.77 2.17 146.93 4.84 146.93 L 645.16 146.93 C 647.83 146.93 650 144.77 650 142.1 L 650 4.84 C 650 2.17 647.83 0 645.16 0 L 4.84 0 C 2.17 0 0 2.17 0 4.84 Z" /></g><g style="--ltx-fill-color:#F9F7FB;" fill="#F9F7FB" fill-opacity="1.0"><path style="stroke:none" d="M 0.9 4.84 L 0.9 142.1 C 0.9 144.27 2.66 146.03 4.84 146.03 L 645.16 146.03 C 647.34 146.03 649.1 144.27 649.1 142.1 L 649.1 4.84 C 649.1 2.66 647.34 0.9 645.16 0.9 L 4.84 0.9 C 2.66 0.9 0.9 2.66 0.9 4.84 Z" /></g><g transform="matrix(1.0 0.0 0.0 1.0 6.92 144.17)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="matrix(1 0 0 1 0 0)"><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.7 L 0 16.65 C 0 18.7 1.66 20.36 3.7 20.36 L 76.61 20.36 C 78.66 20.36 80.31 18.7 80.31 16.65 L 80.31 3.7 C 80.31 1.66 78.66 0 76.61 0 L 3.7 0 C 1.66 0 0 1.66 0 3.7 Z" /></g><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 3.7 L 0.55 16.65 C 0.55 18.39 1.96 19.8 3.7 19.8 L 76.61 19.8 C 78.35 19.8 79.76 18.39 79.76 16.65 L 79.76 3.7 C 79.76 1.96 78.35 0.55 76.61 0.55 L 3.7 0.55 C 1.96 0.55 0.55 1.96 0.55 3.7 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.64 7.76)"><foreignobject style="--ltx-fo-width:5.22em;--ltx-fo-height:0.54em;--ltx-fo-depth:0.15em;font-size:9.06pt;" height="8.61" overflow="visible" transform="matrix(1 0 0 -1 0 6.73)" width="65.44"><span id="S6.p6.pic1.1" style="font-size:70%;--ltx-fg-color:#FFFFFF;">要点 4</span></foreignobject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 7.82 9.12)"><foreignobject style="--ltx-fo-width:45.85em;--ltx-fo-height:9.09em;--ltx-fo-depth:0.19em;font-size:10pt;" height="128.53" overflow="visible" transform="matrix(1 0 0 -1 0 125.84)" width="634.43"><span id="S6.p6.pic1.2" style="width:45.85em;"><span id="S6.p6.pic1.2.1"><span id="S6.p6.pic1.2.1.1" style="--ltx-fg-color:#000000;">AHD 表明，Agentic ESOpt 可以将测试时计算从对冻结 LLM 策略的搜索扩展到启发式-参数耦合优化。在不修改外部 Sample 或 EoH 搜索框架且匹配评估预算的情况下，Agentic ESOpt 改进了 36 次比较中的 28 次，表明轻量级黑盒参数适应可以系统性地增强现有的测试时搜索。</span></span></span></foreignobject></g></g></svg>

## 7 讨论

我们使用本节来阐明参数空间 ES 最具吸引力的应用场景、当前证据的范围以及扩展 Agentic ESOpt 的主要方向。

### 7.1 模型规模的种群缩放

种群规模（Population Size）$G$ 是 ES 的关键超参数。我们当前的结果提供了初步证据，表明更强的骨干模型对小种群的敏感性可能较低。如图 5 所示，对于 4B 模型，在相同的 15 轮数独设置上，将种群从 $G=8$ 增加到 $G=16$ 使最佳测试准确率从 $5.10\%$ 提升到 $35.42\%$，最终测试成功率从 $2.95\%$ 提升到 $22.92\%$。然而，9B 模型对大种群的依赖性显著降低。将 $G$ 从 8 增加到 16 使最佳测试成功率从 $30.21\%$ 变为 $37.50\%$，相对提升 $24.1\%$，而最终测试成功率保持在 $30.21\%$，相对提升为零。

![Refer to caption](imgs/img-003-scaling_es_agent.png)

图 4：Vanilla-ES 在 15 轮数独上的种群敏感性。相对变化比较 G = 16 G=16 与 G=8。

如图 5 所示，一个合理的解释是，预训练的更强骨干模型将导致更有能力的局部区域，因此更大比例的邻近扰动会产生对更新有信息价值的行为。这一解释与文献[^5]的发现一致，即有用的行为多样性在更强的预训练模型周围变得更密集。这支持了使用 Agentic ESOpt 以更少的 FLOPs 微调更大的先进 LLM 的前景；我们将建立通用缩放定律并在前沿 LLM 上实现 Agentic ESOpt 作为未来工作。

<svg id="S7.SS1.p3.pic1" height="98.11" overflow="visible" version="1.1" viewBox="0 0 650 98.11" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,98.11) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.84 L 0 75.68 C 0 78.35 2.17 80.52 4.84 80.52 L 645.16 80.52 C 647.83 80.52 650 78.35 650 75.68 L 650 4.84 C 650 2.17 647.83 0 645.16 0 L 4.84 0 C 2.17 0 0 2.17 0 4.84 Z" /></g><g style="--ltx-fill-color:#F9F7FB;" fill="#F9F7FB" fill-opacity="1.0"><path style="stroke:none" d="M 0.9 4.84 L 0.9 75.68 C 0.9 77.85 2.66 79.62 4.84 79.62 L 645.16 79.62 C 647.34 79.62 649.1 77.85 649.1 75.68 L 649.1 4.84 C 649.1 2.66 647.34 0.9 645.16 0.9 L 4.84 0.9 C 2.66 0.9 0.9 2.66 0.9 4.84 Z" /></g><g transform="matrix(1.0 0.0 0.0 1.0 6.92 77.75)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="matrix(1 0 0 1 0 0)"><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.7 L 0 16.65 C 0 18.7 1.66 20.36 3.7 20.36 L 76.61 20.36 C 78.66 20.36 80.31 18.7 80.31 16.65 L 80.31 3.7 C 80.31 1.66 78.66 0 76.61 0 L 3.7 0 C 1.66 0 0 1.66 0 3.7 Z" /></g><g style="--ltx-fill-color:#7030A0;" fill="#7030A0" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 3.7 L 0.55 16.65 C 0.55 18.39 1.96 19.8 3.7 19.8 L 76.61 19.8 C 78.35 19.8 79.76 18.39 79.76 16.65 L 79.76 3.7 C 79.76 1.96 78.35 0.55 76.61 0.55 L 3.7 0.55 C 1.96 0.55 0.55 1.96 0.55 3.7 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.64 7.76)"><foreignobject style="--ltx-fo-width:5.22em;--ltx-fo-height:0.54em;--ltx-fo-depth:0.15em;font-size:9.06pt;" height="8.61" overflow="visible" transform="matrix(1 0 0 -1 0 6.73)" width="65.44"><span id="S7.SS1.p3.pic1.1" style="font-size:70%;--ltx-fg-color:#FFFFFF;">要点 5</span></foreignobject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 7.82 9.12)"><foreignobject style="--ltx-fo-width:45.85em;--ltx-fo-height:4.29em;--ltx-fo-depth:0.19em;font-size:10pt;" height="62.11" overflow="visible" transform="matrix(1 0 0 -1 0 59.42)" width="634.43"><span id="S7.SS1.p3.pic1.2" style="width:45.85em;"><span id="S7.SS1.p3.pic1.2.1"><span id="S7.SS1.p3.pic1.2.1.1" style="--ltx-fg-color:#000000;">该实验提供了初步的种群敏感性证据：将 <math data-latex="G" display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi style="--ltx-fg-color:#000000;" mathcolor="#000000">G</mi> <annotation encoding="application/x-tex">G</annotation></semantics></math> 加倍使 4B 模型的最终测试成功率变化 <math data-latex="+677.0\%" display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo style="--ltx-fg-color:#000000;" mathcolor="#000000">+</mo> <mrow><mn style="--ltx-fg-color:#000000;" mathcolor="#000000">677.0</mn> <mo style="--ltx-fg-color:#000000;" mathcolor="#000000">%</mo></mrow></mrow> <annotation encoding="application/x-tex">+677.0\%</annotation></semantics></math>，但 9B 模型为 <math data-latex="0.0\%" display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn style="--ltx-fg-color:#000000;" mathcolor="#000000">0.0</mn> <mo style="--ltx-fg-color:#000000;" mathcolor="#000000">%</mo></mrow> <annotation encoding="application/x-tex">0.0\%</annotation></semantics></math>。这表明更强的骨干模型可能需要更少的采样方向。</span></span></span></foreignobject></g></g></svg>

## 8 结论

本文提出了 Agentic ESOpt，一个用于微调长视距（Long-Horizon）LLM 智能体的全参数进化策略（Evolution Strategies）框架，仅需最少的推理级 GPU 内存。我们认为，对于长视距智能体，Agentic ESOpt 是比传统智能体化强化学习（Agentic RL）更自然的优化范式。随着交互视距增长和反馈变得稀疏，策略梯度必须将轨迹级结果分配到不断扩展的动作序列上。Agentic ESOpt 则将结果归因于连贯的策略扰动，避免了显式的视距级动作-得分累积。

我们的实验支持这一转变。在受控的数独任务上，Agentic PPO 和 Agentic GRPO 在较短视距下保持竞争力，而随着最小成功视距增长，Agentic ESOpt 变得最强；这一优势进一步扩展到 ReAct 风格的数学和 DocVQA 任务。Agentic ESOpt 还通过仅前向优化提供了模型缩放优势。其推理级内存占用使得在四块 H100 GPU 上对 Qwen3.5-27B WebArena 智能体进行全参数适应成为可能。

除了训练时微调，其灵活性自然地将参数适应集成到测试时计算的更广泛优化循环中。Agentic ESOpt 改进了技能优化和测试时启发式搜索，包括在固定评估预算下 36 次匹配 AHD 比较中的 28 次。

综合来看，这些结果将 ES 定位为不是 RL 的廉价替代品，而是针对长视距、稀疏反馈 LLM 智能体更匹配的优化机制。随着智能体系统向更长交互、更大模型和越来越复杂的提示空间交互组件发展，Agentic ESOpt 可能成为智能体微调和在线自我改进的核心范式。

当前研究的局限性在附录 A.1 中讨论，未来工作在附录 A.2 中概述。

## 致谢

我们衷心感谢新加坡国立大学的 Jiaying Wu、Penghui Qi、Zichen Liu 和 Ziqiao Meng，以及 humans& 的 Ziang Li 对方法论和论文撰写提出的重要意见。

## 参考文献

## 附录目录

## 附录 A 局限性与未来工作

### A.1 局限性

##### 引入新的超参数

与智能体化 RL 相比，Agentic ESOpt 引入了扰动半径 $\sigma$ 的超参数。它们的最优值可能取决于 LLM、奖励分布和环境，这可能比使用固定训练配方引入额外的调优成本。然而，如附录 D.6 总结的，我们在不同任务和 LLM 上使用了相对一致的配置，在 5 个实验中采用 $\sigma_{0}\approx 1e-3$ 和 $\alpha\approx 5e-4$。这表明这组参数具有高度通用性。我们将把 $\sigma$ 的自动调度作为未来工作。

##### 昂贵评估场景中的顾虑

Agentic ESOpt 以更多独立环境评估换取反向传播成本：在匹配的模型 FLOPs 下，它可以承担更多轨迹，因为每条轨迹只需要一次前向传递。当环境评估本身极其昂贵时，这种权衡可能变得不太有利，在这种情况下，推出成本而非模型计算可能主导总预算。

##### 持续学习仍不明确

当前实验建立了设置内适应。如文献[^11]所述，与通常进行稀疏更新的 GRPO 不同，ES 会在与优化目标无关的方向上导致随机游走，引发对持续学习的担忧。然而，尽管 ES 更新在严格意义上是密集的，但其幅度仍高度集中。如表 6 所示，Agentic ESOpt 更新仍然可以在一定阈值内保持稀疏性：$96.26\%$ 的最终参数更新保持在扰动尺度 $\sigma$ 内，$99.42\%$ 小于 $2.0\times 10^{-3}$。

表 6：使用扰动尺度 $\sigma_{t}=1.5\times 10^{-3}$ 在 Qwen3.5-27B 上对 WebArena 进行 Agentic ESOpt 后的参数更新幅度分布。

| Update-magnitude threshold | Fraction of parameters |
| --- | --- |
| $\|\Delta\theta\|\leq 1.0\times 10^{-3}$ | $83.68\%$ |
| $\|\Delta\theta\|\leq 1.5\times 10^{-3}\;(=\sigma)$ | $96.26\%$ |
| $\|\Delta\theta\|\leq 2.0\times 10^{-3}$ | $99.42\%$ |

### A.2 未来工作

##### 扩展到先进 LLM 和种群缩放定律

Agentic ESOpt 的推理级内存需求提供了对大幅更大 LLM 智能体进行全参数适应的直接路径。我们的 4B/9B 结果进一步表明，更强的骨干模型可能需要更少的扰动方向，促使对种群缩放与模型能力的系统研究。此外，作为未来工作，扩展到先进 LLM[^38] [^8] [^6] 也可能实现在涉及专有工具、API 和复杂工作流的工业场景中的在线适应。

##### 量化 ES 优化

量化可以进一步扩展 Agentic ESOpt 的可扩展性，但将密集参数扰动直接应用于量化权重需要尺度感知的噪声生成和数值稳定的更新[^29]。因此，为大规模 ES 微调开发与量化兼容的扰动和种子重放机制的 ES 专用基础设施是一个重要的系统方向。

##### 更紧密的技能-参数协同进化

我们的实验展示了与 Trace2Skill 的共享推出技能蒸馏以及 EoH 内的在线参数适应。一个自然的下一步是完全耦合的多步技能优化[^46] [^32]，其中外部上下文 $c_{t}$ 和模型参数 $\theta_{t}$ 在可比时间尺度上演化，允许技能和参数空间更新持续重塑彼此看到的数据分布。

## 附录 B 额外相关工作

### B.1 智能体化强化学习

强化学习（Reinforcement Learning, RL）为从可验证任务反馈优化模型参数提供了标准路径[^31] [^20] [^49] [^53] [^37]。这些方法在单轮 LLM 推理上展示了强大的性能。

最近的智能体场景带来了关于长上下文和长视距信用分配（Credit Assignment）的新挑战[^13] [^21]，有研究尝试将 RL 技术应用于智能体场景[^26] [^3] [^41]。一些工作强调了在 GRPO 风格基线中更好的推出[^3] [^9]，也有工作训练评论家网络并使用 PPO 实现更好的信用分配[^10]。这些方法可以改进 RL 的信用分配能力，但我们认为考虑 Agentic ESOpt 应被视为更直接的解决方案。基于梯度的智能体化 RL 通常需要存储推出、估计令牌级策略梯度、管理参考模型或 KL 约束的训练堆栈，并消耗大量内存。Agentic ESOpt 保持了参数适应的相同高层目标，但使用标量黑盒适应度（Fitness），因此环境轨迹不需要可微或保留用于反向传播。

### B.2 单轮 LLM 微调的进化策略

随着先进 LLM 配备越来越大的参数，在中等规模设备上微调它们变得不可承受，ES 已经作为高效的无梯度优化器出现，用于微调单轮 LLM。文献[^27]表明，ES 可以扩展到全参数 LLM 微调，在样本效率和训练稳定性上匹配或超越 GRPO。这种势头推动了基于 ES 的方法进入预训练[^18]、少样本适应[^5] [^14] [^7] [^4]，以及通过锐度感知机制实现内存高效调优[^36]。在这些方法中，ES 实现了更高的 GPU 内存效率，但性能低于 RL 方法。然而，我们认为 ES 的结构优势实际上在微调长视距智能体中更加突出，在这种情况下，Agentic ESOpt 可以优于智能体化 RL，而不仅仅是作为一种更便宜的替代品。

作为无梯度 LLM 微调的类似类别，零阶方法（Zeroth-Order Methods）[^24] [^52] 专注于监督微调（SFT），使它们难以应用于单轮和多轮推理场景。

### B.3 基于技能和记忆的自我改进与测试时计算

一大类自我改进智能体通过优化外部非参数组件来改进行为，同时保持底层 LLM 固定。Reflexion 将失败轨迹转换为口头反馈，为后续试验提供条件[^33]；Voyager 为开放式具身探索积累可执行技能[^39]；最近的记忆系统在长视距交互中组织和检索过去经验[^43] [^42]。更近期的方法如 SkillOpt 和 Trace2Skill 通过将技能文档本身视为可优化对象来明确这种优化[^46] [^25]。

测试时计算（Test-Time Compute）[^34] [^56] 方法遵循相关原则：它们通过在推理时进行额外的搜索[^47]、采样、反思[^23]或进化[^16]来提高任务性能，同时通常保持模型参数不变。这些方法轻量且灵活，但它们的优化仍然受到冻结策略可访问行为的限制。

Agentic ESOpt 通过引入参数适应作为额外的优化维度来补充这些方法。其黑盒更新可以重用已为技能、记忆或测试时搜索收集的相同轨迹级反馈，允许外部空间优化和参数空间优化一起进行。这使 Agentic ESOpt 能够加强现有的自我改进和测试时计算流程，而无需替换其原始的搜索或技能更新机制。

## 附录 C 理论分析与计算核算

### C.1 标量分数 ES 梯度的推导

设 $d$ 为 LLM 参数的数量，令

$$
q_{\sigma}(\vartheta\mid\theta)=\frac{1}{(2\pi\sigma^{2})^{d/2}}\exp\!\left(-\frac{\lVert\vartheta-\theta\rVert_{2}^{2}}{2\sigma^{2}}\right)
$$

表示以 $\theta$ 为中心的高斯扰动（Gaussian perturbation）的密度。等价地，$\vartheta=\theta+\sigma\bm{\epsilon}$，其中 $\bm{\epsilon}\sim\mathcal{N}(0,I)$。平滑后的目标（smoothed objective）可以写为

$$
J_{\sigma}(\theta;c)=\int J(\vartheta;c)q_{\sigma}(\vartheta\mid\theta)\,d\vartheta.
$$

假设微分和积分可以互换，对数导数恒等式（log-derivative identity）给出

$$
\displaystyle\nabla_{\theta}J_{\sigma}(\theta;c)
$$

$$
\displaystyle=\int J(\vartheta;c)\nabla_{\theta}q_{\sigma}(\vartheta\mid\theta)\,d\vartheta
$$

$$
\displaystyle=\mathbb{E}_{\vartheta\sim q_{\sigma}(\cdot\mid\theta)}\left[J(\vartheta;c)\nabla_{\theta}\log q_{\sigma}(\vartheta\mid\theta)\right].
$$

对于高斯密度，

$$
\nabla_{\theta}\log q_{\sigma}(\vartheta\mid\theta)=\frac{\vartheta-\theta}{\sigma^{2}}=\frac{\bm{\epsilon}}{\sigma},
$$

因此

$$
\nabla_{\theta}J_{\sigma}(\theta;c)=\frac{1}{\sigma}\mathbb{E}_{\bm{\epsilon}}\left[J(\theta+\sigma\bm{\epsilon};c)\bm{\epsilon}\right].
$$

现在使用轨迹（trajectory）定义展开内部目标：

$$
J(\theta+\sigma\bm{\epsilon};c)=\mathbb{E}_{\bm{\tau}\sim\pi_{\theta+\sigma\bm{\epsilon}}(\cdot\mid c)}\left[R(\bm{\tau})\right].
$$

因此全期望定律给出

$$
\nabla_{\theta}J_{\sigma}(\theta;c)=\frac{1}{\sigma}\mathbb{E}_{\begin{subarray}{c}\bm{\epsilon}\sim\mathcal{N}(0,I)\\
\bm{\tau}\sim\pi_{\theta+\sigma\bm{\epsilon}}(\cdot\mid c)\end{subarray}}\left[R(\bm{\tau})\bm{\epsilon}\right].
$$

因此，对于独立扰动 $\bm{\epsilon}_{i}$ 和条件展开 $\bm{\tau}_{i}\sim\pi_{\theta+\sigma\bm{\epsilon}_{i}}(\cdot\mid c)$，标准蒙特卡洛估计器（Monte Carlo estimator）为

$$
\widehat{g}_{\mathrm{ES}}=\frac{1}{G\sigma}\sum_{i=1}^{G}R(\bm{\tau}_{i})\bm{\epsilon}_{i},\qquad\mathbb{E}[\widehat{g}_{\mathrm{ES}}]=\nabla_{\theta}J_{\sigma}(\theta;c).
$$

因为 $\mathbb{E}[\bm{\epsilon}]=0$，任何与扰动无关的标量基线（baseline）$b$ 可以用 $R(\bm{\tau}_{i})-b$ 替换 $R(\bm{\tau}_{i})$ 而不改变这个期望。关键的是，该估计器仅使用环境产生标量 $R(\bm{\tau}_{i})$；不需要采样动作、环境转移或奖励函数的导数。Agentic ESOpt 使用的群体内奖励标准化（within-population reward standardization）是主文中描述的实用有限样本变体。

### C.2 高斯平滑偏差展开的证明

本小节证明引理 1。为了可读性，我们省略固定的外部上下文 $c$，并将相应的任务目标写为 $J(\theta)$。

###### 证明

定义高斯平滑目标

$$
J_{\sigma}(\theta)=\mathbb{E}_{\bm{\epsilon}\sim\mathcal{N}(0,I)}\left[J(\theta+\sigma\bm{\epsilon})\right].
$$

通过高斯分数函数恒等式（Gaussian score-function identity），或等价地分部积分，

$$
\nabla_{\theta}J_{\sigma}(\theta)=\frac{1}{\sigma}\mathbb{E}_{\bm{\epsilon}}\left[J(\theta+\sigma\bm{\epsilon})\bm{\epsilon}\right].
$$

因此，群体 ES 估计器对 $\nabla_{\theta}J_{\sigma}(\theta)$ 是无偏的。

为了比较平滑后的目标和原始目标，在 $\theta$ 附近展开 $J(\theta+\sigma\bm{\epsilon})$。中心高斯的奇数矩消失，且 $\mathbb{E}[\bm{\epsilon}\bm{\epsilon}^{\top}]=I$，给出

$$
J_{\sigma}(\theta)=J(\theta)+\frac{\sigma^{2}}{2}\operatorname{Tr}\!\left(\nabla_{\theta}^{2}J(\theta)\right)+O(\sigma^{4})=J(\theta)+\frac{\sigma^{2}}{2}\Delta_{\theta}J(\theta)+O(\sigma^{4}).
$$

对展开式求导得到

$$
\nabla_{\theta}J_{\sigma}(\theta)=\nabla_{\theta}J(\theta)+\frac{\sigma^{2}}{2}\nabla_{\theta}\Delta_{\theta}J(\theta)+O(\sigma^{4}).
$$

因此，相对于 $\nabla_{\theta}J(\theta)$ 的领先平滑偏差与 $\sigma^{2}$ 成正比，如引理 1 所述。∎

### C.3 长视距信用分配

设 $u_{t}=\nabla_{\theta}\log\pi_{\theta}(a_{t}\mid s_{t})$ 为第 $t$ 轮的策略分数（policy score），设 $\widetilde{R}=R-b$ 表示以基线为中心的终端回报（terminal return）。轨迹级策略梯度估计器（trajectory-level policy-gradient estimator）为

$$
\widehat{g}_{\mathrm{PG}}=\widetilde{R}\sum_{t=1}^{H}u_{t}.
$$

作为尺度近似，假设（i）回报与任何单个动作分数的相关性较弱，（ii）不同轮次的分数项近似不相关，（iii）它们的边际协方差在各轮次间可比，$\operatorname{Cov}(u_{t})\approx\Sigma_{u}$。那么

$$
\displaystyle\operatorname{Cov}(\widehat{g}_{\mathrm{PG}})
$$

$$
\displaystyle\approx\operatorname{Var}(\widetilde{R})\operatorname{Cov}\!\left(\sum_{t=1}^{H}u_{t}\right)
$$

$$
\displaystyle=\operatorname{Var}(\widetilde{R})\left[\sum_{t=1}^{H}\operatorname{Cov}(u_{t})+\sum_{s\neq t}\operatorname{Cov}(u_{s},u_{t})\right]
$$

$$
\displaystyle\approx H\,\operatorname{Var}(\widetilde{R})\Sigma_{u}.
$$

因此，在这些假设下，策略分数对估计器方差的贡献随实现视距（realized horizon）$H$ 近似线性增长。

对于 Agentic ESOpt，为完整轨迹采样一个扰动 $\bm{\epsilon}\sim\mathcal{N}(0,I)$，给出

$$
\widehat{g}_{\mathrm{ES}}=\widetilde{R}(\theta+\sigma\bm{\epsilon})\frac{\bm{\epsilon}}{\sigma}.
$$

在类似的弱相关近似下，

$$
\operatorname{Cov}(\widehat{g}_{\mathrm{ES}})\approx\operatorname{Var}(\widetilde{R})\operatorname{Cov}\!\left(\frac{\bm{\epsilon}}{\sigma}\right),
$$

其参数分数因子不包含对轮次的求和。因此该比较隔离了动作空间策略梯度中与视距相关的显式方差来源，而这在 ES 参数分数中不存在。

##### 尺度比较的范围

上述论证涉及两个估计器的与视距相关的分数结构。随着视距增长，Agentic ESOpt 仍可能变得更难优化，因为回报分布可能变得更稀疏或区分度更低。其估计器质量还可能取决于参数维度 $d$、扰动半径 $\sigma$、群体规模 $G$ 以及平滑目标的局部几何。因此，该分析并不断言 ES 的总方差独立于 $H$，也不断言 ES 普遍优于策略梯度。比较性预测是，当其他困难来源大致可比时，增加有效视距会为策略梯度引入额外的动作分数项累积，但不会为 ES 参数分数引入。这遵循 [^28] 的尺度视角。

##### 从实现视距 $H$ 到最小成功视距 $H^{*}$

上述理论量是实现的轨迹长度 $H$。在数独实验中，我们改为控制

$$
H^{*}(x)=\min_{\bm{\tau}:R(\bm{\tau})=1}|\bm{\tau}|,
$$

它给出每个成功轨迹视距的任务相关下界。因为每个有效的数独动作最多填充一个被掩码的单元格，掩码 5、10 和 15 个单元格分别产生 $H^{*}=5$、$10$ 和 $15$。失败、无效或无效率的动作可以使实现视距大大超过 $H^{*}$。

在这个自然任务系列中增加 $H^{*}$ 也会改变任务难度的各个方面，例如缺失信息量和正确完成所有必需动作的概率。因此，数独研究旨在测试实际预测，即随着最小交互需求的增长，参数空间和动作空间归因的相对行为会发生变化，而不是识别在其他方面相同的 MDP 中的纯延迟效应。图 7 中的执行轮次分析通过直接测量优化过程中实现视距如何演变来补充 $H^{*}$。

### C.4 AHD 作为启发式空间优化

自动启发式设计（Automatic heuristic design）在通常意义上不是语言生成任务：生成的文本仅通过其诱导的算法行为才有意义。因此，我们将 AHD 视为启发式空间上的优化 [^54]。设 $\mathcal{X}$ 为问题实例上的分布，设 $h\in\mathcal{H}$ 表示从 LLM 诱导的提议分布中采样的启发式。求解器 $A_{h}$ 使用 $h$ 为实例 $x$ 构造解 $y=A_{h}(x)$。理想化的目标是

$$
\min_{h\in\mathcal{H}}\mathbb{E}_{x\sim\mathcal{X}}\left[S(y,x)\right],\qquad y=A_{h}(x),
$$

其中 $S$ 是选择使得越大越好的有符号分数。本文涵盖 5 个 NP 困难组合优化问题，包括旅行商问题（Traveling Salesman Problem, TSP），寻求访问所有节点恰好一次的最短路径；容量受限车辆路径问题（Capacitated Vehicle Routing Problem, CVRP），最小化为具有容量约束的车辆服务客户需求的总路径成本；0-1 背包问题（0-1 Knapsack Problem, KP），在容量约束下最大化所选物品的总价值；装箱问题（Bin-Packing Problem, BPP），最小化打包所有物品所需的固定容量箱数；以及可接受集问题（Admissible Set Problem, ASP），寻求满足预定义可行性约束集的最大子集。

对于最小化问题如 TSP、CVRP 和 BPP，$S$ 是负目标值；对于最大化问题如 KP 和 ASP，$S$ 是原始目标值。因此，相关对象不是 $h$ 的文本形式，而是从问题状态到算法决策的诱导映射。

在构造式 AHD 中，$A_{h}$ 是序列决策规则。在部分解状态 $s_{t}$ 下，启发式诱导对可行动作的偏好，

$$
a_{t}\in\arg\max_{a\in\mathcal{A}(s_{t})}h(s_{t},a),\qquad s_{t+1}=T(s_{t},a_{t}),
$$

直到获得完整解。TSP 评估构造路径的长度，KP 评估在容量约束下所选物品集的价值，ASP 评估构造的可接受集的基数。这些任务测试 Agentic ESOpt 是否能改变 LLM 提议分布，以产生其局部决策累积成更好全局解的启发式。

在 ACO 风格的 AHD 中，$A_{h}$ 是随机的。启发式定义解组件上的能量或期望函数，这偏置如下形式的采样分布

$$
p_{h}(a_{t}\mid s_{t})\propto\exp\left(\beta h(s_{t},a_{t})\right)\cdot\Phi_{t}(s_{t},a_{t}),
$$

其中 $\Phi_{t}$ 表示非学习的搜索状态，如信息素或可行性项，$\beta$ 控制启发式偏置的强度。目标是此随机求解器采样的解的预期质量。Agentic ESOpt + EoH 保留这个外部优化问题，仅改变模型诱导的 $h$ 分布。

### C.5 训练 FLOPs 计算

我们遵循 [^5] 的 FLOPs 计算。对于具有 $P$ 个参数和包含 $\bar{L}$ 个模型处理令牌的轨迹，前向传播（forward pass）需要约 $2P\bar{L}$ FLOPs，而反向传播（backward pass）需要约 $4P\bar{L}$ FLOPs。

设 $T$、$B$ 和 $G$ 分别表示训练迭代次数、提示批大小（prompt batch size）和每个提示的展开或扰动方向数。特定方法的训练成本为

$$
\displaystyle\mathrm{FLOPs}_{\mathrm{ES}}
$$

$$
\displaystyle=T_{\mathrm{ES}}B_{\mathrm{ES}}G_{\mathrm{ES}}\underbrace{(2)}_{\text{策略前向}}P\bar{L}_{\mathrm{ES}}
$$

$$
\displaystyle=2T_{\mathrm{ES}}B_{\mathrm{ES}}G_{\mathrm{ES}}P\bar{L}_{\mathrm{ES}},
$$
$$
\displaystyle\mathrm{FLOPs}_{\mathrm{GRPO}}
$$

$$
\displaystyle=T_{\mathrm{GRPO}}B_{\mathrm{GRPO}}G_{\mathrm{GRPO}}\underbrace{(2+2+4)}_{\begin{subarray}{c}\text{策略前向}\\
+\;\text{参考前向}\\
+\;\text{策略反向}\end{subarray}}P\bar{L}_{\mathrm{GRPO}}
$$

$$
\displaystyle=8T_{\mathrm{GRPO}}B_{\mathrm{GRPO}}G_{\mathrm{GRPO}}P\bar{L}_{\mathrm{GRPO}},
$$
$$
\displaystyle\mathrm{FLOPs}_{\mathrm{PPO}}
$$

$$
\displaystyle=T_{\mathrm{PPO}}B_{\mathrm{PPO}}G_{\mathrm{PPO}}\underbrace{(2+2+2+4+4)}_{\begin{subarray}{c}\text{策略前向}+\text{参考前向}\\
+\;\text{评论器前向}+\text{策略反向}+\text{评论器反向}\end{subarray}}P\bar{L}_{\mathrm{PPO}}
$$

$$
\displaystyle=14T_{\mathrm{PPO}}B_{\mathrm{PPO}}G_{\mathrm{PPO}}P\bar{L}_{\mathrm{PPO}}.
$$

因此，对于相等长度的轨迹，Agentic ESOpt 每个采样轨迹的成本为一次策略前向传播，而 Agentic GRPO 约为四次前向传播等效，Agentic PPO 为七次。整个比较中排除了环境执行，因为比较测量的是模型侧训练 FLOPs。

##### 数独

在报告的数独比较中，Agentic ESOpt 和 GRPO 都使用 100 次更新轮次，提示批大小为 32，而 Agentic PPO 执行 500 次更新轮次。Agentic ESOpt 每个提示评估 $G_{\mathrm{ES}}=32$ 个扰动方向，而 Agentic GRPO 使用 $G_{\mathrm{GRPO}}=8$ 次展开。因此，假设轨迹长度相等，

$$
\frac{\mathrm{FLOPs}_{\mathrm{ES}}}{\mathrm{FLOPs}_{\mathrm{GRPO}}}\approx\frac{2\times 32}{8\times 8}=1.
$$

因此，ES 群体规模增加四倍恰好被其每轨迹模型侧成本降低四倍所补偿。Agentic PPO 限制在 500 个训练步骤以保持在可比计算预算内。对于表 2 中的测量值，我们进一步用实际模型处理令牌数替换名义轨迹长度，在 $H^{*}\in\{5,10,15\}$ 下，Agentic ESOpt 为 3.1/6.3/9.4 EFLOPs，Agentic GRPO 为 3.2/7.6/10.9 EFLOPs。Agentic GRPO 的额外 FLOPs 是因为该算法在长视距信用分配中的缺陷通常导致更长的轨迹。

##### 数学推理和 DocVQA

对于数学推理，Agentic ESOpt（$G=16$）和 Agentic GRPO（$G=8$）都覆盖 400 个样本的训练集一次。对于 DocVQA，两种方法同样覆盖 640 个训练样本。在可比轨迹长度下，它们的相对模型侧 FLOPs 因此为

$$
\frac{\mathrm{FLOPs}_{\mathrm{ES}}}{\mathrm{FLOPs}_{\mathrm{GRPO}}}\approx\frac{2\times 16}{8\times 8}=\frac{1}{2}.
$$

因此，在这两个任务上，Agentic ESOpt 需要约匹配的 Agentic GRPO 配置模型侧训练 FLOPs 的一半。

## 附录 D 实验设置和额外结果

本附录提供特定任务的实现细节和支持第 4-6 节实验的额外诊断。我们按照主要实验的顺序：数独、数学推理、DocVQA、WebArena-Lite 和自动启发式设计，然后是 Agentic ESOpt 超参数的跨设置总结。

### D.1 数独

#### D.1.1 长视距尺度的经验诊断

为了补充第 4 节和附录 C.3 中的长视距尺度分析，我们在数独上检查三个经验诊断，随着最小成功视距 $H^{*}$ 的增加。我们首先可视化相同 Qwen3.5-4B 检查点周围的局部参数空间奖励景观。然后，我们通过简单的示意性计算将每轮正确性与轨迹级成功联系起来，最后比较 Agentic GRPO 和 Agentic ESOpt 在训练期间的实现视距。

对于景观可视化，我们在 $d$ 维参数空间中选择两个正交方向，并评估由当前检查点周围这些方向张成的二维平面。热图中的每个点对应该切片中的扰动模型，其值是在数独批次上测量的高斯平滑平均终端奖励。由于数独仅提供二元终端反馈，此值可以解释为平滑的局部成功分数。理论要点不是更长的视距使奖励景观保持不变，而是 ES 不会在其参数分数因子中引入对轮次的额外显式累积。

![Refer to caption](imgs/img-004-qwen35_4b_mask5_centered_heatmap.png)

(a) 最小成功视距 $H^{*}=5$

如图 6 所示，增加 $H^{*}$ 显著降低了绝对奖励水平和奖励对比度，使得有信息的终端反馈越来越稀缺。尽管如此，即使在 $H^{*}=15$ 时，局部参数邻域中仍可见有用的变化。这与我们的理论表征一致：更长的视距仍使回报信号更困难，但 Agentic ESOpt 不会在其参数分数因子中产生额外的视距累积。

遵循 [^13] 的数独步骤准确度定义，当它用唯一目标值填充被掩码单元格时，轮次是正确的。为了将每轮正确性与轨迹级成功联系起来，考虑具有同质每轮正确性概率 $p$ 的最小长度成功路径。其成功概率为

$$
S_{H^{*}}=p^{H^{*}}.
$$

这个关系是示意性的，并未拟合测量的数独轨迹；它只是说明残余轮次错误如何随所需动作数量复合。

(a) 从每轮正确性到轨迹成功的示意性复合。

(b) $H^{*}=15$ 设置上测量的实现视距诊断。

图 7：长视距数独的轮次级诊断。左：在示意性关系 $S_{H^{*}}=p^{H^{*}}$ 下，随着最小成功视距增长，相同的每轮错误率产生更大的轨迹级惩罚。例如，$p=0.95$ 对 $H^{*}=5$、$10$ 和 $15$ 分别给出 $S_{H^{*}}=0.774$、$0.599$ 和 $0.463$。右：$H^{*}=15$ 设置上的实现视距诊断。Agentic GRPO 训练使用温度 $1$、top-$p=1$ 和 top-$k=-1$，而其评估使用温度 $0.7$、top-$p=0.8$ 和 top-$k=20$；Agentic ESOpt 训练和评估都使用后者解码器。Agentic GRPO 在第 60 步达到 45 轮交互预算，而 Agentic ESOpt 保持在 15 轮最小值附近，最终为 15.41 轮。

图 7 将理论论证与可观察行为联系起来。左图显示，即使很小的残余轮次错误随着所需视距增长也变得更具破坏性。右图显示相应的训练时分离：Agentic GRPO 逐渐累积过多轮次，而 Agentic ESOpt 保持接近最小长度解路径。这些诊断不是完整的因果证明，但它们与第 4 节中的估计器级区分一致：Agentic ESOpt 使用完整轨迹结果评估一个连贯的参数扰动，而不是通过不断扩展的动作级决策序列传播终端反馈。

#### D.1.2 训练配置

##### Agentic ESOpt 和 Vanilla ES

所有固定 ES 比较使用 Qwen3.5-4B 进行 100 代。在每代中，在相同的 32 个谜题批次上评估 $G=32$ 个全参数方向，并在使用 $\alpha=5\times 10^{-4}$ 更新之前对结果轨迹奖励进行 z 分数标准化。交互预算是掩码数的三倍——$H^{*}=5$、$10$ 和 $15$ 分别为 15、30 和 45 轮——每轮最多生成 64 个令牌。训练和评估使用温度 $0.7$、top-$p=0.8$、top-$k=20$、min-$p=0$、存在惩罚 $1.5$ 和重复惩罚 $1.0$。每 10 代进行一次评估，重复三次运行。

表 7：数独 ES 配置。所有配置使用 $G=32$，每代 32 个谜题，100 代，$\alpha=5\times 10^{-4}$。

| 配置 | 掩码数 | Sigma 调度 | 最大轮次 |
| --- | --- | --- | --- |
| Vanilla ES | 5 / 10 | $1.0\times 10^{-3}$ 常数 | 15 / 30 |
| Vanilla ES | 15 | $5.0\times 10^{-4}$ 常数 | 45 |
| Agentic ESOpt | 5 / 10 | $1.0\times 10^{-3}\!\rightarrow\!2.5\times 10^{-4}$ 余弦 | 15 / 30 |
| Agentic ESOpt | 15 | $7.0\times 10^{-4}\!\rightarrow\!5.0\times 10^{-4}$ 余弦 | 45 |

##### Agentic RL

两个 Agentic GRPO 配置都运行 100 轮展开-更新轮次，全局提示批次为 32，每个提示八次展开。它们使用展开微批次 8、策略训练微批次 2、学习率 $10^{-6}$、KL 系数 $10^{-3}$ 和裁剪系数 $0.2$。GRPO-A 使用温度 $0.7$、top-$p=0.8$ 和 top-$k=20$ 采样，而 GRPO-B 使用温度 $1$、top-$p=1$ 和 top-$k=-1$ 采样。两者在训练前和每 20 轮都使用 Agentic ESOpt 解码器进行评估，重复三次评估。一轮表示完整的展开批次，然后是相应的策略更新。

我们实现 Turn-PPO [^15] 作为 Agentic PPO 基线。Turn-PPO 使用单独的 Qwen3.5-4B 评论器（critic），用新添加的价值头初始化，进行轮次级优势估计（advantage estimation）。我们遵循与其他 Agentic RL 基线相同的交互和评估协议（训练中温度 $1$、top-$p=1$ 和 top-$k=-1$，评估中温度 $0.7$、top-$p=0.8$ 和 top-$k=20$），并最多训练 500 步以对齐 FLOPs。

#### D.1.3 Vanilla-ES 群体敏感性配置

图 5 中的 4B/9B 比较使用 $H^{*}=15$ 数独环境，进行 100 代，每 10 代评估 $G\in\{8,16\}$。为了隔离群体敏感性，所有运行都使用 Vanilla ES，常数扰动尺度 $\sigma=5\times 10^{-4}$，更新尺度 $\alpha=5\times 10^{-4}$，全参数扰动，群体 z 分数标准化，每个方向八个训练谜题。因此每代包含 $8G$ 个方向-谜题展开。扰动展开和评估使用温度 $0.7$、top-$p=0.8$ 和 top-$k=20$，每次评估重复三次。

环境提供二元终端奖励：仅当最终棋盘是保留所有给定单元格的完整合法数独解时，展开才获得 1。无效格式、超出范围值、尝试修改给定单元格和覆写使棋盘保持不变。这个固定设置使群体规模成为消融中唯一变化的 ES 超参数；不使用余弦扰动调度和中间奖励塑造。

### D.2 数学推理

除非另有说明，下面描述的所有特定方法配置都指训练。最终评估在 Agentic ESOpt、Agentic GRPO、Trace2Skill、它们的组合变体和所有其他报告基线共享的通用协议下进行。对于 Trace2Skill，我们为每个实例采样 16 次运行，并遵循 [^25] 的指示，每个实例最多使用一个错误轨迹来蒸馏技能。

#### D.2.1 Agentic ESOpt 训练配置

ReAct 风格数学推理的实验在四个 A100 80GB GPU 上进行。我们首先优化 No Skill Qwen3.5-4B 策略，并记录每次参数更新的种子以及后续提供给 Trace2Skill 的轨迹。对于最终评估，重新加载基础检查点，并确定性地重建更新序列。因此，No Skill 和技能条件评估共享相同的参数阶段，仅在外部技能上下文上有所不同。

表 8：数学推理的 Agentic ESOpt 训练配置。

| 参数 | 值 |
| --- | --- |
| 骨干网 | Qwen3.5-4B |
| ES 代数 | 25 |
| 群体规模 $G$ | 16 |
| 每个方向的案例数 | 16 |
| 更新尺度 $\alpha$ | $5\times 10^{-4}$ |
| $\sigma$ 调度 | $10^{-3}\!\rightarrow\!5\times 10^{-4}$ 余弦 |
| 奖励标准化 | 群体 z 分数 |
| 最大轮次 | 50 |
| 训练生成限制 | 每轮 4096 令牌 |

每个方向在 16 个问题上评估，每代给出 $16\times 16=256$ 个方向-问题轨迹。训练使用温度 $1$、top-$p=1$、top-$k=40$、min-$p=0$、存在惩罚 $2$ 和重复惩罚 $1$。轨迹奖励是解析最终答案的精确匹配。每 10 代进行一次中间评估，每个问题一个样本。

#### D.2.2 多轮 GRPO 训练配置

数学 Agentic GRPO 基线遵循与 Agentic ESOpt 相同的采样配置，学习率 $10^{-6}$，每个提示八次展开，KL 系数 $10^{-3}$（使用低方差 KL 形式），温度 $1$、top-$p=1$、top-$k=40$、存在惩罚 $2$ 和重复惩罚 $1$。

表 9：数学推理的多轮 Agentic GRPO 训练配置。

| 参数 | 值 |
| --- | --- |
| 骨干网 | Qwen3.5-4B |
| 训练记录 | 400 |
| 提示批大小 | 20 |
| 每个提示的展开数 | 8 |
| 训练轮数 | 1 |
| 训练轮次 | 20 |
| 总轨迹数 | 3,200 |
| 学习率 | $10^{-6}$ |
| KL 系数 | $10^{-3}$ |
| 最大交互 | 100 用户 + 100 助手轮次 |

对于全参数 Agentic GRPO 训练，我们适当压缩每轮 4096 令牌生成配置，以满足四个 A100 80GB GPU 上的训练内存约束。此调整仅适用于训练展开配置，不影响最终评估协议。400 个训练记录每轮形成 20 个提示批次。

#### D.2.3 通用评估配置

所有报告的方法和基线使用相同的最终评估协议。对于数学推理，我们在 100 个保留的 DAPO 问题和 30 个 AIME 2026 问题上评估，每个问题四个样本。每个轨迹最多允许 50 个交互轮次，每轮最多生成 4096 个令牌。我们对所有方法使用温度 $1$、top-$p=1$、top-$k=40$、min-$p=0$、存在惩罚 $2$ 和重复惩罚 $1$。这些评估设置由 Agentic ESOpt、Agentic GRPO、Trace2Skill、它们的组合变体和所有其他报告基线共享，独立于它们的特定方法训练配置。

### D.3 DocVQA

与数学推理一样，除非另有说明，以下配置指定相应的训练程序。对于 DocVQA，Agentic ESOpt 和 Agentic GRPO 在训练期间还使用完全对齐的交互和生成设置，所有报告的方法共享相同的最终评估配置。对于 Trace2Skill，我们为每个实例采样 16 次运行，并遵循 [^25] 的指示，每个实例最多使用一个错误轨迹和一个正确轨迹来蒸馏技能。

#### D.3.1 Agentic ESOpt 训练配置

ReAct 风格 DocVQA 的实验也在四个 A100 80GB GPU 上进行。DocVQA 实验遵循与数学相同的配对重建协议：No Skill 运行记录更新种子和源轨迹，最终评估在添加事后技能上下文之前重放相同的参数序列。

表 10：DocVQA 的 Agentic ESOpt 训练配置。

| 参数 | 值 |
| --- | --- |
| 骨干网 | Qwen3.5-4B |
| ES 代数 | 40 |
| 群体规模 $G$ | 16 |
| 每个方向的案例数 | 16 |
| 更新尺度 $\alpha$ | $5\times 10^{-4}$ |
| $\sigma$ 调度 | $10^{-3}\!\rightarrow\!5\times 10^{-4}$ 余弦 |
| 奖励标准化 | 群体 z 分数 |
| 最大轮次 | 50 |
| 生成限制 | 每轮 512 令牌 |
| 总生成上限 | 每轨迹 32,768 令牌 |

每代包含 $16\times 16=256$ 个方向-问题轨迹。训练使用温度 $1$、top-$p=1$、top-$k=40$、min-$p=0$、存在惩罚 $2$ 和重复惩罚 $1$。有效轨迹必须在最终答案之前至少包含一个解析的工具动作。连续 ANLS 用作训练奖励；当 ANLS 严格大于 $0.5$ 时，阈值准确度将答案计为正确。每 10 代进行一次中间评估，每个问题一个样本。

#### D.3.2 多轮 GRPO 训练配置

DocVQA Agentic GRPO 使用与 Agentic ESOpt 相同的采样配置和交互预算，以及与数学基线相同的优化器和 KL 设置。

表 11：DocVQA 的多轮 Agentic GRPO 训练配置。

| 参数 | 值 |
| --- | --- |
| 骨干网 | Qwen3.5-4B |
| 训练记录 | 50 |
| 提示批大小 | 4 |
| 每个提示的展开数 | 8 |
| 训练轮数 | 15 |
| 训练轮次 | 180 |
| 总轨迹数 | 5,760 |
| 学习率 | $10^{-6}$ |
| KL 系数 | $10^{-3}$ |
| 最大轮次 | 50 |
| 生成限制 | 每助手轮次 512 令牌 |
| 总生成上限 | 每轨迹 32,768 令牌 |

因此，Agentic GRPO 和 Agentic ESOpt 在训练期间使用相同的 50 轮交互限制、每轮 512 令牌生成限制、总轨迹生成上限和采样配置。最后两条记录不能填满四问题批次，因此被省略，每轮留下 12 个训练轮次。连续 ANLS 也用作训练奖励。

#### D.3.3 通用评估配置

所有报告的方法和基线也遵循 DocVQA 上相同的最终评估协议。我们在相同的 100 个保留问题集上评估，每个问题四个样本。每个轨迹限制为 50 个交互轮次，每轮 512 个生成令牌，总共 32,768 个生成令牌。采样对每种方法使用温度 $1$、top-$p=1$、top-$k=40$、min-$p=0$、存在惩罚 $2$ 和重复惩罚 $1$。因此，Agentic ESOpt、Agentic GRPO、Trace2Skill、它们的组合变体和所有其他报告基线使用相同的评估设置。此外，Agentic ESOpt 和 Agentic GRPO 的交互和生成预算在 DocVQA 训练期间完全对齐。

#### D.3.4 数学和 DocVQA 的重复采样配置

图 8 将表 3 中的四样本结果扩展到重复采样预算达 $k=32$，每种方法使用相同的评估解码器。

图 8：DAPO、AIME 2026、DocVQA ANLS 和 DocVQA 准确度的重复采样配置。$k=1$ 标记使用报告的 Mean@4 值作为单样本性能的估计，而星号显示原始报告的 Pass@4 或 Max@4 值。曲线报告 $k\in\{2,8,16,32\}$ 的新估计最佳-$k$ 结果；省略新计算的 $k=4$ 曲线点，以使原始四样本测量在视觉上保持独特。

在所有四个面板中，随着采样预算增加，Agentic ESOpt 保持在匹配的 Agentic GRPO 基线之上。在 AIME 2026 和两个 DocVQA 指标上，它还在更大的 $k$ 下保持了相对于基础策略的优势，表明平均性能增益不是通过折叠重复采样覆盖率获得的。

#### D.3.5 DocVQA 训练阶段轮次诊断

图 9：DocVQA 训练期间的平均实现轨迹长度。Agentic GRPO 和 Agentic ESOpt 使用相同的展开解码器；它们的训练调度对齐到通用更新轴，显示到第 40 步。

两种方法开始时具有相似的轨迹长度，但它们的行为在训练后期分化：Agentic GRPO 轨迹快速增长，而 Agentic ESOpt 保持更短且更稳定。此诊断反映了在数独中观察到的过多轮次模式，并提供了在稀疏轨迹级反馈下优化的额外视图。

### D.4 WebArena-Lite

#### D.4.1 目标条件任务形式化

WebArena-Lite 在部分可观察的浏览器环境中评估目标条件控制（goal-conditioned control）。任务 $g$ 指定一个用户目标和一个初始隐藏网页状态。在时刻 $t$，智能体（agent）接收观察 $o_{t}$，采样动作

$$
a_{t}\sim\pi_{\theta}(\cdot\mid\bm{o}_{\leq t},\bm{a}_{<t},g,c),
$$

并转移到下一个浏览器状态。当轨迹（trajectory）$\bm{\tau}=(o_{0},a_{0},\ldots,o_{T})$ 在交互预算内满足目标谓词时，该轨迹成功。基准目标函数为

$$
J_{\mathrm{web}}(\theta;c)=\mathbb{E}_{g\sim\mathcal{G}}\mathbb{E}_{\bm{\tau}\sim\pi_{\theta}(\cdot\mid g,c)}\left[\mathbbm{1}_{\mathrm{succ}}(\bm{\tau},g)\right],
$$

通过保留任务集上的经验成功率来估计。奖励是稀疏的：中间动作不获得直接的任务信用（task credit），最终得分取决于完整轨迹是否到达目标网页状态。

外部上下文 $c$ 代表非参数化适应（non-parametric adaptation）。在 No Skill 条件下，它包含任务指令和交互历史；在 Trace2Skill 条件下，它额外包含从先前轨迹中蒸馏的程序性知识（procedural knowledge）。表4中的 No Skill 对照分离了参数更新，而技能条件对照比较了完整的适应流程。

对于网站类别 $\mathcal{G}_{k}$，报告的条件成功率为

$$
\widehat{J}_{\mathcal{G}_{k}}(\theta;c)=\frac{1}{|\mathcal{G}_{k}|}\sum_{g\in\mathcal{G}_{k}}\mathbbm{1}_{\mathrm{succ}}(\bm{\tau}_{g},g).
$$

数据集平均值将相同的估计器应用于完整集合 $\mathcal{G}$。

#### D.4.2 推出协议与数据划分

所有冻结模型和 Agentic ESOpt 行使用相同的浏览器协议。智能体观察 WebRL 风格的文本浏览器状态，在 WebRL 基于 id 的动作空间中行动，每个任务限制在30个浏览器动作。每个响应有2,048个token的预算，浏览器视口为 $1280\times 720$，评估器在终止后分配二元成功标记。采样使用温度0.7，top-$p=0.8$，top-$k=20$，min-$p=0$，存在惩罚（presence penalty）1.5，重复惩罚（repetition penalty）1。最终结果对所有165个保留任务进行三次完整评估的平均。

训练划分从原始的812个 WebArena 任务中构建，排除了基准提供的对应关系映射到 WebArena-Lite 的165个任务。对剩余647个任务进行站点分层划分，得到582个训练任务和65个验证任务。我们使用三个种子运行每个模型以蒸馏技能（基础模型、Agentic RL 模型、Agentic ESOpt 模型），并选择在验证数据集上表现最好的作为评估技能。WebArena-Lite 任务从不贡献参数更新奖励或轨迹到技能输入，定期评估是只读的。所有六个网站——Reddit、GitLab、Wikipedia、Map、Shopping 和 Shopping Admin——保持启用。

#### D.4.3 参数和技能适应阶段

表12：Qwen3.5-27B 上的 WebArena-Lite 适应阶段。两个 Agentic ESOpt 条件共享相同的 No Skill 参数更新；组合条件添加一个后处理技能蒸馏阶段。

| 条件 | 参数阶段 | 技能阶段 | 每阶段数据 | 额外ES |
| --- | --- | --- | --- | --- |
| No Skill 基线 | 无 | 无 | – | 无 |
| Agentic ESOpt + No Skill | 70个ES世代 | 无 | $G=8$ 方向 $\times$ 8任务/世代 | – |
| Trace2Skill 基线 | 无 | 70个技能迭代 | 8任务 $\times$ 8推出/迭代 | 无 |
| Agentic ESOpt + Trace2Skill | 相同的70世代ES运行 | 一次后处理蒸馏 | 所有完成的ES轨迹 | 无 |

Trace2Skill 基线从空技能开始，运行70个技能演化迭代。每个迭代使用8个训练任务，每个任务有8个采样轨迹，每个任务最多选择一个正轨迹和一个负轨迹。每10个迭代进行一次验证，表4报告最终评估的技能。

No Skill 参数阶段在8个训练任务上执行70次 $G=8$ 的全参数ES更新。因此每个世代包含 $8\times 8=64$ 个方向-任务评估，每个扰动（perturbation）在相同的任务批次上评分。奖励经过 z-score 归一化，$\sigma_{0}=1.5\times 10^{-3}$，$\sigma_{t}=1.5\times 10^{-3}$，$\alpha=2.5\times 10^{-4}$，每10个世代进行一次评估。

参数阶段后，所有完成的ES世代的轨迹被传递一次给 Trace2Skill。最终评估重建相同的 No Skill 更新序列，并将蒸馏的技能添加到系统提示中。这个实验实现了顺序共享轨迹组合；不执行额外的技能条件ES阶段。

#### D.4.4 完整评估曲线

图10：No Skill Qwen3.5-27B Agentic ESOpt 运行期间的完整 WebArena-Lite 评估成功率。每10次ES更新进行一次评估；曲线从29.50%的基础检查点开始，在70次更新后达到35.76%。

完整集合曲线在中间检查点之间是非单调的，但最终更新达到观察到的最佳成功率。表4报告用于主要比较的三次运行最终评估。

### D.5 自动启发式设计

#### D.5.1 实验配置

AHD（Automatic Heuristic Design，自动启发式设计）实验在8个3090 24GB GPU的集群上完成。我们在两个固定的外部搜索脚手架下评估 Agentic ESOpt：独立采样（Sample）和 EoH。在每个配对比较中，提议预算和外部搜索过程不变；Agentic ESOpt 仅添加参数空间更新。所有运行使用 LLaMA-3.1-8B-Instruct。

EoH 维持 $N=10$ 个启发式，进行25个外部世代。算子 e1 和 e2 各生成 $N$ 个候选，而 m1 和 m2 各生成 $kN$ 个候选，总预算为 $25N(2+2k)$。对于 $T=1{,}000$ 我们使用 $k=1$，对于 $T=2{,}000$ 使用 $k=3$。Sample 脚手架以20个批次评估独立的 i1 提议。

表13：AHD 提议预算和 Agentic ESOpt 更新单元。匹配的基线使用相同的提议数量。在 EoH 中，m1 和 m2 每个外部世代各形成一个ES更新批次。

| 外部脚手架 | $T$ | 外部世代 | 候选/世代 | ES更新批次 | 方向/批次 | $\sigma_{0}\!\rightarrow\!\sigma_{T}$ |
| --- | --- | --- | --- | --- | --- | --- |
| EoH ($k=1$) | 1,000 | 25 | $10(2+2k)=40$ | 50 | 10 | $10^{-3}\!\rightarrow 0$ |
| EoH ($k=3$) | 2,000 | 25 | $10(2+2k)=80$ | 50 | 30 | $10^{-3}\!\rightarrow 0$ |
| 独立采样 | 1,000 | 50 | 20 | 50 | 20 | $10^{-3}\!\rightarrow 0$ |
| 独立采样 | 2,000 | 100 | 20 | 100 | 20 | $10^{-3}\!\rightarrow 0$ |

Agentic ESOpt 使用全参数、单侧高斯扰动、余弦衰减（cosine decay）$\sigma:10^{-3}\!\rightarrow 0$、无预热、$\alpha=5\times 10^{-4}$ 和群体 z-score 归一化。在 EoH 中，只有突变算子 m1 和 m2 触发参数更新，每个外部世代产生两个更新批次；e1 和 e2 保持不变。在 Sample 中，每批20个提议形成一次更新。

所有目标在内部转换为最小化成本。EoH 使用选定父代的成本减去扰动子代的成本，而 Sample 使用负子代得分。无效或非有限程序在归一化前接收批次内有限惩罚，全部无效的批次不产生更新。候选生成使用温度1，top-$p=0.98$，无 top-$k$ 截断，最多768个新token。

#### D.5.2 额外的 ACO 风格结果和消融实验

##### ACO 风格结果

表14：总评估预算 $T\in\{1000,2000\}$ 下的 AHD ACO 风格结果。所有列越低越好。$\Delta$ 行报告比率 $\mathrm{EoH}/(\mathrm{Agentic\ ESOpt+EoH})-1$；正比率表示改进。改进用绿色阴影和绿色增量标记；退化保持无阴影和中性灰色增量。

<table><tr><td>方法</td><td>TSP <math><semantics><mrow><mi>N</mi> <mo>=</mo> <mn>50</mn></mrow> <annotation>N=50</annotation></semantics></math></td><td>TSP <math><semantics><mrow><mi>N</mi> <mo>=</mo> <mn>100</mn></mrow> <annotation>N=100</annotation></semantics></math></td><td>CVRP <math><semantics><mrow><mrow><mi>N</mi> <mo>=</mo> <mn>50</mn></mrow><mo>,</mo><mrow><mi>C</mi> <mo>=</mo> <mn>50</mn></mrow></mrow> <annotation>N=50,C=50</annotation></semantics></math></td><td>CVRP <math><semantics><mrow><mrow><mi>N</mi> <mo>=</mo> <mn>100</mn></mrow><mo>,</mo><mrow><mi>C</mi> <mo>=</mo> <mn>50</mn></mrow></mrow> <annotation>N=100,C=50</annotation></semantics></math></td><td>BPP <math><semantics><mrow><mrow><mi>N</mi> <mo>=</mo> <mn>500</mn></mrow><mo>,</mo><mrow><mi>C</mi> <mo>=</mo> <mn>150</mn></mrow></mrow> <annotation>N=500,C=150</annotation></semantics></math></td><td>BPP <math><semantics><mrow><mrow><mi>N</mi> <mo>=</mo> <mn>1,000</mn></mrow><mo>,</mo><mrow><mi>C</mi> <mo>=</mo> <mn>150</mn></mrow></mrow> <annotation>N=1{,}000,C=150</annotation></semantics></math></td></tr><tr><td>ACO</td><td>5.992</td><td>8.948</td><td>11.355</td><td>18.778</td><td>208.828</td><td>417.938</td></tr><tr><td colspan="7">总评估数：<math><semantics><mrow><mi>T</mi> <mo>=</mo> <mn>1000</mn></mrow> <annotation>T=1000</annotation></semantics></math></td></tr><tr><td>EoH</td><td>6.030</td><td>8.859</td><td>9.328</td><td>15.666</td><td>203.162</td><td>405.172</td></tr><tr><td>Agentic ESOpt + EoH</td><td>5.882</td><td>8.391</td><td>9.265</td><td>16.073</td><td>203.500</td><td>405.787</td></tr><tr><td><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> vs EoH</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 2.53%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 5.57%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 0.68%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 2.53%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 0.17%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 0.15%</td></tr><tr><td colspan="7">总评估数：<math><semantics><mrow><mi>T</mi> <mo>=</mo> <mn>2000</mn></mrow> <annotation>T=2000</annotation></semantics></math></td></tr><tr><td>EoH</td><td>5.886</td><td>8.368</td><td>9.379</td><td>15.989</td><td>203.307</td><td>405.688</td></tr><tr><td>Agentic ESOpt + EoH</td><td>5.890</td><td>8.398</td><td>9.179</td><td>15.420</td><td>203.052</td><td>405.276</td></tr><tr><td><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> vs EoH</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 0.08%</td><td><math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math> 0.36%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 2.18%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 3.69%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 0.13%</td><td><math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math> 0.10%</td></tr></table>

在 $T=1{,}000$ 时，Agentic ESOpt + EoH 在六个 ACO 风格测试集中改进了三个：两个 TSP 设置和 CVRP-50。在 $T=2{,}000$ 时，它改进了两个 CVRP 和两个 BPP 设置；两个 TSP 差异很小（$0.08\%$ 和 $0.36\%$）。结合构造性 Sample 和 EoH 比较，Agentic ESOpt 在12个测试集和6个场景的36个匹配方法-预算设置中改进了28个，1个持平，7个退化。

##### 组件和采样温度消融实验

表15：$T=1000$ 下构造性 AHD 的组件消融。TSP 最小化，KP 最大化。

| 方法 | TSP $N=50$ | KP $N=50,W=12.5$ |
| --- | --- | --- |
| EoH（基线） | 6.545 | 19.996 |
| Agentic ESOpt + EoH | 6.463 | 20.001 |
| 无ES（仅噪声） | 6.484 | 19.996 |
| 无余弦调度 | 6.480 | 19.997 |

表16：$T=1000$ 下构造性 TSP（$N=50$）的 EoH 采样温度消融，包含 Agentic ESOpt 以供比较。越低越好。

| 设置 | TSP |
| --- | --- |
| EoH ($0.6$) | 6.51782 |
| EoH ($1.0$) | 6.545 |
| EoH ($1.5$) | 6.95927 |
| Agentic ESOpt | 6.463 |

表16中的组件消融显示，奖励加权参数更新和余弦扰动调度都对最终结果有贡献：保留噪声但不进行ES更新，或固定扰动半径，都会削弱两个任务的性能。表16进一步显示，通过重新调整 EoH 采样温度无法再现这种增益。温度0.6是最强的 EoH 设置，但 Agentic ESOpt 在 TSP 上仍然更好。

##### 重复运行显著性

我们在 TSP（$N=50$）和 KP（$N=100,W=25$）上每种方法重复构造性 EoH 比较20次。表17报告跨运行的均值和样本标准差，以及单侧等方差 $t$ 检验。KP 值从内部取反的求解器成本转换回原生最大化目标。检验从表中总结的相同20个每次运行观察中计算。

表17：构造性 AHD 的重复运行分析。每种方法在20次独立运行中评估。TSP 越低越好，原生 KP 目标越高越好。

| 任务 | EoH | Agentic ESOpt + EoH | $p$ 值 |
| --- | --- | --- | --- |
| TSP（$N=50$，$\downarrow$） | $6.5517\pm 0.0729$ | $6.5007\pm 0.0868$ | $0.0258$ |
| KP（$N=100,W=25$，$\uparrow$） | $40.1562\pm 0.0024$ | $40.1578\pm 0.0017$ | $0.0100$ |

两个比较在0.05水平上都是显著的。因此，重复运行分析支持 AHD 增益在单个搜索种子之外的一致性。

##### 在线运行时开销

我们额外测量了 $T=1000$ 下构造性 Sample 脚手架的端到端墙钟时间。

表18：$T=1000$ 下构造性 Sample 的端到端运行时间。所有值以分钟为单位。

| 任务 | Sample | Agentic ESOpt + Sample |
| --- | --- | --- |
| 为 TSP 设计构造性启发式 | 54.8 | 60.6 |
| 为 KP 设计构造性启发式 | 41.2 | 48.6 |
| 为 ASP 设计构造性启发式 | 51.7 | 56.7 |

在三个任务中，附加 Agentic ESOpt 仅增加5.0-7.4分钟，对应于比原始 Sample 运行时间增加9.7%-18.0%。因此，参数更新几乎在现有候选生成和评估循环中在线完成，无需单独的搜索后训练阶段。

### D.6 跨设置超参数总结

表19汇总了所有报告设置中使用的 Agentic ESOpt 配置。上述特定任务部分提供了相应的交互限制、基线设置和评估协议。

表19：跨设置的 Agentic ESOpt 配置。"更新批次"计数参数更新，"奖励负载/方向"指定每个扰动的评分方式。所有设置使用全参数、单侧高斯扰动和群体 z-score 归一化。

| 设置 | 骨干网络 | 更新批次 | 方向/更新 | 奖励负载/方向 | $\sigma_{0}\!\rightarrow\!\sigma_{T}$ | $\alpha$ |
| --- | --- | --- | --- | --- | --- | --- |
| 数独，$H^{*}=5/10$ | Qwen3.5-4B | 100 | 32 | 32个谜题 | $10^{-3}\!\rightarrow 2.5{\times}10^{-4}$ | $5{\times}10^{-4}$ |
| 数独，$H^{*}=15$ | Qwen3.5-4B | 100 | 32 | 32个谜题 | $7{\times}10^{-4}\!\rightarrow 5{\times}10^{-4}$ | $5{\times}10^{-4}$ |
| 数学 | Qwen3.5-4B | 25 | 16 | 16个问题 | $10^{-3}\!\rightarrow 5{\times}10^{-4}$ | $5{\times}10^{-4}$ |
| DocVQA | Qwen3.5-4B | 40 | 16 | 16个样本 | $10^{-3}\!\rightarrow 5{\times}10^{-4}$ | $5{\times}10^{-4}$ |
| WebArena-Lite | Qwen3.5-27B | 70 | 8 | 8个网页任务 | $1.5{\times}10^{-3}\!\rightarrow 1.5{\times}10^{-3}$ 常量 | $2.5{\times}10^{-4}$ |
| AHD + EoH，$T=1{,}000$ | Llama-3.1-8B | 50 | 10 | 一个启发式目标 | $10^{-3}\!\rightarrow 0$ | $5{\times}10^{-4}$ |
| AHD + EoH，$T=2{,}000$ | Llama-3.1-8B | 50 | 30 | 一个启发式目标 | $10^{-3}\!\rightarrow 0$ | $5{\times}10^{-4}$ |
| AHD + Sample，$T=1{,}000/2{,}000$ | Llama-3.1-8B | 50 / 100 | 20 | 一个启发式目标 | $10^{-3}\!\rightarrow 0$ | $5{\times}10^{-4}$ |

对于候选 $i$，实现应用 $\theta+\sigma_{t}\bm{\epsilon}_{i}$，评估完整轨迹或生成的启发式，恢复扰动，并在更新期间从其整数种子重建相同方向。实现使用单侧扰动，没有显式的 $1/\sigma_{t}$ 乘数，因此 $\alpha$ 是有效的更新尺度。特定任务的基线配置总结在表7、8、9、10、11、12和13中。

扰动调度反映每个设置的优化目标。训练时实验保留非零终端半径以保持邻域平滑：数独使用视距特定的终端半径，数学和 DocVQA 衰减到 $5\times 10^{-4}$，WebArena-Lite 保持 $\sigma=1.5\times 10^{-3}$。AHD 将 $\sigma$ 衰减到零，因为最终目标是测试时搜索期间找到的最佳任务特定启发式。

群体大小 $G$ 和用于对每个方向评分的任务数量是联合选择的。增加任一数量都可以提高更新稳定性或任务覆盖率，同时也会增加推出成本。跨设置，二元奖励任务每个方向使用多个任务，而 AHD 为每个方向分配一个可执行启发式目标。

群体 z-score 归一化在异构奖励之间提供了通用更新接口。它将二元成功、连续 ANLS 和问题相关的启发式目标转换为群体内相对得分。对于 AHD，无效或非有限程序在归一化前映射到批次内有限惩罚，全部无效的批次不产生参数更新。

附录A讨论了超参数敏感性以及 $G$ 和 $\sigma$ 的自适应调度。

## 附录E 提示和技能

### E.1 数独

#### E.1.1 环境提示

数独被实现为多轮动作环境。智能体不在一次响应中提交完整棋盘。相反，在每一轮中，它观察原始谜题、当前棋盘、当前空单元格列表以及来自前一个动作的可选环境反馈，然后以格式 set <row\> <col\> <value\> 发出恰好一个动作。行和列从1开始索引。当所有被掩码的单元格都被填充或动作预算耗尽时，回合终止，最终奖励是二元的。

以下是来自评估划分的包含15个被掩码单元格的第0轮提示。空单元格渲染为"."。

# 附录

## A.5 数独任务详情

### A.5.1 任务设置

数独是一个 $9 \times 9$ 的数字放置谜题，目标是用数字 1-9 填充网格，使得每行、每列和每个 $3 \times 3$ 子网格都包含每个数字恰好一次。智能体一次填充一个单元格，直到网格完全填满。每个谜题都有一个唯一解。

**状态空间：** 在回合 $t$，状态 $s_t$ 由当前的部分填充网格和剩余空单元格列表组成。

**动作空间：** 每个动作指定一个空单元格 $(r,c)$ 和一个数字 $v \in \{1,\ldots,9\}$，表示为 `set r c v`。

**奖励：** 智能体在终止时收到二元奖励：如果最终网格满足所有数独约束则为 +1，否则为 0。中间步骤不提供奖励。

**环境反馈：** 在每次填充后，环境返回更新后的网格和当前空单元格列表。

**终止条件：** 当所有单元格都被填充或达到最大步数限制时，回合终止。

**提示：** 我们使用如图所示的最小提示。智能体被指示一次填充一个单元格，遵守数独规则，并以格式 `set <row> <col> <value>` 输出其动作。不允许推理或解释——响应必须恰好是一个动作。

下图展示了数独任务的提示格式示例：

<svg id="A5.SS1.SSS1.p3.pic1" height="52.6" overflow="visible" version="1.1" viewBox="0 0 650 52.6" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,52.6) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#276A4A;" fill="#276A4A" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.84 L 0 30.17 C 0 32.84 2.17 35.01 4.84 35.01 L 645.16 35.01 C 647.83 35.01 650 32.84 650 30.17 L 650 4.84 C 650 2.17 647.83 0 645.16 0 L 4.84 0 C 2.17 0 0 2.17 0 4.84 Z" /></g><g style="--ltx-fill-color:#F1F8F4;" fill="#F1F8F4" fill-opacity="1.0"><path style="stroke:none" d="M 0.9 4.84 L 0.9 30.17 C 0.9 32.35 2.66 34.11 4.84 34.11 L 645.16 34.11 C 647.34 34.11 649.1 32.35 649.1 30.17 L 649.1 4.84 C 649.1 2.66 647.34 0.9 645.16 0.9 L 4.84 0.9 C 2.66 0.9 0.9 2.66 0.9 4.84 Z" /></g><g transform="matrix(1.0 0.0 0.0 1.0 6.92 32.24)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="matrix(1 0 0 1 0 0)"><g style="--ltx-fill-color:#276A4A;" fill="#276A4A" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.7 L 0 16.65 C 0 18.7 1.66 20.36 3.7 20.36 L 120.21 20.36 C 122.25 20.36 123.91 18.7 123.91 16.65 L 123.91 3.7 C 123.91 1.66 122.25 0 120.21 0 L 3.7 0 C 1.66 0 0 1.66 0 3.7 Z" /></g><g style="--ltx-fill-color:#276A4A;" fill="#276A4A" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 3.7 L 0.55 16.65 C 0.55 18.39 1.96 19.8 3.7 19.8 L 120.21 19.8 C 121.95 19.8 123.36 18.39 123.36 16.65 L 123.36 3.7 C 123.36 1.96 121.95 0.55 120.21 0.55 L 3.7 0.55 C 1.96 0.55 0.55 1.96 0.55 3.7 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.64 7.76)"><foreignobject style="--ltx-fo-width:8.5em;--ltx-fo-height:0.54em;--ltx-fo-depth:0.15em;font-size:9.06pt;" height="8.61" overflow="visible" transform="matrix(1 0 0 -1 0 6.73)" width="106.63"><span id="A5.SS1.SSS1.p3.pic1.1" style="font-size:70%;--ltx-fg-color:#FFFFFF;">数独提示</span></foreignobject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 9.2 11.28)"><foreignobject style="--ltx-fo-width:45.65em;--ltx-fo-height:0.75em;--ltx-fo-depth:0.25em;font-size:10pt;" height="13.84" overflow="visible" transform="matrix(1 0 0 -1 0 10.38)" width="631.67"><span id="A5.SS1.SSS1.p3.pic1.2" style="width:45.65em;"><span id="A5.SS1.SSS1.p3.pic1.2.1"><a download="">⬇</a> <span id="lstnumberx1"><span id="lstnumberx1.1" style="--ltx-fg-color:#000000;">你是一个解决数独的智能体，每次执行一个动作。每回合，恰好填充一个空单元格。行和列从1开始索引。棋盘通过 | 和水平线分割为九个 3x3 方格。每一行、每一列和每个 3x3 方格都必须恰好包含数字 1 到 9 各一次。仅从当前空单元格列表中选择单元格。保持所有原始给定数字不变。你的整个响应必须恰好是包含一个动作的一行，不要有其他内容。不要解释、大声推理、用代码围栏包装答案或输出任何额外的词语。使用恰好以下格式：</span> <span id="lstnumberx2"><span id="lstnumberx2.1" style="--ltx-fg-color:#000000;">set &lt;行&gt; &lt;列&gt; &lt;值&gt;</span> </span><span id="lstnumberx4"><span id="lstnumberx4.1" style="--ltx-fg-color:#000000;">原始谜题，掩码数量=15：</span></span> <span id="lstnumberx5"><span id="lstnumberx5.2" style="--ltx-fg-color:#000000;">c1 c2 c3 | c4 c5 c6 | c7 c8 c9</span> </span><span id="lstnumberx6"><span id="lstnumberx6.1" style="--ltx-fg-color:#000000;">r1: 3 5 9 | 4 . 7 | . . 8</span> </span><span id="lstnumberx7"><span id="lstnumberx7.1" style="--ltx-fg-color:#000000;">r2: 2 1 8 | . 9 5 | 4 . 6</span> </span><span id="lstnumberx8"><span id="lstnumberx8.1" style="--ltx-fg-color:#000000;">r3: 4 7 6 | 2 8 1 | 3 5 9</span> </span><span id="lstnumberx9"><span id="lstnumberx9.2" style="--ltx-fg-color:#000000;">--------+----------+--------</span> </span><span id="lstnumberx10"><span id="lstnumberx10.1" style="--ltx-fg-color:#000000;">r4: 6 2 1 | 8 5 3 | 9 4 7</span> </span><span id="lstnumberx11"><span id="lstnumberx11.1" style="--ltx-fg-color:#000000;">r5: . 3 5 | 9 . 4 | 6 2 1</span> </span><span id="lstnumberx12"><span id="lstnumberx12.1" style="--ltx-fg-color:#000000;">r6: . 4 7 | 6 1 2 | 8 3 5</span> </span><span id="lstnumberx13"><span id="lstnumberx13.2" style="--ltx-fg-color:#000000;">--------+----------+--------</span> </span><span id="lstnumberx14"><span id="lstnumberx14.1" style="--ltx-fg-color:#000000;">r7: 7 6 . | 1 3 8 | 5 9 .</span></span> <span id="lstnumberx15"><span id="lstnumberx15.1" style="--ltx-fg-color:#000000;">r8: 5 9 . | . 2 6 | 1 . 3</span> </span><span id="lstnumberx16"><span id="lstnumberx16.1" style="--ltx-fg-color:#000000;">r9: . 8 3 | 5 4 . | 7 6 2</span> </span><span id="lstnumberx18"><span id="lstnumberx18.1" style="--ltx-fg-color:#000000;">当前棋盘，回合=0，剩余空格=15：</span></span> <span id="lstnumberx19"><span id="lstnumberx19.2" style="--ltx-fg-color:#000000;">c1 c2 c3 | c4 c5 c6 | c7 c8 c9</span> </span><span id="lstnumberx20"><span id="lstnumberx20.1" style="--ltx-fg-color:#000000;">r1: 3 5 9 | 4 . 7 | . . 8</span> </span><span id="lstnumberx21"><span id="lstnumberx21.1" style="--ltx-fg-color:#000000;">r2: 2 1 8 | . 9 5 | 4 . 6</span> </span><span id="lstnumberx22"><span id="lstnumberx22.1" style="--ltx-fg-color:#000000;">r3: 4 7 6 | 2 8 1 | 3 5 9</span> </span><span id="lstnumberx23"><span id="lstnumberx23.2" style="--ltx-fg-color:#000000;">--------+----------+--------</span> </span><span id="lstnumberx24"><span id="lstnumberx24.1" style="--ltx-fg-color:#000000;">r4: 6 2 1 | 8 5 3 | 9 4 7</span> </span><span id="lstnumberx25"><span id="lstnumberx25.1" style="--ltx-fg-color:#000000;">r5: . 3 5 | 9 . 4 | 6 2 1</span> </span><span id="lstnumberx26"><span id="lstnumberx26.1" style="--ltx-fg-color:#000000;">r6: . 4 7 | 6 1 2 | 8 3 5</span> </span><span id="lstnumberx27"><span id="lstnumberx27.2" style="--ltx-fg-color:#000000;">--------+----------+--------</span> </span><span id="lstnumberx28"><span id="lstnumberx28.1" style="--ltx-fg-color:#000000;">r7: 7 6 . | 1 3 8 | 5 9 .</span></span> <span id="lstnumberx29"><span id="lstnumberx29.1" style="--ltx-fg-color:#000000;">r8: 5 9 . | . 2 6 | 1 . 3</span> </span><span id="lstnumberx30"><span id="lstnumberx30.1" style="--ltx-fg-color:#000000;">r9: . 8 3 | 5 4 . | 7 6 2</span> </span><span id="lstnumberx32"><span id="lstnumberx32.1" style="--ltx-fg-color:#000000;">当前空单元格：r1c5, r1c7, r1c8, r2c4, r2c8, r5c1, r5c5, r6c1, r7c3, r7c9, r8c3, r8c4, r8c8, r9c1, r9c6</span></span></span></span></foreignobject></g></g></svg>

例如，对这个提示的一个有效响应是单行命令，如 set 1 5 6。然后环境应用该动作，并通过在原始谜题块之前插入 Last environment feedback: Filled r1c5 with 6. 来构造下一个提示，更新当前棋盘，设置 turn=1，将 remaining\_empty 减少到 14，并从当前空单元格列表中移除 r1c5。

### E.2 数学推理

#### E.2.1 ReAct 提示

AIME 设置不是单次直接答案调用，而是被视为一个智能体化的（Agentic）ReAct 环境。模型可以多次调用命令行 Python 工具，接收每个工具结果作为观察（observation），并在输出最终框定答案时停止。进化策略（ES）奖励是整个轨迹（trajectory）之后最终答案的二元精确匹配分数。

<svg id="A5.SS2.SSS1.p2.pic1" height="50.1" overflow="visible" version="1.1" viewBox="0 0 650 50.1" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,50.1) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#22615E;" fill="#22615E" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.84 L 0 27.67 C 0 30.34 2.17 32.51 4.84 32.51 L 645.16 32.51 C 647.83 32.51 650 30.34 650 27.67 L 650 4.84 C 650 2.17 647.83 0 645.16 0 L 4.84 0 C 2.17 0 0 2.17 0 4.84 Z" /></g><g style="--ltx-fill-color:#EFF8F7;" fill="#EFF8F7" fill-opacity="1.0"><path style="stroke:none" d="M 0.9 4.84 L 0.9 27.67 C 0.9 29.85 2.66 31.61 4.84 31.61 L 645.16 31.61 C 647.34 31.61 649.1 29.85 649.1 27.67 L 649.1 4.84 C 649.1 2.66 647.34 0.9 645.16 0.9 L 4.84 0.9 C 2.66 0.9 0.9 2.66 0.9 4.84 Z" /></g><g transform="matrix(1.0 0.0 0.0 1.0 6.92 29.74)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="matrix(1 0 0 1 0 0)"><g style="--ltx-fill-color:#22615E;" fill="#22615E" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.7 L 0 16.65 C 0 18.7 1.66 20.36 3.7 20.36 L 152.55 20.36 C 154.6 20.36 156.26 18.7 156.26 16.65 L 156.26 3.7 C 156.26 1.66 154.6 0 152.55 0 L 3.7 0 C 1.66 0 0 1.66 0 3.7 Z" /></g><g style="--ltx-fill-color:#22615E;" fill="#22615E" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 3.7 L 0.55 16.65 C 0.55 18.39 1.96 19.8 3.7 19.8 L 152.55 19.8 C 154.29 19.8 155.7 18.39 155.7 16.65 L 155.7 3.7 C 155.7 1.96 154.29 0.55 152.55 0.55 L 3.7 0.55 C 1.96 0.55 0.55 1.96 0.55 3.7 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.64 7.76)"><foreignobject style="--ltx-fo-width:11.08em;--ltx-fo-height:0.54em;--ltx-fo-depth:0.15em;font-size:9.06pt;" height="8.61" overflow="visible" transform="matrix(1 0 0 -1 0 6.73)" width="138.98"><span id="A5.SS2.SSS1.p2.pic1.1" style="font-size:70%;--ltx-fg-color:#FFFFFF;">AIME ReAct 提示词</span></foreignobject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 9.2 10.24)"><foreignobject style="--ltx-fo-width:45.65em;--ltx-fo-height:0.64em;--ltx-fo-depth:0.18em;font-size:10pt;" height="11.34" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="631.67"><span id="A5.SS2.SSS1.p2.pic1.2" style="width:45.65em;"><span id="A5.SS2.SSS1.p2.pic1.2.1"><a download="">⬇</a> <span id="lstnumberx33"><span id="lstnumberx33.1" style="font-size:70%;--ltx-fg-color:#000000;">你正在解决一个数学推理问题</span><span id="lstnumberx33.14" style="font-size:70%;--ltx-fg-color:#000000;">。</span><span id="lstnumberx33.16" style="font-size:70%;--ltx-fg-color:#000000;">你可以逐步思考并在有用时使用命令行</span> <span id="lstnumberx33.36" style="font-size:70%;--ltx-fg-color:#000000;">Python</span><span id="lstnumberx33.41" style="font-size:70%;--ltx-fg-color:#000000;">。</span></span> <span id="lstnumberx35"><span id="lstnumberx35.1" style="font-size:70%;--ltx-fg-color:#000000;">动作格式</span><span id="lstnumberx35.4" style="font-size:70%;--ltx-fg-color:#000000;">:</span></span> <span id="lstnumberx36"><span id="lstnumberx36.1" style="font-size:70%;--ltx-fg-color:#000000;">Action</span><span id="lstnumberx36.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span></span> <span id="lstnumberx37"><span id="lstnumberx37.1" style="font-size:70%;--ltx-fg-color:#000000;">{</span> </span><span id="lstnumberx38"><span id="lstnumberx38.2" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx38.3" style="font-size:70%;--ltx-fg-color:#000000;">name</span> <span id="lstnumberx38.4" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx38.6" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx38.7" style="font-size:70%;--ltx-fg-color:#000000;">bash</span> <span id="lstnumberx38.8" style="font-size:70%;--ltx-fg-color:#000000;">",</span></span> <span id="lstnumberx39"><span id="lstnumberx39.2" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx39.3" style="font-size:70%;--ltx-fg-color:#000000;">arguments</span> <span id="lstnumberx39.4" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx39.6" style="font-size:70%;--ltx-fg-color:#000000;">{"</span> <span id="lstnumberx39.7" style="font-size:70%;--ltx-fg-color:#000000;">command</span> <span id="lstnumberx39.8" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx39.10" style="font-size:70%;--ltx-fg-color:#000000;">"&lt;</span> <span id="lstnumberx39.11" style="font-size:70%;--ltx-fg-color:#000000;">shell</span> <span id="lstnumberx39.13" style="font-size:70%;--ltx-fg-color:#000000;">command</span> <span id="lstnumberx39.14" style="font-size:70%;--ltx-fg-color:#000000;">&gt;"}</span> </span><span id="lstnumberx40"><span id="lstnumberx40.1" style="font-size:70%;--ltx-fg-color:#000000;">}</span> </span><span id="lstnumberx41"><span id="lstnumberx41.1" style="font-size:70%;--ltx-fg-color:#000000;">每次动作之后</span><span id="lstnumberx41.6" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx41.8" style="font-size:70%;--ltx-fg-color:#000000;">环境执行命令并返回一个观察结果</span><span id="lstnumberx41.25" style="font-size:70%;--ltx-fg-color:#000000;">。</span><span id="lstnumberx41.27" style="font-size:70%;--ltx-fg-color:#000000;">根据需要重复思考</span> <span id="lstnumberx41.30" style="font-size:70%;--ltx-fg-color:#000000;">/</span> <span id="lstnumberx41.31" style="font-size:70%;--ltx-fg-color:#000000;">动作</span> <span id="lstnumberx41.32" style="font-size:70%;--ltx-fg-color:#000000;">/</span> <span id="lstnumberx41.33" style="font-size:70%;--ltx-fg-color:#000000;">观察结果的循环</span><span id="lstnumberx41.38" style="font-size:70%;--ltx-fg-color:#000000;">。</span><span id="lstnumberx41.40" style="font-size:70%;--ltx-fg-color:#000000;">完成后</span><span id="lstnumberx41.43" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx41.45" style="font-size:70%;--ltx-fg-color:#000000;">以</span> <span id="lstnumberx41.55" style="font-size:70%;--ltx-fg-color:#000000;">\</span> <span id="lstnumberx41.56" style="font-size:70%;--ltx-fg-color:#000000;">boxed</span> <span id="lstnumberx41.57" style="font-size:70%;--ltx-fg-color:#000000;">{</span> <span id="lstnumberx41.58" style="font-size:70%;--ltx-fg-color:#000000;">answer</span> <span id="lstnumberx41.59" style="font-size:70%;--ltx-fg-color:#000000;">}</span> <span id="lstnumberx41.59" style="font-size:70%;--ltx-fg-color:#000000;">的形式输出最终结果</span><span id="lstnumberx41.59" style="font-size:70%;--ltx-fg-color:#000000;">。</span></span> <span id="lstnumberx43"><span id="lstnumberx43.1" style="font-size:70%;--ltx-fg-color:#000000;">问题</span><span id="lstnumberx43.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span></span> <span id="lstnumberx44"><span id="lstnumberx44.1" style="font-size:70%;--ltx-fg-color:#000000;">Patrick</span> <span id="lstnumberx44.3" style="font-size:70%;--ltx-fg-color:#000000;">以恒定速率沿着一条从学校到公园的直路开始行走</span><span id="lstnumberx44.32" style="font-size:70%;--ltx-fg-color:#000000;">。</span><span id="lstnumberx44.34" style="font-size:70%;--ltx-fg-color:#000000;">在</span> <span id="lstnumberx44.36" style="font-size:70%;--ltx-fg-color:#000000;">Patrick</span> <span id="lstnumberx44.38" style="font-size:70%;--ltx-fg-color:#000000;">出发一小时后</span><span id="lstnumberx44.43" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx44.45" style="font-size:70%;--ltx-fg-color:#000000;">Tanya</span> <span id="lstnumberx44.47" style="font-size:70%;--ltx-fg-color:#000000;">沿着同一条路从学校到公园开始跑步</span><span id="lstnumberx44.68" style="font-size:70%;--ltx-fg-color:#000000;">。</span><span id="lstnumberx44.70" style="font-size:70%;--ltx-fg-color:#000000;">在</span> <span id="lstnumberx44.72" style="font-size:70%;--ltx-fg-color:#000000;">Tanya</span> <span id="lstnumberx44.74" style="font-size:70%;--ltx-fg-color:#000000;">出发一小时后</span><span id="lstnumberx44.79" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx44.81" style="font-size:70%;--ltx-fg-color:#000000;">Jose</span> <span id="lstnumberx44.83" style="font-size:70%;--ltx-fg-color:#000000;">沿着同一条路从学校到公园开始骑自行车</span><span id="lstnumberx44.104" style="font-size:70%;--ltx-fg-color:#000000;">。</span><span id="lstnumberx44.106" style="font-size:70%;--ltx-fg-color:#000000;">Tanya</span> <span id="lstnumberx44.108" style="font-size:70%;--ltx-fg-color:#000000;">以恒定速率跑步</span><span id="lstnumberx44.110" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx44.112" style="font-size:70%;--ltx-fg-color:#000000;">比</span> <span id="lstnumberx44.114" style="font-size:70%;--ltx-fg-color:#000000;">Patrick</span> <span id="lstnumberx44.116" style="font-size:70%;--ltx-fg-color:#000000;">行走的速度快每小时</span> <span id="lstnumberx44.120" style="font-size:70%;--ltx-fg-color:#000000;">2</span> <span id="lstnumberx44.122" style="font-size:70%;--ltx-fg-color:#000000;">英里</span><span id="lstnumberx44.135" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx44.137" style="font-size:70%;--ltx-fg-color:#000000;">Jose</span> <span id="lstnumberx44.139" style="font-size:70%;--ltx-fg-color:#000000;">以恒定速率骑车</span><span id="lstnumberx44.141" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx44.143" style="font-size:70%;--ltx-fg-color:#000000;">比</span> <span id="lstnumberx44.145" style="font-size:70%;--ltx-fg-color:#000000;">Tanya</span> <span id="lstnumberx44.147" style="font-size:70%;--ltx-fg-color:#000000;">跑步的速度快每小时</span> <span id="lstnumberx44.151" style="font-size:70%;--ltx-fg-color:#000000;">7</span> <span id="lstnumberx44.153" style="font-size:70%;--ltx-fg-color:#000000;">英里</span><span id="lstnumberx44.166" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx44.168" style="font-size:70%;--ltx-fg-color:#000000;">三人同时到达公园</span><span id="lstnumberx44.189" style="font-size:70%;--ltx-fg-color:#000000;">。</span><span id="lstnumberx44.191" style="font-size:70%;--ltx-fg-color:#000000;">从学校到公园的距离是</span> <span id="lstnumberx44.209" style="font-size:70%;--ltx-fg-color:#000000;">m</span> <span id="lstnumberx44.210" style="font-size:70%;--ltx-fg-color:#000000;">/</span> <span id="lstnumberx44.211" style="font-size:70%;--ltx-fg-color:#000000;">n</span> <span id="lstnumberx44.213" style="font-size:70%;--ltx-fg-color:#000000;">英里</span><span id="lstnumberx44.214" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx44.216" style="font-size:70%;--ltx-fg-color:#000000;">其中</span> <span id="lstnumberx44.218" style="font-size:70%;--ltx-fg-color:#000000;">m</span> <span id="lstnumberx44.220" style="font-size:70%;--ltx-fg-color:#000000;">和</span> <span id="lstnumberx44.222" style="font-size:70%;--ltx-fg-color:#000000;">n</span> <span id="lstnumberx44.224" style="font-size:70%;--ltx-fg-color:#000000;">是互质的正整数</span><span id="lstnumberx44.233" style="font-size:70%;--ltx-fg-color:#000000;">。</span><span id="lstnumberx44.235" style="font-size:70%;--ltx-fg-color:#000000;">求</span> <span id="lstnumberx44.237" style="font-size:70%;--ltx-fg-color:#000000;">m</span> <span id="lstnumberx44.238" style="font-size:70%;--ltx-fg-color:#000000;">+</span> <span id="lstnumberx44.239" style="font-size:70%;--ltx-fg-color:#000000;">n</span><span id="lstnumberx44.240" style="font-size:70%;--ltx-fg-color:#000000;">。</span></span> <span id="lstnumberx46"><span id="lstnumberx46.1" style="font-size:70%;--ltx-fg-color:#000000;">Thought</span><span id="lstnumberx46.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx46.4" style="font-size:70%;--ltx-fg-color:#000000;">设</span> <span id="lstnumberx46.6" style="font-size:70%;--ltx-fg-color:#000000;">Patrick</span> <span id="lstnumberx46.7" style="font-size:70%;--ltx-fg-color:#000000;">的</span> <span id="lstnumberx46.8" style="font-size:70%;--ltx-fg-color:#000000;">旅行时间为</span> <span id="lstnumberx46.16" style="font-size:70%;--ltx-fg-color:#000000;">T</span><span id="lstnumberx46.18" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx46.20" style="font-size:70%;--ltx-fg-color:#000000;">速度为</span> <span id="lstnumberx46.24" style="font-size:70%;--ltx-fg-color:#000000;">p</span><span id="lstnumberx46.25" style="font-size:70%;--ltx-fg-color:#000000;">。</span><span id="lstnumberx46.27" style="font-size:70%;--ltx-fg-color:#000000;">我可以用</span> <span id="lstnumberx46.45" style="font-size:70%;--ltx-fg-color:#000000;">Python</span> <span id="lstnumberx46.45" style="font-size:70%;--ltx-fg-color:#000000;">求解两个等距离方程</span><span id="lstnumberx46.46" style="font-size:70%;--ltx-fg-color:#000000;">。</span></span> <span id="lstnumberx47"><span id="lstnumberx47.1" style="font-size:70%;--ltx-fg-color:#000000;">Action</span><span id="lstnumberx47.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span></span> <span id="lstnumberx48"><span id="lstnumberx48.1" style="font-size:70%;--ltx-fg-color:#000000;">{</span> </span><span id="lstnumberx49"><span id="lstnumberx49.2" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx49.3" style="font-size:70%;--ltx-fg-color:#000000;">name</span> <span id="lstnumberx49.4" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx49.6" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx49.7" style="font-size:70%;--ltx-fg-color:#000000;">bash</span> <span id="lstnumberx49.8" style="font-size:70%;--ltx-fg-color:#000000;">",</span></span> <span id="lstnumberx50"><span id="lstnumberx50.2" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx50.3" style="font-size:70%;--ltx-fg-color:#000000;">arguments</span> <span id="lstnumberx50.4" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx50.6" style="font-size:70%;--ltx-fg-color:#000000;">{"</span> <span id="lstnumberx50.7" style="font-size:70%;--ltx-fg-color:#000000;">command</span> <span id="lstnumberx50.8" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx50.10" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx50.11" style="font-size:70%;--ltx-fg-color:#000000;">python</span> <span id="lstnumberx50.13" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx50.14" style="font-size:70%;--ltx-fg-color:#000000;">c</span> <span id="lstnumberx50.16" style="font-size:70%;--ltx-fg-color:#000000;">\"</span> <span id="lstnumberx50.17" style="font-size:70%;--ltx-fg-color:#000000;">import</span> <span id="lstnumberx50.19" style="font-size:70%;--ltx-fg-color:#000000;">sympy</span> <span id="lstnumberx50.21" style="font-size:70%;--ltx-fg-color:#000000;">as</span> <span id="lstnumberx50.23" style="font-size:70%;--ltx-fg-color:#000000;">sp</span><span id="lstnumberx50.24" style="font-size:70%;--ltx-fg-color:#000000;">;</span><span id="lstnumberx50.26" style="font-size:70%;--ltx-fg-color:#000000;">T</span><span id="lstnumberx50.27" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx50.28" style="font-size:70%;--ltx-fg-color:#000000;">p</span> <span id="lstnumberx50.29" style="font-size:70%;--ltx-fg-color:#000000;">=</span> <span id="lstnumberx50.30" style="font-size:70%;--ltx-fg-color:#000000;">sp</span><span id="lstnumberx50.31" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx50.32" style="font-size:70%;--ltx-fg-color:#000000;">symbols</span> <span id="lstnumberx50.33" style="font-size:70%;--ltx-fg-color:#000000;">('</span> <span id="lstnumberx50.34" style="font-size:70%;--ltx-fg-color:#000000;">T</span> <span id="lstnumberx50.36" style="font-size:70%;--ltx-fg-color:#000000;">p</span> <span id="lstnumberx50.37" style="font-size:70%;--ltx-fg-color:#000000;">');</span><span id="lstnumberx50.39" style="font-size:70%;--ltx-fg-color:#000000;">sol</span> <span id="lstnumberx50.40" style="font-size:70%;--ltx-fg-color:#000000;">=</span> <span id="lstnumberx50.41" style="font-size:70%;--ltx-fg-color:#000000;">sp</span><span id="lstnumberx50.42" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx50.43" style="font-size:70%;--ltx-fg-color:#000000;">solve</span> <span id="lstnumberx50.44" style="font-size:70%;--ltx-fg-color:#000000;">([</span><span id="lstnumberx50.45" style="font-size:70%;--ltx-fg-color:#000000;">sp</span><span id="lstnumberx50.46" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx50.47" style="font-size:70%;--ltx-fg-color:#000000;">Eq</span> <span id="lstnumberx50.48" style="font-size:70%;--ltx-fg-color:#000000;">(</span><span id="lstnumberx50.49" style="font-size:70%;--ltx-fg-color:#000000;">p</span><span id="lstnumberx50.50" style="font-size:70%;--ltx-fg-color:#000000;">,2*</span> <span id="lstnumberx50.51" style="font-size:70%;--ltx-fg-color:#000000;">T</span> <span id="lstnumberx50.52" style="font-size:70%;--ltx-fg-color:#000000;">-2),</span><span id="lstnumberx50.54" style="font-size:70%;--ltx-fg-color:#000000;">sp</span><span id="lstnumberx50.55" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx50.56" style="font-size:70%;--ltx-fg-color:#000000;">Eq</span> <span id="lstnumberx50.57" style="font-size:70%;--ltx-fg-color:#000000;">(</span><span id="lstnumberx50.58" style="font-size:70%;--ltx-fg-color:#000000;">p</span><span id="lstnumberx50.59" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx50.60" style="font-size:70%;--ltx-fg-color:#000000;">sp</span><span id="lstnumberx50.61" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx50.62" style="font-size:70%;--ltx-fg-color:#000000;">Rational</span> <span id="lstnumberx50.63" style="font-size:70%;--ltx-fg-color:#000000;">(9,2)*</span> <span id="lstnumberx50.64" style="font-size:70%;--ltx-fg-color:#000000;">T</span> <span id="lstnumberx50.65" style="font-size:70%;--ltx-fg-color:#000000;">-9)],[</span><span id="lstnumberx50.66" style="font-size:70%;--ltx-fg-color:#000000;">T</span><span id="lstnumberx50.67" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx50.68" style="font-size:70%;--ltx-fg-color:#000000;">p</span><span id="lstnumberx50.69" style="font-size:70%;--ltx-fg-color:#000000;">]);</span><span id="lstnumberx50.71" style="font-size:70%;--ltx-fg-color:#000000;">T0</span><span id="lstnumberx50.72" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx50.73" style="font-size:70%;--ltx-fg-color:#000000;">p0</span> <span id="lstnumberx50.74" style="font-size:70%;--ltx-fg-color:#000000;">=</span> <span id="lstnumberx50.75" style="font-size:70%;--ltx-fg-color:#000000;">sol</span> <span id="lstnumberx50.76" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx50.77" style="font-size:70%;--ltx-fg-color:#000000;">T</span><span id="lstnumberx50.78" style="font-size:70%;--ltx-fg-color:#000000;">],</span><span id="lstnumberx50.79" style="font-size:70%;--ltx-fg-color:#000000;">sol</span> <span id="lstnumberx50.80" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx50.81" style="font-size:70%;--ltx-fg-color:#000000;">p</span><span id="lstnumberx50.82" style="font-size:70%;--ltx-fg-color:#000000;">];</span><span id="lstnumberx50.84" style="font-size:70%;--ltx-fg-color:#000000;">d</span> <span id="lstnumberx50.85" style="font-size:70%;--ltx-fg-color:#000000;">=</span> <span id="lstnumberx50.86" style="font-size:70%;--ltx-fg-color:#000000;">sp</span><span id="lstnumberx50.87" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx50.88" style="font-size:70%;--ltx-fg-color:#000000;">factor</span> <span id="lstnumberx50.89" style="font-size:70%;--ltx-fg-color:#000000;">(</span><span id="lstnumberx50.90" style="font-size:70%;--ltx-fg-color:#000000;">T0</span> <span id="lstnumberx50.91" style="font-size:70%;--ltx-fg-color:#000000;">*</span> <span id="lstnumberx50.92" style="font-size:70%;--ltx-fg-color:#000000;">p0</span><span id="lstnumberx50.93" style="font-size:70%;--ltx-fg-color:#000000;">);</span><span id="lstnumberx50.95" style="font-size:70%;--ltx-fg-color:#000000;">print</span> <span id="lstnumberx50.96" style="font-size:70%;--ltx-fg-color:#000000;">(</span><span id="lstnumberx50.97" style="font-size:70%;--ltx-fg-color:#000000;">d</span><span id="lstnumberx50.98" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx50.100" style="font-size:70%;--ltx-fg-color:#000000;">sp</span><span id="lstnumberx50.101" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx50.102" style="font-size:70%;--ltx-fg-color:#000000;">numer</span> <span id="lstnumberx50.103" style="font-size:70%;--ltx-fg-color:#000000;">(</span><span id="lstnumberx50.104" style="font-size:70%;--ltx-fg-color:#000000;">d</span><span id="lstnumberx50.105" style="font-size:70%;--ltx-fg-color:#000000;">)+</span> <span id="lstnumberx50.106" style="font-size:70%;--ltx-fg-color:#000000;">sp</span><span id="lstnumberx50.107" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx50.108" style="font-size:70%;--ltx-fg-color:#000000;">denom</span> <span id="lstnumberx50.109" style="font-size:70%;--ltx-fg-color:#000000;">(</span><span id="lstnumberx50.110" style="font-size:70%;--ltx-fg-color:#000000;">d</span><span id="lstnumberx50.111" style="font-size:70%;--ltx-fg-color:#000000;">))\""}</span> </span><span id="lstnumberx51"><span id="lstnumberx51.1" style="font-size:70%;--ltx-fg-color:#000000;">}</span> </span><span id="lstnumberx52"><span id="lstnumberx52.1" style="font-size:70%;--ltx-fg-color:#000000;">Observation</span><span id="lstnumberx52.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx52.4" style="font-size:70%;--ltx-fg-color:#000000;">252/25</span> <span id="lstnumberx52.6" style="font-size:70%;--ltx-fg-color:#000000;">277</span> </span><span id="lstnumberx54"><span id="lstnumberx54.1" style="font-size:70%;--ltx-fg-color:#000000;">Thought</span><span id="lstnumberx54.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx54.4" style="font-size:70%;--ltx-fg-color:#000000;">距离是</span> <span id="lstnumberx54.10" style="font-size:70%;--ltx-fg-color:#000000;">252/25,</span><span id="lstnumberx54.12" style="font-size:70%;--ltx-fg-color:#000000;">因此</span> <span id="lstnumberx54.14" style="font-size:70%;--ltx-fg-color:#000000;">m</span> <span id="lstnumberx54.15" style="font-size:70%;--ltx-fg-color:#000000;">+</span> <span id="lstnumberx54.16" style="font-size:70%;--ltx-fg-color:#000000;">n</span> <span id="lstnumberx54.17" style="font-size:70%;--ltx-fg-color:#000000;">=277.</span></span> <span id="lstnumberx55"><span id="lstnumberx55.1" style="font-size:70%;--ltx-fg-color:#000000;">最终答案</span><span id="lstnumberx55.4" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx55.6" style="font-size:70%;--ltx-fg-color:#000000;">\</span> <span id="lstnumberx55.7" style="font-size:70%;--ltx-fg-color:#000000;">boxed</span> <span id="lstnumberx55.8" style="font-size:70%;--ltx-fg-color:#000000;">{277}</span></span></span></span></foreignobject></g></g></svg>

#### E.2.2 轨迹到技能组合（Trajectory-to-Skill Composition）

数学任务会考虑所有25代ES生成的轨迹，对于400个训练问题中的每一个最多保留一条失败轨迹，并将成功轨迹排除在技能蒸馏（Skill Distillation）之外。单次事后Trace2Skill过程使用GPT-5.4-nano模型，温度参数设为$1$。我们使用三个随机种子运行每个模型来蒸馏技能（基础模型、Agentic RL模型、Agentic ESOpt模型），并选择最佳结果进行报告。最终评估会重建相同的无技能ES更新，并仅将蒸馏得到的技能注入到系统上下文中；它不执行进一步的参数优化。

### E.3 DocVQA

#### E.3.1 ReAct提示

DocVQA同样是一个智能体化的ReAct设置。该任务提供文档图像路径，智能体可以使用命令行或Python工具来检查图像、裁剪区域或运行OCR（光学字符识别），然后提交一个简短答案。评估器使用ANLS对最终提交的答案进行评分。

<svg id="A5.SS3.SSS1.p2.pic1" height="50.1" overflow="visible" version="1.1" viewBox="0 0 650 50.1" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,50.1) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#337655;" fill="#337655" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.84 L 0 27.67 C 0 30.34 2.17 32.51 4.84 32.51 L 645.16 32.51 C 647.83 32.51 650 30.34 650 27.67 L 650 4.84 C 650 2.17 647.83 0 645.16 0 L 4.84 0 C 2.17 0 0 2.17 0 4.84 Z" /></g><g style="--ltx-fill-color:#F0FAF5;" fill="#F0FAF5" fill-opacity="1.0"><path style="stroke:none" d="M 0.9 4.84 L 0.9 27.67 C 0.9 29.85 2.66 31.61 4.84 31.61 L 645.16 31.61 C 647.34 31.61 649.1 29.85 649.1 27.67 L 649.1 4.84 C 649.1 2.66 647.34 0.9 645.16 0.9 L 4.84 0.9 C 2.66 0.9 0.9 2.66 0.9 4.84 Z" /></g><g transform="matrix(1.0 0.0 0.0 1.0 6.92 29.74)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="matrix(1 0 0 1 0 0)"><g style="--ltx-fill-color:#337655;" fill="#337655" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.7 L 0 16.65 C 0 18.7 1.66 20.36 3.7 20.36 L 168.11 20.36 C 170.15 20.36 171.81 18.7 171.81 16.65 L 171.81 3.7 C 171.81 1.66 170.15 0 168.11 0 L 3.7 0 C 1.66 0 0 1.66 0 3.7 Z" /></g><g style="--ltx-fill-color:#337655;" fill="#337655" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 3.7 L 0.55 16.65 C 0.55 18.39 1.96 19.8 3.7 19.8 L 168.11 19.8 C 169.85 19.8 171.26 18.39 171.26 16.65 L 171.26 3.7 C 171.26 1.96 169.85 0.55 168.11 0.55 L 3.7 0.55 C 1.96 0.55 0.55 1.96 0.55 3.7 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.64 7.76)"><foreignobject style="--ltx-fo-width:12.33em;--ltx-fo-height:0.54em;--ltx-fo-depth:0.15em;font-size:9.06pt;" height="8.61" overflow="visible" transform="matrix(1 0 0 -1 0 6.73)" width="154.53"><span id="A5.SS3.SSS1.p2.pic1.1" style="font-size:70%;--ltx-fg-color:#FFFFFF;">DocVQA ReAct 提示</span></foreignobject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 9.2 10.24)"><foreignobject style="--ltx-fo-width:45.65em;--ltx-fo-height:0.64em;--ltx-fo-depth:0.18em;font-size:10pt;" height="11.34" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="631.67"><span id="A5.SS3.SSS1.p2.pic1.2" style="width:45.65em;"><span id="A5.SS3.SSS1.p2.pic1.2.1"><a download="">⬇</a> <span id="lstnumberx56"><span id="lstnumberx56.1" style="font-size:70%;--ltx-fg-color:#000000;">回答文档视觉问题。</span><span id="lstnumberx56.3" style="font-size:70%;--ltx-fg-color:#000000;">仔细检查引用的文档图像。</span><span id="lstnumberx56.5" style="font-size:70%;--ltx-fg-color:#000000;">你可以对图像文件使用命令行和</span> <span id="lstnumberx56.7" style="font-size:70%;--ltx-fg-color:#000000;">Python</span> <span id="lstnumberx56.9" style="font-size:70%;--ltx-fg-color:#000000;">工具。</span></span> <span id="lstnumberx58"><span id="lstnumberx58.1" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx58.2" style="font-size:70%;--ltx-fg-color:#000000;">可选的技能指令会插入到这里。</span><span id="lstnumberx58.8" style="font-size:70%;--ltx-fg-color:#000000;">]</span> </span><span id="lstnumberx60"><span id="lstnumberx60.1" style="font-size:70%;--ltx-fg-color:#000000;">动作格式：</span></span> <span id="lstnumberx61"><span id="lstnumberx61.1" style="font-size:70%;--ltx-fg-color:#000000;">Action</span><span id="lstnumberx61.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span></span> <span id="lstnumberx62"><span id="lstnumberx62.1" style="font-size:70%;--ltx-fg-color:#000000;">{</span> </span><span id="lstnumberx63"><span id="lstnumberx63.2" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx63.3" style="font-size:70%;--ltx-fg-color:#000000;">name</span> <span id="lstnumberx63.4" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx63.6" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx63.7" style="font-size:70%;--ltx-fg-color:#000000;">bash</span> <span id="lstnumberx63.8" style="font-size:70%;--ltx-fg-color:#000000;">",</span></span> <span id="lstnumberx64"><span id="lstnumberx64.2" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx64.3" style="font-size:70%;--ltx-fg-color:#000000;">arguments</span> <span id="lstnumberx64.4" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx64.6" style="font-size:70%;--ltx-fg-color:#000000;">{"</span> <span id="lstnumberx64.7" style="font-size:70%;--ltx-fg-color:#000000;">command</span> <span id="lstnumberx64.8" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx64.10" style="font-size:70%;--ltx-fg-color:#000000;">"&lt;</span> <span id="lstnumberx64.11" style="font-size:70%;--ltx-fg-color:#000000;">shell</span> <span id="lstnumberx64.13" style="font-size:70%;--ltx-fg-color:#000000;">command</span> <span id="lstnumberx64.14" style="font-size:70%;--ltx-fg-color:#000000;">&gt;"}</span> </span><span id="lstnumberx65"><span id="lstnumberx65.1" style="font-size:70%;--ltx-fg-color:#000000;">}</span> </span><span id="lstnumberx66"><span id="lstnumberx66.1" style="font-size:70%;--ltx-fg-color:#000000;">每次动作后，环境会返回一个观察结果。</span><span id="lstnumberx66.6" style="font-size:70%;--ltx-fg-color:#000000;">根据需要重复思考</span> <span id="lstnumberx66.8" style="font-size:70%;--ltx-fg-color:#000000;">/</span> <span id="lstnumberx66.10" style="font-size:70%;--ltx-fg-color:#000000;">动作</span> <span id="lstnumberx66.12" style="font-size:70%;--ltx-fg-color:#000000;">/</span> <span id="lstnumberx66.14" style="font-size:70%;--ltx-fg-color:#000000;">观察的过程。</span><span id="lstnumberx66.17" style="font-size:70%;--ltx-fg-color:#000000;">完成后，仅输出简短的最终答案。</span></span> <span id="lstnumberx68"><span id="lstnumberx68.1" style="font-size:70%;--ltx-fg-color:#000000;">图像路径：</span><span id="lstnumberx68.6" style="font-size:70%;--ltx-fg-color:#000000;">data</span> <span id="lstnumberx68.7" style="font-size:70%;--ltx-fg-color:#000000;">\</span> <span id="lstnumberx68.8" style="font-size:70%;--ltx-fg-color:#000000;">trace2skill</span> <span id="lstnumberx68.9" style="font-size:70%;--ltx-fg-color:#000000;">\</span> <span id="lstnumberx68.10" style="font-size:70%;--ltx-fg-color:#000000;">docvqa</span> <span id="lstnumberx68.11" style="font-size:70%;--ltx-fg-color:#000000;">\</span> <span id="lstnumberx68.12" style="font-size:70%;--ltx-fg-color:#000000;">images</span> <span id="lstnumberx68.13" style="font-size:70%;--ltx-fg-color:#000000;">\49153.</span><span id="lstnumberx68.14" style="font-size:70%;--ltx-fg-color:#000000;">png</span> </span><span id="lstnumberx69"><span id="lstnumberx69.1" style="font-size:70%;--ltx-fg-color:#000000;">问题：</span><span id="lstnumberx69.4" style="font-size:70%;--ltx-fg-color:#000000;">在</span> <span id="lstnumberx69.6" style="font-size:70%;--ltx-fg-color:#000000;">1975</span> <span id="lstnumberx69.8" style="font-size:70%;--ltx-fg-color:#000000;">年，每</span> <span id="lstnumberx69.10" style="font-size:70%;--ltx-fg-color:#000000;">1000</span> <span id="lstnumberx69.11" style="font-size:70%;--ltx-fg-color:#000000;">的</span> <span id="lstnumberx69.12" style="font-size:70%;--ltx-fg-color:#000000;">'</span> <span id="lstnumberx69.14" style="font-size:70%;--ltx-fg-color:#000000;">actual</span> <span id="lstnumberx69.16" style="font-size:70%;--ltx-fg-color:#000000;">'</span> <span id="lstnumberx69.18" style="font-size:70%;--ltx-fg-color:#000000;">值是多少？</span></span> <span id="lstnumberx71"><span id="lstnumberx71.1" style="font-size:70%;--ltx-fg-color:#000000;">Thought</span><span id="lstnumberx71.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx71.4" style="font-size:70%;--ltx-fg-color:#000000;">我需要检查文档图表并找到</span> <span id="lstnumberx71.6" style="font-size:70%;--ltx-fg-color:#000000;">1975</span> <span id="lstnumberx71.8" style="font-size:70%;--ltx-fg-color:#000000;">年标记为</span> <span id="lstnumberx71.10" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx71.12" style="font-size:70%;--ltx-fg-color:#000000;">actual</span> <span id="lstnumberx71.14" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx71.16" style="font-size:70%;--ltx-fg-color:#000000;">序列的值。</span></span> <span id="lstnumberx72"><span id="lstnumberx72.1" style="font-size:70%;--ltx-fg-color:#000000;">Action</span><span id="lstnumberx72.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span></span> <span id="lstnumberx73"><span id="lstnumberx73.1" style="font-size:70%;--ltx-fg-color:#000000;">{</span> </span><span id="lstnumberx74"><span id="lstnumberx74.2" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx74.3" style="font-size:70%;--ltx-fg-color:#000000;">name</span> <span id="lstnumberx74.4" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx74.6" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx74.7" style="font-size:70%;--ltx-fg-color:#000000;">bash</span> <span id="lstnumberx74.8" style="font-size:70%;--ltx-fg-color:#000000;">",</span></span> <span id="lstnumberx75"><span id="lstnumberx75.2" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx75.3" style="font-size:70%;--ltx-fg-color:#000000;">arguments</span> <span id="lstnumberx75.4" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx75.6" style="font-size:70%;--ltx-fg-color:#000000;">{"</span> <span id="lstnumberx75.7" style="font-size:70%;--ltx-fg-color:#000000;">command</span> <span id="lstnumberx75.8" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx75.10" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx75.11" style="font-size:70%;--ltx-fg-color:#000000;">python</span> <span id="lstnumberx75.13" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx75.14" style="font-size:70%;--ltx-fg-color:#000000;">c</span> <span id="lstnumberx75.16" style="font-size:70%;--ltx-fg-color:#000000;">\"</span> <span id="lstnumberx75.17" style="font-size:70%;--ltx-fg-color:#000000;">from</span> <span id="lstnumberx75.19" style="font-size:70%;--ltx-fg-color:#000000;">PIL</span> <span id="lstnumberx75.21" style="font-size:70%;--ltx-fg-color:#000000;">import</span> <span id="lstnumberx75.23" style="font-size:70%;--ltx-fg-color:#000000;">Image</span><span id="lstnumberx75.24" style="font-size:70%;--ltx-fg-color:#000000;">;</span><span id="lstnumberx75.26" style="font-size:70%;--ltx-fg-color:#000000;">p</span> <span id="lstnumberx75.27" style="font-size:70%;--ltx-fg-color:#000000;">=</span> <span id="lstnumberx75.28" style="font-size:70%;--ltx-fg-color:#000000;">r</span> <span id="lstnumberx75.29" style="font-size:70%;--ltx-fg-color:#000000;">'</span> <span id="lstnumberx75.30" style="font-size:70%;--ltx-fg-color:#000000;">data</span> <span id="lstnumberx75.31" style="font-size:70%;--ltx-fg-color:#000000;">\\</span> <span id="lstnumberx75.32" style="font-size:70%;--ltx-fg-color:#000000;">trace2skill</span> <span id="lstnumberx75.33" style="font-size:70%;--ltx-fg-color:#000000;">\\</span> <span id="lstnumberx75.34" style="font-size:70%;--ltx-fg-color:#000000;">docvqa</span> <span id="lstnumberx75.35" style="font-size:70%;--ltx-fg-color:#000000;">\\</span> <span id="lstnumberx75.36" style="font-size:70%;--ltx-fg-color:#000000;">images</span> <span id="lstnumberx75.37" style="font-size:70%;--ltx-fg-color:#000000;">\\49153.</span><span id="lstnumberx75.38" style="font-size:70%;--ltx-fg-color:#000000;">png</span> <span id="lstnumberx75.39" style="font-size:70%;--ltx-fg-color:#000000;">';</span><span id="lstnumberx75.41" style="font-size:70%;--ltx-fg-color:#000000;">img</span> <span id="lstnumberx75.42" style="font-size:70%;--ltx-fg-color:#000000;">=</span> <span id="lstnumberx75.43" style="font-size:70%;--ltx-fg-color:#000000;">Image</span><span id="lstnumberx75.44" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx75.45" style="font-size:70%;--ltx-fg-color:#000000;">open</span> <span id="lstnumberx75.46" style="font-size:70%;--ltx-fg-color:#000000;">(</span><span id="lstnumberx75.47" style="font-size:70%;--ltx-fg-color:#000000;">p</span><span id="lstnumberx75.48" style="font-size:70%;--ltx-fg-color:#000000;">);</span><span id="lstnumberx75.50" style="font-size:70%;--ltx-fg-color:#000000;">print</span> <span id="lstnumberx75.51" style="font-size:70%;--ltx-fg-color:#000000;">(</span><span id="lstnumberx75.52" style="font-size:70%;--ltx-fg-color:#000000;">img</span><span id="lstnumberx75.53" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx75.54" style="font-size:70%;--ltx-fg-color:#000000;">size</span><span id="lstnumberx75.55" style="font-size:70%;--ltx-fg-color:#000000;">)\""}</span> </span><span id="lstnumberx76"><span id="lstnumberx76.1" style="font-size:70%;--ltx-fg-color:#000000;">}</span> </span><span id="lstnumberx77"><span id="lstnumberx77.1" style="font-size:70%;--ltx-fg-color:#000000;">Observation</span><span id="lstnumberx77.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx77.4" style="font-size:70%;--ltx-fg-color:#000000;">(1695,</span><span id="lstnumberx77.6" style="font-size:70%;--ltx-fg-color:#000000;">2200)</span> </span><span id="lstnumberx79"><span id="lstnumberx79.1" style="font-size:70%;--ltx-fg-color:#000000;">Thought</span><span id="lstnumberx79.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx79.4" style="font-size:70%;--ltx-fg-color:#000000;">我应该放大图表区域中</span> <span id="lstnumberx79.6" style="font-size:70%;--ltx-fg-color:#000000;">1975</span> <span id="lstnumberx79.8" style="font-size:70%;--ltx-fg-color:#000000;">刻度附近的位置，并在回答前读取</span> <span id="lstnumberx79.10" style="font-size:70%;--ltx-fg-color:#000000;">actual</span> <span id="lstnumberx79.12" style="font-size:70%;--ltx-fg-color:#000000;">序列的标签</span> <span id="lstnumberx79.14" style="font-size:70%;--ltx-fg-color:#000000;">/</span> <span id="lstnumberx79.16" style="font-size:70%;--ltx-fg-color:#000000;">值。</span></span> <span id="lstnumberx80"><span id="lstnumberx80.1" style="font-size:70%;--ltx-fg-color:#000000;">Action</span><span id="lstnumberx80.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span></span> <span id="lstnumberx81"><span id="lstnumberx81.1" style="font-size:70%;--ltx-fg-color:#000000;">{</span> </span><span id="lstnumberx82"><span id="lstnumberx82.2" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx82.3" style="font-size:70%;--ltx-fg-color:#000000;">name</span> <span id="lstnumberx82.4" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx82.6" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx82.7" style="font-size:70%;--ltx-fg-color:#000000;">bash</span> <span id="lstnumberx82.8" style="font-size:70%;--ltx-fg-color:#000000;">",</span></span> <span id="lstnumberx83"><span id="lstnumberx83.2" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx83.3" style="font-size:70%;--ltx-fg-color:#000000;">arguments</span> <span id="lstnumberx83.4" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx83.6" style="font-size:70%;--ltx-fg-color:#000000;">{"</span> <span id="lstnumberx83.7" style="font-size:70%;--ltx-fg-color:#000000;">command</span> <span id="lstnumberx83.8" style="font-size:70%;--ltx-fg-color:#000000;">":</span><span id="lstnumberx83.10" style="font-size:70%;--ltx-fg-color:#000000;">"</span> <span id="lstnumberx83.11" style="font-size:70%;--ltx-fg-color:#000000;">python</span> <span id="lstnumberx83.13" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx83.14" style="font-size:70%;--ltx-fg-color:#000000;">c</span> <span id="lstnumberx83.16" style="font-size:70%;--ltx-fg-color:#000000;">\"</span> <span id="lstnumberx83.17" style="font-size:70%;--ltx-fg-color:#000000;">from</span> <span id="lstnumberx83.19" style="font-size:70%;--ltx-fg-color:#000000;">PIL</span> <span id="lstnumberx83.21" style="font-size:70%;--ltx-fg-color:#000000;">import</span> <span id="lstnumberx83.23" style="font-size:70%;--ltx-fg-color:#000000;">Image</span><span id="lstnumberx83.24" style="font-size:70%;--ltx-fg-color:#000000;">;</span><span id="lstnumberx83.26" style="font-size:70%;--ltx-fg-color:#000000;">p</span> <span id="lstnumberx83.27" style="font-size:70%;--ltx-fg-color:#000000;">=</span> <span id="lstnumberx83.28" style="font-size:70%;--ltx-fg-color:#000000;">r</span> <span id="lstnumberx83.29" style="font-size:70%;--ltx-fg-color:#000000;">'</span> <span id="lstnumberx83.30" style="font-size:70%;--ltx-fg-color:#000000;">data</span> <span id="lstnumberx83.31" style="font-size:70%;--ltx-fg-color:#000000;">\\</span> <span id="lstnumberx83.32" style="font-size:70%;--ltx-fg-color:#000000;">trace2skill</span> <span id="lstnumberx83.33" style="font-size:70%;--ltx-fg-color:#000000;">\\</span> <span id="lstnumberx83.34" style="font-size:70%;--ltx-fg-color:#000000;">docvqa</span> <span id="lstnumberx83.35" style="font-size:70%;--ltx-fg-color:#000000;">\\</span> <span id="lstnumberx83.36" style="font-size:70%;--ltx-fg-color:#000000;">images</span> <span id="lstnumberx83.37" style="font-size:70%;--ltx-fg-color:#000000;">\\49153.</span><span id="lstnumberx83.38" style="font-size:70%;--ltx-fg-color:#000000;">png</span> <span id="lstnumberx83.39" style="font-size:70%;--ltx-fg-color:#000000;">';</span><span id="lstnumberx83.41" style="font-size:70%;--ltx-fg-color:#000000;">img</span> <span id="lstnumberx83.42" style="font-size:70%;--ltx-fg-color:#000000;">=</span> <span id="lstnumberx83.43" style="font-size:70%;--ltx-fg-color:#000000;">Image</span><span id="lstnumberx83.44" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx83.45" style="font-size:70%;--ltx-fg-color:#000000;">open</span> <span id="lstnumberx83.46" style="font-size:70%;--ltx-fg-color:#000000;">(</span><span id="lstnumberx83.47" style="font-size:70%;--ltx-fg-color:#000000;">p</span><span id="lstnumberx83.48" style="font-size:70%;--ltx-fg-color:#000000;">);</span><span id="lstnumberx83.50" style="font-size:70%;--ltx-fg-color:#000000;">crop</span> <span id="lstnumberx83.51" style="font-size:70%;--ltx-fg-color:#000000;">=</span> <span id="lstnumberx83.52" style="font-size:70%;--ltx-fg-color:#000000;">img</span><span id="lstnumberx83.53" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx83.54" style="font-size:70%;--ltx-fg-color:#000000;">crop</span> <span id="lstnumberx83.55" style="font-size:70%;--ltx-fg-color:#000000;">((250,650,1450,1350));</span><span id="lstnumberx83.57" style="font-size:70%;--ltx-fg-color:#000000;">crop</span><span id="lstnumberx83.58" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx83.59" style="font-size:70%;--ltx-fg-color:#000000;">save</span> <span id="lstnumberx83.60" style="font-size:70%;--ltx-fg-color:#000000;">('</span> <span id="lstnumberx83.61" style="font-size:70%;--ltx-fg-color:#000000;">docvqa_49153_chart_crop</span><span id="lstnumberx83.62" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx83.63" style="font-size:70%;--ltx-fg-color:#000000;">png</span> <span id="lstnumberx83.64" style="font-size:70%;--ltx-fg-color:#000000;">');</span><span id="lstnumberx83.66" style="font-size:70%;--ltx-fg-color:#000000;">print</span> <span id="lstnumberx83.67" style="font-size:70%;--ltx-fg-color:#000000;">('</span> <span id="lstnumberx83.68" style="font-size:70%;--ltx-fg-color:#000000;">saved</span> <span id="lstnumberx83.70" style="font-size:70%;--ltx-fg-color:#000000;">docvqa_49153_chart_crop</span><span id="lstnumberx83.71" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx83.72" style="font-size:70%;--ltx-fg-color:#000000;">png</span> <span id="lstnumberx83.73" style="font-size:70%;--ltx-fg-color:#000000;">')\""}</span> </span><span id="lstnumberx84"><span id="lstnumberx84.1" style="font-size:70%;--ltx-fg-color:#000000;">}</span> </span><span id="lstnumberx85"><span id="lstnumberx85.1" style="font-size:70%;--ltx-fg-color:#000000;">Observation</span><span id="lstnumberx85.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx85.4" style="font-size:70%;--ltx-fg-color:#000000;">saved</span> <span id="lstnumberx85.6" style="font-size:70%;--ltx-fg-color:#000000;">docvqa_49153_chart_crop</span><span id="lstnumberx85.7" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx85.8" style="font-size:70%;--ltx-fg-color:#000000;">png</span> </span><span id="lstnumberx87"><span id="lstnumberx87.1" style="font-size:70%;--ltx-fg-color:#000000;">Thought</span><span id="lstnumberx87.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx87.4" style="font-size:70%;--ltx-fg-color:#000000;">图表裁剪显示</span> <span id="lstnumberx87.6" style="font-size:70%;--ltx-fg-color:#000000;">1975</span> <span id="lstnumberx87.8" style="font-size:70%;--ltx-fg-color:#000000;">年的</span> <span id="lstnumberx87.10" style="font-size:70%;--ltx-fg-color:#000000;">actual</span> <span id="lstnumberx87.12" style="font-size:70%;--ltx-fg-color:#000000;">值为每</span> <span id="lstnumberx87.14" style="font-size:70%;--ltx-fg-color:#000000;">1000</span> <span id="lstnumberx87.16" style="font-size:70%;--ltx-fg-color:#000000;">中的</span> <span id="lstnumberx87.18" style="font-size:70%;--ltx-fg-color:#000000;">0.28</span> <span id="lstnumberx87.20" style="font-size:70%;--ltx-fg-color:#000000;">。</span></span> <span id="lstnumberx88"><span id="lstnumberx88.1" style="font-size:70%;--ltx-fg-color:#000000;">最终答案：</span><span id="lstnumberx88.6" style="font-size:70%;--ltx-fg-color:#000000;">0.28</span></span></span></span></foreignobject></g></g></svg>

#### E.3.2 轨迹到技能组合（Trajectory-to-Skill Composition）

DocVQA 选择最后 50 个任务实例，并对每个实例最多保留一条失败轨迹和一条成功轨迹；当某一结果类型不可用时，仅保留观察到的类型。单次事后 Trace2Skill 处理使用 GPT-5.4-nano 模型，温度参数为 $1$。我们使用三个随机种子运行每个模型（基础模型、Agentic RL 模型、Agentic ESOpt 模型）来提取技能，并选择最佳结果进行报告。最终评估重建相同的无技能 ES 更新（No Skill ES updates），并仅将提取的技能注入到系统上下文中；它不执行进一步的参数优化。

### E.4 WebArena-Lite

#### E.4.1 环境提示（Environment Prompt）

WebArena-Lite 使用基于 ID 的可访问性树提示（id-based accessibility-tree prompt）。完整的观察内容可能很长，因此下面的框图展示了稳定的提示结构和一个真实的 WebArena-Lite 任务目标。

<svg id="A5.SS4.SSS1.p2.pic1" height="50.1" overflow="visible" version="1.1" viewBox="0 0 650 50.1" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,50.1) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#1A5B3C;" fill="#1A5B3C" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.84 L 0 27.67 C 0 30.34 2.17 32.51 4.84 32.51 L 645.16 32.51 C 647.83 32.51 650 30.34 650 27.67 L 650 4.84 C 650 2.17 647.83 0 645.16 0 L 4.84 0 C 2.17 0 0 2.17 0 4.84 Z" /></g><g style="--ltx-fill-color:#EEF7F1;" fill="#EEF7F1" fill-opacity="1.0"><path style="stroke:none" d="M 0.9 4.84 L 0.9 27.67 C 0.9 29.85 2.66 31.61 4.84 31.61 L 645.16 31.61 C 647.34 31.61 649.1 29.85 649.1 27.67 L 649.1 4.84 C 649.1 2.66 647.34 0.9 645.16 0.9 L 4.84 0.9 C 2.66 0.9 0.9 2.66 0.9 4.84 Z" /></g><g transform="matrix(1.0 0.0 0.0 1.0 6.92 29.74)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="matrix(1 0 0 1 0 0)"><g style="--ltx-fill-color:#1A5B3C;" fill="#1A5B3C" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.7 L 0 16.65 C 0 18.7 1.66 20.36 3.7 20.36 L 163.16 20.36 C 165.21 20.36 166.87 18.7 166.87 16.65 L 166.87 3.7 C 166.87 1.66 165.21 0 163.16 0 L 3.7 0 C 1.66 0 0 1.66 0 3.7 Z" /></g><g style="--ltx-fill-color:#1A5B3C;" fill="#1A5B3C" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 3.7 L 0.55 16.65 C 0.55 18.39 1.96 19.8 3.7 19.8 L 163.16 19.8 C 164.9 19.8 166.31 18.39 166.31 16.65 L 166.31 3.7 C 166.31 1.96 164.9 0.55 163.16 0.55 L 3.7 0.55 C 1.96 0.55 0.55 1.96 0.55 3.7 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.64 7.76)"><foreignobject style="--ltx-fo-width:12.01em;--ltx-fo-height:0.54em;--ltx-fo-depth:0.15em;font-size:9.06pt;" height="8.61" overflow="visible" transform="matrix(1 0 0 -1 0 6.73)" width="150.62"><span id="A5.SS4.SSS1.p2.pic1.1" style="font-size:70%;--ltx-fg-color:#FFFFFF;">Prompt for WebArena-Lite</span></foreignobject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 9.2 10.24)"><foreignobject style="--ltx-fo-width:45.65em;--ltx-fo-height:0.64em;--ltx-fo-depth:0.18em;font-size:10pt;" height="11.34" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="631.67"><span id="A5.SS4.SSS1.p2.pic1.2" style="width:45.65em;"><span id="A5.SS4.SSS1.p2.pic1.2.1"><a download="">⬇</a> <span id="lstnumberx89"><span id="lstnumberx89.1" style="font-size:70%;--ltx-fg-color:#000000;">You</span> <span id="lstnumberx89.3" style="font-size:70%;--ltx-fg-color:#000000;">are</span> <span id="lstnumberx89.5" style="font-size:70%;--ltx-fg-color:#000000;">an</span> <span id="lstnumberx89.7" style="font-size:70%;--ltx-fg-color:#000000;">autonomous</span> <span id="lstnumberx89.9" style="font-size:70%;--ltx-fg-color:#000000;">intelligent</span> <span id="lstnumberx89.11" style="font-size:70%;--ltx-fg-color:#000000;">agent</span> <span id="lstnumberx89.13" style="font-size:70%;--ltx-fg-color:#000000;">tasked</span> <span id="lstnumberx89.15" style="font-size:70%;--ltx-fg-color:#000000;">with</span> <span id="lstnumberx89.17" style="font-size:70%;--ltx-fg-color:#000000;">navigating</span> <span id="lstnumberx89.19" style="font-size:70%;--ltx-fg-color:#000000;">a</span> <span id="lstnumberx89.21" style="font-size:70%;--ltx-fg-color:#000000;">web</span> <span id="lstnumberx89.23" style="font-size:70%;--ltx-fg-color:#000000;">browser</span><span id="lstnumberx89.24" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx89.26" style="font-size:70%;--ltx-fg-color:#000000;">You</span> <span id="lstnumberx89.28" style="font-size:70%;--ltx-fg-color:#000000;">will</span> <span id="lstnumberx89.30" style="font-size:70%;--ltx-fg-color:#000000;">be</span> <span id="lstnumberx89.32" style="font-size:70%;--ltx-fg-color:#000000;">given</span> <span id="lstnumberx89.34" style="font-size:70%;--ltx-fg-color:#000000;">web</span> <span id="lstnumberx89.35" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx89.36" style="font-size:70%;--ltx-fg-color:#000000;">based</span> <span id="lstnumberx89.38" style="font-size:70%;--ltx-fg-color:#000000;">tasks</span><span id="lstnumberx89.39" style="font-size:70%;--ltx-fg-color:#000000;">.</span><span id="lstnumberx89.41" style="font-size:70%;--ltx-fg-color:#000000;">The</span> <span id="lstnumberx89.43" style="font-size:70%;--ltx-fg-color:#000000;">information</span> <span id="lstnumberx89.45" style="font-size:70%;--ltx-fg-color:#000000;">includes</span> <span id="lstnumberx89.47" style="font-size:70%;--ltx-fg-color:#000000;">the</span> <span id="lstnumberx89.49" style="font-size:70%;--ltx-fg-color:#000000;">user</span> <span id="lstnumberx89.50" style="font-size:70%;--ltx-fg-color:#000000;">'</span> <span id="lstnumberx89.51" style="font-size:70%;--ltx-fg-color:#000000;">s</span> <span id="lstnumberx89.53" style="font-size:70%;--ltx-fg-color:#000000;">objective</span><span id="lstnumberx89.54" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx89.56" style="font-size:70%;--ltx-fg-color:#000000;">the</span> <span id="lstnumberx89.58" style="font-size:70%;--ltx-fg-color:#000000;">current</span> <span id="lstnumberx89.60" style="font-size:70%;--ltx-fg-color:#000000;">accessibility</span> <span id="lstnumberx89.62" style="font-size:70%;--ltx-fg-color:#000000;">tree</span><span id="lstnumberx89.63" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx89.65" style="font-size:70%;--ltx-fg-color:#000000;">the</span> <span id="lstnumberx89.67" style="font-size:70%;--ltx-fg-color:#000000;">current</span> <span id="lstnumberx89.69" style="font-size:70%;--ltx-fg-color:#000000;">page</span> <span id="lstnumberx89.71" style="font-size:70%;--ltx-fg-color:#000000;">address</span><span id="lstnumberx89.72" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx89.74" style="font-size:70%;--ltx-fg-color:#000000;">open</span> <span id="lstnumberx89.76" style="font-size:70%;--ltx-fg-color:#000000;">tabs</span><span id="lstnumberx89.77" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx89.79" style="font-size:70%;--ltx-fg-color:#000000;">and</span> <span id="lstnumberx89.81" style="font-size:70%;--ltx-fg-color:#000000;">the</span> <span id="lstnumberx89.83" style="font-size:70%;--ltx-fg-color:#000000;">previous</span> <span id="lstnumberx89.85" style="font-size:70%;--ltx-fg-color:#000000;">action</span><span id="lstnumberx89.86" style="font-size:70%;--ltx-fg-color:#000000;">.</span></span> <span id="lstnumberx91"><span id="lstnumberx91.1" style="font-size:70%;--ltx-fg-color:#000000;">Available</span> <span id="lstnumberx91.3" style="font-size:70%;--ltx-fg-color:#000000;">actions</span> <span id="lstnumberx91.5" style="font-size:70%;--ltx-fg-color:#000000;">include</span><span id="lstnumberx91.6" style="font-size:70%;--ltx-fg-color:#000000;">:</span></span> <span id="lstnumberx92"><span id="lstnumberx92.1" style="font-size:70%;--ltx-fg-color:#000000;">click</span> <span id="lstnumberx92.3" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx92.4" style="font-size:70%;--ltx-fg-color:#000000;">id</span><span id="lstnumberx92.5" style="font-size:70%;--ltx-fg-color:#000000;">]</span> </span><span id="lstnumberx93"><span id="lstnumberx93.1" style="font-size:70%;--ltx-fg-color:#000000;">type</span> <span id="lstnumberx93.3" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx93.4" style="font-size:70%;--ltx-fg-color:#000000;">id</span><span id="lstnumberx93.5" style="font-size:70%;--ltx-fg-color:#000000;">]</span> <span id="lstnumberx93.7" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx93.8" style="font-size:70%;--ltx-fg-color:#000000;">content</span><span id="lstnumberx93.9" style="font-size:70%;--ltx-fg-color:#000000;">]</span> <span id="lstnumberx93.11" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx93.12" style="font-size:70%;--ltx-fg-color:#000000;">press_enter_after</span> <span id="lstnumberx93.13" style="font-size:70%;--ltx-fg-color:#000000;">=0|1]</span> </span><span id="lstnumberx94"><span id="lstnumberx94.1" style="font-size:70%;--ltx-fg-color:#000000;">hover</span> <span id="lstnumberx94.3" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx94.4" style="font-size:70%;--ltx-fg-color:#000000;">id</span><span id="lstnumberx94.5" style="font-size:70%;--ltx-fg-color:#000000;">]</span> </span><span id="lstnumberx95"><span id="lstnumberx95.1" style="font-size:70%;--ltx-fg-color:#000000;">press</span> <span id="lstnumberx95.3" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx95.4" style="font-size:70%;--ltx-fg-color:#000000;">key_comb</span><span id="lstnumberx95.5" style="font-size:70%;--ltx-fg-color:#000000;">]</span> </span><span id="lstnumberx96"><span id="lstnumberx96.1" style="font-size:70%;--ltx-fg-color:#000000;">scroll</span> <span id="lstnumberx96.3" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx96.4" style="font-size:70%;--ltx-fg-color:#000000;">down</span> <span id="lstnumberx96.5" style="font-size:70%;--ltx-fg-color:#000000;">|</span> <span id="lstnumberx96.6" style="font-size:70%;--ltx-fg-color:#000000;">up</span><span id="lstnumberx96.7" style="font-size:70%;--ltx-fg-color:#000000;">]</span> </span><span id="lstnumberx97"><span id="lstnumberx97.1" style="font-size:70%;--ltx-fg-color:#000000;">goto</span> <span id="lstnumberx97.3" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx97.4" style="font-size:70%;--ltx-fg-color:#000000;">url</span><span id="lstnumberx97.5" style="font-size:70%;--ltx-fg-color:#000000;">]</span> </span><span id="lstnumberx98"><span id="lstnumberx98.1" style="font-size:70%;--ltx-fg-color:#000000;">go_back</span> </span><span id="lstnumberx99"><span id="lstnumberx99.1" style="font-size:70%;--ltx-fg-color:#000000;">go_forward</span> </span><span id="lstnumberx100"><span id="lstnumberx100.1" style="font-size:70%;--ltx-fg-color:#000000;">stop</span> <span id="lstnumberx100.3" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx100.4" style="font-size:70%;--ltx-fg-color:#000000;">answer</span><span id="lstnumberx100.5" style="font-size:70%;--ltx-fg-color:#000000;">]</span> </span><span id="lstnumberx102"><span id="lstnumberx102.1" style="font-size:70%;--ltx-fg-color:#000000;">OBSERVATION</span><span id="lstnumberx102.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span></span> <span id="lstnumberx103"><span id="lstnumberx103.1" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx103.2" style="font-size:70%;--ltx-fg-color:#000000;">accessibility</span> <span id="lstnumberx103.3" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx103.4" style="font-size:70%;--ltx-fg-color:#000000;">tree</span> <span id="lstnumberx103.6" style="font-size:70%;--ltx-fg-color:#000000;">nodes</span> <span id="lstnumberx103.8" style="font-size:70%;--ltx-fg-color:#000000;">with</span> <span id="lstnumberx103.10" style="font-size:70%;--ltx-fg-color:#000000;">element</span> <span id="lstnumberx103.12" style="font-size:70%;--ltx-fg-color:#000000;">IDs</span><span id="lstnumberx103.13" style="font-size:70%;--ltx-fg-color:#000000;">]</span> </span><span id="lstnumberx104"><span id="lstnumberx104.1" style="font-size:70%;--ltx-fg-color:#000000;">CURRENT</span> <span id="lstnumberx104.3" style="font-size:70%;--ltx-fg-color:#000000;">PAGE</span><span id="lstnumberx104.4" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx104.6" style="font-size:70%;--ltx-fg-color:#000000;">[</span><span id="lstnumberx104.7" style="font-size:70%;--ltx-fg-color:#000000;">administration</span> <span id="lstnumberx104.9" style="font-size:70%;--ltx-fg-color:#000000;">interface</span><span id="lstnumberx104.10" style="font-size:70%;--ltx-fg-color:#000000;">]</span> </span><span id="lstnumberx105"><span id="lstnumberx105.1" style="font-size:70%;--ltx-fg-color:#000000;">OBJECTIVE</span><span id="lstnumberx105.2" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx105.4" style="font-size:70%;--ltx-fg-color:#000000;">What</span> <span id="lstnumberx105.6" style="font-size:70%;--ltx-fg-color:#000000;">is</span> <span id="lstnumberx105.8" style="font-size:70%;--ltx-fg-color:#000000;">the</span> <span id="lstnumberx105.10" style="font-size:70%;--ltx-fg-color:#000000;">top</span> <span id="lstnumberx105.11" style="font-size:70%;--ltx-fg-color:#000000;">-1</span> <span id="lstnumberx105.13" style="font-size:70%;--ltx-fg-color:#000000;">best</span> <span id="lstnumberx105.14" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx105.15" style="font-size:70%;--ltx-fg-color:#000000;">selling</span> <span id="lstnumberx105.17" style="font-size:70%;--ltx-fg-color:#000000;">product</span> <span id="lstnumberx105.19" style="font-size:70%;--ltx-fg-color:#000000;">in</span> <span id="lstnumberx105.21" style="font-size:70%;--ltx-fg-color:#000000;">2022</span> </span><span id="lstnumberx106"><span id="lstnumberx106.1" style="font-size:70%;--ltx-fg-color:#000000;">PREVIOUS</span> <span id="lstnumberx106.3" style="font-size:70%;--ltx-fg-color:#000000;">ACTION</span><span id="lstnumberx106.4" style="font-size:70%;--ltx-fg-color:#000000;">:</span><span id="lstnumberx106.6" style="font-size:70%;--ltx-fg-color:#000000;">None</span> </span><span id="lstnumberx108"><span id="lstnumberx108.1" style="font-size:70%;--ltx-fg-color:#000000;">The</span> <span id="lstnumberx108.3" style="font-size:70%;--ltx-fg-color:#000000;">response</span> <span id="lstnumberx108.5" style="font-size:70%;--ltx-fg-color:#000000;">should</span> <span id="lstnumberx108.7" style="font-size:70%;--ltx-fg-color:#000000;">reason</span> <span id="lstnumberx108.9" style="font-size:70%;--ltx-fg-color:#000000;">briefly</span> <span id="lstnumberx108.11" style="font-size:70%;--ltx-fg-color:#000000;">and</span> <span id="lstnumberx108.13" style="font-size:70%;--ltx-fg-color:#000000;">end</span> <span id="lstnumberx108.15" style="font-size:70%;--ltx-fg-color:#000000;">with</span><span id="lstnumberx108.16" style="font-size:70%;--ltx-fg-color:#000000;">:</span></span> <span id="lstnumberx109"><span id="lstnumberx109.1" style="font-size:70%;--ltx-fg-color:#000000;">In</span> <span id="lstnumberx109.3" style="font-size:70%;--ltx-fg-color:#000000;">summary</span><span id="lstnumberx109.4" style="font-size:70%;--ltx-fg-color:#000000;">,</span><span id="lstnumberx109.6" style="font-size:70%;--ltx-fg-color:#000000;">the</span> <span id="lstnumberx109.8" style="font-size:70%;--ltx-fg-color:#000000;">next</span> <span id="lstnumberx109.10" style="font-size:70%;--ltx-fg-color:#000000;">action</span> <span id="lstnumberx109.12" style="font-size:70%;--ltx-fg-color:#000000;">I</span> <span id="lstnumberx109.14" style="font-size:70%;--ltx-fg-color:#000000;">will</span> <span id="lstnumberx109.16" style="font-size:70%;--ltx-fg-color:#000000;">perform</span> <span id="lstnumberx109.18" style="font-size:70%;--ltx-fg-color:#000000;">is</span> <span id="lstnumberx109.20" style="font-size:70%;--ltx-fg-color:#000000;">'''&lt;</span> <span id="lstnumberx109.21" style="font-size:70%;--ltx-fg-color:#000000;">action</span> <span id="lstnumberx109.22" style="font-size:70%;--ltx-fg-color:#000000;">&gt;'''</span></span></span></span></foreignobject></g></g></svg>

#### E.4.2 Trace2Skill 策略上下文（Trace2Skill Policy Context）

下面的框图展示了由 Trace2Skill 条件化智能体（Trace2Skill-conditioned agents）使用的 WebArena 技能。

<svg id="A5.SS4.SSS2.p2.pic1" height="47.95" overflow="visible" version="1.1" viewBox="0 0 650 47.95" width="650"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,47.95) matrix(1 0 0 -1 0 0)"><g style="--ltx-fill-color:#265A43;" fill="#265A43" fill-opacity="1.0"><path style="stroke:none" d="M 0 4.84 L 0 27.4 C 0 30.07 2.17 32.24 4.84 32.24 L 645.16 32.24 C 647.83 32.24 650 30.07 650 27.4 L 650 4.84 C 650 2.17 647.83 0 645.16 0 L 4.84 0 C 2.17 0 0 2.17 0 4.84 Z" /></g><g style="--ltx-fill-color:#F0F8F3;" fill="#F0F8F3" fill-opacity="1.0"><path style="stroke:none" d="M 0.9 4.84 L 0.9 27.4 C 0.9 29.58 2.66 31.34 4.84 31.34 L 645.16 31.34 C 647.34 31.34 649.1 29.58 649.1 27.4 L 649.1 4.84 C 649.1 2.66 647.34 0.9 645.16 0.9 L 4.84 0.9 C 2.66 0.9 0.9 2.66 0.9 4.84 Z" /></g><g transform="matrix(1.0 0.0 0.0 1.0 6.92 29.47)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="matrix(1 0 0 1 0 0)"><g style="--ltx-fill-color:#265A43;" fill="#265A43" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.7 L 0 14.77 C 0 16.82 1.66 18.47 3.7 18.47 L 164.57 18.47 C 166.62 18.47 168.27 16.82 168.27 14.77 L 168.27 3.7 C 168.27 1.66 166.62 0 164.57 0 L 3.7 0 C 1.66 0 0 1.66 0 3.7 Z" /></g><g style="--ltx-fill-color:#265A43;" fill="#265A43" fill-opacity="1.0"><path style="stroke:none" d="M 0.55 3.7 L 0.55 14.77 C 0.55 16.51 1.96 17.92 3.7 17.92 L 164.57 17.92 C 166.31 17.92 167.72 16.51 167.72 14.77 L 167.72 3.7 C 167.72 1.96 166.31 0.55 164.57 0.55 L 3.7 0.55 C 1.96 0.55 0.55 1.96 0.55 3.7 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 8.64 5.87)"><foreignobject style="--ltx-fo-width:12.21em;--ltx-fo-height:0.54em;--ltx-fo-depth:0em;font-size:9.06pt;" height="6.73" overflow="visible" transform="matrix(1 0 0 -1 0 6.73)" width="153.06"><span id="A5.SS4.SSS2.p2.pic1.1" style="font-size:70%;--ltx-fg-color:#FFFFFF;">Trace2Skill WebArena 技能</span></foreignobject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 9.2 9.97)"><foreignobject style="--ltx-fo-width:45.65em;--ltx-fo-height:0.64em;--ltx-fo-depth:0.16em;font-size:10pt;" height="11.07" overflow="visible" transform="matrix(1 0 0 -1 0 8.92)" width="631.67"><span id="A5.SS4.SSS2.p2.pic1.2" style="width:45.65em;"><span id="A5.SS4.SSS2.p2.pic1.2.1"><a download="">⬇</a> <span id="lstnumberx110"><span id="lstnumberx110.1" style="font-size:70%;--ltx-fg-color:#000000;">#</span> <span id="lstnumberx110.3" style="font-size:70%;--ltx-fg-color:#000000;">WebArena</span> <span id="lstnumberx110.5" style="font-size:70%;--ltx-fg-color:#000000;">技能</span> </span><span id="lstnumberx112"><span id="lstnumberx112.1" style="font-size:70%;--ltx-fg-color:#000000;">##</span> <span id="lstnumberx112.3" style="font-size:70%;--ltx-fg-color:#000000;">WebArena</span> <span id="lstnumberx112.5" style="font-size:70%;--ltx-fg-color:#000000;">工作流</span> </span><span id="lstnumberx113"><span id="lstnumberx113.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx113.3" style="font-size:70%;--ltx-fg-color:#000000;">仅使用可见的页面证据和当前</span> <span id="lstnumberx113.5" style="font-size:70%;--ltx-fg-color:#000000;">WebRL</span> <span id="lstnumberx113.7" style="font-size:70%;--ltx-fg-color:#000000;">ID；在任何导航、刷新、重新渲染或布局更改后重新解析元素。</span></span> <span id="lstnumberx114"><span id="lstnumberx114.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx114.3" style="font-size:70%;--ltx-fg-color:#000000;">从最直接的可见路径开始。在长时间滚动或猜测控件之前，优先使用搜索、过滤器、菜单或内置导航。</span></span> <span id="lstnumberx115"><span id="lstnumberx115.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx115.3" style="font-size:70%;--ltx-fg-color:#000000;">在每次点击、选择、文本输入、滚动或导航步骤后重新读取当前可见页面。如果页面没有明显变化，不要重复相同的操作；选择不同的可见控件或路径。</span></span> <span id="lstnumberx116"><span id="lstnumberx116.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx116.3" style="font-size:70%;--ltx-fg-color:#000000;">在输入之前，验证字段标签、当前值和输入类型。覆盖时清除或全选现有文本。</span></span> <span id="lstnumberx117"><span id="lstnumberx117.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx117.3" style="font-size:70%;--ltx-fg-color:#000000;">对于下拉菜单、选择器、自动完成字段和多选框，首先检查当前控件状态，选择与任务匹配的可见选项，并确认之后显示选定的值。</span></span> <span id="lstnumberx118"><span id="lstnumberx118.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx118.3" style="font-size:70%;--ltx-fg-color:#000000;">将搜索结果、摘要、计数和汇总链接仅视为发现辅助工具。在相关页面或详细视图上明确显示请求的状态之前，不要得出完成结论。</span></span> <span id="lstnumberx119"><span id="lstnumberx119.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx119.3" style="font-size:70%;--ltx-fg-color:#000000;">如果搜索或过滤结果为空、不明确或卡住，调整查询、放宽或缩窄过滤器，或切换到不同的基础视图，而不是重复几乎相同的搜索。</span></span> <span id="lstnumberx120"><span id="lstnumberx120.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx120.3" style="font-size:70%;--ltx-fg-color:#000000;">使用可见的过滤器、标签、行计数和结果范围作为当前视图的真实来源。</span></span> <span id="lstnumberx122"><span id="lstnumberx122.1" style="font-size:70%;--ltx-fg-color:#000000;">##</span> <span id="lstnumberx122.3" style="font-size:70%;--ltx-fg-color:#000000;">列表、分页和完整性</span> </span><span id="lstnumberx123"><span id="lstnumberx123.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx123.3" style="font-size:70%;--ltx-fg-color:#000000;">对于要求所有匹配项目的任务，穷尽分页、滚动和展开，直到没有更多条目且确认请求的基数。</span></span> <span id="lstnumberx124"><span id="lstnumberx124.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx124.3" style="font-size:70%;--ltx-fg-color:#000000;">使用页面控件和可见范围指示器逐步移动到相关范围；不要依赖单个视口或旧的行位置。</span></span> <span id="lstnumberx125"><span id="lstnumberx125.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx125.3" style="font-size:70%;--ltx-fg-color:#000000;">当完整性很重要时，在扫描页面或滚动区域时保持运行计数或去重集合。</span></span> <span id="lstnumberx126"><span id="lstnumberx126.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx126.3" style="font-size:70%;--ltx-fg-color:#000000;">对于计数、总数或范围问题，直接验证每个贡献的可见行或桶，并从源值计算。</span></span> <span id="lstnumberx127"><span id="lstnumberx127.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx127.3" style="font-size:70%;--ltx-fg-color:#000000;">对于最多、最少、最高或最低问题，在选择目标之前，确认完整相关结果集或可信聚合中的全局极值。</span></span> <span id="lstnumberx129"><span id="lstnumberx129.1" style="font-size:70%;--ltx-fg-color:#000000;">##</span> <span id="lstnumberx129.3" style="font-size:70%;--ltx-fg-color:#000000;">详情页面、评论和提取</span> </span><span id="lstnumberx130"><span id="lstnumberx130.1" style="font-size:70%;--ltx-fg-color:#000000;">-</span> <span id="lstnumberx130.3" style="font-size:70%;--ltx-fg-color:#000000;">在回答任何状态、评论、</span></span></span></span></foreignobject></g></g></svg>
