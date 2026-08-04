---
sourceTitle: "GFlowRL: Scaling Distribution-Matching RL to Large Language Models"
title: "GFlowRL：将分布匹配强化学习扩展到大语言模型"
sourceAuthors: "Xiaodong Liu, Michael Xu, Jack W. Stokes, Paul Smolensky, Doug Burger, Jianfeng Gao"
authors: "Xiaodong Liu, Michael Xu, Jack W. Stokes, Paul Smolensky, Doug Burger, Jianfeng Gao"
organization: "Microsoft Research"
sourceDate: "2026-07-27"
translator: "Claude (Opus 5)"
translatedAt: "2026-08-03"
language: "zh-CN"
sourceLanguage: "en"
arxivId: "2607.13394v1"
sourceUrl: "https://arxiv.org/abs/2607.13394"
pdfUrl: "https://arxiv.org/pdf/2607.13394v1.pdf"
htmlUrl: "https://arxiv.org/html/2607.13394v1"
sourceType: "academic-paper"
capturedAt: "2026-08-03"
captureMethod: "baoyu-url-to-markdown"
pipelineRunId: "batch-20260803-213412"
pipelineSource: "translate/batch-20260803-213412/works-ready/gflowrl-translation.md"
sourceFigureCount: 3
translationMode: "normal"
translationWorkflow: "chunked"
---


Xiaodong Liu <sup>†‡∗</sup> Michael Xu <sup>∗</sup> Jack W. Stokes Paul Smolensky Doug Burger Jianfeng Gao <sup>‡</sup>
Microsoft Research
<sup>†</sup> 项目负责人。   <sup>‡</sup> 通讯作者。   <sup>∗</sup> 同等贡献。   

###### 摘要

生成流网络（Generative Flow Networks, GFlowNets）为学习大型推理模型提供了一种有前景的替代方案，不同于奖励最大化（reward-maximizing）强化学习。GFlowNets 通过匹配奖励分布而非坍缩到主导模式，来鼓励多样化的推理路径。最近的工作在数学和代码基准测试上展现出潜力,但将 GFlowNet 风格的强化学习扩展到现代后训练（post-training）流程仍然困难。在对大语言模型后训练至关重要的场景中，模型规模、采样轨迹（rollout）的时域长度、奖励噪声以及分布式系统复杂性都会同时增加,而可学习的提示条件配分函数（prompt-conditional partition function）可能成为梯度不稳定和工程开销的来源，而非有用的归一化器。我们通过系统分析来应对这些挑战，揭示了在大规模基于 GFlowNet 的强化学习中哪些组件真正必要。我们的主要发现是：之前被视为必不可少的可学习配分函数，可以用从训练所需的采样轨迹组中计算的批内蒙特卡洛估计（in-batch Monte Carlo estimate）来替代。基于这一观察，我们提出了 GFlowRL，一种简化的 GFlowNet 风格强化学习算法，它完全移除了辅助配分网络，同时保留了奖励分布匹配目标。该方案由两个稳定器完善：用于采样器-训练器漂移的重要性采样校正（importance-sampling correction），以及用于异常值残差的非对称流间隙裁剪（asymmetric flow-gap clipping）。与先前工作相比，GFlowRL 提升了稳定性和性能。它在数学、代码和对抗性红队测试基准上超越所有对手，在 14B 规模达到 2048 的 Codeforces 评分（与 o3-mini 相差 $25$ Elo），并在 AdvBench 和 HarmBench 上达到最高的平均 ASR@1（分别为 82.5% 和 79.5%），在 FlowRL（之前的 GFlowNet 风格强化学习方法）发散的噪声奖励场景中超越先前的 SOTA 多轮攻击器。同样的方案可迁移到所有评估的 MoE（专家混合模型，Mixture of Experts）配置，最高达 235B 参数，而 FlowRL 在这些配置下再次无法收敛。据我们所知，GFlowRL 是首个在稠密和稀疏架构上都能稳定扩展到大型推理模型的 GFlowNet 风格强化学习算法。代码将在 [https://github.com/microsoft/gflowrl](https://github.com/microsoft/gflowrl) 发布。

## 1 引言

强化学习（RL）已成为当前一代大语言模型后训练的核心范式。当今部署的最强大推理系统，包括 OpenAI o1 [^29] 和 o3 [^30]、DeepSeek-R1 [^9] 以及 Gemini [^16]，都将强化学习作为其后训练流程的核心组件。这一阶段的有效性直接决定了模型在高价值任务上的性能，如数学推理、竞技编程（competitive programming）、科学问题解决和智能体决策。随着模型规模扩大和推理链变长，强化学习算法的选择已成为一阶架构决策。

主流的强化学习方法，包括 PPO [^39]、GRPO [^40]、OMD [^17] 及其变体，本质上都是奖励最大化的。它们将概率质量集中在高奖励模式上，往往导致解决方案多样性的坍缩 [^51]。在链式思维推理（chain-of-thought reasoning）中，多条有效的解决路径对于泛化和鲁棒性至关重要，这种行为引入了结构性限制。现有的缓解措施，如熵奖励（entropy bonuses）[^41]、自适应裁剪和词元级探索，将多样性 [^56] 视为事后正则化器，而保持核心的奖励最大化目标不变。

生成流网络（Generative Flow Networks, GFlowNets）[^3] [^2] [^12] 提供了一种根本不同的范式：它们不是最大化期望奖励，而是学习策略以*按照轨迹奖励的比例*采样。当应用于大语言模型后训练时，这种分布匹配（distribution-matching）视角直接解决了模式坍缩（mode collapse）问题，鼓励覆盖所有高奖励推理路径，而非集中在单一路径上。最近的方法如 FOR [^55] 和 FlowRL [^63] 是最早将这一思想引入大语言模型推理的，它们采用轨迹平衡（trajectory balance）目标和可学习的配分函数近似 $Z_{\phi}(\mathbf{x})$。在 FlowRL 中，$Z_{\phi}(\mathbf{x})$ 被参数化为提示最终隐藏状态之上的三层 MLP，而在 FOR 中，它被表示为所有提示共享的标量。

**扩展问题。** 尽管 FlowRL 展示了 GFlowNet 风格训练的潜力，但在现代大语言模型后训练的关键场景中部署它仍然具有挑战性。即使在稠密（dense）骨干网络上，训练也必须在短优化时域、长推理采样轨迹、大型预训练策略以及紧张的分布式系统预算下运行。在这种设置中，随着规模的增长，与策略一起联合训练新初始化的配分模型变得越来越脆弱：梯度峰值变得更频繁，辅助模块增加了优化器和同步开销，而可学习的归一化器落后于规模大得多的预训练策略。MoE 路由 [^9] [^1] [^8] [^17] [^16] 通过引入非确定性以及采样和训练之间的隐式 off-policy 不匹配 [^43] [^54] 进一步放大了这些问题，但 MoE 最好被视为更普遍的扩展瓶颈的最难压力测试，而非根本原因本身。

**根本原因。** 我们将这些失败归因于一个关键设计选择：可学习的配分函数 $Z_{\phi}$。在大语言模型后训练中，被优化的两个组件之间存在根本性不对称。策略是从具有数十亿个良好训练参数的预训练模型初始化的，只需要适度的更新，而 $Z_{\phi}$ 是随机初始化的，必须在相同的有限训练步骤内从零开始学习复杂的量。因此，在训练的大部分时间里，$\log Z_{\phi}$ 的行为主要是噪声源，导致轨迹平衡目标被方差主导而非奖励信号。第 3.1 节提供了对梯度行为和训练动态的详细分析。从经验上看，我们观察到两个一致的模式：(1) 用随机噪声替换 $Z_{\phi}$ 可获得相当的性能，表明它对建模贡献很小；(2) FlowRL 表现出比 GRPO 大几个数量级的梯度范数，导致显著的优化不稳定性，但并未带来任何可见的准确性提升。

基于这一分析，我们提出了 GFlowRL，一种简化的基于 GFlowNet 的强化学习算法，不使用可学习的配分函数。取而代之的是，我们使用简单而有效的 $\log Z$ 批内蒙特卡洛估计，直接从每个提示在训练期间已采样的采样轨迹组估计。这一简单改变带来三个后果：(i) 梯度范数保持与 GRPO 相同的规模，恢复训练稳定性；(ii) 分布式训练中的辅助网络及其优化器状态全部移除；(iii) 算法使用与 GRPO 几乎相同的基础设施，同时保留轨迹平衡的奖励分布匹配语义。该方案由两个稳定器完善：用于采样器-训练器漂移的重要性采样校正，以及用于异常值残差的非对称流间隙裁剪。据我们所知，GFlowRL 是首个在稠密和稀疏架构上都能稳定扩展的基于 GFlowNet 的强化学习算法。

总之，我们的贡献有三方面：

1. 我们识别出可学习配分函数 $Z_{\phi}$ 的使用是基于 GFlowNet 的大语言模型强化学习训练不稳定的关键来源。我们提供了经验证据表明 $Z_{\phi}$ 对后训练几乎没有益处，反而可能使优化变得不必要地困难。
2. 我们提出了 GFlowRL，其中 $Z_{\phi}$ 被批内蒙特卡洛估计替代，从而移除了辅助网络及其相关的分布式开销，同时保留了轨迹平衡目标的不动点。
3. 我们展示了在稠密和稀疏 MoE 架构上的稳定训练，包括 FlowRL 发散的场景。在 14B 稠密规模上，GFlowRL 达到 2048 Codeforces Elo，比 DeepCoder-14B 高 $+112$，比 FlowRL-14B 高 $+144$，比 OpenAI 的 o1 高 $+157$。在对抗性红队测试上，GFlowRL 在 AdvBench（82.5%）和 HarmBench（79.5%）上达到最高的平均 ASR@1，在 FlowRL 发散的噪声奖励场景中超越先前的 SOTA 多轮攻击器。扩展到 MoE，GFlowRL 在 Qwen3-30B-A3B（仅 3B 激活参数）上达到 1999 Codeforces Elo，同样的方案可迁移到 Qwen3-235B-A22B 的数学任务。

## 2 预备知识

我们将链式思维推理形式化为条件生成任务。给定提示 $\mathbf{x}\in\mathcal{X}$（例如，一个数学问题），策略 $\pi_{\theta}(\mathbf{y}\mid\mathbf{x})$ 自回归地生成响应 $\mathbf{y}=(y_{1},y_{2},\dots,y_{T})$，包含中间推理和最终答案。验证器提供标量奖励 $r(\mathbf{x},\mathbf{y})\in\mathbb{R}$。例如，奖励可以是数学 [^40] 的二元正确性信号或代码 [^15] 的单元测试通过率。

**策略表示法。** 符号 $\pi(\mathbf{y}\mid\mathbf{x})$ 表示策略，即给定提示 $\mathbf{x}$ 的响应 $\mathbf{y}$ 上的条件分布。遵循 [^39] [^40] [^54]，$\pi_{\theta}$ 是可训练策略，$\pi_{\mathrm{ref}}$ 是冻结的参考（例如，预训练模型），$\pi_{\mathrm{old}}$ 是用于采样当前组的采样策略；$\pi_{\mathrm{old}}$ 是 $\pi_{\theta}$ 的快照，用于生成采样轨迹，而 $\pi_{\theta}$ 针对这些采样轨迹进行更新。

**奖励最大化。** 典型的奖励最大化目标如 PPO [^39] 和 GRPO [^40] 优化

$$
\mathcal{J}_{\text{RM}}(\theta)=\mathbb{E}_{\mathbf{x}\sim\mathcal{D},\,\mathbf{y}\sim\pi_{\theta}(\cdot\mid\mathbf{x})}\big[\,r(\mathbf{x},\mathbf{y})\,\big],
$$

这倾向于将概率质量集中在响应分布的单一最高奖励模式上（图 6(a)），导致模式坍缩和减少解决方案多样性 [^51]。

**奖励分布匹配。** 一种替代方案是按照奖励的比例采样响应，而非最大化它。将目标分布定义为 softmax

$$
p^{\star}(\mathbf{y}\mid\mathbf{x})=\frac{1}{Z(\mathbf{x})}\exp\!\left({\beta}r(\mathbf{x},\mathbf{y})\right),\qquad Z(\mathbf{x})=\sum_{\mathbf{y}\in\mathcal{Y}}\exp\!\left({\beta r(\mathbf{x},\mathbf{y})}\right),
$$

其中 $\beta>0$ 是逆温度，控制分布的尖锐度，$Z(\mathbf{x})$ 是（难以处理的）配分函数。最小化反向 KL 散度 $D_{\mathrm{KL}}(\pi_{\theta}\,\|\,p^{\star})$ 鼓励策略按比例覆盖所有高奖励模式，而非坍缩到单一模式 [^3] [^2]。由于难以处理的 $Z(\mathbf{x})$，直接最小化这个 KL 散度是不可行的，这促使了代理目标，它们在学习 $Z$ 作为副产品的同时强制相同的不动点。

**大语言模型推理中的轨迹平衡目标。** FOR [^55] 和 FlowRL [^63] 都通过 GFlowNets 的轨迹平衡（TB）目标 [^24] 实例化了这一分布匹配原则。对于提示 $\mathbf{x}$ 和采样响应 $\mathbf{y}$，TB 损失为

$$
\mathcal{L}_{\text{TB}}(\theta,\phi;\mathbf{x},\mathbf{y})=\left(\log Z_{\phi}(\mathbf{x})+\log\frac{\pi_{\theta}(\mathbf{y}\mid\mathbf{x})}{\pi_{\mathrm{ref}}(\mathbf{y}\mid\mathbf{x})}-{\beta}r(\mathbf{x},\mathbf{y})\right)^{\!2},
$$

其中 $Z_{\phi}(\mathbf{x})$ 是对真实配分函数的*可学习*近似。在 FlowRL 中，$Z_{\phi}(\mathbf{x})$ 是提示最终隐藏状态之上的 3 层 MLP，而在 FOR 中，配分函数是所有提示共享的单个变量。

## 3 方法论

### 3.1 为什么可学习的配分函数会失败

虽然 TB 目标在理论上是有原则的，但我们认为其标准实例化不适合大语言模型后训练场景。关键在于联合优化的两个组件之间的学习时程不匹配。

**学习时程不匹配。** 策略 $\pi_{\theta}$ 从预训练的大语言模型检查点初始化，具有数十亿个参数，这些参数从数万亿词元的预训练 [^34] 中编码了丰富的语言和数学先验。后训练强化学习只需要细化这个强先验，通常运行几百个梯度更新。相比之下，配分函数 $Z_{\phi}$ 是一个新初始化的网络（例如，FlowRL 中的 3 层 MLP），必须在相同的短窗口内从零开始学习复杂的提示条件量。这种不对称意味着在训练的大部分时间里，$\log Z_{\phi}$ 实际上是提示的随机函数。<table><thead><tr><th>模型</th><th colspan="2">LiveCodeBench</th><th colspan="2">Codeforces</th><th>HumanEval+</th></tr><tr><th></th><th>Avg@16</th><th>Pass@16</th><th>Rating</th><th>Percentile</th><th>Avg@16</th></tr><tr><th>[rgb]0.93,0.93,0.93 Backbone</th><th>30.68</th><th>49.46</th><th>886.68</th><th>19.4%</th><th>80.90</th></tr></thead><tbody><tr><td>R++</td><td><math><semantics><msub><mn>30.46</mn> <mrow><mo>−</mo> <mn>0.22</mn></mrow></msub> <annotation>30.46_{{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}-0.22}}</annotation></semantics></math></td><td><math><semantics><msub><mn>52.68</mn> <mrow><mo>+</mo> <mn>3.22</mn></mrow></msub> <annotation>52.68_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+3.22}}</annotation></semantics></math></td><td><math><semantics><msub><mn>1208.03</mn> <mrow><mo>+</mo> <mn>321.35</mn></mrow></msub> <annotation>1208.03_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+321.35}}</annotation></semantics></math></td><td><math><semantics><msub><mrow><mn>56.8</mn> <mo>%</mo></mrow> <mrow><mo>+</mo> <mrow><mn>37.4</mn> <mo>%</mo></mrow></mrow></msub> <annotation>56.8\%_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+37.4\%}}</annotation></semantics></math></td><td><math><semantics><msub><mn>76.61</mn> <mrow><mo>−</mo> <mn>4.29</mn></mrow></msub> <annotation>76.61_{{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}-4.29}}</annotation></semantics></math></td></tr><tr><td>PPO</td><td><math><semantics><msub><mn>35.10</mn> <mrow><mo>+</mo> <mn>4.42</mn></mrow></msub> <annotation>35.10_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+4.42}}</annotation></semantics></math></td><td><math><semantics><msub><mn>54.48</mn> <mrow><mo>+</mo> <mn>5.02</mn></mrow></msub> <annotation>54.48_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+5.02}}</annotation></semantics></math></td><td><math><semantics><msub><mn>1403.07</mn> <mrow><mo>+</mo> <mn>516.39</mn></mrow></msub> <annotation>1403.07_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+516.39}}</annotation></semantics></math></td><td><math><semantics><msub><mrow><mn>73.7</mn> <mo>%</mo></mrow> <mrow><mo>+</mo> <mrow><mn>54.3</mn> <mo>%</mo></mrow></mrow></msub> <annotation>73.7\%_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+54.3\%}}</annotation></semantics></math></td><td><math><semantics><msub><mn>82.32</mn> <mrow><mo>+</mo> <mn>1.42</mn></mrow></msub> <annotation>82.32_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+1.42}}</annotation></semantics></math></td></tr><tr><td>GRPO</td><td><math><semantics><msub><mn>32.75</mn> <mrow><mo>+</mo> <mn>2.07</mn></mrow></msub> <annotation>32.75_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+2.07}}</annotation></semantics></math></td><td><math><semantics><msub><mn>52.32</mn> <mrow><mo>+</mo> <mn>2.86</mn></mrow></msub> <annotation>52.32_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+2.86}}</annotation></semantics></math></td><td><math><semantics><msub><mn>1313.82</mn> <mrow><mo>+</mo> <mn>427.14</mn></mrow></msub> <annotation>1313.82_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+427.14}}</annotation></semantics></math></td><td><math><semantics><msub><mrow><mn>67.1</mn> <mo>%</mo></mrow> <mrow><mo>+</mo> <mrow><mn>47.7</mn> <mo>%</mo></mrow></mrow></msub> <annotation>67.1\%_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+47.7\%}}</annotation></semantics></math></td><td><math><semantics><msub><mn>80.13</mn> <mrow><mo>−</mo> <mn>0.77</mn></mrow></msub> <annotation>80.13_{{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}-0.77}}</annotation></semantics></math></td></tr><tr><td>FlowRL</td><td><math><semantics><msub><mn>37.43</mn> <mrow><mo>+</mo> <mn>6.75</mn></mrow></msub> <annotation>37.43_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+6.75}}</annotation></semantics></math></td><td><math><semantics><msub><mn>56.27</mn> <mrow><mo>+</mo> <mn>6.81</mn></mrow></msub> <annotation>56.27_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+6.81}}</annotation></semantics></math></td><td><math><semantics><msub><mn>1549.47</mn> <mrow><mo>+</mo> <mn>662.79</mn></mrow></msub> <annotation>1549.47_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+662.79}}</annotation></semantics></math></td><td><math><semantics><msub><mrow><mn>83.3</mn> <mo>%</mo></mrow> <mrow><mo>+</mo> <mrow><mn>63.9</mn> <mo>%</mo></mrow></mrow></msub> <annotation>83.3\%_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+63.9\%}}</annotation></semantics></math></td><td><math><semantics><msub><mn>83.28</mn> <mrow><mo>+</mo> <mn>2.38</mn></mrow></msub> <annotation>83.28_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+2.38}}</annotation></semantics></math></td></tr><tr><td>GFlowRL</td><td><math><semantics><msub><mn>38.62</mn> <mrow><mo>+</mo> <mn>7.94</mn></mrow></msub> <annotation>\mathbf{38.62}_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+7.94}}</annotation></semantics></math></td><td><math><semantics><msub><mn>58.06</mn> <mrow><mo>+</mo> <mn>8.60</mn></mrow></msub> <annotation>\mathbf{58.06}_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+8.60}}</annotation></semantics></math></td><td><math><semantics><msub><mn>1646.21</mn> <mrow><mo>+</mo> <mn>759.73</mn></mrow></msub> <annotation>\mathbf{1646.21}_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+759.73}}</annotation></semantics></math></td><td><math><semantics><msub><mrow><mn>88.0</mn> <mo>%</mo></mrow> <mrow><mo>+</mo> <mrow><mn>68.6</mn> <mo>%</mo></mrow></mrow></msub> <annotation>\mathbf{88.0\%}_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+68.6\%}}</annotation></semantics></math></td><td><math><semantics><msub><mn>84.93</mn> <mrow><mo>+</mo> <mn>4.03</mn></mrow></msub> <annotation>\mathbf{84.93}_{{\color[rgb]{0.1328125,0.546875,0.1328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.1328125,0.546875,0.1328125}+4.03}}</annotation></semantics></math></td></tr></tbody></table>

表 4：在 AdvBench（LLM 分类器）和 HarmBench（HarmBench 分类器）上的 ASR@1 $\uparrow$，稠密模型。FlowRL 未能收敛。完整结果见附录 F.3。

<table><thead><tr><th>攻击者 / 受害者模型</th><th colspan="4">AdvBench <sup><a href="#fn:64">64</a></sup></th><th colspan="4">HarmBench <sup><a href="#fn:26">26</a></sup></th></tr><tr><th></th><th>Qwen2.5-3B</th><th>Llama-3.1-8B</th><th>GPT-4.1-mini</th><th>Avg</th><th>Qwen2.5-3B</th><th>Llama-3.1-8B</th><th>GPT-4.1-mini</th><th>Avg</th></tr></thead><tbody><tr><th>X-Teaming <sup><a href="#fn:35">35</a></sup></th><td>39.4</td><td>24.2</td><td>44.2</td><td>36.0</td><td>45.3</td><td>22.0</td><td>44.7</td><td>37.3</td></tr><tr><th>SEMA <sup><a href="#fn:6">6</a></sup></th><td>79.9</td><td>77.2</td><td>83.3</td><td>80.1</td><td>74.5</td><td>70.6</td><td>79.8</td><td>75.0</td></tr><tr><th>GFlowRL</th><td>80.2</td><td>81.2</td><td>86.1</td><td>82.5</td><td>79.9</td><td>73.0</td><td>85.5</td><td>79.5</td></tr></tbody></table>

![Refer to caption](imgs/img-002-x1.png)

图 2：稠密模型和稀疏 MoE 模型系列的 Codeforces 评分对比。

我们在稠密骨干模型上评估 GFlowRL 在数学推理（表 2）、竞技编程和代码生成（表 3，图 2）以及对抗性红队测试（表 4）任务上的表现，与奖励最大化基线（R++、PPO、GRPO）和 FlowRL 进行对比。我们的诊断实验预测了两件事：用批内估计替换 $Z_{\phi}$ 在 FlowRL 能够收敛的地方不应损害准确性，并且应在其无法收敛的地方恢复收敛。32B 规模的结果遵循相同趋势，报告在附录 F.2 中。

**GFlowRL 和 FlowRL 在数学和代码任务上大幅优于 PPO 和 GRPO**。在 7B 数学任务上，GFlowRL 比 GRPO 高出 $+8.44$ 平均分（40.92 vs. 32.48）；在 Codeforces 上，两种生成流网络风格的方法都比 GRPO 开辟了 200-330 Elo 的优势。在两者都收敛的每个基准测试中，GFlowRL 也超过了 FlowRL：最佳 7B 数学平均分（40.92，$+5.29$）、最高 LiveCodeBench Avg@16（38.62，$+1.19$）、最高 Codeforces 评分（1646，$+97$）和百分位（88.0%，$+4.7\%$），以及最高 HumanEval+ 分数（84.93，$+1.65$）。结合第 4.2 节的诊断实验，这证实了 GFlowRL 通过高效且稳定的估计器提供了轨迹平衡的优势。

**GFlowRL 在 FlowRL 未能收敛的地方取得成功**。简化设计的最有力证据来自对抗性红队测试，其中奖励信号比数学或代码任务更嘈杂和稀疏。在 AdvBench 和 HarmBench（表 4）上，FlowRL 未能收敛且没有产生可用的攻击者，而 GFlowRL 训练稳定并在两个基准测试上都达到了最高平均 ASR@1（AdvBench 上 82.5%，HarmBench 上 79.5%），分别超过了最先进的多轮攻击者 SEMA [^6] $+2.4$ 和 $+4.5$ 个百分点。去除高方差的 $\log Z_{\phi}$ 项恢复了稳定的优化，将数学/代码上的平等转变为严格的能力优势：GFlowRL 覆盖了生成流网络风格或 GRPO/PPO 训练任何一种方法有效的任务的并集。

**与前沿推理模型的比较**。图 2a 将 GFlowRL 的 14B Codeforces 评分与开源稠密基线和专有推理系统进行比较。GFlowRL 达到 2048 Elo，比 DeepCoder-14B 高出 $+112$，比 FlowRL-14B 高出 $+144$，在该规模下树立了新的开源最佳水平；DeepCoder 是一个强大的 GRPO 方案，具有针对性的改进，之前保持着这一标准，这使得该优势尤为显著。GFlowRL 还超过了 OpenAI 的 o1 $+157$（1891），并接近 o3-mini（2073）<sup>2</sup> 的 $25$ 以内，这一领域以前仅由前沿闭源系统达到。详细设置报告在附录 11 中。

![Refer to caption](imgs/img-003-x2.png)

表 5：稀疏模型在数学推理基准测试上的结果。我们报告 Avg@16 准确率，相对改进以下标显示。正增益显示为绿色，负变化显示为红色。GFlowRL 在 30B-A3B 和 235B-A22B 模型规模上优于或匹配所有基线。FlowRL 在两种情况下都未能收敛。

## 参考文献

[^1]: GPT-4 technical report. arXiv preprint arXiv:2303.08774. 引用于：§1.

[^2]: Flow network based generative models for non-iterative diverse candidate generation. Neural Information Processing Systems (NeurIPS). 引用于：附录 C、§1、§2、§5.

[^3]: GFlowNet foundations. Journal of Machine Learning Research 24 (210), pp. 1–55. 外部链接：[链接](http://jmlr.org/papers/v24/22-0364.html) 引用于：附录 C、§1、§2、§5.

[^4]: DAPO-math-17k. 外部链接：[链接](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) 引用于：第 1 项、表 8.

[^5]: Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374. 引用于：附录 C、第 2 项、表 8、§4.1.

[^6]: SEMA: simple yet effective learning for multi-turn jailbreak attacks. In The Fourteenth International Conference on Learning Representations, 外部链接：[链接](https://openreview.net/forum?id=6eSNG1VNkl) 引用于：附录 C、第 3 项、§D.3、§F.3、§F.3、§4.1、§4.3、表 4.

[^7]: Backpropagation through the void: optimizing control variates for black-box gradient estimation. In International Conference on Learning Representations, 外部链接：[链接](https://openreview.net/forum?id=SyzKd1bCW) 引用于：附录 C、§5.

[^8]: The llama 3 herd of models. 外部链接：2407.21783, [链接](https://arxiv.org/abs/2407.21783) 引用于：§1.

[^9]: Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning. Nature 645, pp. 633–638. 引用于：附录 C、§D.1、§1、§1、§4.1、§5.

[^10]: Jailbreak-r1: exploring the jailbreak capabilities of llms via reinforcement learning. 外部链接：2506.00782, [链接](https://arxiv.org/abs/2506.00782) 引用于：§F.3.

[^11]: OlympiadBench: a challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Annual Meeting of the Association for Computational Linguistics, 外部链接：[链接](https://api.semanticscholar.org/CorpusID:267770504) 引用于：附录 C、第 1 项、表 8、§4.1.

[^12]: Amortizing intractable inference in large language models. In The Twelfth International Conference on Learning Representations, 外部链接：[链接](https://openreview.net/forum?id=Ouj6p4ca60) 引用于：附录 C、§1.

[^13]: REINFORCE++: stabilizing critic-free policy optimization with global advantage normalization. arXiv preprint arXiv:2501.03262. 引用于：附录 C、附录 C、§D.4、§4.1、§5、§5.

[^14]: HugginfaceH4 Math-500. 外部链接：[链接](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) 引用于：表 8.

[^15]: LiveCodeBench: holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations, Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (Eds.), Vol. 2025, pp. 58791–58831. 外部链接：[链接](https://proceedings.iclr.cc/paper_files/paper/2025/file/94074dd5a072d28ff75a76dabed43767-Paper-Conference.pdf) 引用于：附录 C、第 2 项、表 8、§2、§4.1.

[^16]: Gemini 2.5: Our most intelligent AI model. 注：Google Blog (The Keyword), 发布于 2025 年 3 月 25 日 外部链接：[链接](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) 引用于：§1、§1.

[^17]: Kimi k1.5: scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599. 引用于：附录 C、§1、§1、§5.

[^18]: Solving quantitative reasoning problems with language models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, 引用于：第 1 项、表 8、§4.1.

[^19]: Let's verify step by step. In The Twelfth International Conference on Learning Representations, 引用于：附录 C、第 1 项、表 8、§4.1.

[^20]: FlipAttack: jailbreak llms via flipping. In Proceedings of the 42nd International Conference on Machine Learning, ICML'25. 引用于：§F.3.

[^21]: LiveCodeBench LiveCodeBench. 外部链接：[链接](https://github.com/LiveCodeBench/LiveCodeBench) 引用于：表 8.

[^22]: DeepCoder: a fully open-source 14b coder at o3-mini level. 注：Notion Blog [https://www.together.ai/blog/deepcoder](https://www.together.ai/blog/deepcoder) 引用于：附录 C、§D.3、§D.4、§D.7、表 11、表 11、表 8、附录 I、§4.1、§4.1、§5.

[^23]: Learning gflownets from partial episodes for improved convergence and stability. In Proceedings of the 40th International Conference on Machine Learning, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett (Eds.), Proceedings of Machine Learning Research, Vol. 202, pp. 23467–23483. 外部链接：[链接](https://proceedings.mlr.press/v202/madan23a.html) 引用于：附录 C.

[^24]: Trajectory balance: improved credit assignment in gflownets. In Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35, pp. 5955–5967. 外部链接：[链接](https://proceedings.neurips.cc/paper_files/paper/2022/file/27b51baca8377a0cf109f6ecc15a0f70-Paper-Conference.pdf) 引用于：附录 C、§2、§3.2、§5.

[^25]: GFlownets and variational inference. In The Eleventh International Conference on Learning Representations, 外部链接：[链接](https://openreview.net/forum?id=uKiE0VIluA-) 引用于：附录 C、附录 C.

[^26]: HarmBench: a standardized evaluation framework for automated red teaming and robust refusal. In Proceedings of the 41st International Conference on Machine Learning, ICML'24. 引用于：附录 C、第 3 项、表 8、§F.3、§4.1、表 4.

[^27]: Maximum entropy GFlowNets with soft Q-learning. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, S. Dasgupta, S. Mandt, and Y. Li (Eds.), Proceedings of Machine Learning Research, Vol. 238, pp. 2593–2601. 外部链接：[链接](https://proceedings.mlr.press/v238/mohammadpour24a.html) 引用于：附录 C.

[^28]: Solving high-dimensional hamilton–jacobi–bellman pdes using neural networks: perspectives from the theory of controlled diffusions and measures on path space. Partial Differential Equations and Applications 2. 外部链接：[链接](https://doi.org/10.1007/s42985-021-00102-x) 引用于：附录 C.

[^29]: Introducing openai o1. 注：[https://openai.com/o1/](https://openai.com/o1/) 聚焦推理的大语言模型 引用于：§1.

[^30]: Introducing openai o3 and o4-mini. 注：[https://openai.com/index/introducing-o3-and-o4-mini/](https://openai.com/index/introducing-o3-and-o4-mini/) 大语言模型公告 引用于：§1.

[^31]: Training language models to follow instructions with human feedback. In Proceedings of the 36th International Conference on Neural Information Processing Systems, 引用于：附录 C、§5.

[^32]: Automated red teaming with GOAT: the generative offensive agent tester. In ICLR 2025 Workshop on Building Trust in Language Models and Applications, 外部链接：[链接](https://openreview.net/forum?id=6uyczU6S2M) 引用于：§F.3.

[^33]: CodeForces. Hugging Face. 注：[https://huggingface.co/datasets/open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces) 引用于：附录 C、第 2 项、表 8、§4.1.

[^34]: Qwen2.5: a party of foundation models. 外部链接：[链接](https://qwen.ai/blog?id=qwen2.5) 引用于：§D.1、§3.1、§4.1.

[^35]: X-teaming: multi-turn jailbreaks and defenses with adaptive multi-agents. In Second Conference on Language Modeling, 外部链接：[链接](https://openreview.net/forum?id=gKfj7Jb1kj) 引用于：附录 C、§F.3、表 4.

[^36]: LLMs know their vulnerabilities: uncover safety gaps through natural distribution shifts. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar (Eds.), Vienna, Austria, pp. 24763–24785. 外部链接：[链接](https://aclanthology.org/2025.acl-long.1207/) 引用于：§F.3.

[^37]: VarGrad: a low-variance gradient estimator for variational inference. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33, pp. 13481–13492. 外部链接：[链接](https://proceedings.neurips.cc/paper_files/paper/2020/file/9c22c0b51b3202246463e986c7e205df-Paper.pdf) 引用于：附录 C、§5.

[^38]: Great, now write an article about that: the crescendo multi-turn llm jailbreak attack. In Proceedings of the 34th USENIX Conference on Security Symposium, SEC '25, USA. 引用于：§F.3.

[^39]: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. 引用于：附录 C、§D.4、§1、§2、§2、§3.2、§4.1、§5.

[^40]: Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. 引用于：附录 C、附录 C、§D.4、§1、§2、§2、§2、§4.1、§5、§5.

[^41]: On entropy control in LLM-RL algorithms. In The Fourteenth International Conference on Learning Representations, 外部链接：[链接](https://openreview.net/forum?id=LqazVN5epT) 引用于：§1.

[^42]: ADVLLM: iterative self-tuning llms for enhanced jailbreaking capabilities. 外部链接：2410.18469, [链接](https://arxiv.org/abs/2410.18469) 引用于：§F.3.

[^43]: Defeating nondeterminism in llm inference. 注：[https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) 引用于：§1.

[^44]: Generative flow networks as entropy-regularized RL. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, S. Dasgupta, S. Mandt, and Y. Li (Eds.), Proceedings of Machine Learning Research, Vol. 238, pp. 4213–4221. 外部链接：[链接](https://proceedings.mlr.press/v238/tiapkin24a.html) 引用于：附录 C.

[^45]: REBAR: low-variance, unbiased gradient estimates for discrete latent variable models. In Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.), Vol. 30, pp.. 外部链接：[链接](https://proceedings.neurips.cc/paper_files/paper/2017/file/ebd6d2f5d60ff9afaeda1a81fc53e2d0-Paper.pdf) 引用于：附录 C.

[^46]: Amortizing intractable inference in diffusion models for vision, language, and control. In Proceedings of the 38th International Conference on Neural Information Processing Systems, 引用于：附录 C.

[^47]: The optimal reward baseline for gradient-based reinforcement learning. In Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence, UAI'01, pp. 538–545. 引用于：附录 C.

[^48]: Foot-in-the-door: a multi-turn jailbreak for LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng (Eds.), pp. 1939–1950. 外部链接：[链接](https://aclanthology.org/2025.emnlp-main.100/) 引用于：附录 C、§F.3.

[^49]: Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning 8 (3), pp. 229–256. 引用于：附录 C、§5.

[^50]: Variance reduction for policy gradient with action-dependent factorized baselines. In International Conference on Learning Representations, 外部链接：[链接](https://openreview.net/forum?id=H1tSsb-AW) 引用于：附录 C.

[^51]: Echoes in ai: quantifying lack of plot diversity in llm outputs. Proceedings of the National Academy of Sciences 122 (35), pp. e2504966122. 外部链接：[链接](https://www.pnas.org/doi/abs/10.1073/pnas.2504966122) 引用于：§1、§2.

[^52]: Jigsaw puzzles: splitting harmful questions to jailbreak large language models. 外部链接：2410.11459, [链接](https://arxiv.org/abs/2410.11459) 引用于：§F.3.

[^53]: Chain of attack: a semantic-driven contextual multi-turn attacker for llm. 外部链接：2405.05610, [链接](https://arxiv.org/abs/2405.05610) 引用于：§F.3.

[^54]: Your efficient rl framework secretly brings you off-policy rl training. 注：[https://fengyao.notion.site/off-policy-rl](https://fengyao.notion.site/off-policy-rl) 进行中的工作 引用于：§1、§2、§3.2.

[^55]: Flow of reasoning: training LLMs for divergent reasoning with minimal examples. In Forty-second International Conference on Machine Learning, 外部链接：[链接](https://openreview.net/forum?id=qyMxunrR2j) 引用于：附录 C、§1、§2、§4.4、§5.

[^56]: DAPO: an open-source llm reinforcement learning system at scale. In Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (Eds.), Vol. 38, pp. 113222–113244. 外部链接：[链接](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf) 引用于：附录 C、附录 C、§D.3、§1、§3.2、§3.2、§4.1、§5.

[^57]: Robust scheduling with GFlownets. In The Eleventh International Conference on Learning Representations, 外部链接：[链接](https://openreview.net/forum?id=ZBUthI6wK9h) 引用于：附录 C、§3.2.

[^58]: Generative flow networks for discrete probabilistic modeling. In Proceedings of the 39th International Conference on Machine Learning, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato (Eds.), Proceedings of Machine Learning Research, Vol. 162, pp. 26412–26428. 外部链接：[链接](https://proceedings.mlr.press/v162/zhang22v.html) 引用于：附录 C.

[^59]: Y. Zhang and T. Math-AI Minerva math. 外部链接：[链接](https://huggingface.co/datasets/math-ai/minervamath) 引用于：表 8.

[^60]: American mathematics competitions (amc) 2023. 外部链接：[链接](https://huggingface.co/datasets/math-ai/amc23) 引用于：第 1 项、表 8、§4.1.

[^61]: American invitational mathematics examination (aime) 2024. 引用于：第 1 项、表 8、§4.1.

[^62]: American invitational mathematics examination (aime) 2025. 引用于：第 1 项、表 8、§4.1.

[^63]: FlowRL: matching reward distributions for LLM reasoning. In The Fourteenth International Conference on Learning Representations, 外部链接：[链接](https://openreview.net/forum?id=lObnTKbm9U) 引用于：附录 C、§D.3、§D.4、§D.6、§1、§2、§3.2、§4.1、§4.1、§4.4、§5.

[^64]: Universal and transferable adversarial attacks on aligned language models. 外部链接：2307.15043 引用于：附录 C、第 3 项、§D.3、表 8、表 8、§F.3、§4.1、表 4.
