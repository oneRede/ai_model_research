# 文章索引

> **本文件是文章索引与计数的唯一权威源（single source of truth）。**
>
> **计数规则（machine-checkable）：**
> 一篇文章 = 一个 `### N. {标题}` 形式的编号小节，且不属于本文末尾的"已跟踪产品 / 项目"段落。
> 占位条目（"未找到 / 待补充"）**不写在编号正文里**，而是统一进 `references/AGENTS.md` 的"待补充"列表，避免污染计数。
> 全局连续编号（不按脉络重置），最大编号 = 文章总数。
>
> **下游引用都是本文的冗余缓存：** 根 `README.md` / `README.en.md` 的 badge、`prompts/deep-research-tracker.md` 的去重清单、`references/AGENTS.md` 的概览表。
> 新增/删除文章时，必须**同一次提交**更新本文 + 所有下游缓存。
>
> 当前规模：**15 篇文章**。最近一次同步：2026-08-05。
---

## 📚 文章索引

### 1. Kimi K3：开放前沿智能
- **原标题**：KIMI K3: OPEN FRONTIER INTELLIGENCE
- **作者**：Kimi 团队（月之暗面 Moonshot AI）
- **日期**：2026-07-27
- **类型**：技术报告
- **来源**：arXiv:2607.24653v1
- **译文**：[works/kimi-k3-open-frontier-intelligence.md](../works/kimi-k3-open-frontier-intelligence.md)
- **原文**：https://arxiv.org/abs/2607.24653v1
- **核心创新**：
  - 2.8T 参数 MoE 模型，104B 激活参数，100 万 token 上下文
  - Kimi Delta 注意力（KDA）：混合线性注意力机制
  - 注意力残差（AttnRes）：跨层信息流
  - 稳定潜在混合专家：896 路由专家 + 分位数均衡
  - MXFP4/MXFP8 量化感知训练
  - 百万 token 智能体强化学习
- **技术价值**：S 级（全球首个开放 3T 级模型，开源前沿重大突破）

### 2. 在 NVIDIA GB300 NVL72 上创造 MoE 预训练世界记录
- **原标题**：Setting a World Record for MoE Pre-Training on NVIDIA GB300 NVL72
- **作者**：Kirthi Devleker
- **日期**：2026-07-21
- **类型**：官方技术博客
- **来源**：NVIDIA Technical Blog
- **译文**：[works/nvidia-gb300-moe-training-world-record.md](../works/nvidia-gb300-moe-training-world-record.md)
- **原文**：https://developer.nvidia.com/blog/setting-a-world-record-for-moe-pre-training-on-nvidia-gb300-nvl72/
- **核心创新**：
  - 1,648 TFLOPs/GPU 训练性能世界记录（DeepSeek-V3 671B）
  - 3× 性能提升（GB300 vs GB200）
  - 第五代 NVLink：1.8 TB/s 单 GPU 带宽，130 TB/s 全对全
  - 算法-系统协同设计
  - 6 个月内软件优化 1.5× 性能提升
  - 框架优化：Megatron Core、TorchTitan（6×）、JAX（10×）
- **技术价值**：A 级（MoE 训练基础设施新标杆）

### 3. 大规模隐藏解码：大语言模型的潜在计算扩展
- **原标题**：Hidden Decoding at Scale: Latent Computation Scaling for Large Language Models
- **作者**：WeChat AI Team（微信 AI 团队）
- **日期**：2026-07-09
- **类型**：学术论文
- **来源**：arXiv:2607.08186
- **译文**：[works/hidden-decoding-at-scale.md](../works/hidden-decoding-at-scale.md)
- **原文**：https://arxiv.org/abs/2607.08186
- **核心创新**：
  - Hidden Decoding：序列长度扩展方法（固定 Transformer 骨干）
  - Stream-Factorized Attention：注意力成本从 O(n²) 降至 O(n)
  - 首个在 100B+ MoE 规模展示的序列扩展方法
  - 在 WeLM-HD4-80B 和 WeLM-HD4-617B 上验证
  - 扩展因子 n=1→2→4→8，增益单调增长
  - 与流水线并行兼容，解决循环模型扩展难题
- **技术价值**：A 级（新的模型扩展范式）

### 4. OlmoEarth 平台：行星级地理空间推理
- **原标题**：The OlmoEarth Platform: Geospatial inference at planetary scale
- **作者**：Allen Institute for AI (Ai2)
- **日期**：2026-07-28
- **类型**：工程技术博客
- **来源**：Allen Institute for AI Blog
- **译文**：[works/olmoearth-platform-geospatial-inference.md](../works/olmoearth-platform-geospatial-inference.md)
- **原文**：https://allenai.org/blog/olmoearth-infrastructure
- **核心创新**：
  - OlmoEarth Run：行星级地理空间推理执行层
  - 三阶段流水线：CPU 预处理 → GPU 推理 → CPU 后处理
  - 分区并行：数千实例并行，155× 加速（4,737 小时 → 30.5 小时）
  - 自动故障恢复：检查点 + 重试 + 幂等性，99% 完成率
  - 成本效率：每平方公里不到一美分，大陆级推理约 1 天
  - 多云支持：Google Cloud + 可扩展到其他云环境
- **技术价值**：A 级（大规模地理空间推理基础设施）

### 5. CoSA：通过代理-内核协同设计的稀疏注意力加速长上下文推理
- **原标题**：CoSA: Accelerating Long-Context Inference via Proxy-Kernel Co-Designed Sparse Attention
- **作者**：Yufei Xue, Lin Niu, Hong Liu 等（腾讯）
- **日期**：2026-07-28
- **类型**：学术论文
- **来源**：arXiv:2607.25291
- **译文**：[works/cosa-sparse-attention-translation.md](../works/cosa-sparse-attention-translation.md)
- **原文**：https://arxiv.org/abs/2607.25291
- **核心创新**：
  - 代理-内核协同设计的两阶段稀疏注意力
  - 内核感知代理（KAP）：利用 OSM 行最大值标志重排序键块
  - 有序跳过内核（OSK）：动态跳过低贡献块
  - 性能突破：128K 上下文下 4.93× 注意力加速，2.53× TTFT 减少
  - 在 RULER 和 LongBench-v2 上以最低预算达到最高准确率
- **技术价值**：S 级（长上下文推理优化的重大突破，系统-算法协同设计范式）

### 6. ForgetBench：大语言模型长期参数记忆遗忘动态基准
- **原标题**：ForgetBench: Benchmarking Forgetting Dynamics of Long-Term Parametric Memory in Language Models
- **作者**：Ruxi Gu, Zhenliang Zhang, Wei Wang
- **日期**：2026-07-29
- **类型**：学术论文
- **来源**：arXiv:2607.26455
- **译文**：[works/forgetbench-translation.md](../works/forgetbench-translation.md)
- **原文**：https://arxiv.org/abs/2607.26455
- **核心创新**：
  - ForgetBench 基准：首个系统性评估 LLM 持续知识编辑中遗忘行为的框架
  - 双范式评测：concept-based QA（孤立事实）+ scenario-based QA（关系知识）
  - 时序编辑框架：使用时序有序的知识流进行顺序编辑
  - 统一评估指标：temporal decay（时间衰减）、retention strength（保留强度）、cross-instance stability（跨实例稳定性）
  - 实证发现：现有知识编辑方法难以平衡长期保留与泛化质量
- **技术价值**：A 级（知识编辑领域新评测基准，填补持续学习评估空白）

### 7. 语言模型中的全局工作空间
- **原标题**：A global workspace in language models
- **作者**：Anthropic AI 团队
- **日期**：2026 年
- **类型**：可解释性研究报告
- **来源**：Anthropic 官方研究博客
- **译文**：[works/global-workspace-translation.md](../works/global-workspace-translation.md)
- **原文**：https://www.anthropic.com/research/global-workspace
- **核心创新**：
  - 发现 J-空间（雅可比空间）：Claude 的内部"思维工作空间"
  - 雅可比透镜（Jacobian lens）技术：揭示模型未输出的内部思维
  - 全局工作空间理论在 LLM 的验证：5 大功能特性（可报告、可控制、用于推理、灵活复用、区别自动处理）
  - 实际监控应用：检测测试意识、数据造假、恶意目标
  - 意识问题讨论：通达意识 vs 现象意识，与神经科学的双向启发
- **技术价值**：S 级（可解释性研究重大突破，AI 对齐与安全的实用工具）

### 8. SVR：通过联合判定-置信度强化学习实现自适应测试时计算的自我验证精炼
- **原标题**：SVR: Self-Verifying Refinement via Joint Verdict-Confidence Reinforcement Learning for Adaptive Test-Time Compute
- **作者**：Hongyu Chen, Liang Lin, Guangrun Wang（中山大学）
- **日期**：2026-07-30
- **类型**：学术论文
- **来源**：arXiv:2607.28457
- **译文**：[works/svr-self-verifying-refinement-translation.md](../works/svr-self-verifying-refinement-translation.md)
- **原文**：https://arxiv.org/abs/2607.28457
- **核心创新**：
  - 自我验证精炼（SVR）：无预言机的多轮强化学习框架
  - 联合判定-置信度输出：每轮生成（解决方案 + Correct/Incorrect/Unsure 判定 + 0-1 置信度）
  - 自适应停止规则：仅当判定为 Correct 且置信度 ≥ 阈值时保留答案
  - 基于 GRPO 的固定视界训练：整合求解、自我验证和格式奖励
  - Qwen3.5-2B 在 7 个数学推理基准上达到 0.563 准确率，平均 2.99 轮推理
  - 超越固定预算和预言机引导基线，token 消耗仅为十样本投票的 51%
- **技术价值**：S 级（测试时计算自适应分配的重大突破，无需外部验证器）

### 9. LATCH：扩散语言模型的候选感知解码
- **原标题**：Where and When to Commit: Candidate-Aware Decoding for Diffusion Language Models
- **作者**：Chia-Ming Lee, Ming-Ching Chang, Xin Li, Yu-Lun Liu, Chih-Chung Hsu
- **日期**：2026-07-30
- **类型**：学术论文
- **来源**：arXiv:2607.28166
- **译文**：[works/latch-diffusion-acceleration-translation.md](../works/latch-diffusion-acceleration-translation.md)
- **原文**：https://arxiv.org/abs/2607.28166
- **核心创新**：
  - 双轴解耦设计：CVC（置信度验证提交）控制全局终止，BWEC（分块提前提交）控制局部加速
  - 候选感知提前退出：动态提取候选答案，验证置信度 + argmax 稳定性
  - 任务相关稳定化时机：短答案 s_0.9=0.04 vs 长推理 s_0.9=0.96
  - 单套超参数跨任务/模型通用：无需针对任务或模型调优
  - 短答案任务：9.3-17.8× TPS 加速，长推理任务：2.0-3.3× TPS 加速
  - 在 11 任务、2 模型（LLaDA/Dream）上保持准确率 ±2.0 点以内
- **技术价值**：S 级（扩散语言模型推理加速重大突破，训练无关优化新范式）

### 10. 从专家归约到行为分歧：追踪稀疏 MoE 推理中的数值状态
- **原标题**：From Expert Reduction to Behavioral Divergence: Tracing Numerical State through Sparse MoE Inference
- **作者**：Tianyang Zhu（独立研究者）
- **日期**：2026-07-30
- **类型**：学术论文
- **来源**：arXiv:2607.28097
- **译文**：[works/sparse-moe-numerical-state-translation.md](../works/sparse-moe-numerical-state-translation.md)
- **原文**：https://arxiv.org/abs/2607.28097
- **核心创新**：
  - 揭示数学上等价的专家归约顺序在有限精度下产生可观测的不同执行结果
  - 四种聚合方案（P32/C/A/B）：分离操作数表示与累加器精度的影响
  - 事件方向分化：中文提示"朋友昨天打来电话"产生 202 个裁员 vs. 113 个招聘延续
  - 状态边界验证：mHC 后状态（token 内边界）和完整持久状态（跨 token 边界）的充分性
  - 受控实验：精确端点重建在 DeepSeek-V4-Flash 上复现分支轨迹
  - 确定性重放：10 个 64-token 分支完全可重现，证明分歧非随机性
- **技术价值**：S 级（MoE 数值稳定性的重大发现，为运行时和硬件设计提供数值兼容性契约）

### 11. 虚拟宽度网络
- **原标题**：Virtual Width Networks
- **作者**：ByteDance Seed Team
- **日期**：2025-11-17
- **类型**：学术论文
- **来源**：arXiv:2511.11238
- **译文**：[works/virtual-width-networks-translation.md](../works/virtual-width-networks-translation.md)
- **原文**：https://arxiv.org/abs/2511.11238
- **核心创新**：
  - 虚拟宽度网络（VWN）：解耦嵌入宽度与骨干网络宽度，扩展表征能力而计算成本几乎不变
  - 广义超连接（GHC）：统一 Hyper-Connections 和 Frac-Connections 的轻量级连接机制
  - 对数线性缩放关系：虚拟宽度因子与损失之间的缩放定律
  - 大规模验证：3.3B MoE 模型，8× 虚拟宽度扩展，token 效率提升 2-3 倍
  - 与多 token 预测（MTP）协同：虚拟宽度与密集监督的互补增益
- **技术价值**：S 级（突破宽度-计算二次方耦合，提出模型扩展新维度，理论 + 大规模验证）

### 12. GFlowRL：将分布匹配强化学习扩展到大语言模型
- **原标题**：GFlowRL: Scaling Distribution-Matching RL to Large Language Models
- **作者**：Xiaodong Liu, Michael Xu, Jack W. Stokes, Paul Smolensky, Doug Burger, Jianfeng Gao
- **日期**：2026-07-27
- **类型**：学术论文
- **来源**：arXiv:2607.13394v1
- **译文**：[works/gflowrl-scaling-distribution-matching-llm-post-training.md](../works/gflowrl-scaling-distribution-matching-llm-post-training.md)
- **原文**：https://arxiv.org/abs/2607.13394
- **核心创新**：
  - 用批内蒙特卡洛估计替代可学习配分函数，移除辅助网络
  - 梯度范数从 10^14 降至 10^-2，恢复训练稳定性
  - 重要性采样校正 + 非对称流间隙裁剪两大稳定器
  - 14B 模型 Codeforces 2048 评分（超越 o1 +157 Elo，接近 o3-mini）
  - AdvBench/HarmBench 红队测试 SOTA（82.5%/79.5% ASR@1）
  - 首次在稠密+稀疏架构（最大 235B MoE）稳定扩展 GFlowNets
- **技术价值**：A 级（LLM 后训练核心算法突破，分布匹配 RL 工程化里程碑）

### 13. AURORA-LM：面向连续潜在扩散语言建模的自编码统一表示
- **原标题**：AURORA-LM: Autoencoding Unified Representation for Continuous-Latent Diffusion Language Modeling
- **作者**：Jiajun Liang, Yucheng Liao, Yukang Cao, Jiazhe Wei, Ken Li, Wende Tan, Jiankun Zhang, ZY Cui, Jingkang Yang, Liucheng Guo, Shiqi Yang, B. Yang, Caifeng Shan, Ziwei Liu, Chenyang Si
- **日期**：2026-08-03
- **类型**：学术论文
- **来源**：arXiv:2608.02602v1
- **译文**：[works/aurora-lm-continuous-latent-diffusion-translation.md](../works/aurora-lm-continuous-latent-diffusion-translation.md)
- **原文**：https://arxiv.org/abs/2608.02602
- **核心创新**：
  - 解耦设计：将可解码文本表示构建与生成分布建模分离
  - Query-based Encoder-Decoder：构建高容量、前缀对齐的潜在序列
  - Block-causal Diffusion Transformer：通过流匹配学习全宽度潜在分布
  - 输入瓶颈策略：仅对噪声输入路径应用低秩投影（Db=128），保留全宽度清洁潜在预测目标（D=1024）
  - Self-trajectory Consistency：对齐去噪轨迹上相邻状态的预测
  - 噪声级别校准：tan-d 调度将噪声分配调整到潜在宽度
  - 1B 参数模型超越 1.8B Cola-DLM，在昇腾 NPU 上完成所有实验
- **技术价值**：S 级（连续语言生成范式突破，解耦表示学习与分布建模的方法论创新）

### 14. MA-LoT：基于多智能体 Lean 的长链式思考推理增强形式化定理证明
- **原标题**：MA-LoT: Multi-Agent Lean-based Long Chain-of-Thought Reasoning enhances Formal Theorem Proving
- **作者**：Ruida Wang, Rui Pan, Yuxin Li, Jipeng Zhang, Yizhen Jia, Shizhe Diao, Renjie Pi, Junjie Hu, Tong Zhang
- **日期**：2025-03
- **类型**：学术论文
- **来源**：arXiv:2503.03205
- **译文**：[works/ma-lot-multi-agent-lean-long-cot-theorem-proving-translation.md](../works/ma-lot-multi-agent-lean-long-cot-theorem-proving-translation.md)
- **原文**：https://ar5iv.labs.arxiv.org/html/2503.03205
- **核心创新**：
  - 首个多智能体 Lean4 定理证明框架：Prover Agent（完整证明生成）+ Corrector Agent（错误分析与修正）
  - LoT-Transfer Learning（LoT-TL）训练流程：使形式化推理能力在 Long CoT 中涌现，无需专门标注数据
  - 三阶段训练：NL Long CoT（126K 数据）→ Lean SFT（54K 数据）→ Correction（64K 数据）
  - 系统提示控制 Long CoT 开关：训练时 WITHOUT（占位符）、推理时 WITH（激活能力）
  - MiniF2F-Test 达到 61.07% 准确率（GPT-4: 22.95%, 树搜索: 50.70%, 完整证明: 55.33%）
  - 成功证明 IMO/AIME 级别问题，消融实验验证各组件有效性
- **技术价值**：S 级（形式化验证与 Long CoT 结合的首个框架，多智能体协作新范式）

### 15. 轻量语言模型的检索增强推理
- **原标题**：Retrieval-Augmented Reasoning with Lean Language Models
- **作者**：Ryan Sze-Yin Chan, Federico Nanni, Tomas Lazauskas, Rosie Wood, Penelope Yong, Lionel Tarassenko, Mark Girolami, James Geddes, Andrew Duncan
- **日期**：2025-08-15
- **类型**：学术论文（技术报告）
- **来源**：arXiv:2508.11386
- **译文**：[works/lean-language-model-rag-reasoning-translation.md](../works/lean-language-model-rag-reasoning-translation.md)
- **原文**：https://arxiv.org/abs/2508.11386
- **核心创新**：
  - 在单个轻量模型架构（1.5B-32B）中整合推理与检索增强生成（RAG）
  - 密集检索器 + 微调 Qwen2.5-Instruct：利用 DeepSeek-R1 推理轨迹进行领域特定微调
  - 文档摘要压缩：减少 85% 文档长度，平均上下文从 74,641 tokens 降至 7,544 tokens
  - NHS 医疗知识库案例：990 条目、1000 评估样本、2000 训练样本
  - 合成数据生成 + 推理轨迹蒸馏 + 预算强制（budget-forcing）测试时计算控制
  - 32B 模型病症识别准确率 56%，接近前沿模型（GPT-4o/o3-mini/DeepSeek-R1）
  - 1.5B 模型病症识别 53%，超越 32B 非推理基线（54%）
  - 完整开源实现 + 3700 GPU 小时计算资源详情（Azure/Baskerville/Isambard-AI）
- **技术价值**：A 级（轻量模型 RAG+推理整合工程化方案，隐私保护场景实用突破）

---

## 🔍 观察项 / 候选材料（不计入 15 篇）

| 候选 | 类型 | 去向 | 角度 / 为何只做观察项 | 原文 |
|---|---|---|---|---|
| LLaDA 2.2：全球首个 Agentic 扩散模型 | 科技报道 | 观察项 | 蚂蚁 inclusionAI 团队。扩散模型在 LLM Agent 任务的突破：Levenshtein 编辑范式 + L-EBPO 强化学习 + BlockRouting（128K上下文）。Agent 任务性能接近自回归（差距 <2分），效率提升 1.64×。科技报道非学术论文，但技术信息完整。 | [量子位](https://www.qbitai.com/2026/07/461650.html) · [技术报告](https://github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2_2_tech_report.pdf) · [GitHub](https://github.com/inclusionAI/LLaDA2.X) |
| DualDecoder：通过预测性预取加速长上下文 LLM 推理 | 学术论文摘要 | 观察项 | Zuning Liang 等。稀疏 KV 缓存优化：双 token 解码流水线预测关键 KV 条目，实现预取与计算重叠，消除辅助状态的 GPU 内存开销。2.62× 吞吐量提升。仅摘要页翻译，完整论文待补充。 | [arXiv](https://arxiv.org/abs/2607.26475) · [译文](../translate/20260731-arxiv-2607-26475/translations/dualdecoder/translation.md) |
| OpenAI GPT-5.6 自我进化技术 | 科技报道 | 观察项 | 智东西（2026-07-30）。AI 自主优化 GPU kernel（成本降低 20%）+ speculative decoding（效率提升 >15%）+ Agent 框架优化（lazy loading / prompt caching / append-only）+ ARC-AGI-3 性能 4.9× 提升（7.8%→38.3%）。科技报道非官方技术博客，但包含完整性能数据。 | [36kr](https://eu.36kr.com/en/p/3917509136346498) · [译文](../translate/2026-08-01-batch/works-ready/openai-gpt56-self-evolution-translation.md) |
