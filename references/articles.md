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
> **当前规模：33 篇文章**。最近一次同步：2026-08-21。
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
- **作者**：Allen Institute for AI
- **日期**：2026-07-28
- **类型**：技术博客
- **来源**：Allen Institute for AI Blog
- **译文**：[works/olmoearth-infrastructure-translation.md](../works/olmoearth-infrastructure-translation.md)
- **原文**：https://allenai.org/blog/olmoearth-infrastructure
- **核心创新**：
  - OlmoEarth Run：行星级地理空间推理执行层
  - 三阶段流水线：CPU 预处理 → GPU 推理 → CPU 后处理
  - 大规模并行：19,600 CPUs + 994 GPUs，155× 加速（4,737h → 30.5h）
  - 容错设计：幂等任务、自动重试、多源回退
  - 成本效率：每平方公里不到一美分，大陆级推理约一天
  - 元数据索引：避免压垮外部 STAC API，窗口化读取云优化格式
- **技术价值**：A 级（地球观测基础模型工业级部署架构）

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

### 16. LLaDA MoE v2：扩展混合专家扩散语言模型
- **原标题**：LLaDA MoE v2: Scaling Mixture-of-Experts Diffusion Language Models
- **作者**：Fengqi Zhu, Shaoxuan Xu, Jingyang Ou, Zebin You, Yipeng Xing, Huabin Liu, Xiaolu Zhang, Jun Zhou, Zhenzhong Lan, Yankai Lin, Wayne Xin Zhao, Jianguo Li, Chongxuan Li, Ji-Rong Wen（人大高瓴 AI 学院、蚂蚁集团）
- **日期**：2026-08-04
- **类型**：学术论文
- **来源**：arXiv:2608.03457
- **译文**：[works/llada-moe-v2-scaling-diffusion-language-models-translation.md](../works/llada-moe-v2-scaling-diffusion-language-models-translation.md)
- **原文**：https://arxiv.org/abs/2608.03457
- **核心创新**：
  - 首次系统表征 MoE 扩散语言模型（dLLMs）的缩放行为，识别与 AR 模型的量化差异
  - 优化缩放：最优 batch size 增长指数 0.3481（比 AR 更陡），学习率衰减指数 -0.2447（比 AR 更快）
  - 计算分配：IsoFLOP 分析揭示略偏数据侧的配置（M*∝C^0.475, D*∝C^0.525）
  - MoE 架构缩放：更大规模偏好更低激活比例，适度专家粒度（G=8–16）稳健，共享专家比例（S=33.3%）跨规模稳定
  - 大规模验证：30B-A3B 模型在 23.5T tokens（Qwen3 的 65%）上训练，在知识/推理/编码基准上接近 Qwen3
  - 仅经 SFT 在 8 个推理/编码任务中的 7 个上超越 SDAR Chat，无需 RL 阶段
  - 受控实验：10^17–10^20 FLOPs 多档位扫描，三维架构扫描（激活比例/专家粒度/共享专家比例）
- **技术价值**：S 级（首次系统化 MoE dLLM 缩放定律，填补该领域空白，理论+大规模验证完整）

### 17. 全带宽 Transformer
- **原标题**：Full-bandwidth transformer
- **作者**：Xi Wang, Ziyang Cai, Zheng Zhan, Harry Dong, Ying Fan, Gustavo de Rosa, Tim Pearce, John Langford（Johns Hopkins University, Princeton University, Microsoft）
- **日期**：2026-08-09
- **类型**：学术论文
- **来源**：arXiv:2608.08888
- **译文**：[works/arxiv-2608-08888-translation.md](../works/arxiv-2608-08888-translation.md)
- **原文**：https://arxiv.org/abs/2608.08888
- **核心创新**：
  - 识别自回归 Transformer 的"窄垂直通道"限制：步间仅传递采样 token（log₂|V| bits），顶层隐藏状态（D 维）被丢弃
  - 潜在反馈解码：通过门控线性单元（GLU）将前一步顶层隐藏状态与当前 token 嵌入融合，拓宽反馈通道至完整 D 维
  - 时间并行性训练：多遍前向传播（temporal parallelism）解决递归结构的并行训练难题，保留教师强制效率
  - 渐进调度 + 收缩映射：晚期引入反馈（75% 单遍 + 22% 双遍 + 3% 三遍），3% 三遍批次使反馈映射成为收缩映射，在 1000 遍外推时保持稳定
  - 1B 参数模型在 400B tokens 上训练，数据效率提升 1.5×-5×（匹配使用更多数据的标准 Transformer）
  - 推理开销可忽略（< 1% per token），与 vLLM 等现有基础设施兼容
  - 基础模型生成更短推理轨迹且准确率相当或更好（GSM8K、Math500、HumanEval、MBPP）
  - 改进通过指令微调延续，融合预填充（Fused）改善非生成任务性能
- **技术价值**：S 级（Transformer 架构核心创新，首次系统解决步间反馈带宽问题，理论+训练+验证完整）

### 18. Transformer 是贝叶斯网络
- **原标题**：Transformers are Bayesian Networks
- **作者**：Greg Coppola
- **日期**：2026-03-17
- **类型**：学术论文
- **来源**：arXiv:2603.17063
- **译文**：[works/arxiv-2603-17063-translation.md](../works/arxiv-2603-17063-translation.md)
- **原文**：https://arxiv.org/abs/2603.17063
- **核心创新**：
  - 通过 Lean 形式化验证证明：任何权重的 sigmoid transformer 都在其隐式因子图上实现加权循环信念传播（loopy BP）
  - 构造性证明：显式权重矩阵可在任何声明的因子图上实现精确信念传播，树结构知识库上零幻觉
  - 唯一性定理：产生精确贝叶斯后验的 sigmoid transformer 必然具有 BP 权重，无其他路径
  - 识别 AND/OR 布尔结构：注意力=AND（合取）、FFN=OR（析取），严格交替即 Pearl 的 gather/update 算法
  - 有限概念空间定理：证明可验证推理需要有限验证器，幻觉是无落地概念空间的结构性后果
  - 所有定理在 Lean 4 中形式化验证 + 实验确认（后验概率匹配到三位小数）
- **技术价值**：S 级（Transformer 理论基础重大突破，形式化证明 + 实验双重验证，解答"为什么有效"的根本问题）

### 19. 超循环 Transformer
- **原标题**：Hyperloop Transformers
- **作者**：Abbas Zeitoun, Lucas Torroba-Hennigen, Yoon Kim
- **日期**：2026-04-23
- **类型**：学术论文
- **来源**：arXiv:2604.21254
- **译文**：[works/arxiv-2604-21254-translation.md](../works/arxiv-2604-21254-translation.md)
- **原文**：https://arxiv.org/abs/2604.21254
- **核心创新**：
  - 将循环 Transformer 与超连接结合，在循环级别（而非层级别）应用超连接，实现 50% 参数减少的同时保持性能
  - 三块架构设计：起始块 → 中间块（循环 3 次）→ 结束块，仅中间块参数共享，起始和结束块保持独立
  - 简化的超连接实现：用对角转移矩阵替代 Sinkhorn 归一化的双随机矩阵，降低计算复杂度
  - 循环位置嵌入：为每次循环迭代添加位置信息，增强模型表达能力
  - 在 240M、1B、2B 三个规模上验证，循环版本使用约 50% 参数即可达到深度匹配 Transformer 的性能
  - INT4 量化后性能保持良好，训练吞吐量仅轻微下降（750K vs 786K tokens/s）
  - 6 组消融实验覆盖循环次数、并行流数量、超连接数量、转移矩阵参数化等关键设计选择
- **技术价值**：A 级（参数高效架构创新，实验严谨全面，适用边缘部署场景，工程实用价值高）

### 20. 可变宽度 Transformer
- **原标题**：Variable-Width Transformers
- **作者**：Zhaofeng Wu, Oliver Sieberling, Shawn Tan, Rameswar Panda, Yury Polyanskiy, Yoon Kim
- **日期**：2026-06-16
- **类型**：学术论文
- **来源**：arXiv:2606.18246
- **译文**：[works/arxiv-2606-18246-translation.md](../works/arxiv-2606-18246-translation.md)
- **原文**：https://arxiv.org/abs/2606.18246
- **核心创新**：
  - 提出 ><former（蝴蝶结形 Transformer）架构，打破传统 Transformer 所有层宽度恒定的假设
  - ×形宽度分配：早期和晚期层较宽，中间层较窄，通过瓶颈结构实现非均匀容量分配
  - 无参数残差调整机制：固定全局残差维度，各层读写残差流的特定切片，未使用维度通过复制前向传递
  - 数学证明参数匹配时必然减少平均层宽度和注意力 FLOPs
  - 在 200M-2B 稠密模型和 3B MoE 模型上验证，困惑度提升约 3%，FLOPs 减少 22%，KV 缓存减少 15%
  - 深入分析表明 ><former 缓解中间层表示坍缩，更均匀地利用 MLP 激活维度和表示空间
  - 瓶颈超参数跨规模迁移：ℓ* = 0.75L，d_ℓ* = 0.3d
- **技术价值**：A 级（架构设计新范式，理论证明与实验验证结合紧密，为非均匀容量分配提供系统化研究）

### 21. SlimQwen：大规模 MoE 模型预训练中的剪枝与蒸馏探索
- **原标题**：SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training
- **作者**：Shengkun Tang, Zekun Wang, Bo Zheng, Liangyu Wang, Rui Men, Siqi Zhang, Xiulong Yuan, Zihan Qiu, Zhiqiang Shen, Dayiheng Liu
- **日期**：2026-05-09（v1）、2026-05-18（v2）
- **类型**：学术论文
- **来源**：arXiv:2605.08738
- **译文**：[works/arxiv-2605-08738-translation.md](../works/arxiv-2605-08738-translation.md)
- **原文**：https://arxiv.org/abs/2605.08738
- **核心创新**：
  - 系统性研究大规模预训练中的 MoE 压缩，覆盖深度、宽度、专家三个维度
  - 证明剪枝初始化在相同训练预算下始终优于从零训练（平均提升 11.79 分）
  - 部分保留专家合并策略：保留一半目标专家不变，合并另一半，防止表示同质化
  - 多 token 预测（MTP）蒸馏：扩展蒸馏目标到多个未来 token，提升主干训练质量和推测解码效率
  - 渐进式剪枝调度：深度优先/宽度优先/联合策略均优于一次性压缩，提供更平滑的优化轨迹
  - 实证压缩 Qwen3-Next-80A3B 至 23A2B（4× 压缩），400B token 持续预训练后保持竞争力性能
  - NTP KD + LM 损失混合优于纯 KD，特别在知识密集型任务（MMLU 从 74.16→74.93）
  - MTP KD 使推测解码接受率提升显著（acc_4 从 4.09%→8.24%）
- **技术价值**：S 级（MoE 预训练规模压缩的首个系统性研究，方法创新+大规模实验验证，填补该领域研究空白，对工业界有重要参考价值）

### 22. 超越 RLHF：对齐的统一理论框架
- **原标题**：Beyond RLHF: A Unified Theoretical Framework of Alignment
- **作者**：Jihun Yun, Juno Kim, Jongho Park, Junhyuck Kim, Jongha Jon Ryu, Jaewoong Cho, Kwang-Sung Jun
- **日期**：2025-06-02（v1）、2026-05-18（v2）
- **类型**：学术论文
- **来源**：arXiv:2506.01523
- **译文**：[works/arxiv-2506-01523-translation.md](../works/arxiv-2506-01523-translation.md)
- **原文**：https://arxiv.org/abs/2506.01523
- **核心创新**：
  - 提出对齐的统一理论框架，将对齐重新定义为从配对偏好中进行**分布学习**而非奖励最大化
  - 基于 Bradley-Terry 模型，假设偏好直接依赖于目标语言模型 π*：ℙ(a≻b|x) = π*(a|x)^γ / (π*(a|x)^γ + π*(b|x)^γ)
  - 推导出三种有原则的对齐目标函数：
    1. **PMLE**（偏好最大似然估计）：类似 DPO，但有显式 KL 正则化，避免退化
    2. **偏好蒸馏**：类似 REBEL，但从 Bradley-Terry 模型明确推导，使用期望偏好而非平方损失
    3. **反向 KL**（RKL）：RLHF 的修正版本，带额外熵项，γ→0 时退化为 RLHF
  - 证明三种方法均享有强非渐近 O(1/n) 收敛保证（首个从配对反馈学习分布的此类保证）
  - 理论解释经验发现：RKL（类 RLHF）比 PMLE（类 DPO）有更优保证，依赖 C_ℛ 而非 C_Π
  - 识别 RLHF 的渐近困境：固定 β 导致欠拟合，β→0 导致退化；RKL 避免此问题
  - 实验验证：TL;DR 摘要任务胜率优于基线，通用对话生成更受偏好的响应
- **技术价值**：S 级（为 RLHF 提供严格理论正当性，统一框架允许不同方法理论比较，首次从理论上确认 RLHF 优于 DPO 的经验发现，填补对齐领域理论基础空白）

### 23. VecInfer：通过离群值抑制向量量化实现高效低比特 KV 缓存的 LLM 推理
- **原标题**：VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization
- **作者**：Dingyu Yao, Chenxu Yang, Zhengyang Tong, Zheng Lin, Wei Liu, Jian Luan, Weiping Wang
- **日期**：2026-07（ACL 2026）
- **类型**：学术论文
- **来源**：ACL 2026 (Volume 1: Long Papers), pages 31527–31543
- **译文**：[works/acl-2026-1454-translation.md](../works/acl-2026-1454-translation.md)
- **原文**：https://aclanthology.org/2026.acl-long.1454/
- **代码**：https://github.com/ydyhello/VecInfer
- **核心创新**：
  - 双重变换抑制离群值：平滑变换 + 哈达玛变换，降低键缓存量化难度
  - SVD 分析证明变换有效性：减小通道级方差，产生无离群值的均匀分布
  - 融合 CUDA 内核：反量化-计算融合，细粒度分块 + 异步流水线，最小化内存访问
  - 2-bit 量化达 FP16 精度，Llama-3.1-8B (196k) 上实现 2.7× 自注意力加速、8.3× 端到端延迟降低
  - LongBench 和 MATH 任务验证，1.25/1.5/2/3/4-bit 全面优于 KIVI/ZipCache/CQ/MILLION
  - 任务无关码本：双重变换后的均匀分布使码本全面覆盖数据空间
- **技术价值**：S 级（KV 缓存向量量化的重大突破，解决超低比特量化的核心难题，系统实现完整且开源，填补仓库 KV 缓存量化方向空白）

### 24. ZeroLock：基于模块化更新解耦的并发内存高效 LLM 训练
- **原标题**：ZeroLock: Concurrent Memory-Efficient LLM Training via Modular Update Decoupling
- **作者**：Wentao Dai, Xuanran Li, Yuxiang Zhang, Ming Tang, Chao Huang
- **日期**：2026-08-08
- **类型**：学术论文
- **来源**：arXiv:2608.07974
- **译文**：[works/arxiv-2608-07974-translation.md](../works/arxiv-2608-07974-translation.md)
- **原文**：https://arxiv.org/abs/2608.07974
- **代码**：https://anonymous.4open.science/r/unlock_trainer-105B
- **核心创新**：
  - ZeroLock 算法：通过局部目标构建打破 BP 更新锁定，模块化解耦实现并发更新
  - 首个通用理论框架：Bregman 几何下的收敛性分析，证明收敛率 Õ(1/√T)，与 BP 仅差多重对数因子
  - 系统实现：多 GPU 服务器 + Android 原型，早转发与故障恢复技术
  - 性能提升：相比 BP 基线，内存减少 26.5%，吞吐量提高 4.9%
  - 边缘部署：TinyLlama Android 微调峰值 PSS < 4000 MiB，首个移动端 BP-free 训练原型
  - 理论贡献：建立局部-全局目标映射，为解耦更新的收敛性分析提供系统方法
- **技术价值**：S 级（边缘设备 LLM 微调完整解决方案，填补 BP-free 训练和移动端部署空白，算法-理论-系统三位一体）

### 25. Intern-S2-Preview：科学智能体基础模型
- **原标题**：Intern-S2-Preview: Scientific Agentic Foundation Model
- **作者**：Intern-S2-Preview Team, Shanghai AI Laboratory
- **日期**：2026-08
- **类型**：技术报告
- **来源**：arXiv:2608.13505
- **译文**：[works/arxiv-2608-13505-translation.md](../works/arxiv-2608-13505-translation.md)
- **原文**：https://arxiv.org/abs/2608.13505
- **核心创新**：
  - Memory Decoder：参数化记忆解码器，冻结 397B 主干 + 可插拔领域记忆，实现模块化专精
  - 时间序列模块升级：支持 30 万时间步，5-6× 推理加速，扩展到数值预测（MHz 级雷达信号）
  - 可扩展 RL：部分 rollout + 离策略校正、自适应长度正则化、在线推测解码（2× 加速）、GEPO（群组级熵控制）
  - 智能体 RL 框架：harness × task 抽象，统一黑白盒智能体，21 万+编程/终端任务 + 自演化任务合成
  - 预训练技术：视觉预训练（VP）、交错 PDF 数据构建（视觉增益过滤）、大规模图像检索
  - 系统工程：XTuner + LMDeploy 共置、TITO（token 级输入输出）、PrefixTree 轨迹存储、R3 路由重放
- **技术价值**：S 级（首个系统性整合科学多模态理解、长时域智能体、可扩展 RL 的工作，工程实践深度极高，填补智能体 RL 工程实践空白）

### 26. LLM 预预训练的不稳定性：并非总是有效——多语言调查研究
- **原标题**：Instability of LLM Pre-Pretraining: It Doesn't Always Help. An Investigation on Multiple Languages
- **作者**：Sofiia Riazhskykh, Nam Luu, Ondřej Bojar（查理大学）
- **日期**：2026-08-09
- **类型**：学术论文
- **来源**：arXiv:2608.08800
- **译文**：[works/arxiv-2608-08800-translation.md](../works/arxiv-2608-08800-translation.md)
- **原文**：https://arxiv.org/abs/2608.08800
- **核心创新**：
  - 系统验证预预训练（pre-pretraining）的有效性：在人工语言（Dyck 语言）上预训练后再进行自然语言训练
  - 多语言、多设置实验：覆盖 6 种语言（英语、阿尔巴尼亚语、捷克语、丹麦语、荷兰语、芬兰语）、3 种模型规模（154M-481M）、2 种分词器
  - 揭示方法不稳定性：同一设置下不同随机种子导致显著差异，增益高度依赖实验配置
  - 确认部分有效性：154M+Llama 分词器在 128-Dyck 上预训练表现出稳定增益趋势
  - 语言学分析：评估标记效率增益与语言特征（形态丰富度、句法树深度、交叉依赖）的相关性
  - 方法论贡献：强调多次训练运行的重要性，避免社区采纳不稳定的优化方法
- **技术价值**：A 级（对热门预训练优化方法的批判性验证，强调实验鲁棒性，具有重要方法论意义）

### 27. 多智能体 AI 工作流中的 Token 优化与上下文窗口管理
- **原标题**：Token Optimization and Context Window Management in Multi-Agent AI Workflows
- **作者**：Dvir Shamay
- **日期**：2026-08-17
- **类型**：学术论文（独立研究）
- **来源**：arXiv:2608.17188
- **译文**：[works/arxiv-2608-17188-translation.md](../works/arxiv-2608-17188-translation.md)
- **原文**：https://arxiv.org/abs/2608.17188
- **核心创新**：
  - 六种 token 优化模式：上下文分层、一次抓取/本地处理、模式约束提示、token 感知降级链、语义缓存、智能体间通信压缩
  - 生产环境实测：冷启动延迟从 3.5-10.5 分钟降至 61-116 秒，token 使用量减少 60-70%
  - 反直觉发现：相关性对比上下文（relevance-contrast context）——混入同领域低相关性内容提高模型对目标项的相关性判断准确率
  - 对照实验验证：2,420 次试验跨 11 个模型配置，50:50 信噪比条件下准确率提高 +0.077（Cohen's d = 0.49）
  - 将上下文窗口管理提升为工程学科：六大核心技能（剪枝、信息密度、内存卸载、并行发现、污染检测、校准）
  - Fusion-of-N 研究：简单的集合并集优于学习型融合，提供面向召回的提取部署规则
- **技术价值**：S 级（多智能体工作流优化重大突破，系统化方法论 + 严格实验验证 + 工程实践价值）

### 28. MoNe：面向高效长上下文推理的模块化神经记忆
- **原标题**：MoNe: Modular Neural Memory for Efficient Long Context Inference
- **作者**：Wonguk Cho, Kyubyung Chae, Tribhuvanesh Orekondy, Sunghyun Park, Hyoungwoo Park, Jeongho Kim, Arash Behboodi, Kyuwoong Hwang, Sungrack Yun（高通AI研究院）
- **日期**：2026-08-18
- **类型**：学术论文
- **来源**：arXiv:2608.17616
- **译文**：[works/arxiv-2608-17616-translation.md](../works/arxiv-2608-17616-translation.md)
- **原文**：https://arxiv.org/abs/2608.17616
- **核心创新**：
  - 模块化神经记忆架构：可插拔至任何冻结预训练 Transformer，无需重训练骨干模型，仅 6.4% 参数开销
  - 测试时学习 + 层局部梯度更新：快权重神经记忆网络通过关联记忆损失在线更新，无需跨层反向传播
  - 两阶段解耦设计：O(N) 预处理顺序处理上下文段 + O(1) 查询仅注意记忆 token，突破传统 O(N²) 注意力瓶颈
  - 段局部 RoPE 编码：位置索引始终在 [0,T) 范围内，实现 32 倍外推泛化（4K 训练 → 128K 推理）无需位置插值
  - 恒定显存推理：峰值 GPU 显存不随上下文长度增长，128K token 时相比 ICL 减少约 80% 计算和显存
  - RULER 基准测试：S-NIAH/MK-NIAH/高频词提取任务近乎完美性能（0.94-1.00），而 ICL 在超出原生窗口后崩溃至 0.00-0.28
- **技术价值**：A 级（长上下文推理优化重大突破，模块化设计 + 测试时学习范式创新，工程实用性强）

### 29. Data-DPO：用于大模型后训练数据选择的直接偏好优化
- **原标题**：Data-DPO: Direct Preference Optimization for Target Model Data Selection in LLM Post-Training
- **作者**：Peng Sun, Yi Yang, Antong Zhang, Chunxiao Li, Yanbo Wang, Dianbo Liu, Xin Chen, Kai Yu, Lu Chen, Tianfan Fu
- **日期**：2026-08-05
- **类型**：学术论文
- **来源**：arXiv:2608.16926
- **译文**：[works/arxiv-2608-16926-translation.md](../works/arxiv-2608-16926-translation.md)
- **原文**：https://arxiv.org/abs/2608.16926
- **核心创新**：
  - 面向目标模型的数据选择新范式：将数据价值从静态属性转变为与目标模型能力分布兼容的动态属性
  - 激活探测与偏好构建：通过单步训练反馈观察目标模型的局部训练反馈，将激活差异转化为成对数据偏好
  - 轻量级奖励模型：训练目标模型感知的奖励模型学习数据偏好，结合偏好、质量和多样性三维信号
  - 序贯选择策略：贪心地逐步构建最终训练子集，平衡目标模型偏好、外部质量评分和边际多样性
  - 稳定超越全量训练：Vision-Flan 数据集上实现 100.76%-102.70% 相对性能，LLaVA-CoT 上达到 102.73%-103.93%
  - 跨模型鲁棒性验证：在不同目标模型、质量评分源和嵌入源上均保持稳定性能
- **技术价值**：S 级（SFT 数据选择方法论创新，将 DPO 迁移到数据选择领域，实验充分且性能稳定）

### 30. 基于图结构在线难度估计的高效 RLVR 调度
- **原标题**：Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation
- **作者**：Zhizhao Liu, Zhiliang Tian, Xi Wang, Zhihua Wen, Yihang Xiong, Zhiquan Lai, Dongsheng Li
- **日期**：2026-08-18
- **类型**：学术论文
- **来源**：arXiv:2608.17941
- **译文**：[works/arxiv-2608-17941-translation.md](../works/arxiv-2608-17941-translation.md)
- **原文**：https://arxiv.org/abs/2608.17941
- **核心创新**：
  - 首次将 RLVR 动态难度估计表述为图结构化潜在变量推断问题
  - 难度感知样本图：基于语义和推理相似性构建稀疏图（k-NN + 互为邻居过滤）
  - 图结构化潜在模型：Potts 先验鼓励相邻样本共享潜在状态，Beta-二项模型聚合 rollout 反馈
  - 在线平均场变分推断：坐标上升更新潜在状态分配与状态级成功概率，持续跟踪策略演化下的动态难度
  - 即插即用组件：可集成到样本选择（PCL、GRESO）和 rollout 分配（GVM）调度器
  - 冷启动友好：通过图传播利用相似样本历史反馈，缓解观测稀疏性
  - 实验验证：多模型（Qwen-2.5-Math-1.5B、Llama-3.2-1B）多数据集（MATH500、AIME24/25、OlympiadBench）性能提升，低成本（~0.12h vs 45h 专用探测）
- **技术价值**：A 级（RLVR 训练调度优化重大突破，方法论完整且具备工程实用性）

### 31. Abra：扩散图像训练的缩放
- **原标题**：Abra: Scaling Diffusion Image Training
- **作者**：Kyle Chickering, Wei-An Lin, Swayam Bhanded, Dan Saunders, Akshat Tripathi, Jiaming Song, Shyamal Buch, Xinchen Yan（Luma AI）
- **日期**：2026-08-18
- **类型**：学术论文
- **来源**：arXiv:2608.17286
- **译文**：[works/arxiv-2608-17286-translation.md](../works/arxiv-2608-17286-translation.md)
- **原文**：https://arxiv.org/abs/2608.17286
- **核心创新**：
  - 首次系统研究文生图扩散模型计算最优缩放定律：ABRA 模型家族（60M-2B 参数），训练跨越 10^19-10^22 FLOPs
  - 200 TPP 规则：扩散模型计算最优点在每参数 200 个图像 token，是 LLM Chinchilla 规则（20 TPP）的 10 倍
  - 过训练稳健性：扩散模型 2× 过训练损失惩罚 < 0.5%，远优于 LLM；实践规则：倾向更多数据而非更大模型
  - 可预测缩放：训练损失、FID/KID/CLIPScore 等生成指标、表示质量（线性探测）均遵循幂律
  - 缩放坍缩验证：首次证明扩散模型展现 scaling collapse 现象，训练曲线重缩放后坍缩到普适形式
  - 分辨率效应：更高分辨率需要更多图像 token 达到计算最优（256px: 165 TPP → 768px: 247 TPP）
  - µP 应用：使用最大更新参数化实现超参数跨模型迁移，零样本迁移学习率 4×10^(-4)
- **技术价值**：S 级（视觉生成缩放定律重大突破，填补扩散模型计算最优研究空白，200 TPP 规则具有重要实践指导价值）

### 32. Agentic ESOpt：以最小GPU内存需求微调长视距LLM智能体
- **原标题**：Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Memory Requirements
- **作者**：Zhi Zheng, Rongsheng Chen, Yunpeng Ba, Zhenkun Wang, Yee Whye Teh, Wee Sun Lee
- **机构**：新加坡国立大学（NUS）
- **日期**：2026-08-18
- **类型**：学术论文
- **来源**：arXiv:2608.17310
- **译文**：[works/arxiv-2608-17310-translation.md](../works/arxiv-2608-17310-translation.md)
- **原文**：https://arxiv.org/abs/2608.17310
- **核心创新**：
  - 提出基于进化策略（ES）的长视距LLM智能体微调方法，解决RL的GPU内存瓶颈和信用分配难题
  - 三大核心优势：1) 仅需推理级GPU内存实现全参数优化；2) 黑盒反馈接口易与提示空间优化组合；3) 轨迹级参数归因提供更好的长视距可扩展性
  - 余弦衰减调度机制：扰动尺度σ动态调整，训练时保留非零终止值实现温和正则化，测试时衰减至0实现精细适应
  - 参数-提示协同进化框架：支持在线参数适应与技能空间优化、测试时搜索的组合
  - 数独任务（15轮）：比最强GRPO基线提升+12.5%，验证长视距优势随视距增长而扩大
  - ReAct工具使用：Math和DocVQA任务平均比基础模型提升13.7%，比Agentic GRPO提升8.3%
  - WebArena-Lite：27B模型全参数优化将无技能基线从29.47%提升至36.16%，结合Trace2Skill进一步提升至36.36%
  - 测试时自动启发式设计：在线提示-参数协同进化，在36个设置中的28个超越匹配基线
  - 种群敏感性研究：更强骨干模型（9B）对小种群敏感性低于较弱模型（4B）
- **技术价值**：S 级（长视距智能体微调重大突破，ES vs RL系统对比，理论+实验完整，27B全参数微调工程价值高）

### 33. ArmorOCR: 通过观察迁移自蒸馏实现基础化对抗性视觉感知
- **原标题**：ArmorOCR: Grounded Adversarial Visual Perception via Observation-Transferred Self-Distillation
- **作者**：Linhan Cao, Siyuan Li, Jun Lan, Liangbo He, Guannan Li, Xiaolei Huang, Jun Jia, Shuheng Zhou, Huijia Zhu, Weiqiang Wang, Wei Sun
- **日期**：2026-08-20
- **类型**：学术论文
- **来源**：arXiv:2608.20122
- **译文**：[works/armorocr-translation.md](../works/armorocr-translation.md)
- **原文**：https://arxiv.org/abs/2608.20122
- **代码**：https://github.com/ant-research/ArmorOCR
- **核心创新**：
  - AdvSpot 基准：首个区域定位的对抗性OCR评测基准，390张图像，5大类13种细粒度对抗性OCR类型（成像退化、空间操纵、字形变体、视觉编码、上下文融合）
  - ArmorOCR 框架：两阶段训练（观察迁移自蒸馏OTSD + 群相对策略优化GRPO），内化转换视图揭示的感知能力
  - 阶段1-OTSD：学生模型从特权转换观察中获取对抗性OCR感知，响应区域感知token加权，置信门控JSD选择性迁移
  - 阶段2-GRPO：任务条件化奖励（定位、识别、检测、视觉问答）联合优化基础化OCR能力
  - 系统的对抗性OCR分类法：基于失败机制组织，区域级标注（边界框、转录、类别标签、VQA对）
  - AdvSpot性能：平均准确率55.7%，平均IoU 63.3%，在13种细粒度类型上全面超越基线模型
  - 跨基准验证：AdvOCR平均56.0%（+1% vs VACoT），SmuggleBench平均17.1%（+0.7% vs SmugggleCoT）
  - 保持通用OCR能力：OCRBench上与基线模型性能相当（74.0% vs 74.5%）
- **技术价值**：S 级（首个区域定位对抗性OCR基准，观察迁移自蒸馏方法论创新，系统化分类法与实验验证）

---

## 🔍 观察项 / 候选材料（不计入 33 篇）

| 候选 | 类型 | 去向 | 角度 / 为何只做观察项 | 原文 |
|---|---|---|---|---|
| LLaDA 2.2：全球首个 Agentic 扩散模型 | 科技报道 | 观察项 | 蚂蚁 inclusionAI 团队。扩散模型在 LLM Agent 任务的突破：Levenshtein 编辑范式 + L-EBPO 强化学习 + BlockRouting（128K上下文）。Agent 任务性能接近自回归（差距 <2分），效率提升 1.64×。科技报道非学术论文，但技术信息完整。 | [量子位](https://www.qbitai.com/2026/07/461650.html) · [技术报告](https://github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2_2_tech_report.pdf) · [GitHub](https://github.com/inclusionAI/LLaDA2.X) |
| DualDecoder：通过预测性预取加速长上下文 LLM 推理 | 学术论文摘要 | 观察项 | Zuning Liang 等。稀疏 KV 缓存优化：双 token 解码流水线预测关键 KV 条目，实现预取与计算重叠，消除辅助状态的 GPU 内存开销。2.62× 吞吐量提升。仅摘要页翻译，完整论文待补充。 | [arXiv](https://arxiv.org/abs/2607.26475) · [译文](../translate/20260731-arxiv-2607-26475/translations/dualdecoder/translation.md) |
| OpenAI GPT-5.6 自我进化技术 | 科技报道 | 观察项 | 智东西（2026-07-30）。AI 自主优化 GPU kernel（成本降低 20%）+ speculative decoding（效率提升 >15%）+ Agent 框架优化（lazy loading / prompt caching / append-only）+ ARC-AGI-3 性能 4.9× 提升（7.8%→38.3%）。科技报道非官方技术博客，但包含完整性能数据。 | [36kr](https://eu.36kr.com/en/p/3917509136346498) · [译文](../translate/2026-08-01-batch/works-ready/openai-gpt56-self-evolution-translation.md) |
