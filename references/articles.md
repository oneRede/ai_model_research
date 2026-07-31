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
> 当前规模：**6 篇文章**。最近一次同步：2026-07-31。
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

---

## 🔍 观察项 / 候选材料（不计入 6 篇）

| 候选 | 类型 | 去向 | 角度 / 为何只做观察项 | 原文 |
|---|---|---|---|---|
| LLaDA 2.2：全球首个 Agentic 扩散模型 | 科技报道 | 观察项 | 蚂蚁 inclusionAI 团队。扩散模型在 LLM Agent 任务的突破：Levenshtein 编辑范式 + L-EBPO 强化学习 + BlockRouting（128K上下文）。Agent 任务性能接近自回归（差距 <2分），效率提升 1.64×。科技报道非学术论文，但技术信息完整。 | [量子位](https://www.qbitai.com/2026/07/461650.html) · [技术报告](https://github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2_2_tech_report.pdf) · [GitHub](https://github.com/inclusionAI/LLaDA2.X) |
