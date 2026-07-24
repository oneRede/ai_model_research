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
> 当前规模：**1 篇文章**。最近一次同步：2026-07-24。
---

## 📚 文章索引

### 1. How We Built an Internal Data Analytics Agent (Qubot)

**作者：** GitHub Engineering Team  
**日期：** 2026-07  
**类型：** 系统架构案例  
**主题：** 生产环境 AI Agent 设计 — 企业数据分析应用

GitHub 内部数据分析 Agent 的完整系统设计，展示了三层架构（UI、上下文、查询引擎）和联邦式知识库（Bronze/Silver/Gold 数据分层）。核心发现："结构化上下文让 Qubot 快 3 倍且更准确"。这是一个典型的企业 AI 应用案例，展示了如何将 AI 能力落地到实际业务场景。

**译文：** [works/github-qubot-analytics-agent-translation.md](../works/github-qubot-analytics-agent-translation.md)  
**原文：** https://github.blog/ai-and-ml/github-copilot/how-we-built-an-internal-data-analytics-agent/

---

## 🔍 观察项 / 候选材料（不计入 1 篇）

| 候选 | 类型 | 去向 | 角度 / 为何只做观察项 | 原文 |
|---|---|---|---|---|
| The 2026 Agent Confidence Index | 行业调研 | 参考 | Microsoft × MIT TR 调研：300 人评估 101 任务，高信心领域为自动化报告、样板代码、证书监控等；59% 优先人机协作。行业趋势数据，缺乏实施细节 | [链接](https://www.microsoft.com/en-us/microsoft-cloud/blog/2026/06/29/the-2026-agent-confidence-index-where-300-builders-see-real-momentum/) |
| Jefferies Trade Assistant | 金融 AI 案例 | 参考 | AWS 客户案例：投资银行交易员用 Claude + MCP 查询百万行交易数据，8 步工作流程完整，但无量化效果数据 | [链接](https://aws.amazon.com/blogs/machine-learning/building-trade-assistant-how-jefferies-optimized-front-office-trading-operations-with-ai/) |
| AT&T OTel 2.0 电信 AI | 企业 AI 基础设施 | 参考 | Azure 案例：电信行业专用模型，多模型策略 + 异构 GPU（530 个），处理 1T tokens，节省数千万美元成本，但缺乏应用层技术细节 | [链接](https://azure.microsoft.com/en-us/blog/att-and-microsoft-scale-trillion-token-workloads-with-microsoft-foundry-and-amd/) |
| Self Inspection AI 车检 | 汽车科技 | 参考 | TechCrunch 融资报道，智能手机视觉检测车损 + 估价，年处理百万级，客户包括 Avis、CarOffer，但缺技术细节 | [链接](https://techcrunch.com/2025/02/07/self-inspection-raises-3m-for-its-ai-powered-vehicle-inspections/) |
| LangChain Overview | 框架文档 | 参考 | 官方文档概览页，清晰的架构理念（智能体=模型+框架），但不是深度技术文章 | [链接](https://docs.langchain.com/oss/python/langchain/overview) |
| Anthropic Research Overview | 研究索引 | 参考 | 研究项目列表和团队介绍，提供 2026 年最新研究方向快照，但属于索引页性质 | [链接](https://www.anthropic.com/research) |