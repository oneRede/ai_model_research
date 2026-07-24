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
> 当前规模：**5 篇文章**。最近一次同步：2026-07-24。
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

### 2. AI-Driven Chart Review Accurately Identifies Potential Rare Disease Trial Participants

**作者：** Cleveland Clinic / Dyania Health  
**日期：** 2026-03-03  
**类型：** 医疗 AI 应用案例  
**主题：** 临床试验招募 / EHR chart review / human-in-the-loop

Cleveland Clinic 与 Dyania Health 将 AI 病历审查嵌入真实 EMR 与罕见病临床试验招募流程。一周审查 1,476 名患者，在 7,700 个 trial-specific questions 上达到 96.2% accuracy，并显著提高候选发现与入组速度。需注意单病种/单系统验证与商业利益关系。

**译文：** [works/cleveland-clinic-dyania-chart-review-translation.md](../works/cleveland-clinic-dyania-chart-review-translation.md)  
**原文：** https://newsroom.clevelandclinic.org/2026/03/03/ai-driven-chart-review-accurately-identifies-potential-rare-disease-trial-participants-in-new-study

---

### 3. Harnessing Generative Artificial Intelligence for Predictive Maintenance

**作者：** Production Planning & Control research article  
**日期：** 2026-05-26  
**类型：** 工业 AI 应用案例  
**主题：** 预测性维护 / 轮胎制造 / 自然语言数据湖诊断

轮胎制造企业在 curing presses 环节引入 GenAI-enabled predictive maintenance solution，将 domain-adapted LLM 与 plant process data lake 结合。案例给出生产 KPI 前后对比：dry cycle time 106 秒降至 80 秒、日产能增加 304 条轮胎、复杂根因分析从约 10 天降至约 2 小时。

**译文：** [works/genai-predictive-maintenance-tyre-manufacturing-translation.md](../works/genai-predictive-maintenance-tyre-manufacturing-translation.md)  
**原文：** https://www.tandfonline.com/doi/full/10.1080/09537287.2026.2670524

---

### 4. Strengthening Fraud Detection with Agentic AI

**作者：** Financial Stability Board  
**日期：** 2026-06-10  
**类型：** 金融 AI 应用案例  
**主题：** 反欺诈 / agentic AI / human-in-the-loop 治理

FSB 2026 咨询报告收录的大型国际活跃银行匿名案例：agentic AI 在既有每日 8,000 万+ signals 监控能力基础上，识别新兴欺诈模式并生成检测规则建议，由 fraud analytics team 审批后实施。该 agent 参与约四分之三的 card fraud rules，帮助 2026 财年上半年欺诈损失同比下降 20%+。

**译文：** [works/fsb-agentic-ai-fraud-detection-translation.md](../works/fsb-agentic-ai-fraud-detection-translation.md)  
**原文：** https://www.fsb.org/uploads/P100626.pdf

---

### 5. DiffSyn: A Generative Diffusion Approach to Materials Synthesis Planning

**作者：** MIT News / Nature Computational Science  
**日期：** 2026-02-02  
**类型：** AI for Science 应用案例  
**主题：** 材料合成规划 / 生成式 AI / 实验验证

MIT DiffSyn 将目标材料结构到合成路线建模为 one-to-many 生成问题，基于 50 年文献中的 23,000+ 合成配方训练 diffusion model。研究者使用模型建议路线成功制备新的 zeolite material，数据与代码可用。案例仍处科研验证阶段，尚非产业部署。

**译文：** [works/mit-diffsyn-materials-synthesis-translation.md](../works/mit-diffsyn-materials-synthesis-translation.md)  
**原文：** https://news.mit.edu/2026/how-generative-ai-can-help-scientists-synthesize-complex-materials-0202

---

## 🔍 观察项 / 候选材料（不计入 5 篇）

| 候选 | 类型 | 去向 | 角度 / 为何只做观察项 | 原文 |
|---|---|---|---|---|
| AAA 游戏工作室 AI NPC 上线前红队案例 | 游戏 AI / NPC 安全测试 | 参考 | Alice 供应商案例：匿名 AAA studio 对游戏内 AI NPC 做自动+人工对抗测试，覆盖 4 种语言、多模态与 gameplay scenarios，发现 20,000+ 策略/叙事偏离输出；客户匿名且缺修复后指标 | [链接](https://alice.io/case-studies/aaa-gaming-studio) |
| AI NPCs Are Here. So Why Isn't There a Hit Game Yet? | 游戏 AI / 行业观察 | 参考 | 行业观察：AI NPC 尚未产生爆款，瓶颈在云推理成本、fun evaluation、角色一致性；建议 hybrid design。二级评论，一手数据弱 | [链接](https://aigamingdev.com/blog/ai-npc-reality-check-july-2026/) |
| Roblox Studio is Going Agentic | 游戏开发工具链 | 参考 | Roblox 将 Studio/Assistant 扩展为 plan-build-test agentic workflow，并通过 MCP 暴露项目上下文；44% top 1,000 creators 已使用相关 AI 工具。非游戏内 NPC 案例 | [链接](https://about.roblox.com/newsroom/2026/04/roblox-studio-going-agentic) |
| The 2026 Agent Confidence Index | 行业调研 | 参考 | Microsoft × MIT TR 调研：300 人评估 101 任务，高信心领域为自动化报告、样板代码、证书监控等；59% 优先人机协作。行业趋势数据，缺乏实施细节 | [链接](https://www.microsoft.com/en-us/microsoft-cloud/blog/2026/06/29/the-2026-agent-confidence-index-where-300-builders-see-real-momentum/) |
| Jefferies Trade Assistant | 金融 AI 案例 | 参考 | AWS 客户案例：投资银行交易员用 Claude + MCP 查询百万行交易数据，8 步工作流程完整，但无量化效果数据 | [链接](https://aws.amazon.com/blogs/machine-learning/building-trade-assistant-how-jefferies-optimized-front-office-trading-operations-with-ai/) |
| AT&T OTel 2.0 电信 AI | 企业 AI 基础设施 | 参考 | Azure 案例：电信行业专用模型，多模型策略 + 异构 GPU（530 个），处理 1T tokens，节省数千万美元成本，但缺乏应用层技术细节 | [链接](https://azure.microsoft.com/en-us/blog/att-and-microsoft-scale-trillion-token-workloads-with-microsoft-foundry-and-amd/) |
| Self Inspection AI 车检 | 汽车科技 | 参考 | TechCrunch 融资报道，智能手机视觉检测车损 + 估价，年处理百万级，客户包括 Avis、CarOffer，但缺技术细节 | [链接](https://techcrunch.com/2025/02/07/self-inspection-raises-3m-for-its-ai-powered-vehicle-inspections/) |
| LangChain Overview | 框架文档 | 参考 | 官方文档概览页，清晰的架构理念（智能体=模型+框架），但不是深度技术文章 | [链接](https://docs.langchain.com/oss/python/langchain/overview) |
| Anthropic Research Overview | 研究索引 | 参考 | 研究项目列表和团队介绍，提供 2026 年最新研究方向快照，但属于索引页性质 | [链接](https://www.anthropic.com/research) |
