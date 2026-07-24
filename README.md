# AI 应用进展追踪

> 追踪和收录全球 AI 应用的最新进展、实践案例和商业化趋势

## 前言

这是一个不断生长的学习项目。利用智能体（agent）实现 AI 应用领域内容的自动化收录与整理。

## 收录范围

**聚焦 AI 应用的实际落地场景**，包括但不限于：

- **医疗健康**：AI 辅助诊断、药物研发、医学影像分析、临床决策支持
- **金融科技**：智能投顾、风险控制、反欺诈、量化交易
- **游戏娱乐**：AI NPC、游戏设计、内容生成
- **工业制造**：智能制造、预测性维护、供应链优化、质量控制
- **企业应用**：客服机器人、数据分析、流程自动化、知识管理
- **科研教育**：科学计算、材料发现、AI for Science、个性化学习
- **设计创作**：AI 设计工具、艺术创作、内容生成
- **消费产品**：个人 AI 助手、智能硬件、生活服务

**不收录**：纯开发工具（IDE、Coding Agent 框架）、模型训练技术、纯学术理论（除非有明确应用场景）。

## 核心概念

1. 基于 agent 的搜索、爬取网页能力以及搜索 API，从给定的信源获取 AI 应用相关的信息
2. agent 对获取的文章进行翻译
3. agent 判断文章价值并判断处理方式
4. agent 确保整个项目的健康长久运行

## 📂 仓库结构

```
ai_application_research/
├── README.md              ← 你在这里
├── AGENTS.md              ← 仓库导航入口（给智能体看的）
├── thinking/              # Phase 2：独立思考与质疑
├── feedback/              # Phase 4：踩坑与迭代心得
├── works/                 # Phase 5：可展示的作品
├── prompts/               # 验证有效的提示词积累
└── references/            # 外部资源索引
```

每个子目录都有自己的 `AGENTS.md`，说明该目录的用途和写作约定。这本身就是原文「渐进式披露」的实践。

## 📚 收录内容

### 翻译作品（5 篇）

| 标题 | 来源 | 主题 |
|------|------|------|
| [GitHub 如何构建内部数据分析 Agent](works/github-qubot-analytics-agent-translation.md) | GitHub Blog | 企业数据分析应用 |
| [AI 病历审查如何识别罕见病临床试验候选患者](works/cleveland-clinic-dyania-chart-review-translation.md) | Cleveland Clinic / Dyania Health | 医疗 AI / 临床试验招募 |
| [生成式 AI 在轮胎制造预测性维护中的应用](works/genai-predictive-maintenance-tyre-manufacturing-translation.md) | Production Planning & Control | 工业制造 / 预测性维护 |
| [大型银行如何用 Agentic AI 加强反欺诈检测](works/fsb-agentic-ai-fraud-detection-translation.md) | Financial Stability Board | 金融风控 / Agentic AI |
| [DiffSyn：生成式 AI 如何帮助科学家合成复杂材料](works/mit-diffsyn-materials-synthesis-translation.md) | MIT News / Nature Computational Science | AI for Science / 材料合成 |

完整索引见 [references/articles.md](references/articles.md)

## 🛠️ 开发须知

仓库自带一致性检查脚本 `scripts/check-consistency.sh`，守护数量类漂移，覆盖八层校验：

- **C1** — `references/articles.md` 编号 1..N 连续
- **C2** — N 与下游 3 处声明同步（README、`prompts/deep-research-tracker.md` 头部、`references/AGENTS.md` 概览）。文件含独立行 `<!-- check-consistency: skip-count -->` 时豁免
- **C3** — `thinking/`、`feedback/` 的 `*.md` 实际数与 README 中"X 篇"声明一致
- **C4** — `works/*-translation.md` 文件数 ≡ 翻译计数所有声明（badges、`<details>` 摘要、Phase 5 注释、本文件 Phase 5 快照、READMEs 表格行数）
- **C5** — `references/articles.md` 末尾"不计入 N 篇"中的 N ≡ C1 权威值
- **C6** — 翻译流水线本地守卫：`translate/<...>/sources/<slug>/source-full.md` 存在时，对应 `01-analysis.md` 不得再声称"仅摘要页 / 建议补抓全文"。`translate/` 已 gitignore，CI 与干净 clone 自动 SKIP，仅本地有过程稿时触发
- **C7** — / `thinking/` / `feedback/` 正文不得裸写文库计数（"N 篇文章 / N 篇翻译 / N 大概念"）；历史性提法须带"写作时点 / 当时 / 此前 / 首批 / 首轮 / 截至 / 快照"限定词，否则去数字改链 `references/articles.md`
- **C8** — `works/*-translation.md` 必须在 frontmatter 声明 `pipelineRunId` 与 `pipelineSource`，防止绕过 `translate/<batch>/works-ready/` 直接写入正式档案

**首次 clone 后启用 pre-commit hook：**

```bash
git config core.hooksPath .githooks
```

启用后，每次 commit 涉及 README、`AGENTS.md`、`references/articles.md`、`references/AGENTS.md`、`prompts/deep-research-tracker.md`、或 `thinking/` / `feedback/` / `works/` 中的 `*.md` 时会自动跑检查；不涉及则不打扰。

**手动跑：** `bash scripts/check-consistency.sh`

**CI 兜底：** 即使本地未启用 hook，GitHub Actions（`.github/workflows/consistency.yml`）会在每次 push / PR 时跑同一脚本（不做路径过滤，保证分支保护的必需检查总能得到上报）。本地 hook 是开发期反馈，CI 才是真正的合并门。

详情见根 `AGENTS.md` 的"机械化检查"段。

## 🤖 自动化策展

> 这个仓库通过智能体实现内容的自动化收录。
>
> 收录流程固化为 skill 流水线 [`curate-research`](.claude/skills/curate-research/SKILL.md)：评审由并行 agent 自动完成，`scripts/check-consistency.sh` 守护计数一致性，而"是否收录"的决策权始终由人类掌握。
