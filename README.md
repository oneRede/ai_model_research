# AI 大模型进展追踪

> 追踪和收录全球 AI 大模型的最新技术进展，聚焦模型、算法、数据、硬件层面的创新与突破。

## 前言

这是一个不断生长的学习项目。利用智能体（agent）实现 AI 大模型领域技术进展的自动化收录与整理。

## 收录范围

**聚焦 AI 大模型的技术进展与能力边界**，包括但不限于：

- **新模型发布**：GPT、Claude、Gemini、LLaMA 等主流模型的版本更新与能力演进
- **架构创新**：Transformer 变体、MoE（混合专家）、状态空间模型、新注意力机制
- **训练技术**：预训练方法、持续学习、长上下文扩展、多模态融合训练
- **对齐与安全**：RLHF、Constitutional AI、Red Teaming、数据合成、安全护栏
- **评测基准**：新的评测方法、基准数据集、能力测试框架（推理、规划、工具使用等）
- **模型能力研究**：涌现能力、泛化性、鲁棒性、可解释性、上下文学习
- **推理优化**：量化、剪枝、蒸馏、KV-cache 优化、投机解码
- **硬件与基础设施**：训练集群架构、GPU/TPU 优化、分布式训练、推理加速
- **数据工程**：高质量数据集构建、数据清洗、合成数据生成、数据配比策略

**不收录**：纯应用案例（除非展示模型新能力）、开发工具与框架（LangChain、LlamaIndex 等）、商业分析与市场报告。

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

### 翻译作品（19 篇）

| 标题 | 类型 | 发布日期 | 译文 |
|------|------|---------|------|
| Kimi K3：开放前沿智能 | 技术报告 | 2026-07-27 | [works/kimi-k3-open-frontier-intelligence.md](works/kimi-k3-open-frontier-intelligence.md) |
| 在 NVIDIA GB300 NVL72 上创造 MoE 预训练世界记录 | 技术博客 | 2026-07-21 | [works/nvidia-gb300-moe-training-world-record.md](works/nvidia-gb300-moe-training-world-record.md) |
| OlmoEarth 平台：行星级地理空间推理 | 技术博客 | 2026-07-28 | [works/olmoearth-infrastructure-translation.md](works/olmoearth-infrastructure-translation.md) |
| CoSA：通过代理-内核协同设计的稀疏注意力加速长上下文推理 | 学术论文 | 2026-07-28 | [works/cosa-sparse-attention-translation.md](works/cosa-sparse-attention-translation.md) |
| ForgetBench: 大语言模型长期参数记忆遗忘动态基准 | 学术论文 | 2026-07-29 | [works/forgetbench-translation.md](works/forgetbench-translation.md) |
| 语言模型中的全局工作空间 | 研究报告 | 2026 年 | [works/global-workspace-translation.md](works/global-workspace-translation.md) |
| SVR：通过联合判定-置信度强化学习实现自适应测试时计算的自我验证精炼 | 学术论文 | 2026-07-30 | [works/svr-self-verifying-refinement-translation.md](works/svr-self-verifying-refinement-translation.md) |
| LATCH：扩散语言模型的候选感知解码 | 学术论文 | 2026-07-30 | [works/latch-diffusion-acceleration-translation.md](works/latch-diffusion-acceleration-translation.md) |
| 从专家归约到行为分歧：追踪稀疏 MoE 推理中的数值状态 | 学术论文 | 2026-07-30 | [works/sparse-moe-numerical-state-translation.md](works/sparse-moe-numerical-state-translation.md) |
| 虚拟宽度网络 | 学术论文 | 2025-11-17 | [works/virtual-width-networks-translation.md](works/virtual-width-networks-translation.md) |
| GFlowRL：将分布匹配强化学习扩展到大语言模型 | 学术论文 | 2026-07-27 | [works/gflowrl-scaling-distribution-matching-llm-post-training.md](works/gflowrl-scaling-distribution-matching-llm-post-training.md) |
| AURORA-LM：面向连续潜在扩散语言建模的自编码统一表示 | 学术论文 | 2026-08-03 | [works/aurora-lm-continuous-latent-diffusion-translation.md](works/aurora-lm-continuous-latent-diffusion-translation.md) |
| MA-LoT：基于多智能体 Lean 的长链式思考推理增强形式化定理证明 | 学术论文 | 2025-03 | [works/ma-lot-multi-agent-lean-long-cot-theorem-proving-translation.md](works/ma-lot-multi-agent-lean-long-cot-theorem-proving-translation.md) |
| 轻量语言模型的检索增强推理 | 学术论文 | 2025-08-15 | [works/lean-language-model-rag-reasoning-translation.md](works/lean-language-model-rag-reasoning-translation.md) |
| LLaDA MoE v2：扩展混合专家扩散语言模型 | 学术论文 | 2026-08-04 | [works/llada-moe-v2-scaling-diffusion-language-models-translation.md](works/llada-moe-v2-scaling-diffusion-language-models-translation.md) |
| 全带宽 Transformer | 学术论文 | 2026-08-09 | [works/arxiv-2608-08888-translation.md](works/arxiv-2608-08888-translation.md) |
| Transformer 是贝叶斯网络 | 学术论文 | 2026-03-17 | [works/arxiv-2603-17063-translation.md](works/arxiv-2603-17063-translation.md) |
| 超循环 Transformer | 学术论文 | 2026-04-23 | [works/arxiv-2604-21254-translation.md](works/arxiv-2604-21254-translation.md) |

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

> 这个仓库通过智能体实现大模型技术进展的自动化收录。
>
> 收录流程固化为 skill 流水线 [`curate-research`](.claude/skills/curate-research/SKILL.md)：评审由并行 agent 自动完成，`scripts/check-consistency.sh` 守护计数一致性，而"是否收录"的决策权始终由人类掌握。
