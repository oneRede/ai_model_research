# AI应用 进展追踪

> 追踪和收录全球 AI 应用的最新进展、实践案例和商业化趋势。

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


## 导航

每个子目录都有自己的 AGENTS.md，说明该目录的用途、内容组织方式和写作约定。
从任何一个目录开始，都能找到下一步该看什么。

## 机械化检查

`scripts/check-consistency.sh` 守护"漂移"问题：

- **C1** — `references/articles.md` 编号 1..N 连续
- **C2** — N 与下游 3 处声明同步（README、`prompts/deep-research-tracker.md` 头部、`references/AGENTS.md` 概览）。文件含独立行 `<!-- check-consistency: skip-count -->` 时豁免
- **C3** — `thinking/`、`feedback/` 的 `*.md` 实际数与 README 中"X 篇"声明一致
- **C4** — `works/*-translation.md` 文件数 ≡ 翻译计数所有声明（badges、`<details>` 摘要、Phase 5 注释、本文件 Phase 5 快照、READMEs 表格行数）
- **C5** — `references/articles.md` 末尾"不计入 N 篇"中的 N ≡ C1 权威值
- **C6** — 翻译流水线本地守卫：`translate/<...>/sources/<slug>/source-full.md` 存在时，对应 `01-analysis.md` 不得再声称"仅摘要页 / 建议补抓全文"。`translate/` 已 gitignore，CI 与干净 clone 自动 SKIP，仅本地有过程稿时触发
- **C7** — / `thinking/` / `feedback/` 正文不得裸写文库计数（"N 篇文章 / N 篇翻译 / N 大概念"）；历史性提法须带"写作时点 / 当时 / 此前 / 首批 / 首轮 / 截至 / 快照"限定词，否则去数字改链 `references/articles.md`
- **C8** — `works/*-translation.md` 必须在 frontmatter 声明 `pipelineRunId` 与 `pipelineSource`，防止绕过 `translate/<batch>/works-ready/` 直接写入正式档案

执行：`bash scripts/check-consistency.sh`（仓库根目录）
启用 pre-commit 阻断：`git config core.hooksPath .githooks`

**CI 兜底**：`.github/workflows/consistency.yml` 在每次 push / PR 时跑同一脚本（不做路径过滤，保证必需检查总能上报）。job 显示名固定为 `consistency / check`——分支保护按 check run 名匹配必需检查，改名会让所有 PR 重新被 "Expected" 卡住。
本地 hook 是开发反馈，CI 是合并门——两层独立，本地未启用 hook 不会绕过检查。
