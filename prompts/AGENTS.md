# prompts/ — 提示词积累

学习过程中验证有效的提示词，按场景或工作流沉淀。**只收录亲测有效的，不收录未验证的。**

## 文件形态（两种合法形式）

### 形式 A：单条 Prompt（按场景命名）

- 文件名：`{场景}.md`，如 `code-generation.md`、`code-review.md`
- 必备字段：用途、提示词正文
- 建议字段：效果评价（好/中/差）、改进记录、适用模型 / 不适用场景

### 形式 B：Prompt 工作流（按工具链命名）

- 文件名：`{工作流名称}.md`，如 `deep-research-tracker.md`
- 必备字段：工作流目标、各步 Prompt 正文（A/B/C…）、链路图
- 适合多模型 / 多步骤协作的场景（例如 ChatGPT Deep Research → Claude 深度分析 → 人工确认）

## 场景分类参考

| 场景 | 说明 |
|------|------|
| 项目初始化 | 让智能体生成仓库骨架、AGENTS.md、CI 配置 |
| 代码生成 | 在架构约束下生成业务代码 |
| 代码审查 | 智能体审查智能体生成的代码 |
| 重构清理 | 熵管理 / 垃圾回收 |
| 文档维护 | doc-gardening 智能体的提示词 |
| 情报追踪 | 定期发现领域内新内容（见 deep-research-tracker.md） |

## 已有内容

| 文件 | 形态 | 用途 |
|------|------|------|
| [deep-research-tracker.md](deep-research-tracker.md) | 工作流 | 每周追踪 Harness Engineering / AI 编程领域新内容 |

## 下一步

用了某条 Prompt 之后，把效果（好/中/差/翻车）记到 [feedback/](../feedback/)；
积累到一定量再回到本目录补"效果评价 / 改进记录"。
