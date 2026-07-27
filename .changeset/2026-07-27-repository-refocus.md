# 仓库重新聚焦：从 AI 应用到 AI 大模型技术

**日期**：2026-07-27  
**类型**：重大重构

## 改动概述

将仓库从"AI 应用进展追踪"重新聚焦为"AI 大模型技术进展追踪"。

## 核心变更

### 1. 收录范围调整

**之前**：聚焦 AI 在医疗、金融、游戏、工业等领域的应用案例

**现在**：聚焦 AI 大模型的技术进展，包括：
- 新模型发布（GPT、Claude、Gemini、LLaMA 等）
- 架构创新（Transformer 变体、MoE、新注意力机制）
- 训练技术（预训练、持续学习、长上下文、多模态融合）
- 对齐与安全（RLHF、Constitutional AI、Red Teaming）
- 评测基准（新评测方法、基准数据集、能力测试）
- 模型能力研究（涌现能力、泛化性、可解释性）
- 推理优化（量化、剪枝、蒸馏、KV-cache 优化）
- 硬件与基础设施（训练集群、GPU/TPU 优化）
- 数据工程（数据集构建、清洗、合成数据）

### 2. 内容清空

- 删除 14 篇应用案例翻译（医疗、金融、工业等）
- 清空 `works/imgs/` 图片目录
- 重置 `references/articles.md` 为 0 篇
- 保留仓库架构和流水线机制

### 3. 文档更新

**主要文档**：
- `README.md` — 更新项目定位和收录范围
- `AGENTS.md` — 更新仓库导航说明
- `references/articles.md` — 重置索引为空
- `works/AGENTS.md` — 更新作品方向参考

**辅助文档**：
- `thinking/AGENTS.md` — 更新写作方向示例
- `prompts/deep-research-tracker.md` — 完全重写情报追踪 prompt，覆盖学术论文、技术博客、中英文社区
- `prompts/AGENTS.md` — 更新场景分类
- `.claude/skills/curate-research/SKILL.md` — 更新评审标准

### 4. 保留的架构

✅ 6 阶段流水线（抓取→翻译→评审→收录→校验→清理）  
✅ C1-C12 一致性检查机制  
✅ works/ + references/ 双层结构  
✅ thinking/ + feedback/ 反思空间  
✅ 人类闸门决策机制  
✅ baoyu-translate 和 baoyu-url-to-markdown 子 skill

## 验证结果

```bash
bash scripts/check-consistency.sh
```

**结果**：✅ 全部检查通过（C1-C8，空仓库状态）

## Git 状态

**修改的文件**（9 个）：
- README.md
- AGENTS.md
- references/articles.md
- references/AGENTS.md
- works/AGENTS.md
- thinking/AGENTS.md
- prompts/AGENTS.md
- prompts/deep-research-tracker.md
- .claude/skills/curate-research/SKILL.md

**删除的文件**（14 个翻译作品）：
- works/*-translation.md（全部应用案例）
- works/imgs/（图片目录）

## 下一步

1. **启动内容收录**：使用更新后的 `deep-research-tracker.md` prompt 开始追踪大模型技术进展
2. **首批候选**：可从 arXiv、OpenAI/Anthropic 研究博客、NeurIPS/ICML 最新论文开始
3. **验证流水线**：用 1-2 篇技术论文验证翻译流水线在新主题下的效果

## 技术债务

无。架构保持完整，一致性检查全部通过。

## 兼容性

⚠️ **破坏性变更**：删除了所有历史翻译内容。如需恢复，可从 Git 历史中找回。

## 审批

- [x] 一致性检查通过
- [x] 文档逻辑自洽
- [x] 流水线配置更新完成
