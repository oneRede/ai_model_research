# 仓库重新聚焦：AI 应用信息收集

> 执行日期：2026-07-24  
> 执行者：Claude (Opus 4.8)

## 目标

将仓库从"Harness Engineering 主题"重构为真正的"**AI 应用信息收集**"，聚焦 AI 在各行业的实际落地场景。

---

## 已完成的工作

### 1. ✅ 重新定义收录范围

**新增明确的收录边界：**

**✓ 收录领域：**
- 医疗健康：AI 辅助诊断、药物研发、医学影像分析、临床决策支持
- 金融科技：智能投顾、风险控制、反欺诈、量化交易
- 游戏娱乐：AI NPC、游戏设计、内容生成
- 工业制造：智能制造、预测性维护、供应链优化、质量控制
- 企业应用：客服机器人、数据分析、流程自动化、知识管理
- 科研教育：科学计算、材料发现、AI for Science、个性化学习
- 设计创作：AI 设计工具、艺术创作、内容生成
- 消费产品：个人 AI 助手、智能硬件、生活服务

**✗ 不收录：**
- 纯开发工具（IDE、Coding Agent 框架、Harness）
- 模型训练技术
- 纯学术理论（除非有明确应用场景）

### 2. ✅ 更新核心文档

**修改文件：**
- `AGENTS.md` — 添加收录范围说明，移除 Harness Engineering 相关表述
- `README.md` — 更新前言和核心概念，明确聚焦 AI 应用
- `prompts/deep-research-tracker.md` — 重写信源列表和质量过滤标准

**信源优先级调整：**
- Tier 1：行业媒体（TechCrunch、MIT Tech Review、Nature/Science）、商业案例研究（McKinsey、BCG、HBR）
- Tier 2：公司博客的应用案例部分、行业头部公司技术博客
- Tier 3：arXiv 应用方向论文、Medium、LinkedIn 专家分享

### 3. ✅ 清理不符合定位的内容

**删除：**
- `works/github-copilot-harness-benchmark-translation.md` — 开发工具性能测试
- `works/github-copilot-code-review-optimization-translation.md` — 开发工具工作流设计

**保留：**
- `works/github-qubot-analytics-agent-translation.md` — 符合"企业应用：数据分析"定位

**更新文章计数：** 从 3 篇 → 1 篇

### 4. ✅ 同步所有下游缓存

**已更新：**
- `references/articles.md` — 重写为只包含 1 篇文章（Qubot）
- `README.md` — 翻译作品表格更新为 1 篇
- `works/AGENTS.md` — 清理旧翻译列表，只保留 1 篇

### 5. ✅ 运行一致性检查

```bash
bash scripts/check-consistency.sh
```

**结果：✓ consistency checks passed**

所有 C1-C7 检查通过或正常跳过（部分 badge/计数模式因内容重构而不存在，属于预期行为）。

---

## 文件修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `AGENTS.md` | 修改 | 添加收录范围章节 |
| `README.md` | 修改 | 更新前言、核心概念、翻译表格 |
| `prompts/deep-research-tracker.md` | 修改 | 重写信源、关键词、质量标准 |
| `references/articles.md` | 修改 | 重置为 1 篇文章 |
| `works/AGENTS.md` | 修改 | 清理旧内容 |
| `works/github-copilot-harness-benchmark-translation.md` | 删除 | 不符合新定位 |
| `works/github-copilot-code-review-optimization-translation.md` | 删除 | 不符合新定位 |

---

## 下一步建议

### 立即行动

1. **审查修改** — 检查所有文档修改是否符合预期
2. **提交变更** — 如果满意，提交这次重构
   ```bash
   git add -A
   git commit -m "refactor: 重新聚焦为 AI 应用信息收集（删除 Harness 内容，保留 1 篇企业应用案例）"
   ```

### 短期目标

3. **补充内容** — 运行 `prompts/deep-research-tracker.md` 中的搜索流程，收录真正的 AI 应用案例：
   - 医疗 AI 的临床应用案例
   - 金融 AI 的风控实践
   - 工业 AI 的 ROI 数据
   - 游戏 AI 的用户体验研究

4. **完善文档** — 考虑添加：
   - 各领域的应用现状分析（thinking/）
   - 实践中的经验教训（feedback/）
   - 行业趋势观察

### 长期规划

5. **建立评审标准** — 针对不同应用领域制定收录标准：
   - 医疗：是否有临床验证数据
   - 金融：是否披露真实 ROI
   - 工业：是否有生产环境部署经验

6. **社区建设** — 考虑邀请各领域专家审阅相关内容

---

## 备注

- 仓库中大量的 `D`（删除）状态文件是你在本次重构前已经做的清理工作
- 本次执行只额外删除了 2 篇不符合定位的翻译
- 一致性检查脚本运行正常，所有必要检查通过
- 暂存区（translate/）未受影响（gitignored）

---

**重构完成！仓库现在清晰聚焦于 AI 应用的实际落地场景。**
