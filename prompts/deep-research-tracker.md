# 深度研究追踪 Prompt

> 用途：定期（每周）运行，发现AI应用领域的高价值新内容
> 推荐工具：ChatGPT Deep Research（广度搜索）→ Claude（深度分析 + 项目关联）

---

## Prompt A：ChatGPT Deep Research — 广度发现

```
你是一个技术情报分析师。请对以下领域进行深度网络搜索，找出过去 1 周内（{START_DATE} 至 {END_DATE}）发布的高价值内容。

### 搜索领域

核心主题：
- "AI Application"
- "AI for business"
- "Enterprise AI"
- "AI startup"
- "AI adoption"

相关关键词（中英文）：
- AI healthcare,AI in medicine,medical AI,AI for healthcare,AI drug discovery,AI 药物研发,AI 医疗,clinical AI,diagnostic AI,AI radiology,AI pathology
- AI game, AI NPC
- AI design, AI art
- personal AI,AI assistant,consumer AI
- AI security,cybersecurity AI,AI 安全,网络安全 AI
- AI finance,fintech AI,AI 金融,AI 投资
- Claude science,AI for science,scientific AI,AI research,AI 科研,科学计算 AI
- AI materials,AI new materials,materials science AI,AI 新材料,材料科学 AI
- AI industrial,industrial AI,manufacturing AI,AI automation,AI 工业,工业 AI,智能制造

### 搜索范围

必须覆盖的信源（按优先级）：

**Tier 1 — 高权重（行业媒体 + 案例研究）：**
- TechCrunch (AI 应用报道)
- VentureBeat AI
- MIT Technology Review (AI 板块)
- Nature / Science (AI 应用论文)
- Harvard Business Review (企业 AI 案例)
- McKinsey / BCG (AI 应用报告)
- STAT News (医疗 AI)
- The Information
- 36氪（AI 商业化）
- 机器之心、量子位（AI 应用报道）

**Tier 2 — 中权重（公司博客 + 产品案例）：**
- OpenAI Blog (应用案例，非模型技术)
- Anthropic Blog (应用案例)
- Google AI Blog (应用案例)
- Microsoft AI Blog
- 各行业头部公司技术博客（医疗、金融、游戏、制造等）
- ProductHunt (AI 应用产品)
- Hacker News (AI 应用讨论)
- 知乎专栏、少数派、掘金（AI 应用实践）

**Tier 3 — 低权重但可能有惊喜：**
- arXiv (cs.AI 应用方向)
- Medium AI 专栏
- YouTube (AI 应用案例频道)
- Reddit (r/artificial, r/MachineLearning)
- LinkedIn (行业专家分享)


### 我们已知的内容（用于去重和关联）

> **本节是 Prompt 的去重权威**——给外部搜索器（ChatGPT Deep Research 等）使用。
> 它必须自包含，因为搜索器无法访问 `references/articles.md`。
>
> **维护纪律：** 当 `references/articles.md` 新增/删除条目时，**同一次提交中**必须同步更新本节。两份内容的口径（脉络划分、篇数、产品/项目清单）应保持完全一致。


**已跟踪的开源项目/产品：**

**请重点发现：**
- 上述未覆盖的新作者 / 新视角 / 新组织
- 对上述文章的**深度回应或反驳**（不是简单转述）
- 与上述项目**互补或竞争**的新工具 / harness / 框架
- 中文社区针对上述材料的原创分析（少数派、掘金、知乎专栏等）

### 输出格式

请按以下格式输出，每条内容一个条目：

---

#### [编号]. {标题}

- **类型：** 文章 / 开源项目 / 工具 / 演讲 / 论文
- **链接：** {URL}
- **作者/组织：** {作者}
- **日期：** {发布日期}
- **信源层级：** Tier 1 / Tier 2 / Tier 3
- **推荐指数：** ⭐⭐⭐⭐⭐（1-5 星）

**一句话摘要：** {50 字以内}

**核心洞察（3-5 条）：**
1. ...
2. ...
3. ...

**与已知内容的关联：**
- 支持/挑战/扩展了哪篇已有文章的观点
- 填补了哪个已知缺口

**值得收录的理由 / 不值得的理由：**
{判断}

---

### 质量过滤标准

**必须满足（全部）：**
- 有实质性的技术内容（不是纯营销或产品公告）
- 有原创洞察或数据（不是对已有文章的简单转述）
- 来源可信（有署名，有技术背景）

**加分项（满足越多越好）：**
- 有实际数据或实验结果
- 有代码示例或可复现的方案
- 挑战了主流观点
- 来自一线实践者（不是纯理论）
- 有中文社区尚未覆盖的视角

**排除：**
- 纯产品发布/营销内容
- 对已有文章的简单翻译或摘要（没有新观点）
- 过于初级的入门教程
- 纯开发工具类（IDE、Coding Agent、Harness 框架）
- 纯模型训练技术（与实际应用场景无关）

### 输出数量

- 文章类：推荐 5-10 篇，按推荐指数排序
- 开源项目类：推荐 3-5 个
- 其他（工具/演讲/论文）：如有高质量内容，不限数量
```

---

## Prompt B：Claude — 深度分析与项目关联

> 在 ChatGPT 返回结果后，将结果喂给 Claude（在本项目中），做深度分析

```
以下是最近 1 周的技术情报搜索结果。请基于我们项目的已有内容，做以下分析：

{粘贴 ChatGPT 的输出}

### 请分析：

1. **优先级排序**：哪些内容最值得我们收录？考虑因素：
   - 对已收录文章的补充价值
   - 应用场景的新颖性和实践价值
   - 数据和案例的真实性与深度
   - 对 AI 应用落地的启发意义

2. **缺口分析**：这批内容覆盖了我们的哪些知识缺口？还有哪些缺口未被触及？
   当前已知缺口：
   - 医疗 AI 的临床验证案例
   - 金融 AI 的风险控制实践
   - 工业 AI 的 ROI 数据
   - 游戏 AI 的用户体验研究

3. **趋势信号**：这批内容中是否有新的趋势或方向？是否揭示了新的应用领域或商业模式？

4. **收录建议**：对每条推荐内容给出具体建议：
   - 收录到 references/articles.md（哪个脉络）
   - 值得翻译到 works/
   - 值得在 thinking/ 中写分析
   - 暂不收录，持续观察
```

---

## Prompt C：Manus / OpenClaw — 自动化监控

> 用于设置定时监控的固定信源

```
定时任务：每日检查以下信源的更新

监控列表：
**Tier 1 — 高权重（模型厂商 + 顶级技术博客）：**
- Anthropic Engineering Blog (anthropic.com/engineering)
- OpenAI Blog (openai.com)
- Google DeepMind / Google AI Blog
- Martin Fowler (martinfowler.com)
- Mitchell Hashimoto (mitchellh.com)
- LangChain Blog (blog.langchain.com)
- Simon Willison (simonwillison.net)

**Tier 2 — 中权重（社区 + 行业）：**
- Hacker News (前 100 讨论)
- GitHub Trending (相关仓库)
- X/Twitter 技术社区 (#harness-engineering, #ai-coding, #context-engineering)
- Dev.to, Medium 技术专栏
- HumanLayer, Cursor, Windsurf, Codex 相关博客
- 中文社区：少数派、掘金、知乎专栏
- TechCrunch、STAT News、Import AI、VentureBeat、 Product

**Tier 3 — 低权重但可能有惊喜：**
- arXiv (cs.SE, cs.AI 交叉)
- 个人技术博客
- YouTube 技术频道
- Reddit (r/LocalLLaMA, r/ChatGPT, r/programming)

匹配关键词：
- AI healthcare,AI in medicine,medical AI,AI for healthcare,AI drug discovery,AI 药物研发,AI 医疗,clinical AI,diagnostic AI,AI radiology,AI pathology
- AI game, AI NPC
- AI design, AI art
- personal AI,AI assistant,consumer AI
- AI security,cybersecurity AI,AI 安全,网络安全 AI
- AI finance,fintech AI,AI 金融,AI 投资
- Claude science,AI for science,scientific AI,AI research,AI 科研,科学计算 AI
- AI materials,AI new materials,materials science AI,AI 新材料,材料科学 AI
- AI industrial,industrial AI,manufacturing AI,AI automation,AI 工业,工业 AI,智能制造

输出：
- 有更新时，发送通知（标题 + 链接 + 匹配的关键词）
- 无更新时，静默
```

---

## 工作流总结

```
┌─────────────────────────────────────────────────┐
│  Layer 1: 自动化监控（每日）                       │
│  工具：OpenClaw / Manus                          │
│  输出：固定信源的新内容通知                         │
└──────────────────────┬──────────────────────────┘
                       ↓ 有新内容时触发
┌─────────────────────────────────────────────────┐
│  Layer 2: 广度搜索（每周）                    │
│  工具：ChatGPT Deep Research                     │
│  输入：Prompt A                                   │
│  输出：5-10 篇文章 + 3-5 个项目的结构化摘要         │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  Layer 3: 深度分析（按需）                         │
│  工具：Claude（本项目内）                          │
│  输入：Prompt B + ChatGPT 输出                    │
│  输出：优先级排序 + 缺口分析 + 收录建议              │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  Layer 4: 人工确认                                │
│  你决定：收录 / 翻译 / 写分析 / 跳过               │
└─────────────────────────────────────────────────┘
```
