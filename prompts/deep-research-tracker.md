# 深度研究追踪 Prompt

> 用途：定期（每周）运行，发现 AI 大模型领域的高价值技术进展
> 推荐工具：ChatGPT Deep Research（广度搜索）→ Claude（深度分析 + 项目关联）

---

## Prompt A：ChatGPT Deep Research — 广度发现

```
你是一个 AI大模型 技术情报分析师。请对以下领域进行深度网络搜索，找出过去 1 周内（{START_DATE} 至 {END_DATE}）发布的高价值内容。

### 搜索领域

核心主题：
- "Large Language Models"
- "AI Model Architecture"
- "Foundation Models"
- "LLM Training"
- "Model Evaluation"
- "AI Model Performance"

相关关键词（中英文）：
- GPT, Claude, Gemini, LLaMA, Mixtral, Qwen, 通义千问
- Transformer, MoE, Mixture of Experts, Attention Mechanism, 混合专家
- RLHF, Constitutional AI, Alignment, AI Safety, AI 对齐, 模型安全
- Long Context, 长上下文, Context Window, 上下文窗口
- Multimodal, 多模态, Vision Language Model, VLM
- Model Compression, Quantization, Pruning, Distillation, 模型压缩, 量化
- Training Infrastructure, GPU Cluster, TPU, 训练集群
- Benchmark, MMLU, HumanEval, BigBench, 评测基准
- Synthetic Data, Data Quality, 合成数据, 数据质量
- Reasoning, Planning, Tool Use, 推理能力, 规划能力
- Inference Optimization, KV Cache, Speculative Decoding, 推理优化

### 搜索范围

必须覆盖的信源（按优先级）：

**Tier 1 — 高权重（学术 + 主流大模型官方博客）：**
- arXiv (cs.CL, cs.AI, cs.LG)
- NeurIPS / ICML / ICLR / ACL / EMNLP 会议论文
- Nature / Science (AI 相关论文)

**主流大模型官方博客（必须覆盖）：**
- OpenAI Research Blog (https://openai.com/research/)
- Anthropic Research (https://www.anthropic.com/research)
- Google AI Blog (https://blog.google/technology/ai/)
- Google DeepMind Blog (https://deepmind.google/discover/blog/)
- Meta AI Research (https://ai.meta.com/blog/)
- DeepSeek (https://www.deepseek.com/ 及相关技术报告)
- Kimi / Moonshot AI (月之暗面，https://www.moonshot.cn/ 及公众号)
- 智谱 AI / GLM (https://www.zhipuai.cn/ 及技术博客)
- 阿里 Qwen / 通义千问 (https://qwenlm.github.io/ 及阿里云博客)
- 百度 ERNIE / 文心 (https://wenxin.baidu.com/ 及技术公众号)
- 字节 Doubao / 豆包 (火山引擎技术博客)
- 腾讯混元 (https://hunyuan.tencent.com/)
- Mistral AI Blog (https://mistral.ai/news/)
- Cohere AI Blog (https://cohere.com/blog)
- xAI (https://x.ai/blog/)
- Inflection AI

**Tier 2 — 中权重（学术机构 + 技术社区）：**
- MIT Technology Review (AI 板块)
- The Gradient
- Transformer Circuits Thread (Anthropic 可解释性研究)
- Hugging Face Blog
- Papers with Code (Trending)
- Together AI Blog
- Modal Labs Blog
- Stanford AI Lab Blog / Stanford HAI
- Berkeley AI Research (BAIR)
- Allen Institute for AI (AI2)
- AI Alignment Forum
- LessWrong (AI 讨论)
- EleutherAI Blog

**Tier 3 — 补充覆盖（技术媒体 + 中文社区）：**
- InfoQ AI
- The New Stack (AI Infrastructure)
- VentureBeat AI (重大模型发布)
- TechCrunch (AI 公司新闻)
- 机器之心（模型技术报道，https://www.jiqizhixin.com/）
- 量子位（AI 前沿，https://www.qbitai.com/）
- 新智元（https://www.aiust.com/）
- AI科技评论（https://www.leiphone.com/category/ai）
- PaperWeekly
- 夕小瑶的卖萌屋（技术解读公众号）
- 大模型之美（公众号/知乎专栏）

### 筛选标准

**必须满足以下至少一项才收录：**

1. **新模型发布**：主流厂商新模型 / 重要开源模型
2. **架构创新**：新的模型架构或显著改进
3. **训练方法突破**：预训练、微调、对齐的新方法
4. **评测基准**：新的评测方法或重要的评测结果
5. **能力研究**：对模型能力边界的深入研究
6. **优化技术**：推理或训练的重要优化方法
7. **基础设施**：训练/推理集群的架构创新
8. **数据工程**：高质量数据集构建方法

**排除标准：**
- 纯应用案例（除非展示模型新能力）
- 开发工具/框架（LangChain、LlamaIndex 等）
- 商业分析/市场报告
- 纯新闻稿/融资消息
- 重复已知信息

### 输出格式

对每条发现，按以下格式输出：

**标题：** {原文标题}
**来源：** {作者/机构}
**日期：** {YYYY-MM-DD}
**链接：** {URL}
**分类：** {模型发布/架构创新/训练技术/评测基准/能力研究/优化技术/基础设施/数据工程}
**一句话总结：** {核心创新点}
**收录建议：** {高/中/低}
**理由：** {为什么值得关注 / 为什么不收录}

---

按收录建议分组，高优先级在前。
```

---

## Prompt B：Claude 深度分析 + 项目关联

将 Prompt A 的输出喂给 Claude，并附上以下指令：

```
你是 AI 大模型进展追踪仓库的内容策展助理。

我刚用 ChatGPT Deep Research 做了一轮广度搜索，找到了以下候选内容（见下方）。
请帮我完成以下任务：

### 任务 1：去重与评级

1. 读取当前仓库的 `references/articles.md`，检查哪些内容已经收录
2. 对未收录的内容，按技术价值重新评级（S/A/B/C）：
   - S 级：重大突破，必须收录（例：GPT-5 论文、新架构 SOTA、重要评测基准）
   - A 级：高价值，强烈推荐（例：主流模型改进、重要方法论文）
   - B 级：有价值，可选收录（例：增量优化、特定场景研究）
   - C 级：信息量不足或重复，暂不收录

### 任务 2：技术深度评估

对 S 和 A 级候选，回答：
- **技术创新点**：具体创新在哪里？
- **方法可复现性**：是否有足够的技术细节？
- **实验数据完整性**：是否有充分的实验验证？
- **与现有收录的关系**：是否填补技术空白？是否与已有内容重复？

### 任务 3：收录建议

对每个 S/A 级候选，给出：
- **建议操作**：立即翻译收录 / 加入观察列表 / 等待后续论文
- **处理优先级**：P0（本周）/ P1（本月）/ P2（有空再说）
- **预期工作量**：简单（产品页）/ 中等（博客文章）/ 复杂（长论文需全文翻译）

### 输出格式

```markdown
## 去重结果
- 已收录：X 条
- 新发现：Y 条（S 级 a 条，A 级 b 条，B 级 c 条，C 级 d 条）

## S 级候选（必收）
### 1. {标题}
- **来源**：{作者/机构}
- **分类**：{分类}
- **创新点**：{具体技术创新}
- **可复现性**：{高/中/低 + 理由}
- **数据完整性**：{充分/部分/不足}
- **与仓库关系**：{填补空白 / 补充细节 / 独立价值}
- **建议操作**：{具体行动}
- **优先级**：P0/P1/P2
- **工作量**：{简单/中等/复杂}

## A 级候选（推荐）
{同上格式}

## 已收录内容（跳过）
- {标题} - 已在 articles.md #{编号}

## 本周行动清单
1. [ ] {候选标题} - {预期工作量} - P0
2. [ ] {候选标题} - {预期工作量} - P1
...
```

---

**仓库上下文：**
{粘贴 references/articles.md 当前内容}

**搜索结果：**
{粘贴 Prompt A 的输出}
```

---

## 使用流程

1. **每周一**：用 Prompt A 在 ChatGPT Deep Research 中运行
2. **输出交接**：将 A 的结果 + `references/articles.md` 内容喂给 Claude
3. **执行分析**：Claude 运行 Prompt B，产出本周行动清单
4. **人工决策**：审查 S/A 级候选，确认是否收录
5. **启动流水线**：对确认收录的内容，运行 `curate-research` skill

---

## 改进记录

| 日期 | 改动 | 原因 | 效果 |
|------|------|------|------|
| 2026-07-27 | 从"AI 应用"改为"AI 大模型技术" | 仓库重新聚焦 | 待验证 |

---

## 下一步优化方向

- [ ] 增加 arXiv RSS 订阅自动化
- [ ] 建立重要会议论文跟踪（NeurIPS/ICML/ICLR）
- [ ] 添加 Papers with Code Trending 自动抓取
- [ ] 集成 Semantic Scholar API 自动获取引用关系
