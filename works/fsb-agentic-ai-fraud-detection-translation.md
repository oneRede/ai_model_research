---
title: 大型银行如何用 Agentic AI 加强反欺诈检测
originalTitle: Strengthening Fraud Detection with Agentic AI
author: Financial Stability Board
date: 2026-06-10
source: https://www.fsb.org/uploads/P100626.pdf
translator: Claude (Opus 4.8)
translationDate: 2026-07-24
sourceFigureCount: null
pipelineRunId: 2026-07-ai-applications-strict
pipelineSource: translate/2026-07-ai-applications-strict/works-ready/fsb-agentic-ai-fraud-detection-translation.md
---

# 大型银行如何用 Agentic AI 加强反欺诈检测

Financial Stability Board 在 2026 年关于金融机构负责任采用 AI 的咨询报告中，收录了一个大型国际活跃银行使用 agentic AI 加强反欺诈检测的案例。

这个案例的价值在于，它展示了金融业如何把 agentic AI 放在高风险业务流程中的受控位置：AI 不直接上线规则、不直接做最终处置，而是分析模式、生成建议，再由欺诈分析团队审批。

## 应用场景

银行每天处理数百万笔支付，并监控来自交易、银行卡、线上支付和数字银行交互的海量信号。欺诈模式变化快，传统规则维护容易滞后。

该银行已有 AI 监控能力，每天处理 8,000 万以上 signals。新的 agentic AI 系统被用于增强以下环节：

- 识别新兴欺诈和诈骗模式；
- 评估可疑模式的严重性；
- 分析上下文；
- 生成检测规则建议；
- 将建议提交给 fraud analytics team 审批。

## Human-in-the-loop 设计

案例最重要的治理设计是责任边界：

- agent 负责发现模式和提出规则建议；
- fraud analytics team 审核；
- 只有经人工批准的规则才会实施。

这避免了在金融风控场景中让 AI 自动执行高风险动作，也保留了可审计和可问责的流程。

## 实施与效果

报告披露了若干关键事实：

- 系统由银行内部数据科学和工程团队在三个月内构建；
- 运行在云数据平台和核心银行基础设施上；
- 银行每天监控 8,000 万以上 signals；
- 银行每天处理数百万笔支付，并发送数千条主动欺诈警告；
- agent 参与开发或更新约四分之三的 card fraud rules；
- 2026 财年上半年，欺诈损失同比下降 20% 以上。

这些指标说明，该系统不是试点 demo，而是进入了金融机构核心风控流程。

## 为什么这个案例重要

金融 AI 的难点通常不只是模型效果，而是治理：谁能审批？规则如何上线？误报和漏报如何管理？责任如何划分？

这个案例给出的答案是：让 AI 进入规则发现和规则建议环节，而不是直接替代风控团队。这种设计既利用了 agentic AI 对复杂信号和新模式的敏感性，又保留了人工审批和组织责任。

## 风险与边界

这个案例也有明显限制：

- 银行匿名，无法独立核验机构名称和后续表现；
- FSB 报告是 consultation report，不是审计报告，也不是强制标准；
- 没有披露误报率、召回率、规则上线后的 A/B 测试方法；
- 20% 以上损失下降不应被简单归因为 agentic AI 的单一贡献。

因此，应把它视为一个高价值监管报告案例，而不是可直接复制的完整技术蓝图。

## 为什么值得收录

它补足了金融 AI 应用方向，尤其是 agentic AI 在高风险业务中的可控落地方式。

这个案例说明，金融机构采用 AI 的关键不是“让 agent 自动行动”，而是把 agent 放在能产生高价值建议、同时仍可人工审批的环节中。
