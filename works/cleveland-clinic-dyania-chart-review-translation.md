---
title: AI 病历审查如何识别罕见病临床试验候选患者
originalTitle: AI-Driven Chart Review Accurately Identifies Potential Rare Disease Trial Participants in New Study
author: Cleveland Clinic / Dyania Health
date: 2026-03-03
source: https://newsroom.clevelandclinic.org/2026/03/03/ai-driven-chart-review-accurately-identifies-potential-rare-disease-trial-participants-in-new-study
translator: Claude (Opus 4.8)
translationDate: 2026-07-24
sourceFigureCount: null
pipelineRunId: 2026-07-ai-applications-strict
pipelineSource: translate/2026-07-ai-applications-strict/works-ready/cleveland-clinic-dyania-chart-review-translation.md
---

# AI 病历审查如何识别罕见病临床试验候选患者

Cleveland Clinic 与 Dyania Health 的研究展示了一个医疗 AI 的高价值落地场景：不是直接替代医生诊断，而是嵌入电子病历系统，用于临床试验招募前的病历审查和候选患者预筛。

该系统用于 ATTR-CM 相关罕见病临床试验的预筛。它读取结构化与非结构化 EMR 数据，围绕试验纳入与排除标准生成判断，并给出可审计的依据。医生仍保留最终判断权，AI 的角色是扩大筛查覆盖、减少漏筛、提高招募效率。

## 应用场景

罕见病临床试验招募通常面临两个问题：

1. 合格患者分散在大量病历中，人工审查成本高；
2. 传统转诊路径容易漏掉代表性不足的人群。

在这个案例中，AI 被部署在 Cleveland Clinic 的医疗系统内部，用于从真实 EMR 中识别可能符合 DepleTTR-CM Phase 3 trial 条件的患者。

## 关键结果

研究披露了多项量化指标：

- 一周内审查 1,476 名患者；
- 识别 46 名潜在匹配患者；
- 在 7,700 个 trial-specific questions 上达到 96.2% accuracy；
- 对不合格患者的排除表现为 99% negative predictive value；
- AI + clinician review 找到的 30 个 trial matches 中，有 29 个没有被传统招募路径发现；
- AI 辅助筛查在 6 天内入组 7 人，而传统筛查 90 天入组 10 人；
- AI 识别人群中 Black 患者占比 36.6%，传统筛查为 7.1%。

这些数字说明，AI 的主要价值不是“自动决定谁能入组”，而是在真实临床运营流程中提高筛查速度、覆盖面和可解释性。

## 系统设计要点

### 1. 结合结构化与非结构化病历

传统规则系统更容易处理结构化字段，但大量临床信息存在于医生笔记、检查摘要和历史记录中。该系统的价值在于同时综合这些信息，而不是只依赖单一数据源。

### 2. 试验标准级推理

系统并不是给出一个笼统的“合格 / 不合格”标签，而是围绕试验标准逐条回答问题，并为判断提供依据。这让医生可以审查 AI 的推理路径。

### 3. Clinician-in-the-loop

AI 负责预筛和解释辅助，医生负责最终确认。这个边界非常重要：医疗场景中的 AI 适合先承担高通量、可复核、非最终裁决任务。

### 4. 嵌入现有医疗系统

研究强调该系统在医疗系统防火墙内运行，并嵌入 EMR 工作流。对于医疗 AI 来说，能否进入现有系统和合规流程，往往比模型本身更决定落地价值。

## 风险与边界

这个案例很强，但不应被过度外推：

- 这是单病种、单试验、单一大型 health system 的验证；
- 96.2% accuracy 是针对 trial-specific criteria question 的评估，不等同于所有临床任务的通用准确率；
- Black 患者占比提升是重要信号，但不能简单断言算法天然公平，还需要结合候选池构成、疾病流行率和既有转诊路径分析；
- Cleveland Clinic 已投资 Dyania Health，并可能从技术商业化中获益，存在利益相关关系。

## 为什么值得收录

这篇材料展示了医疗 AI 的一个务实方向：将 AI 用于临床研究运营中的病历审查、候选发现和流程提效。相比“AI 诊断超过医生”这类容易被误读的叙事，它更接近可控、可解释、可部署的医疗 AI 应用。

它的启发是：高价值医疗 AI 未必从最终诊断开始，而可以先从高成本、低风险、可复核的临床运营环节切入。
