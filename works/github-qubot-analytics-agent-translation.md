---
title: GitHub 如何构建内部数据分析 Agent（Qubot）
originalTitle: How We Built an Internal Data Analytics Agent
author: GitHub Engineering Team
date: 2026-07
source: https://github.blog/ai-and-ml/github-copilot/how-we-built-an-internal-data-analytics-agent/
translator: Claude (Opus 4.8)
translationDate: 2026-07-24
sourceFigureCount: null
---

# GitHub 如何构建内部数据分析 AI Agent：Qubot

GitHub 构建了名为 Qubot 的内部数据分析 Agent，让员工可以用自然语言查询数据仓库。

## 系统架构

### 三层架构设计

1. **用户界面层**：支持 Slack、VS Code 和 Copilot CLI 多渠道访问
2. **上下文层**：联邦式知识库，针对不同数据类型定制
3. **查询引擎**：双引擎支持（Kusto 和 Trino）

### 上下文层的分层设计

- **Bronze 数据**：原始事件，由产品团队贡献遥测上下文和元数据
- **Silver 数据**：标准化事实和维度，包含查询示例、使用指南和强制过滤器
- **Gold 数据**：业务规则和指标定义，由数据所有团队维护

## 关键技术实现

### 上下文代理（Context Agent）

通过标准化模板或仓库引用方式收集知识，"ingests, organizes, and normalizes this information into a structured format"。上下文通过 GitHub MCP Server 在运行时加载。

### 评估框架

每次上下文层或 Agent 配置变更都要经过离线评估：

- **测试用例**：带正确答案和基准 SQL 的提示数据集
- **自动化编排**：并行运行多次试验，使用 `gh agent-task create` 启动
- **统计聚合**：计算完成率、准确性和响应时间指标

### 查询引擎智能路由

- 默认使用 Kusto 处理近期事件数据的探索性查询
- 需要复杂连接和深度历史分析时自动切换到 Trino
- 通过自定义 Trino MCP Server 和 Fabric RTI MCP Server 实现

## 核心发现

### 结构化上下文的价值

**"Structured and well curated context not only makes Qubot more accurate, but also three times faster"**

结构化且精心策划的上下文不仅让 Qubot 更准确，而且快了三倍。

### 联邦式执行模式

采用中心辐射型（hub-and-spoke）架构，产品团队拥有遥测数据，业务团队定义黄金数据，避免创建多个孤立工具。

### 零维护成本

帮助团队快速熟悉陌生数据集，减少了数据分析 Slack 频道的咨询量。

## 适用场景

Qubot 定位于探索性问题而非报表工具，例如：

- "哪个用户群体在此功能上留存率最高？"
- "上周哪个产品对指标变动贡献最大？"

结果以 Slack 消息和 markdown PR 形式呈现，支持协作和迭代优化。

## Harness Engineering 视角

Qubot 的成功展示了几个关键的 Harness Engineering 原则：

### 1. 上下文工程的核心地位

**结构化上下文带来 3 倍速度提升**，证明了精心设计的上下文层比单纯提升模型能力更有效。Bronze/Silver/Gold 数据分层提供了清晰的知识组织框架。

### 2. 联邦式架构的可扩展性

中心辐射型设计让不同团队各自维护领域知识，避免了中心化知识库的维护瓶颈。这是大型组织部署 AI Agent 的可持续模式。

### 3. 评估驱动的迭代

每次配置变更都经过离线评估，使用测试用例、自动化编排和统计聚合。这种严格的评估框架保证了系统质量的可控演进。

### 4. 多引擎智能路由

Kusto 和 Trino 双引擎设计展示了如何根据任务特性选择合适的后端，而不是一刀切的单一解决方案。

Qubot 不仅是一个成功的内部工具，更是展示了如何通过 Harness Engineering 方法构建生产级 AI Agent 系统的完整案例。
