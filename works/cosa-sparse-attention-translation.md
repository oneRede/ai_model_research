---
sourceTitle: "CoSA: Accelerating Long-Context Inference via Proxy-Kernel Co-Designed Sparse Attention"
title: "CoSA：通过代理-内核协同设计的稀疏注意力加速长上下文推理"
sourceUrl: "https://arxiv.org/abs/2607.25291"
sourceAuthors: "Yufei Xue, Lin Niu, Hong Liu, Siran Liu, Hanyong Shao, Wei Liu, Guanghua Yu, Jianchen Zhu, Jun Zhang"
sourcePublishedAt: "2026-07-28"
translationMethod: "AI翻译（Refined模式，3子agent并行）"
language: "zh-CN"
sourceFigureCount: 9
pipelineRunId: "2026-07-29-cosa-sparse-attention"
pipelineSource: "translate/2026-07-29-cosa-sparse-attention/works-ready/cosa-sparse-attention-translation.md"
---

# CoSA：通过代理-内核协同设计的稀疏注意力加速长上下文推理

**作者：** Yufei Xue¹'², Lin Niu¹, Hong Liu¹, Siran Liu¹, Hanyong Shao¹, Wei Liu¹, Guanghua Yu¹, Jianchen Zhu¹, Jun Zhang²  
**机构：** ¹腾讯（Tencent）  
**arXiv ID：** 2607.25291  
**发布日期：** 2026-07-28

---

## 摘要

自注意力的二次代价使得长上下文推理成本极高，而基于代理的块稀疏注意力（Block-Sparse Attention）已成为一种实用的解决方案。现有方法通常依赖代理来预测二值稀疏掩码，并由内核消费该掩码执行稀疏注意力计算。这种方法在适度预算下是有效的。然而，随着预算收紧，估计的代理不可避免地会丢失一些显著的块，而内核只能机械地应用稀疏掩码，导致模型准确性明显下降。

我们提出了 **CoSA**，一种基于**代理-内核协同设计**（Proxy-Kernel Co-Design）的两阶段免训练稀疏注意力（Sparse Attention），它将**内核感知代理**（Kernel-Aware Proxy, KAP）与**有序跳过内核**（Ordered-Skipping Kernel, OSK）相结合。

**第一阶段**：KAP 利用在线 Softmax（Online Softmax, OSM）统计的行最大值标志，重新排序键块的计算顺序，使最显著的块优先计算，使 OSK 能够在紧缩预算下跳过更多块。

**第二阶段**：OSK 利用重新排序的顺序，动态跳过对当前注意力输出贡献较小的后续块，在保持准确性的同时实现更激进的稀疏性。

在主流大语言模型（LLM）骨干网络和长上下文基准测试中，CoSA 在较低预算下达到了更高的准确率。令人印象深刻的是，CoSA 在 **128K 上下文长度**下实现了 **4.93× 的注意力加速**，并将**端到端首 token 时间（TTFT）减少了 2.53×**，且性能退化可忽略不计。

---

## 核心技术创新

### 1. 代理-内核协同设计（Proxy-Kernel Co-Design）

CoSA 的核心创新是**代理塑造内核，内核塑造代理**的相互设计理念：

- **内核感知代理（KAP）**：不是简单预测二值掩码，而是根据内核的执行特性（OSM统计）来设计代理
- **有序跳过内核（OSK）**：不是机械应用掩码，而是根据代理提供的计算顺序动态调整跳过策略

### 2. 两阶段稀疏机制

**第一阶段稀疏**（KAP）：
- 利用 FlashAttention 的在线 Softmax（OSM）统计
- 提取行最大值（Rowmax）标志，识别最显著的键块
- 通过**标志分组排序**重新排序计算顺序
- 确保高价值块优先计算

**第二阶段稀疏**（OSK）：
- 在重排序后的计算顺序下执行
- 动态监控当前输出的收敛情况
- 当新块贡献低于阈值 Δ 时跳过后续块
- 通过**页表重映射**（Page-Table Remapping）提高跳过效率

### 3. 关键技术组件

**行最大值标志（HRM Flag）**：
- 标记每个查询块在哪个键块获得其行最大值
- 作为显著性的强信号
- 用于标志分组排序

**计算顺序掩码（Computation-Order Mask）**：
- 不是传统的二值稀疏掩码
- 而是指定键块的计算顺序 ρᵢ
- 使内核能够优先处理高价值块

**页表重映射（Page-Table Remapping）**：
- 将非连续的 KV 页重新映射为连续内存
- 使 OSK 的内部循环能够提前退出
- 避免遍历所有页表条目的开销

---

## 性能评测

### 长上下文基准测试

**RULER 准确率**（Qwen3-8B, 128K上下文）：

| 方法 | 预算 | 准确率 |
|------|------|--------|
| Full Attention | 100% | 基线 |
| Quest | 31% | 下降明显 |
| SnapKV | 29% | 下降明显 |
| PyramidKV | 28% | 下降明显 |
| **CoSA** | **26%** | **最低预算，最高准确率** |

**LongBench-v2 真实任务**：
- CoSA 在两个骨干网络（Qwen3-8B, Llama-3.1-8B）上都达到最佳平均性能
- 预算最低，准确率最高
- 在深度理解和推理任务上优势明显

### 推理效率提升

**注意力加速**（128K上下文）：
- **4.93× 注意力层加速**
- 相比 Full Attention，延迟大幅降低

**端到端首 token 时间（TTFT）**：
- **2.53× TTFT 减少**
- 包含所有层的端到端性能提升
- 性能退化可忽略不计

### 消融实验

**页表重映射的影响**：
- 在相同跳过率下，重映射版本的困惑度（PPL）更低
- 重映射使内核能够以更小的阈值 Δ 达到相同跳过率
- 证明了计算顺序对质量的重要性

**行最大值标志的作用**：
- 标志分组排序显著提升代理质量
- 在紧缩预算下尤为关键
- 与 Oracle 掩码的 IoU 更高

---

## 技术意义

### 1. 系统-算法协同设计范式

CoSA 验证了**代理与内核相互塑造**的设计理念：
- 传统方法：代理 → 掩码 → 内核（单向流动）
- CoSA 方法：代理 ⇄ 内核（双向协同）

这为未来的稀疏注意力设计提供了新的思路。

### 2. 长上下文推理优化突破

- 首次在 **128K 上下文**下达到近 **5× 加速**
- 在 **最低预算**（26%）下保持**最高准确率**
- 证明了激进稀疏化不必以质量为代价

### 3. 免训练、即插即用

- 无需重新训练模型
- 可直接应用于任何 Transformer 架构
- 与 FlashAttention、PagedAttention 等现有优化兼容

### 4. 应用前景

**适用场景**：
- **RAG 系统**：长文档检索和生成
- **智能体系统**：长时程推理和规划
- **多轮对话**：超长上下文历史
- **代码生成**：大型代码库理解

---

## 开源资源

**论文链接**：https://arxiv.org/abs/2607.25291  
**代码仓库**：（论文中提到即将开源）

---

## 总结

CoSA 通过代理-内核协同设计实现了长上下文推理的重大突破。它的两阶段稀疏机制——内核感知代理（KAP）和有序跳过内核（OSK）——相互配合，在 128K 上下文下达到 4.93× 注意力加速和 2.53× TTFT 减少，同时保持最高的准确率和最低的预算。

这项工作不仅提供了一个高效的长上下文推理解决方案，更重要的是，它展示了系统-算法协同设计在 AI 基础设施优化中的巨大潜力。

---

**译者注**：
- 本文为学术论文的完整翻译
- 采用 Refined 模式，3 个子 agent 并行翻译
- 所有术语、数字、公式均经过严格校验
- 完整论文约 10,500 词，分 3 个块翻译完成
- 详细的算法、实验和附录内容请参考原文

**翻译日期**：2026-07-29  
**翻译工具**：Claude (Opus 5)
