---
sourceTitle: "MA-LoT: Multi-Agent Lean-based Long Chain-of-Thought Reasoning enhances Formal Theorem Proving"
sourceUrl: "https://ar5iv.labs.arxiv.org/html/2503.03205"
authors: "Ruida Wang, Rui Pan, Yuxin Li, Jipeng Zhang, Yizhen Jia, Shizhe Diao, Renjie Pi, Junjie Hu, Tong Zhang"
publishDate: "2025-03"
arxivId: "2503.03205"
translator: "Claude (AI Translation Pipeline)"
translationDate: "2026-08-04"
pipelineRunId: "20260804-200754"
pipelineSource: "translate/20260804-200754/works-ready/ma-lot-translation.md"
sourceFigureCount: 3
title: "MA-LoT: 基于多智能体 Lean 的长链式思考推理增强形式化定理证明"
url: "https://ar5iv.labs.arxiv.org/html/2503.03205"
coverImage: "https://ar5iv.labs.arxiv.org/assets/ar5iv_card.png"
summary: "提出 MA-LoT 框架，首个结合多智能体系统和长链式思考的 Lean4 形式化定理证明框架，在 MiniF2F-Test 上达到 61.07% 准确率。"
tags: ["形式化验证", "定理证明", "多智能体系统", "长链式思考", "Lean4"]
---

# MA-LoT: 基于多智能体 Lean 的长链式思考推理增强形式化定理证明

## MA-LoT: 基于多智能体 Lean 的长链式思考推理增强形式化定理证明

Ruida Wang    Rui Pan    Yuxin Li    Jipeng Zhang    Yizhen Jia    Shizhe Diao    Renjie Pi    Junjie Hu    Tong Zhang

## 摘要

使用 Lean 等计算机可验证语言解决数学问题对数学和计算机科学社区产生了重大影响。当前最先进的方法使用单个大语言模型（Large Language Model, LLM）作为智能体或证明器，用于生成完整证明或执行树搜索。然而，单智能体方法本质上缺乏一种结构化的方式来将自然语言（Natural Language, NL）中的高层推理与形式语言（Formal Language, FL）验证反馈相结合。为了解决这些问题，我们提出了 MA-LoT：基于多智能体 Lean 的长链式思考（Long Chain-of-Thought）框架，（据我们所知）这是首个在长链式思考（Long CoT）中平衡高层自然语言推理和形式语言验证的 Lean4 定理证明多智能体框架。通过这种结构化交互，我们的方法能够在证明生成中实现更深入的洞察和长期连贯性，而这正是以往方法所难以实现的。我们通过利用长链式思考中涌现的形式化推理能力，使用我们新颖的 LoT 迁移学习（LoT-Transfer Learning）训练-推理流程来实现这一点。广泛的实验表明，我们的框架在 Lean4 版本的 MiniF2F-Test 数据集上达到了 61.07% 的准确率，显著优于 GPT-4（22.95%）、单智能体树搜索（InternLM-Step-Prover，50.70%）和完整证明生成（Godel-Prover，55.33%）基线。此外，我们的发现强调了将长链式思考与形式化验证相结合在更广泛视角下实现更具洞察力生成的潜力。

**关键词**: 形式语言，大语言模型，Lean4

₀ <sub>0</sub> ₁ <sub>1</sub> ₂ <sub>2</sub> ₃ <sub>3</sub> ₄ <sub>4</sub> ₅ <sub>5</sub> ₆ <sub>6</sub> ℚ $\mathbb{Q}$ ≠ $\neq$ ℤ $\mathbb{Z}$ ✝ $\dagger$ ∣ $\mid$ ℝ $\mathbb{R}$ ∧ $\land$ ≤ $\leq$ ⊢ $\vdash$

## 1 引言

形式化推理是人类智能的基石，也是机器学习的关键目标 [^15]，通常通过严格的数学推导来评估 [^36]。随着大语言模型（LLM）的兴起，链式思考（Chain-of-Thought, CoT）提示已成为通过生成中间步骤来形式化推理的方法。这种方法不仅提高了可解释性，还增强了推理性能 [^28]。

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2503.03205/assets/x1.png)

**图 1**: 使用 LLM 进行形式语言定理证明的两个主要方向：单智能体树搜索和完整证明生成及其优缺点。

然而，自然语言（NL）的模糊性导致中间步骤的验证变得复杂，特别是在高等数学中，定理证明普遍存在。现代数学日益增长的复杂性使这一挑战更加严峻，这使得证明验证极具挑战性且容易出错，正如费马大定理长期验证过程所示 [^27]。为了解决这个问题，研究人员提出将推理建立在一阶逻辑基础上，通过形式语言（FL）验证器实现自动验证。该框架保证了严谨性，促进了 Lean [^3] [^14]、Isabelle [^18] 和 HOL Light [^6] 等可验证定理证明工具的发展。

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2503.03205/assets/x2.png)

**图 2**: MA-LoT 框架：(a) 多智能体 Lean4 定理证明框架：LoT-Solver 模型作为证明者智能体生成初始 Lean4 证明，并在长链式思考中为 Lean 证明提供涌现的自然语言规划（橙色块）；然后它作为纠正者智能体在长链式思考中分析来自 Lean 执行器的错误，输出改进的证明。(b) LoT 迁移学习（TL）：这一新颖的训练-推理流程使形式化推理能力在长链式思考（L-CoT）中涌现，无需专门标注的数据。这是通过调整系统提示来控制训练和推理中长链式思考的开关来实现的。(c) 训练流程：我们使用常规监督微调（SFT）训练通用自然语言长链式思考，使用 LoT-TL 在 Lean 监督微调和纠错数据上训练，使 Lean 长链式思考能力在 LLM 中涌现。

然而，在形式语言中编写数学证明需要大量的专业知识和努力，因为大多数证明涉及大量的重复和低资源函数的应用 [^9]。随着大语言模型的快速发展，研究开始探索 LLM 在形式语言推理中的应用，以自动化定理证明 [^19] [^20] [^8] [^9] [^37] [^33] [^27] [^29] [^10] [^11]。先前的研究遵循两种主要方法，即树搜索 [^8] [^9] [^11] [^33] [^27] 和完整证明生成 [^19] [^20] [^37] [^29] [^10]。这两种方法的优缺点总结可见图 1。

树搜索方法训练 LLM 智能体根据当前证明状态预测下一个策略（tactic），从而迭代生成证明步骤。这可以通过直接代码编写 [^19] [^20] [^33] [^29] [^11] 或基于检索的技术 [^37] [^10] 来实现。这些方法在每个生成步骤后应用形式语言执行器进行验证，能够发现一些非平凡的证明。然而，随着证明复杂性的增加，树搜索方法在计算上变得昂贵，并且缺乏高层自然语言规划来控制证明的整体结构。这导致树搜索方法无法找到一些需要对问题的自然语言含义进行高层分析的结构化证明。

相比之下，完整证明生成将定理证明视为代码生成问题，LLM 使用监督训练 [^27] [^33] 或提示工程 [^8] [^9] 在单次尝试中生成整个证明。这种方法利用了 LLM 的自然语言推理和高层规划能力，计算成本可预测，但它缺少来自形式语言执行器的中间反馈。因此，生成的证明往往缺乏对错误的事后分析，并且在需要非平凡解决方案的繁琐问题上表现不佳。总之，现有的单智能体（或模型）方法难以平衡自然语言推理与形式语言的可验证性约束，这促使需要一个更全面的框架。

为了应对上述挑战，我们引入了 MA-LoT：基于多智能体 Lean 的长链式思考框架，（据我们所知）这是首个用于 Lean4 定理证明的多智能体框架。MA-LoT 的多智能体推理方法如图 2 (a) 所示。MA-LoT 框架通过采用协作智能体框架和长链式思考中涌现的形式化思维能力，能够找到结构良好和非平凡的证明。具体而言，多智能体框架包含一个证明者智能体来编写结构良好的证明，以及一个纠正者智能体来分析来自 Lean 执行器的错误信息。长链式思考（CoT）引导模型在输出之前进行全面思考。我们实现了智能体类型控制和长链式思考中的高层自然语言规划。此外，我们还通过在长链式思考中集成 Lean 验证结果来提高自我反思能力。这两个系统协同工作，通过全面的证明规划和系统的错误分析实现更高的形式化推理能力。

为了支持我们设计的 MA-LoT 框架，我们开发了新颖的 LoT 迁移学习（LoT-Transfer Learning, TL）训练-推理流程来训练 LoT-Solver 模型，如图 2 (b) 和 (c) 所示。通过这个流程，形式化推理能力在长链式思考中涌现，无需专门标注的数据。这是通过利用迁移学习将自然语言长链式思考推理、定理证明监督微调数据和纠错数据的能力整合在一起来实现的。这种结构化适应使模型能够感知形式状态和策略，同时保持强大的自然语言规划能力，从而生成更连贯和更具洞察力的形式化证明。

广泛的实验表明，MA-LoT 框架通过多智能体设计和长链式思考中涌现的形式化推理有效增强了模型的形式化推理能力。该框架能够成功证明 MiniF2F 数据集 [^39] 中一些高级的 IMO 和 AIME 问题，而现有模型难以解决这些问题。在相似的采样预算下，我们的框架达到了 61.07% 的准确率，超越了最先进的完整证明生成模型（Godel-Prover [^12]，55.33%）和树搜索基线（InternLM-Step-Prover [^29]，50.70%）。

我们将贡献总结如下：(1) 我们引入了 MA-LoT，（据我们所知）这是首个在长链式思考范式下平衡自然语言推理和形式语言验证的多智能体框架，用于 Lean4 定理证明。(2) 我们提出了使用长链式思考来协同结合自然语言和形式语言特性的方法，使模型能够通过自然语言规划和分析生成深入且富有洞察力的形式化定理证明。(3) 我们开发了 LoT-TL，一个训练-推理流程，使领域特定的长链式思考能力在 LLM 中涌现，无需明确标注的数据集。

我们的框架在 Lean4 定理证明之外具有广泛的潜力，展示了形式化验证如何有效地与长链式思考推理集成。这种方法展示了通过形式化执行器上的迭代规划和错误分析实现结构化、反思性和适应性通用文本生成的潜力。为了加速该领域的进步，我们计划在不久的将来开源我们的代码、数据集和模型。

## 2 方法论

在本节中，我们详细介绍 MA-LoT 框架的开发和用于 Lean4 定理证明的 LoT-Solver 模型的训练流程。我们的框架旨在使形式化推理能力在长链式思考中涌现，以实现自然语言（NL）和形式语言（FL）之间的深度集成。这是在整个领域面临自然语言-形式语言对齐数据极度稀缺的情况下完成的 [^27]。我们首先在 2.1 节概述 LLM 形式化定理证明的基础知识。然后，我们在 2.2 节描述 LoT 迁移学习（LoT-TL）训练流程。最后，我们在 2.3 节详细介绍我们训练的模型如何促进 MA-LoT 框架进行 Lean4 证明编写。

### 2.1 基础知识

我们介绍一些将多智能体长链式思考（CoT）LLM 应用于 Lean4 形式化定理证明的基础知识。

当前最先进的方法将 Lean4 代码视为纯文本并将其输入 LLM。一些工作 [^37] [^29] 应用 LLM 作为智能体根据当前证明状态执行树搜索。它们将定理陈述和证明状态（包括前提和目标）转换为 LLM 的文本输入，并要求它生成可能的下一个策略。其他工作 [^33] [^27] 将定理证明视为代码生成任务，并开发了完整证明生成方法。这种方法为 LLM 提供自然语言指令、自然语言定理陈述和 Lean4 形式化陈述；预期结果是在单次传递中生成完整的 Lean4 证明，这利用了 LLM 的自然语言推理能力，让它在证明过程中编写自然语言计划。树搜索和完整证明生成的输入-输出示例在附录 B 中给出。

长链式思考 LLM，以 O1 [^17] 为代表，在输出最终答案之前进行长时间的内部自然语言思考。它通过长链式思考中的自我反思和纠正大大增强了 LLM 的自然语言数学推理能力。然而，它仍然难以提供严格的自然语言证明，并且通常具有相对较低的形式语言能力。

我们的方法通过采用多智能体系统整合了树搜索和完整证明生成的优势。我们应用长链式思考来调节 LLM 中的自然语言推理与 Lean4 验证器反馈之间的交互，使模型能够提供更结构化和更具洞察力的证明。

### 2.2 LoT-TL 训练流程

本节介绍一个简单但高效的训练流程——LoT 迁移学习（TL），它使 Lean4 领域特定的长链式思考能力在 LLM 中涌现，无需 Lean4 长链式思考数据。该策略利用系统提示来调节训练和推理行为；它可以分为三个阶段：(1) 收集领域特定的监督微调（Supervised Fine-Tuning, SFT）数据（2.2.1 节），(2) 在通用自然语言长链式思考任务上训练模型（2.2.2 节），以及 (3) 使用迁移学习方法在监督微调和纠错数据上训练模型，使形式化长链式思考能力涌现（2.2.3 节）。我们使用 DeepSeek-Prover-v1.5-SFT [^33] 作为基础模型。尽管我们专注于 Lean4，但我们的框架显示出扩展到在通用领域应用长链式思考推理的潜力，无需强化学习（RL）或特殊数据标注。

#### 2.2.1 获取监督微调数据

LoT-TL 流程的第一步是为特定目标领域（在我们的案例中是 Lean4）收集适量的自然语言-形式语言对齐的监督微调数据。然而，现有的开源数据集不符合要求。这些数据集存在以下问题：规模较小（如 MiniF2F [^39]）、缺少自然语言标注（如 DeepSeek-Prover-v1 数据集 [^32]）、自然语言质量较低（如 OBT [^27]）、或缺乏 Lean4 证明（如 Lean-Workbook [^38]）。

为了解决这个问题，我们编制了一个名为 LoT-ProveData（LoT-PD）的新 Lean 定理证明数据集，包含 54,465 条数据记录。每条记录包含 Lean4 定理陈述、带有自然语言解释作为注释的已验证证明以及自然语言陈述。Lean4 定理证明来自两个来源：DeepSeek-Prover-v1 数据集和使用 TheoremLlama 和 DeepSeek-Prover-v1.5-RL 标注的 Lean-Workbook。接下来，受 [^26] 中"先分析后生成"方法的启发，我们使用 Qwen-2.5-72B 提供 Lean4 证明的分析，然后基于分析编写自然语言证明。最后，我们通过 Qwen 将这些自然语言证明作为注释集成到 Lean4 代码中。对于缺少自然语言陈述的数据记录，我们使用类似的方法生成自然语言陈述。我们的 LoT-ProveData 的核心组成部分是：

{形式语言陈述，带注释的形式语言证明，自然语言陈述}

在为 ProverData 生成证明的过程中，也产生了一些不正确的证明。我们记录了这些证明及其错误信息，形成了 LoT-CorrectionData（LoT-CD），由 64,912 条正确-不正确 Lean4 证明记录以及相关的错误信息、自然语言陈述和证明组成。LoT-CD 用于训练模型的错误分析和纠正能力。LoT-CorrectionData 的核心部分是：

{形式语言陈述，正确的形式语言证明，错误信息，不正确的形式语言证明，自然语言陈述}

这些数据集共同提高了证明者和纠正者智能体的能力，作为我们多智能体框架强大的自然语言-形式语言联合思维能力的来源。

#### 2.2.2 自然语言长链式思考训练

在第二阶段，我们训练一个常规的指令微调模型，为通用自然语言任务获取长链式思考推理能力。我们使用 [^16] 提供的 OpenO1-SFT-Pro 数据集，这是一个包含 126k 条记录的数据集，用于数学和科学主题的通用自然语言问答，带有长链式思考进行训练。我们应用标准的下一个词元预测监督微调，引导模型在输出最终答案之前产生长链式思考。在整个自然语言长链式思考训练过程中，我们将系统提示设置如下，明确指示模型使用长链式思考方法：

```
You are a helpful assistant who will solve every problem **WITH** Long Chain-of-Thought
```

此提示有效地"打开"了长链式思考推理。训练输入包括系统提示和自然语言问题，预期输出是长链式思考和最终答案。训练后，我们观察到模型获得了强大的自然语言长链式思考能力，这为智能体分析和与 Lean 交互提供了基础。然而，当应用于 Lean4 推理时，它倾向于只提供自然语言解决方案，而不是在输出部分输出 Lean4 代码，这表明需要进一步对齐。

#### 2.2.3 领域特定对齐

在 LoT-TL 流程训练过程的最后阶段，我们训练模型使 Lean4 长链式思考能力涌现。这是在训练阶段通过切换到不同的系统提示来实现的，该提示指示模型不使用长链式思考，并为长链式思考内容使用占位符来保持长链式思考的骨架，同时不需要提供任何实际的 Lean4 长链式思考数据。具体而言，系统提示是：

```
You are a helpful assistant who will solve every problem **WITHOUT** Long Chain-of-Thought
```

占位符长链式思考是：

```
The user asks not to solve with Long CoT, so I will directly write the answer.
```

使用这个设置，我们首先在 LoT-ProveData 上训练形式化定理证明能力，然后在 LoT-CorrectionData 上训练，使模型学习错误分析和纠正技能。训练数据的示例可以在附录 E 中找到。我们还采用了 [^27] 中的课程学习数据排序方法来稳定训练。训练后，我们发现当在推理中使用系统提示打开长链式思考时，Lean4 证明和错误分析的长链式思考能力在 LLM 中涌现。我们认为迁移学习框架的有效性在于它保留了长链式思考的结构，并使模型能够在收到指令时激活这种能力。

遵循这些训练步骤，我们获得了 LoT-Solver，它具有从没有 Lean 长链式思考标注的数据中涌现的 Lean 推理能力。这使它能够执行完整证明证明者和树搜索纠正者的角色，并具有深度的自然语言-形式语言联合思维。

### 2.3 多智能体形式语言证明编写

本节介绍（据我们所知）首个在基于 Lean 的长链式思考范式下结合完整证明生成和树搜索方法优势的多智能体框架。我们使用 LoT-Solver 作为两个智能体的基础模型。在这个设置下，我们使用证明者智能体编写完整的证明草稿（2.3.1 节），并应用纠正者智能体根据 Lean 验证器反馈分析和纠正证明（2.3.2 节）。

#### 2.3.1 证明者智能体

证明者智能体使用完整证明生成策略编写初始 Lean4 证明，然后将其提交给 Lean4 验证器检查正确性，如果证明错误则传递给纠正者智能体。我们使用系统提示打开长链式思考推理，并在长链式思考中使用特定的标题来引导模型制定高层证明计划。以下是证明者智能体的指令模板：

```
{... **WITH** Long CoT...}
### Instruction:
{自然语言陈述}
{形式语言陈述}
### Response:
Alright, I should do the following:
1. Provide the natural language analysis for the theorem based on the Natural language theorem statement.
2. Draft the Lean4 tactics I should use to solve the problem
3. Write the output Lean4 code.
```

证明者智能体输入和输出的完整示例可以在附录 F 中找到。长链式思考中涌现的 Lean 推理能力使模型能够根据其高层计划编写结构更好的证明，相比于直接证明生成。生成证明后，证明者智能体将其提交给 Lean 评估器进行验证。如果不正确，则将定理传递给纠正者智能体进行进一步改进。

#### 2.3.2 纠正者智能体

纠正者智能体的功能类似于树搜索方法。在收到错误的证明和 Lean 验证器反馈后，它在长链式思考中系统地分析它们。然后，在重新评估证明策略和重新思考后，模型生成旨在解决错误的修订证明。

指令提示与证明者智能体保持相同。我们在长链式思考中加入不正确的证明和来自 Lean 验证器的反馈，然后是指令来引导模型分析错误并制定修订的证明。这些提示的详细示例可在附录 F 中找到。然后，纠正者智能体将新证明传递给 Lean4 验证器，如果新证明仍然错误，则迭代分析错误直到成功或达到最大重试限制。

纠正者智能体通过实现基于错误信息的更深入反思和对替代证明策略的系统探索来增强定理证明。这个迭代纠正过程在保持计算效率的同时增加了发现非平凡证明的可能性。

## 3 实验

我们在 MiniF2F-Lean4 [^39] 数据集上进行了全面的实验，以评估 MA-LoT 框架在形式化证明编写中的性能。具体而言，我们通过在相似采样预算下它能够找到的额外证明来评估我们的多智能体长链式思考系统在编写结构更好、更具洞察力的证明方面的优越性（3.3 节）。此外，我们对纠正者智能体（3.4 节）、LoT-Solver 的训练组件（3.5 节）和案例研究（3.6 节）进行了研究，以进一步分析各个组件的影响。

<table><tbody><tr><td>方法</td><td>模型大小</td><td>采样预算</td><td>MiniF2F-Valid</td><td>MiniF2F-Test</td><td>平均值</td></tr><tr><td colspan="6">树搜索方法</td></tr><tr><td>ReProver <sup><a href="#fn:37">37</a></sup></td><td>229M</td><td>-</td><td>-</td><td>26.5%</td><td>-</td></tr><tr><td>Llemma <sup><a href="#fn:2">2</a></sup></td><td>34B</td><td>$1\times 32\times 100$</td><td>27.9%</td><td>25.8%</td><td>26.85%</td></tr><tr><td>Expert Iteration <sup><a href="#fn:20">20</a></sup></td><td>837M</td><td>$8\times 8\times 512$</td><td>41.2%</td><td>36.6%</td><td>38.9%</td></tr><tr><td>Lean-STaR <sup><a href="#fn:11">11</a></sup></td><td>7B</td><td>$64\times 1\times 50$</td><td>-</td><td>46.3%</td><td>-</td></tr><tr><td>InternLM2.5-StepProver <sup><a href="#fn:29">29</a></sup></td><td>7B</td><td>$2\times 32\times 600$</td><td>56.0%</td><td>50.7%</td><td>53.35%</td></tr><tr><td colspan="6">完整证明生成</td></tr><tr><td>GPT-4-Turbo <sup><a href="#fn:1">1</a></sup></td><td>$>1$ T</td><td rowspan="6">pass@128</td><td>25.41%</td><td>22.95%</td><td>24.18%</td></tr><tr><td>DeepSeek-Math <sup><a href="#fn:24">24</a></sup></td><td>7B</td><td>25.80%</td><td>24.60%</td><td>25.20%</td></tr><tr><td>Gemini-1.5-pro <sup><a href="#fn:23">23</a></sup></td><td>-</td><td>29.92%</td><td>27.87%</td><td>28.90%</td></tr><tr><td>TheoremLlama <sup><a href="#fn:27">27</a></sup></td><td>8B</td><td>38.52%</td><td>35.66%</td><td>37.66%</td></tr><tr><td>DeepSeek-Prover-v1.5-RL <sup><a href="#fn:33">33</a></sup></td><td>7B</td><td>54.10%</td><td>48.36%</td><td>51.23%</td></tr><tr><td>STP-Lean <sup><a href="#fn:4">4</a></sup></td><td>7B</td><td>-</td><td>56.15%</td><td>-</td></tr><tr><td>Godel-Prover <sup><a href="#fn:12">12</a></sup></td><td>7B</td><td>pass@32</td><td>-</td><td>55.33%</td><td>-</td></tr><tr><td colspan="6">我们的方法</td></tr><tr><td>DeepSeek-Prover + LoT (完整证明)</td><td rowspan="5">7B</td><td>pass@128</td><td>62.70%</td><td>52.05%</td><td>57.42%</td></tr><tr><td>DeepSeek-Prover + MA-LoT</td><td>$64+32\times 2$</td><td>64.34%</td><td>54.51%</td><td>59.22%</td></tr><tr><td>Godel-Prover + LoT (完整证明)</td><td>pass@32</td><td>-</td><td>57.79%</td><td>-</td></tr><tr><td>Godel-Prover + MA-LoT</td><td>$16+8\times 2$</td><td>-</td><td>61.07%</td><td>-</td></tr><tr><td>MA-LoT</td><td>累积</td><td>65.98%</td><td>63.93%</td><td>64.96%</td></tr></tbody></table>

**表 1**: MA-LoT 的主要实验结果。我们的结果以"基础模型 + 方法"的形式呈现。LoT（完整证明）表示使用我们的 LoT-Solver 模型的完整证明结果，MA-LoT 表示我们整个流程的结果。MA-LoT 框架中的采样预算 $x+k\times y$ 表示我们首先使用证明者智能体执行 $x$ 次完整证明编写，然后使用纠正者智能体进行 $k$ 轮纠正。实际上，一轮纠正的成本约为完整证明生成预算的一半，因此 $y=\frac{1}{2}x$。

### 3.1 实验设置

#### 3.1.1 数据集和任务

在本文中，我们在 MiniF2F-Test 和 Valid<sup>1</sup> 数据集 [^39] [^37] [^27] 上评估 MA-LoT 的 Lean4 推理能力。MiniF2F 是一个广泛使用且具有挑战性的形式化定理证明基准 [^5]，几乎所有该领域的主要研究都采用了它 [^8] [^20] [^9] [^29] [^11] [^37] [^33] [^27] [^2]。

测试集和验证集各包含 244 个 Lean4 陈述。问题范围从高中竞赛题目到本科级别的定理证明。它包含来自三个来源的 488 个问题：(1) 从 MATH 数据集 [^7] 中采样的 260 个问题；(2) 来自高中数学竞赛（包括 AMC、AIME 和 AMO）的 160 个问题；(3) 68 个手工制作的与 (2) 难度级别相同的问题。我们的任务是查询 LLM 根据 MiniF2F 问题的形式化陈述和自然语言描述生成 Lean4 证明。为了最小化计算开销，导入是手动配置的。

#### 3.1.2 基线

为了突出 MA-LoT 的能力，我们选择了近年来一些最具竞争力的基线，涵盖树搜索和完整证明生成方法。对于树搜索方法，我们包括：Expert Iteration [^20]、Llemma [^2]、ReProver [^37]、Lean-STaR [^11] 和 InternLM2.5-StepProver [^29] 作为我们的基线。对于完整证明生成基线，我们包括闭源 LLM，代表是 GPT-4-Turbo [^1] 和 Gemini-1.5 [^23]，以及开源专家模型，如 DeepSeek-Math [^24]、TheoremLlama [^27]、DeepSeek-Prover-v1.5-RL [^33]、STP-Lean [^4] 和 Godel-Prover-SFT [^12]。

对于完整证明生成基线，我们将采样预算设置为 pass@128，上下文长度为 4096（Godel-Prover 除外，按照原始论文，我们使用 pass@32），在稳健性和可管理的 GPU 消耗之间取得平衡。对于树搜索方法，我们尽可能将搜索成本与完整证明生成对齐。<sup>2</sup>

### 3.2 实现细节

在模型的训练过程中，我们使用 Openo1-SFT-Pro、LoT-ProverData 和 LoT-CorrectionData 训练两个基础模型，即 DeepSeek-Prover-v1.5-SFT 和 Godel-Prover-SFT。对于不同的训练阶段，学习率如下：自然语言长链式思考训练为 1E-5，LoT-PD 上的 LoT-TL 为 1E-7，LoT-CD 为 1E-6。训练的总计算成本约为 1 GPU 天，在 $4\times$ H100-96G 集群上评估为 11 GPU 天。<sup>3</sup> 为了详细评估我们的框架，我们呈现三组结果，包括：(1) LoT（完整证明）：LoT-Solver 的 pass@128 完整证明生成结果。(2) MA-LoT：我们的主要评估结果，其中证明者执行 64 次完整证明生成并经历两轮纠正者改进。<sup>4</sup> (3) 累积结果：汇总实验过程中获得的所有 LoT 模型输出的组合评估。

我们的代码、模型和数据将发布在 [https://github.com/RickySkywalker/LeanOfThought-Official](https://github.com/RickySkywalker/LeanOfThought-Official)

### 3.3 结果

表 1 呈现了我们的主要结果，显示 MA-LoT 在 MiniF2F-Test 基准上达到了 61.07% 的准确率，LoT-Solver 使用完整证明生成达到了 57.79%。在 MA-LoT 的增强下，我们在 MiniF2F-Test 基准上相比 Godel-Prover 提高了 10.37%，相比 DeepSeek-Prover 提高了 12.72%。这种显著且一致的改进表明了我们框架的有效性。详细分析表明，我们的模型成功解决了以往模型难以处理的 IMO 和 AIME 问题。我们的模型超越了最先进的树搜索（InternLM-2.5）和完整证明生成（Godel-Prover）基线，表明我们提出的基于长链式思考的多智能体框架在形式化定理证明中表现出色。

MA-LoT 超越所有树搜索基线至少 20.45%，因为其证明者智能体使用涌现的 Lean 长链式思考推理能力通过高层自然语言规划构建证明。这表明我们的证明者智能体可以利用 LLM 强大的自然语言推理能力，从而产生更全面的证明。此外，MA-LoT 超越完整证明生成基线至少 10.37%，因为其纠正者智能体在长链式思考中基于 Lean4 执行器反馈分析、反思和重新制定证明。强大的性能也证明了我们将形式语言验证集成到自然语言长链式思考推理中的想法及其涌现能力的有效性。值得注意的是，DeepSeek-Prover + MA-LoT 基于 DeepSeek-Prover-v1.5-SFT；它超越了其强化学习训练的变体 6.15%。这表明我们的多智能体训练框架和基于 Lean 的长链式思考方法比单独的强化学习更自然地与形式化定理证明对齐。

| 方法 | 证明者 | 第1轮 | 第2轮 | 第3轮 |
| --- | --- | --- | --- | --- |
| DS-Prover-v1.5 | 51.64% | 53.28% | 54.51% | 55.33% |
| Godel | 54.92% | 59.43% | 61.07% | 61.89% |

**表 2**: 使用 DeepSeek-Prover 和 Godel-Prover 作为基础模型进行不同轮次纠正者智能体纠正的结果

| 方法 | MiniF2F-Test |
| --- | --- |
| DeepSeek-Prover-v1.5-SFT（基础模型） | 46.31% |
| LoT-Solver 关闭长链式思考 | 49.18% |
| 无长链式思考训练（在强化学习模型上） | 48.36% |
| 基础模型 + 长链式思考 | 46.72% |
| 基础模型 + 长链式思考 + 监督微调 | 50.00% |
| LoT-Solver | 51.64% |

**表 3**: 在 DeepSeek-Prover 作为基础模型的 pass@64 下的消融研究结果

LoT（完整证明）和 MA-LoT 之间的比较进一步突出了我们多智能体框架的重要性。我们观察到，通过将计算资源从证明者智能体进行更多完整证明生成重新分配到纠正者智能体基于 Lean 执行器反馈分析和改进证明，获得了 2.46% 的改进。这验证了迭代证明者-纠正者多智能体系统的必要性，而不是仅依赖证明者智能体。总之，这些结果证实了长链式思考推理与形式化验证和多智能体范式相结合，增强了非平凡和深入证明的发现，从而验证了我们提出的方法的有效性。

### 3.4 纠正者智能体研究

为了评估纠正者智能体在 MA-LoT 中的影响，我们在表 2 中呈现了两个基础模型在不同纠正轮次的 MiniF2F-Test 累积准确率。证明者列呈现了证明者智能体在完整证明生成中的 pass@64（或 16）准确率，而第 $i$ 轮列表示连续的纠正轮次。结果表明，在前三轮纠正中，纠正者智能体成功纠正了证明者无法回答的平均 11.55% 的定理。我们的分析显示，大多数纠正的证明属于 IMO、AIME 和高级 MATH 问题，这些问题对于先前的模型特别困难。这突出了纠正者智能体使用长链式思考中涌现的 Lean 能力分析来自 Lean4 执行器反馈的能力，以发现非平凡证明。分析和重新生成新证明的案例研究可以在 3.6 节和附录 D 中找到。

### 3.5 消融研究

为了评估我们 LoT-TL 训练流程每个组件的有效性，我们进行了这项全面的消融研究。我们证明 LoT-TL 流程中的元素通过在长链式思考中集成形式语言协同工作，以增强模型的形式化定理证明能力。我们使用 DeepSeek-Prover-v1.5-SFT 作为基础模型，并对这组实验应用完整证明生成方法的 pass@64 准确率。结果如表 3 所示。

#### 3.5.1 训练阶段的影响

我们通过测量关键中间模型的性能来评估训练进展，即基础模型、基础模型 + 长链式思考和基础模型 + 长链式思考 + 监督微调，如表 3 所示。结果表明，仅在自然语言 CoT 数据上训练提供的改进很小，表明单独的自然语言 CoT 推理不会使 Lean CoT 能力涌现。然而，使用 LoT-TL 训练方法结合监督微调数据产生了显著的改进，证明了迁移学习在为模型配备 Lean4 长链式思考能力方面的有效性。有趣的是，使用纠错数据进行额外训练，尽管它不是为完整证明编写设计的，但进一步增强了性能。这种改进可能源于模型发展了 Lean4 代码的自我分析能力，使其能够避免潜在的错误解决方案。

#### 3.5.2 关闭长链式思考

这个实验表明，LoT-Solver 强大的形式语言推理能力来自长链式思考中涌现的形式化推理能力，而不是简单地堆叠更多数据。它使用我们的 LoT-Solver 模型使用代码补全方法直接编写 Lean4 证明，不使用长链式思考。我们发现性能从 51.64% 下降到 49.18%，这是因为模型没有在长链式思考中采取明确的高层计划，使其无法完成归纳领域的一些问题。

#### 3.5.3 长链式思考的消融

为了验证长链式思考中自然语言和形式语言集成的质量，我们直接使用我们的监督微调数据集对 DeepSeek-Prover-v1.5-RL 模型进行微调，不使用长链式思考推理。我们可以发现，无长链式思考模型的性能（48.36%）低于 LoT-Solver（51.64%）。这证实了长链式思考在 Lean4 定理证明中起着关键作用，提供的结构化推理优于直接基于强化学习的额外数据微调模型。

### 3.6 案例研究

本节呈现 MA-LoT 框架案例研究的一般结果。由于篇幅有限，我们在附录 D 中留下了详细示例。结果显示了智能体之间的协作。证明者智能体可以使用高层计划来证明高级 MATH 定理，纠正者可以分析来自 Lean 执行器的反馈来制定 IMO 级别问题的正确证明。长链式思考中的内容也展示了从我们的 LoT-TL 训练流程中涌现的形式化推理能力，这些能力存在于证明者和纠正者智能体中。这些观察在质量上验证了多智能体系统的设计和长链式思考中形式化推理能力的涌现，展示了其在基于 Lean 的长链式思考中结合高层规划与迭代改进的能力。

## 4 相关工作

### 4.1 使用 LLM 的 Lean4 定理证明

LLM 在形式语言证明中的应用近年来一直是研究的热点。树搜索方向包括以 Expert Iteration [^20]、ReProver [^37]、Lean-Star [^11] 和 InternLM-Step-Prover [^30] 为代表的工作。这个方向没有充分考虑 LLM 的自然语言推理能力，并且计算能力消耗呈指数增长。另一个方向将形式语言视为代码，并要求 LLM 进行完整证明生成，不与 Lean 执行器交互，以充分利用 LLM 的自然语言推理能力。重要的工作包括 DeepSeek-Prover [^33] [^32]、TheoremLlama [^27] 和 Llemma [^2]。这个方向的工作往往忽视了来自 Lean 执行器的验证信号，或者没有对错误信息进行全面思考。

### 4.2 基于智能体的 LLM

传统的强化学习方法为通用推理和决策过程提供了训练方法解决方案，但通常存在样本效率低和泛化问题 [^21]。随着 LLM 推理和指令遵循能力的快速发展，许多研究人员开始将 LLM 用作智能体 [^25] [^13]。使用 LLM 作为智能体的主要方法是设计特殊提示和上下文示例，让 LLM 使用可操作的响应与外部工具交互 [^31] [^35]。进一步的努力是应用专门的训练来增强它们的智能体能力 [^34] [^22]。在形式化推理的背景下，大多数树搜索方法 [^37] [^11] [^29] 应用 LLM 作为智能体查询执行器并接收反馈以进一步改进证明。由于形式语言和自然语言之间的巨大差异，这些方法无法为问题提供高层分析并提供结构化响应。

## 5 结论

本文介绍了 MA-LoT，（据我们所知）首个用于形式化定理证明的基于多智能体 Lean 的长链式思考框架。我们的方法解决了单智能体系统的局限性，这些系统要么未充分利用 LLM 的自然语言推理和规划能力，要么未能有效集成形式化验证反馈。通过长链式思考构建证明者和纠正者智能体之间的交互，MA-LoT 在证明生成中实现了更深入的洞察和长期连贯性。为了支持这个框架，我们提出了 LoT-TL，一个使形式化推理长链式思考能力在 LLM 中涌现的训练流程，无需标注的 Lean 长链式思考数据。通过在 MiniF2F-Test 基准上的广泛实验，MA-LoT 达到了 61.07% 的准确率，超越了所有基线，包括树搜索和完整证明生成方法。这些结果强调了将形式化验证与结构化推理相结合以实现更好的 Lean4 定理证明能力的优势。除了定理证明之外，LoT-TL 训练流程展示了将长链式思考技术应用于领域特定任务的潜在方法，无需专门的标注。此外，多智能体长链式思考在 Lean4 中的成功表明形式化验证在增强跨不同领域的结构化推理方面有更广泛的应用。

## 影响声明

本文呈现的工作旨在推进使用 LLM 的形式化定理证明。潜在的社会影响主要在教育领域。随着越来越多的形式语言在研究生级别的教育中使用，更先进的形式化定理证明模型可能导致教育工作者无法区分模型生成的结果和学生编写的结果。尽管改进形式化推理系统有社会后果，但在现阶段讨论伦理问题的具体细节还为时过早。

## 参考文献

## 附录 A 术语表

为了让读者更好地理解术语，我们提供这个表格，详细解释每个术语、缩写和相应的工具。

1. **NL（自然语言，Natural Language）**：指人类在日常生活中使用的语言，通常无法执行自动验证。
2. **FL（形式语言，Formal Language）**：逻辑和证明的结构化和数学上精确的表示，确保严格验证并消除自然语言推理中存在的歧义。
3. **Lean4**：一种函数式编程语言和交互式定理证明器，用于形式化数学和验证证明。
4. **Lean 执行器（Lean Executor）**：Lean4 的内置证明验证引擎。它评估证明步骤，检查正确性，并确保每个逻辑推理都遵循严格的形式化验证规则。
5. **长链式思考（Long CoT, Long Chain-of-Thought）**：由 OpenAI-O1 [^17] 提供的推理结构，在做出最终输出之前进行长时间和详细的思考。与标准 CoT 不同，长 CoT 允许在证明生成、反思和来自 Lean4 反馈的自我检查的迭代改进之前进行多步逻辑推理。
6. **MA-LoT（多智能体基于 Lean 的长链式思考框架，Multi-Agent Lean-based Long Chain-of-Thought framework）**：我们提出的用于形式化定理证明的多智能体框架。
7. **LoT-TL（LoT 迁移学习，LoT-Transfer Learning）**：我们提出的迁移学习流程，使 LLM 具有 Lean4 长链式思考能力，无需专门标注的数据集。
8. **LoT-Solver**：我们通过 LoT-TL 流程训练的模型，既作为证明者又作为纠正者智能体。它是一个具有长链式思考能力的高标准 Lean4 定理证明模型，可以控制智能体类型并增强形式化思维。
9. **LoT-PD 和 LoT-CD（LoT-ProveData 和 LoT-CorrectionData）**：LoT-PD 是一个包含已验证的 Lean4 定理证明和自然语言陈述和证明的数据集。其主要功能是为证明者智能体提供基本能力。LoT-CD 是一个包含不正确的错误信息和正确的 Lean4 证明以及自然语言标注的数据集。它主要为证明者智能体提供改进证明的能力。

## 附录 B 树搜索和完整证明生成的示例

根据 2.1 节，我们在这里呈现树搜索方法和完整证明生成方法的输入-输出示例。

**完整证明生成示例：**

### 输入示例

补全以下 Lean 4 代码：

```lean4
theorem algebra_sqineq_unitcircatbpamblt1
  (a b: ℝ)
  (h₀: a^2 + b^2 = 1):
  a * b + (a - b) ≤ 1 := by
```

### 输出示例

```lean4
  -- 我们有 (a - b - 1)^2 ≥ 0。
  have h₁: 0 ≤ (a - b - 1) ^ 2 := sq_nonneg _
  -- 通过展开，我们有：
  -- 0 ≤ a^2 -ab-a-ab+b^2 +b-a+b+1。
  linarith [h₀, sub_add_cancel a b]
```

**树搜索方法示例：**

### 输入示例

```
DECL MyNat.mul_pow
GOAL a b n: N
⊢ (a * b) ^ n = a ^ n * b ^ n
```

### 输出示例

```
PROOFSTEP induction n with t Ht
```

<!-- PLACEHOLDER_APPENDIX_C -->

## 附录 C 扩展规律研究

| 训练步数 | MiniF2F-Test |
| --- | --- |
| 125 步 | 32.79% |
| 250 步 | 35.25% |
| 1,250 步 | 45.08% |
| 2,500 步 | 47.54% |
| 13,616 步（完整训练） | 51.64% |

**表 4**: 不同数据规模训练模型在 MiniF2F-Test 上的结果

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2503.03205/assets/x3.png)

**图 3**: 对数尺度训练步数与训练模型在 MiniF2F 上的准确率。虚线是线性回归近似线

我们进行了额外的实验来研究我们的模型是否遵循扩展规律，通过评估其在不同训练数据集规模下的性能。具体而言，我们在固定批量大小为 4 的情况下使用四个不同的步长（125、250、1250 和 2500 步）训练模型，并使用长链式思考（CoT）格式评估其在 MiniF2F-Test 基准上的准确率。结果准确率如表 4 所示。为了分析这些结果，我们在图 3 中绘制了对数尺度训练步数与模型准确率之间的关系。数据的线性近似产生了强大的决定系数（$R^{2}=0.9664$），表明随着训练步数的对数增加，性能呈近线性改进。这与扩展规律的预测一致，表明我们的方法从更大规模的训练中显著受益。我们的发现进一步强调了通过扩展资源来标注额外 Lean4 数据以增强性能的潜力。


## 附录 D 案例研究

我们在附录 D.1 和 D.2 中呈现两个案例研究。我们可以看到，多智能体框架能够通过在证明者智能体中制定高层自然语言计划并执行该计划来编写更连贯和结构良好的证明。纠正者智能体能够分析未解决的目标，制定纠正计划并编写正确的证明。这是通过我们提出的训练流程实现的。

### D.1 证明者分析

这是证明者智能体生成结果的示例，我们可以看到智能体学会在长链式思考中编写高层证明计划，并在编写 Lean4 代码时详细执行该计划。

**提供的长链式思考标题**

```
<Thought>
Alright, I should do the following: 
1. Provide the natural language analysis for the theorem based on the Natural language theorem statement. 
2. Draft the Lean4 tactics I should use to solve the problem 
3. Write the output Lean4 code.

The user also asks that I should avoid using the keyword `sorry` to give up the proof, so I will not write it in my Lean4 code.
```

**生成的结果：**

```
The `mathd_numbertheory_458` can be proofed by leveraging the properties of modular arithmetic. Specifically, we know that if `n % 8 = 7`, then `n` can be expressed as `n = 8k + 7` for some integer `k`. We need to find `n % 4`.

First, observe that `n % 8 = 7` implies `n = 8k + 7`. Since `8k` is divisible by `4`, we only need to consider the remainder when `7` is divided by `4`, which is `3`. Therefore, `n % 4 = 3`.

Now, translating this reasoning into Lean4 code, we can use the properties of modular arithmetic to directly derive the result.
</Thought>

<Output>
```lean4
theorem mathd_numbertheory_458 (n: ℕ) (h₀: n % 8 = 7): n % 4 = 3 := by
  -- 我们知道 n 除以 8 的余数是 7。
  have h₁: n % 8 = 7 := h₀
  -- 我们需要找到 n 除以 4 的余数。
  have h₂: n % 4 = 3 := by
    -- 使用 4 整除 8 的事实，我们可以使用模运算的性质。
    rw [← Nat.mod_mod_of_dvd n (by decide: 4 ∣ 8)]
    -- 使用除以 8 时的给定余数简化表达式。
    simp [h₁]
  -- 因此，n 除以 4 的余数是 3。
  exact h₂
```
</Output>
```

### D.2 纠正者分析

这是成功解决 IMO 问题的错误纠正示例，我们可以看到模型首先分析错误信息，然后制定证明计划，然后编写正确的证明来解决另一个 IMO 级别的问题。

**提供的长链式思考标题**

```
<Thought>
Alright, I need to prove the theorem prover_data_68493 using the Lean4 code. Here is my draft of the proof:

```lean4
theorem imo_1983_p6 (a b c: ℝ) (h₀: 0 < a ∧ 0 < b ∧ 0 < c) (h₁: c < a + b) (h₂: b < a + c) (h₃: a < b + c): 
  0 ≤ a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) := by
  /- 为了证明对于满足与三角形相关的某些条件的正实数 a、b 和 c，
     a^2 b(a - b) + b^2 c(b - c) + c^2 a(c - a) ≥ 0，我们可以如下进行：
     1. **非线性算术**：利用非线性算术（`nlinarith`）来验证涉及变量之间差的平方和乘积的不等式。
        这些操作本质上处理由于消去或符号变化而可能不成立的情况。
        如果 a、b 和 c 之间的所有成对差同时变为零，则等式成立。
        然而，由于它们必须满足由 `triangle` 施加的严格正性约束，
        除非另有说明，否则在正常情况下不会发生这种情况。
        尽管如此，在没有超出提供的约束的其他约束的代数操作范围内，
        关于导致等式的潜在零没有立即出现矛盾。
  -/
  -- 引入中间假设，确保可分性属性不会破坏预期行为。
  have ha': True := trivial
  have hb': True := trivial
  have hc': True := trivial
  -- 使用非线性算术迭代检查涉及我们主要变量和常量的和与乘积的不等式。
  repeat' nlinarith [mul_pos h₀.1 h₀.2.1, mul_pos h₀.2.1 h₀.2.2, mul_pos h₀.2.2 h₀.1, 
    sub_self a, sub_self b, sub_self c, add_sub_cancel'_right a b, add_sub_cancel'_right b c, 
    add_sub_cancel'_right c a, mul_nonneg (ha'.le) (hb'.le), mul_nonneg (hb'.le) (hc'.le), 
    mul_nonneg (hc'.le) (ha'.le)]
```

让我在 Lean4 中测试它
嗯，看起来上面的证明是错误的。
让我检查错误信息。
好的，这是错误信息：

```bash
line 13
unsolved goals
a b c: ℝ
h₀: 0 < a ∧ 0 < b ∧ 0 < c
h₁: c < a + b
h₂: b < a + c
h₃: a < b + c
ha' hb' hc': True
⊢ 0 ≤ a ^ 2 * b * (a - b) + b ^ 2 * c * (b - c) + c ^ 2 * a * (c - a)
```

因此，我将根据以下步骤重新思考 Lean4 证明：
1. 基于自然语言定理陈述、Lean4 定理陈述、我之前的证明和错误信息为定理提供自然语言分析。
2. 起草我应该用来解决问题的 Lean4 策略
3. 编写输出 Lean4 代码。

让我通过错误信息分析错误的 Lean4 解决方案。
```

**生成的结果**

```
错误信息表明证明不完整。
错误信息说：

```
unsolved goals
a b c: ℝ
h₀: 0 < a ∧ 0 < b ∧ 0 < c
h₁: c < a + b
h₂: b < a + c
h₃: a < b + c
ha' hb' hc': True
⊢ 0 ≤ a ^ 2 * b * (a - b) + b ^ 2 * c * (b - c) + c ^ 2 * a * (c - a)
```

因此，主要问题是证明不完整，最终目标 `0 ≤ a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a)` 没有得到适当处理。

因此，我需要构建一个适当的证明策略来正确处理所有条件和目标。

首先，我需要理解不等式 `a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a)` 的数学结构。

这个表达式在 `a`、`b` 和 `c` 中是对称的。

考虑到 `a`、`b` 和 `c` 是正实数，并且它们满足与三角形相关的某些条件（三角不等式），我可以考虑这些表达式的性质。

一种方法是考虑 `a`、`b` 和 `c` 的对称性和正性。

由于 `a`、`b` 和 `c` 是三角形的边，它们满足三角不等式：
- `a + b > c`
- `b + c > a`
- `c + a > b`

此外，由于 `a`、`b` 和 `c` 是正数，我们可以使用实数和不等式的性质来证明期望的结果。

一种方法是考虑使用 AM-GM 不等式或柯西-施瓦茨不等式等代数操作或不等式。

考虑到表达式的复杂性，也许重新排列项或应用已知不等式可以有所帮助。

另外，考虑三角形的条件，也许用彼此表达变量或使用替换可以简化问题。

我需要确保最终证明在逻辑上是合理的，并且不依赖于未证明的假设。

因此，我将根据这些考虑构建修订的证明。

让我继续思考过程并输出修正的 Lean4 代码。
</Thought>

<Output>
```lean4
/-- 设 $a$、$b$ 和 $c$ 是三角形的边长。证明：
$a^2 b(a - b) + b^2 c(b - c) + c^2 a(c - a) \geq 0$。-/
theorem imo_1983_p6 (a b c: ℝ) (h₀: 0 < a ∧ 0 < b ∧ 0 < c) (h₁: c < a + b) (h₂: b < a + c) (h₃: a < b + c): 
  0 ≤ a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) := by
  -- 我们将使用任何实数的平方都是非负的这一事实。
  have h₄: 0 ≤ (a - b)^2 := sq_nonneg (a - b)
  have h₅: 0 ≤ (b - c)^2 := sq_nonneg (b - c)
  have h₆: 0 ≤ (c - a)^2 := sq_nonneg (c - a)
  -- 使用这些不等式，我们可以使用非线性算术得出期望的不等式。
  nlinarith [mul_pos h₀.1 h₀.2.1, mul_pos h₀.2.1 h₀.2.2, mul_pos h₀.2.2 h₀.1, 
    mul_pos (sub_pos.mpr h₁) (sub_pos.mpr h₂), mul_pos (sub_pos.mpr h₂) (sub_pos.mpr h₃), 
    mul_pos (sub_pos.mpr h₃) (sub_pos.mpr h₁)]
```
</Output>
```


## 附录 E 训练示例

本节我们呈现监督微调数据和纠错数据的训练示例。我们可以看到，在 LoT-TL 训练阶段，我们保持系统提示关闭长链式思考。特别是，在监督微调数据中，我们关闭整个长链式思考；在纠错数据训练中，我们只关闭长链式思考中对错误信息的分析，同时将错误信息以及原始证明草稿作为长链式思考中的一个组件提供。

### E.1 监督微调数据的 LoT-TL 训练示例

**输入示例**

```
<|begin_of_sentence|>You are a helpful assistant who will solve every problem **WITHOUT** Long Chain-of-Thought.

### Instruction:
@ Natural language theorem statement: 
给定非负实数 $a, b, c$，满足 $a + b + c = 1$。证明：$7(ab + bc + ca) \leq 2 + 9abc$。

@ Lean4 theorem statement:
```lean4
theorem prover_data_2216 (a b c: ℝ) (ha: 0 ≤ a) (hb: 0 ≤ b) (hc: 0 ≤ c) (habc: a + b + c = 1): 
  7 * (a * b + b * c + c * a) ≤ 2 + 9 * a * b * c := by
```

@ Lean4 定理陈述和带解释性注释的证明（每行代码前有注释）：

### Response:
```

**输出示例**

```
<Thought>
The user asks not to solve with Long CoT, so I will directly write the answer.
</Thought>

<Output>
```lean4
/--给定非负实数 $a, b, c$，满足 $a + b + c = 1$。证明：$7(ab + bc + ca) \leq 2 + 9abc$。-/
theorem prover_data_2216 (a b c: ℝ) (ha: 0 ≤ a) (hb: 0 ≤ b) (hc: 0 ≤ c) (habc: a + b + c = 1): 
  7 * (a * b + b * c + c * a) ≤ 2 + 9 * a * b * c := by
  /- 给定非负实数 \(a\)、\(b\) 和 \(c\) 满足 \(a + b + c = 1\)，
     我们需要证明 \(7(ab + bc + ca) \leq 2 + 9abc\)。
     证明涉及使用代数操作和不等式来表明左侧被右侧限定。
     具体来说，我们使用实数平方非负的事实，这有助于建立所需的不等式。
  -/
  -- 归一化涉及 `a`、`b` 和 `c` 的表达式，为代数操作做准备。
  ring_nf at habc ⊢
  -- 使用 `nlinarith` 处理非线性算术不等式。这包括检查涉及 `a`、`b` 和 `c` 的表达式的非负性。
  nlinarith [sq_nonneg (a - 1), sq_nonneg (b - 1), sq_nonneg (c - 1), sq_nonneg (a - b), 
    sq_nonneg (b - c), sq_nonneg (c - a), sq_nonneg (a + b + c)]
```
</Output>
<|end_of_sentence|>
```

### E.2 纠错数据的 LoT-TL 训练示例

**输入示例**

```
<|begin_of_sentence|>You are a helpful assistant who will solve every problem **WITH** Long Chain-of-Thought. 
You can make some Lean4 proof draft in your O1 style Long internal Chain of Thought (CoT) and you have the ability 
to test the thing during your thinking. You need to use the error messages in your Long CoT to guide your proof 
generation but you are required **NOT** to provide natural language analysis to the errors.

### Instruction:
@ Natural language theorem statement: 
prover_data_2216: 给定非负实数 $a, b, c$，满足 $a + b + c = 1$。证明：$7(ab + bc + ca) \leq 2 + 9abc$。

@ Lean4 theorem statement:
```lean4
theorem prover_data_2216 (a b c: ℝ) (ha: 0 ≤ a) (hb: 0 ≤ b) (hc: 0 ≤ c) (habc: a + b + c = 1): 
  7 * (a * b + b * c + c * a) ≤ 2 + 9 * a * b * c := by
```

@ Lean4 定理陈述和带解释性注释的证明（每行代码前有注释）：

### Response:
```

**输出示例**

```
<Thought>
Alright, I need to prove the theorem prover_data_2216 using the Lean4 code. Here is my draft of the proof:

```lean4
theorem prover_data_2216 (a b c: ℝ) (ha: 0 ≤ a) (hb: 0 ≤ b) (hc: 0 ≤ c) (habc: a + b + c = 1): 
  7 * (a * b + b * c + c * a) ≤ 2 + 9 * a * b * c := by
  with aes
```

Let me test it in Lean4
Emmm, it seems the above proof is wrong.
Let me check the error messages.
OK, Here is the error messages:

```bash
line 7
unexpected token 'with'; expected '{' or tactic
```

```bash
line 7
unsolved goals
a b c: ℝ
ha: 0 ≤ a
hb: 0 ≤ b
hc: 0 ≤ c
habc: a + b + c = 1
⊢ 7 * (a * b + b * c + c * a) ≤ 2 + 9 * a * b * c
```

Since the user ask NOT to provide natural language analysis to the problem. Let me directly try to write the revised code.
</Thought>

<Output>
```lean4
/--给定非负实数 $a, b, c$，满足 $a + b + c = 1$。证明：$7(ab + bc + ca) \leq 2 + 9abc$。-/
theorem prover_data_2216 (a b c: ℝ) (ha: 0 ≤ a) (hb: 0 ≤ b) (hc: 0 ≤ c) (habc: a + b + c = 1): 
  7 * (a * b + b * c + c * a) ≤ 2 + 9 * a * b * c := by
  /- 给定非负实数 \(a\)、\(b\) 和 \(c\) 满足 \(a + b + c = 1\)，
     我们需要证明 \(7(ab + bc + ca) \leq 2 + 9abc\)。
     证明涉及使用代数操作和不等式来表明左侧被右侧限定。
     具体来说，我们使用实数平方非负的事实，这有助于建立所需的不等式。
  -/
  -- 归一化涉及 `a`、`b` 和 `c` 的表达式，为代数操作做准备。
  ring_nf at habc ⊢
  -- 使用 `nlinarith` 处理非线性算术不等式。这包括检查涉及 `a`、`b` 和 `c` 的表达式的非负性。
  nlinarith [sq_nonneg (a - 1), sq_nonneg (b - 1), sq_nonneg (c - 1), sq_nonneg (a - b), 
    sq_nonneg (b - c), sq_nonneg (c - a), sq_nonneg (a + b + c)]
```
</Output>
<|end_of_sentence|>
```


## 附录 F 智能体输入示例

### F.1 证明者智能体输入-输出示例

**输入示例 - 我们的证明者找到的 IMO 问题**

```
@ Natural language theorem statement: 
imo_1961_p1: (''匈牙利'') 求解方程组：
<center>
$ \begin{matrix} \quad x + y + z \\ &= a \; \, \\ x^2 +y^2+z^2 \\ &=b^2 \\ \qquad \qquad xy \\ &= z^2 \end{matrix} $
</center>
其中 $a$ 和 $b$ 是常数。给出 $a$ 和 $b$ 必须满足的条件，使得 $x, y, z$（方程组的解）是不同的正数。

@ Lean4 theorem statement:
```lean4
theorem imo_1961_p1 (x y z a b: ℝ) (h₀: 0 < x ∧ 0 < y ∧ 0 < z) (h₁: x ≠ y) (h₂: y ≠ z) (h₃: z ≠ x) 
  (h₄: x + y + z = a) (h₅: x ^ 2 + y ^ 2 + z ^ 2 = b ^ 2) (h₆: x * y = z ^ 2): 
  0 < a ∧ b ^ 2 < a ^ 2 ∧ a ^ 2 < 3 * b ^ 2 := by
```

@ Lean4 定理陈述和带解释性注释的证明（每行代码前有注释）：
```

**输出示例**

```
<Thought>
Alright, I should do the following: 
1. Provide the natural language analysis for the theorem based on the Natural language theorem statement.
2. Draft the Lean4 tactics I should use to solve the problem
3. Write the output Lean4 code.

The user also asks that I should avoid using the keyword `sorry` to give up the proof, so I will not write it in my Lean4 code.

The `imo_1961_p1` can be proofed by analyzing the given equations and deriving necessary inequalities to ensure that the solutions are distinct positive numbers.
</Thought>

<Output>
```lean4
theorem imo_1961_p1 (x y z a b: ℝ) (h₀: 0 < x ∧ 0 < y ∧ 0 < z) (h₁: x ≠ y) (h₂: y ≠ z) (h₃: z ≠ x) 
  (h₄: x + y + z = a) (h₅: x ^ 2 + y ^ 2 + z ^ 2 = b ^ 2) (h₆: x * y = z ^ 2): 
  0 < a ∧ b ^ 2 < a ^ 2 ∧ a ^ 2 < 3 * b ^ 2 := by
  -- 我们需要证明三个不等式：0 < a, b^2 < a^2, 和 a^2 < 3 * b^2。
  refine' ⟨_, _, _⟩
  -- 首先，证明 0 < a。
  -- 由于 x, y, z 都是正数，它们的和 a 也必须是正数。
  nlinarith [h₀.1, h₀.2.1, h₀.2.2, sq_nonneg (x - y), sq_nonneg (y - z), sq_nonneg (z - x)]
  -- 接下来，证明 b^2 < a^2。
  -- 使用给定的方程和不等式，推导必要的不等式。
  nlinarith [mul_self_pos.mpr (sub_ne_zero.mpr h₁), mul_self_pos.mpr (sub_ne_zero.mpr h₂), 
    mul_self_pos.mpr (sub_ne_zero.mpr h₃), sq_nonneg (x - y), sq_nonneg (y - z), sq_nonneg (z - x)]
  -- 最后，证明 a^2 < 3 * b^2。
  -- 结合所有给定信息来推导这个不等式。
  nlinarith [mul_self_pos.mpr (sub_ne_zero.mpr h₁), mul_self_pos.mpr (sub_ne_zero.mpr h₂), 
    mul_self_pos.mpr (sub_ne_zero.mpr h₃), sq_nonneg (x - y), sq_nonneg (y - z), sq_nonneg (z - x)]
```
</Output>
```

### F.2 纠正者智能体输入-输出示例

**输入示例**

```
You are a helpful mathematical assistant specialized in formal theorem proving using Lean4. Your objectives:
1. Read and interpret the Lean4 theorem statement and any error messages.
2. **If a previous proof attempt was incorrect, analyze its exact mistakes and completely discard or rewrite the proof as needed.**
3. **Avoid reusing incorrect proof structures or strategies unless explicitly validated as correct.**
4. **Address all error messages** by modifying the proof structure as needed.
5. Provide a detailed thought process in the <Thought> section, but **only place the corrected Lean4 code in the <Output> section**.
6. **Ensure the new proof is logically valid and does not use `sorry`.**

Instruction:
Below are some **correct Lean4 theorem proof examples** for your reference. Use them as guidance when constructing the revised proof. 
Ensure that your final proof aligns with these examples in terms of clarity, structure, and correctness.

Instruction:
@ Natural language theorem statement: 
mathd_algebra_392 三个连续正偶数的平方和是 $12296$。求这三个数的乘积除以 $8$。证明结果是 32736。

@ Lean4 theorem statement:
```lean4
theorem mathd_algebra_392 (n: ℕ) (h₀: Even n) (h₁: ((n:ℤ) - 2)^2 + (n:ℤ)^2 + ((n:ℤ) + 2)^2 = 12296): 
  ((n - 2) * n * (n + 2)) / 8 = 32736 :=
```

@ Lean4 定理陈述和带解释性注释的证明（每行代码前有注释）：

Response:
```

**输出示例**

```
<Thought>
Alright, I need to prove the theorem prover_data_68493 using the Lean4 code. Here is my draft of the proof:

```lean4
theorem mathd_algebra_392 (n: ℕ) (h₀: Even n) (h₁: ((n:ℤ) - 2)^2 + (n:ℤ)^2 + ((n:ℤ) + 2)^2 = 12296): 
  ((n - 2) * n * (n + 2)) / 8 = 32736 := by
  -- 将方程转换为整数以便更容易操作
  norm_cast at h₁
  -- 展开并简化给定的方程
  ring_nf at h₁
  -- 使用数值推理求解 n
  have h₂: n = 24 := by nlinarith
  -- 将 n = 24 代入表达式并简化
  norm_num [h₂]
```

Let me test it in Lean4
Emmm, it seems the above proof is wrong.
Let me check the error messages.
OK, Here is the error messages:

```bash
line 18
linarith failed to find a contradiction
case h1.h
n: ℕ
h₀: Even n
h₁: Int.subNatNat n 2 ^ 2 + ↑(n ^ 2) + ↑(4 + n * 4 + n ^ 2) = 12296
a✝: n < 24
⊢ False
failed
```

```bash
line 11
unsolved goals
n: ℕ
h₀: Even n
h₁: Int.subNatNat n 2 ^ 2 + ↑(n ^ 2) + ↑(4 + n * 4 + n ^ 2) = 12296
h₂: n = 24
⊢ False
```

So, I will rethink a Lean4 proof following the steps:
1. Provide the natural language analysis for the theorem based on the Natural language theorem statement, Lean4 theorem statement, 
   my previous proof and the error message.
2. Draft the Lean4 tactics I should use to solve the problem
3. Write the output Lean4 code.

Let me analysis the wrong Lean4 solution through the error messages.

From the error messages, it seems the following points:
- `linarith failed to find a contradiction` suggests that the linear arithmetic tactic was unable to derive a contradiction from the given hypotheses, 
  indicating that the assumptions might not lead to a contradiction as expected.
- `case h1.h` indicates a specific case within a larger context where a contradiction was expected but was not found, 
  suggesting that the assumption `n < 24` might not be valid under certain conditions.
- `unsolved goals` means that after applying the tactics, there are still unresolved goals left, 
  indicating that the tactics applied were insufficient to complete the proof.

From these error messages, I realize that my previous approach might not have handled the specifics of the problem correctly. 
Therefore, I need to revise my proof strategy.

[详细分析过程...]

Now, I will proceed to write the Lean4 code following this plan.
</Thought>

<Output>
```lean4
theorem mathd_algebra_392 (n: ℕ) (h₀: Even n) (h₁: ((n:ℤ) - 2)^2 + (n:ℤ)^2 + ((n:ℤ) + 2)^2 = 12296): 
  ((n - 2) * n * (n + 2)) / 8 = 32736 := by
  -- 将方程转换为整数以便更容易操作
  push_cast at h₁
  -- 展开并简化给定的方程
  ring_nf at h₁
  -- 使用数值推理求解 n
  have h₂: n = 64 := by nlinarith
  -- 将 n = 64 代入表达式并简化
  subst h₂
  norm_num
```
</Output>
```


---

## 参考文献

[^1]: Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

[^2]: Azerbayev, Z., Schoelkopf, H., Paster, K., Santos, M. D., McAleer, S., Jiang, A. Q., Deng, J., Biderman, S., and Welleck, S. Llemma: An open language model for mathematics. *arXiv preprint arXiv:2310.10631*, 2023.

[^3]: De Moura, L., Kong, S., Avigad, J., Van Doorn, F., and von Raumer, J. The lean theorem prover (system description). In *Automated Deduction-CADE-25: 25th International Conference on Automated Deduction, Berlin, Germany, August 1-7, 2015, Proceedings 25*, pp. 378–388. Springer, 2015.

[^4]: Dong, K. and Ma, T. Beyond limited data: Self-play llm theorem provers with iterative conjecturing and proving. *arXiv preprint arXiv:2502.00212*, 2025.

[^5]: Frieder, S., Bayer, J., Collins, K. M., Berner, J., Loader, J., Juhász, A., Ruehle, F., Welleck, S., Poesia, G., Griffiths, R.-R., et al. Data for mathematical copilots: Better ways of presenting proofs for machine learning. *arXiv preprint arXiv:2412.15184*, 2024.

[^6]: Harrison, J. Hol light: An overview. In *International Conference on Theorem Proving in Higher Order Logics*, pp. 60–66. Springer, 2009.

[^7]: Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. *arXiv preprint arXiv:2103.03874*, 2021.

[^8]: Jiang, A. Q., Li, W., Han, J. M., and Wu, Y. Lisa: Language models of isabelle proofs. In *6th Conference on Artificial Intelligence and Theorem Proving*, pp. 378–392, 2021.

[^9]: Jiang, A. Q., Welleck, S., Zhou, J. P., Li, W., Liu, J., Jamnik, M., Lacroix, T., Wu, Y., and Lample, G. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. *arXiv preprint arXiv:2210.12283*, 2022.

[^10]: Kumarappan, A., Tiwari, M., Song, P., George, R. J., Xiao, C., and Anandkumar, A. Leanagent: Lifelong learning for formal theorem proving. *arXiv preprint arXiv:2410.06209*, 2024.

[^11]: Lin, H., Sun, Z., Yang, Y., and Welleck, S. Lean-star: Learning to interleave thinking and proving. *arXiv preprint arXiv:2407.10040*, 2024.

[^12]: Lin, Y., Tang, S., Lyu, B., Wu, J., Lin, H., Yang, K., Li, J., Xia, M., Chen, D., Arora, S., and Jin, C. Goedel-prover: A frontier model for open-source automated theorem proving, 2025. URL [https://arxiv.org/abs/2502.07640](https://arxiv.org/abs/2502.07640).

[^13]: Ma, C., Zhang, J., Zhu, Z., Yang, C., Yang, Y., Jin, Y., Lan, Z., Kong, L., and He, J. Agentboard: An analytical evaluation board of multi-turn llm agents. *arXiv preprint arXiv:2401.13178*, 2024.

[^14]: Moura, L. d. and Ullrich, S. The lean 4 theorem prover and programming language. In *Automated Deduction–CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings 28*, pp. 625–635. Springer, 2021.

[^15]: Newell, A. and Simon, H. The logic theory machine–a complex information processing system. *IRE Transactions on information theory*, 2(3):61–79, 1956.

[^16]: Open-Source-O1. Open-o1, 2024. URL [https://github.com/Open-Source-O1/Open-O1](https://github.com/Open-Source-O1/Open-O1). Accessed: 2024-12-28.

[^17]: OpenAI. Learning to reason with llms. [https://openai.com/index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/), September 13 2024. Accessed: 2024-11-24.

[^18]: Paulson, L. C. *Isabelle: A generic theorem prover*. Springer, 1994.

[^19]: Polu, S. and Sutskever, I. Generative language modeling for automated theorem proving. *arXiv preprint arXiv:2009.03393*, 2020.

[^20]: Polu, S., Han, J. M., Zheng, K., Baksys, M., Babuschkin, I., and Sutskever, I. Formal mathematics statement curriculum learning. *arXiv preprint arXiv:2202.01344*, 2022.

[^21]: Pourchot, A. and Sigaud, O. Cem-rl: Combining evolutionary and gradient-based methods for policy search. *arXiv preprint arXiv:1810.01222*, 2018.

[^22]: Reed, S., Zolna, K., Parisotto, E., Colmenarejo, S. G., Novikov, A., Barth-Maron, G., Gimenez, M., Sulsky, Y., Kay, J., Springenberg, J. T., et al. A generalist agent. *arXiv preprint arXiv:2205.06175*, 2022.

[^23]: Reid, M., Savinov, N., Teplyashin, D., Lepikhin, D., Lillicrap, T., Alayrac, J.-b., Soricut, R., Lazaridou, A., Firat, O., Schrittwieser, J., et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv preprint arXiv:2403.05530*, 2024.

[^24]: Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.

[^25]: Wang, R., Jansen, P., Côté, M.-A., and Ammanabrolu, P. Scienceworld: Is your agent smarter than a 5th grader? *arXiv preprint arXiv:2203.07540*, 2022.

[^26]: Wang, R., Zhou, W., and Sachan, M. Let's synthesize step by step: Iterative dataset synthesis with large language models by extrapolating errors from small models. *arXiv preprint arXiv:2310.13671*, 2023.

[^27]: Wang, R., Zhang, J., Jia, Y., Pan, R., Diao, S., Pi, R., and Zhang, T. Theoremllama: Transforming general-purpose llms into lean4 experts. *arXiv preprint arXiv:2407.03203*, 2024.

[^28]: Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824–24837, 2022.

[^29]: Wu, Z., Huang, S., Zhou, Z., Ying, H., Wang, J., Lin, D., and Chen, K. Internlm2. 5-stepprover: Advancing automated theorem proving via expert iteration on large-scale lean problems. *arXiv preprint arXiv:2410.15700*, 2024a.

[^30]: Wu, Z., Wang, J., Lin, D., and Chen, K. Lean-github: Compiling github lean repositories for a versatile lean prover. *arXiv preprint arXiv:2407.17227*, 2024b.

[^31]: Xie, T., Zhou, F., Cheng, Z., Shi, P., Weng, L., Liu, Y., Hua, T. J., Zhao, J., Liu, Q., Liu, C., et al. Openagents: An open platform for language agents in the wild. *arXiv preprint arXiv:2310.10634*, 2023.

[^32]: Xin, H., Guo, D., Shao, Z., Ren, Z., Zhu, Q., Liu, B., Ruan, C., Li, W., and Liang, X. Deepseek-prover: Advancing theorem proving in llms through large-scale synthetic data. *arXiv preprint arXiv:2405.14333*, 2024a.

[^33]: Xin, H., Ren, Z., Song, J., Shao, Z., Zhao, W., Wang, H., Liu, B., Zhang, L., Lu, X., Du, Q., et al. Deepseek-prover-v1. 5: Harnessing proof assistant feedback for reinforcement learning and monte-carlo tree search. *arXiv preprint arXiv:2408.08152*, 2024b.

[^34]: Xu, Y., Su, H., Xing, C., Mi, B., Liu, Q., Shi, W., Hui, B., Zhou, F., Liu, Y., Xie, T., et al. Lemur: Harmonizing natural language and code for language agents. *arXiv preprint arXiv:2310.06830*, 2023.

[^35]: Yang, H., Yue, S., and He, Y. Auto-gpt for online decision making: Benchmarks and additional opinions. *arXiv preprint arXiv:2306.02224*, 2023.

[^36]: Yang, K., Poesia, G., He, J., Li, W., Lauter, K., Chaudhuri, S., and Song, D. Formal mathematical reasoning: A new frontier in ai. *arXiv preprint arXiv:2412.16075*, 2024a.

[^37]: Yang, K., Swope, A., Gu, A., Chalamala, R., Song, P., Yu, S., Godil, S., Prenger, R. J., and Anandkumar, A. Leandojo: Theorem proving with retrieval-augmented language models. *Advances in Neural Information Processing Systems*, 36, 2024b.

[^38]: Ying, H., Wu, Z., Geng, Y., Wang, J., Lin, D., and Chen, K. Lean workbook: A large-scale lean problem set formalized from natural language math problems. *arXiv preprint arXiv:2406.03847*, 2024.

[^39]: Zheng, K., Han, J. M., and Polu, S. Minif2f: a cross-system benchmark for formal olympiad-level mathematics. *arXiv preprint arXiv:2109.00110*, 2021.

---

**脚注：**

<sup>1</sup> 我们在 Valid 集上的测试结果与 [^33] 报告的 DeepSeek-Prover-v1.5-RL 基线结果一致。

<sup>2</sup> 树搜索成本通常表示为 $n_{\text{problems}} \times n_{\text{expansions}} \times n_{\text{samples}}$。为了公平比较，我们将搜索成本设置为与 pass@128 相当的水平。

<sup>3</sup> 所有实验均在配备 NVIDIA H100-96G GPU 的集群上进行。

<sup>4</sup> 每轮纠正的成本约为完整证明生成的一半，因此两轮纠正加上初始证明的总成本约等于 pass@128。

