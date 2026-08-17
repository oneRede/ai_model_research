---
sourceTitle: "Instability of LLM Pre-Pretraining: It Doesn't Always Help. An Investigation on Multiple Languages"
sourceUrl: "https://arxiv.org/html/2608.08800v1"
sourceArxivId: "2608.08800"
title: "LLM 预预训练的不稳定性：并非总是有效——多语言调查研究"
pipelineRunId: "20260816-220641"
pipelineSource: "translate/20260816-220641/works-ready/arxiv-2608-08800-translation.md"
sourceFigureCount: null
sourceAuthors: "Sofiia Riazhskykh, Nam Luu, Ondřej Bojar"
sourceDate: "2026-08-09"
translatedAt: "2026-08-17"
---

# LLM 预预训练的不稳定性：并非总是有效——多语言调查研究

Sofiia Riazhskykh, Nam Luu, Ondřej Bojar

查理大学数学与物理学院

sofiia.riazhskykh890@student.cuni.cz, {luu,bojar}@ufal.mff.cuni.cz

###### 摘要

在人工语言上预训练大语言模型（LLM）（即"预预训练"，pre-pretraining）是一种据报道可将标记效率（token efficiency）提高 33% 的技术，即节省高达 33% 的训练标记来达到特定性能。我们在更大的自然语言集合上验证了这一先前针对英语的结果，涵盖四个语系的语言，使用两种不同的分词器和不同的模型大小。我们还将观察到的标记效率增益（或损失）与语言的量化语言学属性相关联，如句子长度、形态丰富度（morphological richness）以及依赖句法树（dependency syntactic trees）的特征（树深度、子节点数量、交叉依赖数量）。我们的实证结果表明，所报道的增益高度依赖于实验设置和随机种子的选择，尽管我们可以确认对于大多数被检验语言，使用 Llama 分词器的小型模型在 128-Dyck 预训练下确实存在稳定增益的趋势。总体而言，我们主张应该至少对部分实验进行多次训练运行，以避免社区采纳不稳定的方法。

## 1 引言

语言模型随着规模增长需要更多训练数据，这对低资源语言和特定领域构成挑战。迁移学习（transfer learning），即从不同数据集中受益的知识，是应对这种情况的既定技术之一。机器翻译领域的先前工作 [^8] [^10] 表明，在相关甚至完全不相关的语言对上进行初始训练可以提高目标语言对的性能。后来，在人工语言上预训练大语言模型（LLM）（或称"预预训练"）[^15] [^12] [^7] 时也观察到了增益。然而，这些关于 LLM 的最新研究仅聚焦于英语。

据我们所知，该方法是否能在其他自然语言、模型大小和架构上持续提升标记效率——即减少达到给定性能所需的数据量——仍不清楚。

在本工作中，我们在两个主要方向上超越了先前关于 LLM 从人工语言迁移学习的研究：我们扩展了 (1) 使用的形式语言集合，以及 (2) 不同语系的自然语言集合，寻求影响该方法有效性的人工语言和自然语言属性之间的关系。

实验结果表明，该预训练方法可能对不同的自然语言有帮助，尽管观察到的增益高度依赖于特定的训练运行，并且对设置细节也很敏感（图 1）。

图 1：使用两种不同分词器的所有模型的平均标记效率增益，其中 LT 代表 Llama，GT 代表 Gemma 分词器。

## 2 相关工作

**迁移学习**在机器翻译和低资源 NLP 中有着悠久的历史 [^8] [^10]。最近的工作探索了在人工语言上预训练 LLM 以改进自然语言性能的可能性。

Ri 和 Tsuruoka [^15] 首次系统研究了在人工语言（包括 Dyck 语言）上预训练对后续自然语言建模的影响。Papadimitriou 和 Jurafsky [^12] 进一步研究了这种预训练如何向模型注入结构偏置。

Hu 等人 [^7] 报告了在英语上使用 Dyck 预训练可获得 33% 的标记效率增益，这是本工作的主要验证目标。其他相关工作包括 Kumar 等人 [^9] 探索 LoRA 与人工语言的结合，以及 Budnikov 和 Yamshchikov [^2] 研究结构知识的迁移。

然而，所有这些工作主要关注英语，跨语言和跨设置的鲁棒性仍是未知数。

## 3 数据集

### 3.1 预训练的形式语言

在形式语言中，我们考虑 $k=64$ 和 $k=128$ 的 $k$-Dyck（展示嵌套依赖）和 $k$-Shuffle Dyck（平面依赖）语言，其定义如下：

$k$-Dyck 包含 $k$ 种不同类型的结构嵌套依赖，每种依赖由匹配的开闭括号对表示，例如 ( \[ { } \] ) 或 ( \[ \] ) { }。

$k$-Shuffle Dyck 允许打破良好嵌套性（well-nestedness）（例如 ( \[ \] { ) }）并包含交叉序列依赖（cross-serial dependencies）（例如 ( \[ { ) \] }）。

我们遵循 [^7] 的方法，用非负整数表示括号符号。换句话说，如果一个开符号表示为数字 $n$，其对应的闭符号表示为 $n+k$，其中 $k$ 是不同依赖的数量，且 $0\leq n<k$。对于 $k=128$ 的一个开闭符号对示例是（"81", "209"）。我们请读者参阅 [^7] 了解具体实现。

### 3.2 自然语言

除英语外，我们分析了来自四个语系的其他五种类型多样的欧洲语言的预训练效果，即阿尔巴尼亚语（阿尔巴尼亚语族）、捷克语（斯拉夫语族）、丹麦语和荷兰语（日耳曼语族）以及芬兰语（乌拉尔语系）。

我们通过混合 FineWeb（用于英语；[^13]）或 FineWeb 2（用于其他五种语言；[^14]）与 OpenSubtitles [^11] 和 EUBookshop [^17]，按 8:1:1 的比例，为每种语言构建单语、多领域的训练数据集。

重要的是，我们试图隔离 [^8] 观察到的预训练语料库大小的影响：更大的预训练语料库总是带来更大的增益。在所有实验中，我们的预训练语料库由 16,000 个 2,048 标记的 Dyck 序列组成（即批次大小为 32 的 500 个训练步数），之后使用的自然语言语料库为 320,000 个 2,048 标记的序列（即批次大小为 32 的 10,000 步），由于分词会有微小变化。

对于评估，我们使用多语言海量清洁爬取语料库（Multilingual Colossal Clean Crawled Corpus, mC4）数据集验证集的前 2,000 行。

## 4 实验与结果

### 4.1 实验设置

我们基于 Llama 3 架构 [^5] 实验了三种模型大小。每个模型配备 Llama 3 (LT) 或 Gemma 3 分词器 (GT) [^16]。这产生了六个不同的模型，参数量从 253M 到 884M 不等。这些模型及其后续使用的简写别名概述于表 1。

**表 1：模型概览。LT 和 GT 分别代表使用 Llama 和 Gemma 分词器的模型。**

| 配置 | 有效参数 | 嵌入参数 | 总计 | 名称 |
|------|---------|---------|------|------|
| 0 | 154M | 99M | 253M | 154M+LT |
| 0 | 154M | 201M | 355M | 154M+GT |
| 1 | 308M | 131M | 440M | 308M+LT |
| 1 | 308M | 268M | 577M | 308M+GT |
| 2 | 481M | 197M | 678M | 481M+LT |
| 2 | 481M | 403M | 884M | 481M+GT |

在固定的自然语言标记训练预算下，我们力求使模型在标记参数比方面与 [^7] 的模型具有可比性（表 2）。重要的是，我们注意到我们和 [^7] 实验中的所有比率值都不能被视为 Chinchilla 最优 [^6]，Chinchilla 最优提出了大约 20-25 个标记（甚至更多）每参数的最优值，这在最近的开放权重 LLM [^5] [^16] [^4] 中可以找到。尽管如此，我们看到训练和评估损失曲线在最后 3,000 个训练步数中保持稳定（例如图 2 和 A.4），这表明我们的模型经过了充分训练并与 [^7] 的模型具有可比性。关键的是，我们几乎看不到设置（基线 vs. 预训练）会在更多训练后改变顺序的可能性。

图 2：丹麦语 481M+GT 模型在最后 3,000 个训练步数中的评估损失。越低越好。

**表 2：标记参数比的比较**

|  | 模型 | 大小 | 标记数 | 比率 |
|---|------|------|--------|------|
| [^7] | Pythia | 160M | 655M | ≈4:1 |
| [^7] | Pythia | 1B | 1.63B | 1.63:1 |
| 我们的 | 154M+LT | 253M | 655M | ≈2.6:1 |
| （基于 Llama-3） | 154M+GT | 355M | 655M | ≈1.8:1 |
|  | 308M+LT | 440M | 655M | ≈1.5:1 |
|  | 308M+GT | 577M | 655M | ≈1.1:1 |
|  | 481M+LT | 678M | 655M | ≈0.97:1 |
|  | 481M+GT | 884M | 655M | ≈0.74:1 |

我们的人工语言表示的一个重要区别是 GT 将每个数字拆分为单独的数位，与 LT 不同。

每个模型使用三种方法在完整训练数据集上训练一个轮次（epoch）（见第 3.2 节）：(1) 无预训练（基线，Base）；(2) 在 $k$-Dyck 上预训练后；(3) 在 $k$-Shuffle Dyck 上预训练后。对于 (2) 和 (3)，$k=64$ 或 $128$。对于每个设置，我们收集三次训练运行，使用三个不同的随机种子选择。

遵循 [^7]，我们通过标记效率增益来量化改进，它反映了在人工语言上预训练的模型比相应的基线模型提前多少训练标记达到相同的验证损失。如果预训练模型性能更差，我们对该值取负。我们的具体计算方法在第 A.3 节中提供。我们训练中使用的超参数详情见附录 A。

### 4.2 结果

三次运行的结果揭示了每个相同设置内出乎意料的高方差，在最小模型（图 3）和更大模型（见附录 B 中的图 11 和 12）中都观察到了这一点。

**表 3：154M 模型预训练结果的一致性。** 我们仅指出该设置是否在所有三次运行中导致相对于基线模型的一致增益（+）、一致损失（-），或混合结果（×），其中一些运行受益而一些运行损失。在每个预训练语言组中，左列和右列分别代表 LT 和 GT 的结果。总结报告了相应设置中一致增益、混合结果和损失的数量。

| 自然语言 | 64-Dyck | 128-Dyck | 64-Shuff Dyck | 128-Shuff Dyck | LT | GT | 总体 |
|---------|---------|----------|---------------|----------------|----|----|------|
| 阿尔巴尼亚语 | × × | + - | × × | × × | 1+ 3× 0- | 0+ 3× 1- | 1+ 6× 1- |
| 捷克语 | + × | + × | + × | × - | 3+ 1× 0- | 0+ 3× 1- | 3+ 4× 1- |
| 丹麦语 | × + | + × | + × | × × | 2+ 2× 0- | 1+ 3× 0- | 3+ 5× 0- |
| 荷兰语 | + × | + + | × × | - - | 2+ 1× 1- | 1+ 2× 1- | 3+ 3× 2- |
| 英语 | + + | + + | + × | + + | 4+ 0× 0- | 3+ 1× 0- | 7+ 1× 0- |
| 芬兰语 | + × | + - | + - | × × | 3+ 1× 0- | 0+ 2× 2- | 3+ 3× 2- |

然而，在一个特定设置中，即在 128-Dyck 上预训练的 154M+LT，预训练确实带来了一致的改进（图 4 和 3）。此外，我们在表 3 中显示的英语实验结果表明，使用大多数人工语言（仅排除 64-Shuff Dyck）预训练最小的 154M 模型也会导致持续增益；因此，我们可以安全地证实先前研究对英语的发现（见第 2 节）。

图 3：154M 模型按语言的平均标记效率增益/损失。连线连接三次运行中的相同设置；因此水平线表示稳定行为，而穿过 x 轴的线表示高度不稳定的情况，其中一些训练运行导致增益，一些导致损失。

值得注意的是，使用 LT 的模型平均具有比使用 GT 的模型更高的增益（图 1）。对于前者，$k$-Dyck 在增益方面通常优于 $k$-Shuffle Dyck，而对于后者模型，这种模式并不那么明确（见附录 B 中的图 11）。

图 4：按语言的平均标记效率增益（对于在 128-Dyck 上预训练的 154M+LT）

我们还对最不稳定的设置之一进行了额外实验——在 64-Dyck 上预训练并随后在捷克语上训练的 154M+GT 模型——我们分别用 500、1,000 和 2,000 步进行预训练，每个值三次运行。图 5 表明，使用 1,000 步预训练可能是最优的。

图 5：在 64-Dyck 上预训练的 154M+GT 用于捷克语的平均标记效率增益，使用三个不同的预训练检查点。越靠右越好。

## 5 语言特定分析

增益与语言学属性之间可能存在相关性吗？由于观察到的不一致性，我们将分析限制在 154M 模型并进行显著性检验。所使用的语言学指标的所有定义以及我们语言特定分析的所有细节分别在附录 C 和 D 中提供。

## 6 结论

我们在多种语言、模型大小和分词器上系统地研究了 LLM 预预训练的有效性。我们的主要发现如下：

1. **不稳定性**：预预训练的增益高度依赖于随机种子和实验设置，即使在相同配置下也存在显著方差。

2. **部分有效性**：我们确认了一个稳定的积极结果：使用 Llama 分词器的 154M 模型在 128-Dyck 上预训练后，在大多数测试语言上表现出一致的增益。

3. **英语验证**：对于英语，我们大体上复现了先前报告的增益，证实了该方法在英语上的有效性。

4. **语言差异**：不同语言的响应模式不同，但我们未能找到语言学特征与增益之间的强相关性。

5. **方法论意义**：这些结果强调了在采纳看似有前景的训练方法之前，进行多次训练运行和跨设置验证的重要性。

### 局限性

本工作存在几个局限性：

- 我们的模型规模和训练预算低于 Chinchilla 最优 [^6]，这可能影响了观察到的增益模式。
- 我们仅测试了欧洲语言，更广泛的语言覆盖可能揭示不同的模式。
- 语言学特征分析基于有限的特征集，更全面的语言学表征可能提供更多见解。

尽管存在这些局限性，我们的研究为理解预预训练方法的鲁棒性和适用性提供了重要的实证证据。

---

## 参考文献

[^1]: O. Bojar. Machine Translation. In The Oxford Handbook of Inflection, 2019.

[^2]: M. Budnikov and I. Yamshchikov. Transfer of Structural Knowledge from Synthetic Languages. In XLLM 2025, pp. 242–251, 2025.

[^3]: C. Christodouloupoulos and M. Steedman. A massively parallel corpus: the bible in 100 languages. Language resources and evaluation 49 (2), pp. 375–395.

[^4]: DeepSeek-AI. DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence.

[^5]: A. Grattafiori, et al. The Llama 3 Herd of Models. arXiv:2407.21783, 2024.

[^6]: J. Hoffmann, et al. Training Compute-Optimal Large Language Models. arXiv:2203.15556, 2022.

[^7]: M. Y. Hu, et al. Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases. In ACL 2025, pp. 9691–9709, 2025.

[^8]: T. Kocmi and O. Bojar. Trivial Transfer Learning for Low-Resource Neural Machine Translation. In WMT 2018, pp. 244–252, 2018.

[^9]: N. Kumar, M. Lango, and O. Dusek. Pretraining Language Models with LoRA and Artificial Languages. In BabyLM Workshop, pp. 525–530, 2025.

[^10]: Y. Lin, et al. Choosing Transfer Languages for Cross-Lingual Learning. In ACL 2019, pp. 3125–3135, 2019.

[^11]: P. Lison and J. Tiedemann. OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. In LREC 2016, pp. 923–929, 2016.

[^12]: I. Papadimitriou and D. Jurafsky. Injecting structural hints: Using language models to study inductive biases in language learning. In EMNLP 2023 Findings, pp. 8402–8413, 2023.

[^13]: G. Penedo, et al. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. In NeurIPS 2024 Datasets and Benchmarks Track, 2024.

[^14]: G. Penedo, et al. FineWeb2: One Pipeline to Scale Them All. arXiv:2506.20920, 2025.

[^15]: R. Ri and Y. Tsuruoka. Pretraining with Artificial Language: Studying Transferable Knowledge in Language Models. In ACL 2022, pp. 7302–7315, 2022.

[^16]: G. Team, et al. Gemma 3 Technical Report. arXiv:2503.19786, 2025.

[^17]: J. Tiedemann. Parallel Data, Tools and Interfaces in OPUS. In LREC 2012, 2012.
