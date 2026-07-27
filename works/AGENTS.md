# works/ — 作品输出

可展示的成果：技术论文翻译、模型分析报告、技术综述等。

## 文件约定

- 每个作品一个独立文件或子目录
- 作品应该是**可独立理解的**，不依赖仓库其他部分的上下文
- 适合放到技术博客、GitHub、个人知识库中展示

## 已有作品

### 翻译

**元信息头约定**：每篇 `*-translation.md` 以 YAML frontmatter 开头，必备字段：

```yaml
title:             # 中文标题
sourceTitle:       # 原文标题
sourceUrl:         # 原文链接
sourceAuthor:      # 原作者（可含所属机构）
sourcePublishedAt: # 原文日期（未知可为 null）
translationMethod: # 翻译方式
language: "zh-CN"
sourceFigureCount: # 原文插图数（数字；null = 原文不可得、未审计）
pipelineRunId:     # 策展批次 ID；正式收录必须存在
pipelineSource:    # 对应 translate/<batch>/works-ready/<file>；存量手工条目写 legacy/...
```

**插图与外链约定**：

- 新收录译文的原文插图应下载到 `works/imgs/<slug>/`，以本地相对路径嵌入
- 译文正文保留原文中的超链接，不得在翻译时丢弃
- `scripts/check-consistency.sh` C10 会校验 `sourceFigureCount` 与正文嵌图数

| 文件 | 原文 | 来源 | 领域 |
|------|------|------|------|
| _待补充_ | - | - | - |

## 作品方向参考

- 前沿模型技术论文翻译
- 模型架构与训练方法解析
- 评测基准与能力研究分析
- 硬件基础设施与优化技术
- 数据工程与对齐方法综述

## 下一步

作品发出后，把外部读者的反馈（评论、转发、质疑）回流到 [feedback/](../feedback/)；
新出现的洞见、被挑战的论点回到 [thinking/](../thinking/) 继续打磨。
