# works/ — 作品输出

可展示的成果：文章、工具、模板、教程等。

## 文件约定

- 每个作品一个子目录或单独文件
- 作品应该是**可独立理解的**，不依赖仓库其他部分的上下文
- 适合放到博客、GitHub、求职作品集中展示

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
```

**插图与外链约定**：

- 新收录译文的原文插图应下载到 `works/imgs/<slug>/`，以本地相对路径嵌入
- 译文正文保留原文中的超链接，不得在翻译时丢弃
- `scripts/check-consistency.sh` C10 会校验 `sourceFigureCount` 与正文嵌图数

| 文件 | 原文 | 来源 | 领域 |
|------|------|------|------|
| [github-qubot-analytics-agent-translation.md](github-qubot-analytics-agent-translation.md) | How We Built an Internal Data Analytics Agent | GitHub Engineering | 企业应用 |

## 作品方向参考

- AI 应用实践案例研究
- 行业 AI 落地方案分析
- AI 产品设计与用户体验
- AI 商业化路径探索

## 下一步

作品发出后，把外部读者的反馈（评论、转发、质疑）回流到 [feedback/](../feedback/)；
新出现的洞见、被挑战的论点回到 [thinking/](../thinking/) 继续打磨。
