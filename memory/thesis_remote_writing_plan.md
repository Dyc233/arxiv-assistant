---
name: 论文远程撰写计划
description: 4.25出发去北京旅游一周，5.1要交初稿，利用GitHub代码+平板远程写论文的策略和前情提要
type: project
---

# 情境

2026-04-25 出发去北京旅游一周，不能带电脑。导师要求 5.1 交论文初稿。代码已上传 GitHub（不含 chroma_db/、parquet、.venv 等），远程可用平板/朋友电脑 clone 源码阅读。

# 核心策略

**旅途中写文字密集型章节，回家后补数据密集型章节。**

| 章节 | 远程能否写 | 原因 |
|------|-----------|------|
| 绪论（背景/意义） | 能 | 纯文字 |
| 相关工作（文献综述） | 能 | 可用自己的检索系统查相关文献 |
| 系统设计（架构/模块） | 能 | GitHub 有完整源码 + README + ANALYSIS.md + MANUAL.md |
| 实现细节 | 能 | 源码可读，每个模块描述实现思路 |
| 数据分析 | 部分能 | Git 上有分析产出的 CSV/PNG，但解释文字需基于图表 |
| 实验评估 | 不能 | 需跑系统测 recall/precision，无运行环境 |
| 检索效果展示 | 部分能 | 如果出发前截图了就能写，否则不能 |

# 出发前准备清单

1. **截图**（最重要）：登录页、分析看板、检索流程（至少3个示例：语义/混合/精确查找）、所有分析图表
2. **代码 push**：确认所有改动已提交并 push 到 GitHub
3. **备份无法 push 的数据**到网盘：parquet 文件、chroma_db 目录
4. **浏览参考稿**结构（docs/1毕设参考稿.pdf，56页），了解每章写法

# 论文大纲与对应代码/数据

## 绪论
- 背景：NLP 论文爆炸式增长，研究者需要高效检索工具
- 意义：基于 LLM + 向量检索的论文助手
- 参考：README.md 项目概述

## 相关工作
- 学术搜索引擎现状
- RAG（检索增强生成）技术
- 向量数据库与语义检索
- LLM Agent 框架
- 参考：docs/former/技术文档.txt、docs/former/cite.txt

## 系统设计
- 总体架构图（用 mermaid 画，GitHub 渲染）
- 数据采集模块 → data/spider.py
- 数据处理流水线 → analysis/process.py
- 向量化入库 → data/embedder.py
- 检索系统设计（三种模式）→ search/searcher.py, search/utils.py
- Agent 编排 → agent/agent.py, agent/schemas.py, agent/prompts.py
- 前端设计 → app/app.py, app/search.py, app/dashboard.py
- 增量更新 pipeline → data/updater.py
- 参考：README.md, agent/MANUAL.md, analysis/ANALYSIS.md

## 系统实现
- 每个模块对照源码写实现细节
- 关键技术：BGE-M3 embedding, BGE-Reranker, 混合检索, 元数据过滤
- 核心创新点：自适应动态加权（方案C，README 中提到）
- 调试经验：docs/过程文档/进度.txt 中有 GPU OOM、fp16 优化、upsert 幂等等记录

## 数据分析
- 基于 Git 上已有的 CSV/PNG：
  - 词云 → analysis/wordcloud/global_wordcloud.png
  - 发文趋势 → analysis/trends/publish_trend.png
  - 顶会分布 → analysis/confs/conference_distribution.png
  - 高产作者 → analysis/authors/top_authors.png
  - 关键词趋势热力图 → analysis/keyword_trend/keyword_trend.png
  - 跨学科渗透 → analysis/crossdomain/crossdomain_*.png
- 分析说明文档：analysis/ANALYSIS.md 有每个模块的详细方法论

## 实验评估（回家后写）
- 检索性能：recall/precision
- 消融实验：三种检索模式对比
- 自适应动态加权的效果对比

# 代码规模

共 2374 行 Python，24 个文件。在本科毕设中属于偏上水平。覆盖五段数据管道、三种检索模式、Agent 三段编排、7 个分析维度、完整前端。

# 与参考稿对比

- **优势**：系统完整性强（真能跑的全链路系统 vs 跑实验写论文）、检索技术有工程深度、分析维度更丰富
- **劣势**：实验评估不如参考稿规范（缺 recall/precision 对比、消融实验、显著性检验）、论文写作包装需要加强
- **风险**：系统亮点如果论文不写清楚、不做对比实验，评委感知不到
