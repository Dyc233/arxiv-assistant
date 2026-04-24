---
name: 项目概况
description: BS毕设项目：arXiv NLP论文RAG系统，技术栈/架构/模块结构/当前进度
type: project
---

# 项目概况

本科毕设项目，arXiv NLP 论文检索助手（RAG 系统）。**截至 2026-04-24 系统前后端已整合完毕，处于调试阶段。**

## 技术栈

- **LLM**：Moonshot (Kimi) `kimi-k2-turbo-preview`，需环境变量 `MOONSHOT_API_KEY`
- **Agent 框架**：`agno`（`AgnoResearchAssistant` 类，router/retrieve/renderer 三段编排）
- **向量库**：ChromaDB（本地持久化 `chroma_db/`，collection: `arxiv_nlp_papers`，cosine 距离）
- **Embedding**：`BAAI/bge-m3`，**Reranker**：`BAAI/bge-reranker-v2-m3`
- **元数据存储**：Parquet 文件（`analysis/cleaned_papers.parquet` + 增量 `cleaned_papers_incremental.parquet`）
- **数据采集**：arXiv 爬虫（`data/spider.py`）→ SQLite → 清洗 → Parquet → ChromaDB
- **前端**：Streamlit（`app/app.py`），带用户登录/权限系统（JSON 文件 `data/users.json`）
- **CUDA**：优先 GPU，fallback CPU

## 架构

**Agent 三层：Route → Retrieve → Render**
- **Router**：LLM 解析用户意图为结构化 `RoutingDecision`（task_type/search_mode/filters/response_mode/top_k）
- **Retrieve**：三种检索模式 semantic / metadata / hybrid（`PaperSearcher` 类）
  - semantic：embedding 向量召回 → Reranker 重排序
  - metadata：字段过滤 + 匹配得分排序（可选 Reranker）
  - hybrid：元数据过滤 → 向量召回 → post-filter → Reranker
- **Renderer**：按 `ResponseMode`（raw_list / list_with_insights / report）生成中文回答

## 模块结构

```
src/
├── app/          # Streamlit 前端
│   ├── app.py          # 主入口：登录/检索/管理面板
│   └── dashboard.py    # 分析看板（KPI + 图表分组）
├── agent/        # Agent 核心
│   ├── cli.py          # CLI 入口 / agno Agent 初始化
│   ├── agent.py        # AgnoResearchAssistant 实现
│   ├── prompts.py      # Router/Renderer prompt 模板
│   └── schemas.py      # RoutingDecision, ResponseMode 定义
├── search/       # 检索系统
│   ├── searcher.py     # PaperSearcher（三种检索模式）
│   └── utils.py        # 过滤 + 匹配打分辅助函数
├── data/         # 数据采集
│   ├── spider.py       # arXiv API 爬虫 → SQLite
│   ├── updater.py      # 增量更新 pipeline（爬虫→清洗→合并→upsert）
│   └── embedder.py     # embedding 生成 + ChromaDB upsert
└── analysis/     # 数据分析（已全部完成）
    ├── process.py       # 清洗/合并/关键词提取/顶会正则
    ├── trends/          # 月度发文趋势
    ├── confs/           # 顶会录用分布
    ├── authors/         # 高产作者 Top 20
    ├── wordcloud/       # 词云
    ├── crossdomain/     # 跨学科渗透分析
    ├── keyword_trend/   # 技术术语兴衰热力图
    └── timezone/        # 时区分布
```

## 当前进度（2026-04-24）

- ✅ 数据分析模块全部完成（7 个子分析）
- ✅ Streamlit 前端完整搭建（登录/管理/检索/Dashboard）
- ✅ Agent 精简重构完成（agno 框架）
- ✅ 增量更新 pipeline 完整
- ⏳ 系统调试阶段（最近 commit: "debug"）

## 工作目录

`D:/CODING/BS/src`
