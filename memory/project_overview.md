---
name: 项目概况
description: BS毕设项目基本信息：arXiv NLP论文RAG系统，技术栈，架构要点
type: project
---

本科毕设项目，arXiv NLP 论文检索助手（RAG 系统）。

**技术栈**
- LLM：Moonshot (Kimi) `kimi-k2-turbo-preview`，需环境变量 `MOONSHOT_API_KEY`
- Agent 框架：`agno`
- 向量库：ChromaDB（本地持久化，路径 `chroma_db/`）
- Embedding：`BAAI/bge-m3`，Reranker：`BAAI/bge-reranker-v2-m3`
- 元数据存储：Parquet 文件（`analysis/cleaned_papers.parquet`）
- 数据采集：arXiv 爬虫 → SQLite → Parquet

**架构**：Router → Retrieve → Render 三段线性编排
- Router：解析用户意图为结构化 `RoutingDecision`（task_type / search_mode / filters）
- Retrieve：三种检索模式（semantic / metadata / hybrid）
- Renderer：按 response_mode 生成中文回答（raw_list / list_with_insights / report）

**工作目录**：`D:/CODING/BS/src`
