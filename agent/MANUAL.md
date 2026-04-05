# Agent 模式说明

Agent 的"模式"分两层，**互相独立、自由组合**。

---

## 第一层：`task_type`（用户意图）

Router 判断用户想干什么，决定 `response_mode`：

| task_type | 含义 | 对应 response_mode |
|---|---|---|
| `lookup` | 精准定位某篇/某几篇论文 | `raw_list` |
| `search` | 找一批相关论文，不要深度分析 | `list_with_insights` |
| `report` | 明确要总结、综述、趋势分析 | `report` |
| `lookup_then_report` | 先定位论文，再做报告 | `report` |

`response_mode` 只影响 Renderer 的输出风格，**不影响检索逻辑**。

---

## 第二层：`search_mode`（检索方法）

Router 根据用户提供的条件类型决定，实际调用 `PaperSearcher` 的不同方法：

| search_mode | 触发条件 | 调用方法 | 底层逻辑 |
|---|---|---|---|
| `semantic` | 纯主题探索，无具体约束 | `semantic_search()` | BGE-M3 向量召回（top-50）→ BGE Reranker 重排序 → 返回 top-k |
| `metadata` | 只有 title/author/date/categories/comment 等硬条件 | `metadata_search()` | 从 Parquet 文件过滤 → 按发布时间倒排 → 固定 score=1.0 |
| `hybrid` | 既有主题又有元数据约束 | `hybrid_search()` | 先 Parquet 元数据过滤 → 再对候选集做向量相似度 → Reranker 重排序 |

---

## 已知问题

`metadata` 模式的 score 全部硬编码为 `1.0`，结果只按**发布时间**倒排，没有任何相关性排序。这导致该模式只返回最新论文而非最相关的论文。
