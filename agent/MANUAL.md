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

## 测试用例

  ---
  用例1 — hybrid + 时间过滤（刚修的 bug）
  最近关于 KV Cache 压缩的研究有哪些？最好是2023年以后的
  预期：路由 hybrid，结果全是 KV Cache 相关论文，publish_date >= 2023，score > 0.9

  ---
  用例2 — 纯语义检索
  有哪些关于 prompt tuning 的论文？
  预期：路由 semantic，无时间过滤，结果关于 prompt/soft prompt/prefix tuning，score 合理

  ---
  用例3 — 元数据检索（按标题查具体论文）
  帮我找一篇叫 Attention is All You Need 的论文
  预期：路由 metadata，title 字段被填充，能精确命中或接近命中该论文

  ---
  用例4 — hybrid + 类别过滤
  找一些 cs.CL 分类下关于机器翻译的论文
  预期：路由 hybrid，categories = cs.CL，结果是机器翻译相关论文

  ---
  用例5 — 边界：查无此内容
  最近有哪些关于量子计算纠错码的论文？
  预期：检索结果为空或 score 极低（数据库是 NLP 论文，不应有量子计算内容），renderer 应回复"未检索到"

  ---
  重点验证：

- 用例1 验证 hybrid 时间过滤修复是否生效
- 用例3 验证 metadata 路径
- 用例5 验证空结果处理是否正常（不乱编）
