---
name: 分析模块三大升级
description: 任务一：Renderer动态引用分析CSV数据；任务二：Dashboard从静态PNG迁移Plotly交互图；任务三：BGE-M3→UMAP→HDBSCAN→Plotly主题聚类
type: project
---

# 任务总览

当前分析模块产出 CSV 和静态 PNG，但存在三个问题：
1. 检索回答完全不引用分析数据——检索和分析是两个割裂的世界
2. 分析看板是静态图片——答辩演示时毫无交互感
3. 缺少主题聚类——关键词趋势告诉你"什么词火了"，但无法回答"NLP研究分成了哪几大方向"

本任务分三部分，互不依赖，可独立执行。

---

---

## 任务一：后端——让 Renderer 动态引用分析数据

## 目标

用户搜 "KV cache compression"，系统不只返回论文列表 + LLM 泛泛总结，而是在回答中融入具体统计数据（技术趋势、高峰季度等）。数据来源是已有的分析 CSV，不是 LLM 编的。

## 涉及文件

### 新建 `agent/analysis_context.py`（约 50-70 行）

核心函数：`lookup_analysis_context(query_text: str) -> str`

逻辑：
1. 对 `query_text` 做简单分词（lower + strip + 去常见停用词）
2. 读取 `analysis/keyword_trend/keyword_trend.csv`（行=术语，列=季度），在行名中做子串匹配
3. 对匹配到的术语，读取各季度数值，识别：峰值季度、近 4 季度趋势方向（上升/下降/平稳）
4. 如有匹配，返回 2-3 句中文洞察；匹配不到返回空字符串 `""`
5. 静默降级——CSV 不存在或匹配失败都不抛异常

### 修改 `agent/prompts.py`

- `build_render_prompt()` 内部调用 `lookup_analysis_context()`，将结果插入 prompt 的"分析上下文"段落
- 当 context 为空时，prompt 中不出现分析上下文相关内容

### 修改 `agent/agent.py`（可能需要）

- `agent/agent.py` 的 `respond()` 方法当前已走 `build_render_prompt()`，如果 `build_render_prompt()` 内部完成了查找，则无需改动
- 注意 `app/search.py` 第 51 行直接 import 并调用 `build_render_prompt()`，不是通过 `agent.respond()`——所以需要在 `build_render_prompt()` 内部完成查找调用，而不是在 `agent.respond()` 里

### 关于 Renderer System Prompt

在 `build_render_prompt()` 返回的 prompt 里，当有分析上下文时，加一句指令：

> 以下是从本库统计中提取的分析洞察，请在回答中自然地引用这些数据（不是照搬原文，而是融入你的分析中）。如果洞察与检索结果一致或矛盾，请指出。

## 前端联动

`app/search.py` 第 51 行直接调用 `build_render_prompt()`，且第 40 行 `st.markdown(turn["insights"])` 直接渲染返回的文本。只要 `build_render_prompt()` 内部完成了分析注入，**前端搜索页无需额外改动**，分析洞察会自动出现在 AI 回答中。

### 可选增强：将分析洞察与论文列表视觉分离

在 `app/search.py` 的 `render_search()` 中，如果想把分析洞察从 AI 回答中拆出来单独高亮展示（例如用一个 `st.info()` 框），可以在 Agent 返回的结果中增加一个字段携带分析上下文。但这不是必须的——直接混在回答文本里已经足够。

---

---

## 任务二：前端——分析看板从静态 PNG 迁移到 Plotly 交互图

## 目标

`app/dashboard.py` 当前用 `st.image()` 展示 7 张静态 PNG。改为 `st.plotly_chart()` 后，用户可以 hover 看数值、缩放、框选，答辩演示时明显提升交互感。

## 涉及文件

### 修改 `app/dashboard.py`（主要改动）

当前架构是读取 PNG 文件路径然后 `st.image()`。改为：

**方案 A（推荐——最小改动）：** 不改分析脚本，在 dashboard 中读取分析 CSV 直接用 Plotly 重绘。

每个 `_chart()` 调用替换为对应的 Plotly 绘图逻辑，例如：
- 发文趋势折线图：读 `publish_trend.csv` → `px.line()`
- 顶会分布条形图：读 `conference_distribution.csv` → `px.bar()`
- 作者排行条形图：读 `top_authors.csv` → `px.bar(orientation='h')`
- 跨学科渗透率折线图：读 `crossdomain_penetration_rate.csv` → `px.line()`
- 关键词热力图：读 `keyword_trend.csv`（行归一化）→ `px.imshow()`
- 词云：Plotly 不支持原生词云，可以保留 PNG 或用 `wordcloud` 库生成后在 Plotly 中展示为 image

**方案 B（重写分析脚本）：** 让分析脚本直接输出 Plotly figure 对象，dashboard 只负责展示。不推荐——改动太大且背离"最少改动"原则。

## 关键约束

- 每个图数据量不大（几十行 CSV），无需缓存优化
- 词云如果难迁移可以暂时保留 PNG，其他 6 张图必须交互化
- 图表配色统一，适合答辩投影（深色背景慎用）
- 所有图保持 `use_container_width=True` 自适应布局

## 前端联动（已有代码无需改）

`app/app.py` 中按钮切换 "分析看板" / "论文检索" 的逻辑不变，`render_dashboard()` 的调用方式不变。这是纯函数体内替换。

---

---

## 任务三：新增——论文主题聚类（BGE-M3 → UMAP → HDBSCAN → Plotly）

### 为什么有必要

关键词趋势热力图告诉你"每个术语什么时候火"，但无法回答"整个 NLP 领域分成了哪几大研究方向"。这是两个不同层次的分析。答辩时评委若问"你的系统对 NLP 领域的整体结构有什么洞察"，只有热力图你答不了。

### 方案选择

| 选项 | 为什么不用 |
|------|-----------|
| K-Means | 需要预设 K，你无法预知 NLP 论文天然分几簇 |
| BERTopic | 底层就是 UMAP + HDBSCAN + c-TF-IDF，封装太厚学习成本高，手写更可控 |
| **UMAP + HDBSCAN** ✅ | 自动决定簇数 + 噪声点标灰，学术论文聚类场景最佳选择 |

### 核心管线

```
Parquet(content_to_embed)                    # 已有
    ↓ batch encode (BGE-M3, 已在 searcher 中加载)
向量矩阵 (N × 1024)
    ↓ UMAP (n_components=2, metric='cosine', random_state=42)
2D 坐标 (N × 2)
    ↓ HDBSCAN (min_cluster_size = N的1%~2%, metric='euclidean')
簇标签 (每个论文一个 cluster_id)
    ↓ 每簇提取 Top-5 关键词 (TF-IDF from title+abstract)
簇名称 = 该簇最具区分性的词汇
    ↓ Plotly 散点图 (x, y, color=cluster, hover=论文标题)
```

### 涉及文件

#### 新建 `analysis/clustering/topic_cluster.py`（约 120-150 行）

脚本结构：
1. `load_cleaned()` 读取全量论文的 `id` + `title` + `content_to_embed`
2. 用 `SentenceTransformer("BAAI/bge-m3")` 做 batch encode（batch_size=32，`show_progress_bar=True`）
3. `umap.UMAP(n_components=2, metric='cosine', random_state=42)` 降维
4. `hdbscan.HDBSCAN(min_cluster_size=..., metric='euclidean')` 聚类
5. 对每个簇，用 `CountVectorizer(stop_words='english', max_features=10)` 提取 top-5 TF-IDF 词作为簇名
6. 输出：
   - `topic_cluster_assignments.csv`（论文 ID + cluster_id + x + y）
   - `topic_cluster_labels.csv`（cluster_id + 簇名）
   - `topic_cluster_scatter.png`（静态预览，调试用）

#### 修改 `app/dashboard.py`

在分析看板中新增一个 section "🧩 论文主题聚类"，用 `st.plotly_chart()` 展示散点图：
- X/Y = UMAP 坐标
- Color = 簇标签（用 `px.scatter` 的 `color` 参数 + 离散色板）
- Hover = 论文标题（截断到 80 字符）
- Size = 4（固定，论文量大会遮盖）
- Legend 放在右侧，支持双击图例项单独查看某个簇

#### 新增依赖

```bash
uv add umap-learn hdbscan
```

两个包总计 < 5MB，无需额外系统依赖。

### 参数调优指南

`min_cluster_size` 是唯一需要调的关键参数：

| 论文总量 | 建议初值 | 预期簇数 |
|---------|---------|---------|
| ~1,000 | 10 | 8~15 |
| ~5,000 | 50 | 15~30 |
| ~10,000+ | 100 | 20~50 |

策略：先跑一次看 `len(set(labels))` 和 `sum(labels == -1)`（噪声点比例），如果噪声点 > 30% 则减小 min_cluster_size；如果簇数 > 50 则增大。迭代 2-3 次即可找到合适值。

### 关键约束

- `random_state=42` 固定，保证可复现
- 编码时 device 设为 `"cuda" if torch.cuda.is_available() else "cpu"`
- 如果论文量 > 20000，UMAP 会很慢，考虑先随机采样 15000 篇聚类后其余论文用 `umap.transform()` 映射
- 聚类是离线计算，不需要做增量更新支持
- 散点图 hover 内容长会卡，只展示截断标题

---

# 优先级建议

| 优先级 | 任务 | 理由 |
|--------|------|------|
| 1 | 任务一（注入 Renderer） | 代码量最小（~60行），答辩叙事价值最高，直接回答"检索和分析是什么关系" |
| 2 | 任务三（主题聚类） | 产出全新分析维度，答辩时最有视觉冲击力的一张图，且能体现技术深度 |
| 3 | 任务二（看板交互化） | 本质是展示层升级，演示效果好但信息增量不如前两者 |

三个任务互不依赖，可并行执行。如果时间只够做两个：**一 + 三**。
