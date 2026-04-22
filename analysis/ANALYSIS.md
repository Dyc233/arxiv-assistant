# 分析模块说明

数据来源：`cleaned_papers.parquet` + `cleaned_papers_incremental.parquet`（由 `process.py` 从 SQLite 清洗生成）。各脚本通过 `process.load_cleaned()` 统一加载，去重后分析。

---

## 数据处理 `process.py`

**做什么**：从 SQLite 读取爬虫原始数据，清洗后提取以下字段：

| 字段 | 说明 |
|------|------|
| `publish_date` / `publish_hour` | 解析自 arXiv 发布时间戳 |
| `author_list` | 逗号分隔的作者列表 |
| `category_list` | arXiv 分类标签列表（如 `cs.CL`, `stat.ML`） |
| `top_conference` | 从 `comment` 字段用正则匹配出的顶会（ACL/EMNLP 等） |
| `keywords` | 从标题+摘要提取的去重词列表（简单分词，非 NLP 管道） |
| `content_to_embed` | 供向量化的标题+摘要拼接文本 |

**公共入口**：`load_cleaned()` — 合并主库和增量库，按 arXiv ID 去重，返回完整 DataFrame。

---

## 词云 `wordcloud/global_wordcloud.py`

**方法**：对全量论文标题跑 `CountVectorizer`（1\~3 gram，binary=True），统计每个词组在多少篇论文里出现（文档频率），过滤掉通用学术停用词后，按词频生成词云图。

**为什么用文档频率而非词频**：直接词频会让出现在少数论文中但反复使用的词被放大；文档频率反映的是"有多少篇论文在研究这个方向"，更能代表领域热点。

**输出**
- `global_wordcloud.png`：词云图，词越大表示覆盖论文数越多
- `global_word_frequencies_lib.csv`：Top 160 词组及其文档数

**意义**：一眼看出整个 cs.CL 库的研究热点分布。generation、retrieval、reasoning、alignment 等词大小的相对关系，直接反映了语料库的主题构成。

---

## 发文量趋势 `trends/trend.py`

**方法**：将 `publish_date` 转成月份粒度（`Period('M')`），按月聚合论文数，折线图展示。

**输出**
- `publish_trend.png`：月度发文量折线图
- `publish_trend.csv`：月份 + 论文数

**意义**：可以看出 NLP 论文整体爆发节点（GPT-3/ChatGPT 前后的增速拐点），以及 arXiv 投稿的周期性规律（大会截止日前后的脉冲）。

---

## 顶会录用分布 `confs/conf.py`

**方法**：论文的 `comment` 字段通常包含投稿信息，用正则 `(ACL|EMNLP|NAACL|...)\\s*\\d{4}` 匹配。匹配到的按会议名（去年份）统计数量，水平条形图展示。

**局限**：覆盖率依赖 arXiv 作者是否填写 comment，未填写的归为 None，不参与统计——实际顶会论文数会被低估。

**输出**
- `conference_distribution.png`：各顶会论文数条形图
- `conference_distribution.csv`

**意义**：直观反映本地论文库中各顶会的覆盖比例，也侧面说明 ACL 系列在 cs.CL 爬取结果中的占比。

---

## 高产作者排行 `authors/author.py`

**方法**：展开每篇论文的 `author_list`，对所有作者名做 `Counter`，取 Top 20。

**输出**
- `top_authors.png`：Top 20 作者横向条形图
- `top_authors.csv`

**意义**：识别 cs.CL 领域高产研究者，结合论文库规模可用于衡量数据覆盖的头部作者完整性。注意同名作者会被合并计算（arXiv 没有唯一作者 ID）。

---

## 关键词趋势热力图 `keyword_trend/keyword_trend.py`

**方法**：
1. 先在全量标题上跑 `CountVectorizer`，选出全局 Top 30 高频词组（1\~2 gram）
2. 按季度分组，对每个季度的标题用相同词表统计文档频率，填入矩阵
3. 对矩阵做**行归一化**（每行除以该行最大值），使每个词的峰值 = 1

行归一化的目的是消除绝对数量差异（"generation" 基数大，"alignment" 基数小），让热力图专注展示**各词自身的兴衰曲线**，而非词间的频率对比。

**输出**
- `keyword_trend.png`：关键词 × 季度热力图（颜色越深表示该词在该季度相对越活跃）
- `keyword_trend.csv`：原始（未归一化）计数矩阵

**意义**：这是分析模块里信息密度最高的图。可以读出：某术语（如 "chain thought"、"instruction tuning"）从什么季度开始爆发、持续了多久、是否已经降温——即技术热点的生命周期曲线。

---

## 跨学科渗透度 `crossdomain/crossdomain.py`

**方法**：arXiv 论文可以挂多个分类标签。对于 cs.CL 论文，剔除所有 `cs.*` 标签后，统计剩余非计算机领域标签（eess、q-bio、q-fin、stat、math、physics）的出现频率。按季度计算每个领域的渗透率（有该领域标签的论文占当季总数的百分比）。

**输出**
- `crossdomain_top_categories.png`：全量非 cs.* 标签 Top 20 条形图
- `crossdomain_top_categories.csv`
- `crossdomain_penetration_rate.png`：各领域渗透率随时间变化折线图
- `crossdomain_penetration_rate.csv`

**意义**：反映 NLP 技术向其他学科渗透的趋势。stat.ML 的渗透率变化说明 NLP 与统计机器学习的融合程度；q-bio/q-fin 的上升则直接对应生物医学 NLP 和金融 NLP 的兴起。是说明"NLP 研究影响力扩张"最直观的数据视角。
