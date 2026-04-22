# 项目概述

arXiv NLP 论文检索与分析助手，本科毕设原型。围绕 cs.CL 类别论文，搭建从数据采集到检索再到中文结果呈现的完整链路。

## 系统架构

```
data/spider.py          ← 从 arXiv 抓取论文，落地 SQLite
analysis/process.py     ← 清洗，输出 Parquet；提取 publish_date / top_conference / keywords 等字段
data/embedder.py        ← 向量化 content_to_embed，写入本地 ChromaDB
search/searcher.py      ← 三种检索模式：semantic / metadata / hybrid
agent/agent.py          ← 两阶段 Agent：Router 判意图 → Renderer 输出中文结果
```

增量更新由 `data/updater.py` 驱动，`analysis/process.py` 提供 `load_cleaned()` 作为所有分析脚本的统一数据入口。

## 分析模块（`analysis/`）

| 脚本 | 输出 | 说明 |
|------|------|------|
| `wordcloud/global_wordcloud.py` | 词云图 + CSV | 标题 n-gram 文档频率，反映研究热点分布 |
| `trends/trend.py` | 折线图 + CSV | 按月发文量，可见 ChatGPT 前后的增速拐点 |
| `confs/conf.py` | 条形图 + CSV | 从 comment 字段提取顶会（ACL/EMNLP 等）录用分布 |
| `authors/author.py` | 条形图 + CSV | 高产作者 Top 20 |
| `keyword_trend/keyword_trend.py` | 热力图 + CSV | Top 30 词组 × 季度，行归一化，展示术语兴衰曲线 |
| `crossdomain/crossdomain.py` | 条形图 + 折线图 + CSV | 非 cs.* 标签渗透率，反映 NLP 向其他学科扩张的趋势 |

详细方法说明见 `analysis/ANALYSIS.md`。
