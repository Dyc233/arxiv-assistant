# Findings



  1. 重复实现了两套检索后端，这最明显地违反了“不重复造轮子”。
     retrieval/ 已经有完整的资源加载、过滤、混合检索、重排和格式化链路，但 agent/utils/ 又复制了一套非常接近的实现，包括模型加载、Chroma 连接、候选集向量打
     分、时间过滤和结果格式化。核心重复点在 /D:/CODING/bs/src/agent/utils/search_backend.py:36、/D:/CODING/bs/src/retrieval/resources.py:21、/D:/CODING/bs/
     src/retrieval/service.py:16、/D:/CODING/bs/src/agent/utils/formatters.py:4、/D:/CODING/bs/src/retrieval/formatters.py:4。
     这不只是“代码多一点”，而是维护两套逻辑分支，答辩时老师如果细看，会觉得你在自己堆框架，而不是基于现有模块复用。
  2. 检索主链路里自己实现了较重的“候选集内向量检索 + 手工融合打分”，对本科项目来说显得偏重。
     在混合检索里，你先用 Pandas 做 metadata 过滤，再自己批量 collection.get(..., include=["embeddings"]) 拉 embedding，之后用 torch.mv 手算相似度，再叠加自
     定义的 0.85/0.15 融合权重。见 /D:/CODING/bs/src/retrieval/service.py:124、/D:/CODING/bs/src/retrieval/service.py:182、/D:/CODING/bs/src/retrieval/
     service.py:205、/D:/CODING/bs/src/retrieval/service.py:153。
     从“本科水平代码”这个原则看，这块已经超出了“调用现成向量库完成检索”的叙事，更像是在自己做一个小型检索引擎。工程上不一定错，但展示上容易显得用力过猛。
  3. 增量更新流程做了接近生产级的回滚与恢复机制，超出本科项目通常需要的复杂度。
     updater.py 里有 staged 文件、备份路径、替换恢复、SQLite 显式事务、embedding 回滚删除、异常后状态清理这一整套机制。见 /D:/CODING/bs/src/crawler/
     updater.py:67、/D:/CODING/bs/src/crawler/updater.py:80、/D:/CODING/bs/src/crawler/updater.py:97、/D:/CODING/bs/src/crawler/updater.py:160、/D:/CODING/
     bs/src/crawler/updater.py:183。
     这类设计当然更健壮，但如果你的目标是“看起来像一个合适的本科项目”，这里已经明显偏向生产系统思路了。老师如果按“是否过度设计”来问，这块会是最容易被点名的
     地方。
  4. metadata 检索规则自己造得比较多，尤其是自定义时间 DSL 和模糊评分，不够“借力现成工具”。
     你自己实现了 normalize_text、tokenize_text、SequenceMatcher 模糊匹配、作者/分类/comment 多条件解析，以及 after:/before:/between:/recent: 这套发布时间过
     滤语法。见 /D:/CODING/bs/src/retrieval/filters.py:15、/D:/CODING/bs/src/retrieval/filters.py:30、/D:/CODING/bs/src/retrieval/filters.py:104、/D:/
     CODING/bs/src/retrieval/filters.py:118、/D:/CODING/bs/src/retrieval/filters.py:176。
     如果你的目标是“显得合理、不过度拔高”，这一层完全可以只保留简单的 contains/filter 逻辑。现在这版会让人感觉你在定义一门小查询语言。
  5. 存在一份明显的旧版/平行版检索入口，进一步加重了“重复造轮子”的观感。
     /D:/CODING/bs/src/analysis/query.py:12 里又有一个 PaperSearcher，重复加载 SentenceTransformer、CrossEncoder、Chroma collection，并重复了召回 + rerank
     的流程。这和 retrieval/service.py 的职责高度重合。
     如果这是历史遗留文件，它现在仍在仓库里，就会在答辩材料里形成噪音，因为它让项目看起来像多套方案并存，而不是一条清晰主线。
  6. 数据清洗里有一些手写基础 NLP 工具，严格按你的原则，也属于轻度“重复造轮子”。
     你手写了 stopwords 集、关键词提取和 n-gram 片段拼接逻辑，而项目依赖里其实已经有 nltk，聚类里也用了 TfidfVectorizer。相关位置在 /D:/CODING/bs/src/
     analysis/data_process.py:18、/D:/CODING/bs/src/analysis/data_process.py:71、/D:/CODING/bs/src/analysis/data_process.py:124。
     这块严重性不高，因为实现本身不复杂，也符合教学项目常见写法；但如果非常严格按“尽量用现成工具”，它不算最优。

  判断

  按你给的两个原则看，当前代码库不完全符合。

  更具体地说：

  1. “不重复造轮子”这一条，当前问题比较明显，主要体现在 retrieval/、agent/utils/、analysis/query.py 三处存在平行检索实现。
  2. “本科水平代码，不要做多余工作”这一条，也有几处超线，最典型的是 crawler/updater.py 的恢复机制和 retrieval/service.py 的手工混合检索流程。

  相对符合原则的部分

  /D:/CODING/bs/src/crawler/spider.py:17 和 /D:/CODING/bs/src/analysis/embedder.py:27 这两块整体是合适的。它们基本是在用现成库做清晰直接的事情：arxiv 拉数
  据，SentenceTransformer 编码，ChromaDB 存储，复杂度对本科项目来说是自然的。

  /D:/CODING/bs/src/agent/brain.py:22 这版 Agent 主线也相对克制。它实际上是在复用 retrieval 模块，并借助 Agno 的结构化输出做路由，这个思路本身没问题，反而比
  自己在 Agent 层再写一套检索后端更符合你的两个原则。

  结论

  如果你现在的目标是“让代码看起来像一个成熟但不过度炫技的本科毕设”，那我会给这个检查结果下一个直白判断：

  当前仓库里真正“不像本科项目”的，不是模型或数据规模，而是重复实现太多，以及局部工程化程度过高。
  尤其是下面三块最影响观感：

  1. /D:/CODING/bs/src/agent/utils/search_backend.py:36
  2. /D:/CODING/bs/src/retrieval/service.py:182
  3. /D:/CODING/bs/src/crawler/updater.py:115