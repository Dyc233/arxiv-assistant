"""简化版论文检索系统 - 支持三种检索模式"""
import os
import math
import torch
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

from search.utils import filter_by_time, filter_by_metadata


# 配置
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
PARQUET_PATH = "analysis/cleaned_papers.parquet"


class PaperSearcher:
    """论文检索器 - 支持语义检索、元数据检索、混合检索"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"初始化检索系统 (设备: {self.device})...")

        # 加载模型
        print(f"加载 Embedding 模型: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=self.device)

        print(f"加载 Reranker 模型: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL, device=self.device)

        # 连接向量数据库
        print(f"连接向量数据库: {CHROMA_DB_DIR}")
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.client.get_collection(name="arxiv_nlp_papers")

        # 加载元数据
        if os.path.exists(PARQUET_PATH):
            print(f"加载元数据: {PARQUET_PATH}")
            self.df = pd.read_parquet(PARQUET_PATH)
        else:
            print("警告: 未找到元数据文件，元数据检索将不可用")
            self.df = None

        print("初始化完成！")

    @staticmethod
    def sigmoid(x):
        """将分数转换为0-1之间的置信度"""
        return 1 / (1 + math.exp(-x))

    def semantic_search(self, query_text, recall_top_k=50, final_top_k=5, use_reranker=True):
        """
        纯语义检索：基于向量相似度

        Args:
            query_text: 查询文本
            recall_top_k: 召回数量
            final_top_k: 最终返回数量
            use_reranker: 是否使用重排序
        """
        # 向量召回
        query_embedding = self.embedder.encode(query_text, normalize_embeddings=True).tolist()
        recall_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=recall_top_k,
            include=["documents", "metadatas", "distances"]
        )

        docs = recall_results['documents'][0]
        metas = recall_results['metadatas'][0]
        ids = recall_results['ids'][0]
        distances = recall_results['distances'][0]

        if not docs:
            return []

        # 重排序（可选）
        if use_reranker:
            sentence_pairs = [
                [query_text, f"Title: {m.get('title', '')}\nAbstract: {d}"]
                for d, m in zip(docs, metas)
            ]
            scores = self.reranker.predict(sentence_pairs)
            results = sorted(
                zip(scores, ids, docs, metas),
                key=lambda x: x[0],
                reverse=True
            )
        else:
            # 不使用重排序，直接用距离转换为相似度分数
            scores = [1.0 - d for d in distances]
            results = list(zip(scores, ids, docs, metas))

        return results[:final_top_k]

    def metadata_search(self, title=None, authors=None, categories=None, comment=None, published=None, top_k=10):
        """
        纯元数据检索：基于字段匹配

        Args:
            title: 标题关键词
            authors: 作者关键词
            categories: 分类关键词
            comment: 评论/会议关键词
            published: 发布时间（支持 "2023", "after:2023", "before:2024"）
            top_k: 返回数量
        """
        if self.df is None:
            print("错误: 元数据文件未加载")
            return []

        # 转换为字典列表
        papers = self.df.to_dict('records')

        # 应用过滤
        filtered = filter_by_metadata(papers, title, authors, categories, comment)
        filtered = filter_by_time(filtered, published)

        # 按发布时间排序（最新的在前）
        filtered.sort(key=lambda x: x.get('publish_date', ''), reverse=True)

        # 格式化返回结果（与语义检索保持一致的格式）
        results = []
        for paper in filtered[:top_k]:
            results.append((
                1.0,  # 元数据匹配给固定分数
                paper.get('id', ''),
                paper.get('summary', paper.get('abstract', '')),
                {
                    'title': paper.get('title', ''),
                    'publish_date': paper.get('publish_date', ''),
                    'authors': paper.get('authors', ''),
                    'categories': paper.get('categories', ''),
                    'top_conference': paper.get('top_conference', ''),
                    'comment': paper.get('comment', ''),
                    'url': paper.get('url', ''),
                }
            ))

        return results

    def hybrid_search(self, query_text, title=None, authors=None, categories=None, comment=None, published=None, recall_top_k=50, final_top_k=5, use_reranker=True):
        """
        混合检索：先用元数据过滤，再做语义检索

        Args:
            query_text: 查询文本
            title, authors, categories, comment, published: 元数据过滤条件
            recall_top_k: 召回数量
            final_top_k: 最终返回数量
            use_reranker: 是否使用重排序
        """
        if self.df is None:
            print("警告: 元数据文件未加载，降级为纯语义检索")
            return self.semantic_search(query_text, recall_top_k, final_top_k, use_reranker)

        # 第一步：元数据过滤
        papers = self.df.to_dict('records')
        filtered = filter_by_metadata(papers, title, authors, categories, comment)
        filtered = filter_by_time(filtered, published)

        if not filtered:
            print("元数据过滤后无结果")
            return []

        # 获取过滤后的 ID 列表
        candidate_ids = [str(p['id']) for p in filtered]
        print(f"元数据过滤后剩余 {len(candidate_ids)} 篇论文")

        # 第二步：在候选集中做语义检索
        query_embedding = self.embedder.encode(query_text, normalize_embeddings=True).tolist()

        # 从 ChromaDB 中只查询候选集
        recall_results = self.collection.get(
            ids=candidate_ids[:recall_top_k],
            include=["documents", "metadatas", "embeddings"]
        )

        ids = recall_results.get('ids', [])
        docs = recall_results.get('documents', [])
        metas = recall_results.get('metadatas', [])
        embeddings = recall_results.get('embeddings', [])

        if not ids:
            return []

        # 计算相似度
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
        scores = []
        for emb in embeddings:
            emb_tensor = torch.tensor(emb, dtype=torch.float32)
            similarity = torch.dot(query_tensor, emb_tensor).item()
            scores.append(similarity)

        # 第三步：重排序（可选）
        if use_reranker:
            sentence_pairs = [
                [query_text, f"Title: {m.get('title', '')}\nAbstract: {d}"]
                for d, m in zip(docs, metas)
            ]
            rerank_scores = self.reranker.predict(sentence_pairs)
            results = sorted(
                zip(rerank_scores, ids, docs, metas),
                key=lambda x: x[0],
                reverse=True
            )
        else:
            results = sorted(
                zip(scores, ids, docs, metas),
                key=lambda x: x[0],
                reverse=True
            )

        return results[:final_top_k]

    def search(self, mode="semantic", query_text=None, **kwargs):
        """
        统一搜索接口

        Args:
            mode: 检索模式 ("semantic", "metadata", "hybrid")
            query_text: 查询文本（semantic 和 hybrid 必需）
            **kwargs: 其他参数
        """
        if mode == "semantic":
            if not query_text:
                raise ValueError("语义检索需要提供 query_text")
            return self.semantic_search(query_text, **kwargs)
        elif mode == "metadata":
            return self.metadata_search(**kwargs)
        elif mode == "hybrid":
            if not query_text:
                raise ValueError("混合检索需要提供 query_text")
            return self.hybrid_search(query_text, **kwargs)
        else:
            raise ValueError(f"不支持的检索模式: {mode}")

    def format_results(self, results):
        """格式化输出结果"""
        if not results:
            print("未找到相关论文")
            return

        print(f"\n找到 {len(results)} 篇相关论文：")
        print("-" * 80)

        for i, (score, paper_id, doc, meta) in enumerate(results):
            conf = self.sigmoid(score) if score < 10 else score
            title = meta.get('title', 'N/A')
            date = meta.get('publish_date', 'N/A')
            conf_name = meta.get('top_conference', 'None')

            print(f"【{i+1}】 得分: {conf:.4f} | 日期: {date} | 会议: {conf_name}")
            print(f"    标题: {title}")
            print(f"    摘要: {doc[:150]}...")
            print("-" * 80)

