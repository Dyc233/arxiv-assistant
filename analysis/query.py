import os
import math
import torch
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- 配置区 ---
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL_PATH = "BAAI/bge-m3"  # 如果你挪了位置，这里写 D 盘的绝对路径
RERANKER_MODEL_PATH = "BAAI/bge-reranker-v2-m3"

class PaperSearcher:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在初始化环境 (设备: {self.device})...")
        
        # 1. 加载模型（只在初始化时加载一次）
        print(f"加载 Embedding 模型: {EMBEDDING_MODEL_PATH}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_PATH, device=self.device)
        
        print(f"加载 Reranker 模型: {RERANKER_MODEL_PATH}")
        self.reranker = CrossEncoder(RERANKER_MODEL_PATH, device=self.device)
        
        # 2. 连接数据库
        print(f"🗄️ 连接向量数据库: {CHROMA_DB_DIR}")
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.client.get_collection(name="arxiv_nlp_papers")
        print("初始化完成！系统准备就绪。")

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def search(self, query_text, recall_top_k=50, final_top_k=5):
        # 阶段一：向量召回
        query_embeddings = self.embedder.encode(query_text, normalize_embeddings=True).tolist()
        recall_results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=recall_top_k,
            include=["documents", "metadatas", "distances"]
        )

        docs = recall_results['documents'][0]
        metas = recall_results['metadatas'][0]
        ids = recall_results['ids'][0]

        if not docs:
            return []

        # 阶段二：重排序
        sentence_pairs = [[query_text, f"Title: {m.get('title','')}\nAbstract: {d}"] for d, m in zip(docs, metas)]
        scores = self.reranker.predict(sentence_pairs)

        # 组合排序
        results = sorted(zip(scores, ids, docs, metas), key=lambda x: x[0], reverse=True)
        return results[:final_top_k]

def main():
    # 实例化搜索器
    searcher = PaperSearcher()
    
    print("\n" + "="*50)
    print("      NLP 论文语义搜索系统 (BGE-M3 + Reranker)")
    print("      输入 'q', 'exit' 或 '退出' 即可结束对话")
    print("="*50)

    while True:
        query = input("\n请输入查询问题: ").strip()
        
        # 退出逻辑
        if query.lower() in ['q', 'exit', 'quit', '退出', '再见']:
            print("再见！")
            break
        
        if not query:
            continue

        try:
            print(f"正在检索中...")
            results = searcher.search(query)

            if not results:
                print("未找到相关论文，请尝试换个关键词。")
                continue

            print(f"\n为您找到以下最相关的 {len(results)} 篇论文：")
            print("-" * 60)
            
            for i, (score, doc_id, doc, meta) in enumerate(results):
                conf = searcher.sigmoid(score)
                title = meta.get('title', 'N/A')
                date = meta.get('publish_date', 'N/A')
                conf_name = meta.get('top_conference', 'None')
                
                print(f"【{i+1}】 得分: {conf:.4f} | 日期: {date} | 会议: {conf_name}")
                print(f"    标题: {title}")
                print(f"    摘要: {doc[:180]}...")
                print("-" * 40)
                
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main()