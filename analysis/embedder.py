import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# 1. 基础配置
PARQUET_PATH = "analysis/cleaned_papers.parquet"  
CHROMA_DB_DIR = "chroma_db"         
MODEL_NAME = "BAAI/bge-m3"              
BATCH_SIZE = 32                         

def build_vector_db():
    # 强制开启 CUDA 优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True 
    
    print("1. 检查 GPU 环境...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   当前使用设备: {device.upper()}")
    if device == "cpu":
        print("   [警告] 未检测到 CUDA，将使用 CPU 缓慢运行！请检查 PyTorch 是否安装了 GPU 版本。")

    print(f"2. 正在加载 {MODEL_NAME} 模型到显存... (首次运行会自动下载约 2GB 的模型权重，请保持网络畅通)")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("3. 初始化 ChromaDB 本地向量库...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    # 使用余弦相似度进行检索，这是业界标准
    collection = chroma_client.get_or_create_collection(
        name="arxiv_nlp_papers",
        metadata={"hnsw:space": "cosine"} 
    )

    print(f"4. 正在读取 Parquet 数据: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    
    # 过滤掉可能的空异常值
    df = df.dropna(subset=['id', 'content_to_embed'])
    total_docs = len(df)
    print(f"   共发现 {total_docs} 篇有效论文准备入库。")

    print("5. 开始批量 Embedding 并存入 ChromaDB...")
    
    ids = df['id'].astype(str).tolist()
    documents = df['content_to_embed'].tolist()
    
    # 构建元数据 (Metadata)。
    # 注意：ChromaDB 的元数据值不能是数组 (List)，只能是字符串或数字。
    # 所以我们不存 keywords，只存后面检索 Agent 需要用到的结构化过滤字段。
    metadatas = df[['title', 'publish_date', 'top_conference']].copy()
    metadatas['top_conference'] = metadatas['top_conference'].fillna("None")
    metadatas['publish_date'] = metadatas['publish_date'].astype(str)
    metadata_list = metadatas.to_dict('records')

    current_count = collection.count()
    print(f"检测到数据库已有 {current_count} 条记录，将跳过已处理部分。")
    # 使用进度条分批处理
    for i in tqdm(range(current_count, total_docs, BATCH_SIZE), desc="向量化进度"):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_metas = metadata_list[i : i + BATCH_SIZE]

        # 增加异常处理，防止单批次损坏导致整个任务崩掉
        try:
            embeddings = model.encode(
                batch_docs, 
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=BATCH_SIZE # 利用模型内部的并行
            ).tolist()

            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas
            )
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

    print(f"\n大功告成！{total_docs} 篇论文的语义向量已全部落盘至 {CHROMA_DB_DIR}。")

if __name__ == "__main__":
    build_vector_db()