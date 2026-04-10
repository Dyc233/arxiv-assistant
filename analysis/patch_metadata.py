"""一次性脚本：把 authors/categories/comment/url 补写进 ChromaDB（不重新 embedding）"""
import sys
from pathlib import Path

import chromadb
import pandas as pd
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.embedder import CHROMA_DB_DIR, COLLECTION_NAME
from analysis.data_process import DEFAULT_PARQUET_PATH

BATCH = 500


def main():
    df = pd.read_parquet(DEFAULT_PARQUET_PATH)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    print(f"共 {len(df)} 篇论文，开始 patch metadata...")

    for start in tqdm(range(0, len(df), BATCH)):
        batch = df.iloc[start:start + BATCH]
        collection.update(
            ids=batch["id"].astype(str).tolist(),
            metadatas=[
                {
                    "title":          str(r.get("title", "") or ""),
                    "publish_date":   str(r.get("publish_date", "") or ""),
                    "top_conference": str(r.get("top_conference", "") or "无"),
                    "authors":        str(r.get("authors", "") or ""),
                    "categories":     str(r.get("categories", "") or ""),
                    "comment":        str(r.get("comment", "") or ""),
                    "url":            str(r.get("url", "") or ""),
                }
                for _, r in batch.iterrows()
            ],
        )

    print("patch 完成！")


if __name__ == "__main__":
    main()
