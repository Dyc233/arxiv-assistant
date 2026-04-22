from pathlib import Path

import chromadb
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from analysis.process import DEFAULT_PARQUET_PATH


CHROMA_DB_DIR = Path(__file__).resolve().parents[1] / "chroma_db"
COLLECTION_NAME = "arxiv_nlp_papers"
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32


def _build_metadata(df: pd.DataFrame) -> list[dict[str, str]]:
    cols = {
        "title": "",
        "publish_date": "",
        "top_conference": "无",
        "authors": "",
        "categories": "",
        "comment": "",
        "url": "",
    }
    meta = {}
    for col, default in cols.items():
        if col in df.columns:
            meta[col] = df[col].fillna(default).astype(str)
        else:
            meta[col] = default
    return pd.DataFrame(meta).to_dict("records")


def upsert_dataframe(
    df: pd.DataFrame,
    chroma_dir: Path = CHROMA_DB_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> int:
    valid_df = df.dropna(subset=["id", "content_to_embed"]).copy()
    if valid_df.empty:
        print("没有可向量化的数据，跳过 Upsert。")
        return 0

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前使用设备: {device.upper()}")
    model = SentenceTransformer(model_name, device=device)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = valid_df["id"].astype(str).tolist()
    documents = valid_df["content_to_embed"].tolist()
    metadatas = _build_metadata(valid_df)

    for start in tqdm(range(0, len(ids), batch_size), desc="Embedding"):
        end = start + batch_size
        batch_docs = documents[start:end]
        embeddings = model.encode(
            batch_docs,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=batch_size,
        ).tolist()
        collection.upsert(
            ids=ids[start:end],
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=metadatas[start:end],
        )

    print(f"已完成 {len(ids)} 篇论文的 Chroma Upsert。")
    return len(ids)


def delete_ids(
    ids: list[str],
    chroma_dir: Path = CHROMA_DB_DIR,
    collection_name: str = COLLECTION_NAME,
) -> int:
    normalized_ids = [str(value) for value in ids if str(value).strip()]
    if not normalized_ids or not Path(chroma_dir).exists():
        return 0

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        return 0

    collection.delete(ids=normalized_ids)
    return len(normalized_ids)


def build_vector_db(
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    chroma_dir: Path = CHROMA_DB_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> int:
    print(f"正在读取清洗结果: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    return upsert_dataframe(
        df,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        model_name=model_name,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    build_vector_db()
