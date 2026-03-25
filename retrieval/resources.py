import os
from functools import lru_cache
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import chromadb
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from analysis.data_process import DEFAULT_PARQUET_PATH, read_cleaned_papers
from analysis.embedder import CHROMA_DB_DIR, COLLECTION_NAME, MODEL_NAME


RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class RetrievalResources:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = self._load_embedder()
        self.reranker = self._load_reranker()
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.df = read_cleaned_papers(DEFAULT_PARQUET_PATH)
        self.df_by_id = self.df.set_index("id", drop=False)

    def _load_embedder(self) -> SentenceTransformer:
        model_source = os.getenv("BGE_M3_PATH", MODEL_NAME)
        try:
            return SentenceTransformer(
                model_source,
                device=self.device,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "无法离线加载 embedding 模型。请先缓存 `BAAI/bge-m3`，或设置环境变量 `BGE_M3_PATH` 指向本地模型目录。"
            ) from exc

    def _load_reranker(self) -> CrossEncoder | None:
        model_source = os.getenv("BGE_RERANKER_PATH", RERANKER_MODEL_NAME)
        try:
            return CrossEncoder(
                model_source,
                device=self.device,
                local_files_only=True,
            )
        except Exception:
            return None

    def row_for_id(self, paper_id: str) -> dict[str, Any]:
        if paper_id not in self.df_by_id.index:
            return {
                "id": paper_id,
                "title": "",
                "summary": "",
                "publish_date": "",
                "authors": "",
                "categories": "",
                "comment": "",
                "top_conference": "",
                "url": "",
            }
        row = self.df_by_id.loc[paper_id]
        return {
            "id": str(row.get("id", paper_id)),
            "title": str(row.get("title", "") or ""),
            "summary": str(row.get("summary", "") or ""),
            "publish_date": self._stringify_value(row.get("publish_date", "")),
            "authors": str(row.get("authors", "") or ""),
            "categories": str(row.get("categories", "") or ""),
            "comment": str(row.get("comment", "") or ""),
            "top_conference": str(row.get("top_conference", "") or ""),
            "url": str(row.get("url", "") or ""),
        }

    @staticmethod
    def stringify_filter(value: str | None) -> str:
        return str(value or "").strip()

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return str(value)


@lru_cache(maxsize=1)
def get_retrieval_resources() -> RetrievalResources:
    return RetrievalResources()
