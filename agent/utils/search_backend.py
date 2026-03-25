import math
import os
import re
from dataclasses import dataclass
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
DEFAULT_RECALL_TOP_K = 50
DEFAULT_FINAL_TOP_K = 5
EMBEDDING_BATCH_SIZE = 512


@dataclass
class SearchResult:
    score: float
    paper_id: str
    document: str
    metadata: dict[str, Any]
    row: dict[str, Any]


class AgentPaperSearcher:
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
                "无法离线加载 embedding 模型。"
                "请先在本机缓存 `BAAI/bge-m3`，或设置环境变量 `BGE_M3_PATH` 指向本地模型目录。"
            ) from exc

    def _load_reranker(self) -> CrossEncoder | None:
        model_source = os.getenv("BGE_RERANKER_PATH", RERANKER_MODEL_NAME)
        try:
            return CrossEncoder(
                model_source,
                device=self.device,
                local_files_only=True,
            )
        except Exception as exc:
            print(f"[工具执行] -> Reranker 加载失败，回退为纯向量排序: {exc}")
            return None

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def pure_search(
        self,
        query_text: str,
        recall_top_k: int = DEFAULT_RECALL_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> list[SearchResult]:
        query_embeddings = self.embedder.encode(query_text, normalize_embeddings=True).tolist()
        recall_results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=recall_top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = recall_results.get("ids", [[]])[0]
        docs = recall_results.get("documents", [[]])[0]
        metas = recall_results.get("metadatas", [[]])[0]
        if not ids:
            return []

        results: list[SearchResult] = []
        if self.reranker is None:
            distances = recall_results.get("distances", [[]])[0]
            for paper_id, doc, meta, distance in zip(ids, docs, metas, distances, strict=False):
                row = self._row_for_id(paper_id)
                score = 1.0 - float(distance)
                results.append(
                    SearchResult(
                        score=score,
                        paper_id=paper_id,
                        document=doc,
                        metadata=meta or {},
                        row=row,
                    )
                )
            return results[:final_top_k]

        sentence_pairs = [
            [query_text, self._build_rerank_text(doc, self._row_for_id(paper_id))]
            for paper_id, doc in zip(ids, docs, strict=False)
        ]
        scores = self.reranker.predict(sentence_pairs)
        ranked = sorted(
            (
                SearchResult(
                    score=float(score),
                    paper_id=paper_id,
                    document=doc,
                    metadata=meta or {},
                    row=self._row_for_id(paper_id),
                )
                for score, paper_id, doc, meta in zip(scores, ids, docs, metas, strict=False)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:final_top_k]

    def filtered_search(
        self,
        query_text: str,
        published: str | None = None,
        authors: str | None = None,
        categories: str | None = None,
        comment: str | None = None,
        recall_top_k: int = DEFAULT_RECALL_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> tuple[list[SearchResult], int]:
        filtered_df = self.df
        filtered_df = self._apply_published_filter(filtered_df, published)
        filtered_df = self._apply_contains_all_filter(filtered_df, "authors", authors)
        filtered_df = self._apply_contains_all_filter(filtered_df, "categories", categories)
        filtered_df = self._apply_comment_filter(filtered_df, comment)
        candidate_count = len(filtered_df)
        if candidate_count == 0:
            return [], 0

        candidate_ids = filtered_df["id"].astype(str).tolist()
        vector_results = self._vector_search_in_candidates(
            query_text=query_text,
            candidate_ids=candidate_ids,
            recall_top_k=recall_top_k,
        )
        if not vector_results:
            return [], candidate_count

        if self.reranker is None:
            return vector_results[:final_top_k], candidate_count

        sentence_pairs = [
            [query_text, self._build_rerank_text(result.document, result.row)]
            for result in vector_results
        ]
        scores = self.reranker.predict(sentence_pairs)
        reranked = sorted(
            (
                SearchResult(
                    score=float(score),
                    paper_id=result.paper_id,
                    document=result.document,
                    metadata=result.metadata,
                    row=result.row,
                )
                for score, result in zip(scores, vector_results, strict=False)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return reranked[:final_top_k], candidate_count

    def _vector_search_in_candidates(
        self,
        query_text: str,
        candidate_ids: list[str],
        recall_top_k: int,
    ) -> list[SearchResult]:
        query_embedding = self.embedder.encode(query_text, normalize_embeddings=True)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)

        scored_results: list[SearchResult] = []
        for start in range(0, len(candidate_ids), EMBEDDING_BATCH_SIZE):
            batch_ids = candidate_ids[start:start + EMBEDDING_BATCH_SIZE]
            payload = self.collection.get(
                ids=batch_ids,
                include=["documents", "metadatas", "embeddings"],
            )
            ids = payload.get("ids", [])
            docs = payload.get("documents", [])
            metas = payload.get("metadatas", [])
            embeddings = payload.get("embeddings", [])
            if not ids:
                continue

            embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
            scores = torch.mv(embedding_tensor, query_tensor).tolist()
            for score, paper_id, doc, meta in zip(scores, ids, docs, metas, strict=False):
                scored_results.append(
                    SearchResult(
                        score=float(score),
                        paper_id=paper_id,
                        document=doc,
                        metadata=meta or {},
                        row=self._row_for_id(paper_id),
                    )
                )

        scored_results.sort(key=lambda item: item.score, reverse=True)
        return scored_results[:recall_top_k]

    def _row_for_id(self, paper_id: str) -> dict[str, Any]:
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
    def _stringify_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return str(value)

    @staticmethod
    def _split_terms(value: str | None) -> list[str]:
        if not value:
            return []
        return [term.strip() for term in re.split(r"[;,]", value) if term.strip()]

    def _apply_contains_all_filter(
        self,
        df: pd.DataFrame,
        column: str,
        raw_value: str | None,
    ) -> pd.DataFrame:
        terms = self._split_terms(raw_value)
        if not terms:
            return df

        series = df[column].fillna("").astype(str)
        mask = pd.Series(True, index=df.index)
        for term in terms:
            mask &= series.str.contains(re.escape(term), case=False, regex=True, na=False)
        return df.loc[mask]

    def _apply_comment_filter(self, df: pd.DataFrame, raw_value: str | None) -> pd.DataFrame:
        terms = self._split_terms(raw_value)
        if not terms:
            return df

        series = (
            df["comment"].fillna("").astype(str)
            + " "
            + df["top_conference"].fillna("").astype(str)
        )
        mask = pd.Series(True, index=df.index)
        for term in terms:
            mask &= series.str.contains(re.escape(term), case=False, regex=True, na=False)
        return df.loc[mask]

    def _apply_published_filter(self, df: pd.DataFrame, raw_value: str | None) -> pd.DataFrame:
        if not raw_value:
            return df

        value = raw_value.strip()
        series = pd.to_datetime(df["published_ts"], utc=True, errors="coerce")
        lowered = value.lower()

        if re.fullmatch(r"\d{4}", lowered):
            start = pd.Timestamp(f"{lowered}-01-01", tz="UTC")
            end = pd.Timestamp(f"{lowered}-12-31 23:59:59", tz="UTC")
            return df.loc[series.between(start, end)]

        if lowered.startswith("after:"):
            target = self._parse_published_target(value.split(":", 1)[1].strip(), end_of_year=True)
            return df.loc[series > target]

        if lowered.startswith("before:"):
            target = self._parse_published_target(value.split(":", 1)[1].strip(), end_of_year=False)
            return df.loc[series < target]

        if lowered.startswith("equal:"):
            target = value.split(":", 1)[1].strip()
            return self._apply_published_filter(df, target)

        if lowered.startswith("since:"):
            target = self._parse_published_target(value.split(":", 1)[1].strip(), end_of_year=False)
            return df.loc[series >= target]

        if lowered.startswith("between:"):
            payload = value.split(":", 1)[1]
            parts = [part.strip() for part in payload.split(",", 1)]
            if len(parts) != 2:
                raise ValueError("published='between:' 必须写成 between:开始日期,结束日期")
            start = self._parse_published_target(parts[0], end_of_year=False)
            end = self._parse_published_target(parts[1], end_of_year=True)
            return df.loc[series.between(start, end)]

        if lowered.startswith("recent:"):
            payload = value.split(":", 1)[1].strip().lower()
            match = re.fullmatch(r"(\d+)([ymd])", payload)
            if not match:
                raise ValueError("published='recent:' 仅支持 recent:2y / recent:6m / recent:30d")
            amount = int(match.group(1))
            unit = match.group(2)
            now = pd.Timestamp.now(tz="UTC")
            if unit == "y":
                start = now - pd.DateOffset(years=amount)
            elif unit == "m":
                start = now - pd.DateOffset(months=amount)
            else:
                start = now - pd.Timedelta(days=amount)
            return df.loc[series >= start]

        target = self._parse_published_target(value, end_of_year=False)
        return df.loc[series >= target]

    @staticmethod
    def _parse_published_target(value: str, end_of_year: bool) -> pd.Timestamp:
        stripped = value.strip()
        if re.fullmatch(r"\d{4}", stripped):
            suffix = "-12-31 23:59:59" if end_of_year else "-01-01 00:00:00"
            return pd.Timestamp(f"{stripped}{suffix}", tz="UTC")

        parsed = pd.Timestamp(stripped)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")
        return parsed

    @staticmethod
    def _build_rerank_text(document: str, row: dict[str, Any]) -> str:
        title = row.get("title", "")
        summary = row.get("summary", "")
        categories = row.get("categories", "")
        comment = row.get("comment", "")
        return (
            f"Title: {title}\n"
            f"Summary: {summary or document}\n"
            f"Categories: {categories}\n"
            f"Comment: {comment}"
        )


@lru_cache(maxsize=1)
def get_searcher() -> AgentPaperSearcher:
    return AgentPaperSearcher()
