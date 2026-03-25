from functools import lru_cache

import pandas as pd
import torch

from retrieval.filters import apply_metadata_filters, row_metadata_score
from retrieval.resources import RetrievalResources, get_retrieval_resources
from retrieval.schemas import SearchDiagnostics, SearchMode, SearchRequest, SearchResponse, SearchResult


SEMANTIC_SCORE_WEIGHT = 0.85
METADATA_SCORE_WEIGHT = 0.15
EMBEDDING_BATCH_SIZE = 512


class PaperRetrievalService:
    def __init__(self, resources: RetrievalResources | None = None) -> None:
        self.resources = resources or get_retrieval_resources()

    def search(self, request: SearchRequest) -> SearchResponse:
        if request.mode == SearchMode.METADATA:
            return self.metadata_search(request)
        if request.mode == SearchMode.SEMANTIC:
            return self.semantic_search(request)
        if request.mode == SearchMode.HYBRID:
            return self.hybrid_search(request)
        raise ValueError(f"Unsupported search mode: {request.mode}")

    def metadata_search(self, request: SearchRequest) -> SearchResponse:
        filtered_df = apply_metadata_filters(self.resources.df, request)
        diagnostics = SearchDiagnostics(candidate_count=len(filtered_df), vector_candidate_count=0, reranker_used=False)

        if filtered_df.empty:
            return self._empty_response(request, diagnostics)

        scored_results: list[SearchResult] = []
        for row in filtered_df.to_dict("records"):
            breakdown = row_metadata_score(row, request)
            score = self._aggregate_metadata_score(breakdown, row)
            scored_results.append(
                SearchResult(
                    score=score,
                    paper_id=str(row["id"]),
                    row=self.resources.row_for_id(str(row["id"])),
                    match_source=SearchMode.METADATA.value,
                    score_breakdown=breakdown,
                    metadata={"publish_date": str(row.get("publish_date", ""))},
                )
            )

        scored_results.sort(
            key=lambda item: (item.score, item.row.get("publish_date", "")),
            reverse=True,
        )
        return SearchResponse(
            request=request,
            applied_filters=self._applied_filters(request),
            results=scored_results[: request.top_k],
            diagnostics=diagnostics,
        )

    def semantic_search(self, request: SearchRequest) -> SearchResponse:
        if not request.query_text:
            raise ValueError("Semantic search requires query_text.")

        diagnostics = SearchDiagnostics(reranker_used=False)
        query_embedding = self.resources.embedder.encode(request.query_text, normalize_embeddings=True).tolist()
        recall_results = self.resources.collection.query(
            query_embeddings=[query_embedding],
            n_results=request.recall_top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = recall_results.get("ids", [[]])[0]
        docs = recall_results.get("documents", [[]])[0]
        metas = recall_results.get("metadatas", [[]])[0]
        distances = recall_results.get("distances", [[]])[0]
        diagnostics.candidate_count = len(ids)
        diagnostics.vector_candidate_count = len(ids)
        if not ids:
            return self._empty_response(request, diagnostics)

        results = [
            SearchResult(
                score=1.0 - float(distance),
                paper_id=paper_id,
                row=self.resources.row_for_id(paper_id),
                match_source=SearchMode.SEMANTIC.value,
                score_breakdown={"semantic": 1.0 - float(distance)},
                metadata=meta or {},
                document=doc,
            )
            for paper_id, doc, meta, distance in zip(ids, docs, metas, distances, strict=False)
        ]

        reranked = self._rerank_results(
            query_text=request.query_text,
            results=results,
            use_reranker=request.use_reranker,
            diagnostics=diagnostics,
        )
        return SearchResponse(
            request=request,
            applied_filters=self._applied_filters(request),
            results=reranked[: request.top_k],
            diagnostics=diagnostics,
        )

    def hybrid_search(self, request: SearchRequest) -> SearchResponse:
        has_metadata = any(
            [
                request.title,
                request.authors,
                request.categories,
                request.comment,
                request.published,
            ]
        )
        if not request.query_text:
            if has_metadata:
                return self.metadata_search(request)
            raise ValueError("Hybrid search requires query_text or at least one metadata filter.")

        filtered_df = apply_metadata_filters(self.resources.df, request) if has_metadata else self.resources.df
        diagnostics = SearchDiagnostics(candidate_count=len(filtered_df), reranker_used=False)
        if filtered_df.empty:
            return self._empty_response(request, diagnostics)

        candidate_ids = filtered_df["id"].astype(str).tolist()
        diagnostics.vector_candidate_count = len(candidate_ids)
        semantic_results = self._semantic_search_in_candidates(
            request.query_text,
            candidate_ids=candidate_ids,
            recall_top_k=request.recall_top_k,
        )
        if not semantic_results:
            return self._empty_response(request, diagnostics)

        reranked = self._rerank_results(
            query_text=request.query_text,
            results=semantic_results,
            use_reranker=request.use_reranker,
            diagnostics=diagnostics,
        )

        combined: list[SearchResult] = []
        for result in reranked:
            breakdown = dict(result.score_breakdown)
            row = result.row
            metadata_breakdown = row_metadata_score(row, request)
            metadata_score = self._aggregate_metadata_score(metadata_breakdown, row) if has_metadata else 0.0
            semantic_score = breakdown.get("semantic", result.score)
            combined_score = semantic_score if not has_metadata else (
                semantic_score * SEMANTIC_SCORE_WEIGHT + metadata_score * METADATA_SCORE_WEIGHT
            )
            combined_breakdown = {"semantic": semantic_score}
            combined_breakdown.update(metadata_breakdown)
            if has_metadata:
                combined_breakdown["metadata"] = metadata_score
                combined_breakdown["combined"] = combined_score

            combined.append(
                SearchResult(
                    score=combined_score,
                    paper_id=result.paper_id,
                    row=row,
                    match_source=SearchMode.HYBRID.value,
                    score_breakdown=combined_breakdown,
                    metadata=result.metadata,
                    document=result.document,
                )
            )

        combined.sort(key=lambda item: item.score, reverse=True)
        return SearchResponse(
            request=request,
            applied_filters=self._applied_filters(request),
            results=combined[: request.top_k],
            diagnostics=diagnostics,
        )

    def _semantic_search_in_candidates(
        self,
        query_text: str,
        candidate_ids: list[str],
        recall_top_k: int,
    ) -> list[SearchResult]:
        query_embedding = self.resources.embedder.encode(query_text, normalize_embeddings=True)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)

        scored_results: list[SearchResult] = []
        for start in range(0, len(candidate_ids), EMBEDDING_BATCH_SIZE):
            batch_ids = candidate_ids[start:start + EMBEDDING_BATCH_SIZE]
            payload = self.resources.collection.get(
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
                        row=self.resources.row_for_id(paper_id),
                        match_source=SearchMode.HYBRID.value,
                        score_breakdown={"semantic": float(score)},
                        metadata=meta or {},
                        document=doc,
                    )
                )

        scored_results.sort(key=lambda item: item.score, reverse=True)
        return scored_results[:recall_top_k]

    def _rerank_results(
        self,
        query_text: str,
        results: list[SearchResult],
        use_reranker: bool,
        diagnostics: SearchDiagnostics,
    ) -> list[SearchResult]:
        if not results or not use_reranker or self.resources.reranker is None:
            if use_reranker and self.resources.reranker is None:
                diagnostics.notes.append("Reranker unavailable, kept vector ranking.")
            return results

        sentence_pairs = [
            [query_text, self._build_rerank_text(result)]
            for result in results
        ]
        scores = self.resources.reranker.predict(sentence_pairs)
        reranked = []
        for score, result in zip(scores, results, strict=False):
            breakdown = dict(result.score_breakdown)
            breakdown["rerank"] = float(score)
            reranked.append(
                SearchResult(
                    score=float(score),
                    paper_id=result.paper_id,
                    row=result.row,
                    match_source=result.match_source,
                    score_breakdown=breakdown,
                    metadata=result.metadata,
                    document=result.document,
                )
            )

        diagnostics.reranker_used = True
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    @staticmethod
    def _build_rerank_text(result: SearchResult) -> str:
        row = result.row
        return (
            f"Title: {row.get('title', '')}\n"
            f"Summary: {row.get('summary', '') or result.document}\n"
            f"Categories: {row.get('categories', '')}\n"
            f"Comment: {row.get('comment', '')}"
        )

    @staticmethod
    def _aggregate_metadata_score(breakdown: dict[str, float], row: dict[str, str]) -> float:
        if breakdown:
            return sum(breakdown.values()) / len(breakdown)
        publish_date = str(row.get("publish_date", "") or "")
        return float(publish_date.replace("-", "")) if publish_date else 0.0

    @staticmethod
    def _applied_filters(request: SearchRequest) -> dict[str, str]:
        return {
            "title": request.title or "",
            "authors": request.authors or "",
            "categories": request.categories or "",
            "comment": request.comment or "",
            "published": request.published or "",
        }

    @staticmethod
    def _empty_response(request: SearchRequest, diagnostics: SearchDiagnostics) -> SearchResponse:
        return SearchResponse(
            request=request,
            applied_filters=PaperRetrievalService._applied_filters(request),
            results=[],
            diagnostics=diagnostics,
        )


@lru_cache(maxsize=1)
def get_retrieval_service() -> PaperRetrievalService:
    return PaperRetrievalService()
