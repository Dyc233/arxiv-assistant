from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SearchMode(StrEnum):
    METADATA = "metadata"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass(slots=True)
class SearchRequest:
    mode: SearchMode
    query_text: str | None = None
    title: str | None = None
    authors: str | None = None
    categories: str | None = None
    comment: str | None = None
    published: str | None = None
    top_k: int = 5
    recall_top_k: int = 50
    use_reranker: bool = True


@dataclass(slots=True)
class SearchResult:
    score: float
    paper_id: str
    row: dict[str, Any]
    match_source: str
    score_breakdown: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    document: str = ""


@dataclass(slots=True)
class SearchDiagnostics:
    candidate_count: int = 0
    vector_candidate_count: int = 0
    reranker_used: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SearchResponse:
    request: SearchRequest
    applied_filters: dict[str, str]
    results: list[SearchResult]
    diagnostics: SearchDiagnostics
