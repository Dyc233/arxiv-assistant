"""简化的 retrieval 桥接层 - 供 Agent 使用"""
from enum import Enum
from typing import Any
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from search import PaperSearcher


class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    METADATA = "metadata"
    HYBRID = "hybrid"


class SearchRequest:
    def __init__(self, mode, query_text=None, title=None, authors=None,
                 categories=None, comment=None, published=None, top_k=5, **kwargs):
        self.mode = mode
        self.query_text = query_text
        self.title = title
        self.authors = authors
        self.categories = categories
        self.comment = comment
        self.published = published
        self.top_k = top_k


class SearchResponse:
    def __init__(self, results):
        self.results = results


_searcher = None


def get_retrieval_service():
    """获取检索服务"""
    return RetrievalService()


class RetrievalService:
    def __init__(self):
        global _searcher
        if _searcher is None:
            _searcher = PaperSearcher()
        self.searcher = _searcher

    def search(self, request: SearchRequest) -> SearchResponse:
        """执行检索"""
        if request.mode == SearchMode.SEMANTIC:
            results = self.searcher.semantic_search(
                request.query_text,
                final_top_k=request.top_k
            )
        elif request.mode == SearchMode.METADATA:
            results = self.searcher.metadata_search(
                query_text=request.query_text,
                title=request.title,
                authors=request.authors,
                categories=request.categories,
                comment=request.comment,
                published=request.published,
                top_k=request.top_k
            )
        elif request.mode == SearchMode.HYBRID:
            results = self.searcher.hybrid_search(
                query_text=request.query_text,
                title=request.title,
                authors=request.authors,
                categories=request.categories,
                comment=request.comment,
                published=request.published,
                final_top_k=request.top_k
            )
        else:
            results = []

        return SearchResponse(results)


def format_search_response(response: SearchResponse) -> str:
    """格式化检索结果"""
    if not response.results:
        return "未找到相关论文"

    output = []
    for i, (score, paper_id, doc, meta) in enumerate(response.results):
        title = meta.get('title', 'N/A')
        date = meta.get('publish_date', 'N/A')
        output.append(f"【{i+1}】{title}")
        output.append(f"   发布: {date}")
        output.append(f"   摘要: {doc[:150]}...")
        output.append("")

    return "\n".join(output)


__all__ = [
    "SearchMode",
    "SearchRequest",
    "SearchResponse",
    "get_retrieval_service",
    "format_search_response"
]
