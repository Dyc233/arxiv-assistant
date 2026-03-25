from retrieval.formatters import format_search_response
from retrieval.service import PaperRetrievalService, get_retrieval_service
from retrieval.schemas import SearchDiagnostics, SearchMode, SearchRequest, SearchResponse, SearchResult

__all__ = [
    "PaperRetrievalService",
    "SearchDiagnostics",
    "SearchMode",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "format_search_response",
    "get_retrieval_service",
]

'''
  - /D:/CODING/BS/src/retrieval/__init__.py
    对外导出统一入口，给别的模块直接 from retrieval import ... 用。
  - /D:/CODING/BS/src/retrieval/schemas.py
    放统一的数据结构：SearchMode、SearchRequest、SearchResult、SearchResponse、SearchDiagnostics。
  - /D:/CODING/BS/src/retrieval/resources.py
    管理底层资源：parquet 数据、Chroma collection、embedding 模型、reranker 模型。
  - /D:/CODING/BS/src/retrieval/filters.py
    处理纯 metadata 过滤和字段匹配打分，比如 title/authors/categories/comment/published。
  - /D:/CODING/BS/src/retrieval/service.py
    检索主入口，统一调度 metadata_search、semantic_search、hybrid_search。
  - /D:/CODING/BS/src/retrieval/formatters.py
    把结构化 SearchResponse 渲染成可读文本，方便 CLI 和后续 agent 直接复用。
  - /D:/CODING/BS/src/retrieval/test_retrieval.py
    独立测试脚本，用来绕开 agent 直接验证 retrieval 后端。
'''
