"""简化版检索工具 - 供 Agent 调用"""
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from search import PaperSearcher


# 全局单例，避免重复加载模型
_searcher = None


def get_searcher():
    """获取检索器单例"""
    global _searcher
    if _searcher is None:
        _searcher = PaperSearcher()
    return _searcher


def pure_semantic_search(query: str) -> str:
    """
    纯语义检索：当用户只表达研究主题，没有时间、作者等限制时使用

    Args:
        query: 研究主题，例如 "transformer attention mechanism"
    """
    print(f"\n[工具执行] 纯语义检索 | query='{query}'")
    searcher = get_searcher()
    results = searcher.semantic_search(query, final_top_k=5)
    return _format_results(results)


def metadata_filtered_search(
    query: str,
    published: str | None = None,
    authors: str | None = None,
    categories: str | None = None,
    comment: str | None = None,
) -> str:
    """
    混合检索：先用元数据过滤，再做语义检索

    Args:
        query: 研究主题
        published: 时间过滤，支持 "2024", "after:2023", "before:2024"
        authors: 作者关键词
        categories: 分类关键词
        comment: 会议/评论关键词
    """
    print(f"\n[工具执行] 混合检索")
    print(f"   query={query}")
    print(f"   published={published}, authors={authors}")
    print(f"   categories={categories}, comment={comment}")

    searcher = get_searcher()
    results = searcher.hybrid_search(
        query_text=query,
        title=None,
        authors=authors,
        categories=categories,
        comment=comment,
        published=published,
        final_top_k=5
    )
    return _format_results(results)


def _format_results(results) -> str:
    """格式化检索结果为文本"""
    if not results:
        return "未找到相关论文"

    output = []
    for i, (score, paper_id, doc, meta) in enumerate(results):
        title = meta.get('title', 'N/A')
        date = meta.get('publish_date', 'N/A')
        authors = meta.get('authors', 'N/A')
        conf = meta.get('top_conference', 'None')

        output.append(f"【{i+1}】{title}")
        output.append(f"   发布: {date} | 会议: {conf}")
        output.append(f"   作者: {authors[:100]}...")
        output.append(f"   摘要: {doc[:200]}...")
        output.append("")

    return "\n".join(output)

