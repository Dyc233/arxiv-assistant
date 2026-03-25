from agent.utils.search_backend import SearchResult


def format_search_results(
    results: list[SearchResult],
    query: str,
    mode: str,
    candidate_count: int | None = None,
    applied_filters: dict[str, str] | None = None,
) -> str:
    if not results:
        lines = [f"未找到与 `{query}` 相关的论文结果。"]
        if candidate_count is not None:
            lines.append(f"硬筛后的候选论文数: {candidate_count}")
        if applied_filters:
            filter_desc = " | ".join(f"{key}={value}" for key, value in applied_filters.items() if value)
            if filter_desc:
                lines.append(f"已应用筛选: {filter_desc}")
        return "\n".join(lines)

    lines = [f"检索模式: {mode}", f"查询词: {query}"]
    if candidate_count is not None:
        lines.append(f"硬筛后的候选论文数: {candidate_count}")
    if applied_filters:
        filter_desc = " | ".join(f"{key}={value}" for key, value in applied_filters.items() if value)
        if filter_desc:
            lines.append(f"已应用筛选: {filter_desc}")
    lines.append(f"返回结果数: {len(results)}")
    lines.append("")

    for index, result in enumerate(results, start=1):
        row = result.row
        lines.append(f"{index}. {row.get('title') or 'Untitled'}")
        lines.append(f"   score: {result.score:.4f}")
        lines.append(f"   id: {result.paper_id}")
        lines.append(f"   publish_date: {row.get('publish_date', '')}")
        lines.append(f"   authors: {row.get('authors', '')}")
        lines.append(f"   categories: {row.get('categories', '')}")
        lines.append(f"   top_conference: {row.get('top_conference', '')}")
        lines.append(f"   comment: {row.get('comment', '')}")
        lines.append(f"   url: {row.get('url', '')}")
        summary = (row.get("summary", "") or "").replace("\n", " ").strip()
        lines.append(f"   summary: {summary[:300]}")
        lines.append("")

    return "\n".join(lines).strip()
