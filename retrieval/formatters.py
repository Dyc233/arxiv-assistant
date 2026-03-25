from retrieval.schemas import SearchResponse


def format_search_response(response: SearchResponse) -> str:
    request = response.request
    results = response.results
    lines = [
        f"检索模式: {request.mode.value}",
        f"查询词: {request.query_text or ''}",
        f"候选论文数: {response.diagnostics.candidate_count}",
        f"参与向量排序的候选数: {response.diagnostics.vector_candidate_count}",
        f"返回结果数: {len(results)}",
    ]

    filter_desc = " | ".join(f"{key}={value}" for key, value in response.applied_filters.items() if value)
    if filter_desc:
        lines.append(f"已应用筛选: {filter_desc}")
    if response.diagnostics.reranker_used:
        lines.append("已使用 reranker 重排")
    if response.diagnostics.notes:
        lines.extend(response.diagnostics.notes)
    lines.append("")

    if not results:
        lines.append("未找到匹配结果。")
        return "\n".join(lines)

    for index, result in enumerate(results, start=1):
        row = result.row
        lines.append(f"{index}. {row.get('title') or 'Untitled'}")
        lines.append(f"   score: {result.score:.4f}")
        lines.append(f"   source: {result.match_source}")
        lines.append(f"   id: {result.paper_id}")
        lines.append(f"   publish_date: {row.get('publish_date', '')}")
        lines.append(f"   authors: {row.get('authors', '')}")
        lines.append(f"   categories: {row.get('categories', '')}")
        lines.append(f"   top_conference: {row.get('top_conference', '')}")
        lines.append(f"   comment: {row.get('comment', '')}")
        lines.append(f"   url: {row.get('url', '')}")
        if result.score_breakdown:
            breakdown = ", ".join(f"{key}={value:.4f}" for key, value in result.score_breakdown.items())
            lines.append(f"   score_breakdown: {breakdown}")
        summary = (row.get("summary", "") or "").replace("\n", " ").strip()
        lines.append(f"   summary: {summary[:300]}")
        lines.append("")

    return "\n".join(lines).strip()
