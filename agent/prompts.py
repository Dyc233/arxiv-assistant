from agent.schemas import ResponseMode, RoutingDecision
from retrieval import SearchResponse


ROUTER_DESCRIPTION = "你是一个论文检索路由器，只负责把用户请求拆成结构化检索意图。"

ROUTER_INSTRUCTIONS = [
    "1. 你不能直接回答论文内容，只能输出结构化路由结果。",
    "2. 如果用户主要是想精准定位某篇或某几篇论文，task_type 选 lookup。",
    "3. 如果用户主要是想找一批相关论文但不要求长篇分析，task_type 选 search。",
    "4. 如果用户明确要求总结、综述、比较、分析趋势或方法路线，task_type 选 report 或 lookup_then_report。",
    "5. response_mode 规则：精准定位和纯列表需求用 raw_list；找论文并希望有少量见解用 list_with_insights；明确要求综述分析用 report。",
    "6. search_mode 规则：纯 title/author/published/categories/comment 精准条件优先用 metadata；纯主题探索用 semantic；既有主题又有 metadata 约束时用 hybrid。",
    "7. query_text 只保留研究主题，不要把年份、作者、分类、会议这些约束重复塞进去。",
    "8. published 只允许这些格式：2024 / after:2024 / before:2023 / equal:2022 / since:2024-01-01 / between:2023-01-01,2024-12-31 / recent:2y。",
    "9. authors、categories、comment 多个条件用英文逗号分隔。",
    "10. title 字段只在用户明显提供论文标题或标题片段时填写。",
]


def build_render_prompt(
    user_input: str,
    routing: RoutingDecision,
    search_response: SearchResponse,
) -> str:
    filters = {key: value for key, value in search_response.applied_filters.items() if value}
    results = []
    for result in search_response.results:
        row = result.row
        results.append(
            {
                "title": row.get("title", ""),
                "id": result.paper_id,
                "publish_date": row.get("publish_date", ""),
                "authors": row.get("authors", ""),
                "categories": row.get("categories", ""),
                "top_conference": row.get("top_conference", ""),
                "comment": row.get("comment", ""),
                "url": row.get("url", ""),
                "summary": row.get("summary", ""),
                "score": round(result.score, 4),
                "match_source": result.match_source,
                "score_breakdown": result.score_breakdown,
            }
        )

    mode_instruction = _response_mode_instruction(routing.response_mode)
    return (
        "你是一个 NLP 论文助手，负责基于已经检索好的结果向用户作答。\n"
        "你不能编造不在结果里的论文，也不能改写标题、作者、日期、链接。\n"
        "如果结果为空，就明确说明没有检索到匹配论文，并简短复述检索条件。\n"
        f"{mode_instruction}\n\n"
        f"用户原始需求:\n{user_input}\n\n"
        f"路由结果:\n{routing.model_dump_json(indent=2)}\n\n"
        f"检索诊断:\n"
        f"- search_mode: {search_response.request.mode.value}\n"
        f"- candidate_count: {search_response.diagnostics.candidate_count}\n"
        f"- vector_candidate_count: {search_response.diagnostics.vector_candidate_count}\n"
        f"- reranker_used: {search_response.diagnostics.reranker_used}\n"
        f"- applied_filters: {filters}\n\n"
        f"检索结果(JSON):\n{results}\n"
    )


def _response_mode_instruction(mode: ResponseMode) -> str:
    if mode == ResponseMode.RAW_LIST:
        return (
            "输出要求: 只输出忠实的原始结果列表。"
            " 不要写长篇分析。每条结果展示 title、authors、publish_date、categories、venue/comment、url、score。"
        )
    if mode == ResponseMode.LIST_WITH_INSIGHTS:
        return (
            "输出要求: 先给出结果列表，再给 2 到 4 条简短见解。"
            " 见解必须严格基于结果列表，例如主题分布、时间分布、venue 分布、方法趋势。"
        )
    return (
        "输出要求: 基于结果生成一份结构化综述报告。"
        " 先简述检索范围和代表论文，再总结研究主题、方法路线、趋势与局限。"
    )
