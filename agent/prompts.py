from agent.schemas import ResponseMode, RoutingDecision


ROUTER_DESCRIPTION = "你是一个论文检索路由器，只负责把用户请求拆成结构化检索意图。"

ROUTER_INSTRUCTIONS = [
    "1. 你不能直接回答论文内容，只能输出结构化路由结果。",
    "2. response_mode 规则：精准定位和纯列表需求用 raw_list；找论文并希望有少量见解用 list_with_insights；明确要求综述分析用 report。",
    "3. search_mode 规则：纯 title/author/published/categories/comment 精准条件优先用 metadata；纯主题探索用 semantic；既有主题又有 metadata 约束时用 hybrid。",
    "4. query_text 只保留研究主题，不要把年份、作者、分类、会议这些约束重复塞进去。",
    "5. published 只允许这些格式：2024 / 2024-03 / after:2024 / before:2023 / equal:2022 / since:2024-01-01 / between:2023-01-01,2024-12-31 / recent:2y。",
    "6. authors、categories、comment 多个条件用英文逗号分隔。comment 只能填会议/期刊名的缩写，候选范围：ACL,EMNLP,NAACL,EACL,AACL,COLING,CVPR,ICLR,NeurIPS,ICML,AAAI,IJCAI,SIGIR,WWW。禁止填'顶会''顶刊'等描述性词语。",
    "7. title 字段只在用户明显提供论文标题或标题片段时填写。",
]


def build_render_prompt(user_input: str, routing: RoutingDecision, results: list) -> str:
    brief_results = [
        {
            "title": meta.get("title", ""),
            "publish_date": meta.get("publish_date", ""),
            "categories": meta.get("categories", ""),
            "top_conference": meta.get("top_conference", "") or "",
            "summary": doc[:300],
        }
        for _score, _paper_id, doc, meta in results
    ]

    mode_instruction = _response_mode_instruction(routing.response_mode)
    return (
        "你是一个 NLP 论文助手，论文列表已由前端展示，你只需要完成分析部分。\n"
        "不要输出论文列表，不要重复标题和链接。\n"
        "如果结果为空，就明确说明没有检索到匹配论文，并简短复述检索条件。\n"
        f"{mode_instruction}\n\n"
        f"用户原始需求:\n{user_input}\n\n"
        f"路由结果:\n{routing.model_dump_json(indent=2)}\n\n"
        f"检索到的论文（精简版，供分析用）:\n{brief_results}\n"
    )


def _response_mode_instruction(mode: ResponseMode) -> str:
    if mode == ResponseMode.RAW_LIST:
        return (
            "输出要求: 论文列表已由前端卡片展示，你不需要重复输出。"
            " 只需用一句话确认检索完成（如'已为你找到 N 篇相关论文'），然后给出1-2个简短的检索技巧提示（如可尝试缩小时间范围、添加作者筛选等）。"
        )
    if mode == ResponseMode.LIST_WITH_INSIGHTS:
        return (
            "输出要求: 先给出结果列表，再给一些简短的见解。"
            " 见解必须严格基于结果列表，例如主题分布、时间分布、分类分布、方法趋势。"
        )
    return (
        "输出要求: 基于结果生成一份结构化综述报告。"
        " 先简述检索范围和代表论文，再总结研究主题、方法路线、趋势与局限。"
    )
