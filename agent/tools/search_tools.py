from retrieval import SearchMode, SearchRequest, format_search_response, get_retrieval_service


def pure_semantic_search(query: str) -> str:
    """
    当用户只表达研究主题，没有明确的时间、作者、分类或顶会/顶刊限制时，调用纯语义检索。

    Args:
        query: 核心研究主题，尽量是英文短语，例如 "agent reasoning"。
    """
    print(f"\n[工具执行] -> 触发纯向量检索 | query='{query}'")
    service = get_retrieval_service()
    response = service.search(
        SearchRequest(
            mode=SearchMode.SEMANTIC,
            query_text=query,
        )
    )
    return format_search_response(response)


def metadata_filtered_search(
    query: str,
    published: str | None = None,
    authors: str | None = None,
    categories: str | None = None,
    comment: str | None = None,
) -> str:
    """
    先做 metadata 硬筛，再在候选集合里做向量检索。

    Args:
        query: 去掉时间、作者、分类、顶会限制后的核心研究主题。
        published: 发布时间过滤。支持:
            - "2024"
            - "after:2024"
            - "before:2023"
            - "equal:2022"
            - "since:2024-01-01"
            - "between:2023-01-01,2024-12-31"
            - "recent:2y"
        authors: 作者过滤，多个作者片段用英文逗号分隔。
        categories: arXiv 分类过滤，多个分类用英文逗号分隔，例如 "cs.CL, cs.AI"。
        comment: 顶会/顶刊/注释过滤，多个关键词用英文逗号分隔，例如 "ACL, Findings"。
    """
    print("\n[工具执行] -> 触发 metadata 硬筛 + 向量检索")
    print(f"   query={query}")
    print(f"   published={published}")
    print(f"   authors={authors}")
    print(f"   categories={categories}")
    print(f"   comment={comment}")

    service = get_retrieval_service()
    response = service.search(
        SearchRequest(
            mode=SearchMode.HYBRID,
            query_text=query,
            published=published,
            authors=authors,
            categories=categories,
            comment=comment,
        )
    )
    return format_search_response(response)
