"""检索辅助函数"""
import re
from datetime import datetime
from typing import Any


def simple_text_match(query: str, text: str) -> bool:
    """简单的文本匹配：不区分大小写的包含关系"""
    if not query or not text:
        return False
    return query.lower() in text.lower()


def filter_by_time(papers: list[dict[str, Any]], time_query: str | None) -> list[dict[str, Any]]:
    """
    简单的时间过滤
    支持格式：
    - after:2023 (2023年之后)
    - before:2024 (2024年之前)
    - 2023 (包含2023年)
    """
    if not time_query:
        return papers

    time_query = time_query.strip()
    filtered = []

    for paper in papers:
        publish_date = paper.get("publish_date", "")
        if not publish_date:
            continue

        try:
            if isinstance(publish_date, str):
                year = int(publish_date[:4])
            else:
                year = publish_date.year
        except (ValueError, AttributeError):
            continue

        # 简单的时间过滤逻辑
        if time_query.startswith("after:"):
            target_year = int(time_query.split(":")[1])
            if year >= target_year:
                filtered.append(paper)
        elif time_query.startswith("before:"):
            target_year = int(time_query.split(":")[1])
            if year <= target_year:
                filtered.append(paper)
        elif time_query.isdigit():
            target_year = int(time_query)
            if year == target_year:
                filtered.append(paper)
        else:
            filtered.append(paper)

    return filtered


def filter_by_metadata(
    papers: list[dict[str, Any]],
    title: str | None = None,
    authors: str | None = None,
    categories: str | None = None,
    comment: str | None = None,
) -> list[dict[str, Any]]:
    """简单的元数据过滤：基于字符串包含关系"""
    filtered = papers

    if title:
        filtered = [p for p in filtered if simple_text_match(title, p.get("title", ""))]

    if authors:
        filtered = [p for p in filtered if simple_text_match(authors, p.get("authors", ""))]

    if categories:
        filtered = [p for p in filtered if simple_text_match(categories, p.get("categories", ""))]

    if comment:
        filtered = [p for p in filtered if simple_text_match(comment, p.get("comment", ""))]

    return filtered
