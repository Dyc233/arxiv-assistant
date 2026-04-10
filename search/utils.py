"""检索辅助函数"""
from datetime import datetime, timedelta
from typing import Any


def simple_text_match(query: str, text: str) -> bool:
    """简单的文本匹配：不区分大小写的包含关系"""
    if not query or not text:
        return False
    return query.lower() in text.lower()


def _parse_date(publish_date) -> datetime | None:
    """把 publish_date 字段解析为 datetime，失败返回 None"""
    try:
        if isinstance(publish_date, datetime):
            return publish_date
        s = str(publish_date).strip()
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(s[:len(fmt.replace("%Y", "0000").replace("%m", "00").replace("%d", "00"))], fmt)
            except ValueError:
                continue
    except Exception:
        pass
    return None


def filter_by_time(papers: list[dict[str, Any]], time_query: str | None) -> list[dict[str, Any]]:
    """
    时间过滤，支持格式：
    - 2023 / equal:2023         : 等于该年
    - after:2023                : >= 2023年
    - before:2024               : <= 2024年
    - since:2024-01-01          : >= 该日期
    - between:2023-01-01,2024-12-31 : 日期区间
    - recent:2y / recent:6m     : 最近N年/月
    """
    if not time_query:
        return papers

    q = time_query.strip()
    now = datetime.now()
    filtered = []

    for paper in papers:
        pd = paper.get("publish_date", "")
        if not pd:
            continue

        # 优先用年份快速判断（效率高）
        try:
            year = int(str(pd)[:4])
        except (ValueError, TypeError):
            continue

        keep = False
        if q.isdigit() or q.startswith("equal:") or (len(q) == 7 and q[4] == "-"):
            target = q.split(":")[-1].strip()
            if len(target) == 7 and target[4] == "-":  # YYYY-MM
                keep = str(pd).startswith(target)
            else:
                keep = (year == int(target))
        elif q.startswith("after:"):
            keep = (year >= int(q.split(":")[1]))
        elif q.startswith("before:"):
            keep = (year <= int(q.split(":")[1]))
        elif q.startswith("since:"):
            dt = _parse_date(pd)
            since = _parse_date(q[6:])
            keep = (dt is not None and since is not None and dt >= since)
        elif q.startswith("between:"):
            parts = q[8:].split(",")
            if len(parts) == 2:
                dt = _parse_date(pd)
                lo, hi = _parse_date(parts[0]), _parse_date(parts[1])
                keep = (dt is not None and lo is not None and hi is not None and lo <= dt <= hi)
        elif q.startswith("recent:"):
            spec = q[7:]
            if spec.endswith("y"):
                delta = timedelta(days=365 * int(spec[:-1]))
            elif spec.endswith("m"):
                delta = timedelta(days=30 * int(spec[:-1]))
            else:
                delta = timedelta(days=365)
            dt = _parse_date(pd)
            keep = (dt is not None and dt >= now - delta)
        else:
            keep = True  # 未知格式，保留

        if keep:
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
        filtered = [p for p in filtered if any(simple_text_match(a.strip(), p.get("authors", "")) for a in authors.split(","))]

    if categories:
        filtered = [p for p in filtered if any(simple_text_match(c.strip(), p.get("categories", "")) for c in categories.split(","))]

    if comment:
        filtered = [p for p in filtered if any(simple_text_match(c.strip(), p.get("comment", "")) for c in comment.split(","))]

    return filtered


def metadata_match_score(paper: dict[str, Any], title=None, authors=None, categories=None, comment=None) -> float:
    """根据字段匹配程度打分（0~4），用于无语义查询时的排序"""
    score = 0.0
    if title:
        t = paper.get("title", "").lower()
        q = title.lower()
        score += 1.0 if q == t else (0.6 if q in t else 0)
    if authors:
        score += sum(0.5 for a in authors.split(",") if simple_text_match(a.strip(), paper.get("authors", "")))
    if categories:
        score += sum(0.3 for c in categories.split(",") if simple_text_match(c.strip(), paper.get("categories", "")))
    if comment:
        score += 0.3 if simple_text_match(comment, paper.get("comment", "")) else 0
    return score
