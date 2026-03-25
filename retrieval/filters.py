import re
from difflib import SequenceMatcher

import pandas as pd

from retrieval.schemas import SearchRequest


def split_terms(value: str | None) -> list[str]:
    if not value:
        return []
    return [term.strip() for term in re.split(r"[;,]", value) if term.strip()]


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    lowered = str(value).strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s.+:/-]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def tokenize_text(value: str | None) -> list[str]:
    normalized = normalize_text(value)
    return [token for token in normalized.split(" ") if token]


def text_match_score(query: str | None, value: str | None) -> float:
    normalized_query = normalize_text(query)
    normalized_value = normalize_text(value)
    if not normalized_query or not normalized_value:
        return 0.0
    if normalized_query == normalized_value:
        return 1.0
    if normalized_query in normalized_value:
        return 0.95

    query_tokens = set(tokenize_text(normalized_query))
    value_tokens = set(tokenize_text(normalized_value))
    if not query_tokens or not value_tokens:
        return 0.0

    token_overlap = len(query_tokens & value_tokens) / len(query_tokens)
    sequence_ratio = SequenceMatcher(None, normalized_query, normalized_value).ratio()
    return max(token_overlap * 0.85, sequence_ratio * 0.75)


def contains_all_terms(series: pd.Series, raw_value: str | None) -> pd.Series:
    terms = split_terms(raw_value)
    if not terms:
        return pd.Series(True, index=series.index)

    normalized = series.fillna("").astype(str).str.lower()
    mask = pd.Series(True, index=series.index)
    for term in terms:
        mask &= normalized.str.contains(re.escape(term.lower()), regex=True, na=False)
    return mask


def categories_match_score(requested: str | None, actual: str | None) -> float:
    requested_terms = [term.lower() for term in split_terms(requested)]
    actual_terms = {term.strip().lower() for term in str(actual or "").split(",") if term.strip()}
    if not requested_terms or not actual_terms:
        return 0.0
    matches = sum(1 for term in requested_terms if term in actual_terms)
    return matches / len(requested_terms)


def comment_match_score(requested: str | None, comment: str | None, top_conference: str | None) -> float:
    terms = split_terms(requested)
    if not terms:
        return 0.0
    combined = f"{comment or ''} {top_conference or ''}".strip()
    if not combined:
        return 0.0
    scores = [text_match_score(term, combined) for term in terms]
    return sum(scores) / len(scores)


def row_metadata_score(row: dict[str, str], request: SearchRequest) -> dict[str, float]:
    breakdown: dict[str, float] = {}
    if request.title:
        breakdown["title"] = text_match_score(request.title, row.get("title"))
    if request.authors:
        author_terms = split_terms(request.authors)
        if author_terms:
            scores = [text_match_score(term, row.get("authors")) for term in author_terms]
            breakdown["authors"] = sum(scores) / len(scores)
    if request.categories:
        breakdown["categories"] = categories_match_score(request.categories, row.get("categories"))
    if request.comment:
        breakdown["comment"] = comment_match_score(
            request.comment,
            row.get("comment"),
            row.get("top_conference"),
        )
    if request.published:
        breakdown["published"] = 1.0
    return breakdown


def parse_published_target(value: str, end_of_year: bool) -> pd.Timestamp:
    stripped = value.strip()
    if re.fullmatch(r"\d{4}", stripped):
        suffix = "-12-31 23:59:59" if end_of_year else "-01-01 00:00:00"
        return pd.Timestamp(f"{stripped}{suffix}", tz="UTC")

    parsed = pd.Timestamp(stripped)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


def apply_published_filter(df: pd.DataFrame, raw_value: str | None) -> pd.DataFrame:
    if not raw_value:
        return df

    value = raw_value.strip()
    series = pd.to_datetime(df["published_ts"], utc=True, errors="coerce")
    lowered = value.lower()

    if re.fullmatch(r"\d{4}", lowered):
        start = pd.Timestamp(f"{lowered}-01-01", tz="UTC")
        end = pd.Timestamp(f"{lowered}-12-31 23:59:59", tz="UTC")
        return df.loc[series.between(start, end)]

    if lowered.startswith("after:"):
        target = parse_published_target(value.split(":", 1)[1].strip(), end_of_year=True)
        return df.loc[series > target]

    if lowered.startswith("before:"):
        target = parse_published_target(value.split(":", 1)[1].strip(), end_of_year=False)
        return df.loc[series < target]

    if lowered.startswith("equal:"):
        target = value.split(":", 1)[1].strip()
        return apply_published_filter(df, target)

    if lowered.startswith("since:"):
        target = parse_published_target(value.split(":", 1)[1].strip(), end_of_year=False)
        return df.loc[series >= target]

    if lowered.startswith("between:"):
        payload = value.split(":", 1)[1]
        parts = [part.strip() for part in payload.split(",", 1)]
        if len(parts) != 2:
            raise ValueError("published='between:' 必须写成 between:开始日期,结束日期")
        start = parse_published_target(parts[0], end_of_year=False)
        end = parse_published_target(parts[1], end_of_year=True)
        return df.loc[series.between(start, end)]

    if lowered.startswith("recent:"):
        payload = value.split(":", 1)[1].strip().lower()
        match = re.fullmatch(r"(\d+)([ymd])", payload)
        if not match:
            raise ValueError("published='recent:' 仅支持 recent:2y / recent:6m / recent:30d")
        amount = int(match.group(1))
        unit = match.group(2)
        now = pd.Timestamp.now(tz="UTC")
        if unit == "y":
            start = now - pd.DateOffset(years=amount)
        elif unit == "m":
            start = now - pd.DateOffset(months=amount)
        else:
            start = now - pd.Timedelta(days=amount)
        return df.loc[series >= start]

    target = parse_published_target(value, end_of_year=False)
    return df.loc[series >= target]


def apply_metadata_filters(df: pd.DataFrame, request: SearchRequest) -> pd.DataFrame:
    filtered = apply_published_filter(df, request.published)

    if request.title:
        normalized_title = normalize_text(request.title)
        title_series = filtered["title"].fillna("").astype(str).apply(normalize_text)
        filtered = filtered.loc[title_series.str.contains(re.escape(normalized_title), regex=True, na=False)]

    if request.authors:
        filtered = filtered.loc[contains_all_terms(filtered["authors"], request.authors)]

    if request.categories:
        filtered = filtered.loc[contains_all_terms(filtered["categories"], request.categories)]

    if request.comment:
        combined_series = (
            filtered["comment"].fillna("").astype(str)
            + " "
            + filtered["top_conference"].fillna("").astype(str)
        )
        filtered = filtered.loc[contains_all_terms(combined_series, request.comment)]

    return filtered
