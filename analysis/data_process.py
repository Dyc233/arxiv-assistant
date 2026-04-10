import re
import shutil
import sqlite3
import uuid
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "arxiv_papers.db"
DEFAULT_PARQUET_PATH = Path(__file__).resolve().parent / "cleaned_papers.parquet"
DEFAULT_INCREMENTAL_PATH = Path(__file__).resolve().parent / "cleaned_papers_incremental.parquet"
CONFERENCE_REGEX = re.compile(
    r"(ACL|EMNLP|NAACL|EACL|AACL|COLING|CVPR|ICLR|NeurIPS|ICML|IJCNLP|AAAI|IJCAI|SIGIR|WWW)\s*\d{4}",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}
CLEANED_COLUMNS = [
    "id",
    "title",
    "summary",
    "published",
    "published_ts",
    "publish_date",
    "publish_hour",
    "authors",
    "author_list",
    "categories",
    "category_list",
    "comment",
    "top_conference",
    "url",
    "content_to_embed",
    "keywords",
]


def load_raw_papers(db_path: Path = DEFAULT_DB_PATH, sql: str = "SELECT * FROM papers") -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _extract_top_conference(comment: str | None) -> str:
    match = CONFERENCE_REGEX.search(comment or "")
    return match.group(0).upper() if match else "None"


def _extract_keywords(title: str, summary: str, limit: int = 64) -> list[str]:
    """简化版关键词提取：只提取单词，不做 n-gram"""
    text = re.sub(r"[^a-z\s]", " ", f"{title} {summary}".lower())
    tokens = [token for token in text.split() if token not in STOPWORDS and len(token) > 2]
    # 去重并限制数量
    seen = set()
    keywords = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= limit:
                break
    return keywords


def empty_cleaned_papers() -> pd.DataFrame:
    return pd.DataFrame(columns=CLEANED_COLUMNS)


def _normalize_cleaned_papers(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in CLEANED_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    if "published_ts" in normalized.columns:
        normalized["published_ts"] = pd.to_datetime(normalized["published_ts"], utc=True, errors="coerce")
    if "publish_hour" in normalized.columns:
        normalized["publish_hour"] = pd.array(normalized["publish_hour"], dtype="Int64")
    if "publish_date" in normalized.columns:
        normalized["publish_date"] = normalized["publish_date"].apply(
            lambda v: v.isoformat() if hasattr(v, "isoformat") and not isinstance(v, str) else (str(v) if not pd.isna(v) else "")
        ).astype(str)
    return normalized[CLEANED_COLUMNS]


def clean_papers(rows_or_df: list[dict[str, str]] | pd.DataFrame) -> pd.DataFrame:
    raw_df = rows_or_df.copy() if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    if raw_df.empty:
        return empty_cleaned_papers()

    df = raw_df.drop_duplicates(subset=["id"]).copy()
    for column in ["title", "summary", "authors", "categories", "comment", "url", "published"]:
        df[column] = df[column].fillna("").astype(str).str.replace("\n", " ", regex=False).str.strip()

    df["published_ts"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.dropna(subset=["id", "title", "summary", "published_ts"])
    df["publish_date"] = df["published_ts"].dt.strftime("%Y-%m-%d")
    df["publish_hour"] = df["published_ts"].dt.hour.astype("Int64")
    df["top_conference"] = df["comment"].apply(_extract_top_conference)
    df["author_list"] = df["authors"].apply(_split_csv)
    df["category_list"] = df["categories"].apply(_split_csv)
    df["content_to_embed"] = df.apply(
        lambda row: f"Title:\n{row['title']}\nSummary:\n{row['summary']}",
        axis=1,
    )
    df["keywords"] = df.apply(lambda row: _extract_keywords(row["title"], row["summary"]), axis=1)
    return _normalize_cleaned_papers(df).reset_index(drop=True)


def read_cleaned_papers(parquet_path: Path = DEFAULT_PARQUET_PATH) -> pd.DataFrame:
    if not parquet_path.exists():
        return empty_cleaned_papers()
    return _normalize_cleaned_papers(pd.read_parquet(parquet_path))


def write_cleaned_papers(df: pd.DataFrame, parquet_path: Path = DEFAULT_PARQUET_PATH) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists() and parquet_path.is_dir():
        shutil.rmtree(parquet_path)
    _normalize_cleaned_papers(df).to_parquet(parquet_path, index=False)
    return parquet_path


def save_incremental_snapshot(df: pd.DataFrame, parquet_path: Path = DEFAULT_INCREMENTAL_PATH) -> Path:
    return write_cleaned_papers(df, parquet_path)


def write_cleaned_papers_staged(
    df: pd.DataFrame,
    parquet_path: Path = DEFAULT_PARQUET_PATH,
) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path = parquet_path.with_name(f"{parquet_path.name}.{uuid.uuid4().hex}.tmp")
    if staged_path.exists():
        if staged_path.is_dir():
            shutil.rmtree(staged_path)
        else:
            staged_path.unlink()
    _normalize_cleaned_papers(df).to_parquet(staged_path, index=False)
    return staged_path


def merge_cleaned_papers(
    incremental_df: pd.DataFrame,
    parquet_path: Path = DEFAULT_PARQUET_PATH,
) -> pd.DataFrame:
    base_df = read_cleaned_papers(parquet_path)
    merged_df = pd.concat([base_df, incremental_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["id"], keep="last")
    if "published_ts" in merged_df.columns:
        merged_df = merged_df.sort_values("published_ts", ascending=False, kind="stable")
    # staged write：先写临时文件，成功后再替换原路径（兼容文件/目录两种形式）
    tmp_path = parquet_path.with_suffix(".tmp.parquet")
    _normalize_cleaned_papers(merged_df).reset_index(drop=True).to_parquet(tmp_path, index=False)
    if parquet_path.exists():
        if parquet_path.is_dir():
            shutil.rmtree(parquet_path)
        else:
            parquet_path.unlink()
    tmp_path.rename(parquet_path)

    return _normalize_cleaned_papers(merged_df).reset_index(drop=True)


def process_arxiv_data(
    db_path: Path = DEFAULT_DB_PATH,
    parquet_path: Path = DEFAULT_PARQUET_PATH,
) -> pd.DataFrame:
    print("1. 正在从 SQLite 读取原始数据...")
    raw_df = load_raw_papers(db_path)
    print(f"2. 正在清洗 {len(raw_df)} 篇论文...")
    cleaned_df = clean_papers(raw_df)
    write_cleaned_papers(cleaned_df, parquet_path)
    print(f"3. 处理完毕，已写入：{parquet_path}")
    return cleaned_df


if __name__ == "__main__":
    process_arxiv_data()
