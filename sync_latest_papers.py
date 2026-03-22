from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import arxiv
import chromadb
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("arxiv").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "arxiv_papers.db"
DEFAULT_CLEANED_OUTPUT = SRC_ROOT / "analysis" / "cleaned_papers_incremental.parquet"
DEFAULT_CHROMA_DIR = SRC_ROOT / "chroma_db"
DEFAULT_COLLECTION_NAME = "arxiv_nlp_papers"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_CATEGORY = "cs.CL"
DEFAULT_FETCH_LIMIT = 2000
DEFAULT_INSERT_BATCH_SIZE = 50
DEFAULT_EMBEDDING_BATCH_SIZE = 32

CONFERENCE_PATTERN = re.compile(
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
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    "yourselves",
}


@dataclass(slots=True)
class SyncConfig:
    db_path: Path = DEFAULT_DB_PATH
    cleaned_output: Path = DEFAULT_CLEANED_OUTPUT
    chroma_dir: Path = DEFAULT_CHROMA_DIR
    collection_name: str = DEFAULT_COLLECTION_NAME
    model_name: str = DEFAULT_MODEL_NAME
    category: str = DEFAULT_CATEGORY
    max_results: int = DEFAULT_FETCH_LIMIT
    insert_batch_size: int = DEFAULT_INSERT_BATCH_SIZE
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    write_cleaned_output: bool = True
    skip_embedding: bool = False


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            published TEXT,
            authors TEXT,
            categories TEXT,
            comment TEXT,
            url TEXT
        )
        """
    )
    conn.commit()
    return conn


def get_existing_ids(conn: sqlite3.Connection) -> set[str]:
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM papers")
    return {row[0] for row in cursor.fetchall()}


def get_latest_published(conn: sqlite3.Connection) -> datetime | None:
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(published) FROM papers")
    row = cursor.fetchone()
    latest_value = row[0] if row else None
    if not latest_value:
        return None
    return datetime.fromisoformat(latest_value)


def format_arxiv_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%d%H%M")


def sanitize_text(value: str | None) -> str:
    return (value or "").replace("\n", " ").strip()


def split_multi_value_field(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def deduplicate_keep_order(items: Iterable[str], limit: int | None = None) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduplicated.append(item)
        if limit is not None and len(deduplicated) >= limit:
            break
    return deduplicated


def extract_keywords(title: str, summary: str, limit: int = 64) -> list[str]:
    pure_text = re.sub(r"[^a-z\s]", " ", f"{title} {summary}".lower())
    tokens = [token for token in pure_text.split() if token not in STOPWORDS and len(token) > 1]
    bigrams = [" ".join(tokens[index:index + 2]) for index in range(max(0, len(tokens) - 1))]
    trigrams = [" ".join(tokens[index:index + 3]) for index in range(max(0, len(tokens) - 2))]
    return deduplicate_keep_order([*tokens, *bigrams, *trigrams], limit=limit)


def extract_top_conference(comment: str | None) -> str:
    if not comment:
        return "None"
    match = CONFERENCE_PATTERN.search(comment)
    if not match:
        return "None"
    return match.group(0).upper()


def fetch_new_papers(conn: sqlite3.Connection, config: SyncConfig) -> list[dict[str, str]]:
    cursor = conn.cursor()
    existing_ids = get_existing_ids(conn)
    latest_published = get_latest_published(conn)

    query = f"cat:{config.category}"
    if latest_published is not None:
        start_time = latest_published.replace(second=0, microsecond=0)
        end_time = datetime.now(timezone.utc)
        query += f" AND submittedDate:[{format_arxiv_datetime(start_time)} TO {format_arxiv_datetime(end_time)}]"
        logging.info(f"[*] 增量抓取区间：{latest_published.isoformat()} 之后发布的 {config.category} 论文")
    else:
        logging.info(f"[*] 数据库为空，开始抓取最新的 {config.category} 论文")

    search = arxiv.Search(
        query=query,
        max_results=config.max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)

    inserted_rows: list[dict[str, str]] = []
    insert_buffer: list[tuple[str, str, str, str, str, str, str, str]] = []

    with tqdm(total=config.max_results, desc="Fetching", unit="papers") as progress:
        for paper in client.results(search):
            progress.update(1)
            paper_id = paper.get_short_id()

            if paper_id in existing_ids:
                continue

            if latest_published is not None and paper.published <= latest_published:
                break

            row = {
                "id": paper_id,
                "title": sanitize_text(paper.title),
                "summary": sanitize_text(paper.summary),
                "published": str(paper.published),
                "authors": ", ".join(author.name for author in paper.authors),
                "categories": ", ".join(paper.categories),
                "comment": sanitize_text(paper.comment),
                "url": paper.entry_id,
            }

            insert_buffer.append(
                (
                    row["id"],
                    row["title"],
                    row["summary"],
                    row["published"],
                    row["authors"],
                    row["categories"],
                    row["comment"],
                    row["url"],
                )
            )
            inserted_rows.append(row)
            existing_ids.add(paper_id)

            if len(insert_buffer) >= config.insert_batch_size:
                cursor.executemany(
                    "INSERT OR IGNORE INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    insert_buffer,
                )
                conn.commit()
                insert_buffer.clear()

    if insert_buffer:
        cursor.executemany(
            "INSERT OR IGNORE INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            insert_buffer,
        )
        conn.commit()

    logging.info(f"[*] 本次新增入库论文：{len(inserted_rows)} 篇")
    return inserted_rows


def clean_papers(rows: list[dict[str, str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["id"]).copy()
    df["published_ts"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_ts", "title", "summary"])
    df["publish_date"] = df["published_ts"].dt.strftime("%Y-%m-%d")
    df["publish_hour"] = df["published_ts"].dt.hour.astype("Int64")
    df["top_conference"] = df["comment"].apply(extract_top_conference)
    df["author_list"] = df["authors"].apply(split_multi_value_field)
    df["category_list"] = df["categories"].apply(split_multi_value_field)
    df["content_to_embed"] = df.apply(
        lambda row: f"Title:\n{row['title']}\nSummary:\n{row['summary']}",
        axis=1,
    )
    df["keywords"] = df.apply(
        lambda row: extract_keywords(row["title"], row["summary"]),
        axis=1,
    )
    return df[
        [
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
    ].reset_index(drop=True)


def save_cleaned_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logging.info(f"[*] 增量清洗结果已写入：{output_path}")


def build_metadata_records(df: pd.DataFrame) -> list[dict[str, str | int | float]]:
    metadata_records: list[dict[str, str | int | float]] = []
    for row in df.itertuples(index=False):
        metadata: dict[str, str | int | float] = {
            "title": row.title,
            "publish_date": row.publish_date,
            "published": row.published,
            "top_conference": row.top_conference or "None",
            "authors": row.authors or "",
            "categories": row.categories or "",
            "url": row.url or "",
            "keywords": " | ".join(row.keywords[:20]),
        }
        if pd.notna(row.publish_hour):
            metadata["publish_hour"] = int(row.publish_hour)
        metadata_records.append(metadata)
    return metadata_records


def upsert_embeddings(df: pd.DataFrame, config: SyncConfig) -> int:
    valid_df = df.dropna(subset=["id", "content_to_embed"]).copy()
    if valid_df.empty:
        logging.info("[*] 没有可向量化的数据，跳过 Upsert")
        return 0

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[*] 当前向量化设备：{device.upper()}")
    logging.info(f"[*] 正在加载嵌入模型：{config.model_name}")
    model = SentenceTransformer(config.model_name, device=device)

    logging.info(f"[*] 正在连接 ChromaDB：{config.chroma_dir}")
    chroma_client = chromadb.PersistentClient(path=str(config.chroma_dir))
    collection = chroma_client.get_or_create_collection(
        name=config.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = valid_df["id"].astype(str).tolist()
    documents = valid_df["content_to_embed"].tolist()
    metadatas = build_metadata_records(valid_df)

    for start_index in tqdm(
        range(0, len(ids), config.embedding_batch_size),
        desc="Embedding",
        unit="batch",
    ):
        end_index = start_index + config.embedding_batch_size
        batch_ids = ids[start_index:end_index]
        batch_docs = documents[start_index:end_index]
        batch_metadatas = metadatas[start_index:end_index]
        batch_embeddings = model.encode(
            batch_docs,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=config.embedding_batch_size,
        ).tolist()
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )

    logging.info(f"[*] 已完成 Chroma Upsert：{len(ids)} 篇论文")
    return len(ids)


def run_sync(config: SyncConfig) -> None:
    conn = init_db(config.db_path)
    try:
        new_rows = fetch_new_papers(conn, config)
    finally:
        conn.close()

    if not new_rows:
        logging.info("[*] 没有检测到新论文，本次同步结束")
        return

    cleaned_df = clean_papers(new_rows)
    logging.info(f"[*] 清洗完成，可用于向量化的论文：{len(cleaned_df)} 篇")

    if config.write_cleaned_output:
        save_cleaned_output(cleaned_df, config.cleaned_output)

    if not config.skip_embedding:
        upsert_embeddings(cleaned_df, config)

    logging.info("[*] 增量同步完成")


def parse_args() -> SyncConfig:
    parser = argparse.ArgumentParser(description="一键增量抓取、清洗并 Upsert 最新 arXiv 论文")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--cleaned-output", type=Path, default=DEFAULT_CLEANED_OUTPUT)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--max-results", type=int, default=DEFAULT_FETCH_LIMIT)
    parser.add_argument("--insert-batch-size", type=int, default=DEFAULT_INSERT_BATCH_SIZE)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--skip-cleaned-output", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    args = parser.parse_args()

    return SyncConfig(
        db_path=args.db_path,
        cleaned_output=args.cleaned_output,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        model_name=args.model_name,
        category=args.category,
        max_results=args.max_results,
        insert_batch_size=args.insert_batch_size,
        embedding_batch_size=args.embedding_batch_size,
        write_cleaned_output=not args.skip_cleaned_output,
        skip_embedding=args.skip_embedding,
    )


if __name__ == "__main__":
    run_sync(parse_args())
