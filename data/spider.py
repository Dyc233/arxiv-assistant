import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import arxiv
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("arxiv").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

DEFAULT_CATEGORY = "cs.CL"


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


def get_oldest_date(conn: sqlite3.Connection) -> str | None:
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(published) FROM papers")
    result = cursor.fetchone()[0]
    return result[:10].replace("-", "") + "0000" if result else None


def get_latest_published(conn: sqlite3.Connection) -> datetime | None:
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(published) FROM papers")
    result = cursor.fetchone()[0]
    return datetime.fromisoformat(result) if result else None


def _format_arxiv_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%d%H%M")


def _paper_to_row(paper: arxiv.Result) -> dict[str, str]:
    return {
        "id": paper.get_short_id(),
        "title": paper.title.replace("\n", " ").strip(),
        "summary": paper.summary.replace("\n", " ").strip(),
        "published": str(paper.published),
        "authors": ", ".join(author.name for author in paper.authors),
        "categories": ", ".join(paper.categories),
        "comment": (paper.comment or "").replace("\n", " ").strip(),
        "url": paper.entry_id,
    }


def _insert_rows(conn: sqlite3.Connection, rows: list[dict[str, str]], batch_size: int = 50) -> None:
    _insert_rows_with_commit(conn, rows, batch_size=batch_size, commit=True)


def _insert_rows_with_commit(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    batch_size: int = 50,
    commit: bool = True,
) -> None:
    if not rows:
        return

    cursor = conn.cursor()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        cursor.executemany(
            "INSERT OR IGNORE INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
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
                for row in batch
            ],
        )
        if commit:
            conn.commit()


def fetch_new_papers(
    conn: sqlite3.Connection,
    max_results: int = 2000,
    batch_size: int = 50,
    category: str = DEFAULT_CATEGORY,
    newer_than: datetime | None = None,
    persist: bool = True,
) -> list[dict[str, str]]:
    existing_ids = get_existing_ids(conn)
    newer_than = newer_than or get_latest_published(conn)

    query = f"cat:{category}"
    if newer_than:
        query += (
            f" AND submittedDate:[{_format_arxiv_datetime(newer_than.replace(second=0, microsecond=0))}"
            f" TO {_format_arxiv_datetime(datetime.now(timezone.utc))}]"
        )
        logging.info(f"[*] Incremental crawl after {newer_than.isoformat()}...")
    else:
        logging.info("[*] Database empty, fetching latest papers...")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)

    new_rows: list[dict[str, str]] = []
    with tqdm(total=max_results, desc="Fetching", unit="papers") as progress:
        for paper in client.results(search):
            progress.update(1)
            if newer_than and paper.published <= newer_than:
                break

            row = _paper_to_row(paper)
            if row["id"] in existing_ids:
                continue

            existing_ids.add(row["id"])
            new_rows.append(row)

    if persist:
        _insert_rows_with_commit(conn, new_rows, batch_size=batch_size, commit=True)
        logging.info(f"[*] Inserted {len(new_rows)} new papers into SQLite.")
    else:
        logging.info(f"[*] Collected {len(new_rows)} new papers for staged update.")
    return new_rows


def fetch_arxiv_papers(conn: sqlite3.Connection, max_results: int = 50000, batch_size: int = 50) -> None:
    cursor = conn.cursor()
    existing_ids = get_existing_ids(conn)
    oldest_date = get_oldest_date(conn)

    query = f"cat:{DEFAULT_CATEGORY}"
    if oldest_date:
        query += f" AND submittedDate:[199001010000 TO {oldest_date}]"
        logging.info(f"[*] Resuming: Searching for papers older than {oldest_date}...")
    else:
        logging.info("[*] Starting a fresh crawl from the latest papers...")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    papers_batch: list[tuple[str, str, str, str, str, str, str, str]] = []

    with tqdm(total=max_results, desc="Fetching", unit="papers") as progress:
        try:
            for paper in client.results(search):
                paper_id = paper.get_short_id()
                if paper_id in existing_ids:
                    progress.update(1)
                    continue

                row = _paper_to_row(paper)
                papers_batch.append(
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
                existing_ids.add(paper_id)

                if len(papers_batch) >= batch_size:
                    cursor.executemany(
                        "INSERT OR IGNORE INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        papers_batch,
                    )
                    conn.commit()
                    papers_batch = []

                progress.update(1)

            if papers_batch:
                cursor.executemany(
                    "INSERT OR IGNORE INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    papers_batch,
                )
                conn.commit()

        except KeyboardInterrupt:
            logging.warning("\n[!] User stopped the process. Data is safe in SQLite.")
        except Exception as exc:
            logging.error(f"\n[X] Critical Error: {exc}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "arxiv_papers.db"
    db_conn = init_db(db_path)
    try:
        fetch_arxiv_papers(db_conn, max_results=10000, batch_size=50)
    finally:
        db_conn.close()
        logging.info("\n[*] Database safely closed.")
