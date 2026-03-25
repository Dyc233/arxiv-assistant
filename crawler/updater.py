import argparse
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.data_process import (
    CLEANED_COLUMNS,
    DEFAULT_INCREMENTAL_PATH,
    DEFAULT_PARQUET_PATH,
    clean_papers,
    read_cleaned_papers,
    write_cleaned_papers_staged,
)
from analysis.embedder import (
    BATCH_SIZE,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    MODEL_NAME,
    delete_ids,
    upsert_dataframe,
)
from crawler.spider import (
    DEFAULT_CATEGORY,
    _insert_rows_with_commit,
    fetch_new_papers,
    get_latest_published,
    init_db,
)


DEFAULT_DB_PATH = SRC_ROOT.parent / "data" / "arxiv_papers.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch, clean, and stage an incremental arXiv update.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--cleaned-path", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--incremental-path", type=Path, default=DEFAULT_INCREMENTAL_PATH)
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DB_DIR)
    parser.add_argument("--collection-name", default=COLLECTION_NAME)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--max-results", type=int, default=2000)
    parser.add_argument("--insert-batch-size", type=int, default=50)
    parser.add_argument("--embedding-batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--skip-embedding", action="store_true")
    return parser.parse_args()


def _build_merged_dataframe(incremental_df: pd.DataFrame, cleaned_path: Path) -> pd.DataFrame:
    base_df = read_cleaned_papers(cleaned_path)
    merged_df = pd.concat([base_df, incremental_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["id"], keep="last")
    if "published_ts" in merged_df.columns:
        merged_df = merged_df.sort_values("published_ts", ascending=False, kind="stable")
    return merged_df[CLEANED_COLUMNS].reset_index(drop=True)


def _backup_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.{uuid.uuid4().hex}.bak")


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _promote_staged_file(target: Path, staged: Path, replaced_paths: list[tuple[Path, Path | None]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if target.exists():
        backup = _backup_path(target)
        target.replace(backup)

    try:
        staged.replace(target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.replace(target)
        raise

    replaced_paths.append((target, backup))


def _restore_replaced_paths(replaced_paths: list[tuple[Path, Path | None]]) -> None:
    for target, backup in reversed(replaced_paths):
        _remove_path(target)
        if backup is not None and backup.exists():
            backup.replace(target)


def _cleanup_backups(replaced_paths: list[tuple[Path, Path | None]]) -> None:
    for _, backup in replaced_paths:
        if backup is not None:
            _remove_path(backup)


def _cleanup_staged_paths(paths: list[Path]) -> None:
    for path in paths:
        _remove_path(path)


def main() -> None:
    args = parse_args()
    conn = init_db(args.db_path)
    replaced_paths: list[tuple[Path, Path | None]] = []
    staged_paths: list[Path] = []
    vector_ids: list[str] = []
    embedding_applied = False

    try:
        latest_published = get_latest_published(conn)
        new_rows = fetch_new_papers(
            conn,
            max_results=args.max_results,
            batch_size=args.insert_batch_size,
            category=args.category,
            newer_than=latest_published,
            persist=False,
        )

        if not new_rows:
            print("No new papers detected. Nothing changed.")
            return

        incremental_df = clean_papers(new_rows)
        if incremental_df.empty:
            print("Fetched papers did not produce any valid cleaned rows. Nothing changed.")
            return

        vector_ids = incremental_df["id"].astype(str).tolist()
        merged_df = _build_merged_dataframe(incremental_df, args.cleaned_path)

        staged_incremental_path = write_cleaned_papers_staged(incremental_df, args.incremental_path)
        staged_cleaned_path = write_cleaned_papers_staged(merged_df, args.cleaned_path)
        staged_paths.extend([staged_incremental_path, staged_cleaned_path])

        if not args.skip_embedding:
            upsert_dataframe(
                incremental_df,
                chroma_dir=args.chroma_dir,
                collection_name=args.collection_name,
                model_name=args.model_name,
                batch_size=args.embedding_batch_size,
            )
            embedding_applied = True

        conn.execute("BEGIN IMMEDIATE")
        _insert_rows_with_commit(
            conn,
            new_rows,
            batch_size=args.insert_batch_size,
            commit=False,
        )
        _promote_staged_file(args.incremental_path, staged_incremental_path, replaced_paths)
        _promote_staged_file(args.cleaned_path, staged_cleaned_path, replaced_paths)
        conn.commit()

        _cleanup_backups(replaced_paths)
        replaced_paths.clear()
        print(f"Incremental sync completed successfully with {len(incremental_df)} papers.")
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

        if replaced_paths:
            _restore_replaced_paths(replaced_paths)

        if embedding_applied and not args.skip_embedding and vector_ids:
            try:
                delete_ids(
                    vector_ids,
                    chroma_dir=args.chroma_dir,
                    collection_name=args.collection_name,
                )
            except Exception:
                pass

        raise
    finally:
        _cleanup_staged_paths(staged_paths)
        conn.close()


if __name__ == "__main__":
    main()
