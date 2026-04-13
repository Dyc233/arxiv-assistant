"""增量更新脚本：爬虫 → 清洗 → 合并 parquet → upsert ChromaDB
 正常增量更新（最常用）：
  cd D:/CODING/BS/src
  .venv/Scripts/python -m crawler.updater
  自动从 SQLite 最新时间戳往后抓，清洗、合并 parquet、upsert ChromaDB。

  ---
  只抓数据不跑 embedding（网络好但 GPU 不在）：
  .venv/Scripts/python -m crawler.updater --skip-embedding
  之后补跑 embedding：
  .venv/Scripts/python -m crawler.updater --retry-embedding

  ---
  ChromaDB 上次失败了，补跑 embedding：
  .venv/Scripts/python -m crawler.updater --retry-embedding

  ---
  限制抓取数量（测试用）：
  .venv/Scripts/python -m crawler.updater --max-results 100
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.process import (
    DEFAULT_INCREMENTAL_PATH,
    DEFAULT_PARQUET_PATH,
    clean_papers,
    merge_cleaned_papers,
    read_cleaned_papers,
)
from data.embedder import (
    BATCH_SIZE,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    MODEL_NAME,
    upsert_dataframe,
)
from data.spider import (
    DEFAULT_CATEGORY,
    fetch_new_papers,
    get_latest_published,
    init_db,
)


DEFAULT_DB_PATH = SRC_ROOT.parent / "data" / "arxiv_papers.db"


def parse_args():
    parser = argparse.ArgumentParser(description="增量更新 arXiv 论文数据")
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
    parser.add_argument("--retry-embedding", action="store_true",
                         help="跳过爬虫/清洗/合并，直接从增量 parquet 重新 upsert ChromaDB")
    return parser.parse_args()


def _do_upsert(incremental_df, args):
    upsert_dataframe(
        incremental_df,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        model_name=args.model_name,
        batch_size=args.embedding_batch_size,
    )


def main():
    args = parse_args()

    if args.retry_embedding:
        if not args.incremental_path.exists():
            print(f"找不到增量文件: {args.incremental_path}")
            print("请先正常运行一次 updater，或手动指定 --incremental-path")
            return
        incremental_df = read_cleaned_papers(args.incremental_path)
        print(f"重试 upsert，共 {len(incremental_df)} 篇...")
        _do_upsert(incremental_df, args)
        print("✅ 重试完成")
        return

    # ── 正常流程 ──
    conn = init_db(args.db_path)
    latest_published = get_latest_published(conn)
    print(f"最新论文时间: {latest_published}")

    print(f"开始拉取新论文 (类别: {args.category}, 最多: {args.max_results})...")
    new_rows = fetch_new_papers(
        conn,
        max_results=args.max_results,
        batch_size=args.insert_batch_size,
        category=args.category,
        newer_than=latest_published,
        persist=True,
    )

    if not new_rows:
        print("没有新论文，退出")
        return

    print(f"拉取到 {len(new_rows)} 篇新论文")

    print("清洗数据...")
    incremental_df = clean_papers(new_rows)

    if incremental_df.empty:
        print("清洗后无有效数据，退出")
        return

    print(f"清洗后剩余 {len(incremental_df)} 篇论文")

    # 先保存增量快照（合并前），供 --retry-embedding 使用
    args.incremental_path.parent.mkdir(parents=True, exist_ok=True)
    incremental_df.to_parquet(args.incremental_path, index=False)
    print(f"增量快照已保存: {args.incremental_path}")

    # 合并到主数据集（staged write，保护原文件）
    print("合并到主数据集...")
    merge_cleaned_papers(incremental_df, args.cleaned_path)
    print(f"主数据集已更新: {args.cleaned_path}")

    # upsert 向量库
    if args.skip_embedding:
        print("跳过向量化")
    else:
        print("生成向量并存入 ChromaDB...")
        try:
            _do_upsert(incremental_df, args)
            print("向量化完成")
        except Exception as e:
            print(f"[警告] ChromaDB upsert 失败: {e}")
            print("parquet 已更新，向量库未同步。")
            print("修复命令：python -m crawler.updater --retry-embedding")
            return

    print(f"\n✅ 更新完成！新增 {len(incremental_df)} 篇论文")


if __name__ == "__main__":
    main()
