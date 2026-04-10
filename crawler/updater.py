"""简化版增量更新脚本 - 删除了复杂的备份恢复机制"""
import argparse
import sys
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
)
from analysis.embedder import (
    BATCH_SIZE,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    MODEL_NAME,
    upsert_dataframe,
)
from crawler.spider import (
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
    return parser.parse_args()


def main():
    """主函数：拉取新论文、清洗、合并、生成向量"""
    args = parse_args()

    # 初始化数据库
    conn = init_db(args.db_path)

    # 获取最新发布时间
    latest_published = get_latest_published(conn)
    print(f"最新论文时间: {latest_published}")

    # 拉取新论文
    print(f"开始拉取新论文 (类别: {args.category}, 最多: {args.max_results})...")
    new_rows = fetch_new_papers(
        conn,
        max_results=args.max_results,
        batch_size=args.insert_batch_size,
        category=args.category,
        newer_than=latest_published,
        persist=True,  # 直接持久化到数据库
    )

    if not new_rows:
        print("没有新论文，退出")
        return

    print(f"拉取到 {len(new_rows)} 篇新论文")

    # 清洗数据
    print("清洗数据...")
    incremental_df = clean_papers(new_rows)

    if incremental_df.empty:
        print("清洗后无有效数据，退出")
        return

    print(f"清洗后剩余 {len(incremental_df)} 篇论文")

    # 保存增量数据
    print(f"保存增量数据到: {args.incremental_path}")
    args.incremental_path.parent.mkdir(parents=True, exist_ok=True)
    incremental_df.to_parquet(args.incremental_path, index=False)

    # 合并到主数据集
    print("合并到主数据集...")
    if args.cleaned_path.exists():
        base_df = read_cleaned_papers(args.cleaned_path)
        merged_df = pd.concat([base_df, incremental_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=["id"], keep="last")
        if "published_ts" in merged_df.columns:
            merged_df = merged_df.sort_values("published_ts", ascending=False)
        merged_df = merged_df[CLEANED_COLUMNS].reset_index(drop=True)
    else:
        merged_df = incremental_df[CLEANED_COLUMNS]

    # 修复日期类型问题
    if "publish_date" in merged_df.columns:
        merged_df["publish_date"] = merged_df["publish_date"].astype(str)

    print(f"保存合并数据到: {args.cleaned_path}")
    args.cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(args.cleaned_path, index=False)

    # 生成向量并存入 ChromaDB
    if not args.skip_embedding:
        print("生成向量并存入 ChromaDB...")
        upsert_dataframe(
            incremental_df,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection_name,
            model_name=args.model_name,
            batch_size=args.embedding_batch_size,
        )
        print("向量化完成")
    else:
        print("跳过向量化")

    print(f"\n✅ 更新完成！新增 {len(incremental_df)} 篇论文")


if __name__ == "__main__":
    main()



