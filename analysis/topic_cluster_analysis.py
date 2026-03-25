from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from analysis.data_process import (
    DEFAULT_INCREMENTAL_PATH,
    DEFAULT_PARQUET_PATH,
    STOPWORDS,
)


CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "arxiv_nlp_papers"
OUTPUT_DIR = Path(__file__).resolve().parent / "topic_clusters"
RANDOM_STATE = 42

TITLE_STOPWORDS = STOPWORDS | {
    "paper",
    "papers",
    "model",
    "models",
    "method",
    "methods",
    "approach",
    "approaches",
    "task",
    "tasks",
    "using",
    "use",
    "based",
    "learning",
    "language",
    "languages",
    "natural",
    "processing",
    "nlp",
    "large",
    "neural",
    "towards",
    "via",
    "toward",
    "study",
    "analysis",
    "understanding",
    "new",
    "efficient",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster existing Chroma embeddings into topic clusters.")
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DB_DIR)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--incremental-path", type=Path, default=DEFAULT_INCREMENTAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=12, help="Number of topic clusters.")
    parser.add_argument("--pca-dim", type=int, default=50, help="PCA dimensions before clustering.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size when reading Chroma embeddings.")
    parser.add_argument("--max-papers", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--top-categories", type=int, default=6)
    parser.add_argument("--examples-per-cluster", type=int, default=8)
    parser.add_argument("--title-tfidf-features", type=int, default=3000)
    parser.add_argument("--silhouette-sample", type=int, default=5000)
    return parser.parse_args()


def _normalize_list(value: object) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item).strip() for item in value.tolist() if str(item).strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value) or value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def load_metadata(parquet_path: Path, incremental_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in [parquet_path, incremental_path]:
        if path.exists():
            frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError("No parquet metadata file was found.")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["id"], keep="last").copy()
    merged["published_ts"] = pd.to_datetime(merged["published_ts"], utc=True, errors="coerce")
    merged["publish_date"] = merged["publish_date"].astype(str)
    merged["author_list"] = merged["author_list"].apply(_normalize_list)
    merged["category_list"] = merged["category_list"].apply(_normalize_list)
    merged["keywords"] = merged["keywords"].apply(_normalize_list)
    merged["top_conference"] = merged["top_conference"].fillna("").astype(str).replace("None", "")
    return merged


def load_chroma_embeddings(
    chroma_dir: Path,
    collection_name: str,
    batch_size: int,
    max_papers: int | None = None,
) -> tuple[list[str], np.ndarray]:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(collection_name)
    total = collection.count()

    all_ids: list[str] = []
    chunks: list[np.ndarray] = []

    for offset in range(0, total, batch_size):
        limit = min(batch_size, total - offset)
        payload = collection.get(
            limit=limit,
            offset=offset,
            include=["embeddings"],
        )
        ids = payload["ids"]
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Unexpected embedding shape at offset {offset}: {embeddings.shape}")

        all_ids.extend(ids)
        chunks.append(embeddings)

        if max_papers is not None and len(all_ids) >= max_papers:
            all_ids = all_ids[:max_papers]
            used = max_papers - sum(chunk.shape[0] for chunk in chunks[:-1])
            chunks[-1] = chunks[-1][:used]
            break

    matrix = np.vstack(chunks)
    return all_ids, matrix


def attach_metadata(metadata_df: pd.DataFrame, ids: list[str], embeddings: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = metadata_df.drop_duplicates(subset=["id"], keep="last").set_index("id", drop=False)
    available_ids = []
    positions = []

    for index, paper_id in enumerate(ids):
        if paper_id in metadata.index:
            available_ids.append(paper_id)
            positions.append(index)

    if not available_ids:
        raise ValueError("None of the embeddings could be matched to parquet metadata.")

    filtered_df = metadata.loc[available_ids].reset_index(drop=True)
    filtered_embeddings = embeddings[np.asarray(positions, dtype=np.int32)]
    return filtered_df, filtered_embeddings


def run_pca(embeddings: np.ndarray, pca_dim: int) -> tuple[PCA, np.ndarray]:
    max_dim = min(embeddings.shape[0], embeddings.shape[1])
    if pca_dim >= max_dim:
        pca_dim = max_dim - 1
    if pca_dim < 2:
        raise ValueError("PCA dimension is too small for clustering.")

    model = PCA(
        n_components=pca_dim,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
    )
    reduced = model.fit_transform(embeddings)
    return model, reduced.astype(np.float32, copy=False)


def run_kmeans(reduced: np.ndarray, k: int) -> tuple[MiniBatchKMeans, np.ndarray]:
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init="auto",
        batch_size=2048,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(reduced)
    return model, labels


def compute_cluster_terms(
    titles: pd.Series,
    labels: np.ndarray,
    top_terms: int,
    max_features: int,
) -> dict[int, list[str]]:
    vectorizer = TfidfVectorizer(
        stop_words=sorted(TITLE_STOPWORDS),
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
    )
    matrix = vectorizer.fit_transform(titles.fillna(""))
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    cluster_terms: dict[int, list[str]] = {}
    for cluster_id in sorted(np.unique(labels).tolist()):
        cluster_rows = np.where(labels == cluster_id)[0]
        cluster_mean = np.asarray(matrix[cluster_rows].mean(axis=0)).ravel()
        top_idx = np.argsort(cluster_mean)[::-1][:top_terms]
        cluster_terms[cluster_id] = [
            str(feature_names[index])
            for index in top_idx
            if cluster_mean[index] > 0
        ]
    return cluster_terms


def summarize_categories(category_values: pd.Series, top_n: int) -> str:
    exploded = category_values.explode()
    if exploded.empty:
        return ""
    counts = exploded.value_counts().head(top_n)
    return "; ".join(f"{name}:{count}" for name, count in counts.items())


def summarize_keywords(keyword_values: pd.Series, top_n: int) -> str:
    exploded = keyword_values.explode()
    if exploded.empty:
        return ""
    counts = exploded.value_counts().head(top_n)
    return "; ".join(f"{name}:{count}" for name, count in counts.items())


def build_cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    terms_by_cluster: dict[int, list[str]],
    top_categories: int,
    top_terms: int,
) -> pd.DataFrame:
    work_df = df.copy()
    work_df["cluster_id"] = labels
    work_df["year"] = work_df["published_ts"].dt.year

    summary_rows: list[dict[str, object]] = []
    for cluster_id in sorted(work_df["cluster_id"].unique().tolist()):
        cluster_df = work_df[work_df["cluster_id"] == cluster_id].copy()
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "paper_count": len(cluster_df),
                "share": round(len(cluster_df) / len(work_df), 6),
                "start_date": cluster_df["publish_date"].min(),
                "end_date": cluster_df["publish_date"].max(),
                "median_publish_date": cluster_df["published_ts"].sort_values().iloc[len(cluster_df) // 2].date().isoformat(),
                "top_title_terms": " | ".join(terms_by_cluster.get(cluster_id, [])[:top_terms]),
                "top_categories": summarize_categories(cluster_df["category_list"], top_categories),
                "top_keywords": summarize_keywords(cluster_df["keywords"], top_terms),
                "top_conferences": cluster_df["top_conference"].replace("", pd.NA).dropna().value_counts().head(6).to_dict(),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("paper_count", ascending=False)
    summary_df["top_conferences"] = summary_df["top_conferences"].apply(json.dumps, ensure_ascii=False)
    return summary_df.reset_index(drop=True)


def build_cluster_examples(
    df: pd.DataFrame,
    reduced: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    examples_per_cluster: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id in sorted(np.unique(labels).tolist()):
        cluster_idx = np.where(labels == cluster_id)[0]
        cluster_points = reduced[cluster_idx]
        center = centers[cluster_id]
        distances = np.sum((cluster_points - center) ** 2, axis=1)
        nearest_order = np.argsort(distances)[:examples_per_cluster]

        for rank, local_pos in enumerate(nearest_order, start=1):
            source_index = cluster_idx[local_pos]
            row = df.iloc[source_index]
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "rank_in_cluster": rank,
                    "id": row["id"],
                    "title": row["title"],
                    "publish_date": row["publish_date"],
                    "categories": ", ".join(row["category_list"]),
                    "top_conference": row["top_conference"],
                    "distance_to_centroid": float(distances[local_pos]),
                    "url": row["url"],
                }
            )
    return pd.DataFrame(rows)


def build_cluster_yearly_counts(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    work_df = df.copy()
    work_df["cluster_id"] = labels
    work_df["year"] = work_df["published_ts"].dt.year
    return (
        work_df.groupby(["cluster_id", "year"], dropna=False)
        .size()
        .reset_index(name="paper_count")
        .sort_values(["cluster_id", "year"])
        .reset_index(drop=True)
    )


def maybe_compute_silhouette(reduced: np.ndarray, labels: np.ndarray, sample_size: int) -> float | None:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

    if len(reduced) > sample_size:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_index = rng.choice(len(reduced), size=sample_size, replace=False)
        sample_data = reduced[sample_index]
        sample_labels = labels[sample_index]
    else:
        sample_data = reduced
        sample_labels = labels

    if len(np.unique(sample_labels)) < 2:
        return None

    return float(silhouette_score(sample_data, sample_labels))


def save_outputs(
    output_dir: Path,
    clustered_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    examples_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_export_columns = [
        "id",
        "title",
        "publish_date",
        "cluster_id",
        "pca_x",
        "pca_y",
        "author_count",
        "category_count",
        "categories_joined",
        "authors_joined",
        "keywords_joined",
        "top_conference",
        "url",
    ]
    clustered_df[paper_export_columns].to_csv(
        output_dir / "paper_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(output_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig")
    examples_df.to_csv(output_dir / "cluster_examples.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output_dir / "cluster_yearly_counts.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    print("1. Loading parquet metadata...")
    metadata_df = load_metadata(args.parquet_path, args.incremental_path)

    print("2. Loading embeddings from Chroma...")
    ids, embeddings = load_chroma_embeddings(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
        max_papers=args.max_papers,
    )
    print(f"   loaded_embeddings={len(ids)} dim={embeddings.shape[1]}")

    print("3. Attaching metadata to embeddings...")
    clustered_df, filtered_embeddings = attach_metadata(metadata_df, ids, embeddings)
    missing_metadata = len(ids) - len(clustered_df)
    print(f"   matched_papers={len(clustered_df)} missing_metadata={missing_metadata}")

    print("4. Running PCA...")
    pca_model, reduced = run_pca(filtered_embeddings, args.pca_dim)
    print(
        "   explained_variance_ratio="
        f"{float(np.sum(pca_model.explained_variance_ratio_)):.4f}"
    )

    print("5. Running MiniBatchKMeans...")
    kmeans_model, labels = run_kmeans(reduced, args.k)

    print("6. Building cluster summaries...")
    terms_by_cluster = compute_cluster_terms(
        titles=clustered_df["title"],
        labels=labels,
        top_terms=args.top_terms,
        max_features=args.title_tfidf_features,
    )

    clustered_df = clustered_df.copy()
    clustered_df["cluster_id"] = labels
    clustered_df["pca_x"] = reduced[:, 0]
    clustered_df["pca_y"] = reduced[:, 1]
    clustered_df["author_count"] = clustered_df["author_list"].apply(len)
    clustered_df["category_count"] = clustered_df["category_list"].apply(len)
    clustered_df["keywords_joined"] = clustered_df["keywords"].apply(lambda values: " | ".join(values[:20]))
    clustered_df["categories_joined"] = clustered_df["category_list"].apply(lambda values: " | ".join(values))
    clustered_df["authors_joined"] = clustered_df["author_list"].apply(lambda values: " | ".join(values))

    summary_df = build_cluster_summary(
        df=clustered_df,
        labels=labels,
        terms_by_cluster=terms_by_cluster,
        top_categories=args.top_categories,
        top_terms=args.top_terms,
    )
    examples_df = build_cluster_examples(
        df=clustered_df,
        reduced=reduced,
        labels=labels,
        centers=kmeans_model.cluster_centers_,
        examples_per_cluster=args.examples_per_cluster,
    )
    yearly_df = build_cluster_yearly_counts(clustered_df, labels)

    silhouette = maybe_compute_silhouette(reduced, labels, args.silhouette_sample)
    metrics = {
        "paper_count": int(len(clustered_df)),
        "embedding_dim": int(filtered_embeddings.shape[1]),
        "pca_dim": int(reduced.shape[1]),
        "cluster_count": int(args.k),
        "explained_variance_ratio_sum": float(np.sum(pca_model.explained_variance_ratio_)),
        "silhouette_score": silhouette,
        "missing_metadata": int(missing_metadata),
        "chroma_dir": str(args.chroma_dir),
        "collection": args.collection,
    }

    print("7. Saving outputs...")
    save_outputs(
        output_dir=args.output_dir,
        clustered_df=clustered_df,
        summary_df=summary_df,
        examples_df=examples_df,
        yearly_df=yearly_df,
        metrics=metrics,
    )

    print(f"Done. Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
