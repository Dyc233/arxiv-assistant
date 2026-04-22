from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.process import load_cleaned

OUTPUT_DIR = Path(__file__).resolve().parent

EXTRA_STOPS = {
    "model", "models", "method", "methods", "task", "tasks", "paper", "papers",
    "based", "using", "used", "approach", "approaches", "learning", "language",
    "large", "neural", "system", "systems", "framework", "new", "novel", "towards",
    "via", "data", "study", "work", "works", "analysis", "survey", "benchmark",
    "efficient", "effective", "better", "improved", "improving", "improve",
    "evaluation", "evaluating", "training", "trained", "pre", "fine", "tuning",
    "multi", "high", "low", "different", "multiple", "single", "general", "specific",
    "open", "source", "code", "human", "scale", "real", "world", "time", "art",
    "state", "https", "github", "com", "http", "arxiv", "org",
}

TOP_N = 30
MIN_DF = 5


def main() -> None:
    df = load_cleaned()[["id", "title", "publish_date"]]
    df["quarter"] = pd.to_datetime(df["publish_date"], errors="coerce").dt.to_period("Q")
    df = df.dropna(subset=["quarter"])

    vec_global = CountVectorizer(ngram_range=(1, 2), max_features=500, stop_words="english", min_df=20, binary=True)
    X_global = vec_global.fit_transform(df["title"].tolist())
    all_terms = vec_global.get_feature_names_out()
    global_df = np.asarray(X_global.sum(axis=0)).flatten()
    top_terms = [
        t for t, _ in sorted(zip(all_terms, global_df), key=lambda x: -x[1])
        if t not in EXTRA_STOPS and not any(s in t.split() for s in EXTRA_STOPS)
    ][:TOP_N]

    quarters = sorted(df["quarter"].unique())
    matrix = pd.DataFrame(0, index=top_terms, columns=[str(q) for q in quarters])

    for q in quarters:
        slice_df = df[df["quarter"] == q]
        if len(slice_df) < 10:
            continue
        vec = CountVectorizer(vocabulary=top_terms, binary=True)
        X = vec.fit_transform(slice_df["title"].tolist())
        matrix[str(q)] = np.asarray(X.sum(axis=0)).flatten()

    matrix_norm = matrix.div(matrix.max(axis=1).replace(0, 1), axis=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(OUTPUT_DIR / "keyword_trend.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(max(16, len(quarters) * 0.5), 10))
    im = ax.imshow(matrix_norm.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(quarters)))
    ax.set_xticklabels([str(q) for q in quarters], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_terms)))
    ax.set_yticklabels(top_terms, fontsize=9)
    ax.set_title("Keyword Trend by Quarter (row-normalized)")
    plt.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "keyword_trend.png", dpi=150)
    plt.close(fig)
    print(f"Done. {len(quarters)} quarters x {len(top_terms)} terms -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
