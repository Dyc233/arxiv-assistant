"""作者分析 — 高产作者排行，输出 CSV + PNG"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.process import DEFAULT_INCREMENTAL_PATH, DEFAULT_PARQUET_PATH

OUTPUT_DIR = Path(__file__).resolve().parent
TOP_N = 20


def load_data() -> pd.DataFrame:
    frames = []
    for path in [DEFAULT_PARQUET_PATH, DEFAULT_INCREMENTAL_PATH]:
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError("No parquet files found.")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"], keep="last")
    def to_list(v):
        if isinstance(v, (list, np.ndarray)):
            return list(v)
        if pd.isna(v) or v == "":
            return []
        return [x.strip() for x in str(v).split(",") if x.strip()]
    df["author_list"] = df["author_list"].apply(to_list)
    return df


def main() -> None:
    df = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counter: Counter[str] = Counter()
    for authors in df["author_list"]:
        counter.update(authors)

    top = counter.most_common(TOP_N)
    names = [name for name, _ in top]
    counts = [cnt for _, cnt in top]

    pd.DataFrame({"author": names, "paper_count": counts}).to_csv(
        OUTPUT_DIR / "top_authors.csv", index=False, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names[::-1], counts[::-1])
    ax.set_xlabel("Paper Count")
    ax.set_title(f"Top {TOP_N} Most Prolific Authors")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_authors.png", dpi=150)
    plt.close(fig)

    print(f"Done. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
