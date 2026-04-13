"""顶会分析 — 录用分布，输出 CSV + PNG"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.process import DEFAULT_INCREMENTAL_PATH, DEFAULT_PARQUET_PATH

OUTPUT_DIR = Path(__file__).resolve().parent


def load_data() -> pd.DataFrame:
    frames = []
    for path in [DEFAULT_PARQUET_PATH, DEFAULT_INCREMENTAL_PATH]:
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError("No parquet files found.")
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"], keep="last")


def main() -> None:
    df = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 顶会录用分布（排除 None）──────────────────────────
    conf_df = df[df["top_conference"] != "None"].copy()
    # 只取会议名，去掉年份（如 "ACL 2023" → "ACL"）
    conf_df["conf_name"] = conf_df["top_conference"].str.extract(r"([A-Z]+)", expand=False)
    dist = conf_df["conf_name"].value_counts().reset_index()
    dist.columns = ["conference", "count"]

    dist.to_csv(OUTPUT_DIR / "conference_distribution.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(dist["conference"][::-1], dist["count"][::-1])
    ax.set_xlabel("Paper Count")
    ax.set_title("Top Conference Distribution")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "conference_distribution.png", dpi=150)
    plt.close(fig)

    print(f"Done. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
