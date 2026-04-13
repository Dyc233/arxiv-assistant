"""发文量趋势分析 — 按月统计论文数量，输出 CSV + PNG"""
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
    df["month"] = pd.to_datetime(df["publish_date"], errors="coerce").dt.to_period("M")
    trend = df.groupby("month").size().reset_index(name="count").sort_values("month")
    trend["month_str"] = trend["month"].astype(str)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trend[["month_str", "count"]].to_csv(OUTPUT_DIR / "publish_trend.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(trend["month_str"], trend["count"], marker="o", markersize=3, linewidth=1.2)
    # x轴只显示部分刻度，避免拥挤
    step = max(1, len(trend) // 20)
    ax.set_xticks(range(0, len(trend), step))
    ax.set_xticklabels(trend["month_str"].iloc[::step], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Paper Count")
    ax.set_title("Monthly Paper Publication Trend")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "publish_trend.png", dpi=150)
    plt.close(fig)
    print(f"Done. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
