"""交叉学科渗透度分析 — 排除全部 cs.* 标签，只统计真正跨学科领域"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.process import DEFAULT_INCREMENTAL_PATH, DEFAULT_PARQUET_PATH

OUTPUT_DIR = Path(__file__).resolve().parent
TOP_N = 20

# 真正跨学科领域：前缀 -> 展示名（全部排除 cs.*）
FOCUS_DOMAINS = {
    "eess":    "eess.* (Electrical/Signal)",
    "q-bio":   "q-bio.* (Biology)",
    "q-fin":   "q-fin.* (Finance)",
    "stat":    "stat.* (Statistics)",
    "math":    "math.* (Mathematics)",
    "physics": "physics.* (Physics)",
}


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

    df["category_list"] = df["category_list"].apply(to_list)
    df["quarter"] = pd.to_datetime(df["publish_date"], errors="coerce").dt.to_period("Q")
    return df


def _domain_key(tag: str) -> str | None:
    """将 tag 映射到 FOCUS_DOMAINS 的键，无匹配返回 None"""
    for prefix in FOCUS_DOMAINS:
        if tag == prefix or tag.startswith(prefix + "."):
            return prefix
    return None


def plot_bar(counter: Counter, output_dir: Path) -> None:
    top = counter.most_common(TOP_N)
    tags = [t for t, _ in top]
    counts = [c for _, c in top]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(tags[::-1], counts[::-1], color="steelblue")
    ax.set_xlabel("Occurrence Count")
    ax.set_title(f"Top {TOP_N} Non-cs.* Categories in NLP Papers")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "crossdomain_top_categories.png", dpi=150)
    plt.close(fig)


def plot_penetration(df: pd.DataFrame, output_dir: Path) -> None:
    quarters = sorted(df["quarter"].dropna().unique())
    if len(quarters) < 2:
        return

    records = []
    for q in quarters:
        sub = df[df["quarter"] == q]
        total = len(sub)
        if total == 0:
            continue
        domain_counts: dict[str, int] = {k: 0 for k in FOCUS_DOMAINS}
        for cats in sub["category_list"]:
            seen = set()
            for tag in cats:
                if tag.startswith("cs."):
                    continue
                key = _domain_key(tag)
                if key and key not in seen:
                    domain_counts[key] += 1
                    seen.add(key)
        row = {"quarter": str(q), "total": total}
        for key, cnt in domain_counts.items():
            row[key] = cnt / total * 100
        records.append(row)

    rate_df = pd.DataFrame(records)
    rate_df.to_csv(output_dir / "crossdomain_penetration_rate.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14, 6))
    for key, label in FOCUS_DOMAINS.items():
        if key not in rate_df.columns:
            continue
        ax.plot(rate_df["quarter"], rate_df[key], marker="o", markersize=3, linewidth=1.4, label=label)

    step = max(1, len(rate_df) // 16)
    ax.set_xticks(range(0, len(rate_df), step))
    ax.set_xticklabels(rate_df["quarter"].iloc[::step], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_ylabel("Penetration Rate (% of papers per quarter)")
    ax.set_title("Cross-Domain Penetration Rate of NLP Papers Over Time (excl. cs.*)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "crossdomain_penetration_rate.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 统计全量非 cs.* 标签
    counter: Counter[str] = Counter()
    for cats in df["category_list"]:
        for tag in cats:
            if not tag.startswith("cs."):
                counter[tag] += 1

    pd.DataFrame(counter.most_common(), columns=["category", "count"]).to_csv(
        OUTPUT_DIR / "crossdomain_top_categories.csv", index=False, encoding="utf-8-sig"
    )

    plot_bar(counter, OUTPUT_DIR)
    plot_penetration(df, OUTPUT_DIR)
    print(f"Done. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
