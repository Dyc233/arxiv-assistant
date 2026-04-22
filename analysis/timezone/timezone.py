from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.process import load_cleaned

OUTPUT_DIR = Path(__file__).resolve().parent

# UTC 偏移量，工作时段定义为本地 9-18 点
REGIONS = {
    "China":  {"offset": 8,  "color": "#e74c3c"},
    "Europe": {"offset": 1,  "color": "#3498db"},
    "US East":{"offset": -5, "color": "#2ecc71"},
}
WORK_START, WORK_END = 9, 18


def _in_work_hours(utc_hour: int, offset: int) -> bool:
    local = (utc_hour + offset) % 24
    return WORK_START <= local < WORK_END


def main() -> None:
    df = load_cleaned()[["id", "publish_hour", "publish_date"]].dropna(subset=["publish_hour"])
    df["publish_hour"] = df["publish_hour"].astype(int)
    df["year"] = pd.to_datetime(df["publish_date"], errors="coerce").dt.year.astype("Int64")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    hours = np.arange(24)
    years = sorted(y for y in df["year"].unique() if (df["year"] == y).sum() > 50)

    # ── 热力图数据：year × UTC hour，行归一化 ──
    heatmap = pd.DataFrame(0.0, index=years, columns=hours)
    for y in years:
        yr = df[df["year"] == y]["publish_hour"].value_counts().reindex(hours, fill_value=0)
        heatmap.loc[y] = yr.values / yr.values.sum()

    # ── 面积图数据：各年各地区工作时段内论文占比 ──
    region_ratio = pd.DataFrame(index=years, columns=list(REGIONS.keys()), dtype=float)
    for y in years:
        yr_df = df[df["year"] == y]
        total = len(yr_df)
        for name, cfg in REGIONS.items():
            region_ratio.loc[y, name] = yr_df["publish_hour"].apply(
                lambda h, off=cfg["offset"]: _in_work_hours(h, off)
            ).sum() / total

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── 左图：热力图 year × UTC hour ──
    ax = axes[0]
    im = ax.imshow(heatmap.values, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=8)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in hours], fontsize=7)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Year")
    ax.set_title("发布热力图 by Year × UTC Hour")
    plt.colorbar(im, ax=ax, fraction=0.03, label="proportion")

    for name, cfg in REGIONS.items():
        utc_start = (WORK_START - cfg["offset"]) % 24
        utc_end   = (WORK_END   - cfg["offset"]) % 24
        if utc_start < utc_end:
            ax.axvspan(utc_start - 0.5, utc_end - 0.5, alpha=0.15,
                       color=cfg["color"], zorder=0, label=name)
        else:
            ax.axvspan(utc_start - 0.5, 23.5, alpha=0.15, color=cfg["color"], zorder=0)
            ax.axvspan(-0.5, utc_end - 0.5, alpha=0.15, color=cfg["color"], zorder=0, label=name)
    ax.legend(fontsize=7, loc="upper right")

    # ── 右图：堆叠面积图，各地区工作时段占比随年变化 ──
    ax2 = axes[1]
    bottom = np.zeros(len(years))
    for name, cfg in REGIONS.items():
        vals = region_ratio[name].values.astype(float)
        ax2.fill_between(years, bottom, bottom + vals, alpha=0.7,
                         color=cfg["color"], label=name)
        bottom += vals
    
    ax2.set_xticks(years)
    ax2.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Proportion of papers in work hours")
    ax2.set_title("Regional Work-Hour Share by Year") #
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "timezone_distribution.png", dpi=150)
    plt.close(fig)

    region_ratio.index.name = "year"
    region_ratio.to_csv(OUTPUT_DIR / "timezone_distribution.csv", encoding="utf-8-sig")
    print(f"Done → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
