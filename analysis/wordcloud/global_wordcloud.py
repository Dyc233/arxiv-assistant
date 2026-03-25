from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.data_process import DEFAULT_INCREMENTAL_PATH, DEFAULT_PARQUET_PATH, STOPWORDS


try:
    from wordcloud import WordCloud
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: wordcloud. Install `wordcloud` first, then rerun this script."
    ) from exc


OUTPUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
EXTRA_STOPWORDS = {
    "paper", "papers", "model", "models", "method", "methods", "task", "tasks",
    "results", "show", "using", "used", "based", "learning", "language", "nlp",
    "large", "neural", "analysis", "data", "system", "framework", "recent",
    "many", "significant", "present", "however", "across", "via",
}


def normalize_terms(value: object) -> list[str]:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    elif pd.isna(value) or value is None:
        return []
    else:
        values = [part.strip() for part in str(value).split(",") if part.strip()]

    terms: list[str] = []
    for item in values[:16]:
        term = re.sub(r"[^a-z0-9\-\+\s]", " ", str(item).lower())
        term = re.sub(r"\s+", " ", term).strip()
        if len(term) <= 2:
            continue
        if term in STOPWORDS or term in EXTRA_STOPWORDS:
            continue
        terms.append(term)
    return terms


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
    counter: Counter[str] = Counter()
    for terms in df["keywords"].apply(normalize_terms):
        counter.update(terms)

    frequencies = dict(counter.most_common(160))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wc = WordCloud(
        width=2200,
        height=1400,
        background_color="#f4f1e8",
        colormap="tab10",
        max_words=160,
        font_path=str(FONT_PATH),
    ).generate_from_frequencies(frequencies)

    wc.to_file(str(OUTPUT_DIR / "global_wordcloud_lib.png"))
    pd.DataFrame(counter.most_common(160), columns=["term", "paper_count"]).to_csv(
        OUTPUT_DIR / "global_word_frequencies_lib.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Done. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
