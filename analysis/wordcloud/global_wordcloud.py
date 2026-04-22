from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.process import load_cleaned
from wordcloud import WordCloud

OUTPUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")

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


def main() -> None:
    df = load_cleaned()[["id", "title"]]

    vec = CountVectorizer(
        ngram_range=(1, 3),
        max_features=600,
        stop_words="english",
        min_df=20,
        binary=True,
    )
    X = vec.fit_transform(df["title"].tolist())
    terms = vec.get_feature_names_out()
    doc_freq = np.asarray(X.sum(axis=0)).flatten()

    freq_map = {
        t: int(f)
        for t, f in zip(terms, doc_freq)
        if t not in EXTRA_STOPS and not any(s in t.split() for s in EXTRA_STOPS)
    }

    wc = WordCloud(
        width=2200,
        height=1400,
        background_color="#f4f1e8",
        colormap="tab10",
        max_words=160,
        font_path=str(FONT_PATH),
    ).generate_from_frequencies(freq_map)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(OUTPUT_DIR / "global_wordcloud.png"))

    top = sorted(freq_map.items(), key=lambda x: -x[1])[:160]
    pd.DataFrame(top, columns=["term", "doc_count"]).to_csv(
        OUTPUT_DIR / "global_word_frequencies_lib.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Done. top5: {top[:5]}")


if __name__ == "__main__":
    main()
