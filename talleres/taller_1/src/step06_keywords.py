"""
Step 6: Keyword Extraction & Wordcloud
Aggregates keywords per subject, shows top-N, and generates
one wordcloud per subject saved as PNG.
"""

import ast
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

matplotlib.use("Agg")

from config import OUTPUTS_DIR, SUBJECT_COL, TOP_N_KEYWORDS

WORDCLOUD_DIR = OUTPUTS_DIR / "wordclouds"
WORDCLOUD_DIR.mkdir(exist_ok=True)


def _parse_keywords(cell) -> list[str]:
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        try:
            return ast.literal_eval(cell)
        except Exception:
            return []
    return []


def build_keyword_counters(df: pd.DataFrame) -> dict[str, Counter]:
    counters: dict[str, Counter] = {}
    for _, row in df.iterrows():
        subj = row[SUBJECT_COL]
        kws = _parse_keywords(row["keywords"])
        counters.setdefault(subj, Counter()).update(kws)
    return counters


def print_top_keywords(counters: dict[str, Counter]) -> None:
    print("=" * 60)
    print("STEP 6: TOP KEYWORDS PER SUBJECT")
    print("=" * 60)
    for subj in sorted(counters):
        top = counters[subj].most_common(TOP_N_KEYWORDS)
        kw_str = ", ".join(f"{w}({n})" for w, n in top)
        print(f"\n  {subj}:")
        print(f"    {kw_str}")
    print()


def generate_wordclouds(counters: dict[str, Counter]) -> None:
    for subj, counter in counters.items():
        if not counter:
            continue
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=50,
        ).generate_from_frequencies(dict(counter))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(subj, fontsize=14, fontweight="bold")
        safe_name = subj.replace(" ", "_")
        path = WORDCLOUD_DIR / f"wordcloud_{safe_name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved wordcloud → {path}")


def save_keyword_table(counters: dict[str, Counter]) -> None:
    rows = []
    for subj, counter in counters.items():
        for rank, (word, freq) in enumerate(counter.most_common(TOP_N_KEYWORDS), start=1):
            rows.append({"subject": subj, "rank": rank, "keyword": word, "frequency": freq})
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "06_keywords.csv", index=False)
    print(f"Saved → {OUTPUTS_DIR / '06_keywords.csv'}")


def run(df: pd.DataFrame = None) -> dict[str, Counter]:
    if df is None:
        df = pd.read_csv(OUTPUTS_DIR / "05_pos_ner.csv")
    counters = build_keyword_counters(df)
    print_top_keywords(counters)
    print("Generating wordclouds...")
    generate_wordclouds(counters)
    save_keyword_table(counters)
    return counters


if __name__ == "__main__":
    run()
