"""
Step 1: Load & Explore
"""

import re
from collections import Counter

import pandas as pd

from config import DATA_PATH, OUTPUTS_DIR, SUBJECT_COL, TOPIC_COL

NOISE_PATTERNS = [
    r"\d+[-–]\d+\s*(?:am|pm)",           # time ranges like "9-11 am"
    r"\bEVALUACIÓN\b",                   # explicit evaluation markers
    r"\bExamen\b",                       # exam markers
    r"^\s*$",                            # blank topics
]


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[SUBJECT_COL] = df[SUBJECT_COL].str.strip()
    df[TOPIC_COL] = df[TOPIC_COL].str.strip()
    df = df.dropna(subset=[TOPIC_COL])
    df[SUBJECT_COL] = df[SUBJECT_COL].replace(
        {s: _canonical(s) for s in df[SUBJECT_COL].unique()}
    )
    return df


_SUBJECT_ALIASES = {
    "Asignatua 4": "Asignatura 4",   # typo in source data (missing 'r')
}


def _canonical(name: str) -> str:
    """Normalise subject names: collapse internal spaces and fix known typos."""
    name = re.sub(r"\s+", " ", name).strip()
    return _SUBJECT_ALIASES.get(name, name)


def print_stats(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("STEP 1: LOAD & EXPLORE")
    print("=" * 60)
    print(f"Total rows    : {len(df)}")
    print(f"Unique subjects: {df[SUBJECT_COL].nunique()}")
    print(f"Null topics   : {df[TOPIC_COL].isna().sum()}")
    print()

    print("Rows per subject:")
    counts = df[SUBJECT_COL].value_counts().sort_index()
    for subj, n in counts.items():
        print(f"  {subj:<20} {n:>4} topics")
    print()

    print("Noise patterns detected in topics:")
    for pattern in NOISE_PATTERNS:
        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        mask = df[TOPIC_COL].apply(lambda t: bool(compiled.search(t)) if isinstance(t, str) else False)
        print(f"  {pattern!r:<45} → {mask.sum()} matches")
    print()

    print("Sample topics with detected noise:")
    compiled_all = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in NOISE_PATTERNS]
    def _has_noise(text):
        return isinstance(text, str) and any(p.search(text) for p in compiled_all)
    noisy = df[df[TOPIC_COL].apply(_has_noise)]
    for _, row in noisy.head(8).iterrows():
        print(f"  [{row[SUBJECT_COL]}] {row[TOPIC_COL]!r}")
    print()


def save(df: pd.DataFrame) -> pd.DataFrame:
    out = OUTPUTS_DIR / "01_loaded.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    return df


def run() -> pd.DataFrame:
    df = load_raw()
    print_stats(df)
    return save(df)


if __name__ == "__main__":
    run()
