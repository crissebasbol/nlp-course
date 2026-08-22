"""
Step 2: Clean with Regex
Removes noise, normalises unicode, and collapses whitespace.
"""

import re
import unicodedata

import pandas as pd

from config import OUTPUTS_DIR, TOPIC_COL

_PATTERNS = [
    (re.compile(r"\d{1,2}\s*[-–]\s*\d{1,2}\s*(?:am|pm)", re.IGNORECASE), ""),    # "9-11 am"
    (re.compile(r"\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}"), ""),                   # "8:30-10:30"
    (re.compile(r"[\r\n\t]+"), " "),                                             # control chars
    (re.compile(r"\s{2,}"), " "),                                                # multiple spaces
]


def _normalise_unicode(text: str) -> str:
    """Convert to NFC unicode form (fixes misencoded characters)."""
    return unicodedata.normalize("NFC", text)


def clean_topic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _normalise_unicode(text)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["topic_clean"] = df[TOPIC_COL].apply(clean_topic)
    df = df[df["topic_clean"].str.len() > 0].reset_index(drop=True)
    return df


def print_examples(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("STEP 2: CLEAN (regex)")
    print("=" * 60)
    print(f"Rows after cleaning: {len(df)}")
    print()
    print("Before → After examples:")
    changed = df[df["topic_clean"] != df[TOPIC_COL]]
    for _, row in changed.head(10).iterrows():
        print(f"  BEFORE: {row[TOPIC_COL]!r}")
        print(f"  AFTER : {row['topic_clean']!r}")
        print()


def save(df: pd.DataFrame) -> pd.DataFrame:
    out = OUTPUTS_DIR / "02_cleaned.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    return df


def run(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(OUTPUTS_DIR / "01_loaded.csv")
    df = clean(df)
    print_examples(df)
    return save(df)


if __name__ == "__main__":
    run()
