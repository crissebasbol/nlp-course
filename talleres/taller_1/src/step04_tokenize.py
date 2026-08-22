"""
Step 4 - Tokenisation: two approaches compared
  1. Regex - simple split on whitespace and dots; punctuation stays attached to words
  2. spaCy - language-model tokeniser with Spanish rules; splits punctuation into own tokens

Key difference: the naive split leaves symbols like '(', ')', '-', ',' glued to words;
spaCy detaches them correctly, producing cleaner tokens for downstream tasks.
"""

import re

import pandas as pd
import spacy

from config import OUTPUTS_DIR, SPACY_MODEL, TOPIC_COL

_SPLIT_RE = re.compile(r"[\s.]+")


def tokenise_regex(text: str) -> list[str]:
    return [t for t in _SPLIT_RE.split(text.strip()) if t]


def tokenise_spacy(text: str, nlp) -> list[str]:
    return [t.text for t in nlp(text) if not t.is_space]


def compare(df: pd.DataFrame, nlp, n_samples: int = 6) -> None:
    print("=" * 60)
    print("STEP 4: TOKENISATION (regex vs spaCy)")
    print("=" * 60)

    samples = df["topic_spellchecked"].dropna().sample(n_samples, random_state=42)
    for text in samples:
        r = tokenise_regex(text)
        s = tokenise_spacy(text, nlp)
        print(f"TEXT   : {text}")
        print(f"  regex ({len(r):>2} tokens): {r}")
        print(f"  spaCy ({len(s):>2} tokens): {s}")
        print()


def tokenise_all(df: pd.DataFrame, nlp) -> pd.DataFrame:
    df = df.copy()
    col = "topic_spellchecked"
    df["tokens_regex"] = df[col].apply(lambda t: tokenise_regex(t) if isinstance(t, str) else [])
    df["tokens_spacy"] = df[col].apply(lambda t: tokenise_spacy(t, nlp) if isinstance(t, str) else [])
    return df


def print_totals(df: pd.DataFrame) -> None:
    total_regex = df["tokens_regex"].apply(len).sum()
    total_spacy = df["tokens_spacy"].apply(len).sum()
    unique_regex = len({t for tokens in df["tokens_regex"] for t in tokens})
    unique_spacy = len({t for tokens in df["tokens_spacy"] for t in tokens})
    print(f"{'':30} {'regex':>10} {'spaCy':>10}")
    print(f"{'─'*50}")
    print(f"{'Total tokens':30} {total_regex:>10,} {total_spacy:>10,}")
    print(f"{'Unique tokens':30} {unique_regex:>10,} {unique_spacy:>10,}")
    print(f"{'Avg tokens per topic':30} {total_regex/len(df):>10.1f} {total_spacy/len(df):>10.1f}")
    print()


def save(df: pd.DataFrame) -> pd.DataFrame:
    out = OUTPUTS_DIR / "04_tokenised.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    return df


def run(df: pd.DataFrame = None, nlp=None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(OUTPUTS_DIR / "03_spellchecked.csv")
    if nlp is None:
        nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    compare(df, nlp)
    df = tokenise_all(df, nlp)
    print_totals(df)
    return save(df)


if __name__ == "__main__":
    run()
