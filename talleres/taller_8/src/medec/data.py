"""Step 1: load the CSV, rebuild sentences with their IDs and split by clinical case."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SENTENCE_ID = re.compile(r"^(\d+)\s(.*)$")
SPLIT_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")

COLUMN_MAP = {
    "Text ID": "text_id",
    "Text": "text",
    "Sentences": "sentences_raw",
    "Error Flag": "error_flag",
    "Error Sentence ID": "error_sentence_id",
    "Error Sentence": "error_sentence",
    "Corrected Sentence": "corrected_sentence",
    "Corrected Text": "corrected_text",
}


def normalize_text(text: str) -> str:
    """Collapse whitespace and line breaks into single spaces."""
    return " ".join(str(text).split())


def parse_sentences(raw: str) -> list[str]:
    """Turn the `Sentences` column into a list where the index is the sentence ID.

    Some lines (for example lab values like "Hg.") come without a number. They are
    continuation lines and are glued to the previous sentence; otherwise every
    following ID would be shifted.
    """
    sentences: list[str] = []
    for line in str(raw).split("\n"):
        line = line.strip()
        if not line:
            continue
        match = SENTENCE_ID.match(line)
        if match and int(match.group(1)) == len(sentences):
            sentences.append(match.group(2).strip())
        elif sentences:
            sentences[-1] += " " + line
        else:
            sentences.append(line)
    return sentences


def split_free_text(text: str) -> list[str]:
    """Split a text typed by a user into sentences.

    Accepts the dataset format ("0 First sentence\\n1 Second ...") or plain prose.
    """
    text = str(text).strip()
    first = SENTENCE_ID.match(text.split("\n", 1)[0].strip())
    if first and first.group(1) == "0" and "\n" in text:
        return parse_sentences(text)
    return [s.strip() for s in SPLIT_SENTENCES.split(normalize_text(text)) if s.strip()]


def number_sentences(sentences: list[str]) -> str:
    """Numbered text given to the LLMs: one "<id> <sentence>" per line."""
    return "\n".join(f"{i} {s}" for i, s in enumerate(sentences))


def build_case_groups(df: pd.DataFrame) -> pd.Series:
    """Clinical case ID: the correct and the wrong version of a case share the first sentence."""
    return df["sentences"].map(lambda s: normalize_text(s[0]).lower() if s else "")


def load_mediqa_dataset(path: str | Path) -> pd.DataFrame:
    """Read the CSV, rename columns and add parsed sentences and case groups."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.rename(columns=COLUMN_MAP)
    df["sentences"] = df["sentences_raw"].map(parse_sentences)
    df["error_flag"] = df["error_flag"].astype(int)
    df["error_sentence_id"] = df["error_sentence_id"].astype(int)
    for col in ("error_sentence", "corrected_sentence"):
        df[col] = df[col].where(df[col].notna(), None)
    df["group"] = build_case_groups(df)
    return df


def split_by_case(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Train / val / test split where a clinical case never appears in two splits."""
    from sklearn.model_selection import GroupShuffleSplit

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, groups=df["group"]))
    rest, test = df.iloc[rest_idx], df.iloc[test_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(inner.split(rest, groups=rest["group"]))
    splits = {"train": rest.iloc[train_idx], "val": rest.iloc[val_idx], "test": test}
    return {name: part.reset_index(drop=True).assign(split=name) for name, part in splits.items()}


def check_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if a clinical case appears in more than one split."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = set(splits[a]["group"]) & set(splits[b]["group"])
            if shared:
                raise ValueError(f"{len(shared)} cases shared between {a} and {b}")


def save_splits(splits: dict[str, pd.DataFrame], folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(folder / f"{name}.parquet", index=False)


def load_split(folder: str | Path, name: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(folder) / f"{name}.parquet")
    df["sentences"] = df["sentences"].map(list)
    return df


def to_hf_dataset(records: list[dict]):
    """Convert sentence pairs into a Hugging Face Dataset for the Trainer."""
    from datasets import Dataset

    return Dataset.from_list(records)
