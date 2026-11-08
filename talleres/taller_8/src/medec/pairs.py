"""Step 2: one example per sentence, (candidate sentence, full text) -> is it the error?"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_sentence_pairs(row: pd.Series | dict[str, Any]) -> list[dict[str, Any]]:
    """One record per sentence; label is 1 only for the sentence with the error."""
    sentences = list(row["sentences"])
    context = " ".join(sentences)
    n = len(sentences)
    error_id = int(row["error_sentence_id"]) if int(row["error_flag"]) else -1
    return [
        {
            "text_id": row["text_id"],
            "sentence_id": i,
            "sentence": sentence,
            "context": context,
            "rel_pos": i / max(n - 1, 1),
            "label": int(i == error_id),
        }
        for i, sentence in enumerate(sentences)
    ]


def build_pairs_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All sentence pairs of a split."""
    records = [pair for _, row in df.iterrows() for pair in build_sentence_pairs(row)]
    return pd.DataFrame.from_records(records)


def encode_pair(sentence: str, context: str, tokenizer, max_length: int = 512) -> dict[str, Any]:
    """Tokenize as [CLS] sentence [SEP] full text [SEP]; only the text is truncated."""
    return tokenizer(sentence, context, truncation="only_second", max_length=max_length)


def encode_batch(batch: dict[str, list], tokenizer, max_length: int = 512) -> dict[str, Any]:
    """Batched version of encode_pair for datasets.map."""
    return tokenizer(batch["sentence"], batch["context"], truncation="only_second", max_length=max_length)


def compute_class_weights(labels) -> list[float]:
    """Inverse-frequency weights: there are ~12 correct sentences per wrong one."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = len(labels) / (2 * np.maximum(counts, 1))
    return weights.tolist()
