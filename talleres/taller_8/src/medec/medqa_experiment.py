"""Optional experiment: how much does the MedQA hint help?

The MS texts were built from MedQA questions, so retrieving the original question gives
the correct answer ("the answer sheet"). This is NOT part of the main pipeline; it only
measures how much of the result comes from finding the answer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def load_medqa(split: str = "train") -> pd.DataFrame:
    """MedQA-USMLE 4 options from the Hugging Face Hub."""
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    return ds.to_pandas()


def build_medqa_index(questions: list[str], encoder, batch_size: int = 64) -> np.ndarray:
    """Normalized embeddings of the questions (cosine similarity = dot product)."""
    emb = encoder.encode(questions, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
    return np.asarray(emb, dtype=np.float32)


def retrieve_similar_questions(text: str, index: np.ndarray, medqa: pd.DataFrame, encoder, k: int = 1) -> list[dict[str, Any]]:
    q = encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = index @ q
    top = np.argsort(-sims)[:k]
    return [{**medqa.iloc[i][["question", "options", "answer"]].to_dict(), "similarity": float(sims[i])} for i in top]


def format_retrieved_options(hits: list[dict[str, Any]]) -> str:
    """Short context: options of the most similar question and its correct answer."""
    if not hits:
        return ""
    hit = hits[0]
    options = hit["options"]
    opts = ", ".join(options.values()) if isinstance(options, dict) else str(options)
    return f"Similar exam question options: {opts}. Correct answer: {hit['answer']}."


def add_medqa_context(numbered_text: str, hint: str) -> str:
    """Appends the hint to the text given to the LLM (detector and corrector)."""
    return f"{numbered_text}\n\nReference (similar exam question): {hint}" if hint else numbered_text


def fallback_correction(error_sentence: str, hits: list[dict[str, Any]]) -> str | None:
    """If an option of the question appears in the sentence, replace it by the correct answer."""
    if not hits:
        return None
    hit = hits[0]
    options = hit["options"].values() if isinstance(hit["options"], dict) else []
    lower = error_sentence.lower()
    for opt in sorted(options, key=len, reverse=True):
        if opt and opt != hit["answer"] and opt.lower() in lower:
            start = lower.index(opt.lower())
            return error_sentence[:start] + hit["answer"] + error_sentence[start + len(opt):]
    return None


def compare_with_without_medqa(base: pd.DataFrame, medqa: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side table of the same metrics with and without the hint."""
    out = pd.concat({"sin MedQA": base, "con MedQA": medqa}, axis=0)
    return out
